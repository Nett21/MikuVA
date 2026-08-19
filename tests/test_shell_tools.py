"""Testy narzędzia ``shell.run`` i twardych blokad powłoki (Faza 8).

**Żaden test nie uruchamia prawdziwego procesu.** Uruchamianie jest podstawiane
atrapą ``runner``, a odnajdywanie programów atrapą ``shutil.which``. Sprawdzamy
decyzje, nie system: co zostanie odrzucone, z jakim powodem i czy do wykonania w
ogóle doszło.

Testy blokad są tu najważniejsze — to one pilnują wymagań, których nie wolno
złamać: brak ``rm -rf``, brak formatowania nośników, brak podnoszenia uprawnień,
brak wykonywania poleceń jako root.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from config import Settings
from host.paths import Workspace
from host.shell import (
    HARD_BLOCKED_BINARIES,
    CommandBlockedError,
    ShellPolicy,
    build_environment,
    check_arguments,
    resolve_binary,
    run_command,
)
from security.risk import RiskLevel
from tools.base import ToolContext, ToolError
from tools.shell import ShellRunArgs, build_shell_tools


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def ctx() -> ToolContext:
    return ToolContext(settings=make_settings(), language="pl")


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class FakeCompleted:
    """Atrapa wyniku ``subprocess.run``."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SpyRunner:
    """Atrapa uruchamiania: notuje, z czym ją wywołano, i nic nie uruchamia."""

    def __init__(self, result: FakeCompleted | None = None) -> None:
        self.result = result or FakeCompleted(stdout="ok\n")
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> FakeCompleted:
        self.calls.append({"command": list(command), **kwargs})
        return self.result


@pytest.fixture
def fake_git(monkeypatch: pytest.MonkeyPatch) -> str:
    """Udawany ``git`` w katalogu systemowym (bez instalowania czegokolwiek)."""
    import host.shell

    monkeypatch.setattr(
        host.shell.shutil, "which", lambda name: "/usr/bin/git" if name == "git" else None
    )
    monkeypatch.setattr(host.shell.os.path, "realpath", lambda path: path)
    return "/usr/bin/git"


def policy(**overrides: Any) -> ShellPolicy:
    values: dict[str, Any] = {"allowed": ("git",), "timeout_s": 5.0, "max_output_chars": 500}
    values.update(overrides)
    return ShellPolicy(**values)


# --------------------------------------------------------------------------- #
# Twarde blokady treści polecenia
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["rm", "-rf", "/home/uzytkownik"], "rm -rf"),
        (["rm", "-fr", "dane"], "rm -rf"),
        (["rm", "-Rf", "."], "rm -rf"),
        (["mkfs.ext4", "/dev/sda1"], "formatowanie"),
        (["format", "c:"], "formatowanie"),
        (["diskpart"], "formatowanie"),
        (["dd", "if=/dev/zero", "of=/dev/sda"], "urządzenie"),
        (["chmod", "-R", "777", "/"], "katalogu głównego"),
        (["shutdown", "now"], "restart systemu"),
        (["powershell", "Remove-Item", "-Recurse", "-Force", "C:\\"], "Remove-Item"),
        (["cmd", "del", "/s", "/q", "C:\\dane"], "rekurencyjne usuwanie"),
    ],
)
def test_zablokowane_wzorce_polecen(argv: list[str], fragment: str) -> None:
    """Te wzorce są zablokowane na stałe — niezależnie od allowlisty i zgody."""
    with pytest.raises(CommandBlockedError) as blad:
        check_arguments(argv)
    assert fragment in blad.value.message


@pytest.mark.parametrize("program", ["sudo", "doas", "pkexec", "runas", "su", "gsudo"])
def test_podnoszenie_uprawnien_jest_zablokowane(program: str) -> None:
    """Asystent nie prosi o hasło administratora i nie ma jak go użyć."""
    assert program in HARD_BLOCKED_BINARIES
    with pytest.raises(CommandBlockedError) as blad:
        resolve_binary(program, policy(allowed=(program,)))
    assert "zablokowany na stałe" in blad.value.message


@pytest.mark.parametrize("program", ["mkfs", "dd", "diskpart", "shutdown", "reg", "iptables"])
def test_programy_systemowe_sa_zablokowane_mimo_allowlisty(program: str) -> None:
    """Wpisanie ich do SHELL_ALLOWED_BINARIES nie pomaga."""
    with pytest.raises(CommandBlockedError):
        resolve_binary(program, policy(allowed=(program,)))


@pytest.mark.parametrize(
    "argument",
    ["a; rm b", "a | tee b", "a && b", "`whoami`", "$(id)", "> /etc/passwd", "x\ny"],
)
def test_metaznaki_powloki_w_argumencie_sa_odrzucane(argument: str) -> None:
    """Bez powłoki te znaki nic nie znaczą — ich obecność to próba wstrzyknięcia."""
    with pytest.raises(CommandBlockedError) as blad:
        check_arguments(["git", argument])
    assert "znak powłoki" in blad.value.message


@pytest.mark.parametrize("flaga", ["-c", "-lc", "--command", "/c", "-Command", "-EncodedCommand"])
def test_flagi_uruchamiajace_dowolny_tekst_sa_odrzucane(flaga: str) -> None:
    """``bash -c "..."`` to shell=True w przebraniu."""
    with pytest.raises(CommandBlockedError) as blad:
        check_arguments(["bash", flaga, "echo hej"])
    assert "zablokowana" in blad.value.message


def test_zwykle_polecenie_przechodzi() -> None:
    check_arguments(["git", "status", "--short"])  # nie rzuca


# --------------------------------------------------------------------------- #
# Allowlista i lokalizacja programu
# --------------------------------------------------------------------------- #


def test_pusta_allowlista_wylacza_narzedzie(fake_git: str) -> None:
    with pytest.raises(CommandBlockedError) as blad:
        resolve_binary("git", policy(allowed=()))
    assert "wyłączone" in blad.value.message


def test_program_poza_allowlista_jest_odrzucany(fake_git: str) -> None:
    with pytest.raises(CommandBlockedError) as blad:
        resolve_binary("git", policy(allowed=("hostname",)))
    assert "nie jest na liście" in blad.value.message


def test_sciezka_zamiast_nazwy_jest_odrzucana() -> None:
    """Najprostsza próba ominięcia allowlisty: podać ścieżkę."""
    for candidate in ("/usr/bin/git", "./git", "..\\git.exe", "/tmp/git"):
        with pytest.raises(CommandBlockedError) as blad:
            resolve_binary(candidate, policy(allowed=("git",)))
        assert "nie ścieżkę" in blad.value.message


def test_program_z_katalogu_niesystemowego_jest_odrzucany(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Podrzucony „git" w katalogu domowym nie zostanie uruchomiony."""
    import host.shell

    podrobka = tmp_path / "git"
    podrobka.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(host.shell.shutil, "which", lambda name: str(podrobka))

    with pytest.raises(CommandBlockedError) as blad:
        resolve_binary("git", policy())
    assert "poza katalogami systemowymi" in blad.value.message


def test_brak_programu_w_path_daje_czytelny_blad(monkeypatch: pytest.MonkeyPatch) -> None:
    import host.shell

    monkeypatch.setattr(host.shell.shutil, "which", lambda name: None)
    with pytest.raises(CommandBlockedError) as blad:
        resolve_binary("git", policy())
    assert "w PATH" in blad.value.message


# --------------------------------------------------------------------------- #
# Środowisko procesu
# --------------------------------------------------------------------------- #


def test_srodowisko_nie_przekazuje_sekretow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token z sesji użytkownika nie ma prawa trafić do uruchamianego programu."""
    monkeypatch.setenv("OPENAI_API_KEY", "sekret-123")
    monkeypatch.setenv("GITHUB_TOKEN", "sekret-456")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent")
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = build_environment()

    assert "OPENAI_API_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "SSH_AUTH_SOCK" not in environment
    assert environment["PATH"] == "/usr/bin"


# --------------------------------------------------------------------------- #
# Uruchomienie (z atrapą)
# --------------------------------------------------------------------------- #


def test_uruchomienie_idzie_bez_powloki_i_bez_stdin(fake_git: str) -> None:
    runner = SpyRunner(FakeCompleted(stdout="gałąź main\n"))

    result = run_command(["git", "status"], policy(), runner=runner)

    assert result.ok and "gałąź main" in result.stdout
    wywolanie = runner.calls[0]
    assert wywolanie["command"] == ["/usr/bin/git", "status"]
    # Klucz „shell" nie jest nawet przekazywany — subprocess.run domyślnie go nie ma.
    assert "shell" not in wywolanie
    assert wywolanie["stdin"] is not None  # DEVNULL, nie odziedziczone wejście
    assert wywolanie["capture_output"] is True


def test_wyjscie_programu_jest_obcinane(fake_git: str) -> None:
    runner = SpyRunner(FakeCompleted(stdout="x" * 5_000))
    result = run_command(["git", "log"], policy(max_output_chars=100), runner=runner)

    assert result.truncated and len(result.stdout) < 200
    assert "obcięte" in result.stdout


def test_przekroczony_czas_konczy_sie_odmowa(fake_git: str) -> None:
    import subprocess

    def zwlekajacy(command: list[str], **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd=command, timeout=1.0)

    with pytest.raises(CommandBlockedError) as blad:
        run_command(["git", "log"], policy(), runner=zwlekajacy)
    assert "nie zakończył się" in blad.value.message


# --------------------------------------------------------------------------- #
# Narzędzie shell.run
# --------------------------------------------------------------------------- #


def tool_for(root: Path, *, runner: Any = None, **overrides: Any) -> Any:
    settings = make_settings(**overrides)
    workspace = Workspace.for_roots([root])
    return build_shell_tools(settings, workspace=workspace, runner=runner)[0]


def test_shell_run_jest_krytyczne_i_domyslnie_niedostepne(tmp_path: Path) -> None:
    tool = tool_for(tmp_path)

    assert tool.spec.risk is RiskLevel.CRITICAL
    usable, reason = tool.available()
    assert not usable and "SHELL_ALLOWED_BINARIES" in reason


def test_shell_run_dziala_po_wpisaniu_programu_na_liste(
    tmp_path: Path, fake_git: str
) -> None:
    runner = SpyRunner(FakeCompleted(stdout="czysto\n"))
    tool = tool_for(tmp_path, runner=runner, shell_allowed_binaries="git")

    assert tool.available()[0]
    wynik = run(tool.run(ShellRunArgs(argv=["git", "status"]), ctx()))

    assert wynik.ok and "czysto" in wynik.data["stdout"]
    # Wyjście programu to dane z zewnątrz — router postawi po nim barierę.
    assert wynik.untrusted is True
    assert runner.calls[0]["cwd"] == str(tmp_path)


def test_shell_run_odmawia_katalogu_poza_obszarem(tmp_path: Path, fake_git: str) -> None:
    tool = tool_for(tmp_path, runner=SpyRunner(), shell_allowed_binaries="git")
    with pytest.raises(ToolError) as blad:
        run(tool.run(ShellRunArgs(argv=["git", "status"], cwd="/etc"), ctx()))
    assert "poza dozwolonymi katalogami" in blad.value.message


def test_pytanie_o_zgode_pokazuje_pelne_argv(tmp_path: Path) -> None:
    """Przy CRITICAL użytkownik musi widzieć dokładnie to, co się wykona."""
    tool = tool_for(tmp_path, shell_allowed_binaries="git")
    pytanie = tool.confirmation(ShellRunArgs(argv=["git", "push", "--force"]), language="pl")

    assert pytanie is not None and pytanie.risk is RiskLevel.CRITICAL
    assert "git push --force" in pytanie.summary
    assert pytanie.requires_phrase  # samo „tak" nie wystarczy
    tresc = "\n".join(pytanie.details)
    assert "PATH" in tresc  # widać, jakie środowisko dostanie program


def test_na_koncie_root_narzedzie_jest_niedostepne(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wymóg z Fazy 8: NIGDY jako root/administrator."""
    import host.privileges
    import host.shell

    monkeypatch.setattr(host.privileges, "is_privileged", lambda *args, **kwargs: True)
    tool = tool_for(tmp_path, shell_allowed_binaries="git")

    usable, reason = tool.available()
    assert not usable and ("root" in reason or "administrator" in reason)

    # ...i nawet bezpośrednie wywołanie kończy się odmową, nie uruchomieniem.
    runner = SpyRunner()
    with pytest.raises(CommandBlockedError):
        run_command(["git", "status"], policy(), runner=runner)
    assert runner.calls == []
    del host.shell  # import wyłącznie po to, żeby test opisywał zależność


def test_nieznane_uprawnienia_tez_blokuja_narzedzie_krytyczne(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """„Nie wiem, czy jestem rootem" traktujemy jak „jestem" — przy CRITICAL."""
    import host.privileges

    monkeypatch.setattr(host.privileges, "is_privileged", lambda *args, **kwargs: None)
    tool = tool_for(tmp_path, shell_allowed_binaries="git")

    usable, reason = tool.available()
    assert not usable and "nie udało się ustalić" in reason


# --------------------------------------------------------------------------- #
# Zaufane katalogi: porównanie po CZĘŚCIACH ścieżki, nie po prefiksie tekstu
# --------------------------------------------------------------------------- #


def test_zaufany_katalog_porownuje_czesci_sciezki_a_nie_tekst() -> None:
    """``/usr/binfoo`` NIE jest w ``/usr/bin``, choć tekstowo się nim zaczyna.

    Zaufanie oparte na przypadkowej zbieżności liter jest zaufaniem pozornym.
    Ta sama reguła obowiązuje w ``host/paths.py`` przy dozwolonych katalogach.
    """
    from pathlib import Path

    from host.shell import _under

    assert _under(Path("/usr/bin/git"), "/usr/bin") is True
    assert _under(Path("/usr/bin"), "/usr/bin") is True
    assert _under(Path("/usr/local/bin/git"), "/usr/local/bin") is True

    assert _under(Path("/usr/binfoo/evil"), "/usr/bin") is False
    assert _under(Path("/usr/bin-nie-ten/evil"), "/usr/bin") is False
    assert _under(Path("/home/ktos/git"), "/usr/bin") is False
    assert _under(Path("/usr"), "/usr/bin") is False


def test_program_z_katalogu_domowego_nie_jest_uruchamiany() -> None:
    """Plik podrzucony do katalogu zapisywalnego przez użytkownika ma być odrzucony."""
    from pathlib import Path

    from config import OSFamily, PlatformInfo, detect_platform
    from host.shell import _untrusted_location

    info = detect_platform()
    if info.os_family is OSFamily.WINDOWS:  # pragma: no cover - zależne od systemu
        pytest.skip("reguła zaufanych prefiksów dotyczy systemów uniksowych")

    powod = _untrusted_location(Path("/home/ktos/.local/bin/git"), info)
    assert powod is not None
    assert "poza katalogami systemowymi" in powod

    assert _untrusted_location(Path("/usr/bin/git"), info) is None
