"""Testy narzędzi uruchamiania: aplikacje, adresy, procesy (Faza 8).

**Nic tu nie jest naprawdę uruchamiane ani zamykane.** Uruchamianie procesów
podstawia atrapa ``runner``, otwieranie plików na Windowsie — atrapa ``opener``,
a zamykanie procesu — atrapa ``killer``. Lista aplikacji jest czytana z prawdziwych
plików ``.desktop``, ale utworzonych w ``tmp_path`` (parser ma być sprawdzony na
prawdziwym formacie, a nie na wyobrażeniu o nim).

Lista procesów jest podstawiana ustalonym zestawem: prawdziwa zależy od tego, co
akurat działa na maszynie, więc test sprawdzałby system, a nie kod.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from config import Settings, detect_platform
from host.apps import (
    Application,
    LaunchError,
    application_directories,
    desktop_exec_argv,
    find_application,
    has_graphical_session,
    launch_application,
    list_applications,
    open_target,
    url_scheme,
)
from host.paths import Workspace
from host.processes import (
    PROTECTED_NAMES,
    ProcessInfo,
    ProcessRefusedError,
    check_terminate,
    terminate_process,
)
from security.risk import RiskLevel
from tools.base import ToolContext, ToolError
from tools.launcher import (
    AppLaunchArgs,
    AppListArgs,
    OpenPathArgs,
    OpenUrlArgs,
    ProcessKillArgs,
    ProcessListArgs,
    build_launcher_tools,
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def ctx() -> ToolContext:
    return ToolContext(settings=make_settings(), language="pl")


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class SpySpawn:
    """Atrapa ``subprocess.Popen`` — notuje argv i nic nie uruchamia."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> None:
        self.calls.append([str(item) for item in argv])


class SpyOpener:
    """Atrapa ``os.startfile`` (Windows) — notuje, co miało zostać otwarte."""

    def __init__(self) -> None:
        self.targets: list[str] = []

    def __call__(self, target: str) -> None:
        self.targets.append(str(target))


class SpyKiller:
    """Atrapa ``os.kill`` — notuje sygnały, nie zabija niczego."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.error = error

    def __call__(self, pid: int, number: int) -> None:
        self.calls.append((pid, number))
        if self.error is not None:
            raise self.error


@pytest.fixture
def desktop_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Katalog z prawdziwymi plikami ``.desktop`` — tylko w tmp_path."""
    applications = tmp_path / "applications"
    applications.mkdir()
    (applications / "firefox.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Firefox\n"
        "Comment=Przeglądarka\nExec=/usr/bin/firefox %U\n",
        encoding="utf-8",
    )
    (applications / "kalkulator.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Kalkulator\nExec=galculator\n"
        "[Desktop Action Nowe]\nName=Nowe okno\nExec=galculator --new\n",
        encoding="utf-8",
    )
    (applications / "ukryty.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Ukryty\nExec=ukryty\nNoDisplay=true\n",
        encoding="utf-8",
    )
    (applications / "link.desktop").write_text(
        "[Desktop Entry]\nType=Link\nName=Strona\nURL=https://example.org\n", encoding="utf-8"
    )
    (applications / "bez-exec.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Kaleka\n", encoding="utf-8"
    )

    import host.apps

    monkeypatch.setattr(
        host.apps, "application_directories", lambda *args, **kwargs: [applications]
    )
    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    return applications


def launcher_tools(**overrides: Any) -> dict[str, Any]:
    settings = make_settings(**overrides.pop("settings", {}))
    workspace = overrides.pop("workspace", None)
    built = build_launcher_tools(settings, workspace=workspace, **overrides)
    return {tool.spec.name: tool for tool in built}


# --------------------------------------------------------------------------- #
# Lista aplikacji (parser .desktop)
# --------------------------------------------------------------------------- #


def test_lista_aplikacji_pomija_ukryte_linki_i_kaleki(desktop_dir: Path) -> None:
    nazwy = [application.name for application in list_applications(limit=50)]

    assert nazwy == ["Firefox", "Kalkulator"]
    assert "Ukryty" not in nazwy  # NoDisplay=true
    assert "Strona" not in nazwy  # Type=Link, nie aplikacja
    assert "Kaleka" not in nazwy  # brak Exec=


def test_lista_aplikacji_filtruje_po_fragmencie_nazwy(desktop_dir: Path) -> None:
    assert [item.name for item in list_applications(query="fire")] == ["Firefox"]
    assert list_applications(query="czegoś takiego nie ma") == []


def test_dodatkowe_akcje_z_pliku_desktop_nie_sa_aplikacjami(desktop_dir: Path) -> None:
    """Sekcja ``[Desktop Action ...]`` to nie osobny program."""
    kalkulator = find_application("Kalkulator")
    assert kalkulator is not None
    assert desktop_exec_argv(kalkulator) == ["galculator"]


def test_kody_podstawien_sa_usuwane_z_polecenia(desktop_dir: Path) -> None:
    """``%U``/``%f`` bez pulpitu nie mają czym się wypełnić."""
    firefox = find_application("firefox")
    assert firefox is not None
    assert desktop_exec_argv(firefox) == ["/usr/bin/firefox"]


def test_katalogi_aplikacji_pochodza_z_detekcji_systemu() -> None:
    """Nie zakładamy ścieżek: pytamy warstwy platformy (i akceptujemy pustą listę)."""
    directories = application_directories(detect_platform())
    assert all(isinstance(item, Path) for item in directories)


# --------------------------------------------------------------------------- #
# Uruchamianie aplikacji
# --------------------------------------------------------------------------- #


def test_uruchomienie_aplikacji_bez_gio_i_xdg_open_uzywa_exec(
    desktop_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Maszyna bez zainstalowanego pulpitu: zostaje polecenie z pliku .desktop."""
    import host.apps

    monkeypatch.setattr(host.apps.shutil, "which", lambda name: None)
    monkeypatch.setattr(host.apps, "detect_platform", _linux)
    spawn = SpySpawn()

    firefox = find_application("Firefox")
    assert firefox is not None
    note = launch_application(firefox, platform_info=_linux(), runner=spawn)

    assert spawn.calls == [["/usr/bin/firefox"]]
    assert "firefox" in note.lower()


def test_uruchomienie_aplikacji_preferuje_gio(
    desktop_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import host.apps

    monkeypatch.setattr(
        host.apps.shutil, "which", lambda name: "/usr/bin/gio" if name == "gio" else None
    )
    monkeypatch.setattr(host.apps, "detect_platform", _linux)
    spawn = SpySpawn()

    firefox = find_application("Firefox")
    assert firefox is not None
    launch_application(firefox, platform_info=_linux(), runner=spawn)

    assert spawn.calls[0][:2] == ["gio", "launch"]


def test_brak_sesji_graficznej_wylacza_narzedzia_okienkowe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serwer po SSH: nie ma gdzie pokazać okna i to jest poprawny stan."""
    import tools.launcher

    monkeypatch.setattr(tools.launcher, "has_graphical_session", lambda *args, **kwargs: False)
    narzedzia = launcher_tools()

    for nazwa in ("app.list", "app.launch", "open.url", "open.path"):
        usable, reason = narzedzia[nazwa].available()
        assert not usable and "sesji graficznej" in reason


def test_nieznana_aplikacja_daje_podpowiedz(desktop_dir: Path) -> None:
    narzedzia = launcher_tools(runner=SpySpawn())
    with pytest.raises(ToolError) as blad:
        run(narzedzia["app.launch"].run(AppLaunchArgs(name="photoshop"), ctx()))

    assert "nie znalazłam aplikacji" in blad.value.message
    assert "Firefox" in blad.value.message  # model dostaje przykłady


def test_app_launch_ma_poziom_medium_a_nie_safe(desktop_dir: Path) -> None:
    """Uruchomienie programu ma skutek — nie jest odczytem."""
    narzedzia = launcher_tools()
    assert narzedzia["app.list"].spec.risk is RiskLevel.SAFE
    assert narzedzia["app.launch"].spec.risk is RiskLevel.MEDIUM


def test_lista_aplikacji_jako_narzedzie(desktop_dir: Path) -> None:
    narzedzia = launcher_tools()
    wynik = run(narzedzia["app.list"].run(AppListArgs(query="", limit=10), ctx()))

    assert wynik.ok and wynik.data["count"] == 2
    assert "Firefox" in wynik.display


# --------------------------------------------------------------------------- #
# Otwieranie adresów i plików
# --------------------------------------------------------------------------- #


def test_dozwolone_sa_tylko_schematy_z_konfiguracji(monkeypatch: pytest.MonkeyPatch) -> None:
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    narzedzia = launcher_tools(runner=SpySpawn())
    otworz = narzedzia["open.url"]

    for adres in ("file:///etc/passwd", "ftp://serwer/plik", "javascript:alert(1)"):
        with pytest.raises(ToolError) as blad:
            run(otworz.run(OpenUrlArgs(url=adres), ctx()))
        assert "nie jest dozwolony" in blad.value.message


def test_adres_bez_schematu_jest_odrzucany(monkeypatch: pytest.MonkeyPatch) -> None:
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    with pytest.raises(ToolError) as blad:
        run(launcher_tools()["open.url"].run(OpenUrlArgs(url="example.org"), ctx()))
    assert "schematu" in blad.value.message


def test_otwarcie_adresu_https_uzywa_narzedzia_systemu(monkeypatch: pytest.MonkeyPatch) -> None:
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        host.apps.shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None
    )
    spawn = SpySpawn()

    note = open_target("https://example.org", platform_info=_linux(), runner=spawn)

    assert spawn.calls == [["/usr/bin/xdg-open", "https://example.org"]]
    assert "xdg-open" in note
    assert url_scheme("https://example.org") == "https"


def test_otwarcie_pliku_tylko_z_dozwolonego_katalogu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(host.apps.shutil, "which", lambda name: "/usr/bin/xdg-open")
    monkeypatch.setattr(host.apps, "detect_platform", _linux)

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "raport.txt").write_text("treść", encoding="utf-8")
    spawn = SpySpawn()
    narzedzia = launcher_tools(workspace=Workspace.for_roots([root]), runner=spawn)

    wynik = run(narzedzia["open.path"].run(OpenPathArgs(path="raport.txt"), ctx()))
    assert wynik.ok and spawn.calls

    with pytest.raises(ToolError) as blad:
        run(narzedzia["open.path"].run(OpenPathArgs(path="/etc/passwd"), ctx()))
    assert "poza dozwolonymi katalogami" in blad.value.message
    assert len(spawn.calls) == 1  # drugie wywołanie nie doszło do systemu


def test_windows_otwiera_przez_startfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Na Windowsie nie ma xdg-open — jest ``os.startfile``, i tylko tam istnieje."""
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    opener = SpyOpener()

    note = open_target("https://example.org", platform_info=_windows(), opener=opener)

    assert opener.targets == ["https://example.org"]
    assert "domyślnym programem" in note


def test_brak_mechanizmu_otwierania_daje_czytelna_odmowe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import host.apps

    monkeypatch.setattr(host.apps, "has_graphical_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(host.apps.shutil, "which", lambda name: None)
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda target: False)

    with pytest.raises(LaunchError) as blad:
        open_target("https://example.org", platform_info=_linux())
    assert "nie znalazłam programu" in blad.value.message


# --------------------------------------------------------------------------- #
# Procesy
# --------------------------------------------------------------------------- #


def _fake_processes(monkeypatch: pytest.MonkeyPatch, processes: list[ProcessInfo]) -> None:
    import host.processes

    monkeypatch.setattr(host.processes, "backend", lambda *args, **kwargs: "psutil")
    monkeypatch.setattr(
        host.processes, "list_processes", lambda **kwargs: list(processes)
    )
    monkeypatch.setattr(
        host.processes,
        "find_process",
        lambda pid, **kwargs: next((item for item in processes if item.pid == pid), None),
    )
    import tools.launcher

    monkeypatch.setattr(tools.launcher, "list_processes", lambda **kwargs: list(processes))
    monkeypatch.setattr(
        tools.launcher,
        "find_process",
        lambda pid, **kwargs: next((item for item in processes if item.pid == pid), None),
    )


def test_lista_procesow_jest_tylko_odczytem(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_processes(
        monkeypatch,
        [
            ProcessInfo(pid=4242, name="firefox", username="net", memory_mb=812.5),
            ProcessInfo(pid=99, name="systemd", username="root", memory_mb=12.0, own=False),
        ],
    )
    narzedzia = launcher_tools()

    wynik = run(narzedzia["process.list"].run(ProcessListArgs(), ctx()))

    assert narzedzia["process.list"].spec.risk is RiskLevel.SAFE
    assert wynik.data["count"] == 2
    assert wynik.data["processes"][0]["pid"] == 4242


@pytest.mark.parametrize("nazwa", ["systemd", "init", "lsass.exe", "pipewire"])
def test_procesow_systemowych_nie_zamykamy(nazwa: str) -> None:
    """Lista chronionych jest twarda — potwierdzenie jej nie omija."""
    assert nazwa in PROTECTED_NAMES
    with pytest.raises(ProcessRefusedError) as blad:
        check_terminate(ProcessInfo(pid=1234, name=nazwa, username="net"))
    assert "chronionych" in blad.value.message


def test_wlasnego_procesu_ani_rodzica_nie_zamykamy() -> None:
    mine = ProcessInfo(pid=os.getpid(), name="miku", username="net")
    with pytest.raises(ProcessRefusedError) as blad:
        check_terminate(mine)
    assert "nie zamykam siebie" in blad.value.message

    parent = ProcessInfo(pid=os.getppid(), name="terminal", username="net")
    with pytest.raises(ProcessRefusedError) as blad:
        check_terminate(parent)
    assert "nadrzędny" in blad.value.message


def test_procesu_innego_uzytkownika_nie_zamykamy() -> None:
    with pytest.raises(ProcessRefusedError) as blad:
        check_terminate(ProcessInfo(pid=4242, name="nginx", username="www", own=False))
    assert "innego użytkownika" in blad.value.message


def test_procesu_numer_jeden_nie_zamykamy() -> None:
    with pytest.raises(ProcessRefusedError):
        check_terminate(ProcessInfo(pid=1, name="cokolwiek", username="net"))


def test_zamkniecie_wlasnego_programu_wysyla_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    processes = [ProcessInfo(pid=4242, name="firefox", username="net", memory_mb=800.0)]
    _fake_processes(monkeypatch, processes)
    killer = SpyKiller()

    note = terminate_process(4242, killer=killer)

    assert killer.calls == [(4242, signal.SIGTERM)]
    assert "firefox" in note


def test_wymuszone_zamkniecie_wymaga_jawnego_argumentu(monkeypatch: pytest.MonkeyPatch) -> None:
    """SIGKILL nie zostawia programowi szansy na zapis — musi być poproszony wprost."""
    import signal

    processes = [ProcessInfo(pid=4242, name="firefox", username="net")]
    _fake_processes(monkeypatch, processes)
    killer = SpyKiller()

    terminate_process(4242, force=True, killer=killer)

    expected = getattr(signal, "SIGKILL", signal.SIGTERM)
    assert killer.calls == [(4242, expected)]


def test_narzedzie_kill_pyta_o_zgode_z_nazwa_programu(monkeypatch: pytest.MonkeyPatch) -> None:
    processes = [ProcessInfo(pid=4242, name="firefox", username="net", memory_mb=812.0)]
    _fake_processes(monkeypatch, processes)
    narzedzia = launcher_tools(killer=SpyKiller())
    kill = narzedzia["process.kill"]

    assert kill.spec.risk is RiskLevel.HIGH
    pytanie = kill.confirmation(ProcessKillArgs(pid=4242), language="pl")
    assert pytanie is not None
    assert "firefox" in pytanie.summary and "PID 4242" in "\n".join(pytanie.details)


def test_narzedzie_kill_nie_dziala_bez_backendu(monkeypatch: pytest.MonkeyPatch) -> None:
    import host.processes
    import tools.launcher

    monkeypatch.setattr(
        tools.launcher, "processes_available", lambda *args, **kwargs: (False, "brak psutil")
    )
    narzedzia = launcher_tools()
    usable, reason = narzedzia["process.kill"].available()
    assert not usable and "psutil" in reason
    del host.processes


def test_kill_na_koncie_root_jest_niedostepny(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.launcher

    monkeypatch.setattr(
        tools.launcher, "refuse_if_privileged", lambda *args, **kwargs: "konto root"
    )
    narzedzia = launcher_tools()
    usable, reason = narzedzia["process.kill"].available()
    assert not usable and "root" in reason


# --------------------------------------------------------------------------- #
# Pomocnicze: udawane platformy
# --------------------------------------------------------------------------- #


def _linux() -> Any:
    from config import OSFamily

    info = detect_platform()
    return info if info.os_family is OSFamily.LINUX else _replace_family(info, OSFamily.LINUX)


def _windows() -> Any:
    from config import OSFamily

    return _replace_family(detect_platform(), OSFamily.WINDOWS)


def _replace_family(info: Any, family: Any) -> Any:
    import dataclasses

    return dataclasses.replace(info, os_family=family)


def test_wykrywanie_sesji_graficznej_nie_zaklada_pulpitu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sprawdzamy zmienne sesji, nie obecność GNOME/KDE/Hyprlanda."""
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert has_graphical_session(_linux()) is False

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert has_graphical_session(_linux()) is True

    monkeypatch.delenv("WAYLAND_DISPLAY")
    monkeypatch.setenv("DISPLAY", ":0")
    assert has_graphical_session(_linux()) is True

    # Windows i macOS mają pulpit z definicji.
    assert has_graphical_session(_windows()) is True


def test_aplikacja_bez_sciezki_na_windowsie_nie_wywala_sie() -> None:
    with pytest.raises(LaunchError):
        launch_application(
            Application(name="X", identifier="x", source="start-menu", path=None),
            platform_info=_windows(),
            opener=SpyOpener(),
        )
