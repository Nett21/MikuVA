"""Testy uruchamiania: okno domyślnie, Ollama sama, bez drugiego terminala.

Zgłoszone z prawdziwego pulpitu: „aby działało, muszą być odpalone co najmniej 2
okna terminala", „aby odpalić GUI trzeba odpalać venv", „gui nie jest opcją
domyślną". Te testy pilnują, żeby żadna z trzech rzeczy nie wróciła.

Nic tu nie uruchamia prawdziwej Ollamy ani nie otwiera okna: proces jest atrapą,
a sprawdzenie „czy odpowiada" — funkcją, którą podstawiamy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import host.ollama as ollama_module
from config import Settings
from host.ollama import ensure_running


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"ollama_host": "http://127.0.0.1:11434"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


class FakeProcess:
    """Proces, który „żyje" — tyle, ile sprawdza kod uruchamiający."""

    def __init__(self, *, exit_code: int | None = None) -> None:
        self.returncode = exit_code
        self._code = exit_code

    def poll(self) -> int | None:
        return self._code


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Podmień uruchamianie procesu; zapisuj, co miało wystartować."""
    started: list[Path] = []

    def fake_spawn(binary: Path) -> FakeProcess:
        started.append(binary)
        return FakeProcess()

    monkeypatch.setattr(ollama_module, "_spawn", fake_spawn)
    monkeypatch.setattr(ollama_module, "find_binary", lambda: Path("/usr/bin/ollama"))
    return started


def reachable(monkeypatch: pytest.MonkeyPatch, *answers: bool) -> None:
    """Kolejne odpowiedzi na pytanie „czy usługa odpowiada?"."""
    queue = list(answers)

    def fake(_settings: Any = None) -> bool:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(ollama_module, "is_reachable", fake)


# --------------------------------------------------------------------------- #
# Uruchamianie usługi
# --------------------------------------------------------------------------- #


def test_dzialajaca_usluga_nie_jest_ruszana(
    monkeypatch: pytest.MonkeyPatch, spawned: list[Path]
) -> None:
    """Cudzy proces (ręczny albo systemd) zostaje w spokoju."""
    reachable(monkeypatch, True)

    result = ensure_running(make_settings())

    assert result.running and not result.started_by_us
    assert spawned == []


def test_brak_uslugi_uruchamia_ja_w_tle(
    monkeypatch: pytest.MonkeyPatch, spawned: list[Path]
) -> None:
    """To jest cała odpowiedź na „muszą być otwarte dwa terminale"."""
    reachable(monkeypatch, False, True)

    result = ensure_running(make_settings(), timeout_s=2.0)

    assert result.running and result.started_by_us
    assert spawned == [Path("/usr/bin/ollama")]
    assert result.message and result.hint  # mówimy, co zrobiliśmy i co dalej


def test_zdalny_serwer_nie_jest_dotykany(
    monkeypatch: pytest.MonkeyPatch, spawned: list[Path]
) -> None:
    """Na cudzej maszynie nie mamy czego uruchamiać — i nie próbujemy."""
    reachable(monkeypatch, False)

    result = ensure_running(make_settings(ollama_host="http://192.168.1.50:11434"))

    assert spawned == []
    assert not result.running
    assert "192.168.1.50" in result.message


def test_wylaczony_autostart_nic_nie_uruchamia(
    monkeypatch: pytest.MonkeyPatch, spawned: list[Path]
) -> None:
    reachable(monkeypatch, False)

    result = ensure_running(make_settings(ollama_autostart=False))

    assert spawned == [] and not result.running


def test_brak_programu_konczy_sie_podpowiedzia(monkeypatch: pytest.MonkeyPatch) -> None:
    reachable(monkeypatch, False)
    monkeypatch.setattr(ollama_module, "find_binary", lambda: None)

    result = ensure_running(make_settings())

    assert not result.running
    assert result.hint  # dokąd pójść po Ollamę


def test_proces_ktory_od_razu_padl_nie_wisi_w_petli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zajęty port kończy się komunikatem, a nie czekaniem do końca limitu."""
    reachable(monkeypatch, False)
    monkeypatch.setattr(ollama_module, "find_binary", lambda: Path("/usr/bin/ollama"))
    monkeypatch.setattr(ollama_module, "_spawn", lambda _binary: FakeProcess(exit_code=1))

    result = ensure_running(make_settings(), timeout_s=30.0)

    assert not result.running
    assert "1" in result.message  # kod wyjścia trafia do komunikatu


def test_start_nie_rzuca_gdy_wszystko_zawiedzie(monkeypatch: pytest.MonkeyPatch) -> None:
    """Awaria uruchamiania nie może przerwać startu asystenta."""
    reachable(monkeypatch, False)
    monkeypatch.setattr(ollama_module, "find_binary", lambda: Path("/usr/bin/ollama"))
    monkeypatch.setattr(ollama_module, "_spawn", lambda _binary: None)

    result = ensure_running(make_settings(), timeout_s=1.0)

    assert not result.running and result.message


# --------------------------------------------------------------------------- #
# Skrypty startowe (bez aktywowania venv)
# --------------------------------------------------------------------------- #


def test_skrypt_startowy_istnieje_i_jest_wykonywalny() -> None:
    from config import PROJECT_ROOT

    launcher = PROJECT_ROOT / "run.sh"
    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111, "run.sh bez prawa wykonania"

    content = launcher.read_text(encoding="utf-8")
    # Sam znajduje Pythona — o to chodzi w „nie trzeba odpalać venv".
    assert ".venv" in content and "python3" in content
    # Ścieżki liczone od pliku, nie od katalogu roboczego (skrót w menu).
    assert "BASH_SOURCE" in content


def test_skrypt_windowsowy_ma_bom() -> None:
    """Bez BOM-u Windows PowerShell 5.1 czyta plik jako ANSI i psuje polskie znaki."""
    from config import PROJECT_ROOT

    assert (PROJECT_ROOT / "run.ps1").read_bytes().startswith(b"\xef\xbb\xbf")


def test_wpis_w_menu_cytuje_sciezke_ze_spacja(tmp_path: Path) -> None:
    """Katalog „Projekt MikuVA" ma spację — bez cudzysłowów wpis nic nie robi."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from install_desktop_entry import quote_exec

    assert quote_exec(tmp_path / "Projekt Miku" / "run.sh").startswith('"')
    assert quote_exec(tmp_path / "Projekt Miku" / "run.sh").endswith('"')
