"""Wpis w menu aplikacji — uruchamianie asystenta bez terminala.

    python scripts/install_desktop_entry.py            # zainstaluj
    python scripts/install_desktop_entry.py --remove   # usuń
    python scripts/install_desktop_entry.py --print    # tylko pokaż, nic nie zapisuj

Po co: dotąd start wyglądał tak — otwórz terminal, aktywuj środowisko, wpisz
polecenie. Wpis w menu robi to samo jednym kliknięciem, a nazwa w menu bierze się
z ``assistant_name``, więc po zmianie imienia asystenta wystarczy uruchomić ten
skrypt ponownie.

Standard (freedesktop.org) obsługują Linux i BSD. Na Windowsie i macOS-ie
odpowiedniki wyglądają zupełnie inaczej, więc tam skrypt **nie udaje**, że coś
zrobił — wypisuje, jak zrobić skrót ręcznie.

Żadnych praw administratora: wpis ląduje w katalogu użytkownika
(``~/.local/share/applications``), zgodnie z XDG.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from config import get_user_settings  # noqa: E402

ENTRY_NAME = "miku-assistant.desktop"


def data_home() -> Path:
    """Katalog danych użytkownika wg XDG (bez zakładania ``~/.local/share``)."""
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share"


def launcher_path() -> Path:
    """Skrypt startowy — ten sam, którego używa użytkownik z terminala."""
    return CONFIG_ROOT / "run.sh"


def quote_exec(path: Path) -> str:
    """Ścieżka do pola ``Exec`` zgodnie ze specyfikacją freedesktop.

    Cudzysłowy są KONIECZNE, gdy w ścieżce jest spacja — a katalog projektu może
    się tak nazywać. Bez nich pulpit próbuje uruchomić tylko fragment ścieżki do
    pierwszej spacji i wpis po prostu nic nie robi, bez żadnego komunikatu.
    """
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_entry(name: str) -> str:
    """Treść pliku ``.desktop``. Ścieżki są bezwzględne i liczone z tej maszyny."""
    launcher = launcher_path()
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            f"Name={name}",
            "Comment=Lokalny asystent głosowy / Local voice assistant",
            f"Exec={quote_exec(launcher)} %U",
            f"Path={CONFIG_ROOT}",
            # Bez terminala: o to w tym całym wpisie chodzi.
            "Terminal=false",
            "Categories=Utility;",
            "Keywords=assistant;voice;llm;ollama;",
            # Ikona: nazwa ogólna z motywu systemowego. Własnej nie dokładamy,
            # żeby nie wozić grafiki w repozytorium — każdy motyw ma tę pozycję.
            "Icon=audio-input-microphone",
            "StartupNotify=true",
            "",
        ]
    )


def install(*, dry_run: bool = False) -> int:
    if sys.platform.startswith(("win", "darwin")):
        print(
            "[SYSTEM] Ten format skrótów jest linuksowy (freedesktop.org).\n"
            "         Windows: kliknij prawym na run.ps1 → Wyślij do → Pulpit (utwórz skrót).\n"
            "         macOS:   utwórz skrót do run.sh (Automator → Run Shell Script)."
        )
        return 0

    launcher = launcher_path()
    if not launcher.is_file():
        print(f"[ERROR] Brak skryptu startowego: {launcher}")
        return 2
    if not os.access(launcher, os.X_OK):
        print(f"[SYSTEM] Nadaję prawo wykonania: {launcher}")
        if not dry_run:
            launcher.chmod(launcher.stat().st_mode | 0o111)

    name = get_user_settings().assistant_name
    content = build_entry(name)
    target = data_home() / "applications" / ENTRY_NAME

    if dry_run:
        print(f"# {target}")
        print(content)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    target.chmod(0o755)
    print(f"[SYSTEM] Zapisano wpis w menu: {target}")
    print(f"[SYSTEM] Nazwa w menu: {name} (bierze się z config/user_settings.json)")

    # Odświeżenie bazy menu bywa niepotrzebne (część pulpitów czyta na bieżąco),
    # więc jego brak nie jest błędem.
    for command in (["update-desktop-database", str(target.parent)],):
        try:
            subprocess.run(command, check=False, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    return 0


def remove() -> int:
    target = data_home() / "applications" / ENTRY_NAME
    if target.exists():
        target.unlink()
        print(f"[SYSTEM] Usunięto wpis: {target}")
    else:
        print(f"[SYSTEM] Nie ma czego usuwać ({target}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_desktop_entry.py",
        description="Wpis w menu aplikacji, żeby uruchamiać asystenta bez terminala.",
    )
    parser.add_argument("--remove", action="store_true", help="usuń wpis z menu")
    parser.add_argument(
        "--print", dest="dry_run", action="store_true", help="pokaż treść wpisu, nie zapisuj"
    )
    arguments = parser.parse_args(argv)
    if arguments.remove:
        return remove()
    return install(dry_run=arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
