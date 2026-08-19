"""Autostart asystenta w trybie bezobsługowym — Linux i Windows, bez administratora.

    python scripts/install_autostart.py            # zainstaluj
    python scripts/install_autostart.py --print    # pokaż, co powstanie; nic nie zapisuj
    python scripts/install_autostart.py --remove   # usuń
    python scripts/install_autostart.py --status   # sprawdź, czy jest zainstalowany

Co powstaje na której platformie:

Linux (i BSD)
    Jednostka ``systemd --user`` w ``$XDG_CONFIG_HOME/systemd/user``. Świadomie
    **nie** jest to usługa systemowa: PipeWire i PulseAudio działają w sesji
    użytkownika, więc usługa systemowa nie widziałaby ani mikrofonu, ani
    głośnika — a to jedyne wejście i wyjście tego programu. Dodatkowo instalacja
    usługi systemowej wymagałaby ``sudo``, a tutaj nie potrzeba niczego ponad
    prawo zapisu do własnego ``~/.config``.

Windows 10/11
    Zadanie w Harmonogramie zadań (``schtasks``), wyzwalane przy logowaniu,
    uruchamiane **w kontekście zalogowanego użytkownika**. To jest ten wariant,
    który NIE wymaga administratora: prawa administratora są potrzebne dopiero
    przy ``/ru SYSTEM`` albo ``/rl HIGHEST`` — i tego tu nie ma. Gdy ``schtasks``
    z jakiegoś powodu nie zadziała, zapasowo powstaje plik ``.cmd`` w katalogu
    Autostartu użytkownika (``shell:startup``), który też nie wymaga żadnych
    uprawnień.

macOS
    Skrypt nie udaje, że coś zrobił: wypisuje gotowy ``launchd`` plist do
    ręcznego zapisania. Ta platforma nie jest testowana.

Żaden wariant nie zapisuje niczego poza katalogiem użytkownika i nie prosi o
podniesienie uprawnień.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from config import get_user_settings, home_directory  # noqa: E402

# Nazwa jednostki/zadania jest STAŁA i niezależna od `assistant_name`. Imię
# asystenta wolno zmieniać w każdej chwili, a autostart musi dać się odinstalować
# tym samym poleceniem, którym się go zainstalowało.
UNIT_NAME = "miku-assistant.service"
TASK_NAME = "MikuAssistant"
STARTUP_CMD_NAME = "miku-assistant.cmd"
PLIST_NAME = "com.miku.assistant.plist"

TEMPLATE_UNIT = PROJECT_ROOT / "scripts" / "systemd" / UNIT_NAME


# --------------------------------------------------------------------------- #
# Wspólne: gdzie leży Python i jak go zawołać
# --------------------------------------------------------------------------- #


def find_python(*, windowless: bool = False) -> Path:
    """Interpreter, którym ma ruszyć usługa.

    Kolejność: środowisko w katalogu projektu → interpreter, którym uruchomiono
    ten skrypt. Nie zakładamy nazwy „python3”: w środowisku utworzonym na
    Windowsie plik nazywa się ``python.exe`` i leży w ``Scripts``, a nie w
    ``bin``. ``MIKU_VENV_DIR`` pozwala trzymać środowisko poza projektem.

    ``windowless=True`` woli ``pythonw.exe`` — to ta sama maszyna wirtualna, ale
    bez konsoli. Bez tego autostart na Windowsie otwierałby czarne okno przy
    każdym logowaniu.
    """
    venv = Path(os.environ.get("MIKU_VENV_DIR", "") or (CONFIG_ROOT / ".venv"))
    names = ["pythonw.exe", "python.exe"] if windowless else ["python.exe", "python"]
    for directory in ("Scripts", "bin"):
        for name in names:
            candidate = venv / directory / name
            if candidate.is_file():
                return candidate

    current = Path(sys.executable)
    if windowless and current.name.lower() == "python.exe":
        pythonw = current.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw
    return current


def entry_point() -> Path:
    return CONFIG_ROOT / "main.py"


def check_prerequisites() -> list[str]:
    """Ostrzeżenia, które warto zobaczyć PRZED włączeniem autostartu.

    Nie blokują instalacji — usługa ma prawo powstać, zanim mikrofon zostanie
    podłączony. Chodzi o to, żeby użytkownik nie szukał potem po dzienniku, czemu
    nic nie działa.
    """
    warnings: list[str] = []
    if not entry_point().is_file():
        warnings.append(f"nie ma pliku {entry_point()} — sprawdź, czy to katalog projektu")
    python = find_python()
    if not python.is_file():
        warnings.append(f"nie znalazłem interpretera ({python})")
    venv = Path(os.environ.get("MIKU_VENV_DIR", "") or (CONFIG_ROOT / ".venv"))
    if not venv.exists():
        warnings.append(
            f"nie ma środowiska {venv} — usługa ruszy Pythonem systemowym, "
            "który może nie mieć zainstalowanych zależności"
        )
    return warnings


# --------------------------------------------------------------------------- #
# Linux: systemd --user
# --------------------------------------------------------------------------- #


def config_home() -> Path:
    """Katalog konfiguracji użytkownika wg XDG (bez zakładania ``~/.config``).

    ``home_directory()`` zamiast ``Path.home()``: to drugie RZUCA
    ``RuntimeError``, gdy systemu nie da się o katalog domowy zapytać (usługa
    bez ``HOME``, konto bez profilu). Instalator autostartu ma wtedy powiedzieć,
    czego nie wie, a nie wywalić się stack trace'em.
    """
    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    home = home_directory()
    if home is None:
        raise SystemExit(
            "[ERROR] Nie da się ustalić katalogu domowego (brak HOME). "
            "Ustaw XDG_CONFIG_HOME albo uruchom skrypt z sesji użytkownika."
        )
    return home / ".config"


def unit_path() -> Path:
    return config_home() / "systemd" / "user" / UNIT_NAME


def build_unit() -> str:
    """Jednostka z podstawionymi ścieżkami TEJ maszyny.

    Wzorzec z ``scripts/systemd/`` używa ``%h/miku`` jako przykładu; tutaj
    wstawiamy prawdziwy katalog projektu i prawdziwy interpreter. Ścieżki są
    bezwzględne, bo systemd nie rozwija ``~`` ani zmiennych powłoki w
    ``ExecStart``, a katalog projektu może się nazywać ze spacją.
    """
    python = find_python()
    template = TEMPLATE_UNIT.read_text(encoding="utf-8")
    lines: list[str] = []
    for line in template.splitlines():
        if line.startswith("WorkingDirectory="):
            lines.append(f"WorkingDirectory={CONFIG_ROOT}")
        elif line.startswith("ExecStart="):
            # Cudzysłowy są konieczne przy spacji w ścieżce; systemd rozumie je
            # w ExecStart, a bez nich uruchomiłby tylko fragment do pierwszej spacji.
            lines.append(f'ExecStart="{python}" "{entry_point()}" --headless')
        elif line.startswith("Documentation=file:"):
            lines.append(f"Documentation=file:{CONFIG_ROOT / 'README.md'}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _systemctl(*args: str) -> tuple[int, str]:
    """Zawołaj ``systemctl --user``. Brak systemd nie jest wyjątkiem, tylko wynikiem."""
    binary = shutil.which("systemctl")
    if binary is None:
        return 127, "nie ma polecenia systemctl"
    try:
        result = subprocess.run(
            [binary, "--user", *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


def install_systemd(*, dry_run: bool) -> int:
    if not TEMPLATE_UNIT.is_file():
        print(f"[ERROR] Brak wzorca jednostki: {TEMPLATE_UNIT}")
        return 2

    content = build_unit()
    target = unit_path()
    if dry_run:
        print(f"# {target}")
        print(content)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"[SYSTEM] Zapisano jednostkę: {target}")

    code, output = _systemctl("daemon-reload")
    if code != 0:
        print(f"[SYSTEM] `systemctl --user daemon-reload` nie zadziałało ({output}).")
        print("[SYSTEM] Plik jest na miejscu — dokończ ręcznie po zalogowaniu do sesji graficznej.")
        return 0

    code, output = _systemctl("enable", "--now", UNIT_NAME)
    if code != 0:
        print(f"[SYSTEM] Nie udało się włączyć usługi ({output}).")
        print(f"[SYSTEM] Spróbuj ręcznie: systemctl --user enable --now {UNIT_NAME}")
        return 0

    print(f"[SYSTEM] Usługa włączona. Podgląd: journalctl --user -u {UNIT_NAME} -f")
    print(
        "[SYSTEM] Start bez zalogowanej sesji graficznej (opcjonalnie, jednorazowo z sudo):\n"
        f"         sudo loginctl enable-linger {os.environ.get('USER', 'TWOJ_UZYTKOWNIK')}"
    )
    return 0


def remove_systemd() -> int:
    _systemctl("disable", "--now", UNIT_NAME)
    target = unit_path()
    if target.exists():
        target.unlink()
        print(f"[SYSTEM] Usunięto jednostkę: {target}")
    else:
        print(f"[SYSTEM] Nie ma czego usuwać ({target}).")
    _systemctl("daemon-reload")
    return 0


def status_systemd() -> int:
    target = unit_path()
    print(f"[SYSTEM] Plik jednostki: {target} ({'jest' if target.exists() else 'brak'})")
    _, enabled = _systemctl("is-enabled", UNIT_NAME)
    print(f"[SYSTEM] enabled: {enabled or 'nieznane'}")
    _, active = _systemctl("is-active", UNIT_NAME)
    print(f"[SYSTEM] active:  {active or 'nieznane'}")
    return 0


# --------------------------------------------------------------------------- #
# Windows: Harmonogram zadań (bez administratora) + zapasowo katalog Autostart
# --------------------------------------------------------------------------- #


def startup_directory() -> Path:
    """Katalog Autostartu bieżącego użytkownika (``shell:startup``).

    Czytamy ``APPDATA``, a nie sklejamy ścieżki z litery dysku i nazwy konta:
    profil użytkownika może leżeć na innym dysku, a katalog Menu Start bywa
    przekierowany przez zasady domenowe.
    """
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        base = Path(appdata)
    else:
        home = home_directory()
        if home is None:
            raise SystemExit(
                "[ERROR] Nie da się ustalić profilu użytkownika (brak APPDATA). "
                "Uruchom skrypt z sesji zalogowanego użytkownika."
            )
        base = home / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def build_startup_cmd() -> str:
    """Plik ``.cmd`` dla katalogu Autostartu.

    ``start ""`` z pierwszym pustym argumentem jest konieczne: bez niego
    ``cmd`` potraktowałby ścieżkę w cudzysłowie jako TYTUŁ okna i nic by nie
    uruchomił. To klasyczna pułapka ścieżek ze spacją.
    """
    python = find_python(windowless=True)
    return "\r\n".join(
        [
            "@echo off",
            "rem Autostart lokalnego asystenta glosowego (tryb bezobslugowy).",
            "rem Plik wygenerowany przez scripts/install_autostart.py — nie edytuj recznie.",
            f'cd /d "{CONFIG_ROOT}"',
            f'start "" "{python}" "{entry_point()}" --headless',
            "",
        ]
    )


def _schtasks(*args: str) -> tuple[int, str]:
    binary = shutil.which("schtasks")
    if binary is None:
        return 127, "nie ma polecenia schtasks"
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


def build_task_command() -> str:
    """Wartość ``/tr`` dla ``schtasks``.

    Cała komenda jest JEDNYM argumentem, a w środku ma własne cudzysłowy wokół
    ścieżek ze spacjami. Harmonogram przechowuje ją dosłownie, więc bez
    cudzysłowów zadanie z katalogu „Projekt MikuVA” nigdy by nie ruszyło.
    """
    python = find_python(windowless=True)
    return f'"{python}" "{entry_point()}" --headless'


def install_windows(*, dry_run: bool) -> int:
    command = build_task_command()
    cmd_file = startup_directory() / STARTUP_CMD_NAME

    if dry_run:
        print("# Harmonogram zadań (bez administratora):")
        print(
            f'schtasks /create /tn "{TASK_NAME}" /tr {command!r} /sc onlogon /f '
            f'/it /rl LIMITED'
        )
        print()
        print(f"# Wariant zapasowy — {cmd_file}")
        print(build_startup_cmd())
        return 0

    # /sc onlogon  — przy logowaniu tego użytkownika
    # /it          — tylko gdy użytkownik jest zalogowany (jest wtedy sesja
    #                dźwiękowa; bez niej mikrofon i głośnik nie istnieją)
    # /rl LIMITED  — zwykłe uprawnienia. To jest linia, dzięki której NIE trzeba
    #                administratora; /rl HIGHEST wymagałby podniesienia praw.
    # /f           — nadpisz, gdy zadanie już istnieje (ponowna instalacja)
    code, output = _schtasks(
        "/create",
        "/tn",
        TASK_NAME,
        "/tr",
        command,
        "/sc",
        "onlogon",
        "/it",
        "/rl",
        "LIMITED",
        "/f",
    )
    if code == 0:
        print(
            f"[SYSTEM] Utworzono zadanie „{TASK_NAME}” "
            "(start przy logowaniu, bez administratora)."
        )
        print(f"[SYSTEM] Podgląd: schtasks /query /tn {TASK_NAME} /v /fo list")
        return 0

    print(f"[SYSTEM] Harmonogram zadań odmówił ({output}).")
    print("[SYSTEM] Wariant zapasowy: plik w katalogu Autostartu.")
    try:
        cmd_file.parent.mkdir(parents=True, exist_ok=True)
        # newline="" — treść ma już \r\n; bez tego Python dopisałby drugie \r.
        with cmd_file.open("w", encoding="utf-8", newline="") as handle:
            handle.write(build_startup_cmd())
    except OSError as exc:
        print(f"[ERROR] Nie udało się zapisać {cmd_file}: {exc}")
        return 2
    print(f"[SYSTEM] Zapisano: {cmd_file}")
    return 0


def remove_windows() -> int:
    code, output = _schtasks("/delete", "/tn", TASK_NAME, "/f")
    if code == 0:
        print(f"[SYSTEM] Usunięto zadanie „{TASK_NAME}”.")
    else:
        print(f"[SYSTEM] Zadania nie usunięto ({output}).")
    cmd_file = startup_directory() / STARTUP_CMD_NAME
    if cmd_file.exists():
        cmd_file.unlink()
        print(f"[SYSTEM] Usunięto: {cmd_file}")
    return 0


def status_windows() -> int:
    code, output = _schtasks("/query", "/tn", TASK_NAME)
    print(f"[SYSTEM] Zadanie „{TASK_NAME}”: {'jest' if code == 0 else 'brak'}")
    if code == 0:
        print(output)
    cmd_file = startup_directory() / STARTUP_CMD_NAME
    print(f"[SYSTEM] Plik autostartu: {cmd_file} ({'jest' if cmd_file.exists() else 'brak'})")
    return 0


# --------------------------------------------------------------------------- #
# macOS: launchd (do ręcznego zapisania — platforma nietestowana)
# --------------------------------------------------------------------------- #


def build_plist() -> str:
    python = find_python()
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.miku.assistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{entry_point()}</string>
        <string>--headless</string>
    </array>
    <key>WorkingDirectory</key><string>{CONFIG_ROOT}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
    <key>StandardOutPath</key><string>{CONFIG_ROOT / 'logs' / 'launchd.out.log'}</string>
    <key>StandardErrorPath</key><string>{CONFIG_ROOT / 'logs' / 'launchd.err.log'}</string>
</dict>
</plist>
"""


def install_macos(*, dry_run: bool) -> int:
    home = home_directory()
    target = (home or Path("~").expanduser()) / "Library" / "LaunchAgents" / PLIST_NAME
    print("[SYSTEM] macOS nie jest testowaną platformą tego projektu.")
    print(f"[SYSTEM] Zapisz poniższą treść jako {target}, potem:")
    print(f"         launchctl load -w {target}")
    print()
    print(build_plist())
    if not dry_run:
        print(
            "[SYSTEM] Świadomie NIE zapisuję pliku sam: dostęp do mikrofonu na macOS wymaga\n"
            "         zgody przyznanej konkretnej aplikacji, a proces uruchomiony przez\n"
            "         launchd bez terminala tej zgody nie dostanie i usługa milczałaby\n"
            "         bez żadnego błędu."
        )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_autostart.py",
        description=(
            "Autostart asystenta w trybie bezobsługowym (--headless). "
            "Nie wymaga praw administratora na żadnej platformie."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove", action="store_true", help="usuń autostart")
    group.add_argument("--status", action="store_true", help="sprawdź, czy jest zainstalowany")
    group.add_argument(
        "--print", dest="dry_run", action="store_true", help="pokaż treść, nic nie zapisuj"
    )
    arguments = parser.parse_args(argv)
    system = platform_key()

    if arguments.status:
        return {"linux": status_systemd, "windows": status_windows}.get(
            system, lambda: (print("[SYSTEM] Brak sprawdzenia dla tej platformy.") or 0)
        )()

    if arguments.remove:
        return {"linux": remove_systemd, "windows": remove_windows}.get(
            system, lambda: (print("[SYSTEM] Usuń plist ręcznie: launchctl unload -w …") or 0)
        )()

    if not arguments.dry_run:
        name = get_user_settings().assistant_name
        print(f"[SYSTEM] Autostart dla „{name}” — tryb bezobsługowy, bez okna.")
        for warning in check_prerequisites():
            print(f"[SYSTEM] Uwaga: {warning}")

    if system == "windows":
        return install_windows(dry_run=arguments.dry_run)
    if system == "macos":
        return install_macos(dry_run=arguments.dry_run)
    return install_systemd(dry_run=arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
