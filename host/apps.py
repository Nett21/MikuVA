"""Aplikacje: lista, uruchamianie, otwieranie adresów i plików (Faza 8).

Każdy system trzyma listę aplikacji gdzie indziej i uruchamia je czym innym:

============ ================================== ==================================
system       skąd lista                          czym uruchamiamy
============ ================================== ==================================
Linux/BSD    pliki ``*.desktop`` w katalogach     ``gio launch`` → ``xdg-open`` →
             XDG (``/usr/share/applications``,    wprost ``Exec=`` z pliku
             ``~/.local/share/applications``)
Windows      skróty ``*.lnk`` w menu Start        ``os.startfile``
macOS        pakiety ``*.app`` w ``/Applications````open -a``
============ ================================== ==================================

**Nie zakładamy żadnego środowiska graficznego.** Na Linuksie nie ma tu ani słowa
o GNOME, KDE czy Hyprlandzie: kolejność prób jest od najbardziej standardowej
(``gio`` z glib, obecne praktycznie wszędzie) do najbardziej podstawowej
(uruchomienie polecenia z ``Exec=`` bez pośrednika, co działa nawet na goły
X11/Wayland bez żadnego pulpitu). Brak sesji graficznej to poprawny stan — wtedy
narzędzia z tego modułu są po prostu niedostępne i model ich nie widzi.

Uruchamianie zawsze idzie przez ``argv`` (albo ``os.startfile``), nigdy przez
powłokę — z tego samego powodu co w ``host/shell.py``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess  # nosec B404 - wyłącznie argv, nigdy shell=True
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from config import (
    PlatformInfo,
    detect_platform,
    home_directory,
    path_from_env,
    subprocess_no_window_kwargs,
    user_data_directories,
)

logger = logging.getLogger(__name__)


class LaunchError(Exception):
    """Nie da się (albo nie wolno) tego uruchomić. Powód jest do pokazania."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# Pola ``Exec=`` w plikach .desktop zawierają podstawienia (%f, %U, %i, %c...).
# Bez pulpitu nie mamy czym ich wypełnić, więc po prostu je usuwamy.
_FIELD_CODES: Final[re.Pattern[str]] = re.compile(r"%[fFuUdDnNickvm]")

# Ile plików aplikacji oglądamy, zanim uznamy, że wystarczy. Katalog
# ``/usr/share/applications`` na pełnej instalacji ma ich kilkaset.
_MAX_SCANNED_ENTRIES: Final[int] = 2_000


@dataclass(frozen=True, slots=True)
class Application:
    """Aplikacja, którą da się uruchomić na tej maszynie."""

    name: str
    identifier: str
    source: str
    path: Path | None = None
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "id": self.identifier,
            "source": self.source,
            "comment": self.comment,
        }


# --------------------------------------------------------------------------- #
# Sesja graficzna
# --------------------------------------------------------------------------- #


def has_graphical_session(platform_info: PlatformInfo | None = None) -> bool:
    """Czy jest gdzie pokazać okno aplikacji?

    Na Linuksie pytamy o zmienne sesji (``WAYLAND_DISPLAY``, ``DISPLAY``), a nie
    o konkretny pulpit — asystent ma działać i pod Hyprlandem, i pod samym X11,
    i w kontenerze z przekazanym gniazdem. Serwer bez grafiki (SSH, usługa) po
    prostu nie ma czym otworzyć okna i to jest poprawna odpowiedź.
    """
    info = platform_info or detect_platform()
    if info.is_windows or info.is_macos:
        return True
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


def session_label(platform_info: PlatformInfo | None = None) -> str:
    """Krótki opis sesji graficznej do raportu zależności."""
    info = platform_info or detect_platform()
    if info.is_windows:
        return "pulpit Windows"
    if info.is_macos:
        return "pulpit macOS"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "Wayland"
    if os.environ.get("DISPLAY"):
        return "X11"
    return "brak sesji graficznej"


# --------------------------------------------------------------------------- #
# Lista aplikacji
# --------------------------------------------------------------------------- #


def application_directories(platform_info: PlatformInfo | None = None) -> list[Path]:
    """Katalogi, w których dany system trzyma listę aplikacji."""
    info = platform_info or detect_platform()
    directories: list[Path] = []

    if info.is_windows:
        for variable in ("APPDATA", "PROGRAMDATA"):
            base = path_from_env(variable)
            if base is not None:
                directories.append(base / "Microsoft" / "Windows" / "Start Menu" / "Programs")
        return [item for item in directories if item.is_dir()]

    if info.is_macos:
        directories.append(Path("/Applications"))
        home = home_directory()
        if home is not None:
            directories.append(home / "Applications")
        return [item for item in directories if item.is_dir()]

    # Linux i pozostałe systemy zgodne z XDG: katalogi danych z config.py
    # (XDG_DATA_HOME, XDG_DATA_DIRS, /usr/share) + podkatalog ``applications``.
    for base in user_data_directories(info):
        directories.append(base / "applications")
    home = home_directory()
    if home is not None:
        directories.append(home / ".local" / "share" / "applications")
    return [item for item in directories if item.is_dir()]


def _parse_desktop_entry(path: Path) -> Application | None:
    """Wyciągnij nazwę i polecenie z pliku ``.desktop``.

    Świadomie nie używamy ``configparser``: pliki .desktop bywają niezgodne
    z INI (powtórzone klucze, wartości z ``%``), a nam wystarczą trzy pola.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    in_entry = False
    name = ""
    executable = ""
    comment = ""
    hidden = False
    is_application = True
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            # Interesuje nas wyłącznie główna sekcja; „[Desktop Action ...]"
            # to dodatkowe akcje, których nie pokazujemy.
            in_entry = stripped == "[Desktop Entry]"
            continue
        if not in_entry or "=" not in stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "Name" and not name:
            name = value
        elif key == "Exec" and not executable:
            executable = value
        elif key == "Comment" and not comment:
            comment = value
        elif key in ("NoDisplay", "Hidden") and value.lower() == "true":
            hidden = True
        elif key == "Type" and value and value != "Application":
            is_application = False

    if hidden or not is_application or not name or not executable:
        return None
    return Application(
        name=name, identifier=path.stem, source="desktop", path=path, comment=comment
    )


def list_applications(
    *, limit: int = 100, query: str = "", platform_info: PlatformInfo | None = None
) -> list[Application]:
    """Aplikacje zainstalowane na tej maszynie (posortowane po nazwie).

    ``query`` filtruje po nazwie bez rozróżniania wielkości liter — model zwykle
    pyta o „firefox", a nie o pełną nazwę wpisu.
    """
    info = platform_info or detect_platform()
    needle = query.strip().casefold()
    found: dict[str, Application] = {}
    scanned = 0

    for directory in application_directories(info):
        try:
            entries = sorted(directory.iterdir())
        except OSError:  # pragma: no cover - zależne od uprawnień
            continue
        for entry in entries:
            scanned += 1
            if scanned > _MAX_SCANNED_ENTRIES:
                break
            application = _application_from_entry(entry, info)
            if application is None:
                continue
            if needle and needle not in application.name.casefold():
                continue
            found.setdefault(application.name.casefold(), application)
        if scanned > _MAX_SCANNED_ENTRIES:
            break

    ordered = sorted(found.values(), key=lambda item: item.name.casefold())
    return ordered[: max(1, limit)]


def _application_from_entry(entry: Path, info: PlatformInfo) -> Application | None:
    if info.is_windows:
        if entry.suffix.lower() not in (".lnk", ".url"):
            return None
        return Application(name=entry.stem, identifier=entry.stem, source="start-menu", path=entry)
    if info.is_macos:
        if entry.suffix.lower() != ".app":
            return None
        return Application(name=entry.stem, identifier=entry.stem, source="app-bundle", path=entry)
    if entry.suffix.lower() != ".desktop":
        return None
    return _parse_desktop_entry(entry)


def find_application(
    name: str, *, platform_info: PlatformInfo | None = None
) -> Application | None:
    """Znajdź aplikację po nazwie (dokładnie, potem po fragmencie)."""
    info = platform_info or detect_platform()
    needle = str(name or "").strip().casefold()
    if not needle:
        return None
    candidates = list_applications(limit=500, platform_info=info)
    for application in candidates:
        if application.name.casefold() == needle or application.identifier.casefold() == needle:
            return application
    for application in candidates:
        if needle in application.name.casefold():
            return application
    return None


# --------------------------------------------------------------------------- #
# Uruchamianie
# --------------------------------------------------------------------------- #


def _spawn(argv: Sequence[str], *, runner: Any | None = None) -> None:
    """Uruchom proces w tle i zapomnij o nim (aplikacja żyje dłużej niż my)."""
    execute = runner if runner is not None else subprocess.Popen
    try:
        execute(  # nosec B603 - argv, bez powłoki
            [str(item) for item in argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=not detect_platform().is_windows,
            **subprocess_no_window_kwargs(),
        )
    except (OSError, ValueError) as exc:
        raise LaunchError(f"nie udało się uruchomić: {exc}") from exc


def launch_application(
    application: Application,
    *,
    platform_info: PlatformInfo | None = None,
    runner: Any | None = None,
    opener: Any | None = None,
) -> str:
    """Uruchom aplikację. Zwraca opis tego, co zrobiono.

    ``runner``/``opener`` służą do podstawienia atrap w testach — żaden test nie
    uruchamia prawdziwej przeglądarki.
    """
    info = platform_info or detect_platform()
    if not has_graphical_session(info):
        raise LaunchError(
            "na tej maszynie nie ma sesji graficznej — nie ma gdzie pokazać okna aplikacji"
        )

    if info.is_windows:
        if application.path is None:  # pragma: no cover - lista zawsze ma ścieżkę
            raise LaunchError(f"nie wiem, jak uruchomić '{application.name}'")
        _start_file(application.path, opener=opener)
        return f"uruchomiono {application.name} (skrót z menu Start)"

    if info.is_macos:
        _spawn(["open", "-a", application.name], runner=runner)
        return f"uruchomiono {application.name} (open -a)"

    # Linux: od najbardziej standardowego do najbardziej podstawowego.
    if application.path is not None and shutil.which("gio"):
        _spawn(["gio", "launch", str(application.path)], runner=runner)
        return f"uruchomiono {application.name} (gio launch)"
    if application.path is not None and shutil.which("xdg-open"):
        _spawn(["xdg-open", str(application.path)], runner=runner)
        return f"uruchomiono {application.name} (xdg-open)"

    argv = desktop_exec_argv(application)
    if not argv:
        raise LaunchError(
            f"nie wiem, jak uruchomić '{application.name}' — brak gio, xdg-open i "
            "polecenia Exec w pliku .desktop"
        )
    _spawn(argv, runner=runner)
    return f"uruchomiono {application.name} ({argv[0]})"


def desktop_exec_argv(application: Application) -> list[str]:
    """Polecenie z pola ``Exec=`` rozbite na argv, bez podstawień ``%f``/``%U``.

    Ostatnia deska ratunku, gdy w systemie nie ma ani ``gio``, ani ``xdg-open`` —
    czyli na maszynie bez zainstalowanego pulpitu.
    """
    if application.path is None or application.source != "desktop":
        return []
    entry = _parse_desktop_exec(application.path)
    if not entry:
        return []
    import shlex

    cleaned = _FIELD_CODES.sub("", entry).strip()
    try:
        argv = shlex.split(cleaned)
    except ValueError:  # pragma: no cover - uszkodzony plik .desktop
        return []
    return [item for item in argv if item]


def _parse_desktop_exec(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - zależne od uprawnień
        return ""
    in_entry = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_entry = stripped == "[Desktop Entry]"
            continue
        if in_entry and stripped.startswith("Exec="):
            return stripped.partition("=")[2].strip()
    return ""


def _start_file(target: Path | str, *, opener: Any | None = None) -> None:
    """``os.startfile`` — istnieje TYLKO na Windowsie, stąd ``getattr``."""
    start = opener if opener is not None else getattr(os, "startfile", None)
    if start is None:  # pragma: no cover - inne systemy nie trafiają tutaj
        raise LaunchError("ten system nie ma mechanizmu otwierania plików powłoką")
    try:
        start(str(target))
    except OSError as exc:
        raise LaunchError(f"nie udało się otworzyć: {exc}") from exc


def open_target(
    target: str,
    *,
    platform_info: PlatformInfo | None = None,
    runner: Any | None = None,
    opener: Any | None = None,
) -> str:
    """Otwórz adres albo ścieżkę programem domyślnym dla tego systemu.

    Kolejność na Linuksie: ``xdg-open`` → ``gio open`` → ``webbrowser`` z biblioteki
    standardowej (ten ostatni radzi sobie też bez xdg-utils). Na Windowsie
    ``os.startfile``, na macOS ``open``.
    """
    info = platform_info or detect_platform()
    if not has_graphical_session(info):
        raise LaunchError(
            "na tej maszynie nie ma sesji graficznej — nie ma czym otworzyć adresu"
        )

    if info.is_windows:
        _start_file(target, opener=opener)
        return f"otwarto '{target}' domyślnym programem systemu"
    if info.is_macos:
        _spawn(["open", target], runner=runner)
        return f"otwarto '{target}' przez open"

    for command in ("xdg-open", "gio"):
        found = shutil.which(command)
        if not found:
            continue
        argv = [found, "open", target] if command == "gio" else [found, target]
        _spawn(argv, runner=runner)
        return f"otwarto '{target}' przez {command}"

    # Brak xdg-utils i glib: zostaje biblioteka standardowa, która sama zna
    # kilka mechanizmów (m.in. zmienną BROWSER).
    import webbrowser

    if not webbrowser.open(target):
        raise LaunchError(
            "nie znalazłam programu, którym otworzyć ten adres (brak xdg-open, gio i "
            "przeglądarki w zmiennej BROWSER)"
        )
    return f"otwarto '{target}' przez mechanizm biblioteki standardowej"


def url_scheme(url: str) -> str:
    """Schemat adresu (``http``, ``file``…) albo pusty łańcuch."""
    try:
        return (urlsplit(str(url or "").strip()).scheme or "").lower()
    except ValueError:  # pragma: no cover - adres nie do sparsowania
        return ""


def allowed_schemes(raw: str) -> tuple[str, ...]:
    """Lista dozwolonych schematów z konfiguracji."""
    parts = str(raw or "").replace(";", ",").split(",")
    items = [item.strip().lower().rstrip(":") for item in parts]
    return tuple(item for item in items if item)


def describe_backend(platform_info: PlatformInfo | None = None) -> str:
    """Jedna linijka do raportu zależności."""
    info = platform_info or detect_platform()
    directories = application_directories(info)
    session = session_label(info)
    openers: list[str] = []
    if info.is_windows:
        openers.append("os.startfile")
    elif info.is_macos:
        openers.append("open")
    else:
        openers.extend(name for name in ("gio", "xdg-open") if shutil.which(name))
        openers.append("webbrowser")
    return (
        f"{len(directories)} katalogów z aplikacjami, sesja: {session}, "
        f"otwieranie: {', '.join(openers) or 'brak'}"
    )


def iter_names(applications: Iterable[Application]) -> list[str]:
    return [application.name for application in applications]


__all__ = [
    "Application",
    "LaunchError",
    "allowed_schemes",
    "application_directories",
    "describe_backend",
    "desktop_exec_argv",
    "find_application",
    "has_graphical_session",
    "launch_application",
    "list_applications",
    "open_target",
    "session_label",
    "url_scheme",
]
