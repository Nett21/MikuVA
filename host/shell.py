"""Uruchamianie programów: argv, czyste środowisko, twarde blokady (Faza 8).

Ten moduł jest osobno i jest najbardziej rygorystyczny, bo powłoka to najczęstsze
źródło katastrof. Reguły, które nie mają wyjątków:

* **Nigdy ``shell=True``, nigdy pojedynczy łańcuch znaków.** Wyłącznie
  ``argv: list[str]``. Bez powłoki nie ma interpretacji ``;``, ``|``, ``&&``,
  ``$(...)`` ani globów — a więc nie ma klasycznego wstrzyknięcia polecenia.
* **Program musi być na liście ``SHELL_ALLOWED_BINARIES``.** Lista jest domyślnie
  PUSTA, więc narzędzie ``shell.run`` jest domyślnie wyłączone.
* **Nie uruchamiamy „poleceń w łańcuchu".** Flagi w rodzaju ``-c``, ``-Command``,
  ``/c`` są zablokowane, bo są równoważne ``shell=True``: przekazują powłoce
  dowolny tekst do interpretacji. Konsekwencja jest jawna i celowa — potoki i
  przekierowania nie są obsługiwane. To nie brak funkcji, to warunek istnienia
  blokad z punktu niżej.
* **Twarde blokady treści**: ``rm -rf``, formatowanie nośników (``mkfs``,
  ``format``, ``diskpart``), zapis wprost na urządzenie (``dd of=/dev/...``),
  rekurencyjne usuwanie z wymuszeniem, wyłączanie i restart systemu, zmiana
  uprawnień na katalogach systemowych, bomby procesowe.
* **Żadnego podnoszenia uprawnień**: ``sudo``, ``doas``, ``su``, ``pkexec``,
  ``runas``, ``gsudo`` są zablokowane, a na koncie root/administratora narzędzie
  nie działa wcale (``host/privileges.py``).
* **Środowisko budowane od zera**: tylko ``PATH``, katalog domowy, ``LANG`` i to,
  co system musi mieć, żeby program w ogóle wystartował. Żadnych tokenów, kluczy
  API ani zmiennych z sesji użytkownika.
* **Katalog roboczy z dozwolonego obszaru** (``host/paths.py``), twardy limit
  czasu, przechwycone i obcięte ``stdout``/``stderr``, brak ``stdin``.

Co z tym zostaje: uruchomienie jednego, wskazanego wprost programu z argumentami —
np. ``git status`` w katalogu roboczym. I tyle. To celowo mało.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess  # nosec B404 - wyłącznie argv, nigdy shell=True; patrz docstring
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from config import (
    PlatformInfo,
    Settings,
    detect_platform,
    get_settings,
    home_directory,
    path_from_env,
    subprocess_no_window_kwargs,
)
from host.privileges import refuse_if_privileged

logger = logging.getLogger(__name__)


class CommandBlockedError(Exception):
    """Polecenie odrzucone przed uruchomieniem. Powód jest do pokazania."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# Programy, których nie uruchomimy nigdy, choćby ktoś wpisał je na allowlistę.
# Podnoszenie uprawnień, przejmowanie sesji, wyłączanie maszyny, niszczenie danych.
HARD_BLOCKED_BINARIES: Final[frozenset[str]] = frozenset(
    {
        # podnoszenie uprawnień
        "sudo", "doas", "su", "pkexec", "runas", "gsudo", "sudoedit",
        # nośniki i systemy plików
        "mkfs", "mkfs.ext4", "mkfs.fat", "mkfs.ntfs", "mkswap", "fdisk", "sfdisk",
        "parted", "gparted", "diskpart", "format", "format.com", "cfdisk", "wipefs",
        "dd", "shred", "badblocks", "hdparm",
        # zamykanie i restart systemu
        "shutdown", "reboot", "poweroff", "halt", "systemctl-poweroff", "init",
        # zarządzanie pakietami i konfiguracją systemu
        "pacman", "apt", "apt-get", "dnf", "yum", "zypper", "rpm", "dpkg",
        "winget", "choco", "msiexec", "reg", "regedit", "bcdedit", "sc",
        # inne noże w plecach
        "chown", "chmod", "chattr", "takeown", "icacls", "cipher", "vssadmin",
        "netsh", "route", "iptables", "nft", "crontab", "at", "schtasks",
    }
)

# Flagi „wykonaj ten tekst" — równoważne shell=True, więc zablokowane.
INLINE_SCRIPT_FLAGS: Final[frozenset[str]] = frozenset(
    {"-c", "-ec", "-lc", "-e", "--command", "-command", "/c", "/k", "-encodedcommand", "-e:"}
)

# Metaznaki powłoki. Bez powłoki nic nie znaczą, więc ich obecność w argumencie
# oznacza albo pomyłkę modelu, albo próbę wstrzyknięcia — jedno i drugie odrzucamy.
SHELL_METACHARACTERS: Final[tuple[str, ...]] = (
    ";", "|", "&", "`", "$(", "${", ">", "<", "\n", "\r", "&&", "||", "$IFS",
)

# Wzorce sprawdzane na CAŁYM poleceniu (program + argumenty, złożone w tekst).
# Nazwa programu mogła przejść allowlistę, ale zestaw argumentów bywa zabójczy.
_BLOCKED_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\brm\b.*(?:-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)", re.IGNORECASE),
     "rekurencyjne usuwanie z wymuszeniem (rm -rf) jest zablokowane na stałe"),
    (re.compile(r"\brm\b\s+(?:-[^\s]+\s+)*/\s*$", re.IGNORECASE),
     "usuwanie katalogu głównego jest zablokowane na stałe"),
    (re.compile(r"\bdel\b.*(?:/s|/q).*|\brd\b.*/s", re.IGNORECASE),
     "rekurencyjne usuwanie (del /s, rd /s) jest zablokowane na stałe"),
    (re.compile(r"remove-item.*-recurse.*-force|remove-item.*-force.*-recurse", re.IGNORECASE),
     "Remove-Item -Recurse -Force jest zablokowane na stałe"),
    (re.compile(r"\bmkfs\b|\bformat\b\s+[a-z]:|\bdiskpart\b|\bwipefs\b", re.IGNORECASE),
     "formatowanie nośników jest zablokowane na stałe"),
    (re.compile(r"\bdd\b.*\bof=\s*(?:/dev/|\\\\\.\\)", re.IGNORECASE),
     "zapis wprost na urządzenie (dd of=/dev/...) jest zablokowany na stałe"),
    (re.compile(r">\s*/dev/(?:sd|nvme|hd|mmcblk)", re.IGNORECASE),
     "zapis wprost na urządzenie blokowe jest zablokowany na stałe"),
    (re.compile(r"\bchmod\b\s+(?:-R\s+)?777\s+/(?:\s|$)", re.IGNORECASE),
     "zmiana uprawnień katalogu głównego jest zablokowana na stałe"),
    (re.compile(r":\s*\(\s*\)\s*\{.*\}\s*;\s*:", re.DOTALL),
     "bomba procesowa jest zablokowana na stałe"),
    (re.compile(r"\b(?:curl|wget|iwr|invoke-webrequest)\b.*\|\s*(?:ba)?sh", re.IGNORECASE),
     "pobieranie skryptu i uruchamianie go w powłoce jest zablokowane na stałe"),
    (re.compile(r"\bshutdown\b|\breboot\b|\bpoweroff\b|stop-computer|restart-computer",
                re.IGNORECASE),
     "wyłączanie i restart systemu są zablokowane na stałe"),
)

# Zmienne środowiskowe przekazywane programowi. Nic poza nimi — w szczególności
# żadnych ``*_TOKEN``, ``*_KEY``, ``OPENAI_*`` czy ``SSH_AUTH_SOCK``.
_ENV_ALLOWLIST_POSIX: Final[tuple[str, ...]] = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM")
_ENV_ALLOWLIST_WINDOWS: Final[tuple[str, ...]] = (
    "PATH", "USERPROFILE", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)

# Katalogi, z których wolno uruchomić program. Chodzi o to, żeby „git" nie okazał
# się plikiem ``git`` podrzuconym do katalogu zapisywalnego przez użytkownika.
_TRUSTED_PREFIXES_POSIX: Final[tuple[str, ...]] = (
    "/usr/bin", "/usr/sbin", "/bin", "/sbin", "/usr/local/bin", "/usr/local/sbin",
    "/opt", "/var/lib/flatpak/exports/bin", "/snap/bin", "/run/current-system/sw/bin",
    "/nix/store",
)


@dataclass(frozen=True, slots=True)
class ShellResult:
    """Wynik uruchomienia programu."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool = False
    cwd: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def describe(self) -> str:
        head = " ".join(self.argv)
        state = "ok" if self.ok else f"kod {self.returncode}"
        return f"{head} → {state} ({self.duration_ms} ms)"


@dataclass(frozen=True, slots=True)
class ShellPolicy:
    """Co wolno uruchomić na tej maszynie."""

    allowed: tuple[str, ...] = ()
    timeout_s: float = 20.0
    max_output_chars: int = 4_000
    platform_info: PlatformInfo | None = field(default=None, compare=False)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ShellPolicy:
        active = settings or get_settings()
        raw = str(active.shell_allowed_binaries or "")
        names = [item.strip().lower() for item in raw.replace(";", ",").split(",")]
        return cls(
            allowed=tuple(name for name in names if name),
            timeout_s=active.shell_timeout_s,
            max_output_chars=active.shell_max_output_chars,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.allowed)

    def describe(self) -> str:
        if not self.enabled:
            return "wyłączone (SHELL_ALLOWED_BINARIES jest pusta)"
        return f"dozwolone programy: {', '.join(self.allowed)}; limit {self.timeout_s:.0f} s"


# --------------------------------------------------------------------------- #
# Powłoka systemu (informacyjnie — do opisu, nie do uruchamiania łańcuchów)
# --------------------------------------------------------------------------- #


def system_shell(platform_info: PlatformInfo | None = None) -> Path | None:
    """Powłoka właściwa dla tego systemu — albo ``None``, gdy żadnej nie ma.

    Używane wyłącznie w opisach („na tej maszynie powłoką jest bash"). Sam
    ``shell.run`` powłoki NIE uruchamia: wykonuje wskazany program bez pośrednika.
    """
    info = platform_info or detect_platform()
    candidates = ("pwsh", "powershell") if info.is_windows else ("bash", "zsh", "sh")
    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def build_environment(platform_info: PlatformInfo | None = None) -> dict[str, str]:
    """Środowisko dla procesu potomnego — zbudowane od zera, z allowlisty.

    Kopiowanie ``os.environ`` byłoby wyciekiem: w środowisku sesji siedzą tokeny,
    klucze API i uchwyty agentów SSH, a uruchamiany program nie ma powodu ich
    widzieć.
    """
    info = platform_info or detect_platform()
    names = _ENV_ALLOWLIST_WINDOWS if info.is_windows else _ENV_ALLOWLIST_POSIX
    environment: dict[str, str] = {}
    for name in names:
        value = os.environ.get(name)
        if value:
            environment[name] = value

    if not info.is_windows and "HOME" not in environment:
        home = home_directory()
        if home is not None:
            environment["HOME"] = str(home)
    if info.is_windows and "USERPROFILE" not in environment:
        profile = path_from_env("USERPROFILE") or home_directory()
        if profile is not None:
            environment["USERPROFILE"] = str(profile)
    return environment


def resolve_binary(
    name: str, policy: ShellPolicy, *, platform_info: PlatformInfo | None = None
) -> Path:
    """Znajdź program o tej nazwie. Rzuca :class:`CommandBlockedError` z powodem.

    Kolejność sprawdzeń: nazwa (nie ścieżka!) → allowlista → twarda blokada →
    ``PATH`` → katalog wykonywalny w zaufanym prefiksie.
    """
    info = platform_info or detect_platform()
    raw = str(name or "").strip()
    if not raw:
        raise CommandBlockedError("nie podano programu do uruchomienia")

    # Ścieżka zamiast nazwy to najprostszy sposób ominięcia allowlisty
    # („/tmp/git"), więc nie przyjmujemy ścieżek w ogóle.
    if any(separator in raw for separator in ("/", "\\")) or raw.startswith("."):
        raise CommandBlockedError(
            f"podaj nazwę programu z listy dozwolonych, nie ścieżkę ('{raw}')"
        )

    lowered = raw.lower()
    stem = lowered.removesuffix(".exe").removesuffix(".cmd").removesuffix(".bat")
    if stem in HARD_BLOCKED_BINARIES or lowered in HARD_BLOCKED_BINARIES:
        raise CommandBlockedError(
            f"program '{raw}' jest zablokowany na stałe (podnoszenie uprawnień, "
            "operacje na nośnikach albo zmiany systemowe)"
        )
    if not policy.enabled:
        raise CommandBlockedError(
            "uruchamianie programów jest wyłączone — lista SHELL_ALLOWED_BINARIES jest pusta"
        )
    if stem not in policy.allowed and lowered not in policy.allowed:
        raise CommandBlockedError(
            f"program '{raw}' nie jest na liście SHELL_ALLOWED_BINARIES "
            f"({', '.join(policy.allowed)})"
        )

    found = shutil.which(raw)
    if not found:
        raise CommandBlockedError(f"nie znalazłam programu '{raw}' w PATH tej maszyny")
    resolved = Path(os.path.realpath(found))

    reason = _untrusted_location(resolved, info)
    if reason:
        raise CommandBlockedError(reason)
    return resolved


def _under(candidate: Path, prefix: str) -> bool:
    """Czy ścieżka leży w katalogu ``prefix``?

    Porównujemy CZĘŚCI ścieżki, nie prefiks tekstu. ``str.startswith`` uznałby
    ``/usr/binfoo/evil`` za leżące w ``/usr/bin`` — utworzenie takiego katalogu
    wymaga wprawdzie roota, ale zaufanie oparte na przypadkowej zbieżności
    liter jest zaufaniem pozornym, a poprawny sposób jest tak samo krótki.
    (Ta sama reguła co w ``host/paths.py``.)
    """
    root = Path(prefix).parts
    parts = candidate.parts
    return len(parts) >= len(root) and parts[: len(root)] == root


def _untrusted_location(binary: Path, info: PlatformInfo) -> str | None:
    """Czy program leży w miejscu, któremu wolno zaufać?

    Na Uniksie sprawdzamy zaufane prefiksy (``/usr/bin`` i pokrewne) — plik
    podrzucony do katalogu domowego nie zostanie uruchomiony. Na Windowsie nie ma
    ustalonej listy takich katalogów (programy instalują się w ``Program Files``,
    ``LOCALAPPDATA``, ``ProgramData``…), więc wymagamy tylko, żeby ścieżka nie
    była w katalogu tymczasowym — reszta i tak przechodzi przez allowlistę nazw.
    """
    text = str(binary)
    if info.is_windows:
        for variable in ("TEMP", "TMP"):
            temporary = path_from_env(variable)
            # Porównanie tekstowe jest tu w porządku, bo błędem bezpiecznym jest
            # NADmiarowa odmowa: `C:\Temp2\x` przy TEMP=`C:\Temp` zostanie
            # odrzucone, a to gorzej dla wygody, nie dla bezpieczeństwa.
            if temporary is not None and text.casefold().startswith(str(temporary).casefold()):
                return f"program '{binary.name}' leży w katalogu tymczasowym — nie uruchamiam"
        return None
    if any(_under(binary, prefix) for prefix in _TRUSTED_PREFIXES_POSIX):
        return None
    return (
        f"program '{binary.name}' leży w {binary.parent} — poza katalogami systemowymi. "
        "Uruchamiam tylko programy zainstalowane w systemie."
    )


def check_arguments(argv: Sequence[str]) -> None:
    """Sprawdź argumenty: metaznaki, flagi „wykonaj tekst", twarde wzorce."""
    if not argv:
        raise CommandBlockedError("puste polecenie")

    for index, argument in enumerate(argv):
        text = str(argument)
        if "\x00" in text:
            raise CommandBlockedError("argument zawiera bajt zerowy")
        if index == 0:
            continue
        lowered = text.strip().lower()
        if lowered in INLINE_SCRIPT_FLAGS:
            raise CommandBlockedError(
                f"flaga '{text}' uruchamia dowolny tekst w powłoce i jest zablokowana. "
                "Uruchom program wprost, z argumentami — potoki i przekierowania nie są "
                "obsługiwane (i to jest celowe)."
            )
        for metacharacter in SHELL_METACHARACTERS:
            if metacharacter in text:
                raise CommandBlockedError(
                    f"argument '{text}' zawiera znak powłoki '{metacharacter}'. "
                    "Program jest uruchamiany BEZ powłoki, więc taki znak nie zadziała — "
                    "a wygląda na próbę wstrzyknięcia polecenia."
                )

    joined = " ".join(str(item) for item in argv)
    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(joined):
            raise CommandBlockedError(reason)


def check_command(
    argv: Sequence[str], policy: ShellPolicy, *, platform_info: PlatformInfo | None = None
) -> Path:
    """Pełne sprawdzenie polecenia. Zwraca ścieżkę programu do uruchomienia."""
    info = platform_info or detect_platform()
    refusal = refuse_if_privileged("Uruchamianie programów", platform_info=info, strict=True)
    if refusal:
        raise CommandBlockedError(refusal)
    check_arguments(argv)
    return resolve_binary(argv[0], policy, platform_info=info)


def run_command(
    argv: Sequence[str],
    policy: ShellPolicy,
    *,
    cwd: Path | None = None,
    platform_info: PlatformInfo | None = None,
    runner: object | None = None,
) -> ShellResult:
    """Uruchom program po przejściu wszystkich sprawdzeń.

    ``runner`` pozwala podstawić atrapę ``subprocess.run`` (testy nie uruchamiają
    prawdziwych procesów). W kodzie produkcyjnym jest ``None``.
    """
    info = platform_info or detect_platform()
    binary = check_command(argv, policy, platform_info=info)
    command = [str(binary), *[str(item) for item in argv[1:]]]

    execute = runner if runner is not None else subprocess.run
    started = time.perf_counter()
    try:
        completed = execute(  # type: ignore[operator]  # nosec B603 - argv, bez powłoki
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=policy.timeout_s,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            env=build_environment(info),
            stdin=subprocess.DEVNULL,
            **subprocess_no_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandBlockedError(
            f"program '{argv[0]}' nie zakończył się w {policy.timeout_s:.0f} s — przerwany"
        ) from exc
    except (OSError, ValueError) as exc:
        raise CommandBlockedError(f"nie udało się uruchomić '{argv[0]}': {exc}") from exc

    duration_ms = int((time.perf_counter() - started) * 1000)
    stdout, cut_out = _limit(getattr(completed, "stdout", "") or "", policy.max_output_chars)
    stderr, cut_err = _limit(getattr(completed, "stderr", "") or "", policy.max_output_chars)
    return ShellResult(
        argv=tuple(str(item) for item in command),
        returncode=int(getattr(completed, "returncode", 0) or 0),
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        truncated=cut_out or cut_err,
        cwd=str(cwd) if cwd is not None else "",
    )


def _limit(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n[...] wyjście obcięte", True


def describe_backend(
    settings: Settings | None = None, *, platform_info: PlatformInfo | None = None
) -> str:
    """Jedna linijka o stanie uruchamiania programów."""
    info = platform_info or detect_platform()
    policy = ShellPolicy.from_settings(settings)
    shell = system_shell(info)
    where = f"powłoka systemu: {shell.name}" if shell is not None else "brak powłoki w PATH"
    return f"{policy.describe()}; {where}"


def environment_names(platform_info: PlatformInfo | None = None) -> tuple[str, ...]:
    """Nazwy zmiennych środowiskowych przekazywanych programom (do dokumentacji)."""
    info = platform_info or detect_platform()
    return _ENV_ALLOWLIST_WINDOWS if info.is_windows else _ENV_ALLOWLIST_POSIX


__all__ = [
    "HARD_BLOCKED_BINARIES",
    "INLINE_SCRIPT_FLAGS",
    "SHELL_METACHARACTERS",
    "CommandBlockedError",
    "ShellPolicy",
    "ShellResult",
    "build_environment",
    "check_arguments",
    "check_command",
    "describe_backend",
    "environment_names",
    "resolve_binary",
    "run_command",
    "system_shell",
]
