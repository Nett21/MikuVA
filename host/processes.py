"""Procesy: lista i zamykanie (Faza 8).

Dwa źródła danych, w tej kolejności:

1. **``psutil``** — jeśli jest zainstalowany. Daje nazwę, właściciela i użycie
   pamięci na każdym systemie tym samym kodem.
2. **``/proc``** — na Linuksie i pokrewnych. Czysta biblioteka standardowa, bez
   żadnej dodatkowej zależności.

Gdy nie ma ani jednego, ani drugiego (Windows bez ``psutil``), narzędzia procesowe
są po prostu **niedostępne** i model ich nie widzi. Świadomie nie wołamy tu
``tasklist``/``ps`` przez ``subprocess``: parsowanie ich wyjścia różni się między
wersjami systemu i lokalizacją (kolumny bywają tłumaczone), a od takiego kodu
łatwiej dostać złe PID-y niż brak funkcji.

Zamykanie procesu ma twarde bezpieczniki, niezależne od potwierdzenia użytkownika:

* nie zamykamy procesu numer 0/1 (``init``/``systemd``) ani innych procesów
  systemowych z listy,
* nie zamykamy **siebie** ani swojego procesu nadrzędnego (to byłoby zamknięcie
  asystenta w środku tury albo terminala użytkownika),
* nie zamykamy procesów należących do innego użytkownika — do tego trzeba
  uprawnień administratora, a na nich narzędzia HIGH i tak nie działają,
* domyślnie wysyłamy sygnał „zakończ się" (``SIGTERM``/``TerminateProcess``);
  wymuszenie (``SIGKILL``) wymaga jawnego argumentu, bo nie zostawia programowi
  szansy na zapis danych.
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from config import PlatformInfo, detect_platform

logger = logging.getLogger(__name__)

BACKEND_PSUTIL: Final[str] = "psutil"
BACKEND_PROCFS: Final[str] = "procfs"
BACKEND_NONE: Final[str] = "none"

# Procesy, których nie ruszamy nigdy — niezależnie od zgody użytkownika.
# Zamknięcie któregokolwiek z nich kończy się w najlepszym razie utratą sesji.
PROTECTED_NAMES: Final[frozenset[str]] = frozenset(
    {
        # Linux / POSIX
        "init", "systemd", "systemd-logind", "systemd-journald", "systemd-udevd",
        "dbus-daemon", "dbus-broker", "kthreadd", "kernel_task", "launchd",
        "sshd", "login", "agetty", "polkitd", "pipewire", "wireplumber", "pulseaudio",
        # Windows
        "system", "wininit.exe", "csrss.exe", "smss.exe", "services.exe",
        "lsass.exe", "winlogon.exe", "svchost.exe", "dwm.exe", "fontdrvhost.exe",
        # my sami
        "python", "python3", "python.exe", "pythonw.exe",
    }
)


class ProcessRefusedError(Exception):
    """Nie wolno (albo nie da się) tego zrobić. Powód jest do pokazania."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """Proces widoczny dla użytkownika."""

    pid: int
    name: str
    username: str = ""
    memory_mb: float = 0.0
    command: str = ""
    own: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "user": self.username,
            "memory_mb": round(self.memory_mb, 1),
            "own": self.own,
        }


def _psutil() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def backend(platform_info: PlatformInfo | None = None) -> str:
    """Którym mechanizmem czytamy listę procesów na tej maszynie."""
    if _psutil() is not None:
        return BACKEND_PSUTIL
    info = platform_info or detect_platform()
    if not info.is_windows and Path("/proc").is_dir():
        return BACKEND_PROCFS
    return BACKEND_NONE


def available(platform_info: PlatformInfo | None = None) -> tuple[bool, str]:
    """Czy narzędzia procesowe mają na czym pracować."""
    which = backend(platform_info)
    if which == BACKEND_NONE:
        return False, (
            "brak sposobu odczytu listy procesów na tej maszynie — zainstaluj pakiet "
            "psutil (pip install psutil)"
        )
    return True, ""


def current_user() -> str:
    """Nazwa bieżącego użytkownika albo pusty łańcuch.

    ``getpass.getuser`` czyta ``LOGNAME``/``USER``/``USERNAME``, a gdy ich nie ma —
    pyta systemu. W kontenerze bez ``/etc/passwd`` potrafi rzucić, więc łapiemy.
    """
    import getpass

    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - zależne od środowiska
        return ""


def list_processes(
    *, limit: int = 40, query: str = "", platform_info: PlatformInfo | None = None
) -> list[ProcessInfo]:
    """Lista procesów, posortowana po użyciu pamięci (największe pierwsze)."""
    which = backend(platform_info)
    if which == BACKEND_PSUTIL:
        processes = _list_with_psutil(query)
    elif which == BACKEND_PROCFS:
        processes = _list_with_procfs(query)
    else:
        usable, reason = available(platform_info)
        del usable
        raise ProcessRefusedError(reason)

    processes.sort(key=lambda item: (-item.memory_mb, item.name.casefold()))
    return processes[: max(1, limit)]


def _list_with_psutil(query: str) -> list[ProcessInfo]:
    psutil = _psutil()
    assert psutil is not None  # nosec B101 - sprawdzone przez backend()
    needle = query.strip().casefold()
    me = current_user().casefold()
    found: list[ProcessInfo] = []
    for process in psutil.process_iter(["pid", "name", "username", "memory_info"]):
        try:
            info = process.info
            name = str(info.get("name") or "")
            if needle and needle not in name.casefold():
                continue
            memory = info.get("memory_info")
            username = str(info.get("username") or "")
            found.append(
                ProcessInfo(
                    pid=int(info.get("pid") or 0),
                    name=name,
                    username=username,
                    memory_mb=(getattr(memory, "rss", 0) or 0) / (1024 * 1024),
                    own=bool(me) and username.casefold().endswith(me),
                )
            )
        except Exception:  # proces zniknął w trakcie czytania — normalne
            continue
    return found


def _list_with_procfs(query: str) -> list[ProcessInfo]:
    """Lista procesów z ``/proc`` — tylko biblioteka standardowa."""
    needle = query.strip().casefold()
    my_uid = getattr(os, "getuid", lambda: -1)()
    found: list[ProcessInfo] = []
    try:
        entries = sorted(Path("/proc").iterdir())
    except OSError:  # pragma: no cover - zależne od systemu
        return found

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
            if needle and needle not in name.casefold():
                continue
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
            uid = -1
            memory_kb = 0
            for line in status.splitlines():
                if line.startswith("Uid:"):
                    parts = line.split()
                    uid = int(parts[1]) if len(parts) > 1 else -1
                elif line.startswith("VmRSS:"):
                    numbers = [part for part in line.split() if part.isdigit()]
                    memory_kb = int(numbers[0]) if numbers else 0
            found.append(
                ProcessInfo(
                    pid=int(entry.name),
                    name=name,
                    username="" if uid < 0 else str(uid),
                    memory_mb=memory_kb / 1024,
                    own=uid == my_uid,
                )
            )
        except (OSError, ValueError):  # proces zniknął albo brak dostępu
            continue
    return found


def find_process(pid: int, *, platform_info: PlatformInfo | None = None) -> ProcessInfo | None:
    """Pojedynczy proces albo ``None``, gdy nie istnieje."""
    for process in list_processes(limit=100_000, platform_info=platform_info):
        if process.pid == pid:
            return process
    return None


def check_terminate(
    process: ProcessInfo, *, platform_info: PlatformInfo | None = None
) -> None:
    """Bezpieczniki zamykania procesu. Rzuca :class:`ProcessRefusedError` z powodem."""
    if process.pid <= 1:
        raise ProcessRefusedError(
            f"proces {process.pid} to proces systemowy — nie zamykam go w żadnym wypadku"
        )
    if process.name.casefold() in PROTECTED_NAMES:
        raise ProcessRefusedError(
            f"proces '{process.name}' jest na liście chronionych (system, sesja, dźwięk) "
            "— zamknięcie go zerwałoby sesję użytkownika"
        )
    own_pid = os.getpid()
    if process.pid == own_pid:
        raise ProcessRefusedError("to proces samego asystenta — nie zamykam siebie")
    try:
        parent_pid = os.getppid()
    except (AttributeError, OSError):  # pragma: no cover - zależne od systemu
        parent_pid = -1
    if process.pid == parent_pid:
        raise ProcessRefusedError(
            "to proces nadrzędny asystenta (zwykle terminal) — zamknięcie go zamknęłoby rozmowę"
        )
    if not process.own:
        raise ProcessRefusedError(
            f"proces {process.pid} należy do innego użytkownika — zamknięcie wymagałoby "
            "uprawnień administratora, a na nich narzędzia nie działają"
        )


def terminate_process(
    pid: int,
    *,
    force: bool = False,
    platform_info: PlatformInfo | None = None,
    killer: Any | None = None,
) -> str:
    """Zamknij proces. Zwraca opis tego, co zrobiono.

    ``killer`` pozwala podstawić atrapę w testach — żaden test nie zabija
    prawdziwego procesu.
    """
    process = find_process(pid, platform_info=platform_info)
    if process is None:
        raise ProcessRefusedError(f"nie ma procesu o numerze {pid}")
    check_terminate(process, platform_info=platform_info)

    info = platform_info or detect_platform()
    # SIGTERM istnieje wszędzie; SIGKILL tylko na POSIX-ie. Na Windowsie
    # ``os.kill`` z SIGTERM woła TerminateProcess, czyli i tak zamknięcie twarde.
    hard = force and not info.is_windows
    number = getattr(signal, "SIGKILL", signal.SIGTERM) if hard else signal.SIGTERM
    send = killer if killer is not None else os.kill
    try:
        send(pid, number)
    except ProcessLookupError as exc:
        raise ProcessRefusedError(f"proces {pid} zdążył się zakończyć") from exc
    except PermissionError as exc:
        raise ProcessRefusedError(
            f"brak uprawnień do zamknięcia procesu {pid} — należy do innego użytkownika"
        ) from exc
    except OSError as exc:
        raise ProcessRefusedError(f"nie udało się zamknąć procesu {pid}: {exc}") from exc

    how = "wymuszone zamknięcie" if force and not info.is_windows else "prośba o zamknięcie"
    return f"{how} procesu {process.name} (PID {pid})"


def describe_backend(platform_info: PlatformInfo | None = None) -> str:
    """Jedna linijka do raportu zależności."""
    which = backend(platform_info)
    labels = {
        BACKEND_PSUTIL: "psutil (nazwa, właściciel, pamięć)",
        BACKEND_PROCFS: "/proc (biblioteka standardowa)",
        BACKEND_NONE: "brak — narzędzia procesowe niedostępne",
    }
    return labels[which]


__all__ = [
    "BACKEND_NONE",
    "BACKEND_PROCFS",
    "BACKEND_PSUTIL",
    "PROTECTED_NAMES",
    "ProcessInfo",
    "ProcessRefusedError",
    "available",
    "backend",
    "check_terminate",
    "current_user",
    "describe_backend",
    "find_process",
    "list_processes",
    "terminate_process",
]
