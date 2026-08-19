"""Usługi użytkownika — jedyna „zmiana konfiguracji", na jaką pozwalamy (Faza 8).

Na Linuksie: ``systemctl --user``. **Wyłącznie ``--user``** — usługi systemowe
wymagają roota, a narzędzia asystenta na koncie roota nie działają w ogóle
(``host/privileges.py``). Nie ma tu ``sudo`` i nie da się go dopisać: ``sudo`` jest
na liście programów zablokowanych na stałe.

Na Windowsie i macOS: **niedostępne**. Usługi Windows (``sc``, ``Set-Service``)
wymagają administratora, a ``launchctl`` na macOS bywa różnie skonfigurowany
w zależności od wersji systemu. Zamiast udawać obsługę, mówimy wprost, że jej nie
ma — narzędzie jest wtedy niewidoczne dla modelu.

Świadomie NIE pozwalamy na ``enable``/``disable``: to zmiana trwała, wykraczająca
poza „teraz uruchom/zatrzymaj". Zostają ``start``, ``stop``, ``restart`` i odczyt
stanu.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess  # nosec B404 - wyłącznie argv, nigdy shell=True
from dataclasses import dataclass
from typing import Any, Final

from config import PlatformInfo, detect_platform, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

# Dozwolone działania. „enable"/„disable" celowo nie ma — patrz docstring modułu.
ALLOWED_ACTIONS: Final[tuple[str, ...]] = ("start", "stop", "restart")

# Nazwa jednostki systemd: litery, cyfry, kropki, kreski, podkreślenia, @ dla
# instancji i opcjonalny sufiks typu. Wzorzec jest wąski celowo — nazwa jednostki
# idzie do argv, a im mniej z niej można zrobić, tym lepiej.
_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9@._\-\\]{1,120}(?:\.(?:service|timer|socket|target|path))?$"
)

_TIMEOUT_S: Final[float] = 15.0


class ServiceRefusedError(Exception):
    """Nie da się (albo nie wolno) tego zrobić. Powód jest do pokazania."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class ServiceInfo:
    """Usługa użytkownika."""

    unit: str
    active: str
    sub: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "active": self.active,
            "state": self.sub,
            "description": self.description,
        }


def systemctl_path() -> str | None:
    """Ścieżka do ``systemctl`` albo ``None``."""
    return shutil.which("systemctl")


def available(platform_info: PlatformInfo | None = None) -> tuple[bool, str]:
    """Czy zarządzanie usługami jest na tej maszynie możliwe."""
    info = platform_info or detect_platform()
    if info.is_windows:
        return False, (
            "usługami Windows nie da się zarządzać bez uprawnień administratora — "
            "narzędzie jest niedostępne"
        )
    if info.is_macos:
        return False, "na macOS narzędzie usług nie jest obsługiwane (launchctl)"
    if systemctl_path() is None:
        return False, "brak polecenia systemctl na tej maszynie (system bez systemd)"
    return True, ""


def _run(argv: list[str], *, runner: Any | None = None) -> subprocess.CompletedProcess[str]:
    execute = runner if runner is not None else subprocess.run
    try:
        return execute(  # type: ignore[no-any-return,operator]  # nosec B603 - argv, bez powłoki
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
            **subprocess_no_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ServiceRefusedError("systemctl nie odpowiedział w wyznaczonym czasie") from exc
    except (OSError, ValueError) as exc:
        raise ServiceRefusedError(f"nie udało się wywołać systemctl: {exc}") from exc


def check_unit(unit: str) -> str:
    """Sprawdź i znormalizuj nazwę jednostki."""
    name = str(unit or "").strip()
    if not name:
        raise ServiceRefusedError("nie podano nazwy usługi")
    if not _UNIT_PATTERN.match(name):
        raise ServiceRefusedError(
            f"'{name}' nie wygląda na nazwę usługi systemd (dozwolone: litery, cyfry, "
            "kropka, kreska, podkreślenie, @)"
        )
    return name


def list_services(
    *, limit: int = 30, platform_info: PlatformInfo | None = None, runner: Any | None = None
) -> list[ServiceInfo]:
    """Usługi użytkownika (``systemctl --user list-units``)."""
    usable, reason = available(platform_info)
    if not usable:
        raise ServiceRefusedError(reason)

    binary = systemctl_path() or "systemctl"
    completed = _run(
        [
            binary, "--user", "list-units", "--type=service", "--all",
            "--no-pager", "--no-legend", "--plain",
        ],
        runner=runner,
    )
    services: list[ServiceInfo] = []
    for line in (getattr(completed, "stdout", "") or "").splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 4:
            continue
        unit, _load, active, sub = parts[0], parts[1], parts[2], parts[3]
        description = parts[4] if len(parts) > 4 else ""
        services.append(
            ServiceInfo(unit=unit, active=active, sub=sub, description=description[:120])
        )
        if len(services) >= max(1, limit):
            break
    return services


def service_status(
    unit: str, *, platform_info: PlatformInfo | None = None, runner: Any | None = None
) -> ServiceInfo:
    """Stan jednej usługi użytkownika."""
    usable, reason = available(platform_info)
    if not usable:
        raise ServiceRefusedError(reason)
    name = check_unit(unit)
    binary = systemctl_path() or "systemctl"
    completed = _run(
        [
            binary, "--user", "show", name,
            "--property=ActiveState,SubState,Description", "--no-pager",
        ],
        runner=runner,
    )
    values: dict[str, str] = {}
    for line in (getattr(completed, "stdout", "") or "").splitlines():
        key, _, value = line.partition("=")
        if key:
            values[key.strip()] = value.strip()
    return ServiceInfo(
        unit=name,
        active=values.get("ActiveState", "unknown"),
        sub=values.get("SubState", ""),
        description=values.get("Description", "")[:120],
    )


def control_service(
    unit: str,
    action: str,
    *,
    platform_info: PlatformInfo | None = None,
    runner: Any | None = None,
) -> str:
    """Uruchom, zatrzymaj albo zrestartuj usługę UŻYTKOWNIKA."""
    usable, reason = available(platform_info)
    if not usable:
        raise ServiceRefusedError(reason)
    name = check_unit(unit)
    verb = str(action or "").strip().lower()
    if verb not in ALLOWED_ACTIONS:
        raise ServiceRefusedError(
            f"dozwolone działania to {', '.join(ALLOWED_ACTIONS)} — 'enable' i 'disable' "
            "są świadomie niedostępne, bo zmieniają konfigurację na stałe"
        )

    binary = systemctl_path() or "systemctl"
    completed = _run([binary, "--user", verb, name], runner=runner)
    code = int(getattr(completed, "returncode", 1) or 0)
    if code != 0:
        detail = (getattr(completed, "stderr", "") or "").strip()[:200]
        raise ServiceRefusedError(f"systemctl --user {verb} {name} zakończyło się błędem: {detail}")
    return f"{verb} usługi użytkownika {name} wykonane"


def describe_backend(platform_info: PlatformInfo | None = None) -> str:
    """Jedna linijka do raportu zależności."""
    usable, reason = available(platform_info)
    if not usable:
        return reason
    return "systemctl --user (bez usług systemowych, bez sudo)"


__all__ = [
    "ALLOWED_ACTIONS",
    "ServiceInfo",
    "ServiceRefusedError",
    "available",
    "check_unit",
    "control_service",
    "describe_backend",
    "list_services",
    "service_status",
    "systemctl_path",
]
