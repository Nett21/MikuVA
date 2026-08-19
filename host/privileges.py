"""Czy działamy z uprawnieniami administratora — i dlaczego to blokuje narzędzia (Faza 8).

Zasada Fazy 8: **narzędzia o wysokim i krytycznym ryzyku nie działają na koncie
root/administratora.** Nie chodzi o nieufność do użytkownika, a o skalę skutku:
to samo wywołanie, które na zwykłym koncie może najwyżej uszkodzić pliki jednego
użytkownika, z podniesionymi uprawnieniami może uszkodzić system. Model językowy
nie ma jak ocenić tej różnicy, a pomyłka jest nieodwracalna.

Nie ma tu też żadnego podnoszenia uprawnień: ``sudo``, ``doas``, ``su``,
``pkexec``, ``runas`` i ``gsudo`` są na liście zablokowanych programów
(``host/shell.py``). Asystent nie prosi o hasło administratora i nie ma sposobu,
żeby go o to poprosić.

Wynik ``None`` znaczy „nie wiem" (nietypowy system, brak API) i jest traktowany
jak „tak" tam, gdzie stawka jest wysoka — przy uprawnieniach zgadywanie na korzyść
działania byłoby złym domyślnym wyborem.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from config import PlatformInfo, detect_platform

logger = logging.getLogger(__name__)

# Nazwy kont administracyjnych na Windowsie bywają lokalizowane („Administrator",
# „Administrador"...), więc NIE opieramy się na nazwie — pytamy systemu przez API.
_WINDOWS_ADMIN_API: Final[str] = "shell32.IsUserAnAdmin"


def is_privileged(platform_info: PlatformInfo | None = None) -> bool | None:
    """Czy proces ma uprawnienia administratora?

    ``True`` = root/administrator, ``False`` = zwykłe konto, ``None`` = nie udało
    się ustalić (wtedy wywołujący traktuje to jak ``True`` dla operacji o wysokim
    ryzyku — patrz :func:`refuse_if_privileged`).
    """
    info = platform_info or detect_platform()

    if info.is_windows:
        return _windows_is_admin()

    # POSIX: root ma efektywny UID 0. ``geteuid`` istnieje tylko na POSIX-ie,
    # dlatego przez ``getattr`` — na Windowsie tej funkcji w ``os`` nie ma.
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:  # pragma: no cover - nietypowy port Pythona
        return None
    try:
        return int(geteuid()) == 0
    except OSError:  # pragma: no cover - zależne od systemu
        return None


def _windows_is_admin() -> bool | None:
    """Pytanie do Windowsa: ``IsUserAnAdmin()``.

    Import ``ctypes`` jest lokalny i opakowany: na Windowsie bez pełnego API
    (kontener, obcy runtime) ta funkcja może nie istnieć, a wtedy odpowiedzią jest
    „nie wiem", nie wyjątek.
    """
    try:  # pragma: no cover - ścieżka wykonywana tylko na Windowsie
        import ctypes

        shell32 = getattr(ctypes, "windll", None)
        if shell32 is None:
            return None
        return bool(shell32.shell32.IsUserAnAdmin())
    except Exception as exc:  # pragma: no cover - zależne od systemu
        logger.debug("Nie udało się sprawdzić uprawnień administratora: %s", exc)
        return None


def account_label(platform_info: PlatformInfo | None = None) -> str:
    """Krótki opis konta do komunikatów („zwykłe konto", „ROOT", „administrator")."""
    info = platform_info or detect_platform()
    state = is_privileged(info)
    if state is None:
        return "uprawnienia nieznane"
    if not state:
        return "zwykłe konto użytkownika"
    return "konto administratora" if info.is_windows else "konto root"


def refuse_if_privileged(
    action: str, *, platform_info: PlatformInfo | None = None, strict: bool = True
) -> str | None:
    """Powód odmowy, gdy nie wolno wykonać ``action`` na tym koncie.

    ``strict=True`` (narzędzia CRITICAL): odmowa także wtedy, gdy uprawnień nie
    dało się ustalić. ``strict=False`` (narzędzia HIGH): odmowa tylko przy
    pewnym root/administratorze.
    """
    info = platform_info or detect_platform()
    state = is_privileged(info)
    if state is True:
        who = "administratora" if info.is_windows else "roota"
        return (
            f"{action} nie jest wykonywane na koncie {who} — skutek błędu byłby "
            "systemowy, a nie ograniczony do plików jednego użytkownika. "
            "Uruchom asystenta na zwykłym koncie."
        )
    if state is None and strict:
        return (
            f"{action} wymaga pewności, że nie działamy z uprawnieniami "
            "administratora, a tego na tym systemie nie udało się ustalić."
        )
    return None


def describe(platform_info: PlatformInfo | None = None) -> str:
    """Jedna linijka do ``/narzedzia`` i do raportu zależności."""
    info = platform_info or detect_platform()
    label = account_label(info)
    if is_privileged(info) is False:
        return f"{label} — narzędzia HIGH/CRITICAL dozwolone"
    return f"{label} — narzędzia HIGH/CRITICAL zablokowane"


__all__ = [
    "account_label",
    "describe",
    "is_privileged",
    "refuse_if_privileged",
]
