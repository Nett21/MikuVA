"""Interfejs graficzny asystenta (Faza 10).

Pakiet jest **opcjonalny**: asystent uruchamia się i działa bez niego (tryb
terminalowy z Faz 1–9). Ten plik jest jedynym miejscem, które wie, że GUI może się
nie dać uruchomić, i potrafi to powiedzieć po ludzku.

Trzy różne braki, trzy różne komunikaty — bo każdy naprawia się inaczej:

``customtkinter``
    zwykły pakiet Pythona, instaluje ``pip``;
``tkinter`` / Tk
    biblioteka **systemowa** (``libtk``). ``pip`` jej nie zainstaluje: na
    Debianie to ``python3-tk``, na Archu ``tk``, na Fedorze ``python3-tkinter``,
    na macOS-ie ``python-tk`` z Homebrew, a na Windowsie opcja „tcl/tk and IDLE"
    w instalatorze Pythona;
``brak ekranu``
    poprawna instalacja, ale nie ma gdzie rysować (serwer bez pulpitu, SSH bez
    przekierowania X11, usługa systemowa). Wtedy jedyną sensowną odpowiedzią jest
    „uruchom tryb terminalowy".

Importy CustomTkintera są **leniwe** — samo ``import gui`` nie wciąga tkintera,
więc ``main.py --check-deps`` działa też tam, gdzie Tk nie istnieje.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Final

from i18n import t

logger = logging.getLogger(__name__)

TOOLKIT_PACKAGE: Final[str] = "customtkinter"

# Zmienne, po których na systemach uniksowych widać sesję graficzną. Świadomie
# obie: sesja Waylanda bez Xwaylanda ma tylko WAYLAND_DISPLAY, a klasyczny X11 —
# tylko DISPLAY. Na Windowsie i macOS-ie nie ma czego sprawdzać.
_DISPLAY_VARIABLES: Final[tuple[str, ...]] = ("DISPLAY", "WAYLAND_DISPLAY")


@dataclass(frozen=True, slots=True)
class ToolkitStatus:
    """Czy da się otworzyć okno na TEJ maszynie i co zrobić, jeśli nie."""

    ok: bool
    tk_available: bool
    toolkit_available: bool
    display_available: bool
    detail: str = ""
    hint: str = ""

    def describe(self) -> str:
        return self.detail if not self.ok else t("gui.ready")


def _has_display() -> bool:
    """Czy jest gdzie rysować? Na Windowsie i macOS-ie zakładamy, że tak.

    Nie próbujemy tu tworzyć okna: na maszynie z pulpitem mignęłoby ono na
    ekranie przy każdym ``--check-deps``. Prawdziwą weryfikacją jest próba
    otwarcia okna w :func:`run_gui`, która łapie ``TclError``.
    """
    if sys.platform.startswith(("win", "darwin")):
        return True
    return any(os.environ.get(variable, "").strip() for variable in _DISPLAY_VARIABLES)


def _tk_hint() -> str:
    """Jak zainstalować Tk na tym konkretnym systemie."""
    try:
        from config import OSFamily, PackageManager, detect_platform

        info = detect_platform()
        manager = info.package_manager
        commands = {
            PackageManager.APT: "sudo apt install python3-tk",
            PackageManager.PACMAN: "sudo pacman -S tk",
            PackageManager.DNF: "sudo dnf install python3-tkinter",
            PackageManager.ZYPPER: "sudo zypper install python3-tk",
            PackageManager.APK: "sudo apk add python3-tkinter",
            PackageManager.BREW: "brew install python-tk",
        }
        if manager in commands:
            return commands[manager]
        if info.os_family is OSFamily.WINDOWS:
            return t("gui.hint.tk_windows")
    except Exception:  # pragma: no cover - detekcja nie może blokować komunikatu
        logger.debug("Nie udało się rozpoznać systemu dla podpowiedzi o Tk", exc_info=True)
    return t("gui.hint.tk_generic")


def toolkit_status() -> ToolkitStatus:
    """Sprawdź, czy GUI ma z czego działać. Nie rzuca i nie otwiera okna."""
    tk_available = True
    tk_detail = ""
    try:
        import tkinter  # noqa: F401 - sprawdzamy sam import
    except Exception as exc:  # ImportError, ale też OSError przy braku libtk
        tk_available = False
        tk_detail = str(exc)

    toolkit_available = True
    toolkit_detail = ""
    try:
        import customtkinter  # noqa: F401 - sprawdzamy sam import
    except Exception as exc:
        toolkit_available = False
        toolkit_detail = str(exc)

    display = _has_display()

    if not tk_available:
        return ToolkitStatus(
            ok=False,
            tk_available=False,
            toolkit_available=toolkit_available,
            display_available=display,
            detail=t("gui.missing_tk", error=tk_detail),
            hint=_tk_hint(),
        )
    if not toolkit_available:
        from config import pip_install_hint

        return ToolkitStatus(
            ok=False,
            tk_available=True,
            toolkit_available=False,
            display_available=display,
            detail=t("gui.missing_toolkit", package=TOOLKIT_PACKAGE, error=toolkit_detail),
            hint=f"{pip_install_hint()}  albo: python -m pip install {TOOLKIT_PACKAGE}",
        )
    if not display:
        return ToolkitStatus(
            ok=False,
            tk_available=True,
            toolkit_available=True,
            display_available=False,
            detail=t("gui.no_session"),
            hint=t("gui.terminal_hint"),
        )
    return ToolkitStatus(
        ok=True,
        tk_available=True,
        toolkit_available=True,
        display_available=True,
        detail=t("gui.ready"),
    )


def run_gui(
    settings: Any,
    report: Any = None,
    *,
    speech_enabled: bool = True,
    start_in_voice_mode: bool = False,
) -> int:
    """Uruchom okno asystenta. Zwraca kod wyjścia procesu.

    ``2`` znaczy „GUI nie da się uruchomić na tej maszynie" — z wypisanym powodem
    i podpowiedzią. Program nigdy nie kończy się w tym miejscu stack trace'em:
    brak Tk to stan środowiska, nie błąd asystenta.
    """
    from config import TAG_ERROR, TAG_SYSTEM

    status = toolkit_status()
    if not status.ok:
        print(f"{TAG_ERROR} {t('gui.unavailable', reason=status.detail)}")
        if status.hint:
            print(f"{TAG_SYSTEM} {status.hint}")
        print(f"{TAG_SYSTEM} {t('gui.terminal_hint')}")
        return 2

    from gui.app import run_window

    try:
        return run_window(
            settings,
            report,
            speech_enabled=speech_enabled,
            start_in_voice_mode=start_in_voice_mode,
        )
    except Exception as exc:
        # ``TclError`` (i pochodne) to najczęściej brak dostępu do ekranu — nawet
        # gdy zmienne środowiskowe sugerowały inaczej (np. cudzy DISPLAY, do
        # którego nie mamy uprawnień).
        name = type(exc).__name__
        if name == "TclError":
            print(f"{TAG_ERROR} {t('gui.window_failed', error=exc)}")
            print(f"{TAG_SYSTEM} {t('gui.no_display_hint')}")
            return 2
        logger.exception("Interfejs graficzny zakończył się błędem")
        print(f"{TAG_ERROR} {t('gui.crashed', error=exc)}")
        return 1


__all__ = ["TOOLKIT_PACKAGE", "ToolkitStatus", "run_gui", "toolkit_status"]
