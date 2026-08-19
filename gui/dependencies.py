"""Sprawdzenia środowiska GUI dopisane do wspólnego mechanizmu z Fazy 1.

Import tego modułu rejestruje pozycje widoczne w ``python main.py --check-deps``
i w ``config/dependency_status.json``. Żadna nie jest wymagana — brak Tk ma
wyłączyć okno, a nie zatrzymać asystenta.

Trzy pozycje, bo to trzy różne problemy z trzema różnymi rozwiązaniami: pakiet
``customtkinter`` (pip), biblioteka systemowa Tk (menedżer pakietów systemu) i
sesja graficzna (której na serwerze po prostu nie ma).
"""

from __future__ import annotations

import logging

from config import (
    DependencyCheck,
    DependencyContext,
    PackageRequirement,
    pip_install_hint,
    register_dependency_check,
    register_package_requirement,
)
from gui import TOOLKIT_PACKAGE, toolkit_status
from i18n import t

logger = logging.getLogger(__name__)

# Pakiet trafia do wspólnej listy, więc raport zależności pokazuje go razem z
# resztą — z wersją, ścieżką i informacją, że jest opcjonalny.
register_package_requirement(
    PackageRequirement(
        TOOLKIT_PACKAGE,
        "customtkinter",
        required=False,
        phase=10,
        # Klucz, nie gotowy tekst: lista pakietów powstaje przy imporcie, a język
        # interfejsu bywa ustawiony później (config tłumaczy go przy renderowaniu).
        purpose="deps.gui.purpose",
    )
)


def _package_installed() -> bool:
    """Czy pakiet leży na dysku (bez jego importowania).

    ``find_spec`` samo w sobie NIE wykonuje modułu, więc odpowiada na pytanie
    „czy pip go zainstalował" nawet wtedy, gdy import wywala się na braku Tk.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("customtkinter") is not None
    except (ImportError, ValueError, AttributeError):  # pragma: no cover - skrajne
        return False


@register_dependency_check
def _check_gui(context: DependencyContext) -> list[DependencyCheck]:
    """Czy da się otworzyć okno: Tk, CustomTkinter i sesja graficzna."""
    status = toolkit_status()
    checks: list[DependencyCheck] = [
        DependencyCheck(
            name=t("deps.tk.name"),
            category="system",
            required=False,
            ok=status.tk_available,
            detail=t("deps.tk.ok") if status.tk_available else t("deps.tk.missing"),
            hint="" if status.tk_available else status.hint,
            phase=10,
        ),
        DependencyCheck(
            name=t("deps.gui.name"),
            category="package",
            required=False,
            ok=status.toolkit_available,
            # Rozróżnienie, które oszczędza szukania: pakiet bywa zainstalowany,
            # a mimo to nie da się go zaimportować, bo pod spodem brakuje Tk.
            # „Brak pakietu" wysłałoby użytkownika do pipa po nic.
            detail=(
                t("deps.gui.ok")
                if status.toolkit_available
                else (
                    t("deps.gui.needs_tk") if _package_installed() else t("deps.gui.missing")
                )
            ),
            hint=(
                ""
                if status.toolkit_available
                else (
                    status.hint
                    if _package_installed()
                    else f"{pip_install_hint(context.offline)}  "
                    f"albo: python -m pip install {TOOLKIT_PACKAGE}"
                )
            ),
            phase=10,
        ),
        DependencyCheck(
            name=t("deps.display.name"),
            category="system",
            required=False,
            ok=status.display_available,
            detail=(
                t("deps.display.ok")
                if status.display_available
                else t("deps.display.missing")
            ),
            hint="" if status.display_available else t("deps.display.hint"),
            phase=10,
        ),
    ]
    return checks


__all__ = ["TOOLKIT_PACKAGE"]
