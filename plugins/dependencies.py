"""Pluginy w raporcie ``python main.py --check-deps`` (Faza 11).

Plugin, którego nie widać, jest gorszy od plugina, którego nie ma: użytkownik
nie wie, czy zapomniał go włączyć, czy brakuje tokenu, czy po prostu się nie
ładuje. Dlatego raport pokazuje **wszystkie znalezione** pluginy razem z powodem,
dla którego któryś nie działa.

Sprawdzenie jest tanie i bez skutków ubocznych: ładuje moduły pluginów i pyta je
o dostępność. Nie wywołuje żadnego narzędzia i nie łączy się z niczym — pytanie
„czy Home Assistant odpowiada" celowo NIE pada tutaj, bo opóźniałoby raport o
limit czasu za każdym uruchomieniem.
"""

from __future__ import annotations

import logging
from typing import Any

from config import (
    DependencyCheck,
    DependencyContext,
    Settings,
    register_dependency_check,
)

logger = logging.getLogger(__name__)


def _open_database_for_check(settings: Settings) -> Any | None:
    """Baza na potrzeby SAMEGO sprawdzenia — bez migracji i bez zapisów.

    Pluginy trzymające stan (przypomnienia) pytają w ``available()``, czy baza
    jest. Bez niej raport mówiłby „niedostępny" nawet wtedy, gdy w normalnym
    uruchomieniu plugin działa — czyli straszyłby użytkownika bez powodu.
    ``apply_schema=False`` gwarantuje, że sprawdzenie niczego nie zmienia.
    """
    active = getattr(settings, "memory_enabled", False)
    if not active:
        return None
    try:
        from config import database_file
        from database.database import Database

        return Database(database_file(settings), apply_schema=False)
    except Exception as exc:  # pragma: no cover - zależne od uprawnień do pliku
        logger.info("Sprawdzenie pluginów bez bazy (%s)", exc)
        return None


@register_dependency_check
def check_plugins(context: DependencyContext) -> list[DependencyCheck]:
    """Jedna pozycja zbiorcza + po jednej na każdy znaleziony plugin."""
    settings = context.settings
    if not settings.plugins_enabled:
        return [
            DependencyCheck(
                name="Pluginy",
                category="feature",
                required=False,
                ok=False,
                detail="wyłączone (PLUGINS_ENABLED=false)",
                hint="ustaw PLUGINS_ENABLED=true, żeby wczytać rozszerzenia z plugins/",
                phase=11,
            )
        ]

    database = _open_database_for_check(settings)
    try:
        from plugins.manager import PluginContext, PluginManager

        manager = PluginManager(
            settings, context=PluginContext(settings=settings, database=database)
        )
        loaded = manager.load()
    except Exception as exc:  # pragma: no cover - awaria całej warstwy
        logger.warning("Nie udało się sprawdzić pluginów: %s", exc)
        return [
            DependencyCheck(
                name="Pluginy",
                category="feature",
                required=False,
                ok=False,
                detail=f"warstwa pluginów niedostępna ({exc})",
                phase=11,
            )
        ]
    finally:
        if database is not None:
            try:
                database.close()
            except Exception:  # pragma: no cover - zamykanie nie może zepsuć raportu
                logger.debug("Nie udało się zamknąć bazy po sprawdzeniu pluginów")

    checks: list[DependencyCheck] = [
        DependencyCheck(
            name="Pluginy",
            category="feature",
            required=False,
            ok=any(item.ok for item in loaded),
            detail=manager.describe(),
            hint="" if loaded else "katalog plugins/ jest pusty",
            phase=11,
        )
    ]

    for item in loaded:
        narzedzia = ", ".join(tool.spec.name for tool in item.tools)
        if item.ok:
            detail = narzedzia or "bez narzędzi (tylko powiadomienia)"
        else:
            detail = item.error or item.disabled_reason
        checks.append(
            DependencyCheck(
                name=f"Plugin: {item.name}",
                category="feature",
                required=False,
                ok=item.ok,
                detail=detail,
                # Powód wyłączenia jest zarazem podpowiedzią — pluginy piszą go
                # tak, żeby mówił, czego brakuje (patrz plugins/przyklad).
                hint=item.disabled_reason if item.error else "",
                path=str(item.source) if item.source else "",
                phase=11,
            )
        )
    return checks


__all__ = ["check_plugins"]
