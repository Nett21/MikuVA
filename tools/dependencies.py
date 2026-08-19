"""Sprawdzenia Fazy 7 dopięte do ``python main.py --check-deps``.

Import tego modułu dokłada pozycje do raportu zależności i do
``config/dependency_status.json``. Sprawdzenia są **tanie i bez skutków
ubocznych**: budują rejestr narzędzi (to czysty kod Pythona), czytają politykę
z konfiguracji i sprawdzają, czy jest z kim rozmawiać o potwierdzeniach. Żadne
narzędzie nie jest przy tym wywoływane.

Pozycje są opcjonalne: brak narzędzi ma odebrać modelowi możliwość ich użycia,
a nie zatrzymać asystenta.
"""

from __future__ import annotations

import logging

from config import (
    DependencyCheck,
    DependencyContext,
    register_dependency_check,
)
from i18n import t

logger = logging.getLogger(__name__)


def _check_tools(context: DependencyContext) -> DependencyCheck:
    """Ile narzędzi widzi model i z jakim ryzykiem."""
    settings = context.settings
    if not settings.tools_enabled:
        return DependencyCheck(
            name=t("deps.tools.name"),
            category="package",
            required=False,
            ok=False,
            detail=t("deps.tools.disabled"),
            hint=t("deps.tools.disabled_hint"),
            phase=7,
        )

    try:
        from security.policy import SecurityPolicy
        from tools.registry import build_registry

        registry = build_registry(settings)
        policy = SecurityPolicy(settings)
        visible = registry.visible(policy)
    except Exception as exc:  # pragma: no cover - błąd rejestracji narzędzia
        logger.warning("Nie udało się zbudować rejestru narzędzi: %s", exc)
        return DependencyCheck(
            name=t("deps.tools.name"),
            category="package",
            required=False,
            ok=False,
            detail=t("deps.tools.registry_failed", error=exc),
            hint=t("deps.tools.registry_hint"),
            phase=7,
        )

    names = ", ".join(tool.spec.name for tool in visible) or t("common.none")
    hidden = len(registry) - len(visible)
    detail = t("deps.tools.visible", visible=len(visible), total=len(registry), names=names)
    if hidden:
        detail += t("deps.tools.hidden", hidden=hidden)
    return DependencyCheck(
        name=t("deps.tools.name"),
        category="package",
        required=False,
        ok=bool(visible),
        detail=detail,
        hint="" if visible else t("deps.tools.all_disabled"),
        phase=7,
    )


def _check_permissions(context: DependencyContext) -> DependencyCheck:
    """Czy jest komu zadać pytanie o zgodę na narzędzia HIGH/CRITICAL."""
    from security.confirm import TerminalConfirmationBroker
    from security.policy import SecurityPolicy

    policy = SecurityPolicy(context.settings)
    interactive = TerminalConfirmationBroker().available
    detail = policy.describe()
    if interactive:
        detail += t("deps.perm.terminal")
    else:
        detail += t("deps.perm.no_terminal")
    return DependencyCheck(
        name=t("deps.perm.name"),
        category="feature",
        required=False,
        # Brak kanału potwierdzeń nie jest awarią: narzędzia SAFE/MEDIUM działają,
        # a wyższe są odrzucane — dokładnie tak, jak ma być bez zgody człowieka.
        ok=True,
        detail=detail,
        phase=7,
    )


def _check_workspace(context: DependencyContext) -> DependencyCheck:
    """Gdzie narzędzia plikowe mogą pracować (Faza 8)."""
    from host.paths import Workspace

    workspace = Workspace.from_settings(context.settings)
    roots = list(workspace.roots)
    existing = [root for root in roots if root.is_dir()]
    detail = t(
        "deps.workspace.detail",
        count=len(roots),
        roots=", ".join(str(root) for root in roots),
    )
    if not existing:
        detail += t("deps.workspace.missing")
    return DependencyCheck(
        name=t("deps.workspace.name"),
        category="storage",
        required=False,
        # Brak katalogu nie jest awarią: powstanie przy pierwszym zapisie.
        ok=True,
        detail=detail,
        path=str(workspace.primary),
        hint=(
            ""
            if len(roots) > 1
            else t("deps.workspace.hint")
        ),
        phase=8,
    )


def _check_host_backends(context: DependencyContext) -> DependencyCheck:
    """Czym na tej maszynie da się uruchamiać aplikacje, czytać procesy i usługi."""
    from host.apps import describe_backend as apps_backend
    from host.processes import describe_backend as processes_backend
    from host.services import describe_backend as services_backend
    from host.shell import describe_backend as shell_backend
    from tools.pdf import reader_backend

    parts = [
        t("deps.host.apps", detail=apps_backend(context.platform_info)),
        t("deps.host.processes", detail=processes_backend(context.platform_info)),
        t("deps.host.services", detail=services_backend(context.platform_info)),
        t(
            "deps.host.shell",
            detail=shell_backend(context.settings, platform_info=context.platform_info),
        ),
        t("deps.host.pdf", detail=reader_backend() or t("deps.host.pdf_missing")),
    ]
    return DependencyCheck(
        name=t("deps.host.name"),
        category="feature",
        required=False,
        ok=True,
        detail="; ".join(parts),
        phase=8,
    )


def _check_network_tools(context: DependencyContext) -> DependencyCheck:
    """Stan narzędzi sieciowych i to, co je włącza (Faza 9).

    Sprawdzenie **nie wychodzi do sieci**: pytanie „czy internet działa" i tak
    trzeba by zadać w chwili użycia, a raport zależności ma być szybki i bez
    skutków ubocznych. Sprawdzamy konfigurację, tryb pracy i obecność kluczy —
    przy czym samych kluczy nigdzie nie pokazujemy, tylko fakt ich obecności.
    """
    from host.http import describe_backend, network_available
    from tools.webtext import split_list

    settings = context.settings
    usable, reason = network_available(settings)
    if not usable:
        return DependencyCheck(
            name=t("deps.web.name"),
            category="feature",
            required=False,
            ok=False,
            detail=reason,
            hint=t("deps.web.hint"),
            phase=9,
        )

    feeds = split_list(settings.news_feeds)
    parts = [
        t("deps.web.search", provider=settings.search_provider),
        t("deps.web.weather", provider=settings.weather_provider),
        (
            t("deps.web.news_feeds", count=len(feeds))
            if feeds
            else t("deps.web.news_search")
        ),
        # O kluczach mówimy „jest/brak" — nigdy wartości. Klucze są w SecretStr,
        # więc nawet przypadkowe wypisanie obiektu nie pokaże treści.
        t(
            "deps.web.youtube_key",
            state=(
                t("common.available")
                if settings.youtube_api_key.get_secret_value()
                else t("common.missing")
            ),
        ),
    ]
    return DependencyCheck(
        name=t("deps.web.name"),
        category="feature",
        required=False,
        ok=True,
        detail="; ".join(parts) + f"; {describe_backend(settings)}",
        phase=9,
    )


@register_dependency_check
def check_tools(context: DependencyContext) -> list[DependencyCheck]:
    """Sprawdzenia Faz 7–9 zebrane w jedną pozycję rejestru."""
    checks = [_check_tools(context), _check_permissions(context)]
    try:
        checks.append(_check_workspace(context))
        checks.append(_check_host_backends(context))
        checks.append(_check_network_tools(context))
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Nie udało się sprawdzić narzędzi systemowych: %s", exc)
    return checks


__all__ = ["check_tools"]
