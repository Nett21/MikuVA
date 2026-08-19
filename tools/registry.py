"""Rejestr narzędzi: jedno miejsce, które wie, co istnieje (Faza 7).

Router nie importuje narzędzi z osobna — pyta rejestr. Dzięki temu ``brain/``
nie zależy od tego, jakie narzędzia są w danej wersji dostępne, a Faza 8 dopisze
swoje w jednym miejscu (:func:`build_registry`).

Rejestr **filtruje to, co widzi model**: narzędzie wyłączone w konfiguracji,
niedostępne na tej maszynie albo o ryzyku CRITICAL przy ``SECURITY_ALLOW_CRITICAL=false``
nie trafia do listy wysyłanej modelowi. Model nie wywoła czegoś, o czym nie wie —
a nawet gdyby wywołał na pamięć, bramka i tak by go zatrzymała.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Final

from config import Settings, get_settings
from security.policy import SecurityPolicy
from security.risk import describe_risk
from tools.base import Tool

logger = logging.getLogger(__name__)

# Nazwa narzędzia: ``obszar.czynność``, małymi literami. Kropka nie jest ozdobą —
# porządkuje listę, którą czyta model, i grupuje uprawnienia w konfiguracji.
NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]+)+$")


class ToolRegistry:
    """Zbiór narzędzi dostępnych w tym uruchomieniu."""

    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for item in tools:
            self.register(item)

    # --- rejestracja ------------------------------------------------------ #

    def register(self, tool: Tool[Any]) -> Tool[Any]:
        """Dodaj narzędzie. Zła nazwa albo duplikat to błąd programisty, nie danych."""
        name = tool.spec.name
        if not NAME_PATTERN.match(name):
            raise ValueError(
                f"nazwa narzędzia '{name}' musi mieć postać obszar.czynność (małymi literami)"
            )
        if name in self._tools:
            raise ValueError(f"narzędzie '{name}' jest już zarejestrowane")
        self._tools[name] = tool
        logger.debug("Zarejestrowano narzędzie %s (%s)", name, tool.spec.risk.value)
        return tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # --- odczyt ----------------------------------------------------------- #

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(str(name or "").strip())

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool[Any]]:
        return [self._tools[name] for name in self.names()]

    def visible(self, policy: SecurityPolicy | None = None) -> list[Tool[Any]]:
        """Narzędzia, które wolno pokazać modelowi."""
        active = policy or SecurityPolicy()
        visible: list[Tool[Any]] = []
        for tool in self.all():
            if not active.is_visible_to_llm(tool.spec.name, tool.spec.risk):
                continue
            usable, reason = tool.available()
            if not usable:
                logger.debug("Narzędzie %s pominięte: %s", tool.spec.name, reason)
                continue
            visible.append(tool)
        return visible

    def schemas_for_llm(self, policy: SecurityPolicy | None = None) -> list[dict[str, Any]]:
        """Deklaracje narzędzi w formacie ``tools`` Ollamy."""
        return [tool.spec.llm_schema() for tool in self.visible(policy)]

    def describe(
        self, policy: SecurityPolicy | None = None, *, language: str = "en"
    ) -> list[str]:
        """Lista narzędzi dla człowieka (``/narzedzia``) — z ryzykiem i stanem."""
        active = policy or SecurityPolicy()
        lines: list[str] = []
        for tool in self.all():
            spec = tool.spec
            note = spec.summary or spec.description
            state: list[str] = []
            enabled, reason = active.is_enabled(spec.name)
            if not enabled:
                state.append(reason)
            usable, unavailable_reason = tool.available()
            if not usable:
                state.append(unavailable_reason or "niedostępne na tej maszynie")
            if enabled and usable and not active.is_visible_to_llm(spec.name, spec.risk):
                state.append("ukryte przed modelem")
            suffix = f"  — {'; '.join(state)}" if state else ""
            lines.append(
                f"{spec.name}  [{describe_risk(spec.risk, language=language)}]  {note}{suffix}"
            )
        return lines

    # --- protokoły -------------------------------------------------------- #

    def __contains__(self, name: object) -> bool:
        return str(name) in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool[Any]]:
        return iter(self.all())

    def __repr__(self) -> str:
        return f"ToolRegistry({', '.join(self.names()) or 'pusty'})"


def build_registry(
    settings: Settings | None = None,
    *,
    extra: Sequence[Tool[Any]] = (),
    memory: Any | None = None,
    workspace: Any | None = None,
    plugins: Any | None = None,
) -> ToolRegistry:
    """Rejestr z narzędziami wbudowanymi w tę wersję programu.

    Grupy są rejestrowane niezależnie: awaria importu jednego modułu (brak
    opcjonalnej biblioteki, egzotyczny system) odbiera modelowi tę jedną grupę
    narzędzi, a nie cały rejestr.

    * ``memory`` — pamięć asystenta dla narzędzi notatek. Bez niej ``notes.*`` są
      niedostępne (model ich nie widzi),
    * ``workspace`` — obszar dla narzędzi plikowych (:class:`host.paths.Workspace`).
      Domyślnie budowany z ``FS_ALLOWED_ROOTS``; testy podstawiają własny katalog,
    * ``plugins`` — menedżer pluginów z Fazy 11 (:class:`plugins.manager.PluginManager`).
      ``None`` = zbuduj domyślny. To JEDYNE miejsce, w którym rdzeń wie o
      pluginach: kolejny plugin to nowy katalog, a nie zmiana w tym pliku.
    """
    active = settings or get_settings()
    registry = ToolRegistry()

    def _register(group: str, factory: Any) -> None:
        # Import i budowa w środku funkcji: brak jednej biblioteki nie może
        # wywrócić rejestracji pozostałych grup.
        try:
            for tool in factory():
                registry.register(tool)
        except Exception as exc:  # pragma: no cover - zależne od instalacji
            logger.warning("Nie zarejestrowano narzędzi (%s): %s", group, exc)

    def system_group() -> Sequence[Tool[Any]]:
        from tools.system import build_system_tools

        return build_system_tools(active)

    def filesystem_group() -> Sequence[Tool[Any]]:
        from tools.filesystem import build_filesystem_tools

        return build_filesystem_tools(active, workspace=workspace)

    def launcher_group() -> Sequence[Tool[Any]]:
        from tools.launcher import build_launcher_tools

        return build_launcher_tools(active, workspace=workspace)

    def notes_group() -> Sequence[Tool[Any]]:
        from tools.notes import build_notes_tools

        return build_notes_tools(active, memory=memory)

    def pdf_group() -> Sequence[Tool[Any]]:
        from tools.pdf import build_pdf_tools

        return build_pdf_tools(active, workspace=workspace)

    def shell_group() -> Sequence[Tool[Any]]:
        from tools.shell import build_shell_tools

        return build_shell_tools(active, workspace=workspace)

    def web_group() -> Sequence[Tool[Any]]:
        from tools.web import build_web_tools

        return build_web_tools(active)

    def weather_group() -> Sequence[Tool[Any]]:
        from tools.weather import build_weather_tools

        return build_weather_tools(active)

    def news_group() -> Sequence[Tool[Any]]:
        from tools.news import build_news_tools

        return build_news_tools(active)

    def youtube_group() -> Sequence[Tool[Any]]:
        from tools.youtube import build_youtube_tools

        return build_youtube_tools(active)

    _register("system", system_group)
    _register("pliki", filesystem_group)
    _register("uruchamianie", launcher_group)
    _register("notatki", notes_group)
    _register("pdf", pdf_group)
    _register("powłoka", shell_group)
    _register("sieć", web_group)
    _register("pogoda", weather_group)
    _register("wiadomości", news_group)
    _register("youtube", youtube_group)

    _register("pluginy", lambda: _plugin_tools(active, plugins, memory=memory, workspace=workspace))

    for tool in extra:
        registry.register(tool)
    return registry


def _plugin_tools(
    settings: Settings,
    manager: Any | None,
    *,
    memory: Any | None,
    workspace: Any | None,
) -> Sequence[Tool[Any]]:
    """Narzędzia z pluginów (Faza 11).

    Pluginy przechodzą przez ten sam rejestr, ten sam router i te same bramki co
    narzędzia wbudowane. Rejestracja jest ostatnia, więc plugin nie nadpisze
    narzędzia wbudowanego: duplikat nazwy kończy się błędem rejestracji, który
    ``_register`` zamienia w ostrzeżenie — dla pluginu, nie dla całego rejestru.
    """
    if not settings.plugins_enabled:
        return ()

    from plugins.manager import PluginContext, PluginManager

    if manager is None:
        database = getattr(memory, "database", None)
        manager = PluginManager(
            settings,
            context=PluginContext(
                settings=settings, database=database, memory=memory, workspace=workspace
            ),
        )
    return manager.tools()


__all__ = ["NAME_PATTERN", "ToolRegistry", "build_registry"]
