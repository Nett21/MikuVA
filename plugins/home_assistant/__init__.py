"""Plugin Home Assistanta — sterowanie domem przez jego REST API (Faza 11).

Drugi z dwóch przykładowych pluginów i celowo przeciwieństwo pierwszego:
``reminders`` nie dotyka sieci ani konfiguracji, ten wymaga obu. Razem
pokazują, że kontrakt pluginu obejmuje oba przypadki bez wyjątków „na skróty".

Konfiguracja (``.env``)::

    HOME_ASSISTANT_URL=http://homeassistant.local:8123
    HOME_ASSISTANT_TOKEN=<long-lived access token z profilu użytkownika>

Bez obu tych pól plugin **sam się wyłącza**: model nie zobaczy ani jednego jego
narzędzia, a użytkownik dostanie w raporcie zależności zdanie o tym, czego
brakuje. Świadomie NIE ma tu żadnej wartości domyślnej adresu — instalacja
każdego jest inna, a zgadywanie ``localhost:8123`` kończyłoby się pukaniem do
przypadkowego portu na cudzej maszynie.

Poziomy ryzyka są opisane w :mod:`plugins.home_assistant.tools`. Najważniejsze:
przełączenie zamka, bramy, rolety czy alarmu jest HIGH i przechodzi przez
potwierdzenie z Fazy 7 — tak samo jak usunięcie pliku.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from plugins.home_assistant.client import (
    HomeAssistantClient,
    HomeAssistantConfig,
    HomeAssistantError,
)
from plugins.home_assistant.tools import build_home_assistant_tools
from plugins.manager import BasePlugin, PluginContext, PluginInfo
from tools.base import Tool

logger = logging.getLogger(__name__)

INFO = PluginInfo(
    name="home_assistant",
    description="Odczyt stanu i sterowanie urządzeniami w Home Assistancie.",
    version="1.0",
    requires="HOME_ASSISTANT_URL i HOME_ASSISTANT_TOKEN w .env",
)


class HomeAssistantPlugin(BasePlugin):
    """Narzędzia do rozmowy z własną instancją Home Assistanta."""

    def __init__(self) -> None:
        super().__init__(INFO)

    def available(self, ctx: PluginContext) -> tuple[bool, str]:
        """Bez adresu i tokenu plugin jest nieaktywny — i mówi dlaczego.

        Sprawdzamy KONFIGURACJĘ, nie połączenie: pytanie serwera przy starcie
        opóźniałoby uruchomienie asystenta o limit czasu, a chwilowo wyłączony
        Home Assistant nie powinien znikać z listy narzędzi na całą sesję.
        Nieosiągalny serwer w momencie wywołania kończy się czytelnym błędem
        narzędzia (patrz ``_HomeAssistantTool.guarded``).
        """
        try:
            config = HomeAssistantConfig.from_settings(ctx.settings)
        except HomeAssistantError as exc:
            return False, exc.user_message
        if not config.base_url:
            return False, "brak adresu — ustaw HOME_ASSISTANT_URL w .env"
        if not config.token:
            return False, "brak tokenu — ustaw HOME_ASSISTANT_TOKEN w .env"
        return True, ""

    def tools(self, ctx: PluginContext) -> Sequence[Tool[Any]]:
        # Klient bierzemy z kontekstu, jeśli ktoś go podstawił (tak robią testy
        # i tak może zrobić inny plugin). Inaczej budujemy z ustawień.
        client = ctx.extras.get("home_assistant_client")
        if not isinstance(client, HomeAssistantClient):
            client = HomeAssistantClient(settings=ctx.settings)
        return build_home_assistant_tools(client, settings=ctx.settings)


PLUGIN = HomeAssistantPlugin()


def create_plugin() -> HomeAssistantPlugin:
    return PLUGIN


__all__ = ["INFO", "PLUGIN", "HomeAssistantPlugin", "create_plugin"]
