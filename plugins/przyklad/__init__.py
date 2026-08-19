"""Szkielet pluginu — skopiuj ten katalog i zacznij od niego (Faza 11).

Ten plugin nic nie robi i to jest zamierzone: pokazuje WYŁĄCZNIE strukturę.
Zostaw go w repozytorium jako wzór albo skopiuj pod własną nazwą::

    cp -r plugins/przyklad plugins/moj_plugin

Plugin składa się z trzech rzeczy:

1. **wizytówki** (:class:`plugins.manager.PluginInfo`) — nazwa, opis, czego wymaga,
2. **narzędzi** — zwykłych narzędzi z Fazy 7 (``tools/base.py``); przechodzą przez
   ten sam router, walidację argumentów, budżet tury i potwierdzenia,
3. **obiektu ``PLUGIN``** (albo funkcji ``create_plugin()``), który menedżer znajdzie.

Trzy zasady, których warto się trzymać:

* **Ryzyko deklaruje się uczciwie.** Domyślne ryzyko to CRITICAL, czyli
  zablokowane — to nie jest złośliwość, tylko wybór strony, po której ma być
  błąd. SAFE = tylko odczyt, nic nie zmienia. MEDIUM = zmienia coś odwracalnie.
  HIGH = skutki są trudne albo niemożliwe do cofnięcia i użytkownik MUSI
  potwierdzić. Ryzyko można podnieść po zajrzeniu w argumenty (``dynamic_risk``),
  ale nigdy obniżyć.
* **Nie licz na to, że coś jest zainstalowane.** Sprawdź to w ``available()`` i
  powiedz, czego brakuje. Plugin, którego nie da się użyć, ma o tym MÓWIĆ, a nie
  wywalać się przy pierwszym wywołaniu.
* **Nie trzymaj stanu w plikach obok kodu.** Od tego jest baza z Fazy 5, którą
  dostajesz w :class:`plugins.manager.PluginContext` — przeżyje restart, trafi do
  kopii zapasowej i nie zaśmieci katalogu z programem. Wzór: ``plugins/reminders``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from plugins.manager import BasePlugin, PluginContext, PluginInfo, PluginNotice
from tools.base import Tool

# ——— Krok 1: wizytówka ————————————————————————————————————————————————————— #

INFO = PluginInfo(
    name="przyklad",
    description="Pusty szkielet pluginu — punkt startowy do własnych rozszerzeń.",
    version="1.0",
    requires="niczego",
)


# ——— Krok 2: narzędzia ————————————————————————————————————————————————————— #
#
# Tutaj zbuduj i zwróć swoje narzędzia. Wzór najprostszego narzędzia:
#
#     from pydantic import Field
#     from security.risk import RiskLevel
#     from tools.base import BaseTool, ToolArgs, ToolContext, ToolResult, ToolSpec
#
#     class PowitanieArgs(ToolArgs):
#         imie: str = Field(default="", max_length=60)
#
#     class PowitanieTool(BaseTool[PowitanieArgs]):
#         async def run(self, args: PowitanieArgs, ctx: ToolContext) -> ToolResult:
#             kogo = args.imie or "świecie"
#             return ToolResult.success({"tekst": f"Cześć, {kogo}!"},
#                                       display=f"Cześć, {kogo}!")
#
#     def build_tools() -> list[Tool[Any]]:
#         return [PowitanieTool(ToolSpec(
#             name="przyklad.powitanie",              # obszar.czynność, małymi literami
#             description="Say hello. Example tool from the plugin skeleton.",
#             args_model=PowitanieArgs,
#             risk=RiskLevel.SAFE,                     # nic nie zmienia w świecie
#         ))]
#
# Opis (``description``) czyta MODEL — pisz go po angielsku i konkretnie, bo to
# na jego podstawie model decyduje, kiedy narzędzia użyć.


class PrzykladowyPlugin(BasePlugin):
    """Plugin, który świadomie nie wystawia żadnego narzędzia."""

    def __init__(self) -> None:
        super().__init__(INFO)

    def tools(self, ctx: PluginContext) -> Sequence[Tool[Any]]:
        # Zwróć tutaj listę narzędzi (patrz wzór wyżej). Pusta lista jest
        # poprawna: plugin może służyć wyłącznie do powiadomień z ``poll()``.
        return ()

    def available(self, ctx: PluginContext) -> tuple[bool, str]:
        """Czy plugin ma sens na tej maszynie?

        Zwróć ``(False, "powód")``, gdy czegoś brakuje — brakuje klucza w ``.env``,
        nie ma opcjonalnej biblioteki, sprzęt jest inny. Powód zobaczy użytkownik
        w raporcie zależności, więc niech mówi, co zrobić.
        """
        return True, ""

    def poll(self, ctx: PluginContext) -> Sequence[PluginNotice]:
        """Czy coś się wydarzyło od ostatniego sprawdzenia?

        Wołane co jakiś czas przez interfejs. Zwróć powiadomienia, gdy coś ma
        się odezwać samo z siebie (minął termin, przyszła wiadomość). Ma być
        SZYBKIE i nie może rzucać wyjątkiem — działający wzór: ``plugins/reminders``.
        """
        return ()


# ——— Krok 3: to znajduje menedżer ——————————————————————————————————————————— #
#
# Wystarczy jedno z dwojga: stała ``PLUGIN`` albo funkcja ``create_plugin()``.
# Fabryka przydaje się, gdy plugin musi coś policzyć przy starcie.

PLUGIN = PrzykladowyPlugin()


def create_plugin() -> PrzykladowyPlugin:
    return PLUGIN


__all__ = ["INFO", "PLUGIN", "PrzykladowyPlugin", "create_plugin"]
