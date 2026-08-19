"""Narzędzia Home Assistanta: odczyt stanu i przełączanie (Faza 11).

================= ========== ===============================================
narzędzie         poziom     uzasadnienie
================= ========== ===============================================
``ha.state``      SAFE       odczyt jednej encji, nic nie zmienia
``ha.list``       SAFE       lista encji (z limitem), nic nie zmienia
``ha.switch``     MEDIUM     światło, gniazdko, wentylator — odwracalne
``ha.switch``     **HIGH**   zamek, brama, roleta, alarm, zawór, bojler
================= ========== ===============================================

Ostatni wiersz jest powodem, dla którego ten plugin w ogóle wygląda tak, jak
wygląda. W API Home Assistanta ``lock.unlock`` i ``light.turn_on`` to to samo
wywołanie usługi — różnią się jednym słowem. Dla człowieka to różnica między
„zapaliłem lampkę" a „otworzyłem drzwi wejściowe, gdy nie było mnie w domu".

Dlatego poziom ryzyka nie jest stały: liczy go :meth:`SwitchTool.dynamic_risk` po
zajrzeniu w argumenty, na podstawie DOMENY encji. Eskalacja jest jednokierunkowa
(``BaseTool.effective_risk``), więc żaden argument nie obniży wymagań, a lista
domen wysokiego ryzyka jest w konfiguracji (``HOME_ASSISTANT_HIGH_RISK_DOMAINS``)
— kto ma w domu coś, czego nie przewidzieliśmy, dopisze to bez zmiany kodu.

Potwierdzenie dla HIGH nie jest tu implementowane: robi to router z Fazy 7,
dokładnie tak samo jak dla ``fs.delete``. Narzędzie buduje tylko TREŚĆ pytania.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from pydantic import Field, field_validator

from config import Settings, get_settings
from plugins.home_assistant.client import (
    EntityState,
    HomeAssistantClient,
    HomeAssistantError,
)
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from i18n import t

logger = logging.getLogger(__name__)

# Usługi, na które pozwalamy. Świadomie WĄSKA lista: „wywołaj dowolną usługę"
# dałoby modelowi dostęp do wszystkiego, co potrafi Home Assistant — łącznie z
# restartem instancji i uruchamianiem skryptów użytkownika.
ALLOWED_ACTIONS: Final[dict[str, str]] = {
    "on": "turn_on",
    "off": "turn_off",
    "toggle": "toggle",
}

# Domeny, w których „włącz/wyłącz" nie ma sensu albo jest niebezpieczne bez
# jawnej zgody. Wartość domyślna pokrywa typowy dom; resztę dopisuje użytkownik.
DEFAULT_HIGH_RISK: Final[tuple[str, ...]] = (
    "lock",
    "cover",
    "gate",
    "alarm_control_panel",
    "valve",
    "water_heater",
)


def high_risk_domains(settings: Settings | None = None) -> frozenset[str]:
    """Domeny encji, których przełączenie wymaga zgody użytkownika."""
    active = settings or get_settings()
    raw = active.home_assistant_high_risk_domains
    names = {item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()}
    # Wartości z konfiguracji DODAJEMY do wbudowanych, a nie zastępujemy nimi:
    # literówka w .env nie może po cichu obniżyć wymagań dla zamka w drzwiach.
    return frozenset(names | set(DEFAULT_HIGH_RISK))


class EntityArgs(ToolArgs):
    entity_id: str = Field(min_length=3, max_length=120)

    @field_validator("entity_id")
    @classmethod
    def _looks_like_entity(cls, value: str) -> str:
        text = value.strip().lower()
        if "." not in text:
            raise ValueError(t("ha.bad_entity_id"))
        return text


class ListArgs(ToolArgs):
    domain: str = Field(default="", max_length=40)
    limit: int = Field(default=30, ge=1, le=200)


class SwitchArgs(ToolArgs):
    entity_id: str = Field(min_length=3, max_length=120)
    action: str = Field(default="on", max_length=10)

    @field_validator("entity_id")
    @classmethod
    def _looks_like_entity(cls, value: str) -> str:
        text = value.strip().lower()
        if "." not in text:
            raise ValueError(t("ha.bad_entity_id"))
        return text

    @field_validator("action")
    @classmethod
    def _known_action(cls, value: str) -> str:
        text = value.strip().lower()
        if text not in ALLOWED_ACTIONS:
            allowed = ", ".join(sorted(ALLOWED_ACTIONS))
            raise ValueError(f"dozwolone akcje: {allowed}")
        return text

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]


class _HomeAssistantTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza: wspólny klient i wspólna zamiana błędów na komunikat dla modelu."""

    def __init__(self, spec: ToolSpec, client: HomeAssistantClient) -> None:
        super().__init__(spec)
        self._client = client

    def available(self) -> tuple[bool, str]:
        if not self._client.config.configured:
            return False, (
                "Home Assistant nie jest skonfigurowany "
                "(ustaw HOME_ASSISTANT_URL i HOME_ASSISTANT_TOKEN w .env)"
            )
        return True, ""

    async def guarded(self, coroutine: Any) -> Any:
        """Wykonaj operację, zamieniając awarię serwera w czytelny błąd.

        Niedostępny Home Assistant to normalna sytuacja (serwer się aktualizuje,
        ktoś wyjął wtyczkę switcha, sieć siadła) — asystent ma o tym POWIEDZIEĆ,
        a nie przerwać rozmowę wyjątkiem.
        """
        try:
            return await coroutine
        except HomeAssistantError as exc:
            raise ToolError(exc.user_message) from exc


class StateTool(_HomeAssistantTool[EntityArgs]):
    """Odczytaj stan jednej encji."""

    async def run(self, args: EntityArgs, ctx: ToolContext) -> ToolResult:
        entity: EntityState = await self.guarded(self._client.state(args.entity_id))
        return ToolResult.success(
            {
                "entity_id": entity.entity_id,
                "state": entity.state,
                "name": entity.friendly_name,
                "changed_at": entity.changed_at,
            },
            display=entity.describe(),
            # Nazwy encji i ich stany pochodzą z konfiguracji użytkownika, ale
            # nazwę encji nadaje człowiek — traktujemy to jak treść z zewnątrz.
            untrusted=True,
        )


class ListTool(_HomeAssistantTool[ListArgs]):
    """Pokaż encje (opcjonalnie z jednej domeny)."""

    async def run(self, args: ListArgs, ctx: ToolContext) -> ToolResult:
        entities: list[EntityState] = await self.guarded(
            self._client.states(domain=args.domain, limit=args.limit)
        )
        if not entities:
            where = f" ({args.domain})" if args.domain else ""
            return ToolResult.success(
                {"entities": []}, display=t("ha.no_entities", detail=where)
            )
        return ToolResult.success(
            {
                "entities": [
                    {"entity_id": item.entity_id, "state": item.state, "name": item.friendly_name}
                    for item in entities
                ]
            },
            display="\n".join(item.describe() for item in entities),
            untrusted=True,
        )


class SwitchTool(_HomeAssistantTool[SwitchArgs]):
    """Włącz, wyłącz albo przełącz encję."""

    def __init__(
        self, spec: ToolSpec, client: HomeAssistantClient, *, high_risk: frozenset[str]
    ) -> None:
        super().__init__(spec, client)
        self._high_risk = high_risk

    def dynamic_risk(self, args: SwitchArgs) -> RiskLevel:
        """HIGH dla zamków, bram, rolet i alarmów — MEDIUM dla reszty.

        To jedyne miejsce w pluginie, które decyduje o tym, czy użytkownik
        zostanie zapytany o zgodę. Reszta (samo pytanie, limit czasu, audyt)
        jest po stronie routera z Fazy 7.
        """
        return RiskLevel.HIGH if args.domain in self._high_risk else RiskLevel.MEDIUM

    def confirmation(self, args: SwitchArgs, *, language: str = "en") -> ConfirmationRequest | None:
        risk = self.effective_risk(args)
        action = ALLOWED_ACTIONS[args.action]
        if language == "pl":
            czynnosc = {"turn_on": "włączy", "turn_off": "wyłączy", "toggle": "przełączy"}[action]
            summary = f"{czynnosc} {args.entity_id}"
            details = [f"Home Assistant: {self._client.config.base_url}"]
            if risk is RiskLevel.HIGH:
                details.append("to urządzenie należy do grupy podwyższonego ryzyka")
        else:
            summary = f"{action.replace('_', ' ')} {args.entity_id}"
            details = [f"Home Assistant: {self._client.config.base_url}"]
            if risk is RiskLevel.HIGH:
                details.append("this device is in the elevated-risk group")
        return ConfirmationRequest.build(
            tool=self.spec.name, risk=risk, summary=summary, details=details, language=language
        )

    async def preview(self, args: SwitchArgs, ctx: ToolContext) -> str:
        return f"{ALLOWED_ACTIONS[args.action]} → {args.entity_id}"

    async def run(self, args: SwitchArgs, ctx: ToolContext) -> ToolResult:
        service = ALLOWED_ACTIONS[args.action]
        changed: list[EntityState] = await self.guarded(
            self._client.call_service(args.domain, service, entity_id=args.entity_id)
        )
        after = next(
            (item for item in changed if item.entity_id == args.entity_id),
            None,
        )
        state = after.state if after is not None else "?"
        return ToolResult.success(
            {
                "entity_id": args.entity_id,
                "service": f"{args.domain}.{service}",
                "state": state,
                "changed": [item.entity_id for item in changed],
            },
            display=f"{args.entity_id} → {state}",
        )


def build_home_assistant_tools(
    client: HomeAssistantClient, *, settings: Settings | None = None
) -> list[Tool[Any]]:
    """Narzędzia pluginu. Opisy po angielsku — czyta je model."""
    active = settings or get_settings()
    risky = high_risk_domains(active)
    return [
        StateTool(
            ToolSpec(
                name="ha.state",
                description=(
                    "Read the current state of one Home Assistant entity, "
                    "e.g. light.kitchen or sensor.outside_temperature."
                ),
                args_model=EntityArgs,
                risk=RiskLevel.SAFE,
                requires_network=True,
                summary="Odczytaj stan encji Home Assistanta.",
            ),
            client,
        ),
        ListTool(
            ToolSpec(
                name="ha.list",
                description=(
                    "List Home Assistant entities, optionally limited to one domain "
                    "(light, switch, sensor, lock, cover). Use it to find entity ids."
                ),
                args_model=ListArgs,
                risk=RiskLevel.SAFE,
                requires_network=True,
                summary=t("spec.ha_list"),
            ),
            client,
        ),
        SwitchTool(
            ToolSpec(
                name="ha.switch",
                description=(
                    "Turn a Home Assistant entity on or off (action: on, off, toggle). "
                    "Works for lights, switches, fans and similar devices. Locks, covers, "
                    "gates and alarms require the user's confirmation."
                ),
                args_model=SwitchArgs,
                # Deklaracja to MEDIUM, a dynamic_risk podnosi ją do HIGH dla
                # domen wrażliwych. Odwrotnie się nie da — eskalacja jest
                # jednokierunkowa, więc nie ma jak zejść poniżej tej wartości.
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                idempotent=False,
                summary=t("spec.ha_switch"),
            ),
            client,
            high_risk=risky,
        ),
    ]


__all__ = [
    "ALLOWED_ACTIONS",
    "DEFAULT_HIGH_RISK",
    "EntityArgs",
    "ListArgs",
    "ListTool",
    "StateTool",
    "SwitchArgs",
    "SwitchTool",
    "build_home_assistant_tools",
    "high_risk_domains",
]
