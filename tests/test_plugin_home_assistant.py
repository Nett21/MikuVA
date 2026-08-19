"""Testy pluginu Home Assistant (Faza 11).

Żaden test nie wychodzi do sieci i żaden nie wymaga działającego Home Assistanta:
API jest podstawione przez ``httpx.MockTransport``, który odpowiada tak jak
prawdziwy serwer (razem z 401, 404 i awarią połączenia).

Trzy rzeczy, o które tu naprawdę chodzi:

1. **zamek to nie lampka** — przełączenie ``lock``/``cover``/``alarm`` jest HIGH
   i przechodzi przez potwierdzenie z Fazy 7, a ``light``/``switch`` to MEDIUM,
2. **token jest sekretem** — nie pojawia się w żadnym komunikacie ani w opisie
   konfiguracji, także wtedy, gdy serwer go odrzuci,
3. **niedostępny serwer to komunikat, nie awaria** — asystent ma powiedzieć, co
   się stało, i rozmawiać dalej.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from conftest import SpyBroker, frozen_clock

from brain.tool_router import ToolCall, ToolRouter
from config import Settings
from plugins.home_assistant import HomeAssistantPlugin
from plugins.home_assistant.client import (
    HomeAssistantClient,
    HomeAssistantConfig,
    HomeAssistantError,
    normalize_base_url,
    redact,
)
from plugins.home_assistant.tools import (
    EntityArgs,
    ListArgs,
    SwitchArgs,
    build_home_assistant_tools,
    high_risk_domains,
)
from plugins.manager import PluginContext
from security.policy import SecurityPolicy
from security.risk import RiskLevel
from tools.base import ToolContext, ToolError
from tools.registry import ToolRegistry

TOKEN = "sekretny-token-uzytkownika"
BASE = "http://dom.example:8123"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"home_assistant_url": BASE, "home_assistant_token": TOKEN}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def entity(entity_id: str, state: str, **attributes: Any) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": attributes or {"friendly_name": entity_id},
        "last_changed": "2026-08-17T12:00:00+00:00",
    }


class FakeHomeAssistant:
    """Atrapa serwera: pamięta stany i zapisuje, o co ją pytano."""

    def __init__(self, *, states: dict[str, str] | None = None) -> None:
        self.states = states or {"light.salon": "off", "lock.wejscie": "locked"}
        self.requests: list[httpx.Request] = []
        self.status_override: int | None = None
        self.connection_error = False

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.connection_error:
            raise httpx.ConnectError("nie ma takiego hosta", request=request)
        if self.status_override is not None:
            return httpx.Response(self.status_override, json={"message": "nie"})

        path = request.url.path
        if path == "/api/":
            return httpx.Response(200, json={"message": "API running."})
        if path.startswith("/api/states/"):
            entity_id = path.removeprefix("/api/states/")
            if entity_id not in self.states:
                return httpx.Response(404, json={"message": "Entity not found."})
            return httpx.Response(200, json=entity(entity_id, self.states[entity_id]))
        if path == "/api/states":
            return httpx.Response(
                200, json=[entity(name, state) for name, state in sorted(self.states.items())]
            )
        if path.startswith("/api/services/"):
            _, _, _, _domain, service = path.split("/")
            body = json.loads(request.content or b"{}")
            target = str(body.get("entity_id", ""))
            if target in self.states:
                self.states[target] = {
                    "turn_on": "on",
                    "turn_off": "off",
                    "toggle": "on" if self.states[target] == "off" else "off",
                }.get(service, self.states[target])
            return httpx.Response(200, json=[entity(target, self.states.get(target, "unknown"))])
        return httpx.Response(404, json={"message": "nie ma"})


def make_client(server: FakeHomeAssistant, **overrides: Any) -> HomeAssistantClient:
    settings = make_settings(**overrides)
    return HomeAssistantClient(HomeAssistantConfig.from_settings(settings), client=server.client())


def tools_for(server: FakeHomeAssistant, **overrides: Any) -> dict[str, Any]:
    settings = make_settings(**overrides)
    built = build_home_assistant_tools(make_client(server, **overrides), settings=settings)
    return {tool.spec.name: tool for tool in built}


def tool_context() -> ToolContext:
    return ToolContext(settings=make_settings(), now=frozen_clock())


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --------------------------------------------------------------------------- #
# Adres i konfiguracja
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("wpisane", "oczekiwane"),
    [
        ("http://dom.local:8123", "http://dom.local:8123"),
        ("http://dom.local:8123/", "http://dom.local:8123"),
        ("http://dom.local:8123/api", "http://dom.local:8123"),
        ("http://dom.local:8123/api/", "http://dom.local:8123"),
        # Bez schematu zakładamy http — instalacja domowa rzadko ma certyfikat.
        ("dom.local:8123", "http://dom.local:8123"),
        ("192.168.1.10:8123", "http://192.168.1.10:8123"),
    ],
)
def test_adres_jest_sprowadzany_do_jednej_postaci(wpisane: str, oczekiwane: str) -> None:
    assert normalize_base_url(wpisane) == oczekiwane


def test_pusty_adres_znaczy_brak_konfiguracji() -> None:
    config = HomeAssistantConfig.from_settings(Settings(_env_file=None))

    assert not config.configured
    assert "HOME_ASSISTANT_URL" in config.describe()


def test_opis_konfiguracji_nigdy_nie_pokazuje_tokenu() -> None:
    config = HomeAssistantConfig.from_settings(make_settings())

    opis = config.describe()

    assert TOKEN not in opis
    assert "token ustawiony" in opis


def test_token_nie_wycieka_przez_ustawienia() -> None:
    """``SecretStr`` — token nie pokaże się w logu ani w raporcie zależności."""
    settings = make_settings()

    assert TOKEN not in repr(settings)
    assert TOKEN not in str(settings.home_assistant_token)
    assert settings.home_assistant_token.get_secret_value() == TOKEN


def test_redakcja_usuwa_token_z_tekstu() -> None:
    assert TOKEN not in redact(f"Authorization: Bearer {TOKEN}")


# --------------------------------------------------------------------------- #
# Odczyt
# --------------------------------------------------------------------------- #


def test_odczyt_stanu_encji() -> None:
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.state"]

    result = run(tool.run(EntityArgs(entity_id="light.salon"), tool_context()))

    assert result.ok and result.data["state"] == "off"
    assert "light.salon" in result.display


def test_token_jedzie_w_naglowku_zadania() -> None:
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.state"]

    run(tool.run(EntityArgs(entity_id="light.salon"), tool_context()))

    assert server.requests[-1].headers["Authorization"] == f"Bearer {TOKEN}"


def test_lista_encji_da_sie_zawezic_do_domeny() -> None:
    server = FakeHomeAssistant(
        states={"light.salon": "off", "light.kuchnia": "on", "lock.wejscie": "locked"}
    )
    tool = tools_for(server)["ha.list"]

    result = run(tool.run(ListArgs(domain="light"), tool_context()))

    identyfikatory = [item["entity_id"] for item in result.data["entities"]]
    assert identyfikatory == ["light.kuchnia", "light.salon"]


def test_lista_ma_limit_zeby_nie_zapchac_kontekstu() -> None:
    server = FakeHomeAssistant(states={f"light.l{i}": "off" for i in range(50)})
    tool = tools_for(server, home_assistant_max_entities=5)["ha.list"]

    result = run(tool.run(ListArgs(limit=5), tool_context()))

    assert len(result.data["entities"]) == 5


def test_wynik_z_home_assistanta_jest_oznaczony_jako_niezaufany() -> None:
    """Nazwy encji nadaje człowiek — to treść z zewnątrz, jak strona WWW."""
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.state"]

    result = run(tool.run(EntityArgs(entity_id="light.salon"), tool_context()))

    assert result.untrusted


# --------------------------------------------------------------------------- #
# Przełączanie i poziomy ryzyka
# --------------------------------------------------------------------------- #


def test_wlaczenie_swiatla_zmienia_stan() -> None:
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.switch"]

    result = run(tool.run(SwitchArgs(entity_id="light.salon", action="on"), tool_context()))

    assert result.ok and result.data["state"] == "on"
    assert server.states["light.salon"] == "on"


@pytest.mark.parametrize(
    ("entity_id", "poziom"),
    [
        ("light.salon", RiskLevel.MEDIUM),
        ("switch.gniazdko", RiskLevel.MEDIUM),
        ("fan.sypialnia", RiskLevel.MEDIUM),
        ("lock.wejscie", RiskLevel.HIGH),
        ("cover.roleta", RiskLevel.HIGH),
        ("gate.brama", RiskLevel.HIGH),
        ("alarm_control_panel.dom", RiskLevel.HIGH),
    ],
)
def test_ryzyko_zalezy_od_domeny_encji(entity_id: str, poziom: RiskLevel) -> None:
    """W API to jedno wywołanie; dla człowieka różnica między lampką a zamkiem."""
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.switch"]

    assert tool.effective_risk(SwitchArgs(entity_id=entity_id, action="on")) is poziom


def test_wlasne_domeny_wysokiego_ryzyka_z_konfiguracji() -> None:
    domeny = high_risk_domains(make_settings(home_assistant_high_risk_domains="humidifier"))

    assert "humidifier" in domeny
    # Wbudowanych nie da się usunąć literówką w .env — zamek zostaje zamkiem.
    assert "lock" in domeny


def test_zamek_wymaga_zgody_uzytkownika() -> None:
    """Pełna droga przez router z Fazy 7 — nie przez własną logikę pluginu."""
    server = FakeHomeAssistant()
    settings = make_settings()
    broker = SpyBroker(approve=False, reason="użytkownik odmówił")
    router = ToolRouter(
        ToolRegistry(build_home_assistant_tools(make_client(server), settings=settings)),
        settings=settings,
        policy=SecurityPolicy(settings),
        broker=broker,
    )

    outcome = run(
        router.dispatch(
            ToolCall(name="ha.switch", arguments={"entity_id": "lock.wejscie", "action": "off"}),
            tool_context(),
        )
    )

    assert not outcome.ok
    assert len(broker.requests) == 1 and broker.requests[0].risk is RiskLevel.HIGH
    # Odmowa znaczy, że do serwera nie poszło ŻADNE żądanie zmiany stanu.
    assert server.states["lock.wejscie"] == "locked"
    assert not any(r.url.path.startswith("/api/services/") for r in server.requests)


def test_swiatlo_nie_pyta_o_zgode_przy_domyslnej_polityce() -> None:
    server = FakeHomeAssistant()
    settings = make_settings()
    broker = SpyBroker(approve=True)
    router = ToolRouter(
        ToolRegistry(build_home_assistant_tools(make_client(server), settings=settings)),
        settings=settings,
        policy=SecurityPolicy(settings),
        broker=broker,
    )

    outcome = run(
        router.dispatch(
            ToolCall(name="ha.switch", arguments={"entity_id": "light.salon", "action": "on"}),
            tool_context(),
        )
    )

    assert outcome.ok and broker.requests == []


def test_pytanie_o_zgode_mowi_co_sie_stanie() -> None:
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.switch"]

    request = tool.confirmation(SwitchArgs(entity_id="lock.wejscie", action="off"), language="pl")

    assert request is not None
    assert "lock.wejscie" in request.summary
    assert TOKEN not in str(request)


@pytest.mark.parametrize("action", ["restart", "delete", "explode"])
def test_nieznana_akcja_jest_odrzucana(action: str) -> None:
    """Wąska lista akcji: model nie wywoła dowolnej usługi Home Assistanta."""
    with pytest.raises(ValueError, match="dozwolone akcje"):
        SwitchArgs(entity_id="light.salon", action=action)


def test_identyfikator_bez_kropki_jest_odrzucany() -> None:
    with pytest.raises(ValueError, match=r"domain\.name"):
        SwitchArgs(entity_id="salon", action="on")


# --------------------------------------------------------------------------- #
# Gdy Home Assistanta nie ma
# --------------------------------------------------------------------------- #


def test_niedostepny_serwer_to_czytelny_blad_a_nie_awaria() -> None:
    server = FakeHomeAssistant()
    server.connection_error = True
    tool = tools_for(server)["ha.state"]

    with pytest.raises(ToolError) as error:
        run(tool.run(EntityArgs(entity_id="light.salon"), tool_context()))

    assert "connect" in str(error.value)
    assert TOKEN not in str(error.value)


def test_odrzucony_token_mowi_co_zrobic_ale_nie_pokazuje_tokenu() -> None:
    server = FakeHomeAssistant()
    server.status_override = 401
    tool = tools_for(server)["ha.state"]

    with pytest.raises(ToolError) as error:
        run(tool.run(EntityArgs(entity_id="light.salon"), tool_context()))

    message = str(error.value)
    assert "token" in message.lower() and TOKEN not in message


def test_nieznana_encja_konczy_sie_podpowiedzia() -> None:
    server = FakeHomeAssistant()
    tool = tools_for(server)["ha.state"]

    with pytest.raises(ToolError, match=r"ha\.list"):
        run(tool.run(EntityArgs(entity_id="light.nie_ma"), tool_context()))


def test_bez_konfiguracji_narzedzia_sa_niewidoczne_dla_modelu() -> None:
    puste = Settings(_env_file=None)
    client = HomeAssistantClient(HomeAssistantConfig.from_settings(puste))

    for tool in build_home_assistant_tools(client, settings=puste):
        usable, reason = tool.available()
        assert not usable
        assert "HOME_ASSISTANT_URL" in reason


def test_wywolanie_bez_konfiguracji_nie_probuje_sieci() -> None:
    puste = Settings(_env_file=None)
    client = HomeAssistantClient(HomeAssistantConfig.from_settings(puste))

    with pytest.raises(HomeAssistantError, match="nie jest skonfigurowany"):
        run(client.state("light.salon"))


# --------------------------------------------------------------------------- #
# Plugin jako całość
# --------------------------------------------------------------------------- #


def test_plugin_bez_tokenu_mowi_czego_brakuje() -> None:
    plugin = HomeAssistantPlugin()
    ctx = PluginContext(settings=Settings(_env_file=None, home_assistant_url=BASE))

    usable, reason = plugin.available(ctx)

    assert not usable and "HOME_ASSISTANT_TOKEN" in reason


def test_plugin_z_konfiguracja_daje_trzy_narzedzia() -> None:
    plugin = HomeAssistantPlugin()
    ctx = PluginContext(settings=make_settings())

    names = {tool.spec.name for tool in plugin.tools(ctx)}

    assert names == {"ha.state", "ha.list", "ha.switch"}
    assert plugin.available(ctx)[0]


def test_plugin_uzywa_klienta_z_kontekstu() -> None:
    """Dzięki temu testy (i inne pluginy) mogą podstawić własny transport."""
    server = FakeHomeAssistant()
    plugin = HomeAssistantPlugin()
    ctx = PluginContext(
        settings=make_settings(), extras={"home_assistant_client": make_client(server)}
    )

    tools = {tool.spec.name: tool for tool in plugin.tools(ctx)}
    result = run(tools["ha.state"].run(EntityArgs(entity_id="light.salon"), tool_context()))

    assert result.ok and server.requests
