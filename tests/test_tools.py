"""Testy kontraktu narzędzi, rejestru i przykładowego narzędzia SAFE (Faza 7).

Żaden test nie sięga do systemu: jedyne zarejestrowane narzędzie czyta czas z
zegara **wstrzykniętego** przez kontekst, a nie z zegara maszyny. Dzięki temu
sprawdzamy formatowanie daty na ustalonej chwili i wynik jest ten sam wszędzie.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any

import pytest
from conftest import FROZEN_MOMENT, frozen_clock, make_fake_tool

from config import Settings
from security.policy import SecurityPolicy
from security.risk import RiskLevel
from tools.base import (
    BaseTool,
    ToolArgs,
    ToolContext,
    ToolResult,
    ToolSpec,
    make_tool,
)
from tools.registry import ToolRegistry, build_registry
from tools.system import TimeNowArgs, build_time_tool, format_moment, time_now


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def make_context(**overrides: Any) -> ToolContext:
    values: dict[str, Any] = {
        "settings": make_settings(),
        "language": "pl",
        "now": frozen_clock(),
    }
    values.update(overrides)
    return ToolContext(**values)


def run(coroutine: Any) -> Any:
    import asyncio

    return asyncio.run(coroutine)


# --------------------------------------------------------------------------- #
# Kontrakt: argumenty i wynik
# --------------------------------------------------------------------------- #


def test_nieznany_argument_jest_bledem_a_nie_ignorowany() -> None:
    """Halucynacja parametru nie może przejść po cichu."""

    class Args(ToolArgs):
        text: str = "x"

    with pytest.raises(ValueError):
        Args.model_validate({"text": "ok", "force": True})


def test_argumenty_sa_niezmienne() -> None:
    class Args(ToolArgs):
        text: str = "x"

    args = Args()
    with pytest.raises(ValueError):
        args.text = "inne"  # type: ignore[misc]


def test_brak_poziomu_ryzyka_daje_najwyzszy_a_nie_najnizszy() -> None:
    """Narzędzie bez zadeklarowanego ryzyka ma być blokowane, nie przepuszczane."""
    spec = ToolSpec(name="test.cos", description="bez ryzyka")
    assert spec.risk is RiskLevel.CRITICAL


def test_schemat_dla_modelu_ma_format_tool_callingu() -> None:
    tool = build_time_tool()
    schema = tool.spec.llm_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "time.now"
    parameters = schema["function"]["parameters"]
    assert parameters["type"] == "object"
    # ``extra="forbid"`` musi być widoczne także w schemacie wysyłanym modelowi.
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"zone", "include_date"}


def test_wynik_dla_modelu_jest_zwiezlym_jsonem() -> None:
    ok = ToolResult.success({"b": 2, "a": 1}, display="ładnie")
    # Klucze sortowane: ten sam wynik daje ten sam tekst, więc audyt i testy są stabilne.
    assert ok.to_json() == '{"data": {"a": 1, "b": 2}, "ok": true}'
    assert ToolResult.failure("nie wyszło").to_json() == '{"error": "nie wyszło", "ok": false}'


def test_eskalacja_ryzyka_dziala_tylko_w_gore() -> None:
    """Narzędzie może podnieść swoje ryzyko na podstawie argumentów, nie obniżyć."""

    class Args(ToolArgs):
        wszystko: bool = False

    class Wredne(BaseTool):
        def dynamic_risk(self, args: Any) -> RiskLevel:
            # Próba obniżenia: deklaracja mówi HIGH, a to zwraca SAFE.
            return RiskLevel.SAFE if not args.wszystko else RiskLevel.CRITICAL

    tool = Wredne(
        ToolSpec(name="test.wredne", description="x", args_model=Args, risk=RiskLevel.HIGH)
    )
    assert tool.effective_risk(Args()) is RiskLevel.HIGH
    assert tool.effective_risk(Args(wszystko=True)) is RiskLevel.CRITICAL


def test_funkcja_synchroniczna_tez_jest_narzedziem() -> None:
    """Zwykła funkcja ma działać bez opakowań — pójdzie do wątku roboczego."""

    def zwykla(args: Any, ctx: ToolContext) -> str:
        return "gotowe"

    tool = make_tool(
        name="test.sync",
        description="x",
        args_model=ToolArgs,
        risk=RiskLevel.SAFE,
        function=zwykla,
    )
    result = run(tool.run(ToolArgs(), make_context()))
    assert result.ok and result.data["value"] == "gotowe"


def test_domyslne_pytanie_o_zgode_pokazuje_prawdziwe_argumenty() -> None:
    """Treść pytania buduje narzędzie z argumentów, nie model."""
    tool = make_fake_tool(name="test.pisz", risk=RiskLevel.HIGH)
    args = tool.spec.args_model.model_validate({"text": "usuń wszystko"})

    request = tool.confirmation(args, language="pl")

    assert request is not None
    assert request.tool == "test.pisz" and request.risk is RiskLevel.HIGH
    assert any("usuń wszystko" in detail for detail in request.details)


# --------------------------------------------------------------------------- #
# Rejestr
# --------------------------------------------------------------------------- #


def test_rejestr_odrzuca_zla_nazwe_i_duplikat() -> None:
    registry = ToolRegistry()
    registry.register(make_fake_tool(name="test.echo"))

    with pytest.raises(ValueError):
        registry.register(make_fake_tool(name="test.echo"))
    with pytest.raises(ValueError):
        registry.register(make_fake_tool(name="ZleNazwane"))
    with pytest.raises(ValueError):
        registry.register(make_fake_tool(name="bezkropki"))


def test_rejestr_ukrywa_przed_modelem_krytyczne_i_wylaczone() -> None:
    registry = ToolRegistry(
        [
            make_fake_tool(name="test.bezpieczne", risk=RiskLevel.SAFE),
            make_fake_tool(name="test.krytyczne", risk=RiskLevel.CRITICAL),
            make_fake_tool(name="test.wylaczone", risk=RiskLevel.SAFE),
        ]
    )
    policy = SecurityPolicy(make_settings(tools_disabled="test.wylaczone"))

    widoczne = [tool.spec.name for tool in registry.visible(policy)]
    assert widoczne == ["test.bezpieczne"]
    assert len(registry) == 3  # ukrycie nie znaczy usunięcie


def test_rejestr_pokazuje_krytyczne_gdy_wlaczone_jawnie() -> None:
    registry = ToolRegistry([make_fake_tool(name="test.krytyczne", risk=RiskLevel.CRITICAL)])
    policy = SecurityPolicy(make_settings(security_allow_critical=True))
    assert [tool.spec.name for tool in registry.visible(policy)] == ["test.krytyczne"]


def test_rejestr_pomija_narzedzie_niedostepne_na_tej_maszynie() -> None:
    """Brak zależności = narzędzia nie ma na liście dla modelu, ale jest w opisie."""
    registry = ToolRegistry(
        [make_fake_tool(name="test.brak", available=(False, "brak biblioteki xyz"))]
    )
    assert registry.visible(SecurityPolicy(make_settings())) == []
    assert "brak biblioteki xyz" in "\n".join(registry.describe())


def test_allowlista_zawezajaca_dziala() -> None:
    registry = ToolRegistry(
        [make_fake_tool(name="test.jedno"), make_fake_tool(name="test.drugie")]
    )
    policy = SecurityPolicy(make_settings(tools_allowed="test.jedno"))
    assert [tool.spec.name for tool in registry.visible(policy)] == ["test.jedno"]


def test_wbudowany_rejestr_ma_narzedzia_wszystkich_grup() -> None:
    """Rejestr wbudowany: czas (Faza 7) i grupy Fazy 8, każde z poziomem ryzyka."""
    registry = build_registry(make_settings())
    names = registry.names()

    assert "time.now" in names and registry.get("time.now").spec.risk is RiskLevel.SAFE
    for prefix in ("fs.", "app.", "notes.", "pdf.", "shell.", "process.", "service."):
        assert any(name.startswith(prefix) for name in names), prefix
    # Nazwy są unikalne, a żadne narzędzie nie zostało bez zadeklarowanego ryzyka.
    assert len(names) == len(set(names))
    assert all(tool.spec.risk in tuple(RiskLevel) for tool in registry)


def test_narzedzia_o_ryzyku_high_maja_wlasne_pytanie_o_zgode() -> None:
    """HIGH/CRITICAL bez własnego pytania pokazywałoby użytkownikowi surowe argumenty."""
    registry = build_registry(make_settings())
    for tool in registry:
        if tool.spec.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            assert type(tool).confirmation is not BaseTool.confirmation, tool.spec.name


# --------------------------------------------------------------------------- #
# Narzędzie time.now
# --------------------------------------------------------------------------- #


def test_godzina_jest_liczona_z_wstrzyknietego_zegara() -> None:
    result = time_now(TimeNowArgs(zone="utc"), make_context(language="en"))

    assert result.ok
    assert result.data["date"] == "2026-08-17"
    assert result.data["time"] == "13:42"
    assert result.data["weekday"] == "Monday"
    assert "Monday, 17 August 2026, 13:42" in result.display


def test_nazwy_dni_nie_zaleza_od_ustawien_regionalnych_maszyny() -> None:
    """Nazwa dnia pochodzi z kodu, nie ze ``strftime('%A')`` i nie z locale."""
    polski = time_now(TimeNowArgs(zone="utc"), make_context(language="pl"))
    angielski = time_now(TimeNowArgs(zone="utc"), make_context(language="en"))

    assert "poniedziałek" in polski.display and "17 sierpnia 2026" in polski.display
    assert "Monday" in angielski.display and "17 August 2026" in angielski.display


def test_sama_godzina_bez_daty() -> None:
    result = time_now(TimeNowArgs(zone="utc", include_date=False), make_context())
    assert result.display.startswith("13:42")
    assert "sierpnia" not in result.display


def test_strefa_lokalna_zwraca_przesuniecie_wzgledem_utc() -> None:
    """Nie zakładamy żadnej konkretnej strefy — sprawdzamy spójność wyniku."""
    result = time_now(TimeNowArgs(zone="local"), make_context())
    lokalny = datetime.fromisoformat(result.data["iso"])

    assert lokalny.utcoffset() is not None
    assert lokalny.astimezone(timezone.utc) == FROZEN_MOMENT
    assert result.data["utc_offset_minutes"] == int(
        (lokalny.utcoffset() or timedelta(0)).total_seconds() // 60
    )


def test_nazwa_strefy_bez_tzname_spada_na_przesuniecie() -> None:
    """``tzname()`` bywa puste (zależy od systemu) — wtedy liczymy przesunięcie."""

    class BezNazwy(tzinfo):
        def utcoffset(self, moment: datetime | None) -> timedelta:
            return timedelta(hours=2)

        def tzname(self, moment: datetime | None) -> str | None:
            return None

        def dst(self, moment: datetime | None) -> timedelta:
            return timedelta(0)

    moment = FROZEN_MOMENT.astimezone(BezNazwy())
    tekst = format_moment(moment, language="pl", include_date=False)
    assert tekst == "15:42 (UTC+02:00)"


def test_nieznana_strefa_jest_odrzucana_przez_walidacje() -> None:
    """Model nie poda „Europe/Warsaw" — bo bazy stref nie ma na każdym systemie."""
    with pytest.raises(ValueError):
        TimeNowArgs.model_validate({"zone": "Europe/Warsaw"})
