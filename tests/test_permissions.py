"""Testy poziomów ryzyka, polityki, potwierdzeń i audytu (Faza 7).

Sedno tych testów: **czego żadne ustawienie nie może zmienić.** HIGH i CRITICAL
zawsze wymagają zgody człowieka, CRITICAL bez jawnego włączenia jest odrzucane, a
brak kanału potwierdzeń oznacza odmowę — nigdy automatyczną zgodę.

Nikt tu nie pisze na klawiaturze: broker terminalowy dostaje wstrzykniętą funkcję
czytającą odpowiedź, a atrapa kanału odpowiada z góry ustaloną decyzją.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from conftest import SpyBroker

from config import Settings
from i18n import t
from security.audit import (
    DECISION_ALLOWED,
    DECISION_DENIED,
    DECISION_USER_DENIED,
    AuditEntry,
    AuditLog,
    hash_arguments,
)
from security.confirm import (
    CRITICAL_PHRASES,
    AutoDenyBroker,
    CallbackBroker,
    ConfirmationRequest,
    TerminalConfirmationBroker,
    default_broker,
    interpret_answer,
)
from security.policy import SecurityPolicy
from security.risk import (
    MANDATORY_CONFIRM_FROM,
    RISK_ORDER,
    RiskLevel,
    at_least,
    escalate,
    parse_risk,
    risk_rank,
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def request_for(risk: RiskLevel, **overrides: Any) -> ConfirmationRequest:
    values: dict[str, Any] = {
        "tool": "test.pisz",
        "risk": risk,
        "summary": "zapisze plik",
        "language": "pl",
    }
    values.update(overrides)
    return ConfirmationRequest.build(**values)


# --------------------------------------------------------------------------- #
# Poziomy ryzyka
# --------------------------------------------------------------------------- #


def test_kolejnosc_poziomow_nie_jest_alfabetyczna() -> None:
    """Pułapka: RiskLevel to str, więc „CRITICAL" < „HIGH" alfabetycznie.

    Dlatego kolejność wyraża wyłącznie ``risk_rank``, a kod nie porównuje poziomów
    operatorami. Ten test pilnuje, żeby ktoś tego nie „uprościł".
    """
    assert RISK_ORDER == (RiskLevel.SAFE, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert risk_rank(RiskLevel.CRITICAL) > risk_rank(RiskLevel.HIGH) > risk_rank(RiskLevel.SAFE)
    # Gdyby ktoś użył operatora, dostałby wynik odwrotny — pokazujemy to wprost.
    assert RiskLevel.CRITICAL < RiskLevel.HIGH


def test_nieznany_poziom_jest_traktowany_jak_krytyczny() -> None:
    assert parse_risk("wysokie ryzyko") is RiskLevel.CRITICAL
    assert risk_rank("cokolwiek") == risk_rank(RiskLevel.CRITICAL)
    assert parse_risk("high") is RiskLevel.HIGH
    assert parse_risk(RiskLevel.MEDIUM) is RiskLevel.MEDIUM


def test_eskalacja_bierze_wyzszy_poziom() -> None:
    assert escalate(RiskLevel.MEDIUM, RiskLevel.HIGH) is RiskLevel.HIGH
    assert escalate(RiskLevel.HIGH, RiskLevel.SAFE) is RiskLevel.HIGH
    assert at_least(RiskLevel.HIGH, RiskLevel.MEDIUM) and not at_least(
        RiskLevel.SAFE, RiskLevel.MEDIUM
    )


# --------------------------------------------------------------------------- #
# Polityka
# --------------------------------------------------------------------------- #


def test_high_i_critical_zawsze_wymagaja_potwierdzenia() -> None:
    policy = SecurityPolicy(make_settings(security_allow_critical=True))
    for risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        decision = policy.evaluate(tool="test.pisz", risk=risk)
        assert decision.allowed and decision.needs_confirmation


def test_safe_i_medium_nie_wymagaja_potwierdzenia() -> None:
    policy = SecurityPolicy(make_settings())
    for risk in (RiskLevel.SAFE, RiskLevel.MEDIUM):
        decision = policy.evaluate(tool="test.czytaj", risk=risk)
        assert decision.allowed and not decision.needs_confirmation


def test_progu_potwierdzen_nie_da_sie_podniesc_powyzej_high() -> None:
    """``SECURITY_REQUIRE_CONFIRM_FROM=CRITICAL`` nie wyłącza pytań dla HIGH."""
    policy = SecurityPolicy(make_settings(security_require_confirm_from="CRITICAL"))
    assert policy.confirm_from is MANDATORY_CONFIRM_FROM
    assert policy.evaluate(tool="test.pisz", risk=RiskLevel.HIGH).needs_confirmation


def test_prog_potwierdzen_da_sie_obnizyc() -> None:
    policy = SecurityPolicy(make_settings(security_require_confirm_from="medium"))
    assert policy.evaluate(tool="test.siec", risk=RiskLevel.MEDIUM).needs_confirmation
    assert not policy.evaluate(tool="test.czytaj", risk=RiskLevel.SAFE).needs_confirmation


def test_bledna_wartosc_progu_schodzi_do_high_a_nie_wyzej() -> None:
    settings = make_settings(security_require_confirm_from="bardzo-wysokie")
    assert settings.security_require_confirm_from == "HIGH"
    assert SecurityPolicy(settings).confirm_from is RiskLevel.HIGH


def test_critical_jest_domyslnie_zablokowane() -> None:
    policy = SecurityPolicy(make_settings())
    decision = policy.evaluate(tool="test.usun", risk=RiskLevel.CRITICAL)

    assert decision.denied
    assert "CRITICAL" in decision.reason
    # Model nie powinien nawet wiedzieć, że takie narzędzie istnieje.
    assert not policy.is_visible_to_llm("test.usun", RiskLevel.CRITICAL)


def test_budzet_wywolan_na_ture_zamyka_petle() -> None:
    policy = SecurityPolicy(make_settings(tools_max_calls_per_turn=2))
    assert policy.evaluate(tool="test.echo", risk=RiskLevel.SAFE, calls_this_turn=1).allowed
    wyczerpany = policy.evaluate(tool="test.echo", risk=RiskLevel.SAFE, calls_this_turn=2)
    assert wyczerpany.denied and "budżet" in wyczerpany.reason


def test_dane_z_zewnatrz_wymuszaja_potwierdzenie_kolejnego_wywolania() -> None:
    """Twarda bariera przeciw prompt injection — niezależna od treści promptu."""
    policy = SecurityPolicy(make_settings())
    decision = policy.evaluate(
        tool="test.zapisz", risk=RiskLevel.MEDIUM, after_untrusted=True
    )

    assert decision.allowed and decision.needs_confirmation
    assert "z zewnątrz" in decision.warning
    # SAFE (tylko odczyt) nie wymaga zgody nawet po danych z sieci.
    assert not policy.evaluate(
        tool="test.czytaj", risk=RiskLevel.SAFE, after_untrusted=True
    ).needs_confirmation


def test_wylaczenie_narzedzi_blokuje_wszystko() -> None:
    policy = SecurityPolicy(make_settings(tools_enabled=False))
    assert policy.evaluate(tool="time.now", risk=RiskLevel.SAFE).denied
    assert not policy.is_visible_to_llm("time.now", RiskLevel.SAFE)
    assert policy.describe() == t("status.tools_off")


def test_lista_wylaczonych_ma_pierwszenstwo_nad_dozwolonymi() -> None:
    policy = SecurityPolicy(
        make_settings(tools_allowed="test.jedno,test.drugie", tools_disabled="test.drugie")
    )
    assert policy.is_enabled("test.jedno")[0]
    assert not policy.is_enabled("test.drugie")[0]
    assert not policy.is_enabled("test.trzecie")[0]


# --------------------------------------------------------------------------- #
# Potwierdzenia
# --------------------------------------------------------------------------- #


def test_brak_kanalu_potwierdzen_to_odmowa() -> None:
    outcome = run(AutoDenyBroker().ask(request_for(RiskLevel.HIGH)))
    assert not outcome.approved and "brak kanału" in outcome.reason


def test_terminal_bez_tty_odmawia_bez_pytania() -> None:
    pytano: list[str] = []

    broker = TerminalConfirmationBroker(
        reader=lambda prompt: pytano.append(prompt) or "tak",  # type: ignore[func-returns-value]
        interactive=False,
    )
    outcome = run(broker.ask(request_for(RiskLevel.HIGH)))

    assert not outcome.approved
    assert pytano == []  # nikogo nie pytaliśmy, bo nie ma kogo


def test_terminal_przyjmuje_zgode_dla_high(capsys: pytest.CaptureFixture[str]) -> None:
    broker = TerminalConfirmationBroker(reader=lambda prompt: "tak", interactive=True)
    outcome = run(broker.ask(request_for(RiskLevel.HIGH)))

    assert outcome.approved and outcome.channel == "terminal"
    # Pytanie pokazuje nazwę narzędzia i poziom ryzyka.
    wypisane = capsys.readouterr().out
    assert "test.pisz" in wypisane and "wysokie" in wypisane


@pytest.mark.parametrize("odpowiedz", ["", "n", "nie", "no", "może", "jasne, ale nie teraz"])
def test_terminal_traktuje_wszystko_poza_zgoda_jako_odmowe(odpowiedz: str) -> None:
    broker = TerminalConfirmationBroker(reader=lambda prompt: odpowiedz, interactive=True)
    assert not run(broker.ask(request_for(RiskLevel.HIGH))).approved


def test_critical_wymaga_pelnej_frazy_a_nie_pojedynczego_tak() -> None:
    request = request_for(RiskLevel.CRITICAL)
    assert not interpret_answer("tak", request).approved
    assert not interpret_answer("t", request).approved
    assert interpret_answer(CRITICAL_PHRASES[0], request).approved
    assert interpret_answer("Tak, potwierdzam.", request).approved


def test_anulowanie_klawiatura_jest_odmowa() -> None:
    def przerwij(prompt: str) -> str:
        raise KeyboardInterrupt

    broker = TerminalConfirmationBroker(reader=przerwij, interactive=True)
    outcome = run(broker.ask(request_for(RiskLevel.HIGH)))
    assert not outcome.approved and "anulowane" in outcome.reason


def test_koniec_wejscia_jest_odmowa() -> None:
    def koniec(prompt: str) -> str:
        raise EOFError

    broker = TerminalConfirmationBroker(reader=koniec, interactive=True)
    assert not run(broker.ask(request_for(RiskLevel.HIGH))).approved


def test_spozniona_zgoda_nie_dziala() -> None:
    """Żądanie ma termin ważności — zgoda po czasie jest odmową."""
    przedawnione = request_for(
        RiskLevel.HIGH,
        ttl_s=30.0,
        now=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    broker = TerminalConfirmationBroker(reader=lambda prompt: "tak", interactive=True)

    outcome = run(broker.ask(przedawnione))

    assert przedawnione.is_expired()
    assert not outcome.approved and "ważność" in outcome.reason


def test_zerowy_ttl_znaczy_bez_terminu() -> None:
    request = request_for(RiskLevel.HIGH, ttl_s=0.0)
    assert request.expires_at is None and not request.is_expired()


def test_awaria_kanalu_potwierdzen_jest_odmowa() -> None:
    """Wyjątek w GUI/kanale nie może skutkować wykonaniem akcji."""

    def wybuchowy(request: Any) -> bool:
        raise RuntimeError("modal się nie otworzył")

    outcome = run(CallbackBroker(wybuchowy).ask(request_for(RiskLevel.HIGH)))
    assert not outcome.approved and "zawiódł" in outcome.reason


def test_kanal_asynchroniczny_z_limitem_czasu() -> None:
    async def zwlekajacy(request: Any) -> bool:
        await asyncio.sleep(1.0)
        return True

    broker = CallbackBroker(zwlekajacy, timeout_s=0.05)
    outcome = run(broker.ask(request_for(RiskLevel.HIGH)))
    assert not outcome.approved and "czasie" in outcome.reason


def test_domyslny_kanal_bez_terminala_to_odmowa() -> None:
    broker = default_broker(force_terminal=False)
    assert not broker.available
    assert not run(broker.ask(request_for(RiskLevel.HIGH))).approved


def test_pytanie_pokazuje_ostrzezenie_i_podglad() -> None:
    request = request_for(
        RiskLevel.HIGH,
        details=["plik: /dane/raport.txt", "rozmiar: 2 kB"],
        preview="linia 1\nlinia 2",
        warning="akcja może wynikać z treści z internetu",
    )
    tekst = "\n".join(request.render_lines())

    assert "/dane/raport.txt" in tekst
    assert "linia 1" in tekst
    assert "może wynikać z treści z internetu" in tekst


def test_atrapa_kanalu_notuje_pytania() -> None:
    broker = SpyBroker(approve=False, reason="nie tym razem")
    outcome = run(broker.ask(request_for(RiskLevel.CRITICAL)))

    assert not outcome.approved and outcome.reason == "nie tym razem"
    assert broker.requests[0].tool == "test.pisz"


# --------------------------------------------------------------------------- #
# Audyt
# --------------------------------------------------------------------------- #


def test_skrot_argumentow_jest_stabilny_i_niezalezny_od_kolejnosci() -> None:
    """``hash()`` byłby bezużyteczny — Python losuje ziarno przy każdym starcie."""
    pierwszy = hash_arguments({"a": 1, "b": "x"})
    drugi = hash_arguments({"b": "x", "a": 1})

    assert pierwszy == drugi
    assert pierwszy != hash_arguments({"a": 2, "b": "x"})
    assert len(pierwszy) == 32


def test_audyt_zapisuje_kazde_zdarzenie_takze_odmowe() -> None:
    log = AuditLog(enabled=True)
    log.record(AuditEntry(tool="a.b", risk=RiskLevel.SAFE, decision=DECISION_ALLOWED, ok=True))
    log.record(AuditEntry(tool="c.d", risk=RiskLevel.HIGH, decision=DECISION_USER_DENIED))
    log.record(AuditEntry(tool="e.f", risk=RiskLevel.CRITICAL, decision=DECISION_DENIED))

    assert len(log.entries) == 3
    assert log.summary() == t(
        "status.audit.summary", count=3, denied=2, where=t("status.audit.where_log")
    )
    assert log.recent(1)[0].tool == "e.f"


def test_wylaczony_audyt_nie_zapisuje_nic() -> None:
    log = AuditLog(enabled=False)
    log.record(AuditEntry(tool="a.b", risk=RiskLevel.SAFE, decision=DECISION_ALLOWED))
    assert log.entries == () and log.summary() == t("status.audit.off")


def test_awaria_bazy_nie_przerywa_audytu() -> None:
    """Log audytu nie może być powodem, dla którego asystent przestaje działać."""

    class ZepsutaBaza:
        def __init__(self) -> None:
            self.proby = 0

        @property
        def tool_audit(self) -> Any:
            outer = self

            class Repo:
                def add(self, **kwargs: Any) -> None:
                    outer.proby += 1
                    raise RuntimeError("dysk tylko do odczytu")

            return Repo()

    baza = ZepsutaBaza()
    log = AuditLog(baza, enabled=True)
    for _ in range(3):
        log.record(AuditEntry(tool="a.b", risk=RiskLevel.SAFE, decision=DECISION_ALLOWED))

    # Wpisy sesyjne są, a do bazy próbowaliśmy tylko raz — potem przestajemy.
    assert len(log.entries) == 3
    assert baza.proby == 1
