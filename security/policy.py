"""Polityka bezpieczeństwa: co wolno wywołać, co wymaga zgody (Faza 7).

Polityka jest **po stronie Pythona**, poza zasięgiem modelu. Model dostaje jej
skutki (odmowę, pytanie o zgodę) jako zwykły komunikat i może spróbować inaczej,
ale nie ma sposobu, by ją obejść — nie ma dostępu do tych obiektów.

Reguły, których żadne ustawienie nie zmienia:

* HIGH i CRITICAL **zawsze** wymagają zgody człowieka. ``SECURITY_REQUIRE_CONFIRM_FROM``
  może obniżyć próg (np. pytać już od MEDIUM), ale nie podnieść go powyżej HIGH.
* CRITICAL bez ``SECURITY_ALLOW_CRITICAL=true`` jest odrzucane i **nie jest nawet
  pokazywane modelowi** — nie może wywołać czegoś, o czym nie wie.
* Liczba wywołań w jednej turze jest ograniczona (``TOOLS_MAX_CALLS_PER_TURN``) —
  to zabezpieczenie przed pętlą narzędzie→wynik→narzędzie.
* Wywołanie o ryzyku ≥ MEDIUM następujące PO wyniku narzędzia z danymi
  niezaufanymi (sieć, plik, treść cudzego autorstwa) wymaga zgody nawet wtedy,
  gdy normalnie by jej nie wymagało. To twarda bariera przeciw prompt injection:
  strona WWW nie zdoła namówić modelu na akcję bez pytania człowieka.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from config import Settings, get_settings
from i18n import t
from security.risk import (
    MANDATORY_CONFIRM_FROM,
    RiskLevel,
    at_least,
    parse_risk,
    risk_rank,
)

logger = logging.getLogger(__name__)

# Od tego poziomu wynik narzędzia wymusza potwierdzenie następnego wywołania,
# gdy poprzedni wynik pochodził z niezaufanego źródła.
_UNTRUSTED_BARRIER_FROM: Final[RiskLevel] = RiskLevel.MEDIUM


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Rozstrzygnięcie polityki dla jednego wywołania."""

    allowed: bool
    risk: RiskLevel
    needs_confirmation: bool = False
    reason: str = ""
    warning: str = ""

    @property
    def denied(self) -> bool:
        return not self.allowed


def _split_list(raw: str) -> tuple[str, ...]:
    """Rozbij listę z ``.env`` („a, b ,c") na nazwy. Puste wpisy wypadają."""
    parts = str(raw or "").replace(";", ",").split(",")
    return tuple(item.strip() for item in parts if item.strip())


class SecurityPolicy:
    """Bramki 2 (ENABLED), 5 (POLICY) i 6 (CONFIRM) routera narzędzi."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._allowed = _split_list(self._settings.tools_allowed)
        self._disabled = _split_list(self._settings.tools_disabled)
        # Próg potwierdzeń: konfiguracja może go OBNIŻYĆ, nigdy podnieść powyżej
        # HIGH. Nierozpoznana wartość schodzi do HIGH, nie do CRITICAL.
        requested = parse_risk(self._settings.security_require_confirm_from, default=RiskLevel.HIGH)
        self._confirm_from = (
            requested
            if risk_rank(requested) < risk_rank(MANDATORY_CONFIRM_FROM)
            else MANDATORY_CONFIRM_FROM
        )

    # --- właściwości ----------------------------------------------------- #

    @property
    def tools_enabled(self) -> bool:
        return bool(self._settings.tools_enabled)

    @property
    def confirm_from(self) -> RiskLevel:
        return self._confirm_from

    @property
    def allow_critical(self) -> bool:
        return bool(self._settings.security_allow_critical)

    @property
    def max_calls_per_turn(self) -> int:
        return int(self._settings.tools_max_calls_per_turn)

    @property
    def confirm_timeout_s(self) -> float:
        return float(self._settings.security_confirm_timeout_s)

    @property
    def dry_run(self) -> bool:
        return bool(self._settings.security_dry_run)

    @property
    def audit_enabled(self) -> bool:
        return bool(self._settings.security_audit_enabled)

    # --- bramka ENABLED --------------------------------------------------- #

    def is_enabled(self, name: str) -> tuple[bool, str]:
        """Czy narzędzie o tej nazwie wolno wywołać? Zwraca też powód odmowy."""
        if not self.tools_enabled:
            return False, "narzędzia są wyłączone (TOOLS_ENABLED=false)"
        if name in self._disabled:
            return False, f"narzędzie '{name}' jest wyłączone w konfiguracji (TOOLS_DISABLED)"
        if self._allowed and "*" not in self._allowed and name not in self._allowed:
            return False, f"narzędzie '{name}' nie jest na liście TOOLS_ALLOWED"
        return True, ""

    def is_visible_to_llm(self, name: str, risk: RiskLevel) -> bool:
        """Czy pokazywać to narzędzie modelowi?

        CRITICAL bez jawnej zgody w konfiguracji jest ukryte: model nie wywoła
        czegoś, czego nie widzi, a odmowa po fakcie i tak byłaby pewna.
        """
        enabled, _ = self.is_enabled(name)
        if not enabled:
            return False
        return self.allow_critical or risk is not RiskLevel.CRITICAL

    # --- bramka POLICY + CONFIRM ------------------------------------------ #

    def evaluate(
        self,
        *,
        tool: str,
        risk: RiskLevel,
        calls_this_turn: int = 0,
        after_untrusted: bool = False,
    ) -> PolicyDecision:
        """Rozstrzygnij: wolno, wolno po zgodzie, czy nie wolno wcale."""
        enabled, reason = self.is_enabled(tool)
        if not enabled:
            return PolicyDecision(allowed=False, risk=risk, reason=reason)

        if risk is RiskLevel.CRITICAL and not self.allow_critical:
            return PolicyDecision(
                allowed=False,
                risk=risk,
                reason=t("policy.critical_disabled"),
            )

        if calls_this_turn >= self.max_calls_per_turn:
            return PolicyDecision(
                allowed=False,
                risk=risk,
                reason=(
                    t("policy.budget_spent", limit=self.max_calls_per_turn)
                ),
            )

        needs_confirmation = at_least(risk, self._confirm_from)
        warning = ""
        if after_untrusted and at_least(risk, _UNTRUSTED_BARRIER_FROM):
            # Barierę stawiamy niezależnie od promptu: instrukcja w treści strony
            # nie ma jak jej wyłączyć, bo decyzja jest tutaj, w Pythonie.
            needs_confirmation = True
            warning = (
                "to wywołanie następuje po wyniku narzędzia z danymi z zewnątrz — "
                "może wynikać z treści, której nie napisał użytkownik"
            )

        return PolicyDecision(
            allowed=True,
            risk=risk,
            needs_confirmation=needs_confirmation,
            reason="",
            warning=warning,
        )

    # --- opis ------------------------------------------------------------- #

    def describe(self) -> str:
        """Jedna linijka do ``/status`` i do raportu zależności."""
        if not self.tools_enabled:
            return t("status.tools_off")
        parts = [t("status.policy.confirm_from", level=self._confirm_from.value)]
        parts.append(
            t("status.policy.critical_allowed")
            if self.allow_critical
            else t("status.policy.critical_blocked")
        )
        parts.append(t("status.policy.limit", count=self.max_calls_per_turn))
        if self.dry_run:
            parts.append(t("status.policy.dry_run"))
        if self._allowed and "*" not in self._allowed:
            parts.append(t("status.policy.allowed", names=", ".join(self._allowed)))
        if self._disabled:
            parts.append(t("status.policy.disabled", names=", ".join(self._disabled)))
        return ", ".join(parts)


__all__ = ["PolicyDecision", "SecurityPolicy"]
