"""Poziomy ryzyka narzędzi (Faza 7).

Poziom jest **atrybutem narzędzia**, deklarowanym w kodzie — nigdy nie przychodzi
od modelu językowego. Model wybiera *co* wywołać, a nie *jak groźne* to jest.

Cztery poziomy, w kolejności rosnącej:

======== ======================================================== =============
poziom   znaczenie                                                potwierdzenie
======== ======================================================== =============
SAFE     tylko odczyt, bez efektów ubocznych                      nie
MEDIUM   ruch sieciowy albo zapis we własnych danych asystenta     nie
HIGH     zapis poza własnymi danymi, uruchamianie programów        ZAWSZE
CRITICAL nieodwracalne albo o zasięgu systemowym                   ZAWSZE + zgoda
======== ======================================================== =============

**Uwaga na porównania.** ``RiskLevel`` jest podklasą ``str``, więc operatory
``<`` i ``>`` porównywałyby *alfabetycznie* („CRITICAL" < „HIGH" < „MEDIUM" <
„SAFE" — czyli dokładnie odwrotnie niż trzeba). Dlatego kolejność jest wyrażona
wyłącznie przez :func:`risk_rank`, :func:`at_least` i :func:`escalate`, a w kodzie
nie wolno porównywać poziomów operatorami.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class RiskLevel(StrEnum):
    """Poziom ryzyka narzędzia."""

    SAFE = "SAFE"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Kolejność rosnąca — jedyne źródło prawdy o tym, co jest „wyżej".
RISK_ORDER: Final[tuple[RiskLevel, ...]] = (
    RiskLevel.SAFE,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.CRITICAL,
)

_RANKS: Final[dict[str, int]] = {level.value: index for index, level in enumerate(RISK_ORDER)}

# Poziom, od którego potwierdzenie jest OBOWIĄZKOWE. Konfiguracja może obniżyć
# próg (żądać potwierdzeń już od MEDIUM), ale nigdy go nie podniesie powyżej tej
# wartości — ustawienie ``SECURITY_REQUIRE_CONFIRM_FROM=CRITICAL`` nie wyłącza
# potwierdzeń dla HIGH.
MANDATORY_CONFIRM_FROM: Final[RiskLevel] = RiskLevel.HIGH

RISK_LABELS_PL: Final[dict[RiskLevel, str]] = {
    RiskLevel.SAFE: "bezpieczne (tylko odczyt)",
    RiskLevel.MEDIUM: "umiarkowane (sieć albo własne dane)",
    RiskLevel.HIGH: "wysokie (wymaga potwierdzenia)",
    RiskLevel.CRITICAL: "krytyczne (nieodwracalne)",
}

RISK_LABELS_EN: Final[dict[RiskLevel, str]] = {
    RiskLevel.SAFE: "safe (read-only)",
    RiskLevel.MEDIUM: "medium (network or own data)",
    RiskLevel.HIGH: "high (needs confirmation)",
    RiskLevel.CRITICAL: "critical (irreversible)",
}


def risk_rank(level: RiskLevel | str) -> int:
    """Pozycja poziomu w kolejności rosnącej (SAFE = 0, CRITICAL = 3).

    Nieznana wartość dostaje rangę CRITICAL — nierozpoznany poziom traktujemy
    jak najgroźniejszy, nigdy jak najłagodniejszy.
    """
    key = str(level).strip().upper()
    return _RANKS.get(key, _RANKS[RiskLevel.CRITICAL.value])


def parse_risk(value: object, *, default: RiskLevel = RiskLevel.CRITICAL) -> RiskLevel:
    """Zamień dowolny zapis („high", „HIGH", ``RiskLevel.HIGH``) na poziom.

    Domyślnie CRITICAL: literówka w konfiguracji nie może obniżyć rygoru.
    """
    if isinstance(value, RiskLevel):
        return value
    key = str(value or "").strip().upper()
    try:
        return RiskLevel(key)
    except ValueError:
        return default


def at_least(level: RiskLevel | str, threshold: RiskLevel | str) -> bool:
    """Czy ``level`` jest co najmniej tak wysoki jak ``threshold``?"""
    return risk_rank(level) >= risk_rank(threshold)


def escalate(level: RiskLevel, other: RiskLevel) -> RiskLevel:
    """Wyższy z dwóch poziomów.

    Eskalacja jest jednokierunkowa: narzędzie może na podstawie argumentów
    podnieść swój poziom (zapis poza katalogiem = HIGH), ale nigdy go obniżyć.
    """
    return other if risk_rank(other) > risk_rank(level) else level


def describe_risk(level: RiskLevel, *, language: str = "en") -> str:
    labels = RISK_LABELS_PL if language == "pl" else RISK_LABELS_EN
    return labels.get(level, str(level))


__all__ = [
    "MANDATORY_CONFIRM_FROM",
    "RISK_LABELS_EN",
    "RISK_LABELS_PL",
    "RISK_ORDER",
    "RiskLevel",
    "at_least",
    "describe_risk",
    "escalate",
    "parse_risk",
    "risk_rank",
]
