"""Tool risk levels (Phase 7).

The level is an **attribute of the tool**, declared in the code — it never comes
from the language model. The model chooses *what* to call, not *how dangerous*
it is.

Four levels, in ascending order:

======== ======================================================== =============
level    meaning                                                  confirmation
======== ======================================================== =============
SAFE     read only, no side effects                               no
MEDIUM   network traffic, or a write within the assistant's data  no
HIGH     writing outside its own data, launching programs         ALWAYS
CRITICAL irreversible or system-wide                              ALWAYS + consent
======== ======================================================== =============

**Mind the comparisons.** ``RiskLevel`` is a subclass of ``str``, so the ``<``
and ``>`` operators would compare *alphabetically* ("CRITICAL" < "HIGH" <
"MEDIUM" < "SAFE" — that is, exactly backwards). The ordering is therefore
expressed solely through :func:`risk_rank`, :func:`at_least` and
:func:`escalate`, and comparing levels with operators is forbidden in the code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class RiskLevel(StrEnum):
    """A tool's risk level."""

    SAFE = "SAFE"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Ascending order — the single source of truth about what counts as "higher".
RISK_ORDER: Final[tuple[RiskLevel, ...]] = (
    RiskLevel.SAFE,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.CRITICAL,
)

_RANKS: Final[dict[str, int]] = {level.value: index for index, level in enumerate(RISK_ORDER)}

# The level from which confirmation is MANDATORY. Configuration may lower the
# threshold (demanding confirmations from MEDIUM upwards) but can never raise it
# above this value — setting ``SECURITY_REQUIRE_CONFIRM_FROM=CRITICAL`` does not
# disable confirmations for HIGH.
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
    """The level's position in ascending order (SAFE = 0, CRITICAL = 3).

    An unknown value is given the CRITICAL rank — an unrecognised level is
    treated as the most dangerous one, never as the mildest.
    """
    key = str(level).strip().upper()
    return _RANKS.get(key, _RANKS[RiskLevel.CRITICAL.value])


def parse_risk(value: object, *, default: RiskLevel = RiskLevel.CRITICAL) -> RiskLevel:
    """Turn any spelling ("high", "HIGH", ``RiskLevel.HIGH``) into a level.

    CRITICAL by default: a typo in the configuration must not lower the rigour.
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
    """The higher of two levels.

    Escalation is one-directional: a tool may raise its level based on the
    arguments (a write outside the directory = HIGH) but never lower it.
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
