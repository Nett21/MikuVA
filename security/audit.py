"""Log wywołań narzędzi — co model próbował zrobić i co mu na to pozwolono (Faza 7).

Każde przejście przez router zostawia wpis: także odmowa na pierwszej bramce i
także anulowanie przez użytkownika. Bez tego nie da się odpowiedzieć na pytanie
„dlaczego to się stało?", a przy narzędziach o wysokim ryzyku to pytanie pada
zawsze i zwykle po fakcie.

Dwie decyzje warte uzasadnienia:

* **zapisujemy skrót argumentów (sha256), nie ich treść.** Do zbadania zdarzenia
  wystarczy nazwa narzędzia, poziom ryzyka, decyzja i skrót (widać, czy to samo
  wywołanie powtarzało się w pętli). Prywatne dane użytkownika nie muszą leżeć
  w drugim miejscu, a hash nadal pozwala porównać dwa wywołania;
* **log nigdy nie przerywa wywołania.** Brak bazy, dysk tylko do odczytu, błąd
  SQLite — wszystko to zostaje ostrzeżeniem w logu aplikacji. Log audytu nie może
  być powodem, dla którego asystent przestaje działać, ale też nie może po cichu
  udawać, że zapisał.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from i18n import t
from security.risk import RiskLevel

logger = logging.getLogger(__name__)

# Nazwy decyzji. Zwykłe łańcuchy, bo trafiają do bazy i do logu, a nowa decyzja
# w kolejnej fazie nie może wymagać migracji schematu.
DECISION_ALLOWED: Final[str] = "allowed"
DECISION_CONFIRMED: Final[str] = "confirmed"
DECISION_DENIED: Final[str] = "denied"
DECISION_USER_DENIED: Final[str] = "user_denied"
DECISION_INVALID: Final[str] = "invalid_arguments"
DECISION_UNKNOWN_TOOL: Final[str] = "unknown_tool"
DECISION_DRY_RUN: Final[str] = "dry_run"
DECISION_ERROR: Final[str] = "error"
# Model poprosił drugi raz o dokładnie to samo, o co użytkownik był już pytany
# w tej turze. Osobna decyzja, bo w audycie to co innego niż odmowa człowieka:
# tu nikt nikogo nie pytał — pytanie zostało zdjęte z drogi użytkownika.
DECISION_REPEATED: Final[str] = "repeated_call"


def hash_arguments(arguments: Mapping[str, Any] | None) -> str:
    """Skrót argumentów — stabilny między uruchomieniami i maszynami.

    Kolejność kluczy jest sortowana, a wynik zapisany jako UTF-8: dwa identyczne
    wywołania dają ten sam skrót niezależnie od kolejności pól w JSON-ie od modelu
    i niezależnie od systemu. ``hash()`` byłby tu bezużyteczny — Python losuje
    jego ziarno przy każdym starcie procesu.
    """
    try:
        payload = json.dumps(dict(arguments or {}), ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - argumenty nie do serializacji
        payload = repr(arguments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Pojedyncze zdarzenie do zapisania."""

    tool: str
    risk: RiskLevel
    decision: str
    ok: bool = False
    confirmed: bool = False
    arguments_hash: str = ""
    duration_ms: int = 0
    detail: str = ""
    conversation_id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_line(self) -> str:
        state = t("common.yes") if self.ok else t("common.no")
        line = t(
            "status.audit.line",
            tool=self.tool,
            risk=self.risk.value,
            decision=self.decision,
            confirmed=t("status.audit.confirmed") if self.confirmed else "",
            state=state,
            ms=self.duration_ms,
        )
        return line + (f": {self.detail}" if self.detail else "")


class AuditLog:
    """Zapis zdarzeń do bazy i do logu aplikacji.

    Baza jest opcjonalna — bez niej zostaje sam log tekstowy, co przy pracy bez
    pamięci trwałej (``MEMORY_ENABLED=false``) jest jedynym możliwym śladem.
    """

    def __init__(self, database: Any | None = None, *, enabled: bool = True) -> None:
        self._db = database
        self._enabled = bool(enabled)
        self._entries: list[AuditEntry] = []
        self._db_failed = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        """Zdarzenia z tej sesji (także wtedy, gdy bazy nie ma)."""
        return tuple(self._entries)

    def record(self, entry: AuditEntry) -> AuditEntry:
        """Zapisz zdarzenie. Nigdy nie rzuca."""
        if not self._enabled:
            return entry

        self._entries.append(entry)
        logger.info("AUDYT %s", entry.as_line())

        if self._db is None or self._db_failed:
            return entry
        try:
            self._db.tool_audit.add(
                tool=entry.tool,
                risk=entry.risk.value,
                decision=entry.decision,
                ok=entry.ok,
                confirmed=entry.confirmed,
                arguments_hash=entry.arguments_hash,
                duration_ms=entry.duration_ms,
                detail=entry.detail,
                conversation_id=entry.conversation_id,
            )
        except Exception as exc:
            # Raz ostrzegamy i przestajemy próbować — inaczej każde wywołanie
            # narzędzia zasypywałoby log tym samym błędem bazy.
            self._db_failed = True
            logger.warning("Nie zapisano wpisu audytu do bazy (dalsze pomijam): %s", exc)
            logger.debug("Szczegóły błędu zapisu audytu", exc_info=True)
        return entry

    def recent(self, limit: int = 20) -> list[AuditEntry]:
        """Ostatnie zdarzenia z tej sesji (od najnowszego)."""
        return list(reversed(self._entries[-max(1, limit) :]))

    def summary(self) -> str:
        """Jedna linijka do ``/status``."""
        if not self._enabled:
            return t("status.audit.off")
        if not self._entries:
            return t("status.audit.empty")
        denied = sum(
            1
            for entry in self._entries
            if entry.decision in (DECISION_DENIED, DECISION_USER_DENIED)
        )
        where = (
            t("status.audit.where_db")
            if self._db is not None and not self._db_failed
            else t("status.audit.where_log")
        )
        return t("status.audit.summary", count=len(self._entries), denied=denied, where=where)


__all__ = [
    "DECISION_ALLOWED",
    "DECISION_CONFIRMED",
    "DECISION_DENIED",
    "DECISION_DRY_RUN",
    "DECISION_ERROR",
    "DECISION_INVALID",
    "DECISION_REPEATED",
    "DECISION_UNKNOWN_TOOL",
    "DECISION_USER_DENIED",
    "AuditEntry",
    "AuditLog",
    "hash_arguments",
]
