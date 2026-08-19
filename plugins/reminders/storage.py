"""Przechowywanie przypomnień w bazie z Fazy 5 (Faza 11).

Dlaczego w SQLite, a nie w pamięci procesu
------------------------------------------

Przypomnienie ustawione na 7:00 ma zadziałać także wtedy, gdy asystent zostanie
po drodze zamknięty i uruchomiony ponownie — a to jest normalna rzecz przy
programie na własnym komputerze. Stan w pamięci procesu znikałby przy każdym
restarcie i „obudź mnie o 7" byłoby obietnicą bez pokrycia.

Dlaczego plugin tworzy tabelę sam
---------------------------------

Migracje w ``database/migrations.py`` opisują schemat ASYSTENTA. Gdyby każdy
plugin musiał tam coś dopisać, „plugin bez zmian w kodzie głównym" przestałoby
być prawdą przy pierwszym pluginie trzymającym stan. Dlatego plugin zakłada
własną tabelę (``CREATE TABLE IF NOT EXISTS``) z przedrostkiem ``plugin_``.

Cena tego wyboru jest jawna: tabele pluginów nie mają historii migracji, więc
zmiana ich schematu w przyszłej wersji pluginu jest zadaniem samego pluginu.
Przy jednej tabeli i kilku kolumnach to uczciwy zamian za brak sprzężenia.

Czas trzymamy w UTC (ISO 8601), tak jak reszta bazy. Zamiana na czas lokalny
jest sprawą warstwy, która pokazuje wynik człowiekowi.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

logger = logging.getLogger(__name__)

TABLE_NAME: Final[str] = "plugin_reminders"

# Stany przypomnienia. „fired" zostaje w tabeli, bo „co mi dzwoniło rano?" to
# sensowne pytanie, a skasowany wiersz nie umie na nie odpowiedzieć.
STATE_ACTIVE: Final[str] = "active"
STATE_FIRED: Final[str] = "fired"
STATE_CANCELLED: Final[str] = "cancelled"

_SCHEMA: Final[tuple[str, ...]] = (
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL,
        due_at      TEXT NOT NULL,
        text        TEXT NOT NULL,
        state       TEXT NOT NULL DEFAULT '{STATE_ACTIVE}',
        fired_at    TEXT,
        -- Nazwa narzędzia, które ma pójść w ruch po terminie (puste = samo
        -- przypomnienie). To pole decyduje o poziomie ryzyka przy planowaniu.
        action      TEXT NOT NULL DEFAULT '',
        source      TEXT NOT NULL DEFAULT ''
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_due ON {TABLE_NAME}(state, due_at)",
)


class ReminderError(RuntimeError):
    """Błąd przechowywania przypomnień, nadający się do pokazania człowiekowi."""


@dataclass(frozen=True, slots=True)
class Reminder:
    """Jedno zaplanowane przypomnienie."""

    id: int
    text: str
    due_at: datetime
    created_at: datetime
    state: str = STATE_ACTIVE
    action: str = ""
    source: str = ""
    fired_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.state == STATE_ACTIVE

    def local_due(self) -> datetime:
        """Termin w strefie czasowej TEJ maszyny — do pokazania człowiekowi.

        Baza trzyma UTC; użytkownik myśli w swojej strefie. Konwersja jest tu, a
        nie w bazie, bo ta sama baza może zostać przeniesiona na inną maszynę.
        """
        return self.due_at.astimezone()

    def describe(self, *, now: datetime | None = None) -> str:
        moment = self.local_due()
        stamp = moment.strftime("%Y-%m-%d %H:%M")
        if now is not None and self.active:
            minutes = int((self.due_at - now).total_seconds() // 60)
            if 0 <= minutes < 60:
                return f"#{self.id} {self.text} — za {minutes} min ({stamp})"
        return f"#{self.id} {self.text} — {stamp}"


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:  # pragma: no cover - uszkodzony wiersz
        logger.warning("Nieczytelna data w tabeli %s: %r", TABLE_NAME, value)
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _row_to_reminder(row: sqlite3.Row) -> Reminder:
    due = _parse(row["due_at"]) or datetime.now(UTC)
    created = _parse(row["created_at"]) or due
    return Reminder(
        id=int(row["id"]),
        text=str(row["text"]),
        due_at=due,
        created_at=created,
        state=str(row["state"]),
        action=str(row["action"] or ""),
        source=str(row["source"] or ""),
        fired_at=_parse(row["fired_at"]),
    )


class ReminderStore:
    """Dostęp do tabeli przypomnień. Cała wiedza o SQL-u jest tutaj.

    Obiekt bazy jest wstrzykiwany (``Database`` z Fazy 5 albo atrapa w teście),
    więc ten kod nie wie, gdzie leży plik ani czy w ogóle jest plikiem.
    """

    def __init__(self, database: Any) -> None:
        if database is None:
            raise ReminderError("przypomnienia wymagają działającej bazy danych")
        self._db = database
        self._ready = False

    # --- schemat ----------------------------------------------------------- #

    def ensure_schema(self) -> None:
        """Załóż tabelę, jeśli jej nie ma. Wołane leniwie, przy pierwszym użyciu."""
        if self._ready:
            return
        try:
            with self._db.transaction() as connection:
                for statement in _SCHEMA:
                    connection.execute(statement)
        except Exception as exc:  # sqlite3.Error, DatabaseError, atrapy w testach
            raise ReminderError(f"nie mogę przygotować tabeli przypomnień ({exc})") from exc
        self._ready = True

    # --- zapis ------------------------------------------------------------- #

    def add(
        self,
        text: str,
        due_at: datetime,
        *,
        action: str = "",
        source: str = "",
        now: datetime | None = None,
    ) -> Reminder:
        """Zaplanuj przypomnienie. Zwraca gotowy wpis razem z nadanym numerem."""
        self.ensure_schema()
        moment = now or datetime.now(UTC)
        due = due_at.astimezone(UTC)
        try:
            with self._db.transaction() as connection:
                cursor = connection.execute(
                    f"INSERT INTO {TABLE_NAME} (created_at, due_at, text, state, action, source) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        moment.isoformat(),
                        due.isoformat(),
                        text.strip(),
                        STATE_ACTIVE,
                        action.strip(),
                        source.strip(),
                    ),
                )
                identifier = int(cursor.lastrowid or 0)
        except Exception as exc:
            raise ReminderError(f"nie udało się zapisać przypomnienia ({exc})") from exc

        return Reminder(
            id=identifier,
            text=text.strip(),
            due_at=due,
            created_at=moment,
            action=action.strip(),
            source=source.strip(),
        )

    def cancel(self, reminder_id: int) -> Reminder | None:
        """Odwołaj przypomnienie. ``None`` = nie było takiego (albo już nieaktywne)."""
        self.ensure_schema()
        found = self.get(reminder_id)
        if found is None or not found.active:
            return None
        with self._db.transaction() as connection:
            connection.execute(
                f"UPDATE {TABLE_NAME} SET state = ? WHERE id = ?",
                (STATE_CANCELLED, reminder_id),
            )
        return found

    def mark_fired(self, reminder_id: int, *, now: datetime | None = None) -> None:
        """Zaznacz, że przypomnienie już się odezwało.

        Osobny stan, a nie usunięcie wiersza: dzięki temu jedno przypomnienie
        nie odezwie się drugi raz po restarcie, a użytkownik może zapytać, co
        mu dzwoniło.
        """
        self.ensure_schema()
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        with self._db.transaction() as connection:
            connection.execute(
                f"UPDATE {TABLE_NAME} SET state = ?, fired_at = ? WHERE id = ? AND state = ?",
                (STATE_FIRED, moment.isoformat(), reminder_id, STATE_ACTIVE),
            )

    # --- odczyt ------------------------------------------------------------ #

    def get(self, reminder_id: int) -> Reminder | None:
        self.ensure_schema()
        row = self._db.query_one(f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (reminder_id,))
        return _row_to_reminder(row) if row is not None else None

    def active(self, *, limit: int = 50) -> list[Reminder]:
        """Zaplanowane przypomnienia, od najbliższego terminu."""
        self.ensure_schema()
        rows = self._db.query(
            f"SELECT * FROM {TABLE_NAME} WHERE state = ? ORDER BY due_at ASC LIMIT ?",
            (STATE_ACTIVE, int(limit)),
        )
        return [_row_to_reminder(row) for row in rows]

    def count_active(self) -> int:
        self.ensure_schema()
        row = self._db.query_one(
            f"SELECT COUNT(*) AS ile FROM {TABLE_NAME} WHERE state = ?", (STATE_ACTIVE,)
        )
        return int(row["ile"]) if row is not None else 0

    def due(self, now: datetime | None = None, *, limit: int = 20) -> list[Reminder]:
        """Przypomnienia, których termin już minął, a które jeszcze nie dzwoniły."""
        self.ensure_schema()
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        rows = self._db.query(
            f"SELECT * FROM {TABLE_NAME} WHERE state = ? AND due_at <= ? "
            "ORDER BY due_at ASC LIMIT ?",
            (STATE_ACTIVE, moment.isoformat(), int(limit)),
        )
        return [_row_to_reminder(row) for row in rows]

    def recent(self, *, limit: int = 20) -> list[Reminder]:
        self.ensure_schema()
        rows = self._db.query(
            f"SELECT * FROM {TABLE_NAME} ORDER BY due_at DESC LIMIT ?", (int(limit),)
        )
        return [_row_to_reminder(row) for row in rows]

    # --- sprzątanie -------------------------------------------------------- #

    def purge_older_than(self, days: int, *, now: datetime | None = None) -> int:
        """Usuń stare, już zrealizowane wpisy. ``days=0`` = nie sprzątaj."""
        if days <= 0:
            return 0
        self.ensure_schema()
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = moment.timestamp() - days * 86_400
        border = datetime.fromtimestamp(cutoff, UTC).isoformat()
        with self._db.transaction() as connection:
            cursor = connection.execute(
                f"DELETE FROM {TABLE_NAME} WHERE state != ? AND due_at < ?",
                (STATE_ACTIVE, border),
            )
            return int(cursor.rowcount or 0)


__all__: Sequence[str] = [
    "STATE_ACTIVE",
    "STATE_CANCELLED",
    "STATE_FIRED",
    "TABLE_NAME",
    "Reminder",
    "ReminderError",
    "ReminderStore",
]
