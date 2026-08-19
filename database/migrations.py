"""Migracje schematu bazy — wersjonowane, tylko „w przód", w transakcji.

Wersja schematu żyje w ``PRAGMA user_version`` (liczba w nagłówku pliku bazy,
czytana bez żadnej tabeli) i jest dublowana w tabeli ``schema_migrations``, żeby
dało się zobaczyć, co i kiedy zostało zastosowane.

Zasady:

* migracje wykonują się po kolei, każda we własnej transakcji — przerwana
  migracja nie zostawia bazy w połowie zmiany,
* przed pierwszą migracją NA ISTNIEJĄCEJ bazie robimy kopię pliku (sqlite3
  ``Connection.backup()``, a nie kopiowanie pliku „na żywca" — to jedyny sposób
  bezpieczny przy włączonym WAL),
* niczego nie cofamy; naprawa błędnej migracji to kolejna migracja.

Świadomie bez Alembica: jedna lokalna baza pliku nie potrzebuje SQLAlchemy, a
warstwa repozytoriów i tak nie zna SQL-a spoza tego pakietu.

FTS5 NIE jest częścią łańcucha migracji — patrz :func:`ensure_fulltext_index`.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Migration:
    """Jeden krok zmiany schematu."""

    version: int
    name: str
    statements: tuple[str, ...]


_MIGRATION_001 = Migration(
    version=1,
    name="pamiec_podstawowa",
    statements=(
        # --- ślad zastosowanych migracji ---
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            applied_at  TEXT NOT NULL
        )
        """,
        # --- rozmowy ---
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at    TEXT NOT NULL,
            ended_at      TEXT,
            title         TEXT NOT NULL DEFAULT '',
            source        TEXT NOT NULL DEFAULT 'terminal',
            model         TEXT NOT NULL DEFAULT '',
            message_count INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_conversations_started ON conversations(started_at DESC)",
        # --- wiadomości ---
        """
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            language        TEXT NOT NULL DEFAULT '',
            metadata        TEXT NOT NULL DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC)",
        # --- streszczenia (zastępują obcięte wiadomości) ---
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id      INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            content              TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            message_count        INTEGER NOT NULL DEFAULT 0,
            covers_to_message_id INTEGER,
            generation           INTEGER NOT NULL DEFAULT 1,
            method               TEXT NOT NULL DEFAULT 'llm'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_summaries_conversation ON summaries(conversation_id, id)",
        # --- fakty o użytkowniku ---
        """
        CREATE TABLE IF NOT EXISTS facts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            key        TEXT NOT NULL UNIQUE,
            value      TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'user',
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            pinned     INTEGER NOT NULL DEFAULT 1
        )
        """,
        # --- preferencje ---
        """
        CREATE TABLE IF NOT EXISTS preferences (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            key        TEXT NOT NULL UNIQUE,
            value      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        # --- notatki ---
        """
        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL DEFAULT '',
            body       TEXT NOT NULL,
            tags       TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT 'user'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at DESC)",
        # --- metadane wspomnień (warstwa nad treścią) ---
        """
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            kind         TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_id    INTEGER NOT NULL,
            summary      TEXT NOT NULL DEFAULT '',
            importance   REAL NOT NULL DEFAULT 0.5,
            created_at   TEXT NOT NULL,
            last_used_at TEXT,
            use_count    INTEGER NOT NULL DEFAULT 0,
            expires_at   TEXT,
            metadata     TEXT NOT NULL DEFAULT '{}',
            UNIQUE(source_table, source_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind, importance DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_expiry ON memories(expires_at)",
    ),
)


_MIGRATION_002 = Migration(
    version=2,
    name="embeddingi",
    statements=(
        # Wektory wspomnień (Faza 6). Trzymamy je TUTAJ, a nie w pliku indeksu
        # FAISS, bo baza jest jedynym źródłem prawdy: indeks da się odtworzyć
        # w każdej chwili, a plik indeksu jest związany z wersją biblioteki i
        # nie przenosi się między maszynami tak dobrze jak SQLite.
        #
        # ``model`` i ``dim`` są przy KAŻDYM wektorze, bo wektorów z różnych
        # modeli nie wolno ze sobą porównywać — walidacja następuje przy odczycie.
        # ``vector`` to surowe float32 little-endian (patrz database/models.py).
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source_table TEXT NOT NULL,
            source_id    INTEGER NOT NULL,
            text         TEXT NOT NULL,
            model        TEXT NOT NULL,
            dim          INTEGER NOT NULL,
            vector       BLOB NOT NULL,
            created_at   TEXT NOT NULL,
            UNIQUE(source_table, source_id, model)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model, dim)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_table, source_id)",
    ),
)


_MIGRATION_003 = Migration(
    version=3,
    name="audyt_narzedzi",
    statements=(
        # Log wywołań narzędzi (Faza 7). Tabela jest **tylko do dopisywania**:
        # repozytorium nie ma metody usuwającej, a narzędzia nie mają dostępu do
        # bazy wprost — inaczej narzędzie mogłoby zatrzeć po sobie ślad.
        #
        # Zapisujemy SKRÓT argumentów (sha256), nie ich treść: do zbadania „co
        # się stało i czy było potwierdzone" wystarczy, a prywatnych danych
        # użytkownika nie duplikujemy w drugim miejscu.
        """
        CREATE TABLE IF NOT EXISTS tool_audit (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            conversation_id INTEGER,
            tool            TEXT NOT NULL,
            risk            TEXT NOT NULL,
            decision        TEXT NOT NULL,
            ok              INTEGER NOT NULL DEFAULT 0,
            confirmed       INTEGER NOT NULL DEFAULT 0,
            arguments_hash  TEXT NOT NULL DEFAULT '',
            duration_ms     INTEGER NOT NULL DEFAULT 0,
            detail          TEXT NOT NULL DEFAULT ''
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_tool_audit_created ON tool_audit(created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_tool_audit_tool ON tool_audit(tool, created_at DESC)",
    ),
)


MIGRATIONS: tuple[Migration, ...] = (_MIGRATION_001, _MIGRATION_002, _MIGRATION_003)

SCHEMA_VERSION: int = max(migration.version for migration in MIGRATIONS)


# --------------------------------------------------------------------------- #
# Wersja schematu
# --------------------------------------------------------------------------- #


def current_version(connection: sqlite3.Connection) -> int:
    """Wersja schematu zapisana w pliku bazy (0 = pusta/nowa baza)."""
    row = connection.execute("PRAGMA user_version").fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):  # pragma: no cover - uszkodzony nagłówek
        return 0


def pending_migrations(connection: sqlite3.Connection) -> list[Migration]:
    """Migracje, których ta baza jeszcze nie ma."""
    version = current_version(connection)
    return [migration for migration in MIGRATIONS if migration.version > version]


def is_up_to_date(connection: sqlite3.Connection) -> bool:
    return not pending_migrations(connection)


# --------------------------------------------------------------------------- #
# Kopia zapasowa przed migracją
# --------------------------------------------------------------------------- #


def backup_database(connection: sqlite3.Connection, target: Path) -> Path | None:
    """Skopiuj bazę do ``target`` mechanizmem sqlite3 (bezpiecznym przy WAL).

    Zwraca ścieżkę kopii albo ``None``, gdy kopii nie dało się zrobić — brak
    kopii jest ostrzeżeniem, nie powodem do zatrzymania asystenta.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        destination = sqlite3.connect(str(target))
        try:
            with destination:
                connection.backup(destination)
        finally:
            destination.close()
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Nie udało się zrobić kopii bazy w %s: %s", target, exc)
        return None
    logger.info("Kopia bazy przed migracją: %s", target)
    return target


def backup_name(source: Path, version: int) -> Path:
    """Nazwa pliku kopii: ``miku.sqlite3.v1.bak`` obok oryginału."""
    return source.with_name(f"{source.name}.v{version}.bak")


# --------------------------------------------------------------------------- #
# Uruchamianie migracji
# --------------------------------------------------------------------------- #


def _record_migration(connection: sqlite3.Connection, migration: Migration) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (
            migration.version,
            migration.name,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def apply_migrations(
    connection: sqlite3.Connection,
    *,
    database_path: Path | None = None,
    backup: bool = True,
) -> list[Migration]:
    """Doprowadź schemat do :data:`SCHEMA_VERSION`. Zwraca zastosowane migracje.

    ``database_path`` służy wyłącznie do nazwania kopii zapasowej; baza w pamięci
    (``:memory:``) kopii nie dostaje, bo nie ma czego kopiować.
    """
    pending = pending_migrations(connection)
    if not pending:
        return []

    version_before = current_version(connection)
    if backup and version_before > 0 and database_path is not None and database_path.exists():
        backup_database(connection, backup_name(database_path, version_before))

    applied: list[Migration] = []
    for migration in pending:
        logger.info("Migracja bazy %s → %s (%s)", version_before, migration.version, migration.name)
        try:
            # BEGIN IMMEDIATE od razu bierze blokadę zapisu: druga instancja
            # asystenta uruchomiona w tej samej sekundzie dostanie „database is
            # locked" i poczeka (busy_timeout), zamiast migrować równolegle.
            connection.execute("BEGIN IMMEDIATE")
            for statement in migration.statements:
                connection.execute(statement)
            _record_migration(connection, migration)
            # PRAGMA user_version nie przyjmuje parametru — stąd literał.
            # Wartość pochodzi z kodu (int w dataclassie), nie od użytkownika.
            connection.execute(f"PRAGMA user_version = {int(migration.version)}")
            connection.execute("COMMIT")
        except sqlite3.Error:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover - rollback po zerwanym połączeniu
                pass
            logger.exception("Migracja %s (%s) nie powiodła się", migration.version, migration.name)
            raise
        applied.append(migration)
        version_before = migration.version

    return applied


# --------------------------------------------------------------------------- #
# Indeks pełnotekstowy (FTS5) — opcjonalny z konieczności
# --------------------------------------------------------------------------- #


def supports_fts5(connection: sqlite3.Connection) -> bool:
    """Czy TA biblioteka SQLite ma wkompilowany moduł FTS5?

    FTS5 to opcja kompilacji, a nie część standardu: Python na Windowsie i
    macOS ma ją zwykle włączoną, ale SQLite z dystrybucji Linuksa czy z
    minimalnego obrazu kontenera bywa budowany bez niej. Dlatego indeks jest
    tworzony warunkowo, a wyszukiwanie ma wariant zapasowy na ``LIKE``.
    """
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
    except sqlite3.Error:
        return False
    try:
        connection.execute("DROP TABLE temp.__fts5_probe")
    except sqlite3.Error:  # pragma: no cover - sprzątanie po sondzie
        pass
    return True


# Indeks nie należy do łańcucha migracji celowo: gdyby należał, baza założona na
# maszynie BEZ FTS5 zatrzymałaby się na tej wersji i nie dostała już żadnej
# kolejnej migracji. Tutaj jest idempotentnie dokładany przy każdym otwarciu —
# przenosiny bazy na maszynę z FTS5 same włączają wyszukiwanie pełnotekstowe.
_FTS_STATEMENTS: tuple[str, ...] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        content='messages',
        content_rowid='id',
        tokenize="unicode61 remove_diacritics 2"
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
        title,
        body,
        content='notes',
        content_rowid='id',
        tokenize="unicode61 remove_diacritics 2"
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_insert AFTER INSERT ON notes BEGIN
        INSERT INTO notes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_delete AFTER DELETE ON notes BEGIN
        INSERT INTO notes_fts(notes_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_update AFTER UPDATE ON notes BEGIN
        INSERT INTO notes_fts(notes_fts, rowid, title, body)
        VALUES ('delete', old.id, old.title, old.body);
        INSERT INTO notes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
    END
    """,
)


def ensure_fulltext_index(connection: sqlite3.Connection) -> bool:
    """Załóż indeks FTS5 nad wiadomościami i notatkami, jeśli SQLite to potrafi.

    Zwraca ``True``, gdy indeks jest gotowy do użycia. ``False`` = ta maszyna nie
    ma FTS5 albo indeks się nie założył; wyszukiwanie zejdzie wtedy na ``LIKE``,
    co przy skali osobistego asystenta (10⁴–10⁵ wiadomości) nadal działa.
    """
    if not supports_fts5(connection):
        logger.info("SQLite bez modułu FTS5 — wyszukiwanie w pamięci zejdzie na LIKE.")
        return False

    try:
        connection.execute("BEGIN IMMEDIATE")
        fresh = _table_missing(connection, "messages_fts")
        for statement in _FTS_STATEMENTS:
            connection.execute(statement)
        if fresh:
            # Indeks założony na już istniejących danych startuje pusty —
            # 'rebuild' wypełnia go treścią z tabel źródłowych.
            connection.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
            connection.execute("INSERT INTO notes_fts(notes_fts) VALUES ('rebuild')")
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover
            pass
        logger.warning("Nie udało się przygotować indeksu pełnotekstowego: %s", exc)
        return False
    return True


def _table_missing(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (name,)
    ).fetchone()
    return row is None


def describe_migrations(connection: sqlite3.Connection) -> Sequence[tuple[int, str, str]]:
    """Zastosowane migracje (wersja, nazwa, kiedy) — do diagnostyki."""
    try:
        rows = connection.execute(
            "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()
    except sqlite3.Error:
        return ()
    return tuple((int(row[0]), str(row[1]), str(row[2])) for row in rows)


__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Migration",
    "apply_migrations",
    "backup_database",
    "backup_name",
    "current_version",
    "describe_migrations",
    "ensure_fulltext_index",
    "is_up_to_date",
    "pending_migrations",
    "supports_fts5",
]
