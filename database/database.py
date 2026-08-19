"""Połączenie z SQLite, transakcje i repozytoria pamięci długoterminowej (Faza 5).

Jedyne miejsce w projekcie, które pisze SQL. ``brain/memory.py`` rozmawia z
repozytoriami (``db.facts.set(...)``), więc podmiana SQLite na cokolwiek innego
nie dotyka warstwy „mózgu".

Czego ten moduł NIE robi:

* nie wie, gdzie leży plik bazy — ścieżkę dostaje z :func:`config.database_file`,
* nie zakłada, że katalog danych istnieje ani że da się w nim pisać — brak prawa
  zapisu kończy się :class:`DatabaseError` z podpowiedzią, a nie stack trace'em,
* nie zakłada, że SQLite ma FTS5 ani że dziennik WAL da się włączyć (nie da się
  na udziale sieciowym) — jedno i drugie jest wykrywane i degraduje łagodnie.

Wątki: ``sqlite3`` zabrania używania jednego połączenia w wielu wątkach, więc
każdy wątek dostaje własne (``threading.local``). Zapisy są dodatkowo
serializowane blokadą — WAL pozwala na wielu czytelników, ale tylko jednego
piszącego naraz.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

from config import (
    MEMORY_DATABASE,
    DependencyCheck,
    DependencyContext,
    Settings,
    database_file,
    get_settings,
    register_dependency_check,
)
from database.migrations import (
    SCHEMA_VERSION,
    apply_migrations,
    current_version,
    describe_migrations,
    ensure_fulltext_index,
)
from database.models import (
    MEMORY_KIND_FACT,
    MEMORY_KIND_NOTE,
    MEMORY_KIND_PREFERENCE,
    Conversation,
    EmbeddingRecord,
    Fact,
    MemoryEntry,
    MemoryStats,
    Note,
    Preference,
    SearchHit,
    StoredMessage,
    Summary,
    ToolAuditRecord,
    dump_metadata,
    encode_vector,
    from_iso,
    to_iso,
    utc_now,
)
from i18n import t

logger = logging.getLogger(__name__)

# Tokeny zapytania pełnotekstowego. Wszystko, co nie jest literą ani cyfrą,
# wypada — FTS5 ma własną składnię (``*``, ``"``, ``-``, ``NEAR``) i tekst
# użytkownika wklejony do niej wprost potrafi wywrócić zapytanie błędem składni.
_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\W_]+", re.UNICODE)
# Znaki specjalne LIKE-a. Escape'ujemy je, żeby „100%" szukało procenta, a nie
# czegokolwiek.
_LIKE_SPECIAL: Final[re.Pattern[str]] = re.compile(r"([\\%_])")


class DatabaseError(RuntimeError):
    """Błąd bazy nadający się do pokazania użytkownikowi."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n       Podpowiedź: {self.hint}"
        return self.message


def _escape_like(text: str) -> str:
    return _LIKE_SPECIAL.sub(r"\\\1", text)


def _fts_query(text: str) -> str:
    """Zamień tekst użytkownika na bezpieczne zapytanie FTS5 (wszystkie słowa).

    Każde słowo dostaje gwiazdkę (dopasowanie po przedrostku), bo FTS5 nie zna
    odmiany: bez tego „rower" nie znalazłby „roweru", a po polsku rozmawia się
    właśnie odmienionymi formami. Przy okazji upodabnia to wynik do wariantu
    zapasowego na ``LIKE``, który zawsze szuka po fragmencie.
    """
    words = [match.group(0) for match in _WORD_PATTERN.finditer(text)]
    return " ".join(f'"{word}"*' for word in words)


# --------------------------------------------------------------------------- #
# Połączenie
# --------------------------------------------------------------------------- #


class Database:
    """Baza SQLite asystenta: połączenia, pragmy, migracje, repozytoria."""

    def __init__(
        self,
        path: Path | str,
        *,
        timeout: float = 5.0,
        backup_before_migration: bool = True,
        apply_schema: bool = True,
    ) -> None:
        raw = str(path)
        self._in_memory = raw == MEMORY_DATABASE
        self._path = Path(raw)
        self._timeout = max(0.1, float(timeout))
        self._backup = backup_before_migration
        self._lock = threading.RLock()
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._shared: sqlite3.Connection | None = None
        self._closed = False
        self._journal_mode = "unknown"
        self._has_fulltext = False

        self.conversations = ConversationRepository(self)
        self.messages = MessageRepository(self)
        self.summaries = SummaryRepository(self)
        self.facts = FactRepository(self)
        self.preferences = PreferenceRepository(self)
        self.notes = NoteRepository(self)
        self.memories = MemoryRepository(self)
        self.embeddings = EmbeddingRepository(self)
        self.tool_audit = ToolAuditRepository(self)

        connection = self._open_connection()
        if self._in_memory:
            # Baza w pamięci ŻYJE w połączeniu: drugie połączenie to druga,
            # pusta baza. Dlatego tu — i tylko tu — wszystkie wątki dzielą jedno.
            self._shared = connection
        if apply_schema:
            self._prepare_schema(connection)

    # --- otwieranie ------------------------------------------------------ #

    @classmethod
    def open(cls, settings: Settings | None = None) -> Database:
        """Otwórz bazę wskazaną przez konfigurację (``DATABASE_PATH`` albo katalog systemu)."""
        active = settings or get_settings()
        return cls(
            database_file(active),
            timeout=active.database_timeout_s,
            backup_before_migration=active.database_backup_before_migration,
        )

    def _open_connection(self) -> sqlite3.Connection:
        if not self._in_memory:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise DatabaseError(
                    f"Nie mogę utworzyć katalogu na bazę pamięci: {self._path.parent}",
                    hint=(
                        "wskaż inne miejsce zmienną MIKU_DATA_DIR albo wpisem "
                        "DATABASE_PATH w .env, albo wyłącz pamięć: MEMORY_ENABLED=false"
                    ),
                ) from exc

        try:
            connection = sqlite3.connect(
                str(self._path) if not self._in_memory else MEMORY_DATABASE,
                timeout=self._timeout,
                # Baza w pamięci jest dzielona między wątkami świadomie (patrz
                # wyżej); serializuje ją ta sama blokada, co zapisy do pliku.
                check_same_thread=not self._in_memory,
                # Transakcjami sterujemy sami (BEGIN IMMEDIATE w transaction()),
                # bo tylko wtedy wiadomo, co dokładnie jest w transakcji.
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Nie mogę otworzyć bazy pamięci: {self._path} ({exc})",
                hint=(
                    "sprawdź prawa zapisu do katalogu, wskaż inny ścieżką DATABASE_PATH "
                    "w .env albo wyłącz pamięć: MEMORY_ENABLED=false"
                ),
            ) from exc

        connection.row_factory = sqlite3.Row
        self._configure(connection)
        with self._lock:
            self._connections.append(connection)
        return connection

    def _configure(self, connection: sqlite3.Connection) -> None:
        """Pragmy ustawiane na KAŻDYM połączeniu (nie zapisują się w pliku)."""
        # Klucze obce są w SQLite domyślnie WYŁĄCZONE i włącza się je osobno na
        # każdym połączeniu — bez tego kasowanie rozmowy zostawiałoby sieroty.
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")

        if self._in_memory:
            self._journal_mode = "memory"
            return

        # WAL jest zapisywany w pliku bazy raz, ale nie działa na wszystkim:
        # na udziale sieciowym (SMB/NFS) SQLite odmawia i zostaje przy dzienniku
        # DELETE. To nie jest błąd — po prostu wolniej i bez równoległych odczytów.
        try:
            row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            self._journal_mode = str(row[0]).lower() if row else "unknown"
        except sqlite3.Error as exc:  # pragma: no cover - zależne od systemu plików
            logger.warning("Nie udało się ustawić dziennika WAL: %s", exc)
            self._journal_mode = "unknown"

        if self._journal_mode != "wal":
            logger.warning(
                "Baza %s pracuje w trybie dziennika %r zamiast WAL — "
                "prawdopodobnie leży na dysku sieciowym. Będzie działać wolniej.",
                self._path,
                self._journal_mode,
            )
        else:
            # NORMAL jest bezpieczne przy WAL (przy pełnym zaniku zasilania można
            # stracić ostatnie transakcje, ale baza nie ulega uszkodzeniu).
            connection.execute("PRAGMA synchronous = NORMAL")

    def _prepare_schema(self, connection: sqlite3.Connection) -> None:
        try:
            applied = apply_migrations(
                connection,
                database_path=None if self._in_memory else self._path,
                backup=self._backup,
            )
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"Nie udało się przygotować schematu bazy: {exc}",
                hint=(
                    f"plik bazy: {self._path} — jeśli jest uszkodzony, przenieś go na bok; "
                    "asystent założy nowy (pamięć zostanie utracona)"
                ),
            ) from exc
        if applied:
            logger.info(
                "Zastosowano migracje: %s", ", ".join(f"{m.version}:{m.name}" for m in applied)
            )
        self._has_fulltext = ensure_fulltext_index(connection)

    # --- właściwości ----------------------------------------------------- #

    @property
    def path(self) -> Path:
        return self._path

    @property
    def in_memory(self) -> bool:
        return self._in_memory

    @property
    def journal_mode(self) -> str:
        return self._journal_mode

    @property
    def has_fulltext(self) -> bool:
        """Czy działa wyszukiwanie FTS5 (``False`` = wariant zapasowy na LIKE)."""
        return self._has_fulltext

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def schema_version(self) -> int:
        return current_version(self.connection())

    def describe(self) -> str:
        """Jedna linijka do ``/status`` i do raportu zależności."""
        where = t("status.db.in_memory") if self._in_memory else str(self._path)
        return t(
            "status.db.describe",
            where=where,
            version=self.schema_version,
            latest=SCHEMA_VERSION,
            journal=self._journal_mode,
            search="FTS5" if self._has_fulltext else "LIKE",
        )

    def migration_history(self) -> Sequence[tuple[int, str, str]]:
        return describe_migrations(self.connection())

    # --- połączenia i transakcje ----------------------------------------- #

    def connection(self) -> sqlite3.Connection:
        """Połączenie dla BIEŻĄCEGO wątku (tworzone przy pierwszym użyciu)."""
        if self._closed:
            raise DatabaseError("Baza pamięci jest już zamknięta.")
        if self._shared is not None:
            return self._shared
        existing: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if existing is not None:
            return existing
        connection = self._open_connection()
        self._local.connection = connection
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transakcja zapisu. Zagnieżdżenie jest bezpieczne (liczy się głębokość)."""
        connection = self.connection()
        with self._lock:
            depth = int(getattr(self._local, "depth", 0))
            self._local.depth = depth + 1
            try:
                if depth == 0:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
            except BaseException:
                if depth == 0:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error:  # pragma: no cover
                        pass
                raise
            else:
                if depth == 0:
                    connection.execute("COMMIT")
            finally:
                self._local.depth = depth

    # --- pomocnicze zapytania -------------------------------------------- #

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Zapis w transakcji. Zwraca kursor (``lastrowid`` po INSERT-cie)."""
        try:
            with self.transaction() as connection:
                return connection.execute(sql, tuple(parameters))
        except sqlite3.Error as exc:
            raise self._wrap(exc) from exc

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        try:
            return list(self.connection().execute(sql, tuple(parameters)).fetchall())
        except sqlite3.Error as exc:
            raise self._wrap(exc) from exc

    def query_one(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Row | None:
        try:
            row: sqlite3.Row | None = self.connection().execute(sql, tuple(parameters)).fetchone()
        except sqlite3.Error as exc:
            raise self._wrap(exc) from exc
        return row

    def _wrap(self, exc: sqlite3.Error) -> DatabaseError:
        logger.error("Błąd bazy %s: %s", self._path, exc, exc_info=True)
        message = str(exc)
        if "locked" in message or "busy" in message:
            return DatabaseError(
                "Baza pamięci jest zajęta przez inny proces.",
                hint="zamknij drugą instancję asystenta albo zwiększ DATABASE_TIMEOUT_S w .env",
            )
        if "readonly" in message or "attempt to write" in message:
            return DatabaseError(
                f"Baza pamięci jest tylko do odczytu: {self._path}",
                hint="sprawdź prawa zapisu albo wskaż inne miejsce zmienną MIKU_DATA_DIR",
            )
        return DatabaseError(f"Błąd bazy pamięci: {message}", hint="szczegóły w logs/errors.log")

    # --- utrzymanie ------------------------------------------------------ #

    def stats(self) -> MemoryStats:
        """Ile czego jest w bazie."""
        counts: dict[str, int] = {}
        for table in (
            "conversations",
            "messages",
            "summaries",
            "facts",
            "preferences",
            "notes",
            "memories",
            "embeddings",
            "tool_audit",
        ):
            row = self.query_one(f"SELECT COUNT(*) AS liczba FROM {table}")  # nazwa z kodu
            counts[table] = int(row["liczba"]) if row else 0
        return MemoryStats(**counts)

    def purge_older_than(self, days: int) -> int:
        """Usuń rozmowy starsze niż ``days`` dni. 0 = nic nie usuwaj.

        Fakty, preferencje i notatki zostają — to pamięć trwała, a nie historia.
        """
        if days <= 0:
            return 0
        cutoff = to_iso(utc_now() - timedelta(days=days))
        cursor = self.execute("DELETE FROM conversations WHERE started_at < ?", (cutoff,))
        removed = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        if removed:
            logger.info("Usunięto %s rozmów starszych niż %s dni.", removed, days)
        return removed

    def close(self) -> None:
        """Zamknij wszystkie połączenia. Wywołanie po zamknięciu nic nie robi."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for connection in self._connections:
                # Połączenia założone w INNYCH wątkach odmówią tu współpracy
                # (sqlite3.ProgrammingError) — to nie błąd: po wyczyszczeniu
                # listy nikt już ich nie trzyma i zamyka je licznik referencji.
                try:
                    # Zalecenie dokumentacji SQLite: przed zamknięciem dać
                    # planiście okazję na aktualizację statystyk.
                    connection.execute("PRAGMA optimize")
                except sqlite3.Error:  # pragma: no cover - połączenie z innego wątku
                    pass
                try:
                    connection.close()
                except sqlite3.Error:  # pragma: no cover - jak wyżej
                    pass
            self._connections.clear()
            self._shared = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Database({self._path}, schema=v{SCHEMA_VERSION}, fts={self._has_fulltext})"


# --------------------------------------------------------------------------- #
# Repozytoria — jedyne miejsce z SQL-em dla danego rodzaju rekordu
# --------------------------------------------------------------------------- #


class _Repository:
    def __init__(self, database: Database) -> None:
        self._db = database


class ConversationRepository(_Repository):
    """Sesje rozmów."""

    def start(
        self, *, title: str = "", source: str = "terminal", model: str = ""
    ) -> Conversation:
        record = Conversation(title=title, source=source, model=model)
        cursor = self._db.execute(
            "INSERT INTO conversations (started_at, ended_at, title, source, model, message_count)"
            " VALUES (?, NULL, ?, ?, ?, 0)",
            (to_iso(record.started_at), record.title, record.source, record.model),
        )
        record.id = int(cursor.lastrowid or 0)
        return record

    def finish(self, conversation_id: int) -> None:
        self._db.execute(
            "UPDATE conversations SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (to_iso(utc_now()), conversation_id),
        )

    def set_title(self, conversation_id: int, title: str) -> None:
        self._db.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title.strip()[:200], conversation_id),
        )

    def get(self, conversation_id: int) -> Conversation | None:
        row = self._db.query_one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        return Conversation.from_row(row) if row else None

    def recent(self, limit: int = 10) -> list[Conversation]:
        rows = self._db.query(
            "SELECT * FROM conversations ORDER BY started_at DESC, id DESC LIMIT ?",
            (max(1, limit),),
        )
        return [Conversation.from_row(row) for row in rows]

    def last_finished(self, *, exclude_id: int | None = None) -> Conversation | None:
        """Poprzednia rozmowa — źródło „pamiętam, o czym mówiliśmy wcześniej"."""
        rows = self._db.query(
            "SELECT * FROM conversations WHERE id <> ? ORDER BY id DESC LIMIT 1",
            (exclude_id if exclude_id is not None else -1,),
        )
        return Conversation.from_row(rows[0]) if rows else None

    def delete(self, conversation_id: int) -> None:
        self._db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


class MessageRepository(_Repository):
    """Wiadomości rozmów (historia w sensie trwałym)."""

    def add(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        language: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredMessage:
        record = StoredMessage(
            conversation_id=conversation_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            language=language,
            metadata=dict(metadata or {}),
        )
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at, language,"
                " metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    record.role,
                    record.content,
                    to_iso(record.created_at),
                    record.language,
                    dump_metadata(record.metadata),
                ),
            )
            record.id = int(cursor.lastrowid or 0)
            connection.execute(
                "UPDATE conversations SET message_count = message_count + 1 WHERE id = ?",
                (conversation_id,),
            )
        return record

    def for_conversation(
        self, conversation_id: int, *, limit: int | None = None, newest_first: bool = False
    ) -> list[StoredMessage]:
        order = "DESC" if newest_first else "ASC"
        sql = f"SELECT * FROM messages WHERE conversation_id = ? ORDER BY id {order}"
        parameters: list[Any] = [conversation_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(1, limit))
        return [StoredMessage.from_row(row) for row in self._db.query(sql, parameters)]

    def count(self, conversation_id: int | None = None) -> int:
        if conversation_id is None:
            row = self._db.query_one("SELECT COUNT(*) AS liczba FROM messages")
        else:
            row = self._db.query_one(
                "SELECT COUNT(*) AS liczba FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
        return int(row["liczba"]) if row else 0

    def search(self, text: str, *, limit: int = 10) -> list[SearchHit]:
        """Znajdź wiadomości zawierające podany tekst (FTS5 albo LIKE)."""
        query = text.strip()
        if not query:
            return []
        limit = max(1, limit)

        if self._db.has_fulltext:
            match = _fts_query(query)
            if match:
                rows = self._db.query(
                    "SELECT m.id, m.content, m.created_at, m.role FROM messages_fts f"
                    " JOIN messages m ON m.id = f.rowid"
                    " WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, limit),
                )
                return [self._to_hit(row) for row in rows]

        rows = self._db.query(
            "SELECT id, content, created_at, role FROM messages"
            " WHERE content LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
            (f"%{_escape_like(query)}%", limit),
        )
        return [self._to_hit(row) for row in rows]

    @staticmethod
    def _to_hit(row: Any) -> SearchHit:
        return SearchHit(
            kind="message",
            source_id=int(row["id"]),
            text=str(row["content"]),
            created_at=from_iso(row["created_at"]),
            context=str(row["role"]),
        )


class SummaryRepository(_Repository):
    """Streszczenia — to one zastępują wiadomości wypchnięte z okna rozmowy."""

    def add(
        self,
        conversation_id: int,
        content: str,
        *,
        message_count: int = 0,
        covers_to_message_id: int | None = None,
        generation: int = 1,
        method: str = "llm",
    ) -> Summary:
        record = Summary(
            conversation_id=conversation_id,
            content=content,
            message_count=message_count,
            covers_to_message_id=covers_to_message_id,
            generation=generation,
            method=method,
        )
        cursor = self._db.execute(
            "INSERT INTO summaries (conversation_id, content, created_at, message_count,"
            " covers_to_message_id, generation, method) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                record.content,
                to_iso(record.created_at),
                record.message_count,
                record.covers_to_message_id,
                record.generation,
                record.method,
            ),
        )
        record.id = int(cursor.lastrowid or 0)
        return record

    def latest(self, conversation_id: int) -> Summary | None:
        row = self._db.query_one(
            "SELECT * FROM summaries WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        return Summary.from_row(row) if row else None

    def for_conversation(self, conversation_id: int) -> list[Summary]:
        rows = self._db.query(
            "SELECT * FROM summaries WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        )
        return [Summary.from_row(row) for row in rows]

    def count(self, conversation_id: int | None = None) -> int:
        if conversation_id is None:
            row = self._db.query_one("SELECT COUNT(*) AS liczba FROM summaries")
        else:
            row = self._db.query_one(
                "SELECT COUNT(*) AS liczba FROM summaries WHERE conversation_id = ?",
                (conversation_id,),
            )
        return int(row["liczba"]) if row else 0


class FactRepository(_Repository):
    """Fakty o użytkowniku — klucz jest unikalny, nowa wartość nadpisuje starą."""

    def set(
        self,
        key: str,
        value: str,
        *,
        source: str = "user",
        confidence: float = 1.0,
        pinned: bool = True,
        expires_at: Any = None,
    ) -> Fact:
        record = Fact(
            key=key,
            value=value,
            source=source,  # type: ignore[arg-type]
            confidence=confidence,
            pinned=pinned,
        )
        # Świadomie SELECT + UPDATE/INSERT zamiast „UPSERT" (ON CONFLICT DO
        # UPDATE): ten drugi wymaga SQLite >= 3.24, a Python gwarantuje tylko
        # 3.15. Na starszej bibliotece systemowej upsert wysypałby się składnią.
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT id, created_at FROM facts WHERE key = ?", (record.key,)
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    "INSERT INTO facts (key, value, source, confidence, created_at, updated_at,"
                    " pinned) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.key,
                        record.value,
                        record.source,
                        record.confidence,
                        to_iso(record.created_at),
                        to_iso(record.updated_at),
                        int(record.pinned),
                    ),
                )
                record.id = int(cursor.lastrowid or 0)
            else:
                record.id = int(row["id"])
                connection.execute(
                    "UPDATE facts SET value = ?, source = ?, confidence = ?, updated_at = ?,"
                    " pinned = ? WHERE id = ?",
                    (
                        record.value,
                        record.source,
                        record.confidence,
                        to_iso(record.updated_at),
                        int(record.pinned),
                        record.id,
                    ),
                )
        self._db.memories.record(
            kind=MEMORY_KIND_FACT,
            source_table="facts",
            source_id=record.id or 0,
            summary=record.as_line(),
            importance=min(1.0, 0.6 + confidence * 0.4),
            expires_at=expires_at,
        )
        return record

    def get(self, key: str) -> Fact | None:
        row = self._db.query_one("SELECT * FROM facts WHERE key = ?", (key.strip().lower(),))
        return Fact.from_row(row) if row else None

    def get_by_id(self, fact_id: int) -> Fact | None:
        """Fakt po identyfikatorze — tak wracają trafienia z indeksu wektorowego."""
        row = self._db.query_one("SELECT * FROM facts WHERE id = ?", (fact_id,))
        return Fact.from_row(row) if row else None

    def all(self, *, limit: int = 100) -> list[Fact]:
        rows = self._db.query(
            "SELECT * FROM facts ORDER BY pinned DESC, updated_at DESC LIMIT ?", (max(1, limit),)
        )
        return [Fact.from_row(row) for row in rows]

    def search(self, text: str, *, limit: int = 20) -> list[Fact]:
        pattern = f"%{_escape_like(text.strip())}%"
        rows = self._db.query(
            "SELECT * FROM facts WHERE key LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\'"
            " ORDER BY updated_at DESC LIMIT ?",
            (pattern, pattern, max(1, limit)),
        )
        return [Fact.from_row(row) for row in rows]

    def delete(self, key: str) -> bool:
        normalized = key.strip().lower()
        existing = self.get(normalized)
        if existing is None:
            return False
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM facts WHERE id = ?", (existing.id,))
            connection.execute(
                "DELETE FROM memories WHERE source_table = 'facts' AND source_id = ?",
                (existing.id,),
            )
        return True

    def count(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS liczba FROM facts")
        return int(row["liczba"]) if row else 0


class PreferenceRepository(_Repository):
    """Preferencje użytkownika (sterują zachowaniem, nie opisują świata)."""

    def set(self, key: str, value: str, *, expires_at: Any = None) -> Preference:
        record = Preference(key=key, value=value)
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT id FROM preferences WHERE key = ?", (record.key,)
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    "INSERT INTO preferences (key, value, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        record.key,
                        record.value,
                        to_iso(record.created_at),
                        to_iso(record.updated_at),
                    ),
                )
                record.id = int(cursor.lastrowid or 0)
            else:
                record.id = int(row["id"])
                connection.execute(
                    "UPDATE preferences SET value = ?, updated_at = ? WHERE id = ?",
                    (record.value, to_iso(record.updated_at), record.id),
                )
        self._db.memories.record(
            kind=MEMORY_KIND_PREFERENCE,
            source_table="preferences",
            source_id=record.id or 0,
            summary=record.as_line(),
            importance=0.8,
            expires_at=expires_at,
        )
        return record

    def get(self, key: str) -> Preference | None:
        row = self._db.query_one("SELECT * FROM preferences WHERE key = ?", (key.strip().lower(),))
        return Preference.from_row(row) if row else None

    def get_by_id(self, preference_id: int) -> Preference | None:
        row = self._db.query_one("SELECT * FROM preferences WHERE id = ?", (preference_id,))
        return Preference.from_row(row) if row else None

    def value_of(self, key: str, default: str = "") -> str:
        found = self.get(key)
        return found.value if found is not None else default

    def all(self, *, limit: int = 100) -> list[Preference]:
        rows = self._db.query(
            "SELECT * FROM preferences ORDER BY key LIMIT ?", (max(1, limit),)
        )
        return [Preference.from_row(row) for row in rows]

    def delete(self, key: str) -> bool:
        existing = self.get(key)
        if existing is None:
            return False
        with self._db.transaction() as connection:
            connection.execute("DELETE FROM preferences WHERE id = ?", (existing.id,))
            connection.execute(
                "DELETE FROM memories WHERE source_table = 'preferences' AND source_id = ?",
                (existing.id,),
            )
        return True


class NoteRepository(_Repository):
    """Notatki. Treść w bazie, żeby dało się je przeszukiwać razem z rozmowami."""

    def add(
        self,
        body: str,
        *,
        title: str = "",
        tags: Sequence[str] = (),
        source: str = "user",
        expires_at: Any = None,
    ) -> Note:
        record = Note(title=title.strip()[:200], body=body, tags=list(tags), source=source)
        cursor = self._db.execute(
            "INSERT INTO notes (title, body, tags, created_at, updated_at, source)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.title,
                record.body,
                ",".join(record.tags),
                to_iso(record.created_at),
                to_iso(record.updated_at),
                record.source,
            ),
        )
        record.id = int(cursor.lastrowid or 0)
        self._db.memories.record(
            kind=MEMORY_KIND_NOTE,
            source_table="notes",
            source_id=record.id,
            summary=record.title or record.preview,
            importance=0.7,
            expires_at=expires_at,
        )
        return record

    def get(self, note_id: int) -> Note | None:
        row = self._db.query_one("SELECT * FROM notes WHERE id = ?", (note_id,))
        return Note.from_row(row) if row else None

    def append(self, note_id: int, text: str) -> Note | None:
        """Dopisz akapit do istniejącej notatki. ``None`` = nie ma takiej notatki.

        Dopisanie, a nie nadpisanie: identyfikator zostaje, więc wektor z Fazy 6 i
        metadane wspomnienia nadal wskazują tę samą notatkę (trzeba tylko przeliczyć
        wektor — robi to ``brain/memory.py``).
        """
        addition = str(text or "").strip()
        if not addition:
            return self.get(note_id)
        existing = self.get(note_id)
        if existing is None:
            return None
        body = f"{existing.body}\n\n{addition}"
        self._db.execute(
            "UPDATE notes SET body = ?, updated_at = ? WHERE id = ?",
            (body, to_iso(utc_now()), note_id),
        )
        return self.get(note_id)

    def recent(self, *, limit: int = 20) -> list[Note]:
        rows = self._db.query(
            "SELECT * FROM notes ORDER BY updated_at DESC, id DESC LIMIT ?", (max(1, limit),)
        )
        return [Note.from_row(row) for row in rows]

    def search(self, text: str, *, limit: int = 10) -> list[SearchHit]:
        query = text.strip()
        if not query:
            return []
        limit = max(1, limit)

        if self._db.has_fulltext:
            match = _fts_query(query)
            if match:
                rows = self._db.query(
                    "SELECT n.id, n.title, n.body, n.created_at FROM notes_fts f"
                    " JOIN notes n ON n.id = f.rowid"
                    " WHERE notes_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, limit),
                )
                return [self._to_hit(row) for row in rows]

        pattern = f"%{_escape_like(query)}%"
        rows = self._db.query(
            "SELECT id, title, body, created_at FROM notes"
            " WHERE body LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\'"
            " ORDER BY id DESC LIMIT ?",
            (pattern, pattern, limit),
        )
        return [self._to_hit(row) for row in rows]

    def delete(self, note_id: int) -> bool:
        with self._db.transaction() as connection:
            cursor = connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            connection.execute(
                "DELETE FROM memories WHERE source_table = 'notes' AND source_id = ?", (note_id,)
            )
        return bool(cursor.rowcount)

    @staticmethod
    def _to_hit(row: Any) -> SearchHit:
        return SearchHit(
            kind="note",
            source_id=int(row["id"]),
            text=str(row["body"]),
            created_at=from_iso(row["created_at"]),
            context=str(row["title"] or ""),
        )


class EmbeddingRepository(_Repository):
    """Wektory wspomnień (Faza 6). Sama arytmetyka podobieństwa jest w ``brain/``.

    Repozytorium świadomie NIE liczy podobieństwa w SQL-u: SQLite go nie umie
    bez rozszerzenia (``sqlite-vec``), którego nie ma na każdej maszynie. Baza
    przechowuje i oddaje wektory, a szuka w nich indeks z ``brain/vectorstore.py``.
    """

    def set(
        self,
        *,
        source_table: str,
        source_id: int,
        text: str,
        model: str,
        vector: Sequence[float],
    ) -> EmbeddingRecord:
        """Zapisz (albo nadpisz) wektor dla danego źródła i modelu."""
        record = EmbeddingRecord(
            source_table=source_table,
            source_id=source_id,
            text=text,
            model=model,
            dim=len(vector),
            vector=list(vector),
        )
        blob = encode_vector(record.vector)
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT id FROM embeddings WHERE source_table = ? AND source_id = ? AND model = ?",
                (source_table, source_id, model),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    "INSERT INTO embeddings (source_table, source_id, text, model, dim, vector,"
                    " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        source_table,
                        source_id,
                        record.text,
                        model,
                        record.dim,
                        blob,
                        to_iso(record.created_at),
                    ),
                )
                record.id = int(cursor.lastrowid or 0)
            else:
                record.id = int(row["id"])
                connection.execute(
                    "UPDATE embeddings SET text = ?, dim = ?, vector = ?, created_at = ?"
                    " WHERE id = ?",
                    (record.text, record.dim, blob, to_iso(record.created_at), record.id),
                )
        return record

    def for_model(self, model: str, *, dim: int | None = None) -> list[EmbeddingRecord]:
        """Wszystkie wektory policzone TYM modelem (i tylko nim).

        Wektory z innych modeli leżą w innych przestrzeniach — mieszanie ich
        dałoby wyniki wyglądające sensownie i będące bez sensu.
        """
        if dim is None:
            rows = self._db.query("SELECT * FROM embeddings WHERE model = ? ORDER BY id", (model,))
        else:
            rows = self._db.query(
                "SELECT * FROM embeddings WHERE model = ? AND dim = ? ORDER BY id", (model, dim)
            )
        records = [EmbeddingRecord.from_row(row) for row in rows]
        return [record for record in records if record.vector]

    def get(self, source_table: str, source_id: int, model: str) -> EmbeddingRecord | None:
        row = self._db.query_one(
            "SELECT * FROM embeddings WHERE source_table = ? AND source_id = ? AND model = ?",
            (source_table, source_id, model),
        )
        return EmbeddingRecord.from_row(row) if row else None

    def has(self, source_table: str, source_id: int, model: str) -> bool:
        row = self._db.query_one(
            "SELECT 1 FROM embeddings WHERE source_table = ? AND source_id = ? AND model = ?",
            (source_table, source_id, model),
        )
        return row is not None

    def delete(self, source_table: str, source_id: int, *, model: str | None = None) -> int:
        if model is None:
            cursor = self._db.execute(
                "DELETE FROM embeddings WHERE source_table = ? AND source_id = ?",
                (source_table, source_id),
            )
        else:
            cursor = self._db.execute(
                "DELETE FROM embeddings WHERE source_table = ? AND source_id = ? AND model = ?",
                (source_table, source_id, model),
            )
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    def clear(self, *, model: str | None = None) -> int:
        """Wyczyść indeks (np. przed reindeksacją po zmianie modelu)."""
        if model is None:
            cursor = self._db.execute("DELETE FROM embeddings")
        else:
            cursor = self._db.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    def count(self, *, model: str | None = None) -> int:
        if model is None:
            row = self._db.query_one("SELECT COUNT(*) AS liczba FROM embeddings")
        else:
            row = self._db.query_one(
                "SELECT COUNT(*) AS liczba FROM embeddings WHERE model = ?", (model,)
            )
        return int(row["liczba"]) if row else 0

    def models(self) -> list[tuple[str, int, int]]:
        """Jakie modele siedzą w indeksie: ``[(nazwa, wymiar, ile wektorów)]``."""
        rows = self._db.query(
            "SELECT model, dim, COUNT(*) AS liczba FROM embeddings GROUP BY model, dim"
            " ORDER BY liczba DESC"
        )
        return [(str(row["model"]), int(row["dim"]), int(row["liczba"])) for row in rows]

    def prune_orphans(self) -> int:
        """Usuń wektory wskazujące na nieistniejące już rekordy.

        Klucz obcy nie wchodzi w grę: ``source_table`` bywa różny, a SQLite nie
        zna kluczy obcych „polimorficznych". Sprzątamy więc jawnie — po
        skasowaniu rozmowy jej wiadomości znikają kaskadą, a wektory zostają.
        """
        removed = 0
        for table in ("messages", "facts", "notes", "summaries", "preferences"):
            cursor = self._db.execute(
                f"DELETE FROM embeddings WHERE source_table = '{table}'"  # nazwa z kodu
                f" AND source_id NOT IN (SELECT id FROM {table})"
            )
            if cursor.rowcount and cursor.rowcount > 0:
                removed += cursor.rowcount
        return removed


class ToolAuditRepository(_Repository):
    """Log wywołań narzędzi (Faza 7) — **tylko do dopisywania**.

    Świadomie NIE ma tu metody usuwającej ani aktualizującej. Log audytu, który
    da się wyczyścić z tego samego procesu, co wykonuje narzędzia, nie jest logiem
    audytu. Historia rozmów kasuje się osobno i nie rusza tej tabeli.
    """

    def add(
        self,
        *,
        tool: str,
        risk: str,
        decision: str,
        ok: bool = False,
        confirmed: bool = False,
        arguments_hash: str = "",
        duration_ms: int = 0,
        detail: str = "",
        conversation_id: int | None = None,
    ) -> ToolAuditRecord:
        record = ToolAuditRecord(
            tool=tool,
            risk=risk,
            decision=decision,
            ok=ok,
            confirmed=confirmed,
            arguments_hash=arguments_hash,
            duration_ms=int(duration_ms),
            detail=detail[:500],
            conversation_id=conversation_id,
        )
        cursor = self._db.execute(
            "INSERT INTO tool_audit (created_at, conversation_id, tool, risk, decision, ok,"
            " confirmed, arguments_hash, duration_ms, detail)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                to_iso(record.created_at),
                record.conversation_id,
                record.tool,
                record.risk,
                record.decision,
                int(record.ok),
                int(record.confirmed),
                record.arguments_hash,
                record.duration_ms,
                record.detail,
            ),
        )
        record.id = int(cursor.lastrowid or 0)
        return record

    def recent(self, *, limit: int = 20, tool: str | None = None) -> list[ToolAuditRecord]:
        if tool:
            rows = self._db.query(
                "SELECT * FROM tool_audit WHERE tool = ? ORDER BY id DESC LIMIT ?",
                (tool, max(1, limit)),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM tool_audit ORDER BY id DESC LIMIT ?", (max(1, limit),)
            )
        return [ToolAuditRecord.from_row(row) for row in rows]

    def count(self, *, tool: str | None = None) -> int:
        if tool:
            row = self._db.query_one(
                "SELECT COUNT(*) AS liczba FROM tool_audit WHERE tool = ?", (tool,)
            )
        else:
            row = self._db.query_one("SELECT COUNT(*) AS liczba FROM tool_audit")
        return int(row["liczba"]) if row else 0


class MemoryRepository(_Repository):
    """Metadane wspomnień: waga, użycie, wygasanie. Treść żyje w swojej tabeli."""

    def record(
        self,
        *,
        kind: str,
        source_table: str,
        source_id: int,
        summary: str = "",
        importance: float = 0.5,
        expires_at: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            kind=kind,
            source_table=source_table,
            source_id=source_id,
            summary=summary[:500],
            importance=importance,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT id, use_count FROM memories WHERE source_table = ? AND source_id = ?",
                (source_table, source_id),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    "INSERT INTO memories (kind, source_table, source_id, summary, importance,"
                    " created_at, last_used_at, use_count, expires_at, metadata)"
                    " VALUES (?, ?, ?, ?, ?, ?, NULL, 0, ?, ?)",
                    (
                        entry.kind,
                        entry.source_table,
                        entry.source_id,
                        entry.summary,
                        entry.importance,
                        to_iso(entry.created_at),
                        to_iso(entry.expires_at),
                        dump_metadata(entry.metadata),
                    ),
                )
                entry.id = int(cursor.lastrowid or 0)
            else:
                entry.id = int(row["id"])
                entry.use_count = int(row["use_count"])
                connection.execute(
                    "UPDATE memories SET kind = ?, summary = ?, importance = ?, expires_at = ?,"
                    " metadata = ? WHERE id = ?",
                    (
                        entry.kind,
                        entry.summary,
                        entry.importance,
                        to_iso(entry.expires_at),
                        dump_metadata(entry.metadata),
                        entry.id,
                    ),
                )
        return entry

    def touch(self, entry_id: int) -> None:
        """Odnotuj, że wspomnienie się przydało (podstawa przyszłego rankingu)."""
        self._db.execute(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
            (to_iso(utc_now()), entry_id),
        )

    def top(self, *, kind: str | None = None, limit: int = 20) -> list[MemoryEntry]:
        if kind is None:
            rows = self._db.query(
                "SELECT * FROM memories ORDER BY importance DESC, use_count DESC LIMIT ?",
                (max(1, limit),),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM memories WHERE kind = ?"
                " ORDER BY importance DESC, use_count DESC LIMIT ?",
                (kind, max(1, limit)),
            )
        return [MemoryEntry.from_row(row) for row in rows]

    def expired(self) -> list[MemoryEntry]:
        """Wspomnienia po terminie ważności (informacje uznane za chwilowe)."""
        rows = self._db.query(
            "SELECT * FROM memories WHERE expires_at IS NOT NULL AND expires_at < ? ORDER BY id",
            (to_iso(utc_now()),),
        )
        return [MemoryEntry.from_row(row) for row in rows]

    def purge_expired(self, *, delete_sources: bool = False) -> int:
        """Usuń przeterminowane wspomnienia. Zwraca liczbę usuniętych wpisów.

        ``delete_sources`` usuwa też treść, do której prowadziły — ale wyłącznie
        z tabel, które są „pamięcią" (fakty, preferencje, notatki). Historia
        rozmowy nie wygasa: wpis w ``memories`` może zniknąć, wiadomość zostaje.
        """
        entries = self.expired() if delete_sources else []
        with self._db.transaction() as connection:
            for entry in entries:
                if entry.source_table not in ("facts", "preferences", "notes"):
                    continue
                connection.execute(
                    f"DELETE FROM {entry.source_table} WHERE id = ?",  # nazwa z białej listy
                    (entry.source_id,),
                )
                connection.execute(
                    "DELETE FROM embeddings WHERE source_table = ? AND source_id = ?",
                    (entry.source_table, entry.source_id),
                )
            cursor = connection.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (to_iso(utc_now()),),
            )
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    def count(self, *, kind: str | None = None) -> int:
        if kind is None:
            row = self._db.query_one("SELECT COUNT(*) AS liczba FROM memories")
        else:
            row = self._db.query_one(
                "SELECT COUNT(*) AS liczba FROM memories WHERE kind = ?", (kind,)
            )
        return int(row["liczba"]) if row else 0


# --------------------------------------------------------------------------- #
# Sprawdzenie środowiska (dopisane do mechanizmu z Fazy 1)
# --------------------------------------------------------------------------- #


@register_dependency_check
def check_memory(context: DependencyContext) -> list[DependencyCheck]:
    """Czy pamięć długoterminowa ma gdzie mieszkać i co potrafi ta biblioteka SQLite.

    Sprawdzenie świadomie NIE otwiera docelowej bazy: ``--check-deps`` ma być
    diagnostyką, a nie zakładać plików. Sprawdzamy prawo zapisu do katalogu i
    możliwości SQLite na bazie tymczasowej w pamięci.
    """
    settings = context.settings
    if not settings.memory_enabled:
        return [
            DependencyCheck(
                name="Pamięć długoterminowa",
                category="storage",
                required=False,
                ok=False,
                detail="wyłączona ustawieniem MEMORY_ENABLED=false",
                hint="ustaw MEMORY_ENABLED=true, aby asystent pamiętał między uruchomieniami",
                phase=5,
            )
        ]

    target = database_file(settings)
    checks: list[DependencyCheck] = []

    if str(target) == MEMORY_DATABASE:
        checks.append(
            DependencyCheck(
                name="Pamięć długoterminowa",
                category="storage",
                required=False,
                ok=True,
                detail=(
                    "baza wyłącznie w pamięci procesu (DATABASE_PATH=:memory:) —"
                    " nic nie przetrwa zamknięcia programu"
                ),
                phase=5,
            )
        )
    else:
        exists = target.exists()
        size = target.stat().st_size if exists else 0
        writable = True
        problem = ""
        probe = target.parent / f".write-test-{os.getpid()}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            writable = False
            problem = str(exc)
        checks.append(
            DependencyCheck(
                name="Pamięć długoterminowa",
                category="storage",
                required=False,
                ok=writable,
                detail=(
                    (f"baza istnieje ({size // 1024} KiB)" if exists else "baza zostanie założona")
                    if writable
                    else f"brak prawa zapisu: {problem}"
                ),
                path=str(target),
                hint=(
                    ""
                    if writable
                    else "wskaż inne miejsce zmienną MIKU_DATA_DIR albo DATABASE_PATH w .env"
                ),
                phase=5,
            )
        )

    # Możliwości samej biblioteki SQLite — różne na różnych systemach.
    probe_connection = sqlite3.connect(MEMORY_DATABASE)
    try:
        from database.migrations import supports_fts5

        fts = supports_fts5(probe_connection)
    finally:
        probe_connection.close()
    checks.append(
        DependencyCheck(
            name="SQLite (wyszukiwanie pełnotekstowe)",
            category="storage",
            required=False,
            ok=fts,
            detail=(
                f"biblioteka {sqlite3.sqlite_version}, moduł FTS5 dostępny"
                if fts
                else f"biblioteka {sqlite3.sqlite_version} bez FTS5 — "
                "szukanie w pamięci zejdzie na LIKE (wolniej, ale działa)"
            ),
            phase=5,
        )
    )
    return checks


__all__ = [
    "ConversationRepository",
    "Database",
    "DatabaseError",
    "EmbeddingRepository",
    "FactRepository",
    "MemoryRepository",
    "MessageRepository",
    "NoteRepository",
    "PreferenceRepository",
    "SummaryRepository",
    "check_memory",
]
