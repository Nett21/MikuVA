"""Pamięć długoterminowa asystenta (Faza 5) — SQLite.

Import pakietu rejestruje sprawdzenia środowiska w mechanizmie z Fazy 1
(``python main.py --check-deps``) i wystawia wszystko, czego potrzebuje
``brain/memory.py``::

    from database import Database

    with Database.open(settings) as db:
        conversation = db.conversations.start(source="terminal")
        db.messages.add(conversation.id, "user", "Cześć!")
        db.facts.set("imie", "Mariusz")

Ten pakiet nie wie, gdzie leży plik bazy — pyta o to ``config.database_file()``,
bo katalog danych jest inny na każdym systemie.
"""

from __future__ import annotations

from database.database import (
    ConversationRepository,
    Database,
    DatabaseError,
    FactRepository,
    MemoryRepository,
    MessageRepository,
    NoteRepository,
    PreferenceRepository,
    SummaryRepository,
    check_memory,
)
from database.migrations import (
    MIGRATIONS,
    SCHEMA_VERSION,
    Migration,
    apply_migrations,
    current_version,
    ensure_fulltext_index,
    supports_fts5,
)
from database.models import (
    Conversation,
    Fact,
    MemoryEntry,
    MemoryStats,
    Note,
    Preference,
    SearchHit,
    StoredMessage,
    Summary,
)

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Conversation",
    "ConversationRepository",
    "Database",
    "DatabaseError",
    "Fact",
    "FactRepository",
    "MemoryEntry",
    "MemoryRepository",
    "MemoryStats",
    "MessageRepository",
    "Migration",
    "Note",
    "NoteRepository",
    "Preference",
    "PreferenceRepository",
    "SearchHit",
    "StoredMessage",
    "Summary",
    "SummaryRepository",
    "apply_migrations",
    "check_memory",
    "current_version",
    "ensure_fulltext_index",
    "supports_fts5",
]
