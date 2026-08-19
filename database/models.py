"""Rekordy pamięci długoterminowej (Faza 5).

Modele opisują WYŁĄCZNIE kształt danych — nie znają SQL-a, nie otwierają
połączeń i nie wiedzą, gdzie leży plik bazy. Zamiana SQLite na cokolwiek innego
nie wymaga tknięcia tego pliku.

Konwencje wspólne dla całej warstwy:

* czas jest zawsze w UTC i zapisywany jako tekst ISO-8601 (``2026-08-15T10:00:00+00:00``);
  czasu lokalnego nie zapisujemy nigdzie, bo baza bywa przenoszona między
  maszynami i strefami czasowymi,
* ``metadata`` jedzie do bazy jako JSON w kolumnie ``TEXT`` — jedno miejsce na
  drobiazgi (język, model, latencje), bez migracji przy każdym nowym polu,
* identyfikatory nadaje baza (``INTEGER PRIMARY KEY``); rekord przed zapisem ma
  ``id = None``.

Uwaga na nazwy (patrz ARCHITECTURE.md §3.2): ``models/`` w katalogu projektu to
WAGI MODELI (dane), a ten moduł to REKORDY BAZY. To dwie różne rzeczy.
"""

from __future__ import annotations

import json
import logging
import struct
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from i18n import t

logger = logging.getLogger(__name__)

# Role wiadomości. Świadomie powtórzone zamiast importu z ``brain.conversation``:
# warstwa bazy leży NIŻEJ niż „mózg" i nie może od niego zależeć (inaczej
# powstaje cykl importów przy pierwszym module, który potrzebuje obu).
MessageRole = Literal["system", "user", "assistant", "tool"]

# Skąd wziął się fakt/preferencja. „user" = powiedział to wprost użytkownik,
# „inferred" = wywnioskował model, „system" = zapisał program.
FactSource = Literal["user", "inferred", "system", "import"]

# Rodzaj wspomnienia w tabeli metadanych. Lista jest otwarta z rozmysłem —
# kolejne fazy (embeddingi, notatki z plików) dopisują własne rodzaje.
MEMORY_KIND_FACT: Final[str] = "fact"
MEMORY_KIND_PREFERENCE: Final[str] = "preference"
MEMORY_KIND_NOTE: Final[str] = "note"
MEMORY_KIND_SUMMARY: Final[str] = "summary"
MEMORY_KIND_MESSAGE: Final[str] = "message"


def utc_now() -> datetime:
    """Bieżąca chwila w UTC (świadoma strefy)."""
    return datetime.now(timezone.utc)


def to_iso(moment: datetime | None) -> str | None:
    """Zamień chwilę na tekst ISO-8601 w UTC. ``None`` przechodzi bez zmian."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        # Czas bez strefy traktujemy jako UTC — nie zgadujemy strefy maszyny,
        # bo ten sam plik bazy bywa otwierany na komputerze w innej strefie.
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def from_iso(value: Any) -> datetime | None:
    """Odczytaj chwilę zapisaną przez :func:`to_iso`; śmieci dają ``None``."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        logger.debug("Nie rozpoznano znacznika czasu %r — pomijam.", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def dump_metadata(metadata: Mapping[str, Any] | None) -> str:
    """Zserializuj metadane do JSON-a. Wartości niezdatne do JSON-a lecą jako tekst."""
    if not metadata:
        return "{}"
    try:
        return json.dumps(dict(metadata), ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - egzotyczne obiekty
        return "{}"


def load_metadata(value: Any) -> dict[str, Any]:
    """Odczytaj metadane; uszkodzony JSON daje pusty słownik, nie wyjątek."""
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.debug("Uszkodzone metadane w bazie: %r", value)
        return {}
    return parsed if isinstance(parsed, dict) else {}


class Record(BaseModel):
    """Wspólna baza rekordów: identyfikator nadany przez bazę i konwersja z wiersza."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None

    @classmethod
    def from_row(cls, row: Any) -> Any:
        """Zbuduj rekord z wiersza ``sqlite3.Row`` (albo dowolnego mapowania).

        Typ argumentu jest celowo luźny: ``sqlite3.Row`` zachowuje się jak
        mapowanie, ale formalnie nim nie jest, a warstwa rekordów nie ma powodu
        wiedzieć, że dane przyszły akurat ze sqlite3.
        """
        return cls(**cls._convert_row(dict(row)))

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        return data


class Conversation(Record):
    """Jedna sesja rozmowy — od uruchomienia asystenta do ``/nowa`` albo wyjścia."""

    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    title: str = ""
    # Skąd przyszła rozmowa: „terminal", „voice", „gui"... Zwykły tekst, bo
    # kolejne fazy dokładają własne źródła bez migracji schematu.
    source: str = "terminal"
    model: str = ""
    message_count: int = 0

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["started_at"] = from_iso(data.get("started_at")) or utc_now()
        data["ended_at"] = from_iso(data.get("ended_at"))
        return data

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


class StoredMessage(Record):
    """Wiadomość zapisana w bazie (odpowiednik ``brain.conversation.Message``)."""

    conversation_id: int
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        data["metadata"] = load_metadata(data.get("metadata"))
        data["language"] = data.get("language") or ""
        return data


class Summary(Record):
    """Streszczenie fragmentu rozmowy — to ono zastępuje obcięte wiadomości."""

    conversation_id: int
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    # Ile wiadomości zostało w nim zwiniętych i do której z nich sięga.
    message_count: int = 0
    covers_to_message_id: int | None = None
    # Streszczenie streszczenia: kolejne kompaktowanie bierze poprzednie za punkt
    # wyjścia, więc warto wiedzieć, które to pokolenie.
    generation: int = 1
    # „llm" = streszczał model, „fallback" = mechaniczny skrót (model milczał).
    method: str = "llm"

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        return data


class Fact(Record):
    """Trwały fakt o użytkowniku: ``imie = Mariusz``, ``miasto = Wrocław``."""

    key: str
    value: str
    source: FactSource = "user"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    # Fakt przypięty przeżywa czyszczenie starych danych (retention_days).
    pinned: bool = True

    @field_validator("key")
    @classmethod
    def _normalize_key(cls, value: str) -> str:
        key = value.strip().lower()
        if not key:
            raise ValueError(t("rec.fact_key_empty"))
        return key[:120]

    @field_validator("value")
    @classmethod
    def _clean_value(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(t("rec.fact_value_empty"))
        return cleaned

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        data["updated_at"] = from_iso(data.get("updated_at")) or utc_now()
        data["pinned"] = bool(data.get("pinned", True))
        return data

    def as_line(self) -> str:
        return f"{self.key}: {self.value}"


class Preference(Record):
    """Preferencja użytkownika: ``jezyk_odpowiedzi = polski``, ``dlugosc = krotko``.

    Osobno od :class:`Fact`, bo ma inny cykl życia: preferencje sterują
    zachowaniem asystenta i są nadpisywane, fakty tylko przyrastają.
    """

    key: str
    value: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("key")
    @classmethod
    def _normalize_key(cls, value: str) -> str:
        key = value.strip().lower()
        if not key:
            raise ValueError(t("rec.pref_key_empty"))
        return key[:120]

    @field_validator("value")
    @classmethod
    def _clean_value(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(t("rec.pref_value_empty"))
        return cleaned

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        data["updated_at"] = from_iso(data.get("updated_at")) or utc_now()
        return data

    def as_line(self) -> str:
        return f"{self.key}: {self.value}"


class Note(Record):
    """Notatka użytkownika. Treść w bazie, bo ma być wyszukiwalna razem z rozmowami."""

    title: str = ""
    body: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    # Skąd notatka: „user", „assistant", „import".
    source: str = "user"

    @field_validator("body")
    @classmethod
    def _clean_body(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(t("rec.note_body_empty"))
        return cleaned

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for tag in value:
            normalized = str(tag).strip().lower()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized[:40])
        return cleaned

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        data["updated_at"] = from_iso(data.get("updated_at")) or utc_now()
        raw_tags = data.get("tags")
        if isinstance(raw_tags, str):
            data["tags"] = [tag for tag in raw_tags.split(",") if tag]
        return data

    @property
    def preview(self) -> str:
        single_line = " ".join(self.body.split())
        return single_line if len(single_line) <= 80 else single_line[:77] + "..."


class MemoryEntry(Record):
    """Metadane wspomnienia — warstwa NAD treścią, wspólna dla wszystkich rodzajów.

    Treść mieszka w swojej tabeli (``facts``, ``notes``, ``summaries``...), a tutaj
    trzyma się to, co potrzebne do przypominania i zapominania: waga, kiedy
    ostatnio się przydało, ile razy, kiedy wygasa. Dzięki temu Faza z
    embeddingami dopisuje wektory do JEDNEJ tabeli, zamiast do każdej z osobna.
    """

    kind: str
    source_table: str
    source_id: int
    summary: str = ""
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None
    use_count: int = 0
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        data["last_used_at"] = from_iso(data.get("last_used_at"))
        data["expires_at"] = from_iso(data.get("expires_at"))
        data["metadata"] = load_metadata(data.get("metadata"))
        return data

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or utc_now()) >= self.expires_at


class EmbeddingRecord(Record):
    """Wektor jednego fragmentu pamięci (Faza 6).

    ``model`` i ``dim`` jadą razem z wektorem, bo wektory policzone różnymi
    modelami leżą w różnych przestrzeniach i porównywanie ich nie ma sensu.
    Zmiana modelu = nowe wiersze i reindeksacja, nigdy ciche mieszanie.
    """

    source_table: str
    source_id: int
    text: str
    model: str
    dim: int
    vector: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        raw = data.get("vector")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            data["vector"] = decode_vector(raw)
        return data

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_table, self.source_id)


class ToolAuditRecord(Record):
    """Jeden wpis w logu wywołań narzędzi (Faza 7).

    Zapisujemy SKRÓT argumentów, nie ich treść: do odpowiedzi na pytanie „co się
    stało i czy użytkownik to potwierdził" skrót wystarcza, a prywatne dane nie
    lądują w drugim miejscu. Wpisy tylko powstają — nie ma metody, która je
    modyfikuje albo usuwa.
    """

    created_at: datetime = Field(default_factory=utc_now)
    conversation_id: int | None = None
    tool: str
    risk: str
    # allowed | denied | confirmed | user_denied | expired | dry_run | error
    decision: str
    ok: bool = False
    confirmed: bool = False
    arguments_hash: str = ""
    duration_ms: int = 0
    detail: str = ""

    @classmethod
    def _convert_row(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["created_at"] = from_iso(data.get("created_at")) or utc_now()
        data["ok"] = bool(data.get("ok"))
        data["confirmed"] = bool(data.get("confirmed"))
        return data

    def as_line(self) -> str:
        state = "ok" if self.ok else "błąd"
        confirmed = ", potwierdzone" if self.confirmed else ""
        return (
            f"{self.created_at.strftime('%Y-%m-%d %H:%M')}  {self.tool}  "
            f"[{self.risk}] {self.decision}{confirmed} → {state} ({self.duration_ms} ms)"
        )


class SemanticHit(BaseModel):
    """Wspomnienie odnalezione „po znaczeniu" wraz z miarą podobieństwa."""

    model_config = ConfigDict(frozen=True)

    source_table: str
    source_id: int
    text: str
    score: float
    created_at: datetime | None = None

    @property
    def preview(self) -> str:
        single_line = " ".join(self.text.split())
        return single_line if len(single_line) <= 120 else single_line[:117] + "..."


def encode_vector(values: Sequence[float]) -> bytes:
    """Zapisz wektor jako surowe ``float32`` **little-endian**.

    Kolejność bajtów jest wymuszona (``<``), a nie „taka jak na tej maszynie":
    plik bazy bywa przenoszony, a na maszynie big-endian (s390x, część MIPS-ów)
    natywny zapis dałby po odczycie kompletne śmieci zamiast wektora.
    """
    count = len(values)
    return struct.pack(f"<{count}f", *(float(item) for item in values))


def decode_vector(blob: bytes | bytearray | memoryview) -> list[float]:
    """Odczytaj wektor zapisany przez :func:`encode_vector`.

    Uszkodzony wpis (ucięty blob) daje pustą listę zamiast wyjątku — pojedynczy
    wadliwy wektor ma wypaść z indeksu, a nie zatrzymać asystenta.
    """
    raw = bytes(blob)
    count, remainder = divmod(len(raw), 4)
    if remainder or count == 0:
        logger.warning("Pominięto uszkodzony wektor w bazie (%s bajtów).", len(raw))
        return []
    return list(struct.unpack(f"<{count}f", raw))


class SearchHit(BaseModel):
    """Wynik wyszukiwania w pamięci — wspólny kształt dla wiadomości i notatek."""

    model_config = ConfigDict(frozen=True)

    kind: str
    source_id: int
    text: str
    created_at: datetime | None = None
    context: str = ""

    @property
    def preview(self) -> str:
        single_line = " ".join(self.text.split())
        return single_line if len(single_line) <= 100 else single_line[:97] + "..."


class MemoryStats(BaseModel):
    """Licznik zawartości bazy — do ``/status`` i do raportu zależności."""

    model_config = ConfigDict(frozen=True)

    conversations: int = 0
    messages: int = 0
    summaries: int = 0
    facts: int = 0
    preferences: int = 0
    notes: int = 0
    memories: int = 0
    embeddings: int = 0
    tool_audit: int = 0

    def describe(self) -> str:
        return t(
            "status.stats.describe",
            conversations=self.conversations,
            messages=self.messages,
            summaries=self.summaries,
            facts=self.facts,
            preferences=self.preferences,
            notes=self.notes,
            embeddings=self.embeddings,
            audit=self.tool_audit,
        )


__all__ = [
    "MEMORY_KIND_FACT",
    "MEMORY_KIND_MESSAGE",
    "MEMORY_KIND_NOTE",
    "MEMORY_KIND_PREFERENCE",
    "MEMORY_KIND_SUMMARY",
    "Conversation",
    "EmbeddingRecord",
    "Fact",
    "FactSource",
    "MemoryEntry",
    "MemoryStats",
    "MessageRole",
    "Note",
    "Preference",
    "Record",
    "SearchHit",
    "SemanticHit",
    "StoredMessage",
    "Summary",
    "ToolAuditRecord",
    "decode_vector",
    "dump_metadata",
    "encode_vector",
    "from_iso",
    "load_metadata",
    "to_iso",
    "utc_now",
]
