"""Indeks wektorowy i wyszukiwanie wspomnień „po znaczeniu" (Faza 6).

Dlaczego FAISS, a nie ChromaDB
------------------------------
Wybrany został **FAISS** (pakiet ``faiss-cpu``), i to jako element *opcjonalny*,
z wariantem zapasowym liczonym w NumPy. Powody, po kolei:

1. **Jedno źródło prawdy.** Wektory mieszkają w tabeli ``embeddings`` w SQLite
   (Faza 5) — razem z tekstem, z którego powstały. FAISS jest tylko strukturą
   wyszukiwania budowaną z tej tabeli przy starcie. ChromaDB przyniosłoby
   *drugą bazę* z własnym plikiem, własnym schematem i własnymi migracjami, a
   wtedy „zapomnij o X" trzeba by wykonać w dwóch miejscach i pilnować, żeby
   nie rozjechały się po awarii. Skasowanie rozmowy z SQLite ma usuwać jej
   ślady — nie zostawiać ich w drugim magazynie.
2. **Ciężar zależności.** ChromaDB ciągnie za sobą duży zestaw pakietów
   (m.in. własny stos serwera i ``onnxruntime`` na domyślne embeddingi),
   przypina wersje ``pydantic`` i wymaga nowszego SQLite niż ten, który bywa w
   dystrybucji — to dokładnie ten rodzaj kłopotu, który psuje instalację na
   cudzej maszynie. ``faiss-cpu`` to jedna biblioteka bez usług w tle.
3. **Skala jest mała.** Osobisty asystent to rząd 10⁴–10⁵ wektorów. Przy takiej
   liczbie przegląd zupełny w NumPy zajmuje pojedyncze milisekundy, więc brak
   FAISS-a niczego nie psuje — daje tylko wolniejsze wyszukiwanie. Dlatego FAISS
   jest przyspieszeniem, a nie warunkiem działania; nie ma go w
   ``requirements.txt``, bo koła ``faiss-cpu`` nie istnieją dla każdej wersji
   Pythona i każdej platformy, a nieudana instalacja jednego pakietu nie może
   zatrzymać instalacji całej reszty (tak samo traktujemy tu Pipera i webrtcvad).
4. **Plik indeksu jest niewymienny między maszynami.** Zapis indeksu FAISS jest
   związany z wersją biblioteki i architekturą — dlatego świadomie NIE zapisujemy
   go na dysk. Indeks powstaje w pamięci z bazy przy każdym uruchomieniu (dla
   10⁵ wektorów to ułamek sekundy), a jedynym trwałym artefaktem zostaje SQLite,
   który przenosi się wszędzie.

Interfejs :class:`VectorIndex` pozwala podmienić implementację (hnswlib,
``sqlite-vec``, cokolwiek) bez ruszania warstwy pamięci.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from brain.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    create_embedding_provider,
    describe_provider,
)
from config import Settings, get_settings
from i18n import t

if TYPE_CHECKING:  # importy tylko dla typów — moduł działa bez warstwy bazy
    from database.database import Database

logger = logging.getLogger(__name__)

# Tabele, których treść trafia do pamięci semantycznej, wraz z zapytaniem
# zwracającym (id, tekst). Kolejność ma znaczenie tylko dla reindeksacji.
INDEXED_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("facts", "SELECT id, key || ': ' || value AS text FROM facts ORDER BY id"),
    ("preferences", "SELECT id, key || ': ' || value AS text FROM preferences ORDER BY id"),
    ("notes", "SELECT id, CASE WHEN title = '' THEN body ELSE title || ': ' || body END"
              " AS text FROM notes ORDER BY id"),
    ("summaries", "SELECT id, content AS text FROM summaries ORDER BY id"),
    ("messages", "SELECT id, content AS text FROM messages WHERE role = 'user' ORDER BY id"),
)


def _numpy() -> Any | None:
    """NumPy albo ``None``. Bez niego pamięć semantyczna jest wyłączona.

    NumPy jest w projekcie zależnością OPCJONALNĄ (warstwa audio też sobie bez
    niej radzi), więc nie zakładamy jego obecności — sprawdzamy.
    """
    try:
        import numpy
    except ImportError:  # pragma: no cover - zależne od instalacji
        return None
    return numpy


@runtime_checkable
class VectorIndex(Protocol):
    """Struktura wyszukiwania po podobieństwie. Identyfikatory to id z bazy."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None: ...

    def remove(self, ids: Sequence[int]) -> None: ...

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[int, float]]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


class NumpyVectorIndex:
    """Przegląd zupełny na macierzy NumPy — wariant zawsze dostępny.

    Wektory są znormalizowane do długości 1 (patrz ``brain.embeddings``), więc
    podobieństwo kosinusowe to zwykły iloczyn skalarny: jedno mnożenie macierzy.

    Macierz rośnie z zapasem (podwajanie pojemności), a nie o jeden wiersz na
    raz. To nie kosmetyka: reindeksacja dopisuje wektory pojedynczo, więc przy
    kopiowaniu całości na każde dopisanie koszt byłby kwadratowy — dla 10⁵
    wektorów po 384 wymiary oznaczałoby to przepisanie setek gigabajtów pamięci
    na maszynie, która i tak nie miała koła ``faiss-cpu``.
    """

    # Od tylu wierszy zaczyna każda nowa macierz — tyle, żeby krótka rozmowa
    # nigdy nie musiała się powiększać.
    _INITIAL_CAPACITY: Final[int] = 64

    def __init__(self, dimension: int) -> None:
        numpy = _numpy()
        if numpy is None:  # pragma: no cover - sprawdzane wcześniej przez fabrykę
            raise RuntimeError("NumPy jest wymagany do wyszukiwania wektorowego")
        self._numpy = numpy
        self._dimension = int(dimension)
        self._ids: list[int] = []
        # Pozycja identyfikatora w macierzy — bez tego każde dopisanie musiałoby
        # przeglądać całą listę id (też koszt kwadratowy przy reindeksacji).
        self._positions: dict[int, int] = {}
        self._matrix = numpy.zeros((self._INITIAL_CAPACITY, self._dimension), dtype=numpy.float32)

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def _rows(self) -> Any:
        """Wypełniona część macierzy (bez zapasu na przyszłe wektory)."""
        return self._matrix[: len(self._ids)]

    def _reserve(self, extra: int) -> None:
        """Zapewnij miejsce na ``extra`` nowych wierszy, podwajając pojemność."""
        needed = len(self._ids) + extra
        capacity = self._matrix.shape[0]
        if needed <= capacity:
            return
        numpy = self._numpy
        while capacity < needed:
            capacity = max(self._INITIAL_CAPACITY, capacity * 2)
        grown = numpy.zeros((capacity, self._dimension), dtype=numpy.float32)
        grown[: len(self._ids)] = self._rows
        self._matrix = grown

    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not ids:
            return
        numpy = self._numpy
        block = numpy.asarray(vectors, dtype=numpy.float32).reshape(len(ids), -1)
        if block.shape[1] != self._dimension:
            raise ValueError(
                t("vec.dim_mismatch", actual=block.shape[1], expected=self._dimension)
            )

        # Powtórzone id nadpisuje poprzedni wektor (aktualizacja treści źródła).
        fresh: list[tuple[int, Any]] = []
        for identifier, row in zip(ids, block, strict=True):
            position = self._positions.get(int(identifier))
            if position is None:
                fresh.append((int(identifier), row))
            else:
                self._matrix[position] = row

        if not fresh:
            return
        self._reserve(len(fresh))
        for identifier, row in fresh:
            position = len(self._ids)
            self._matrix[position] = row
            self._positions[identifier] = position
            self._ids.append(identifier)

    def remove(self, ids: Sequence[int]) -> None:
        if not ids or not self._ids:
            return
        unwanted = {int(identifier) for identifier in ids if int(identifier) in self._positions}
        if not unwanted:
            return
        keep = [position for position, item in enumerate(self._ids) if item not in unwanted]
        kept_rows = self._rows[keep].copy() if keep else None
        self._ids = [self._ids[position] for position in keep]
        self._positions = {identifier: index for index, identifier in enumerate(self._ids)}
        self._matrix = self._numpy.zeros(
            (max(self._INITIAL_CAPACITY, len(self._ids)), self._dimension),
            dtype=self._numpy.float32,
        )
        if kept_rows is not None:
            self._matrix[: len(self._ids)] = kept_rows

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        if not self._ids or limit <= 0:
            return []
        numpy = self._numpy
        query = numpy.asarray(vector, dtype=numpy.float32).reshape(-1)
        if query.shape[0] != self._dimension:
            return []
        scores = self._rows @ query
        count = min(limit, scores.shape[0])
        # argpartition wybiera top-k bez sortowania całości — przy 10⁵ wektorach
        # to zauważalna różnica względem pełnego sortowania.
        top = numpy.argpartition(-scores, count - 1)[:count]
        top = top[numpy.argsort(-scores[top])]
        return [(self._ids[int(position)], float(scores[int(position)])) for position in top]

    def clear(self) -> None:
        self._ids = []
        self._positions = {}
        self._matrix = self._numpy.zeros(
            (self._INITIAL_CAPACITY, self._dimension), dtype=self._numpy.float32
        )

    def __len__(self) -> int:
        return len(self._ids)


class FaissVectorIndex:
    """Indeks FAISS (``IndexFlatIP`` + mapowanie identyfikatorów).

    ``IndexFlatIP`` to też przegląd zupełny, ale liczony w C++ z wektoryzacją —
    przy tej samej dokładności jest kilkanaście razy szybszy od NumPy. Indeksów
    przybliżonych (IVF/HNSW) świadomie nie używamy: przy 10⁴–10⁵ wektorach nic
    nie dają, a wymagają uczenia i strojenia.
    """

    def __init__(self, dimension: int) -> None:
        numpy = _numpy()
        if numpy is None:  # pragma: no cover - sprawdzane wcześniej przez fabrykę
            raise RuntimeError("NumPy jest wymagany do wyszukiwania wektorowego")
        import faiss

        self._numpy = numpy
        self._faiss = faiss
        self._dimension = int(dimension)
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))

    @property
    def name(self) -> str:
        return "faiss"

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not ids:
            return
        numpy = self._numpy
        block = numpy.asarray(vectors, dtype=numpy.float32).reshape(len(ids), -1)
        if block.shape[1] != self._dimension:
            raise ValueError(
                t("vec.dim_mismatch", actual=block.shape[1], expected=self._dimension)
            )
        identifiers = numpy.asarray([int(item) for item in ids], dtype=numpy.int64)
        # IndexIDMap2 nie nadpisuje istniejącego id — najpierw usuwamy stare.
        self.remove(ids)
        self._index.add_with_ids(block, identifiers)

    def remove(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        numpy = self._numpy
        selector = numpy.asarray([int(item) for item in ids], dtype=numpy.int64)
        try:
            self._index.remove_ids(selector)
        except Exception:  # pragma: no cover - starsze wydania bywają wybredne
            logger.debug("FAISS nie usunął identyfikatorów %s", list(ids), exc_info=True)

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        if limit <= 0 or len(self) == 0:
            return []
        numpy = self._numpy
        query = numpy.asarray(vector, dtype=numpy.float32).reshape(1, -1)
        if query.shape[1] != self._dimension:
            return []
        scores, identifiers = self._index.search(query, min(limit, len(self)))
        results: list[tuple[int, float]] = []
        for identifier, score in zip(identifiers[0], scores[0], strict=True):
            # FAISS oznacza „brak wyniku" identyfikatorem -1.
            if int(identifier) >= 0:
                results.append((int(identifier), float(score)))
        return results

    def clear(self) -> None:
        self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self._dimension))

    def __len__(self) -> int:
        return int(self._index.ntotal)


def create_vector_index(dimension: int, *, prefer_faiss: bool = True) -> VectorIndex | None:
    """Zbuduj indeks: FAISS, jeśli jest; inaczej NumPy; ``None`` bez NumPy."""
    if dimension <= 0:
        return None
    if _numpy() is None:
        logger.info("Brak pakietu numpy — pamięć semantyczna wyłączona.")
        return None
    if prefer_faiss:
        try:
            return FaissVectorIndex(dimension)
        except ImportError:
            logger.debug("Brak pakietu faiss — wyszukiwanie wektorowe policzy NumPy.")
        except Exception as exc:  # pragma: no cover - uszkodzona instalacja faiss
            logger.warning("FAISS jest zainstalowany, ale nie działa (%s) — używam NumPy.", exc)
    return NumpyVectorIndex(dimension)


# --------------------------------------------------------------------------- #
# Pamięć semantyczna: baza + model + indeks
# --------------------------------------------------------------------------- #


class SemanticMemory:
    """Zapisywanie wspomnień z embeddingiem i odnajdywanie ich po znaczeniu.

    Wszystko jest tu „miękkie": brak modelu, brak NumPy, padnięta Ollama czy
    uszkodzony wektor mają wyłączyć wyszukiwanie semantyczne i zostawić resztę
    pamięci (fakty, notatki, szukanie po słowach) nietkniętą.
    """

    def __init__(
        self,
        database: Database,
        settings: Settings | None = None,
        *,
        provider: EmbeddingProvider | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        self._db = database
        self._settings = settings or get_settings()
        self._provider = provider if provider is not None else create_embedding_provider(
            self._settings
        )
        self._index: VectorIndex | None = index
        self._built = index is not None
        self._disabled = self._provider is None
        self._error = "" if self._provider is not None else "brak silnika embeddingów"
        # Zapytanie i zapis wypowiedzi użytkownika dotyczą tego samego tekstu —
        # bez tego bufora liczylibyśmy ten sam wektor dwa razy w każdej turze.
        self._last_text: str | None = None
        self._last_vector: list[float] | None = None

    # --- stan ------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return not self._disabled and self._provider is not None

    @property
    def model_name(self) -> str:
        return self._provider.name if self._provider is not None else ""

    @property
    def error(self) -> str:
        return self._error

    @property
    def size(self) -> int:
        return len(self._index) if self._index is not None else 0

    def describe(self) -> str:
        if self._provider is None:
            return t("status.semantic.no_engine")
        if self._disabled:
            return t("status.semantic.unavailable", reason=self._error)
        engine = (
            self._index.name if self._index is not None else t("status.semantic.index_off")
        )
        return t(
            "status.semantic.describe",
            provider=describe_provider(self._provider),
            index=engine,
            count=self.size,
        )

    def _disable(self, reason: str) -> None:
        """Wyłącz wyszukiwanie semantyczne do końca sesji.

        Bez tego każda kolejna wypowiedź próbowałaby ładować model, którego nie
        ma — i za każdym razem czekała kilka sekund na ten sam błąd.
        """
        if not self._disabled:
            logger.warning("Pamięć semantyczna wyłączona: %s", reason)
        self._disabled = True
        self._error = reason

    # --- liczenie i indeksowanie ----------------------------------------- #

    def embed(self, text: str) -> list[float] | None:
        """Wektor tekstu albo ``None``, gdy się nie da. Nigdy nie rzuca."""
        if not self.available or not text.strip():
            return None
        if text == self._last_text and self._last_vector is not None:
            return self._last_vector
        try:
            vectors = self._provider.embed([text])  # type: ignore[union-attr]
        except EmbeddingError as exc:
            self._disable(exc.message)
            return None
        except Exception as exc:  # pragma: no cover - nieprzewidziany błąd modelu
            logger.exception("Liczenie embeddingu nie powiodło się")
            self._disable(str(exc))
            return None
        if not vectors or not vectors[0]:
            return None
        self._last_text, self._last_vector = text, vectors[0]
        return vectors[0]

    def _new_index(self, dimension: int) -> VectorIndex | None:
        """Indeks zgodny z ustawieniem ``VECTOR_INDEX`` (patrz ``config.py``)."""
        requested = getattr(self._settings, "vector_index", "auto")
        index = create_vector_index(dimension, prefer_faiss=requested != "numpy")
        if index is not None and requested == "faiss" and index.name != "faiss":
            logger.warning(
                "VECTOR_INDEX=faiss, ale pakietu faiss nie da się tu użyć — liczy NumPy."
            )
        return index

    def _ensure_index(self, dimension: int) -> VectorIndex | None:
        if self._index is not None:
            return self._index
        self._index = self._new_index(dimension)
        if self._index is None:
            self._disable("brak pakietu numpy")
        return self._index

    def remember(
        self, source_table: str, source_id: int, text: str, *, vector: Sequence[float] | None = None
    ) -> bool:
        """Zapisz wspomnienie razem z embeddingiem. ``False`` = nie udało się."""
        if not self.available or source_id <= 0:
            return False
        values = list(vector) if vector else self.embed(text)
        if not values:
            return False

        index = self._ensure_index(len(values))
        if index is None:
            return False

        try:
            record = self._db.embeddings.set(
                source_table=source_table,
                source_id=source_id,
                text=text,
                model=self.model_name,
                vector=values,
            )
        except Exception as exc:
            logger.warning("Nie zapisano embeddingu (%s #%s): %s", source_table, source_id, exc)
            return False

        if record.id:
            try:
                index.add([record.id], [values])
            except ValueError as exc:
                # Zmiana modelu w trakcie sesji: wymiary przestały pasować.
                logger.warning("Wektor nie pasuje do indeksu (%s) — przebudowuję.", exc)
                self.rebuild()
        return True

    def forget(self, source_table: str, source_id: int) -> bool:
        """Usuń wspomnienie z indeksu i z bazy wektorów."""
        try:
            record = self._db.embeddings.get(source_table, source_id, self.model_name)
            removed = self._db.embeddings.delete(source_table, source_id)
        except Exception as exc:
            logger.warning("Nie usunięto embeddingu (%s #%s): %s", source_table, source_id, exc)
            return False
        if record is not None and record.id and self._index is not None:
            self._index.remove([record.id])
        return removed > 0

    # --- budowa indeksu -------------------------------------------------- #

    def rebuild(self) -> int:
        """Zbuduj indeks od nowa z wektorów leżących w bazie. Zwraca ich liczbę.

        Wołane przy pierwszym wyszukiwaniu. Wektory policzone INNYM modelem są
        pomijane — nie da się porównywać wektorów z różnych przestrzeni.
        """
        if not self.available:
            return 0
        try:
            records = self._db.embeddings.for_model(self.model_name)
        except Exception as exc:
            logger.warning("Nie udało się wczytać wektorów z bazy: %s", exc)
            self._built = True
            return 0

        self._built = True
        if not records:
            return 0

        dimension = len(records[0].vector)
        # Wektor o innej długości niż reszta = ślad po zmianie modelu bez zmiany
        # nazwy albo uszkodzony wiersz. Wypada z indeksu, zostaje w bazie.
        usable = [record for record in records if len(record.vector) == dimension and record.id]
        index = self._new_index(dimension)
        if index is None:
            self._disable("brak pakietu numpy")
            return 0

        index.add([record.id or 0 for record in usable], [record.vector for record in usable])
        self._index = index
        skipped = len(records) - len(usable)
        if skipped:
            logger.warning("Pominięto %s wektorów o niezgodnej długości.", skipped)
        logger.info("Indeks semantyczny: %s wektorów (%s).", len(usable), index.name)
        return len(usable)

    def reindex(self, *, progress: Callable[[str, int], None] | None = None) -> int:
        """Policz embeddingi dla WSZYSTKIEGO, co jest w bazie. Zwraca liczbę wektorów.

        Potrzebne po zmianie modelu i po włączeniu pamięci semantycznej na bazie,
        która powstała wcześniej.
        """
        if not self.available:
            return 0

        total = 0
        batch_size = max(1, self._settings.embedding_batch_size)
        for table, query in INDEXED_SOURCES:
            try:
                rows = self._db.query(query)
            except Exception as exc:
                logger.warning("Reindeksacja pominęła tabelę %s: %s", table, exc)
                continue

            pending: list[tuple[int, str]] = [
                (int(row["id"]), str(row["text"]))
                for row in rows
                if str(row["text"] or "").strip()
            ]
            done = 0
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                if not self._index_chunk(table, chunk):
                    return total + done
                done += len(chunk)
            total += done
            if progress is not None and done:
                progress(table, done)
        return total

    def _index_chunk(self, table: str, chunk: Sequence[tuple[int, str]]) -> bool:
        """Policz i zapisz wektory dla porcji rekordów. ``False`` = przerwij."""
        if not chunk or self._provider is None:
            return True
        try:
            vectors = self._provider.embed([text for _, text in chunk])
        except EmbeddingError as exc:
            self._disable(exc.message)
            return False
        except Exception as exc:  # pragma: no cover - nieprzewidziany błąd modelu
            logger.exception("Reindeksacja przerwana")
            self._disable(str(exc))
            return False

        for (source_id, text), vector in zip(chunk, vectors, strict=True):
            self.remember(table, source_id, text, vector=vector)
        return True

    # --- wyszukiwanie ---------------------------------------------------- #

    def search(
        self, query: str, *, limit: int = 5, min_score: float = 0.0
    ) -> list[tuple[str, int, float]]:
        """Znajdź podobne wspomnienia: ``[(tabela, id źródła, podobieństwo)]``.

        Zwracane są identyfikatory, a nie treść — po treść (i po datę) idzie
        :meth:`recall`, żeby nie trzymać całej bazy w pamięci indeksu.
        """
        if not self.available or limit <= 0:
            return []
        vector = self.embed(query)
        if not vector:
            return []
        if not self._built:
            self.rebuild()
        if self._index is None or len(self._index) == 0:
            return []

        hits = self._index.search(vector, limit)
        if not hits:
            return []

        rows = self._fetch_rows([identifier for identifier, _ in hits])
        results: list[tuple[str, int, float]] = []
        for identifier, score in hits:
            row = rows.get(identifier)
            if row is None or score < min_score:
                continue
            results.append((row[0], row[1], score))
        return results

    def recall(
        self, query: str, *, limit: int = 5, min_score: float = 0.0
    ) -> list[tuple[str, int, str, float, datetime | None]]:
        """To co :meth:`search`, ale z treścią i datą wspomnienia."""
        if not self.available or limit <= 0:
            return []
        vector = self.embed(query)
        if not vector:
            return []
        if not self._built:
            self.rebuild()
        if self._index is None or len(self._index) == 0:
            return []

        hits = self._index.search(vector, limit)
        rows = self._fetch_rows([identifier for identifier, _ in hits])
        recalled: list[tuple[str, int, str, float, datetime | None]] = []
        for identifier, score in hits:
            row = rows.get(identifier)
            if row is None or score < min_score:
                continue
            table, source_id, text, created_at = row
            recalled.append((table, source_id, text, score, created_at))
        return recalled

    def _fetch_rows(
        self, identifiers: Sequence[int]
    ) -> dict[int, tuple[str, int, str, datetime | None]]:
        """Dociągnij treść wskazanych wektorów jednym zapytaniem."""
        if not identifiers:
            return {}
        from database.models import from_iso

        placeholders = ",".join("?" for _ in identifiers)
        try:
            rows = self._db.query(
                "SELECT id, source_table, source_id, text, created_at FROM embeddings"
                f" WHERE id IN ({placeholders})",  # same znaki zapytania, bez danych
                [int(identifier) for identifier in identifiers],
            )
        except Exception as exc:
            logger.warning("Nie udało się odczytać wspomnień z bazy: %s", exc)
            return {}
        return {
            int(row["id"]): (
                str(row["source_table"]),
                int(row["source_id"]),
                str(row["text"]),
                from_iso(row["created_at"]),
            )
            for row in rows
        }

    def close(self) -> None:
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()


__all__ = [
    "INDEXED_SOURCES",
    "FaissVectorIndex",
    "NumpyVectorIndex",
    "SemanticMemory",
    "VectorIndex",
    "create_vector_index",
]
