"""Testy indeksu wektorowego i pamięci semantycznej (Faza 6).

Embeddingi liczy atrapa (``conftest.FakeEmbeddingProvider``) — deterministyczna
i niezależna od maszyny. Baza to plik SQLite w ``tmp_path``. Testy FAISS-a
uruchamiają się tylko wtedy, gdy pakiet jest zainstalowany: jego brak jest
przewidzianym stanem, a nie awarią.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from conftest import FAKE_EMBEDDING_DIM, FakeEmbeddingProvider, fake_vector

from brain.embeddings import EmbeddingUnavailableError
from brain.vectorstore import (
    FaissVectorIndex,
    NumpyVectorIndex,
    SemanticMemory,
    create_vector_index,
)
from config import Settings
from database.database import Database
from i18n import t

HAS_FAISS = importlib.util.find_spec("faiss") is not None
faiss_only = pytest.mark.skipif(not HAS_FAISS, reason="pakiet faiss-cpu nie jest zainstalowany")


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "pamiec.sqlite3")
    yield database
    database.close()


@pytest.fixture
def semantic(db: Database, tmp_path: Path) -> SemanticMemory:
    settings = Settings(_env_file=None, database_path=str(tmp_path / "pamiec.sqlite3"))
    return SemanticMemory(db, settings, provider=FakeEmbeddingProvider())


# --------------------------------------------------------------------------- #
# Indeksy
# --------------------------------------------------------------------------- #


def _index_classes() -> list[type]:
    return [NumpyVectorIndex, FaissVectorIndex] if HAS_FAISS else [NumpyVectorIndex]


@pytest.mark.parametrize("klasa", _index_classes())
def test_indeks_znajduje_najbardziej_podobny_wektor(klasa: type) -> None:
    index = klasa(FAKE_EMBEDDING_DIM)
    index.add([1, 2, 3], [fake_vector("rower górski"), fake_vector("kawa"), fake_vector("rower")])

    wyniki = index.search(fake_vector("rower"), 2)
    assert wyniki[0][0] == 3
    assert wyniki[0][1] > wyniki[1][1]
    assert len(index) == 3


@pytest.mark.parametrize("klasa", _index_classes())
def test_indeks_usuwa_wektor(klasa: type) -> None:
    index = klasa(FAKE_EMBEDDING_DIM)
    index.add([1, 2], [fake_vector("rower"), fake_vector("kawa")])
    index.remove([1])

    assert len(index) == 1
    assert [identifier for identifier, _ in index.search(fake_vector("rower"), 5)] == [2]


@pytest.mark.parametrize("klasa", _index_classes())
def test_powtorzony_identyfikator_nadpisuje_wektor(klasa: type) -> None:
    index = klasa(FAKE_EMBEDDING_DIM)
    index.add([1], [fake_vector("rower")])
    index.add([1], [fake_vector("kawa")])

    assert len(index) == 1
    wynik = index.search(fake_vector("kawa"), 1)
    assert wynik[0][0] == 1 and wynik[0][1] > 0.9


@pytest.mark.parametrize("klasa", _index_classes())
def test_pusty_indeks_nic_nie_zwraca(klasa: type) -> None:
    index = klasa(FAKE_EMBEDDING_DIM)
    assert index.search(fake_vector("cokolwiek"), 5) == []
    assert len(index) == 0


@pytest.mark.parametrize("klasa", _index_classes())
def test_wektor_o_zlej_dlugosci_jest_odrzucany(klasa: type) -> None:
    index = klasa(FAKE_EMBEDDING_DIM)
    with pytest.raises(ValueError):
        index.add([1], [[0.5, 0.5]])
    # Zapytanie o złym wymiarze nie wywraca się, tylko nic nie znajduje.
    assert index.search([0.5, 0.5], 3) == []


@pytest.mark.parametrize("klasa", _index_classes())
def test_indeks_rosnie_ponad_poczatkowa_pojemnosc(klasa: type) -> None:
    """Dopisywanie po jednym wektorze (tak działa reindeksacja) musi być poprawne.

    Macierz NumPy rośnie z zapasem, więc test przechodzi przez granicę
    podwojenia i sprawdza, że nic się przy tym nie gubi ani nie przestawia.
    """
    # Wektory jednostkowe w osobnych wymiarach: każdy jest sam sobie najbliższy,
    # więc dopasowanie jest jednoznaczne (atrapa embeddingów przy 150 tekstach
    # i 64 wymiarach zaczęłaby dawać kolizje i test badałby ją, a nie indeks).
    wymiary = 256
    liczba = 150  # ponad początkową pojemność macierzy (64)

    def jednostkowy(pozycja: int) -> list[float]:
        wektor = [0.0] * wymiary
        wektor[pozycja] = 1.0
        return wektor

    index = klasa(wymiary)
    for numer in range(1, liczba + 1):
        index.add([numer], [jednostkowy(numer - 1)])

    assert len(index) == liczba
    wynik = index.search(jednostkowy(136), 1)
    assert wynik[0][0] == 137 and wynik[0][1] == pytest.approx(1.0)

    # Usunięcie kompaktuje macierz — pozostałe wektory nadal muszą się znajdować.
    index.remove(list(range(1, 100)))
    assert len(index) == liczba - 99
    assert index.search(jednostkowy(136), 1)[0][0] == 137
    assert [identifier for identifier, _ in index.search(jednostkowy(49), 5)].count(50) == 0


def test_fabryka_zawsze_daje_dzialajacy_indeks() -> None:
    index = create_vector_index(FAKE_EMBEDDING_DIM)
    assert index is not None
    assert index.name == ("faiss" if HAS_FAISS else "numpy")
    assert create_vector_index(FAKE_EMBEDDING_DIM, prefer_faiss=False).name == "numpy"
    assert create_vector_index(0) is None


def test_ustawienie_wymusza_indeks_numpy(db: Database, tmp_path: Path) -> None:
    """VECTOR_INDEX=numpy to wyjście awaryjne dla procesorów bez AVX2.

    Koło ``faiss-cpu`` bywa zbudowane pod instrukcje, których starszy procesor nie
    ma — wtedy biblioteka nie zgłasza błędu, tylko przerywa proces. Musi więc
    istnieć sposób jej pominięcia bez odinstalowywania pakietu.
    """
    settings = Settings(
        _env_file=None, database_path=str(tmp_path / "pamiec.sqlite3"), vector_index="numpy"
    )
    semantic = SemanticMemory(db, settings, provider=FakeEmbeddingProvider())
    semantic.remember("notes", 1, "rower stoi w piwnicy")

    # Opis mówi, CZYM liczone jest podobieństwo — nazwa indeksu jest techniczna,
    # więc nie zmienia się z językiem interfejsu.
    assert "numpy" in semantic.describe()
    assert semantic.search("rower") != []


def test_brak_numpy_wylacza_wyszukiwanie(monkeypatch: pytest.MonkeyPatch) -> None:
    """Na maszynie bez NumPy pamięć semantyczna ma się wyłączyć, a nie wysypać."""
    import brain.vectorstore

    monkeypatch.setattr(brain.vectorstore, "_numpy", lambda: None)
    assert create_vector_index(FAKE_EMBEDDING_DIM) is None


@faiss_only
def test_faiss_i_numpy_daja_te_same_wyniki() -> None:
    """Wariant zapasowy musi zwracać to samo co przyspieszony — inaczej nie jest zapasowy."""
    teksty = [
        "rower górski",  # 1.00 — identyczne
        "rower górski stoi w piwnicy",  # 0.63
        "rower miejski",  # 0.50
        "górski szlak w tatrach",  # 0.35
        "kawa z mlekiem",  # 0.00 — nic wspólnego
    ]
    numpy_index = NumpyVectorIndex(FAKE_EMBEDDING_DIM)
    faiss_index = FaissVectorIndex(FAKE_EMBEDDING_DIM)
    for index in (numpy_index, faiss_index):
        index.add(list(range(1, len(teksty) + 1)), [fake_vector(text) for text in teksty])

    zapytanie = fake_vector("rower górski")
    z_numpy = numpy_index.search(zapytanie, 5)
    z_faiss = faiss_index.search(zapytanie, 5)

    # Wyniki podobieństwa muszą się zgadzać co do wartości...
    assert [round(score, 5) for _, score in z_numpy] == [round(score, 5) for _, score in z_faiss]
    # ...a kolejność wszędzie tam, gdzie wyniki są różne. Ostatni wektor ma
    # podobieństwo zerowe i przy remisie każda z bibliotek może go ustawić po
    # swojemu — to nie jest błąd.
    assert [identifier for identifier, _ in z_numpy[:4]] == [
        identifier for identifier, _ in z_faiss[:4]
    ]
    assert [identifier for identifier, _ in z_numpy[:4]] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Pamięć semantyczna
# --------------------------------------------------------------------------- #


def test_wspomnienie_jest_zapisywane_z_wektorem(semantic: SemanticMemory, db: Database) -> None:
    notatka = db.notes.add("Rower stoi w piwnicy")
    assert semantic.remember("notes", notatka.id, "Rower stoi w piwnicy")

    zapisany = db.embeddings.get("notes", notatka.id, semantic.model_name)
    assert zapisany is not None
    assert zapisany.dim == FAKE_EMBEDDING_DIM
    assert len(zapisany.vector) == FAKE_EMBEDDING_DIM
    assert db.embeddings.count() == 1


def test_wyszukiwanie_znajduje_po_znaczeniu(semantic: SemanticMemory, db: Database) -> None:
    for tekst in ("rower górski stoi w piwnicy", "ulubiona kawa to flat white"):
        notatka = db.notes.add(tekst)
        semantic.remember("notes", notatka.id, tekst)

    wyniki = semantic.recall("rower", limit=1)
    assert len(wyniki) == 1
    tabela, _, tekst, wynik, _ = wyniki[0]
    assert tabela == "notes" and "rower" in tekst and wynik > 0.3


def test_prog_podobienstwa_odcina_slabe_trafienia(
    semantic: SemanticMemory, db: Database
) -> None:
    notatka = db.notes.add("zupełnie inny temat")
    semantic.remember("notes", notatka.id, "zupełnie inny temat")

    assert semantic.recall("rower", limit=5, min_score=0.99) == []
    assert semantic.recall("zupełnie inny temat", limit=5, min_score=0.99) != []


def test_indeks_odbudowuje_sie_z_bazy(db: Database, tmp_path: Path) -> None:
    """Plik indeksu nie jest zapisywany — po restarcie powstaje z SQLite."""
    settings = Settings(_env_file=None, database_path=str(tmp_path / "pamiec.sqlite3"))
    pierwsza = SemanticMemory(db, settings, provider=FakeEmbeddingProvider())
    notatka = db.notes.add("rower górski")
    pierwsza.remember("notes", notatka.id, "rower górski")

    druga = SemanticMemory(db, settings, provider=FakeEmbeddingProvider())
    assert druga.rebuild() == 1
    assert druga.recall("rower", limit=1) != []
    assert list(tmp_path.rglob("*.faiss")) == []


def test_wektory_innego_modelu_sa_pomijane(db: Database, tmp_path: Path) -> None:
    """Wektorów z różnych modeli nie wolno porównywać — mają rozłączne przestrzenie."""
    settings = Settings(_env_file=None, database_path=str(tmp_path / "pamiec.sqlite3"))
    stary = SemanticMemory(db, settings, provider=FakeEmbeddingProvider(name="model-stary"))
    notatka = db.notes.add("rower górski")
    stary.remember("notes", notatka.id, "rower górski")

    nowy = SemanticMemory(db, settings, provider=FakeEmbeddingProvider(name="model-nowy"))
    assert nowy.rebuild() == 0
    assert nowy.recall("rower") == []
    # Stare wektory zostają w bazie (do reindeksacji), po prostu nie są używane.
    assert db.embeddings.count() == 1


def test_zapominanie_usuwa_wektor(semantic: SemanticMemory, db: Database) -> None:
    notatka = db.notes.add("rower górski")
    semantic.remember("notes", notatka.id, "rower górski")
    assert semantic.recall("rower") != []

    assert semantic.forget("notes", notatka.id) is True
    assert db.embeddings.count() == 0
    assert semantic.recall("rower") == []


def test_reindeksacja_obejmuje_wszystkie_zrodla(semantic: SemanticMemory, db: Database) -> None:
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "mam rower górski")
    db.messages.add(rozmowa.id, "assistant", "to odpowiedź asystenta")
    db.facts.set("imie", "Mariusz")
    db.notes.add("notatka o kawie")
    db.summaries.add(rozmowa.id, "streszczenie rozmowy")

    postep: list[tuple[str, int]] = []
    policzone = semantic.reindex(progress=lambda table, done: postep.append((table, done)))

    # Wypowiedź asystenta nie jest indeksowana — pamięć dotyczy użytkownika.
    assert policzone == 4
    assert dict(postep) == {"facts": 1, "notes": 1, "summaries": 1, "messages": 1}
    assert semantic.recall("rower") != []


def test_awaria_modelu_wylacza_warstwe_i_nie_probuje_w_kolko(
    db: Database, tmp_path: Path
) -> None:
    settings = Settings(_env_file=None, database_path=str(tmp_path / "pamiec.sqlite3"))
    provider = FakeEmbeddingProvider()
    provider.error = EmbeddingUnavailableError("model padł")
    semantic = SemanticMemory(db, settings, provider=provider)

    assert semantic.embed("cokolwiek") is None
    assert not semantic.available
    assert "model padł" in semantic.error

    semantic.embed("jeszcze raz")
    semantic.recall("i jeszcze")
    # Po wyłączeniu nie wołamy modelu ponownie — inaczej każda tura czekałaby
    # kilka sekund na ten sam błąd.
    assert len(provider.calls) == 1


def test_ten_sam_tekst_liczony_jest_raz(semantic: SemanticMemory, db: Database) -> None:
    """Zapytanie i zapis wypowiedzi dotyczą tego samego tekstu."""
    provider = semantic._provider  # test sprawdza liczbę wywołań
    semantic.embed("mam rower")
    semantic.embed("mam rower")
    assert len(provider.calls) == 1


def test_uszkodzony_wektor_wypada_z_indeksu(semantic: SemanticMemory, db: Database) -> None:
    notatka = db.notes.add("rower")
    semantic.remember("notes", notatka.id, "rower")
    # Symulacja uszkodzonego wiersza: blob o długości niepodzielnej przez 4.
    db.execute("UPDATE embeddings SET vector = ? WHERE source_id = ?", (b"abc", notatka.id))

    swieza = SemanticMemory(
        db,
        Settings(_env_file=None),
        provider=FakeEmbeddingProvider(),
    )
    assert swieza.rebuild() == 0
    assert swieza.recall("rower") == []


def test_wektory_osierocone_da_sie_posprzatac(semantic: SemanticMemory, db: Database) -> None:
    rozmowa = db.conversations.start()
    wiadomosc = db.messages.add(rozmowa.id, "user", "mam rower")
    semantic.remember("messages", wiadomosc.id, "mam rower")
    assert db.embeddings.count() == 1

    db.conversations.delete(rozmowa.id)  # wiadomości znikają kaskadą
    assert db.embeddings.prune_orphans() == 1
    assert db.embeddings.count() == 0


def test_opis_warstwy_mowi_czym_liczy(semantic: SemanticMemory, db: Database) -> None:
    notatka = db.notes.add("rower")
    semantic.remember("notes", notatka.id, "rower")
    opis = semantic.describe()
    assert "atrapa" in opis
    assert ("faiss" if HAS_FAISS else "numpy") in opis
    assert opis.endswith(t("status.semantic.describe", provider="", index="", count=1).split(": ")[-1])


def test_lista_modelow_w_indeksie(semantic: SemanticMemory, db: Database) -> None:
    notatka = db.notes.add("rower")
    semantic.remember("notes", notatka.id, "rower")
    modele = db.embeddings.models()
    assert modele == [("atrapa-embeddingow", FAKE_EMBEDDING_DIM, 1)]
    assert db.embeddings.clear(model="atrapa-embeddingow") == 1
