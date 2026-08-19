"""Testy pamięci długoterminowej — warstwa bazy (Faza 5).

Każdy test dostaje własny plik SQLite w ``tmp_path``. Żaden nie dotyka bazy
asystenta na maszynie dewelopera: ścieżka jest podawana wprost, a dodatkowo
autouse'owa atrapa katalogu danych (``conftest.isolated_data_dir``) przekierowuje
domyślną lokalizację w katalog tymczasowy.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from config import MEMORY_DATABASE, Settings
from database.database import Database, DatabaseError
from database.migrations import (
    SCHEMA_VERSION,
    apply_migrations,
    backup_database,
    backup_name,
    current_version,
    pending_migrations,
    supports_fts5,
)
from database.models import Fact, Note, from_iso, load_metadata, to_iso, utc_now
from i18n import t


@pytest.fixture
def db(tmp_path: Path):
    """Baza w pliku tymczasowym — zamykana po teście."""
    database = Database(tmp_path / "pamiec.sqlite3")
    yield database
    database.close()


# --------------------------------------------------------------------------- #
# Schemat i migracje
# --------------------------------------------------------------------------- #


def test_swieza_baza_dostaje_pelny_schemat(tmp_path: Path) -> None:
    with Database(tmp_path / "nowa.sqlite3") as database:
        assert database.schema_version == SCHEMA_VERSION
        tables = {
            row["name"]
            for row in database.query("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"conversations", "messages", "summaries", "facts", "preferences", "notes"} <= tables


def test_ponowne_otwarcie_nie_powtarza_migracji(tmp_path: Path) -> None:
    path = tmp_path / "pamiec.sqlite3"
    with Database(path) as first:
        first.conversations.start()
        historia = list(first.migration_history())

    with Database(path) as second:
        assert second.schema_version == SCHEMA_VERSION
        assert list(second.migration_history()) == historia
        assert second.conversations.recent() != []


def test_migracje_sa_idempotentne_na_poziomie_polaczenia(tmp_path: Path) -> None:
    connection = sqlite3.connect(str(tmp_path / "reczna.sqlite3"), isolation_level=None)
    try:
        assert current_version(connection) == 0
        assert apply_migrations(connection, backup=False)
        assert current_version(connection) == SCHEMA_VERSION
        # Druga próba nie ma czego robić.
        assert apply_migrations(connection, backup=False) == []
        assert pending_migrations(connection) == []
    finally:
        connection.close()


def test_kopia_zapasowa_powstaje_i_da_sie_ja_otworzyc(tmp_path: Path) -> None:
    path = tmp_path / "pamiec.sqlite3"
    with Database(path) as database:
        database.facts.set("imie", "Mariusz")
        kopia = backup_database(database.connection(), backup_name(path, 1))

    assert kopia is not None and kopia.exists()
    with Database(kopia) as odczytana:
        assert odczytana.facts.get("imie") is not None


def test_uszkodzona_baza_daje_czytelny_blad(tmp_path: Path) -> None:
    path = tmp_path / "smieci.sqlite3"
    path.write_bytes(b"to zdecydowanie nie jest baza SQLite" * 10)
    with pytest.raises(DatabaseError) as blad:
        Database(path)
    assert "logs" in blad.value.user_message or "uszkodzony" in blad.value.user_message


def test_brak_miejsca_na_baze_konczy_sie_wyjatkiem_z_podpowiedzia(tmp_path: Path) -> None:
    """Katalog nadrzędny jest PLIKIEM — nie da się w nim założyć bazy (na każdym systemie)."""
    przeszkoda = tmp_path / "to-jest-plik"
    przeszkoda.write_text("blokada", encoding="utf-8")

    with pytest.raises(DatabaseError) as blad:
        Database(przeszkoda / "podkatalog" / "pamiec.sqlite3")
    assert "MIKU_DATA_DIR" in blad.value.user_message


# --------------------------------------------------------------------------- #
# Ustawienia połączenia
# --------------------------------------------------------------------------- #


def test_klucze_obce_kasuja_wiadomosci_razem_z_rozmowa(db: Database) -> None:
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "pierwsza")
    db.messages.add(rozmowa.id, "assistant", "druga")
    assert db.messages.count(rozmowa.id) == 2

    db.conversations.delete(rozmowa.id)
    assert db.messages.count(rozmowa.id) == 0


def test_dziennik_i_wyszukiwanie_sa_raportowane_bez_zalozen(db: Database) -> None:
    """Nie zakładamy, że WAL i FTS5 są dostępne — sprawdzamy, że stan jest znany."""
    assert db.journal_mode in {"wal", "delete", "truncate", "persist", "memory", "off", "unknown"}
    assert isinstance(db.has_fulltext, bool)
    assert str(db.path) in db.describe()


def test_baza_w_pamieci_dziala_i_nie_tworzy_pliku(tmp_path: Path) -> None:
    with Database(MEMORY_DATABASE) as database:
        assert database.in_memory
        rozmowa = database.conversations.start()
        database.messages.add(rozmowa.id, "user", "cześć")
        assert database.messages.count() == 1
    assert list(tmp_path.rglob("*.sqlite3")) == []


def test_settings_wskazuje_plik_bazy(tmp_path: Path) -> None:
    cel = tmp_path / "wskazana.sqlite3"
    settings = Settings(_env_file=None, database_path=str(cel))
    with Database.open(settings) as database:
        assert database.path == cel
    assert cel.exists()


# --------------------------------------------------------------------------- #
# Rozmowy i wiadomości
# --------------------------------------------------------------------------- #


def test_wiadomosci_wracaja_w_kolejnosci_zapisu(db: Database) -> None:
    rozmowa = db.conversations.start(source="terminal", model="testowy")
    db.messages.add(rozmowa.id, "user", "pierwsza", language="pl")
    db.messages.add(rozmowa.id, "assistant", "druga", metadata={"latencja_ms": 120})

    wiadomosci = db.messages.for_conversation(rozmowa.id)
    assert [item.role for item in wiadomosci] == ["user", "assistant"]
    assert wiadomosci[0].language == "pl"
    assert wiadomosci[1].metadata["latencja_ms"] == 120

    zapisana = db.conversations.get(rozmowa.id)
    assert zapisana is not None and zapisana.message_count == 2


def test_zamkniecie_rozmowy_ustawia_koniec(db: Database) -> None:
    rozmowa = db.conversations.start()
    assert db.conversations.get(rozmowa.id).is_open

    db.conversations.finish(rozmowa.id)
    zamknieta = db.conversations.get(rozmowa.id)
    assert not zamknieta.is_open and zamknieta.ended_at is not None


def test_poprzednia_rozmowa_jest_odnajdywana(db: Database) -> None:
    stara = db.conversations.start()
    db.conversations.finish(stara.id)
    nowa = db.conversations.start()

    znaleziona = db.conversations.last_finished(exclude_id=nowa.id)
    assert znaleziona is not None and znaleziona.id == stara.id


# --------------------------------------------------------------------------- #
# Wyszukiwanie
# --------------------------------------------------------------------------- #


def test_szukanie_znajduje_wiadomosc(db: Database) -> None:
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "Mam rower marki Kellys i jeżdżę nim do pracy.")
    db.messages.add(rozmowa.id, "assistant", "Zapamiętam to sobie.")

    trafienia = db.messages.search("rower")
    assert len(trafienia) == 1
    assert "Kellys" in trafienia[0].text


def test_szukanie_dziala_takze_bez_fts5(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ścieżka zapasowa (LIKE) musi dawać ten sam wynik co FTS5."""
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "Ulubiona herbata to yerba mate.")

    monkeypatch.setattr(type(db), "has_fulltext", property(lambda self: False))
    trafienia = db.messages.search("yerba")
    assert len(trafienia) == 1 and "yerba" in trafienia[0].text


@pytest.mark.parametrize("fraza", ['"cudzysłów"', "100%", "a_b", "*", "NEAR(x)", "-minus", ""])
def test_szukanie_nie_wywraca_sie_na_znakach_specjalnych(db: Database, fraza: str) -> None:
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "zwykła treść bez znaków specjalnych")
    assert isinstance(db.messages.search(fraza), list)


def test_like_nie_traktuje_procentu_jak_wieloznacznika(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "bateria na 100% naładowana")
    db.messages.add(rozmowa.id, "user", "zupełnie inna wiadomość")

    monkeypatch.setattr(type(db), "has_fulltext", property(lambda self: False))
    assert len(db.messages.search("100%")) == 1


# --------------------------------------------------------------------------- #
# Fakty, preferencje, notatki
# --------------------------------------------------------------------------- #


def test_fakt_jest_nadpisywany_a_nie_duplikowany(db: Database) -> None:
    db.facts.set("miasto", "Wrocław")
    db.facts.set("MIASTO", "Kraków")  # klucz jest normalizowany do małych liter

    assert db.facts.count() == 1
    fakt = db.facts.get("miasto")
    assert fakt is not None and fakt.value == "Kraków"


def test_fakt_tworzy_wpis_metadanych_wspomnienia(db: Database) -> None:
    fakt = db.facts.set("imie", "Mariusz")
    wspomnienia = db.memories.top(kind="fact")
    assert [item.source_id for item in wspomnienia] == [fakt.id]
    assert wspomnienia[0].summary == "imie: Mariusz"


def test_zapominanie_faktu_usuwa_tez_metadane(db: Database) -> None:
    db.facts.set("imie", "Mariusz")
    assert db.facts.delete("imie") is True
    assert db.facts.get("imie") is None
    assert db.memories.count(kind="fact") == 0
    assert db.facts.delete("imie") is False


def test_pusty_fakt_jest_odrzucany(db: Database) -> None:
    with pytest.raises(ValueError):
        db.facts.set("   ", "cokolwiek")
    with pytest.raises(ValueError):
        db.facts.set("klucz", "   ")


def test_preferencje_maja_wartosc_domyslna(db: Database) -> None:
    assert db.preferences.value_of("dlugosc", "krotko") == "krotko"
    db.preferences.set("dlugosc", "bardzo krótko")
    assert db.preferences.value_of("dlugosc") == "bardzo krótko"
    assert db.preferences.delete("dlugosc") is True


def test_notatki_da_sie_zapisac_i_znalezc(db: Database) -> None:
    notatka = db.notes.add("Kupić mleko i chleb", title="Zakupy", tags=["dom", "DOM"])
    assert notatka.tags == ["dom"]

    trafienia = db.notes.search("mleko")
    assert len(trafienia) == 1 and trafienia[0].source_id == notatka.id
    assert db.notes.delete(notatka.id) is True
    assert db.notes.search("mleko") == []


# --------------------------------------------------------------------------- #
# Streszczenia, statystyki, sprzątanie
# --------------------------------------------------------------------------- #


def test_streszczenia_wracaja_od_najnowszego(db: Database) -> None:
    rozmowa = db.conversations.start()
    db.summaries.add(rozmowa.id, "pierwsze", message_count=3, generation=1)
    db.summaries.add(rozmowa.id, "drugie", message_count=5, generation=2)

    ostatnie = db.summaries.latest(rozmowa.id)
    assert ostatnie is not None and ostatnie.content == "drugie" and ostatnie.generation == 2
    assert len(db.summaries.for_conversation(rozmowa.id)) == 2


def test_statystyki_licza_wszystkie_tabele(db: Database) -> None:
    rozmowa = db.conversations.start()
    db.messages.add(rozmowa.id, "user", "raz")
    db.facts.set("imie", "Mariusz")
    db.notes.add("notatka")

    stats = db.stats()
    assert (stats.conversations, stats.messages, stats.facts, stats.notes) == (1, 1, 1, 1)
    assert stats.describe() == t(
        "status.stats.describe",
        conversations=stats.conversations,
        messages=stats.messages,
        summaries=stats.summaries,
        facts=stats.facts,
        preferences=stats.preferences,
        notes=stats.notes,
        embeddings=stats.embeddings,
        audit=stats.tool_audit,
    )
    assert stats.facts == 1


def test_czyszczenie_starych_rozmow_oszczedza_fakty(db: Database) -> None:
    stara = db.conversations.start()
    db.messages.add(stara.id, "user", "dawna rozmowa")
    db.facts.set("imie", "Mariusz")
    # Cofnięcie daty startu — jedyny sposób na „starą" rozmowę bez czekania.
    db.execute(
        "UPDATE conversations SET started_at = ? WHERE id = ?",
        (to_iso(utc_now().replace(year=utc_now().year - 1)), stara.id),
    )

    assert db.purge_older_than(30) == 1
    assert db.messages.count() == 0
    assert db.facts.get("imie") is not None
    assert db.purge_older_than(0) == 0


def test_wygasle_wspomnienia_da_sie_usunac(db: Database) -> None:
    notatka = db.notes.add("krótkotrwała")
    db.memories.record(
        kind="note",
        source_table="notes",
        source_id=notatka.id,
        expires_at=utc_now().replace(year=utc_now().year - 1),
    )
    assert db.memories.purge_expired() == 1


def test_licznik_uzyc_rosnie(db: Database) -> None:
    fakt = db.facts.set("imie", "Mariusz")
    wpis = db.memories.top(kind="fact")[0]
    db.memories.touch(wpis.id)

    odswiezony = db.memories.top(kind="fact")[0]
    assert odswiezony.use_count == 1 and odswiezony.last_used_at is not None
    assert odswiezony.source_id == fakt.id


# --------------------------------------------------------------------------- #
# Współbieżność i cykl życia
# --------------------------------------------------------------------------- #


def test_zapis_z_wielu_watkow_nie_gubi_wiadomosci(db: Database) -> None:
    rozmowa = db.conversations.start()
    bledy: list[BaseException] = []

    def pisz(numer: int) -> None:
        try:
            for krok in range(5):
                db.messages.add(rozmowa.id, "user", f"wątek {numer} krok {krok}")
        except BaseException as exc:  # pragma: no cover - sygnalizacja błędu do testu
            bledy.append(exc)

    watki = [threading.Thread(target=pisz, args=(numer,)) for numer in range(4)]
    for watek in watki:
        watek.start()
    for watek in watki:
        watek.join()

    assert bledy == []
    assert db.messages.count(rozmowa.id) == 20


def test_zamknieta_baza_odmawia_pracy(tmp_path: Path) -> None:
    database = Database(tmp_path / "pamiec.sqlite3")
    database.close()
    database.close()  # powtórzone zamknięcie jest bezpieczne

    with pytest.raises(DatabaseError):
        database.conversations.start()


def test_transakcja_wycofuje_zmiany_przy_bledzie(db: Database) -> None:
    rozmowa = db.conversations.start()
    with pytest.raises(RuntimeError):
        with db.transaction() as connection:
            connection.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at)"
                " VALUES (?, 'user', 'nieudana', ?)",
                (rozmowa.id, to_iso(utc_now())),
            )
            raise RuntimeError("coś poszło nie tak")

    assert db.messages.count(rozmowa.id) == 0


# --------------------------------------------------------------------------- #
# Modele rekordów
# --------------------------------------------------------------------------- #


def test_czas_jest_zapisywany_w_utc() -> None:
    moment = utc_now()
    assert to_iso(moment).endswith("+00:00")
    assert from_iso(to_iso(moment)).tzinfo is not None
    # Czas bez strefy jest traktowany jako UTC, a nie jako czas maszyny.
    naiwny = moment.replace(tzinfo=None)
    assert to_iso(naiwny) == to_iso(moment)


def test_uszkodzone_dane_nie_wywracaja_odczytu() -> None:
    assert from_iso("zupełnie nie data") is None
    assert from_iso(None) is None
    assert load_metadata("{niepoprawny json") == {}
    assert load_metadata(None) == {}


def test_rekordy_powstaja_z_wiersza_bazy() -> None:
    fakt = Fact.from_row(
        {
            "id": 7,
            "key": "imie",
            "value": "Mariusz",
            "source": "user",
            "confidence": 1.0,
            "created_at": "2026-08-15T10:00:00+00:00",
            "updated_at": "2026-08-15T10:00:00+00:00",
            "pinned": 1,
        }
    )
    assert fakt.id == 7 and fakt.pinned is True and fakt.as_line() == "imie: Mariusz"

    notatka = Note.from_row(
        {
            "id": 1,
            "title": "",
            "body": "x" * 200,
            "tags": "dom,praca",
            "created_at": "2026-08-15T10:00:00+00:00",
            "updated_at": "2026-08-15T10:00:00+00:00",
            "source": "user",
        }
    )
    assert notatka.tags == ["dom", "praca"] and notatka.preview.endswith("...")


def test_wykrywanie_fts5_nie_zostawia_smieci() -> None:
    connection = sqlite3.connect(MEMORY_DATABASE)
    try:
        wynik = supports_fts5(connection)
        assert isinstance(wynik, bool)
        zostalo = connection.execute(
            "SELECT name FROM temp.sqlite_master WHERE name LIKE '%fts5_probe%'"
        ).fetchall()
        assert zostalo == []
    finally:
        connection.close()
