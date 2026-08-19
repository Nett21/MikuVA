"""Testy pamięci asystenta (Faza 5) — okno rozmowy, streszczanie, zapis.

Model językowy jest atrapą: żaden test nie potrzebuje działającej Ollamy. Baza
to plik w ``tmp_path``, więc testy nie zależą od tego, co ktoś ma zapisane na
swojej maszynie.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from brain.conversation import ConversationHistory, Message
from brain.memory import ConversationMemory
from config import Settings
from database.database import Database
from i18n import t

# --------------------------------------------------------------------------- #
# Atrapy i pomocniki
# --------------------------------------------------------------------------- #


class FakeBackend:
    """Atrapa modelu: zapamiętuje, o co ją poproszono, i oddaje ustaloną treść."""

    def __init__(self, answer: str = "Użytkownik ma na imię Mariusz i pyta o rower.") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def chat(self, messages: Sequence[Message], *, system: str | None = None) -> str:
        self.calls.append({"messages": list(messages), "system": system})
        if self.error is not None:
            raise self.error
        return self.answer


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Ustawienia z małym oknem rozmowy — inaczej nic by z niego nie wypadało."""
    values: dict[str, Any] = {
        "database_path": str(tmp_path / "pamiec.sqlite3"),
        "history_max_messages": 4,
        "history_max_chars": 2_000,
        "memory_trim_ratio": 0.5,
        "memory_summary_max_chars": 400,
        # Warstwa semantyczna ma tu być WYŁĄCZONA: te testy dotyczą streszczania,
        # a nie embeddingów. Gdy w środowisku pojawi się sentence-transformers,
        # każdy taki test ładowałby prawdziwy model (7 s zamiast 0,1 s) — testy
        # zależałyby wtedy od tego, co ktoś ma zainstalowane.
        "embeddings_enabled": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --------------------------------------------------------------------------- #
# Okno rozmowy (brain/conversation.py)
# --------------------------------------------------------------------------- #


def test_okno_oddaje_wiadomosci_zamiast_je_gubic() -> None:
    wyrzucone: list[Message] = []
    historia = ConversationHistory(
        max_messages=4,
        max_chars=1_000,
        trim_ratio=0.5,
        on_evict=lambda items: wyrzucone.extend(items),
    )
    for numer in range(6):
        historia.add_user(f"wiadomość {numer}")

    assert len(historia) <= 4
    assert [item.content for item in wyrzucone][0] == "wiadomość 0"
    # Przycinanie schodzi PONIŻEJ limitu, więc nie dzieje się w każdej turze.
    assert len(historia) < 4


def test_okno_przycina_tez_po_liczbie_znakow() -> None:
    historia = ConversationHistory(max_messages=100, max_chars=300, trim_ratio=0.5)
    for numer in range(10):
        historia.add_user("x" * 100 + str(numer))

    assert historia.char_count <= 300
    assert len(historia) >= 1


def test_ostatnia_wiadomosc_nigdy_nie_wypada() -> None:
    historia = ConversationHistory(max_messages=4, max_chars=100)
    historia.add_user("x" * 5_000)
    assert len(historia) == 1


def test_blad_odbiorcy_nie_przerywa_rozmowy() -> None:
    def wybuchaj(_: Sequence[Message]) -> None:
        raise RuntimeError("awaria odbiorcy")

    historia = ConversationHistory(max_messages=2, max_chars=1_000, on_evict=wybuchaj)
    for numer in range(5):
        historia.add_user(f"wiadomość {numer}")
    assert len(historia) >= 1


# --------------------------------------------------------------------------- #
# Streszczanie zamiast obcinania
# --------------------------------------------------------------------------- #


def test_dluga_rozmowa_jest_streszczana_a_nie_obcinana(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    backend = FakeBackend("Użytkownik przedstawił się jako Mariusz i pytał o rower.")

    with ConversationMemory(settings) as memory:
        for numer in range(6):
            memory.add_user(f"pytanie numer {numer}")
            memory.add_assistant(f"odpowiedź numer {numer}")

        assert memory.needs_compaction
        streszczenie = run(memory.compact(backend))

        assert streszczenie == "Użytkownik przedstawił się jako Mariusz i pytał o rower."
        assert not memory.needs_compaction
        assert backend.calls, "model powinien zostać poproszony o streszczenie"
        # Materiał do streszczenia zawiera treść wiadomości, które wypadły z okna.
        assert "pytanie numer 0" in backend.calls[0]["messages"][0].content
        # Streszczenie trafia do promptu systemowego następnej tury.
        assert "Mariusz" in memory.context_block("pl")


def test_streszczenie_ladu_je_w_bazie(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as memory:
        for numer in range(6):
            memory.add_user(f"wiadomość {numer}")
        run(memory.compact(FakeBackend("krótkie streszczenie")))
        rozmowa = memory.conversation_id

    with Database(tmp_path / "pamiec.sqlite3") as db:
        zapisane = db.summaries.latest(rozmowa)
        assert zapisane is not None
        assert zapisane.content == "krótkie streszczenie"
        assert zapisane.method == "llm" and zapisane.message_count > 0


def test_awaria_modelu_daje_skrot_mechaniczny(tmp_path: Path) -> None:
    """Brak Ollamy nie może oznaczać utraty starszej części rozmowy."""
    settings = make_settings(tmp_path)
    backend = FakeBackend()
    backend.error = RuntimeError("Ollama nie odpowiada")

    with ConversationMemory(settings) as memory:
        for numer in range(6):
            memory.add_user(f"tajne hasło numer {numer}")
        streszczenie = run(memory.compact(backend))

        assert "tajne hasło numer 0" in streszczenie
        rozmowa = memory.conversation_id

    with Database(tmp_path / "pamiec.sqlite3") as db:
        assert db.summaries.latest(rozmowa).method == "fallback"


def test_streszczanie_bez_modelu_takze_dziala(tmp_path: Path) -> None:
    with ConversationMemory(make_settings(tmp_path)) as memory:
        for numer in range(6):
            memory.add_user(f"wiadomość {numer}")
        assert "wiadomość 0" in run(memory.compact(None))


def test_streszczenie_da_sie_wylaczyc(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, memory_summary_enabled=False)
    backend = FakeBackend()
    with ConversationMemory(settings) as memory:
        for numer in range(6):
            memory.add_user(f"wiadomość {numer}")
        run(memory.compact(backend))
        assert backend.calls == [], "model nie powinien być wołany"


def test_kolejne_streszczenie_bazuje_na_poprzednim(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    backend = FakeBackend("streszczenie pierwsze")
    with ConversationMemory(settings) as memory:
        for numer in range(6):
            memory.add_user(f"wiadomość {numer}")
        run(memory.compact(backend))

        backend.answer = "streszczenie drugie"
        for numer in range(6, 12):
            memory.add_user(f"wiadomość {numer}")
        run(memory.compact(backend))

        assert memory.summary == "streszczenie drugie"
        # Drugie wywołanie dostało poprzednie streszczenie jako punkt wyjścia.
        assert "streszczenie pierwsze" in backend.calls[1]["messages"][0].content


def test_streszczenie_jest_przycinane_do_limitu(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, memory_summary_max_chars=200)
    with ConversationMemory(settings) as memory:
        for numer in range(6):
            memory.add_user(f"wiadomość {numer}")
        streszczenie = run(memory.compact(FakeBackend("bardzo długie zdanie " * 100)))
        assert len(streszczenie) <= 200


def test_nic_do_streszczenia_nie_wola_modelu(tmp_path: Path) -> None:
    backend = FakeBackend()
    with ConversationMemory(make_settings(tmp_path)) as memory:
        memory.add_user("krótka rozmowa")
        assert run(memory.compact(backend)) == ""
        assert backend.calls == []


# --------------------------------------------------------------------------- #
# Pamięć trwała
# --------------------------------------------------------------------------- #


def test_fakty_przezywaja_restart_asystenta(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as pierwsza:
        assert pierwsza.remember_fact("imie", "Mariusz")
        assert pierwsza.set_preference("dlugosc", "krotko")
        pierwsza.add_note("Rower stoi w piwnicy")

    with ConversationMemory(settings) as druga:
        kontekst = druga.context_block("pl")
        assert "imie: Mariusz" in kontekst
        assert "dlugosc: krotko" in kontekst
        assert [note.body for note in druga.notes()] == ["Rower stoi w piwnicy"]
        assert druga.forget_fact("imie") is True
        assert "imie: Mariusz" not in druga.context_block("pl")


def test_nowa_sesja_czysci_okno_ale_nie_fakty(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as memory:
        memory.remember_fact("imie", "Mariusz")
        memory.add_user("coś tam")
        stara_rozmowa = memory.conversation_id

        memory.new_session()

        assert len(memory.history) == 0
        assert memory.summary == ""
        assert memory.conversation_id != stara_rozmowa
        assert "imie: Mariusz" in memory.context_block("pl")


def test_poprzednia_rozmowa_jest_przypominana(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as pierwsza:
        for numer in range(6):
            pierwsza.add_user(f"rozmowa o rowerze {numer}")
        run(pierwsza.compact(FakeBackend("Rozmawialiśmy o rowerze.")))

    with ConversationMemory(settings) as druga:
        kontekst = druga.context_block("pl")
        assert "Z wcześniejszej rozmowy" in kontekst
        assert "Rozmawialiśmy o rowerze." in kontekst


def test_bez_streszczenia_przypomina_ostatnie_wypowiedzi(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as pierwsza:
        pierwsza.add_user("mam na imię Mariusz")
        pierwsza.add_assistant("miło mi")

    with ConversationMemory(settings) as druga:
        assert "mam na imię Mariusz" in druga.context_block("pl")


def test_wyszukiwanie_obejmuje_rozmowy_i_notatki(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as memory:
        memory.add_user("Mój rower to Kellys")
        memory.add_note("Serwis roweru w maju")

        trafienia = memory.search("rower")
        assert {hit.kind for hit in trafienia} == {"message", "note"}


def test_kontekst_jest_pusty_na_swiezej_instalacji(tmp_path: Path) -> None:
    with ConversationMemory(make_settings(tmp_path)) as memory:
        assert memory.context_block("pl") == ""


def test_kontekst_ostrzega_ze_to_dane_a_nie_polecenia(tmp_path: Path) -> None:
    with ConversationMemory(make_settings(tmp_path)) as memory:
        memory.remember_fact("imie", "Mariusz")
        assert "nie polecenia" in memory.context_block("pl")
        assert "not instructions" in memory.context_block("en")


# --------------------------------------------------------------------------- #
# Degradacja: brak bazy nie może zatrzymać rozmowy
# --------------------------------------------------------------------------- #


def test_wylaczona_pamiec_nie_dotyka_dysku(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, memory_enabled=False)
    with ConversationMemory(settings) as memory:
        memory.add_user("cześć")
        assert not memory.persistent
        assert memory.remember_fact("imie", "Mariusz") is False
        assert memory.facts() == []
        assert memory.search("cokolwiek") == []
        assert memory.status_text == t("status.memory.off")
    assert not (tmp_path / "pamiec.sqlite3").exists()


def test_niedostepna_baza_zostawia_dzialajaca_rozmowe(tmp_path: Path) -> None:
    """Ścieżka bazy prowadzi „przez plik" — nie da się jej otworzyć na żadnym systemie."""
    przeszkoda = tmp_path / "to-jest-plik"
    przeszkoda.write_text("blokada", encoding="utf-8")
    settings = make_settings(tmp_path, database_path=str(przeszkoda / "podkatalog" / "b.sqlite3"))

    with ConversationMemory(settings) as memory:
        memory.add_user("rozmowa działa mimo braku bazy")
        assert not memory.persistent
        assert memory.error
        assert len(memory.history) == 1
        assert memory.status_text.startswith(t("status.memory.ram_only", reason=""))


def test_awaria_bazy_w_trakcie_nie_przerywa_rozmowy(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with ConversationMemory(settings) as memory:
        memory.add_user("pierwsza wiadomość")
        # Baza znika w trakcie pracy (odłączony dysk sieciowy, USB, sen systemu).
        memory._db.close()  # test celowo psuje wewnętrzny stan obiektu

        memory.add_user("druga wiadomość")
        assert len(memory.history) == 2
        assert memory.error
        assert memory.stats_line() == t("status.memory.stats_failed")


def test_python_bez_sqlite3_nie_blokuje_rozmowy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minimalne buildy Pythona bywają kompilowane bez modułu ``sqlite3``."""
    import sys

    monkeypatch.setitem(sys.modules, "database", None)
    with ConversationMemory(make_settings(tmp_path)) as memory:
        memory.add_user("rozmowa działa bez warstwy bazy")
        assert not memory.persistent
        assert len(memory.history) == 1
        assert "niedostępna" in memory.error


def test_wlasna_baza_nie_jest_zamykana_przez_pamiec(tmp_path: Path) -> None:
    """Bazę podaną z zewnątrz zamyka ten, kto ją otworzył (np. GUI z Fazy 10)."""
    with Database(tmp_path / "wspolna.sqlite3") as db:
        memory = ConversationMemory(make_settings(tmp_path), database=db)
        memory.add_user("cześć")
        memory.close()

        assert not db.closed
        assert db.messages.count() == 1


def test_status_opisuje_gdzie_mieszka_pamiec(tmp_path: Path) -> None:
    with ConversationMemory(make_settings(tmp_path)) as memory:
        assert "pamiec.sqlite3" in memory.status_text
        assert "v3/3" in memory.status_text
        assert "0 rozmów" not in memory.stats_line()


@pytest.mark.parametrize("jezyk", ["pl", "en"])
def test_streszczenie_uzywa_promptu_w_jezyku_rozmowy(tmp_path: Path, jezyk: str) -> None:
    backend = FakeBackend()
    with ConversationMemory(make_settings(tmp_path)) as memory:
        for numer in range(6):
            memory.add_user(f"wiadomość {numer}")
        run(memory.compact(backend, language=jezyk))

    system = backend.calls[0]["system"]
    assert ("memory module" in system) if jezyk == "en" else ("modułem pamięci" in system)
