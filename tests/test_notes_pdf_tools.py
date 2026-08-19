"""Testy narzędzi notatek i PDF (Faza 8).

Notatki działają na prawdziwej bazie SQLite w ``tmp_path`` z atrapą embeddingów —
sprawdzamy więc też to, co najłatwiej przeoczyć: że usunięcie notatki zabiera jej
wektor, a dopisanie akapitu wektor przelicza.

PDF-y są czytane **atrapą biblioteki**: żaden test nie wymaga zainstalowanego
``pypdf`` i żaden nie tworzy prawdziwego pliku PDF. Dzięki temu zestaw testów
przechodzi identycznie na maszynie z biblioteką i bez niej.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest
from conftest import FakeEmbeddingProvider

from brain.memory import ConversationMemory
from brain.vectorstore import SemanticMemory
from config import Settings
from database.database import Database
from host.paths import Workspace
from security.risk import RiskLevel
from tools.base import ToolContext, ToolError
from tools.notes import (
    NotesAppendArgs,
    NotesCreateArgs,
    NotesDeleteArgs,
    NotesReadArgs,
    NotesSearchArgs,
    build_notes_tools,
)
from tools.pdf import PdfReadArgs, PdfSearchArgs, build_pdf_tools, reader_backend


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {"database_path": str(tmp_path / "pamiec.sqlite3")}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def ctx(settings: Settings | None = None) -> ToolContext:
    active = settings or Settings(_env_file=None)  # type: ignore[call-arg]
    return ToolContext(settings=active, language="pl")


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --------------------------------------------------------------------------- #
# Notatki
# --------------------------------------------------------------------------- #


@pytest.fixture
def memory(tmp_path: Path):
    """Pamięć z prawdziwą bazą i atrapą embeddingów."""
    settings = make_settings(tmp_path)
    database = Database(tmp_path / "pamiec.sqlite3")
    semantic = SemanticMemory(database, settings, provider=FakeEmbeddingProvider())
    memory = ConversationMemory(settings, database=database, semantic=semantic)
    yield memory
    memory.close()
    database.close()


def notes_tools(memory: Any, tmp_path: Path) -> dict[str, Any]:
    built = build_notes_tools(make_settings(tmp_path), memory=memory)
    return {tool.spec.name: tool for tool in built}


def test_notatki_bez_pamieci_sa_niedostepne(tmp_path: Path) -> None:
    """Model nie widzi narzędzi, którymi nie da się nic zrobić."""
    narzedzia = notes_tools(None, tmp_path)
    for tool in narzedzia.values():
        usable, reason = tool.available()
        assert not usable and "pamięć" in reason.lower()


def test_poziomy_ryzyka_narzedzi_notatek(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    assert narzedzia["notes.search"].spec.risk is RiskLevel.SAFE
    assert narzedzia["notes.read"].spec.risk is RiskLevel.SAFE
    assert narzedzia["notes.create"].spec.risk is RiskLevel.MEDIUM
    assert narzedzia["notes.append"].spec.risk is RiskLevel.MEDIUM
    assert narzedzia["notes.delete"].spec.risk is RiskLevel.HIGH


def test_utworzenie_notatki_jest_od_razu_wyszukiwalne(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    settings = make_settings(tmp_path)

    wynik = run(
        narzedzia["notes.create"].run(
            NotesCreateArgs(body="Rower stoi w piwnicy", title="rower", tags=["dom"]),
            ctx(settings),
        )
    )

    assert wynik.ok
    note_id = wynik.data["id"]
    # ...po słowach
    znalezione = run(
        narzedzia["notes.search"].run(NotesSearchArgs(query="piwnicy"), ctx(settings))
    )
    assert any(item["id"] == note_id for item in znalezione.data["notes"])
    # ...i po znaczeniu (wektor powstał przy zapisie)
    assert memory.recall("gdzie stoi rower") != []


def test_odczyt_notatki_po_numerze(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    settings = make_settings(tmp_path)
    note = memory.add_note("treść notatki", title="tytuł")

    wynik = run(narzedzia["notes.read"].run(NotesReadArgs(note_id=note.id), ctx(settings)))

    assert wynik.data["title"] == "tytuł" and wynik.data["body"] == "treść notatki"


def test_odczyt_nieistniejacej_notatki_daje_czytelny_blad(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["notes.read"].run(NotesReadArgs(note_id=9999), ctx(make_settings(tmp_path))))
    assert "nie ma notatki" in blad.value.message


def test_dopisanie_nie_nadpisuje_i_przelicza_wektor(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    settings = make_settings(tmp_path)
    note = memory.add_note("pierwsza część", title="dziennik")

    run(
        narzedzia["notes.append"].run(
            NotesAppendArgs(note_id=note.id, text="druga część"), ctx(settings)
        )
    )

    zapisana = memory.note(note.id)
    assert "pierwsza część" in zapisana.body and "druga część" in zapisana.body
    # Wektor odpowiada nowej treści — szukanie po dopisanym fragmencie działa.
    assert any(hit.source_id == note.id for hit in memory.recall("druga część"))


def test_usuniecie_notatki_zabiera_tez_wektor(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    settings = make_settings(tmp_path)
    note = memory.add_note("tajny plan urodzinowy", title="urodziny")
    assert memory.recall("urodziny") != []

    wynik = run(
        narzedzia["notes.delete"].run(NotesDeleteArgs(note_id=note.id), ctx(settings))
    )

    assert wynik.ok
    assert memory.note(note.id) is None
    # Bez tego „zapomnij" zostawiałoby ślad w pamięci semantycznej.
    assert all(hit.source_id != note.id for hit in memory.recall("urodziny"))


def test_pytanie_o_zgode_na_usuniecie_pokazuje_tresc(memory: Any, tmp_path: Path) -> None:
    narzedzia = notes_tools(memory, tmp_path)
    note = memory.add_note("treść do usunięcia", title="do kosza")

    pytanie = narzedzia["notes.delete"].confirmation(
        NotesDeleteArgs(note_id=note.id), language="pl"
    )

    assert pytanie is not None and pytanie.risk is RiskLevel.HIGH
    assert "do kosza" in pytanie.summary
    assert pytanie.preview is not None and "treść do usunięcia" in pytanie.preview


def test_notatki_dzialaja_bez_dostepu_do_dysku_uzytkownika(memory: Any, tmp_path: Path) -> None:
    """Notatki nie potrzebują ani jednego dozwolonego katalogu plikowego."""
    narzedzia = notes_tools(memory, tmp_path)
    for tool in narzedzia.values():
        assert tool.available()[0]


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakePdfReader:
    """Atrapa ``PdfReader``: udaje dokument o zadanej treści stron."""

    instances: ClassVar[list[FakePdfReader]] = []

    def __init__(self, path: str) -> None:
        self.path = path
        FakePdfReader.instances.append(self)
        self.pages = [
            FakePage("Umowa najmu\nStrony ustalają czynsz 2000 zł"),
            FakePage("Termin płatności: 10. dzień miesiąca"),
            FakePage("Załącznik: protokół zdawczo-odbiorczy"),
        ]


class BrokenPdfReader:
    def __init__(self, path: str) -> None:
        raise ValueError("plik nie jest PDF-em")


class EmptyPdfReader:
    def __init__(self, path: str) -> None:
        self.pages = [FakePage("   "), FakePage("")]


@pytest.fixture
def pdf_root(tmp_path: Path) -> Path:
    root = tmp_path / "dokumenty"
    root.mkdir()
    # Treść nie ma znaczenia — czytanie jest podstawione atrapą.
    (root / "umowa.pdf").write_bytes(b"%PDF-1.4 udawany")
    (root / "notatka.txt").write_text("nie pdf", encoding="utf-8")
    return root


def pdf_tools(root: Path, *, reader: Any = FakePdfReader, **overrides: Any) -> dict[str, Any]:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
    built = build_pdf_tools(settings, workspace=Workspace.for_roots([root]), reader=reader)
    return {tool.spec.name: tool for tool in built}


def test_pdf_bez_biblioteki_jest_niedostepny(
    pdf_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brak pypdf = narzędzia nie ma, a nie „jest i się wywala"."""
    import tools.pdf

    monkeypatch.setattr(tools.pdf, "reader_backend", lambda: "")
    narzedzia = pdf_tools(pdf_root, reader=None)

    for tool in narzedzia.values():
        usable, reason = tool.available()
        assert not usable and "pypdf" in reason


def test_pdf_z_biblioteka_jest_dostepny(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root)
    assert all(tool.available()[0] for tool in narzedzia.values())
    assert all(tool.spec.risk is RiskLevel.SAFE for tool in narzedzia.values())


def test_odczyt_pdf_zwraca_tekst_stron(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root)
    wynik = run(narzedzia["pdf.read"].run(PdfReadArgs(path="umowa.pdf"), ctx()))  # type: ignore[call-arg]

    assert wynik.ok
    assert "czynsz 2000 zł" in wynik.data["text"]
    assert wynik.data["pages_total"] == 3
    # Treść dokumentu to dane z zewnątrz — router postawi po niej barierę.
    assert wynik.untrusted is True


def test_odczyt_pdf_respektuje_limit_stron(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root, pdf_max_pages=1)
    wynik = run(narzedzia["pdf.read"].run(PdfReadArgs(path="umowa.pdf"), ctx()))  # type: ignore[call-arg]

    assert wynik.data["pages_read"] == 1
    assert "Termin płatności" not in wynik.data["text"]


def test_odczyt_pdf_od_wskazanej_strony(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root)
    wynik = run(
        narzedzia["pdf.read"].run(
            PdfReadArgs(path="umowa.pdf", start_page=2, pages=1),
            ctx(),  # type: ignore[call-arg]
        )
    )
    assert "Termin płatności" in wynik.data["text"]
    assert "Umowa najmu" not in wynik.data["text"]


def test_szukanie_w_pdf_podaje_numery_stron(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root)
    wynik = run(
        narzedzia["pdf.search"].run(
            PdfSearchArgs(path="umowa.pdf", query="czynsz"),
            ctx(),  # type: ignore[call-arg]
        )
    )

    assert wynik.data["count"] == 1
    assert wynik.data["matches"][0]["page"] == 1
    assert "czynsz" in wynik.data["matches"][0]["excerpt"]


def test_pdf_poza_dozwolonym_katalogiem_jest_odrzucany(pdf_root: Path, tmp_path: Path) -> None:
    obcy = tmp_path / "obcy.pdf"
    obcy.write_bytes(b"%PDF")
    narzedzia = pdf_tools(pdf_root)

    with pytest.raises(ToolError) as blad:
        run(narzedzia["pdf.read"].run(PdfReadArgs(path=str(obcy)), ctx()))  # type: ignore[call-arg]
    assert "poza dozwolonymi katalogami" in blad.value.message


def test_plik_bez_rozszerzenia_pdf_jest_odrzucany(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["pdf.read"].run(PdfReadArgs(path="notatka.txt"), ctx()))  # type: ignore[call-arg]
    assert ".pdf" in blad.value.message


def test_uszkodzony_pdf_daje_czytelny_blad(pdf_root: Path) -> None:
    narzedzia = pdf_tools(pdf_root, reader=BrokenPdfReader)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["pdf.read"].run(PdfReadArgs(path="umowa.pdf"), ctx()))  # type: ignore[call-arg]
    assert "nie udało się otworzyć" in blad.value.message


def test_pdf_bez_tekstu_mowi_o_skanie(pdf_root: Path) -> None:
    """Skan bez OCR to częsty przypadek — komunikat ma to wyjaśniać."""
    narzedzia = pdf_tools(pdf_root, reader=EmptyPdfReader)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["pdf.read"].run(PdfReadArgs(path="umowa.pdf"), ctx()))  # type: ignore[call-arg]
    assert "skanem" in blad.value.message


def test_wykrywanie_biblioteki_pdf_nie_wymaga_jej_obecnosci() -> None:
    """Funkcja ma zwrócić nazwę albo pusty łańcuch — nigdy nie rzucać."""
    assert isinstance(reader_backend(), str)
