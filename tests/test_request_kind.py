"""Testy rozróżniania LOCAL REQUEST vs WEB REQUEST (Faza 9).

Rozpoznawanie jest czysto tekstowe, więc testy są proste i szybkie — i tak samo
działa w programie: bez modelu, bez sieci, bez dodatkowej tury rozmowy.

Dwie własności pilnowane tu najmocniej:

* pytanie o **świeże dane** dokłada do promptu zdanie „nie zgaduj, użyj narzędzia",
* gdy narzędzi sieciowych nie ma, prompt i komunikat dla użytkownika mówią wprost,
  że odpowiedź nie będzie bieżąca — a asystent działa dalej.
"""

from __future__ import annotations

import pytest

from brain.request_kind import RequestKind, classify, prompt_hint, user_notice


@pytest.mark.parametrize(
    "pytanie",
    [
        "Jaka jest dzisiaj pogoda we Wrocławiu?",
        "What's the weather like tomorrow?",
        "Co nowego w wiadomościach?",
        "Sprawdź aktualny kurs euro",
        "ile kosztuje teraz bitcoin",
        "Jakie są najnowsze newsy o Pythonie?",
        "Wyszukaj w internecie recenzje tego roweru",
        "Kto wygrał wczorajszy mecz?",
        "Jaka jest temperatura na zewnatrz",
        "Zobacz co jest na https://example.org/artykul",
        "Znajdz film na youtube o naprawie hamulcow",
    ],
)
def test_pytania_wymagajace_swiezych_danych(pytanie: str) -> None:
    assessment = classify(pytanie)
    assert assessment.kind is RequestKind.WEB, assessment
    assert assessment.needs_web


@pytest.mark.parametrize(
    "pytanie",
    [
        "Ile to jest dwa plus dwa?",
        "Napisz wiersz o rowerze",
        "Co zapamiętałaś o moim imieniu?",
        "Pokaż pliki w katalogu roboczym",
        "Zapamiętaj, że mieszkam we Wrocławiu",
        "Wyjaśnij, jak działa pamięć wirtualna",
        "Jak się nazywasz?",
    ],
)
def test_pytania_lokalne_nie_wymagaja_sieci(pytanie: str) -> None:
    assessment = classify(pytanie)
    assert assessment.kind is RequestKind.LOCAL, assessment
    assert not assessment.needs_web


def test_pytanie_mieszane_rozpoznaje_oba_watki() -> None:
    assessment = classify("Zapisz w notatce aktualny kurs euro")

    assert assessment.kind is RequestKind.MIXED
    assert assessment.needs_web and assessment.needs_local
    assert assessment.web_hits and assessment.local_hits


def test_puste_pytanie_jest_lokalne() -> None:
    assert classify("").kind is RequestKind.LOCAL
    assert classify("   ").kind is RequestKind.LOCAL


def test_rozpoznawanie_nie_zalezy_od_polskich_znakow() -> None:
    """Z rozpoznawania mowy wychodzi i „pogoda", i „pogode" bez ogonków."""
    assert classify("jaka jest pogoda").needs_web
    assert classify("jaka bedzie pogode dzis").needs_web
    assert classify("JAKA JEST POGODA?").needs_web


def test_opis_oceny_nadaje_sie_do_logu() -> None:
    assert "lokalne" in classify("napisz wiersz").describe()
    assert "świeżych danych" in classify("jaka jest pogoda").describe()


# --------------------------------------------------------------------------- #
# Podpowiedzi do promptu
# --------------------------------------------------------------------------- #


def test_pytanie_lokalne_nie_doklada_niczego_do_promptu() -> None:
    assessment = classify("napisz wiersz o rowerze")
    assert prompt_hint(assessment, language="pl") == ""
    assert prompt_hint(assessment, language="en", web_available=False) == ""


def test_prompt_mowi_wprost_zeby_nie_zgadywac() -> None:
    assessment = classify("jaka jest dzisiaj pogoda")

    polski = prompt_hint(assessment, language="pl", web_available=True)
    angielski = prompt_hint(assessment, language="en", web_available=True)

    assert "nie zgaduj" in polski.lower() and "narzędzia" in polski
    assert "do not guess" in angielski.lower() and "tool" in angielski


def test_prompt_bez_sieci_kaze_powiedziec_o_braku_dostepu() -> None:
    assessment = classify("co nowego w wiadomościach")

    polski = prompt_hint(assessment, language="pl", web_available=False)
    angielski = prompt_hint(assessment, language="en", web_available=False)

    assert "niedostępne" in polski and "nie jest to bieżąca" in polski
    assert "unavailable" in angielski and "not current" in angielski
    # W obu wariantach zakaz wymyślania liczb zostaje.
    assert "wymyślonych" in polski and "invent" in angielski


def test_prompt_dla_pytania_mieszanego_wspomina_oba_zrodla() -> None:
    hint = prompt_hint(classify("zapisz w notatce kurs euro"), language="pl")
    assert "notatki" in hint or "pliki" in hint
    assert "narzędzi lokalnych" in hint


# --------------------------------------------------------------------------- #
# Komunikat dla użytkownika
# --------------------------------------------------------------------------- #


def test_komunikat_pojawia_sie_tylko_gdy_sieci_brakuje() -> None:
    assessment = classify("jaka jest pogoda")

    assert user_notice(assessment, language="pl", web_available=True) == ""
    assert user_notice(classify("napisz wiersz"), language="pl", web_available=False) == ""

    notice = user_notice(assessment, language="pl", web_available=False)
    assert "wymaga danych z internetu" in notice
    assert "bez dostępu do sieci" in notice


def test_komunikat_jest_jednym_zdaniem_do_wypowiedzenia() -> None:
    """Ten tekst jest czytany na głos — bez nawiasów, skrótów i list."""
    for language in ("pl", "en"):
        notice = user_notice(classify("jaka jest pogoda"), language=language, web_available=False)
        assert notice and "(" not in notice and "•" not in notice
        assert notice.count("\n") == 0
