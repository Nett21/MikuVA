"""Testy języka odpowiedzi i promptu systemowego.

Sedno: **ustawiony język wiąże.** Przy ``LANGUAGE=en`` pytanie zadane po polsku
ma dostać odpowiedź po angielsku — inaczej ustawienie użytkownika byłoby tylko
sugestią, którą przewraca pierwsze polskie słowo w pytaniu. Rozpoznawanie języka
z wypowiedzi włącza się dopiero przy ``LANGUAGE=auto``.
"""

from __future__ import annotations

import pytest

import config
from brain.personality import (
    DEFAULT_LANGUAGE,
    build_system_prompt,
    detect_language,
    is_auto_language,
    normalize_language,
    resolve_reply_language,
)
from config import Settings, UserSettings, configured_reply_language


def make_settings(**overrides: object) -> Settings:
    # _env_file to argument pydantic-settings, nie pole modelu — stąd wyciszenie.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


# --------------------------------------------------------------------------- #
# Wybór języka odpowiedzi
# --------------------------------------------------------------------------- #


def test_domyslnym_jezykiem_jest_angielski() -> None:
    assert DEFAULT_LANGUAGE == "en"
    assert make_settings().language == "en"


@pytest.mark.parametrize(
    "pytanie",
    [
        "Ile mam zapisanych faktów?",
        "opowiedz mi o rowerze",
        "Czy pamiętasz, gdzie kupiłem rower?",
    ],
)
def test_ustawiony_angielski_wygrywa_z_polskim_pytaniem(pytanie: str) -> None:
    assert resolve_reply_language("en", pytanie) == "en"


def test_ustawiony_polski_wygrywa_z_angielskim_pytaniem() -> None:
    assert resolve_reply_language("pl", "what is the weather like today?") == "pl"


def test_auto_oddaje_decyzje_rozpoznawaniu() -> None:
    assert is_auto_language("auto") and is_auto_language("") and is_auto_language(None)
    assert resolve_reply_language("auto", "Ile mam zapisanych faktów?") == "pl"
    assert resolve_reply_language("auto", "what can you do for me?") == "en"
    # Bez treści zostaje język domyślny.
    assert resolve_reply_language("auto", "") == DEFAULT_LANGUAGE


def test_nieobslugiwany_kod_schodzi_do_domyslnego() -> None:
    assert resolve_reply_language("de", "Wie geht es?") == DEFAULT_LANGUAGE
    assert normalize_language("PL-pl") == "pl"
    assert normalize_language(None) == DEFAULT_LANGUAGE


def test_walidator_ustawien_rozpoznaje_tryb_auto() -> None:
    """„auto" to tryb, nie kod języka — nie wolno go obciąć do dwóch znaków."""
    assert make_settings(language="AUTO").language == "auto"
    assert make_settings(language="en-US").language == "en"
    assert make_settings(language="  ").language == "en"


def test_rozpoznawanie_jezyka_dziala_dalej_dla_trybu_auto() -> None:
    assert detect_language("cześć, co robisz?") == "pl"
    assert detect_language("hello, what are you doing?") == "en"


# --------------------------------------------------------------------------- #
# Prompt systemowy
# --------------------------------------------------------------------------- #


def test_prompt_kaze_trzymac_sie_ustawionego_jezyka() -> None:
    user = UserSettings(assistant_name="Miku")

    angielski = build_system_prompt(user, language="en")
    assert "ALWAYS reply in English" in angielski

    polski = build_system_prompt(user, language="pl")
    assert "ZAWSZE po polsku" in polski


def test_prompt_dla_trybu_auto_pozwala_przelaczac_jezyk() -> None:
    user = UserSettings(assistant_name="Miku")

    angielski = build_system_prompt(user, language="en", lock_language=False)
    assert "switch back when they do" in angielski
    assert "ALWAYS reply in English" not in angielski

    polski = build_system_prompt(user, language="pl", lock_language=False)
    assert "chyba że użytkownik" in polski


def test_prompt_nie_ma_wpisanego_imienia_na_sztywno() -> None:
    prompt = build_system_prompt(UserSettings(assistant_name="Zosia"), language="en")
    assert "Zosia" in prompt and "Miku" not in prompt


# --------------------------------------------------------------------------- #
# Język mowy a język odpowiedzi to dwie różne rzeczy
# --------------------------------------------------------------------------- #


def test_jawnie_ustawiony_language_wygrywa_z_jezykiem_mowy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zgłoszone z użycia: „mówię po pl, wykrywa, lecz nie odpowiada po ang".

    Użytkownik mówi po polsku i po angielsku (``speech_language`` = „pl,en"),
    a odpowiedzi chce po angielsku (``LANGUAGE=en``). Wcześniej ``speech_language``
    szło wprost jako język odpowiedzi — a „pl,en" nie jest kodem języka, więc
    prompt schodził na polski.
    """
    monkeypatch.setattr(config, "get_user_settings", lambda: UserSettings(speech_language="pl,en"))
    settings = Settings(_env_file=None, language="en")

    assert configured_reply_language(settings) == "en"


def test_bez_jawnego_language_decyduje_jezyk_mowy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kto ustawił tylko ``speech_language`` i nie dotknął .env, nie traci polskiego."""
    monkeypatch.setattr(config, "get_user_settings", lambda: UserSettings(speech_language="pl"))
    settings = Settings(_env_file=None)
    settings.model_fields_set.discard("language")

    assert configured_reply_language(settings) == "pl"


def test_lista_jezykow_nigdy_nie_wychodzi_jako_kod(monkeypatch: pytest.MonkeyPatch) -> None:
    """„pl,en" opisuje MOWĘ. Jako język odpowiedzi byłoby śmieciem w prompcie."""
    monkeypatch.setattr(config, "get_user_settings", lambda: UserSettings(speech_language="pl,en"))
    settings = Settings(_env_file=None, language="pl")
    settings.model_fields_set.discard("language")

    assert configured_reply_language(settings) in ("pl", "en")
    assert "," not in configured_reply_language(settings)


def test_prompt_dla_polskiego_pytania_zostaje_angielski(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sprawdzenie do końca: od ustawienia aż po treść promptu systemowego."""
    monkeypatch.setattr(config, "get_user_settings", lambda: UserSettings(speech_language="pl,en"))
    settings = Settings(_env_file=None, language="en")

    ustawiony = configured_reply_language(settings)
    jezyk = resolve_reply_language(ustawiony, "jaka jest pogoda w Warszawie")
    prompt = build_system_prompt(UserSettings(assistant_name="Miku"), language=jezyk)

    assert jezyk == "en"
    assert prompt.startswith("You are a voice assistant")
