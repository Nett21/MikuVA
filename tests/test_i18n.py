"""Testy języka interfejsu (Faza 10).

Interfejs jest domyślnie **angielski**, a polski to pełnoprawny wariant. Trzy
rzeczy, które muszą być pewne, bo inaczej użytkownik zobaczy pusty przycisk albo
klucz zamiast zdania:

* oba katalogi mają **ten sam zestaw kluczy** i żadnego pustego napisu,
* wstawki (``{name}``, ``{model}``) zgadzają się między językami — inaczej
  tłumaczenie wywaliłoby się na formatowaniu,
* brak klucza albo brak tłumaczenia **nie wywraca** interfejsu.

Do tego sprawdzamy wybór trybu uruchomienia: ``--gui``/``--no-gui`` i ustawienie
``GUI_ENABLED``.
"""

from __future__ import annotations

import re

import pytest

import i18n
from config import Settings
from i18n import (
    DEFAULT_UI_LANGUAGE,
    SUPPORTED_UI_LANGUAGES,
    catalog,
    normalize_ui_language,
    set_ui_language,
    t,
    translate_or_text,
    ui_language,
)

PLACEHOLDER = re.compile(r"\{(\w+)\}")


# --------------------------------------------------------------------------- #
# Spójność katalogów
# --------------------------------------------------------------------------- #


def test_domyslny_jezyk_interfejsu_to_angielski() -> None:
    assert DEFAULT_UI_LANGUAGE == "en"
    assert ui_language() == "en"


def test_oba_katalogi_maja_te_same_klucze() -> None:
    english = set(catalog("en"))
    polish = set(catalog("pl"))

    assert english == polish, (
        f"brak po polsku: {sorted(english - polish)}; "
        f"nadmiarowe po polsku: {sorted(polish - english)}"
    )


@pytest.mark.parametrize("language", SUPPORTED_UI_LANGUAGES)
def test_zaden_napis_nie_jest_pusty(language: str) -> None:
    puste = [key for key, value in catalog(language).items() if not value.strip()]
    assert not puste, f"puste teksty w katalogu {language}: {puste}"


def test_wstawki_zgadzaja_sie_miedzy_jezykami() -> None:
    """``{name}`` po jednej stronie i ``{imie}`` po drugiej = błąd formatowania."""
    english, polish = catalog("en"), catalog("pl")
    rozjazd = {
        key: (sorted(PLACEHOLDER.findall(value)), sorted(PLACEHOLDER.findall(polish[key])))
        for key, value in english.items()
        if set(PLACEHOLDER.findall(value)) != set(PLACEHOLDER.findall(polish[key]))
    }
    assert not rozjazd, f"różne wstawki: {rozjazd}"


# --------------------------------------------------------------------------- #
# Wybór języka
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [("en", "en"), ("pl", "pl"), ("PL", "pl"), ("pl-PL", "pl"), ("de", "en"), ("", "en")],
)
def test_normalizacja_kodu_jezyka(value: str, expected: str) -> None:
    assert normalize_ui_language(value) == expected


def test_auto_idzie_za_jezykiem_odpowiedzi() -> None:
    """``UI_LANGUAGE=auto`` oszczędza drugiego wpisu w .env."""
    assert normalize_ui_language("auto", reply_language="pl") == "pl"
    assert normalize_ui_language("auto", reply_language="en") == "en"
    # „auto" po obu stronach (LANGUAGE=auto) schodzi do wzorcowego angielskiego.
    assert normalize_ui_language("auto", reply_language="auto") == "en"


def test_ustawienie_jezyka_zmienia_teksty() -> None:
    set_ui_language("pl")
    assert ui_language() == "pl"
    assert t("gui.send") == "Wyślij"

    set_ui_language("en")
    assert t("gui.send") == "Send"


def test_jezyk_mozna_wymusic_dla_jednego_tekstu() -> None:
    """Potrzebne dla rzeczy MÓWIONYCH: te idą językiem odpowiedzi, nie interfejsu."""
    set_ui_language("en")
    assert t("cli.speech.sample", _lang="pl", name="Miku").startswith("Cześć")
    assert t("gui.send") == "Send"


# --------------------------------------------------------------------------- #
# Odporność
# --------------------------------------------------------------------------- #


def test_brakujacy_klucz_nie_wywraca_interfejsu() -> None:
    assert t("nie.ma.takiego.klucza") == "nie.ma.takiego.klucza"


def test_brak_tlumaczenia_pokazuje_wersje_angielska(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(i18n._EN, "test.tylko.po.angielsku", "English only")
    set_ui_language("pl")

    assert t("test.tylko.po.angielsku") == "English only"


def test_zla_liczba_wstawek_nie_rzuca(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(i18n._EN, "test.wstawka", "Hello {name}")

    # Brakujący parametr: pokazujemy surowy wzorzec zamiast wywalać okno.
    assert t("test.wstawka") == "Hello {name}"
    assert t("test.wstawka", name="Miku") == "Hello Miku"


def test_tekst_ktory_nie_jest_kluczem_zostaje_bez_zmian() -> None:
    """``purpose`` w rejestrze pakietów bywa kluczem, a bywa zwykłym zdaniem."""
    assert translate_or_text("deps.purpose.httpx") == t("deps.purpose.httpx")
    assert translate_or_text("dowolny opis wtyczki") == "dowolny opis wtyczki"


def test_dopisanie_tekstow_nie_nadpisuje_rdzenia() -> None:
    i18n.register_texts("en", {"gui.send": "PODMIENIONE", "test.wtyczka": "Plugin text"})

    assert t("gui.send") == "Send"
    assert t("test.wtyczka") == "Plugin text"


# --------------------------------------------------------------------------- #
# Tryb uruchomienia: okno czy terminal
# --------------------------------------------------------------------------- #


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


def resolve_gui(args_gui: bool, args_no_gui: bool, enabled: bool) -> bool:
    """Ta sama reguła co w ``main.main`` — flaga wygrywa z ustawieniem."""
    return (args_gui or enabled) and not args_no_gui


@pytest.mark.parametrize(
    ("flag_gui", "flag_no_gui", "setting", "expected"),
    [
        (False, False, False, False),  # GUI_ENABLED=false → terminal
        (True, False, False, True),  # --gui
        (False, False, True, True),  # GUI_ENABLED=true
        (False, True, True, False),  # --no-gui wygrywa z ustawieniem
        (True, True, False, False),  # --no-gui wygrywa też z --gui (main odrzuca wcześniej)
    ],
)
def test_wybor_okna_albo_terminala(
    flag_gui: bool, flag_no_gui: bool, setting: bool, expected: bool
) -> None:
    assert resolve_gui(flag_gui, flag_no_gui, setting) is expected


def test_okno_jest_domyslnym_interfejsem() -> None:
    """Od poprawek po Fazie 10 `python main.py` otwiera okno, nie terminal."""
    settings = make_settings()
    assert settings.gui_enabled is True
    assert settings.ui_language == "en"


def test_literowka_w_ui_language_nie_blokuje_startu() -> None:
    """Zły kod ma dać angielskie napisy, a nie odmowę uruchomienia."""
    settings = make_settings(ui_language="klingoński")
    assert normalize_ui_language(settings.ui_language) == "en"


def test_parser_zna_obie_flagi_trybu() -> None:
    import main

    args = main.build_parser().parse_args(["--no-gui"])
    assert args.no_gui is True and args.gui is False

    args = main.build_parser().parse_args(["--gui", "--ui-lang", "pl"])
    assert args.gui is True and args.ui_lang == "pl"


def test_sprzeczne_flagi_koncza_sie_komunikatem(capsys: pytest.CaptureFixture[str]) -> None:
    import main

    assert main.main(["--gui", "--no-gui"]) == main.EXIT_CONFIG_ERROR
    assert t("cli.main.gui_conflict") in capsys.readouterr().out


def test_zaden_klucz_nie_jest_zdefiniowany_dwa_razy() -> None:
    """Powtórzony klucz w słowniku literalnym Pythona przechodzi BEZ ŚLADU.

    Druga definicja po cichu nadpisuje pierwszą, a `len(_EN) == len(_PL)`
    dalej się zgadza — więc test porównujący zestawy kluczy tego nie łapie.
    Efektem jest komunikat, który w kodzie wygląda inaczej niż na ekranie.
    Dlatego czytamy sam PLIK i liczymy wystąpienia.
    """
    import ast
    import collections

    from config import PROJECT_ROOT

    drzewo = ast.parse((PROJECT_ROOT / "i18n.py").read_text(encoding="utf-8"))
    duplikaty: dict[str, list[str]] = {}
    for wezel in ast.walk(drzewo):
        if not isinstance(wezel, ast.AnnAssign) or not isinstance(wezel.value, ast.Dict):
            continue
        if not isinstance(wezel.target, ast.Name) or wezel.target.id not in {"_EN", "_PL"}:
            continue
        klucze = [
            k.value for k in wezel.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
        powtorzone = [k for k, ile in collections.Counter(klucze).items() if ile > 1]
        if powtorzone:
            duplikaty[wezel.target.id] = sorted(powtorzone)

    assert not duplikaty, f"klucze zdefiniowane więcej niż raz: {duplikaty}"
