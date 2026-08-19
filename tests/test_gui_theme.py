"""Testy motywu GUI liczonego z ``ui_accent_color`` (Faza 10).

Te testy nie potrzebują ekranu, tkintera ani CustomTkintera — i to jest celowe:
najważniejsza obietnica Fazy 10 („zmiana jednego pola realnie zmienia wygląd")
jest własnością **czystej funkcji**, więc daje się sprawdzić wszędzie.

Dwie rzeczy pilnowane najmocniej:

* każdy kolor interfejsu **zmienia się** po zmianie akcentu — inaczej gdzieś w
  kodzie zostałaby zaszyta wartość,
* tekst na kolorowym tle jest **czytelny** przy dowolnym akcencie, także skrajnie
  jasnym albo skrajnie ciemnym.
"""

from __future__ import annotations

import re

import pytest

from gui.theme import (
    DEFAULT_ACCENT,
    Metrics,
    Theme,
    build_palette,
    contrast_ratio,
    normalize_accent,
    parse_color,
    pick_font_family,
    readable_text_on,
    to_hex,
)

HEX = re.compile(r"^#[0-9a-f]{6}$")

# Akcenty dobrane pod skrajności: bardzo jasny, bardzo ciemny, nasycony, szary.
ACCENTS = ("#39C5BB", "#FFF176", "#0B1030", "#FF00AA", "#808080", "#FFFFFF", "#000000")


# --------------------------------------------------------------------------- #
# Odczyt koloru
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#39C5BB", (57, 197, 187)),
        ("39c5bb", (57, 197, 187)),
        ("#abc", (170, 187, 204)),
        ("  #FFFFFF  ", (255, 255, 255)),
    ],
)
def test_odczyt_koloru(value: str, expected: tuple[int, int, int]) -> None:
    assert parse_color(value) == expected


@pytest.mark.parametrize("value", ["", None, "zielony", "#12345", "#gggggg", "39C5BB99"])
def test_wartosc_ktora_nie_jest_kolorem(value: str | None) -> None:
    assert parse_color(value) is None


def test_bledny_kolor_uzytkownika_nie_wywraca_motywu() -> None:
    """Pole edytowane ręcznie bywa błędne — motyw wraca wtedy do domyślnego."""
    assert normalize_accent("nie-kolor") == DEFAULT_ACCENT.lower()
    assert normalize_accent("") == DEFAULT_ACCENT.lower()
    assert normalize_accent("#3CB") == "#33ccbb"


# --------------------------------------------------------------------------- #
# Paleta
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("mode", ["dark", "light"])
def test_paleta_ma_same_poprawne_kolory(accent: str, mode: str) -> None:
    palette = build_palette(accent, mode)  # type: ignore[arg-type]
    for name, value in palette.as_dict().items():
        assert HEX.match(value), f"{name}={value!r} nie jest kolorem #rrggbb"


def test_zmiana_akcentu_zmienia_caly_interfejs() -> None:
    """Sedno wymagania: po zmianie akcentu nie może zostać ANI JEDEN stary kolor.

    Gdyby któryś element trzymał własną kopię koloru, ten test by go wskazał —
    porównujemy pełne palety, nie wybrane pola.
    """
    first = build_palette("#39C5BB", "dark").as_dict()
    second = build_palette("#FF6600", "dark").as_dict()

    identyczne = {name for name, value in first.items() if second[name] == value}
    # Kolory tekstu to czerń albo biel wybierana kontrastem — mogą wypaść tak samo
    # dla obu akcentów i to jest poprawne. Wszystko inne musi się zmienić.
    dozwolone = {
        "accent_text",
        "user_text",
        "assistant_text",
        "error_text",
    }
    assert identyczne <= dozwolone, f"te kolory nie zależą od akcentu: {identyczne - dozwolone}"


def test_bable_czatu_i_wskaznik_biora_kolor_z_akcentu() -> None:
    palette = build_palette("#FF6600", "dark")
    assert palette.user_bubble == "#ff6600"
    # Wskaźnik nasłuchiwania jest wzmocnionym akcentem, nie osobnym kolorem.
    assert palette.listening_active != palette.listening_idle
    assert parse_color(palette.listening_active) is not None


def test_paleta_jest_powtarzalna() -> None:
    """Ten sam akcent zawsze daje tę samą paletę (bez losowości i bez zegara)."""
    assert build_palette("#39C5BB", "dark") == build_palette("#39c5bb", "dark")


def test_tryb_jasny_i_ciemny_to_inne_tla() -> None:
    dark = build_palette("#39C5BB", "dark")
    light = build_palette("#39C5BB", "light")
    assert dark.background != light.background
    assert dark.is_dark and not light.is_dark


@pytest.mark.parametrize("accent", ACCENTS)
def test_napis_na_akcencie_jest_czytelny(accent: str) -> None:
    """Kontrast liczony, nie zgadywany — także dla żółtego i dla czerni."""
    palette = build_palette(accent, "dark")
    ratio = contrast_ratio(
        parse_color(palette.user_bubble) or (0, 0, 0),
        parse_color(palette.user_text) or (255, 255, 255),
    )
    assert ratio >= 3.0, f"akcent {accent} daje kontrast {ratio:.2f}"


def test_wybor_koloru_napisu_na_tle() -> None:
    assert to_hex(readable_text_on((10, 10, 20))) == "#ffffff"
    assert to_hex(readable_text_on((250, 250, 200))) == "#18181b"


# --------------------------------------------------------------------------- #
# Czcionki i motyw
# --------------------------------------------------------------------------- #


def test_czcionka_z_ustawien_tylko_gdy_system_ja_zna() -> None:
    """Nazwa nieistniejącego kroju to cichy zastępnik o innych metrykach — unikamy."""
    families = ("DejaVu Sans", "Courier New")
    assert pick_font_family(families, preferred="DejaVu Sans") == "DejaVu Sans"
    assert pick_font_family(families, preferred="Segoe UI") == "DejaVu Sans"


def test_czcionka_bez_zadnego_kandydata_oddaje_decyzje_toolkitowi() -> None:
    assert pick_font_family(("Jakis Dziwny Krój",)) == ""
    # Puste „available" znaczy „nie wiem, co jest zainstalowane" — wtedy wybór
    # użytkownika przechodzi bez zmian, bo nie mamy podstaw go odrzucić.
    assert pick_font_family((), preferred="Segoe UI") == "Segoe UI"


def test_wielkosc_liter_nie_ma_znaczenia_dla_nazwy_kroju() -> None:
    assert pick_font_family(("dejavu sans",), preferred="DejaVu Sans") == "DejaVu Sans"
    assert pick_font_family(("DEJAVU SANS",)) == "DEJAVU SANS"


def test_motyw_z_nowym_akcentem_zachowuje_reszte() -> None:
    theme = Theme.build("#39C5BB", mode="light", font_family="DejaVu Sans", metrics=Metrics(pad=20))
    changed = theme.with_accent("#FF6600")

    assert changed.mode == "light"
    assert changed.font_family == "DejaVu Sans"
    assert changed.metrics.pad == 20
    assert changed.accent == "#ff6600"


def test_motyw_zmienia_tryb_bez_gubienia_akcentu() -> None:
    theme = Theme.build("#FF6600", mode="dark")
    assert theme.with_mode("light").accent == theme.accent
    assert theme.with_mode("light").mode == "light"
