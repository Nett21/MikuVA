"""Testy panelu ustawień: co i gdzie zapisuje (Faza 10).

Najważniejsze pytanie, na które odpowiadają te testy: **czy zapis z jednego
ekranu nie kasuje ustawień, których ten ekran nie pokazuje?** Panel edytuje pięć
rzeczy, a plik ``config/user_settings.json`` zawiera kilkanaście — łatwo o wersję,
która „zapisuje formularz", czyli po cichu zeruje frazę wybudzającą i mapę głosów.

Drugie pytanie: czy panel na pewno pisze do ``user_settings.json``, a nie do
``.env``. To nie jest szczegół: ``.env`` opisuje infrastrukturę i bywa
współdzielony między maszynami, a imię asystenta czy kolor to preferencja
człowieka siedzącego przed tym konkretnym komputerem.

Testy nie potrzebują ekranu — cała logika panelu siedzi w ``gui/settings_form.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT, UserSettings
from gui.settings_form import (
    FORM_FIELDS,
    LIVE_KEYS,
    ChoiceOptions,
    SettingsForm,
    build_payload,
    coerce,
    field_by_key,
    file_request,
    relativize_path,
    validate,
)
from i18n import t

# Plik ustawień z rzeczami, których panel NIE edytuje — łącznie z kluczem,
# którego model w ogóle nie zna.
BASE_FILE = {
    "assistant_name": "Miku",
    "ui_accent_color": "#39C5BB",
    "personality_traits": "",
    "wake_word": "hej komputerze",
    "wake_word_model": "modele/hej.onnx",
    "speech_language": "pl",
    "voice_engine": "piper",
    "piper_model": "pl_PL-pierwszy-medium",
    "piper_voices": {"pl": "pl_PL-pierwszy-medium", "en": "en_US-amy-medium"},
    "piper_speaker": 3,
    "voice_speed": 1.2,
    "voice_volume": 0.7,
    "rvc": {
        "enabled": False,
        "model_path": "",
        "index_path": "",
        "pitch_shift": 0,
        "index_rate": 0.75,
    },
    "moje_wlasne_pole": "nie ruszaj",
}


@pytest.fixture
def settings_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Plik ustawień w katalogu tymczasowym (nigdy prawdziwy plik dewelopera)."""
    import config

    target = tmp_path / "user_settings.json"
    target.write_text(json.dumps(BASE_FILE, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(config, "USER_SETTINGS_FILE", target)
    monkeypatch.setattr(config, "_user_settings_cache", None, raising=False)
    monkeypatch.setattr(config, "_user_settings_mtime", None, raising=False)
    return target


@pytest.fixture
def form(settings_path: Path) -> SettingsForm:
    return SettingsForm(UserSettings.model_validate(BASE_FILE), path=settings_path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Zapis
# --------------------------------------------------------------------------- #


def test_zapis_trafia_do_pliku_ustawien_uzytkownika(
    form: SettingsForm, settings_path: Path
) -> None:
    form.set("assistant_name", "Aiko")
    form.set("ui_accent_color", "#FF6600")

    result = form.save()

    assert result.ok
    saved = read(settings_path)
    assert saved["assistant_name"] == "Aiko"
    assert saved["ui_accent_color"] == "#FF6600"


def test_zapis_nie_kasuje_pol_ktorych_ten_ekran_nie_edytuje(
    form: SettingsForm, settings_path: Path
) -> None:
    """Fraza wybudzająca, mapa głosów i obce klucze muszą przetrwać zapis."""
    form.set("assistant_name", "Aiko")
    form.save()

    saved = read(settings_path)
    assert saved["wake_word"] == "hej komputerze"
    assert saved["wake_word_model"] == "modele/hej.onnx"
    assert saved["piper_voices"] == {"pl": "pl_PL-pierwszy-medium", "en": "en_US-amy-medium"}
    assert saved["piper_speaker"] == 3
    assert saved["voice_speed"] == 1.2
    assert saved["speech_language"] == "pl"
    # Klucz, którego model nie zna, też zostaje — plik należy do użytkownika.
    assert saved["moje_wlasne_pole"] == "nie ruszaj"


def test_zapis_nie_tworzy_ani_nie_rusza_pliku_env(
    form: SettingsForm, settings_path: Path, tmp_path: Path
) -> None:
    """Ustawienia użytkownika NIE mają prawa wylądować w .env."""
    form.set("assistant_name", "Aiko")
    form.set("rvc.pitch_shift", 5)
    form.save()

    assert not list(tmp_path.rglob(".env*"))
    # Zapis dotyka wyłącznie pliku ustawień użytkownika — żaden inny plik nie
    # powstaje, a klucze mają postać z JSON-a, nie ze zmiennych środowiskowych.
    saved = read(settings_path)
    assert all(key == key.lower() for key in saved)
    assert saved["rvc"]["pitch_shift"] == 5


def test_zapisuja_sie_tylko_zmienione_pola(form: SettingsForm, settings_path: Path) -> None:
    form.set("ui_accent_color", "#FF6600")
    result = form.save()

    assert result.changed == ("ui_accent_color",)
    assert read(settings_path)["assistant_name"] == "Miku"


def test_brak_zmian_nie_dotyka_pliku(form: SettingsForm, settings_path: Path) -> None:
    before = settings_path.read_bytes()
    result = form.save()

    assert result.ok and result.changed == ()
    assert result.message() == t("settings.result.nothing")
    assert settings_path.read_bytes() == before


def test_przywrocenie_cofa_zmiany_bez_zapisu(form: SettingsForm, settings_path: Path) -> None:
    form.set("assistant_name", "Aiko")
    form.revert()

    assert form.value("assistant_name") == "Miku"
    assert form.changed_keys() == ()
    assert read(settings_path)["assistant_name"] == "Miku"


# --------------------------------------------------------------------------- #
# Walidacja
# --------------------------------------------------------------------------- #


def test_bledny_kolor_blokuje_zapis_i_tlumaczy_dlaczego(
    form: SettingsForm, settings_path: Path
) -> None:
    form.set("ui_accent_color", "zielony")
    result = form.save()

    assert not result.ok
    assert result.message() == t("settings.problem.color")
    assert read(settings_path)["ui_accent_color"] == "#39C5BB"


def test_puste_imie_asystenta_nie_przechodzi(form: SettingsForm) -> None:
    form.set("assistant_name", "   ")
    result = form.save()

    assert not result.ok
    assert any(problem.key.endswith("assistant_name") for problem in result.problems)


def test_nieistniejacy_plik_rvc_to_ostrzezenie_a_nie_blokada(
    form: SettingsForm, settings_path: Path, tmp_path: Path
) -> None:
    """Ścieżkę wolno zapisać „na zapas", ale użytkownik ma wiedzieć, że pliku nie ma."""
    form.set("rvc.model_path", str(tmp_path / "nie-ma-mnie.pth"))
    result = form.save()

    assert result.ok
    # Ostrzeżenie dotyczy TEGO pola i nie blokuje zapisu — sprawdzamy zachowanie,
    # a nie brzmienie zdania, bo interfejs bywa w różnych językach.
    assert any(
        problem.key == "rvc.model_path" and not problem.blocking
        for problem in result.problems
    )
    assert result.warnings
    assert read(settings_path)["rvc"]["model_path"]


def test_wlaczone_rvc_bez_modelu_ostrzega(form: SettingsForm) -> None:
    form.set("rvc.enabled", True)
    result = form.save()

    assert result.ok
    assert t("settings.problem.rvc_without_model") in result.warnings


def test_zle_rozszerzenie_pliku_jest_zauwazone(form: SettingsForm, tmp_path: Path) -> None:
    wrong = tmp_path / "glos.onnx"
    wrong.write_bytes(b"x")
    form.set("rvc.model_path", str(wrong))

    problems = form.problems()

    assert any(".pth" in problem.message and not problem.blocking for problem in problems)


def test_wartosci_liczbowe_sa_przycinane_do_zakresu(form: SettingsForm) -> None:
    form.set("rvc.pitch_shift", 999)
    form.set("rvc.index_rate", -3)

    assert form.value("rvc.pitch_shift") == 24
    assert form.value("rvc.index_rate") == 0.0
    assert form.save().ok


# --------------------------------------------------------------------------- #
# Normalizacja wartości z widgetów
# --------------------------------------------------------------------------- #


def test_kolor_bez_kratki_jest_przyjmowany() -> None:
    spec = field_by_key("ui_accent_color")
    assert spec is not None
    assert coerce(spec, "39C5BB") == "#39C5BB"
    # Coś, co nie jest kolorem, zostaje bez zmian — walidacja to odrzuci z sensownym
    # komunikatem zamiast doklejać kratkę do bzdury.
    assert coerce(spec, "zielony") == "zielony"


def test_liczby_przyjmuja_przecinek_dziesietny() -> None:
    spec = field_by_key("rvc.index_rate")
    assert spec is not None
    assert coerce(spec, "0,5") == 0.5


def test_sciezka_w_projekcie_zapisuje_sie_wzglednie() -> None:
    """Model w katalogu projektu ma się znaleźć także po przeniesieniu na inny komputer."""
    inside = PROJECT_ROOT / "models" / "rvc" / "glos.pth"
    relative = relativize_path(str(inside))

    assert relative == "models/rvc/glos.pth"
    assert "\\" not in relative


def test_sciezka_spoza_projektu_zostaje_bezwzgledna(tmp_path: Path) -> None:
    outside = tmp_path / "glos.pth"
    assert relativize_path(str(outside)) == str(outside.resolve())


def test_sciezka_w_cudzyslowie_jest_czyszczona(tmp_path: Path) -> None:
    """Przeciągnięcie pliku do terminala/pola wkleja ścieżkę w cudzysłowie."""
    outside = tmp_path / "glos.pth"
    assert relativize_path(f'"{outside}"') == str(outside.resolve())


def test_budowa_zapisu_zagniezdza_pola_rvc() -> None:
    payload = build_payload({"assistant_name": "Aiko", "rvc.pitch_shift": 2, "rvc.index_rate": 0.5})

    assert payload == {"assistant_name": "Aiko", "rvc": {"pitch_shift": 2, "index_rate": 0.5}}


# --------------------------------------------------------------------------- #
# Co działa od razu, a co wymaga przeładowania mowy
# --------------------------------------------------------------------------- #


def test_imie_kolor_i_cechy_dzialaja_natychmiast(form: SettingsForm) -> None:
    form.set("assistant_name", "Aiko")
    form.set("ui_accent_color", "#FF6600")
    form.set("personality_traits", "mówi krótko")

    result = form.save()

    assert result.ok
    assert set(result.applied_live) == {"assistant_name", "ui_accent_color", "personality_traits"}
    assert not result.needs_tts_reload
    labels = ", ".join(
        field_by_key(key).label for key in result.changed  # type: ignore[union-attr]
    )
    assert result.message() == t("settings.result.saved", fields=labels)


def test_zmiana_glosu_wymaga_przeladowania_mowy(form: SettingsForm) -> None:
    form.set("piper_model", "pl_PL-drugi-medium")
    result = form.save()

    assert result.ok and result.needs_tts_reload
    assert result.message() == t(
        "settings.result.saved_reload",
        fields=field_by_key("piper_model").label,  # type: ignore[union-attr]
    )


@pytest.mark.parametrize(
    "key", ["rvc.model_path", "rvc.index_path", "rvc.pitch_shift", "rvc.index_rate", "voice_engine"]
)
def test_kazde_pole_mowy_jest_oznaczone_jako_wymagajace_przeladowania(key: str) -> None:
    spec = field_by_key(key)
    assert spec is not None and spec.reload_tts and not spec.live


def test_pola_natychmiastowe_zgadzaja_sie_z_definicja() -> None:
    """Lista „działa od razu" i oznaczenia pól nie mogą się rozjechać."""
    from_specs = {spec.key for spec in FORM_FIELDS if spec.live}
    assert from_specs == LIVE_KEYS


# --------------------------------------------------------------------------- #
# Okno wyboru pliku
# --------------------------------------------------------------------------- #


def test_okno_wyboru_startuje_od_katalogu_obecnego_pliku(tmp_path: Path) -> None:
    existing = tmp_path / "glosy" / "moj.pth"
    existing.parent.mkdir()
    existing.write_bytes(b"x")

    request = file_request("rvc.model_path", str(existing))

    assert request.initial_dir == str(existing.parent)
    assert request.initial_file == "moj.pth"
    assert (t("settings.filter.rvc_model"), "*.pth") in request.filetypes


def test_okno_wyboru_bez_wskazanego_pliku_nie_wymysla_sciezek() -> None:
    """Katalog startowy musi ISTNIEĆ — inaczej lepiej oddać decyzję systemowi."""
    request = file_request("rvc.index_path", "")

    assert request.initial_file == ""
    if request.initial_dir:
        assert Path(request.initial_dir).is_dir()


def test_argumenty_okna_pomijaja_puste_wartosci() -> None:
    kwargs = file_request("rvc.model_path", "").as_dialog_kwargs()

    assert "title" in kwargs
    assert "initialfile" not in kwargs
    assert all(value != "" for value in kwargs.values())


# --------------------------------------------------------------------------- #
# Listy wyboru
# --------------------------------------------------------------------------- #


def test_lista_glosow_zawiera_wariant_automatyczny() -> None:
    spec = field_by_key("piper_model")
    assert spec is not None
    options = ChoiceOptions(piper_voices=("pl_PL-a", "en_US-b"))

    values = options.values_for(spec)

    assert values[0] == ""  # „dobierany do języka" to poprawny wybór
    assert "pl_PL-a" in values
    assert options.label_for("") == t("settings.auto_voice")


def test_lista_silnikow_ma_wariant_bez_mowy() -> None:
    spec = field_by_key("voice_engine")
    assert spec is not None
    assert "none" in ChoiceOptions(tts_engines=("piper", "none")).values_for(spec)


def test_walidacja_bez_formularza_dziala_na_dowolnej_bazie() -> None:
    base = UserSettings(assistant_name="Miku")
    problems = validate({"ui_accent_color": "#GGG"}, base=base)

    assert problems and problems[0].blocking
