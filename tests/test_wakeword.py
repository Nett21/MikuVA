"""Testy słowa aktywującego — dopasowanie frazy, silniki i wybór silnika.

Nic tu nie dotyka mikrofonu ani modelu: detektor whisperowy dostaje funkcję
transkrypcji jako argument, więc w teście jest nią zwykły słownik odpowiedzi.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from conftest import make_tone

from audio.microphone import AudioFrame
from audio.vad import Utterance
from audio.wakeword import (
    OpenWakeWordEngine,
    PhraseMatcher,
    WakeWordError,
    WhisperWakeWord,
    create_wake_word_engine,
    find_wakeword_models,
    fold_sounds,
    normalize_token,
    resolve_wake_phrase,
    tokenize,
)
from config import Settings, UserSettings


def utterance_of(seconds: float = 1.0) -> Utterance:
    samples = make_tone(int(16_000 * seconds))
    return Utterance(samples=samples, sample_rate=16_000, started_at=0.0, ended_at=seconds)


# --------------------------------------------------------------------------- #
# Fraza pochodzi z ustawień użytkownika, nie z kodu
# --------------------------------------------------------------------------- #


def test_domyslna_fraza_powstaje_z_imienia_asystenta() -> None:
    """Zmiana imienia zmienia zawołanie — bez dotykania kodu i pliku .env."""
    assert UserSettings().effective_wake_word == "hej Miku"
    assert UserSettings(assistant_name="Aiko").effective_wake_word == "hej Aiko"


def test_jawna_fraza_ma_pierwszenstwo_przed_imieniem() -> None:
    user = UserSettings(assistant_name="Aiko", wake_word="komputerze słuchaj")
    assert user.effective_wake_word == "komputerze słuchaj"
    assert resolve_wake_phrase(user) == "komputerze słuchaj"


def test_pusta_fraza_po_oczyszczeniu_wraca_do_imienia() -> None:
    user = UserSettings(assistant_name="Miku", wake_word="   ")
    assert user.effective_wake_word == "hej Miku"


# --------------------------------------------------------------------------- #
# Dopasowanie tekstu
# --------------------------------------------------------------------------- #


def test_normalizacja_zdejmuje_ogonki_i_wielkosc_liter() -> None:
    assert normalize_token("Miku!") == "miku"
    assert normalize_token("Zażółć") == "zazolc"
    assert normalize_token("ŁÓDŹ") == "lodz"


def test_tokenizacja_zwraca_pozycje_w_oryginale() -> None:
    tokens = tokenize("Hej, Miku!")
    assert [token for token, _, _ in tokens] == ["hej", "miku"]
    assert tokens[-1][2] == len("Hej, Miku")


@pytest.mark.parametrize(
    "heard",
    [
        "Hej Miku",
        "hej, miku",
        "Hej Miku!",
        "HEJ MIKU.",
        "Hey Miku",  # Whisper często zapisuje po angielsku
        "Ej Miku",  # przekręcone przez model tiny
        "O, hej Miku",  # wypełniacz na początku
    ],
)
def test_warianty_zapisu_frazy_sa_rozpoznawane(heard: str) -> None:
    matcher = PhraseMatcher("hej miku")
    match = matcher.match(heard)
    assert match is not None, heard
    assert match.score >= matcher.threshold


@pytest.mark.parametrize(
    "heard",
    [
        "jaka jest dzisiaj pogoda",
        "wczoraj byłem w kinie na dobrym filmie",
        "",
        "   ",
        "no to co robimy dalej",
    ],
)
def test_mowa_bez_frazy_nie_jest_dopasowana(heard: str) -> None:
    assert PhraseMatcher("hej miku").match(heard) is None


def test_fraza_w_srodku_dlugiego_zdania_nie_wybudza() -> None:
    """Rozmowa w tle nie może aktywować asystenta w połowie wypowiedzi."""
    matcher = PhraseMatcher("hej miku")
    assert matcher.match("wczoraj rozmawiałem z Michałem i powiedziałem hej miku") is None


def test_polecenie_wypowiedziane_jednym_tchem_jest_odcinane() -> None:
    matcher = PhraseMatcher("hej miku")
    match = matcher.match("Hej Miku, jaka jest pogoda?")

    assert match is not None
    assert match.has_command is True
    assert match.command == "jaka jest pogoda?"
    assert matcher.strip_phrase("Hej Miku, jaka jest pogoda?") == "jaka jest pogoda?"


def test_samo_zawolanie_nie_ma_polecenia() -> None:
    match = PhraseMatcher("hej miku").match("Hej Miku")
    assert match is not None
    assert match.has_command is False


def test_dowolna_fraza_uzytkownika_dziala_tak_samo() -> None:
    """Nic w kodzie nie zakłada brzmienia frazy — także dla innego języka."""
    matcher = PhraseMatcher("okay computer")
    match = matcher.match("Okay, computer — turn on the light")
    assert match is not None
    assert match.command == "turn on the light"


def test_prog_podobienstwa_decyduje_o_czulosci() -> None:
    """Próg rządzi tam, gdzie słowa naprawdę brzmią inaczej.

    „ej micu" celowo NIE jest już przykładem na czułość progu: to ta sama fraza
    zapisana inaczej przez Whispera i po zwinięciu zapisu („c"→„k", „h" nieme)
    wychodzi z niej dokładnie „ejmiku". Rozróżnianie takich wariantów było
    główną przyczyną tego, że asystent ledwo słyszał własne imię.
    """
    tolerant = PhraseMatcher("hej miku", threshold=0.5, name_threshold=1.01)
    strict = PhraseMatcher("hej miku", threshold=0.95, name_threshold=1.01)

    assert tolerant.match("hej miku") is not None
    assert tolerant.match("hej mika") is not None
    assert strict.match("hej mika") is None
    # Zapis inny, brzmienie to samo — łapane niezależnie od progu.
    assert strict.match("ej micu") is not None


def test_pusta_fraza_jest_bledem_konfiguracji() -> None:
    with pytest.raises(WakeWordError):
        PhraseMatcher("   ")
    with pytest.raises(WakeWordError):
        PhraseMatcher("...")


# --------------------------------------------------------------------------- #
# Detektor whisperowy
# --------------------------------------------------------------------------- #


class FakeWakeTranscribe:
    """Podstawia gotowy tekst zamiast modelu i liczy wywołania."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def __call__(self, utterance: Utterance) -> str:
        self.calls += 1
        return self._texts.pop(0) if self._texts else ""


def test_detektor_whisperowy_wykrywa_fraze() -> None:
    transcribe = FakeWakeTranscribe(["Hej Miku, która godzina"])
    engine = WhisperWakeWord(PhraseMatcher("hej miku"), transcribe)

    match = engine.process_utterance(utterance_of(1.2))

    assert match is not None
    assert match.command == "która godzina"
    assert engine.mode == "utterance"


def test_dlugie_wypowiedzi_nie_sa_transkrybowane() -> None:
    """Główna oszczędność CPU: rozmowa w tle nie trafia nawet do modelu tiny."""
    transcribe = FakeWakeTranscribe(["cokolwiek"])
    engine = WhisperWakeWord(PhraseMatcher("hej miku"), transcribe, max_duration_s=2.0)

    assert engine.process_utterance(utterance_of(5.0)) is None
    assert transcribe.calls == 0
    assert engine.skipped_long == 1


def test_blad_transkrypcji_nie_wysadza_detektora() -> None:
    def boom(utterance: Utterance) -> str:
        raise RuntimeError("model padł")

    engine = WhisperWakeWord(PhraseMatcher("hej miku"), boom)
    assert engine.process_utterance(utterance_of(1.0)) is None


# --------------------------------------------------------------------------- #
# openWakeWord
# --------------------------------------------------------------------------- #


def test_brak_modelu_openwakeword_jest_czytelnym_bledem() -> None:
    with pytest.raises(WakeWordError) as error:
        OpenWakeWordEngine(PhraseMatcher("hej miku"), [])
    assert "openWakeWord" in error.value.message


def test_modele_openwakeword_sa_szukane_w_katalogu_projektu(tmp_path: Path) -> None:
    (tmp_path / "hej_miku.onnx").write_bytes(b"model")
    (tmp_path / "notatka.txt").write_text("nie model", encoding="utf-8")

    found = find_wakeword_models(UserSettings(), tmp_path)

    assert [path.name for path in found] == ["hej_miku.onnx"]


def test_wskazana_sciezka_modelu_ma_pierwszenstwo(tmp_path: Path) -> None:
    explicit = tmp_path / "moj_model.tflite"
    explicit.write_bytes(b"model")
    user = UserSettings(wake_word_model=str(explicit))

    found = find_wakeword_models(user, tmp_path)

    assert found[0] == explicit


# --------------------------------------------------------------------------- #
# Wybór silnika
# --------------------------------------------------------------------------- #


def test_wylaczone_slowo_aktywujace_nie_buduje_detektora(settings: Settings) -> None:
    disabled = settings.model_copy(update={"wake_enabled": False})
    assert create_wake_word_engine(disabled, user_settings=UserSettings()) is None

    none_engine = settings.model_copy(update={"wake_engine": "none"})
    assert create_wake_word_engine(none_engine, user_settings=UserSettings()) is None


def test_auto_bez_openwakeword_schodzi_na_detektor_whisperowy(
    settings: Settings, tmp_path: Path
) -> None:
    engine = create_wake_word_engine(
        settings,
        user_settings=UserSettings(),
        transcribe=FakeWakeTranscribe([]),
        models_dir=tmp_path,  # pusty katalog = brak modeli KWS
    )

    assert isinstance(engine, WhisperWakeWord)
    assert engine.phrase == "hej Miku"


def test_wymuszony_openwakeword_bez_modelu_zglasza_blad(settings: Settings, tmp_path: Path) -> None:
    forced = settings.model_copy(update={"wake_engine": "openwakeword"})
    with pytest.raises(WakeWordError):
        create_wake_word_engine(
            forced,
            user_settings=UserSettings(),
            transcribe=FakeWakeTranscribe([]),
            models_dir=tmp_path,
        )


def test_prog_podobienstwa_z_ustawien_trafia_do_detektora(
    settings: Settings, tmp_path: Path
) -> None:
    tuned = settings.model_copy(update={"wake_similarity": 0.9})
    engine = create_wake_word_engine(
        tuned,
        user_settings=UserSettings(wake_word="komputerze"),
        transcribe=FakeWakeTranscribe([]),
        models_dir=tmp_path,
    )

    assert isinstance(engine, WhisperWakeWord)
    assert engine.phrase == "komputerze"
    assert engine.process_utterance(utterance_of(0.5)) is None


def test_openwakeword_konsumuje_ramki_porcjami(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapter musi buforować ramki do porcji 80 ms wymaganych przez modele."""
    chunks: list[int] = []

    class FakeModel:
        def __init__(self, wakeword_models: list[str]) -> None:
            self.models = wakeword_models

        def predict(self, chunk: np.ndarray) -> dict[str, float]:
            chunks.append(chunk.size)
            return {"hej_miku": 0.9 if len(chunks) >= 2 else 0.1}

        def reset(self) -> None:
            pass

    module = types.ModuleType("openwakeword.model")
    module.Model = FakeModel  # type: ignore[attr-defined]
    package = types.ModuleType("openwakeword")
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", module)

    engine = OpenWakeWordEngine(PhraseMatcher("hej miku"), [Path("hej_miku.onnx")], threshold=0.5)
    frame = AudioFrame(
        samples=np.zeros(1_280 * 2, dtype=np.int16), sample_rate=16_000, timestamp=0.0
    )

    match = engine.process_frame(frame)

    assert chunks == [1_280, 1_280]
    assert match is not None
    assert engine.mode == "stream"


# --------------------------------------------------------------------------- #
# Przypadki z korpusu prawdziwych transkrypcji
# --------------------------------------------------------------------------- #
#
# Wszystkie teksty poniżej NAPRAWDĘ wyszły z Whispera przy zdaniach wypowiedzianych
# frazą „Hej Miku" (korpus 480 transkrypcji mowy syntetycznej, modele tiny/base/small,
# warianty: czysto, szum, cicho). Obecne dopasowanie wykrywało 27% z nich.


@pytest.mark.parametrize(
    "uslyszane",
    [
        "Hej Miku",
        "Tej Miku, która godzina",
        "w tej miku, jaka jest pogoda w Warszawie?",
        # Fraza sklejona w jedno słowo — najczęstszy zapis w całym korpusie.
        "Tymiku otwusz przeglądarkę.",
        "w tym micu otwórz przegodarkę.",
        # Whisper gubi albo podmienia „hej", ale nazwa zostaje.
        "Miku zapisz notatkę.",
        "OK, Miku, ile to jest 2 plus 2?",
        "i miku wyłącznie.",
        "Kej miku otwórz przeglądarkę.",
    ],
)
def test_przekrecona_fraza_jest_rozpoznawana(uslyszane: str) -> None:
    matcher = PhraseMatcher("hej Miku", threshold=0.72, name_threshold=0.80)

    assert matcher.match(uslyszane) is not None, f"nie rozpoznano {uslyszane!r}"


@pytest.mark.parametrize(
    "uslyszane",
    [
        # Zdania z korpusu negatywnego: brzmią podobnie, ale nikt nikogo nie woła.
        "Ten mikser jest zepsuty",
        "Mika przyjedzie jutro rano",
        "Ten mikrofon jest za cichy",
        "Poproszę kawę z mlekiem",
        "Hej, słuchaj co się stało",
        "Muszę kupić mleko i chleb",
        "Która jest godzina w Krakowie?",
    ],
)
def test_zwykla_rozmowa_nie_budzi_asystenta(uslyszane: str) -> None:
    """Luźniejsze dopasowanie nie może zamienić się w nasłuch na wszystko.

    Na 240 negatywach z korpusu ta konfiguracja dała ZERO fałszywych pobudek;
    obniżenie progu nazwy do 0,75 podnosiło je do 7,9% — stąd wartość 0,80.
    """
    matcher = PhraseMatcher("hej Miku", threshold=0.72, name_threshold=0.80)

    assert matcher.match(uslyszane) is None, f"fałszywa pobudka na {uslyszane!r}"


def test_sama_nazwa_wystarczy_ale_dalej_od_poczatku_juz_nie() -> None:
    """Nazwa w środku dłuższego zdania to rozmowa O asystencie, nie DO niego."""
    matcher = PhraseMatcher("hej Miku", threshold=0.72, name_threshold=0.80)

    assert matcher.match("Miku, otwórz przeglądarkę") is not None
    assert matcher.match("wczoraj rozmawiałem z kolegą o tym, jak działa Miku") is None


def test_zwijanie_zapisu_nie_scala_roznych_slow() -> None:
    """Zwijanie ma ratować zapis tego samego brzmienia, a nie mieszać słowa."""
    assert fold_sounds("micu") == fold_sounds("miku")
    assert fold_sounds("hej") == fold_sounds("ej")
    assert fold_sounds("mikser") != fold_sounds("miku")
    assert fold_sounds("mika") != fold_sounds("miku")
