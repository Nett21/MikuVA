"""Testy VAD i segmentacji wypowiedzi — na sygnale generowanym w pamięci."""

from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import make_silence, make_tone

from audio.microphone import AudioFrame
from audio.vad import EnergyVAD, UtteranceSegmenter, VADError, WebRTCVAD, create_vad
from config import Settings


def frame_of(samples: np.ndarray, timestamp: float = 0.0) -> AudioFrame:
    return AudioFrame(samples=samples, sample_rate=16_000, timestamp=timestamp)


class ScriptedVAD:
    """VAD sterowany listą wartości logicznych — jedna na ramkę."""

    def __init__(self, script: list[bool]) -> None:
        self._script = list(script)
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    def reset(self) -> None:
        self.calls = 0

    def is_speech(self, frame: AudioFrame) -> bool:
        value = self._script[self.calls] if self.calls < len(self._script) else False
        self.calls += 1
        return value


def test_energy_vad_odroznia_cisze_od_tonu() -> None:
    vad = EnergyVAD(threshold_db=8.0)

    for _ in range(20):  # dostrojenie poziomu tła
        assert vad.is_speech(frame_of(make_silence(320))) is False

    assert vad.is_speech(frame_of(make_tone(320))) is True


def test_stale_glosne_tlo_nie_zostaje_mowa_na_zawsze() -> None:
    """Regresja: mikrofon z ciągłym szumem musi wrócić do stanu 'cisza'.

    Bez tego VAD przepuszczałby do Whispera nieprzerwany strumień szumu.
    Gwarantowana granica to max_active_s + window_s (tu: 5 s + 3 s).
    """
    vad = EnergyVAD(threshold_db=8.0, window_s=3.0, max_active_s=5.0)
    loud_background = make_tone(320, amplitude=1500, frequency=90.0)

    for _ in range(int(8.0 * 1000 / 20)):  # 8 sekund ramek po 20 ms
        vad.is_speech(frame_of(loud_background))

    assert vad.is_speech(frame_of(loud_background)) is False
    # ...a wyraźnie głośniejszy sygnał nadal jest wykrywany jako mowa.
    assert vad.is_speech(frame_of(make_tone(320, amplitude=12000))) is True


def test_dluga_wypowiedz_nie_podnosi_progu_sama_sobie() -> None:
    """Podczas mowy tło się nie uczy — inaczej zdanie urwałoby się w połowie."""
    vad = EnergyVAD(threshold_db=8.0)
    for _ in range(30):
        vad.is_speech(frame_of(make_silence(320)))
    floor_before = vad.noise_floor_dbfs

    speech = make_tone(320, amplitude=9000)
    for _ in range(100):  # 2 sekundy ciągłej mowy
        assert vad.is_speech(frame_of(speech)) is True

    assert vad.noise_floor_dbfs == pytest.approx(floor_before, abs=0.5)


def test_energy_vad_ma_histereze() -> None:
    vad = EnergyVAD(threshold_db=10.0, release_db=6.0)
    for _ in range(20):
        vad.is_speech(frame_of(make_silence(320)))

    assert vad.is_speech(frame_of(make_tone(320, amplitude=9000))) is True
    # Cichszy fragment w środku wypowiedzi nadal liczy się jako mowa.
    assert vad.is_speech(frame_of(make_tone(320, amplitude=2500))) is True


def test_reset_przywraca_stan_poczatkowy() -> None:
    vad = EnergyVAD(initial_floor_dbfs=-50.0)
    for _ in range(200):  # cisza obniża oszacowanie tła
        vad.is_speech(frame_of(make_silence(320)))
    assert vad.noise_floor_dbfs < -55.0

    vad.reset()
    assert vad.noise_floor_dbfs == pytest.approx(-50.0)


def test_create_vad_wybiera_energy_gdy_brak_webrtc(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    settings = Settings(_env_file=None, vad_engine="auto")
    assert create_vad(settings).name == "energy"


def test_wymuszony_webrtc_bez_pakietu_zglasza_blad(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "webrtcvad", None)
    settings = Settings(_env_file=None, vad_engine="webrtc")
    with pytest.raises(VADError, match="webrtcvad"):
        create_vad(settings)


def test_webrtc_odrzuca_nieobslugiwana_czestotliwosc() -> None:
    with pytest.raises(VADError, match="8000"):
        WebRTCVAD(sample_rate=22_050, frame_ms=20)


def test_segmenter_sklada_wypowiedz_z_prerollem(settings: Settings) -> None:
    # 5 ramek ciszy, 15 ramek mowy, 10 ramek ciszy (200 ms → koniec wypowiedzi)
    script = [False] * 5 + [True] * 15 + [False] * 10
    segmenter = UtteranceSegmenter(settings, vad=ScriptedVAD(script))

    utterance = None
    for index in range(len(script)):
        samples = make_tone(320) if script[index] else make_silence(320)
        utterance = segmenter.push(frame_of(samples, timestamp=index * 0.02))
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.truncated is False
    # preroll (100 ms = 5 ramek) + mowa + cisza kończąca
    assert utterance.duration_s > 0.3
    assert segmenter.is_recording is False


def test_krotkie_trzaski_nie_uruchamiaja_nagrywania(settings: Settings) -> None:
    # Pojedyncza ramka „mowy" (20 ms) przy progu 100 ms.
    script = [False, True, False, False, False, False]
    segmenter = UtteranceSegmenter(settings, vad=ScriptedVAD(script))

    for index in range(len(script)):
        assert segmenter.push(frame_of(make_silence(320), timestamp=index * 0.02)) is None
    assert segmenter.is_recording is False


def test_maksymalna_dlugosc_przycina_wypowiedz(settings: Settings) -> None:
    short_limit = settings.model_copy(update={"vad_max_utterance_s": 0.5})
    segmenter = UtteranceSegmenter(short_limit, vad=ScriptedVAD([True] * 200))

    utterance = None
    for index in range(200):
        utterance = segmenter.push(frame_of(make_tone(320), timestamp=index * 0.02))
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.truncated is True
    assert utterance.duration_s <= 0.6


def test_flush_zamyka_trwajace_nagranie(settings: Settings) -> None:
    segmenter = UtteranceSegmenter(settings, vad=ScriptedVAD([True] * 20))
    for index in range(10):
        segmenter.push(frame_of(make_tone(320), timestamp=index * 0.02))

    assert segmenter.is_recording is True
    utterance = segmenter.flush()
    assert utterance is not None
    assert utterance.truncated is True
    assert segmenter.flush() is None


def test_cisza_nigdy_nie_tworzy_wypowiedzi(settings: Settings) -> None:
    """Kluczowa własność: Whisper nie może dostawać ciszy do transkrypcji."""
    segmenter = UtteranceSegmenter(settings, vad=EnergyVAD(threshold_db=8.0))

    for index in range(200):
        assert segmenter.push(frame_of(make_silence(320), timestamp=index * 0.02)) is None
    assert segmenter.is_recording is False


def test_znaczniki_czasu_pochodza_z_ramek(settings: Settings) -> None:
    script = [True] * 10 + [False] * 12
    segmenter = UtteranceSegmenter(settings, vad=ScriptedVAD(script))
    base = time.monotonic()

    utterance = None
    for index in range(len(script)):
        utterance = segmenter.push(frame_of(make_tone(320), timestamp=base + index * 0.02))
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.ended_at > utterance.started_at
