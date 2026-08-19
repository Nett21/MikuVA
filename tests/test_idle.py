"""Zachowanie w ciszy: model nie ma leżeć w pamięci, gdy nikt nie mówi.

Sama pętla nasłuchu nie zużywa procesora — czekanie na ramkę jest blokujące na
kolejce, a nie odpytywaniem. Kosztowna jest PAMIĘĆ: model Whispera wczytany „na
wszelki wypadek" trzyma kilkaset MB RAM-u, a na GPU tyle samo VRAM-u. Te testy
pilnują, że po ustalonym czasie ciszy zostaje zwolniony i że wraca sam przy
pierwszej wypowiedzi.

Żadnego sprzętu ani modelu tu nie ma: mikrofon, VAD, segmenter i transkryber są
atrapami.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from audio.microphone import AudioFrame
from audio.pipeline import SpeechToTextPipeline
from audio.vad import Utterance
from audio.whisper import Transcript
from config import Settings


class FakeMic:
    """Mikrofon oddający zaplanowaną sekwencję ramek; ``None`` = cisza (pusta kolejka)."""

    def __init__(self, frames: list[AudioFrame | None]) -> None:
        self._frames = list(frames)
        self.started = False
        self.reads = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def clear(self) -> None:
        return None

    @property
    def device(self) -> Any:
        return None

    def read(self, timeout: float | None = 0.5) -> AudioFrame | None:
        """Pusta kolejka CZEKA — dokładnie tak, jak ``queue.Queue.get(timeout=…)``.

        Bez tego czekania atrapa oddawałaby ``None`` natychmiast i test
        „pętla nie odpytuje aktywnie" mierzyłby szybkość atrapy, a nie
        zachowanie kodu produkcyjnego.
        """
        self.reads += 1
        if not self._frames:
            time.sleep(timeout if timeout is not None else 0.0)
            return None
        return self._frames.pop(0)


class FakeSegmenter:
    def __init__(self) -> None:
        self.is_recording = False
        self.result: Utterance | None = None

    @property
    def vad_name(self) -> str:
        return "atrapa"

    def reset(self) -> None:
        self.is_recording = False

    def flush(self) -> Utterance | None:
        return None

    def push(self, frame: AudioFrame) -> Utterance | None:
        result, self.result = self.result, None
        return result


class FakeTranscriber:
    """Transkryber liczący ładowania i zwalniania — bez żadnego modelu."""

    def __init__(self, *, name: str = "tiny") -> None:
        self.is_loaded = False
        self.loads = 0
        self.unloads = 0
        self.model_name = name

    def load(self) -> None:
        if not self.is_loaded:
            self.loads += 1
        self.is_loaded = True

    def unload(self) -> None:
        if self.is_loaded:
            self.unloads += 1
        self.is_loaded = False

    def describe(self) -> str:
        return f"{self.model_name} (atrapa)"

    def transcribe(self, utterance: Any, **kwargs: Any) -> Transcript:
        self.load()  # tak samo jak prawdziwy WhisperTranscriber
        return Transcript(
            text="zapaliło się światło",
            language="pl",
            language_probability=0.9,
            audio_duration_s=1.0,
            processing_s=0.1,
        )


def make_frame(settings: Settings) -> AudioFrame:
    size = int(settings.audio_sample_rate * settings.audio_frame_ms / 1000)
    return AudioFrame(
        samples=np.zeros(size, dtype=np.int16),
        sample_rate=settings.audio_sample_rate,
        timestamp=0.0,
    )


def make_utterance(settings: Settings) -> Utterance:
    return Utterance(
        samples=np.zeros(settings.audio_sample_rate, dtype=np.int16),
        sample_rate=settings.audio_sample_rate,
        started_at=0.0,
        ended_at=1.0,
    )


@pytest.fixture
def idle_settings(tmp_path: Any) -> Settings:
    return Settings(
        _env_file=None,
        piper_voices_dir=str(tmp_path / "voices"),
        whisper_idle_unload_s=10.0,
        vad_listen_timeout_s=1.0,
        wake_enabled=False,
        wake_engine="none",
    )


def build(
    settings: Settings, frames: list[AudioFrame | None]
) -> tuple[SpeechToTextPipeline, FakeTranscriber, FakeSegmenter]:
    transcriber = FakeTranscriber()
    segmenter = FakeSegmenter()
    pipeline = SpeechToTextPipeline(
        settings,
        microphone=FakeMic(frames),  # type: ignore[arg-type]
        segmenter=segmenter,  # type: ignore[arg-type]
        transcriber=transcriber,  # type: ignore[arg-type]
        wake=None,
    )
    return pipeline, transcriber, segmenter


# --------------------------------------------------------------------------- #


def test_start_laduje_model(idle_settings: Settings) -> None:
    pipeline, transcriber, _ = build(idle_settings, [])
    pipeline.start()
    assert transcriber.is_loaded
    assert transcriber.loads == 1


def test_cisza_krotsza_niz_prog_nie_zwalnia_modelu(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, transcriber, _ = build(idle_settings, [])
    pipeline.start()

    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()

    zegar["teraz"] += 9.0  # próg to 10 s
    assert pipeline._maybe_unload_idle() is False
    assert transcriber.is_loaded


def test_cisza_dluzsza_niz_prog_zwalnia_model(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, transcriber, _ = build(idle_settings, [])
    pipeline.start()

    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()

    zegar["teraz"] += 11.0
    assert pipeline._maybe_unload_idle() is True
    assert not transcriber.is_loaded
    assert transcriber.unloads == 1
    assert pipeline.is_idle_unloaded


def test_zwolnienie_dzieje_sie_raz_a_nie_przy_kazdej_ramce(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bez znacznika sprawdzenie powtarzałoby się 50 razy na sekundę."""
    pipeline, transcriber, _ = build(idle_settings, [])
    pipeline.start()
    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()

    zegar["teraz"] += 30.0
    for _ in range(20):
        pipeline._maybe_unload_idle()
    assert transcriber.unloads == 1


def test_zero_wylacza_mechanizm(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        piper_voices_dir=str(tmp_path / "voices"),
        whisper_idle_unload_s=0.0,
        wake_enabled=False,
        wake_engine="none",
    )
    pipeline, transcriber, _ = build(settings, [])
    pipeline.start()
    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()
    zegar["teraz"] += 100_000.0
    assert pipeline._maybe_unload_idle() is False
    assert transcriber.is_loaded


def test_nie_zwalniamy_w_srodku_wypowiedzi(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nagrywanie trwa — zwolnienie modelu w tym momencie to utracona wypowiedź."""
    pipeline, transcriber, segmenter = build(idle_settings, [])
    pipeline.start()
    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()

    segmenter.is_recording = True
    zegar["teraz"] += 60.0
    assert pipeline._maybe_unload_idle() is False
    assert transcriber.is_loaded


def test_nie_zwalniamy_przy_otwartym_oknie_rozmowy(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fraza padła, użytkownik zbiera myśli — model ma być gotowy."""
    pipeline, transcriber, _ = build(idle_settings, [])
    pipeline.start()
    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline.wake_up()

    zegar["teraz"] += 11.0  # okno rozmowy trwa domyślnie 30 s
    assert pipeline._maybe_unload_idle() is False
    assert transcriber.is_loaded


def test_model_wraca_sam_przy_pierwszej_wypowiedzi(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sedno mechanizmu: zwolnienie ma być niewidoczne poza jednorazowym opóźnieniem."""
    frame = make_frame(idle_settings)
    pipeline, transcriber, segmenter = build(idle_settings, [frame])
    pipeline.start()

    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()
    zegar["teraz"] += 60.0
    pipeline._maybe_unload_idle()
    assert not transcriber.is_loaded

    segmenter.result = make_utterance(idle_settings)
    transcript = pipeline.listen_once(timeout_s=0)
    assert transcript is not None
    assert transcript.text == "zapaliło się światło"
    assert transcriber.is_loaded
    assert transcriber.loads == 2  # start + powrót po ciszy


def test_wykrycie_mowy_odswieza_znacznik(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, transcriber, _ = build(idle_settings, [])
    pipeline.start()
    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()

    zegar["teraz"] += 9.0
    pipeline._mark_activity()  # jak przy zdarzeniu SPEECH_START
    zegar["teraz"] += 9.0
    assert pipeline._maybe_unload_idle() is False
    assert transcriber.is_loaded


def test_wspoldzielony_model_frazy_nie_jest_zwalniany(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gdy detektor frazy używa TEGO SAMEGO obiektu, zwolnienie kosztowałoby każde zawołanie."""
    transcriber = FakeTranscriber()
    segmenter = FakeSegmenter()
    pipeline = SpeechToTextPipeline(
        idle_settings,
        microphone=FakeMic([]),  # type: ignore[arg-type]
        segmenter=segmenter,  # type: ignore[arg-type]
        transcriber=transcriber,  # type: ignore[arg-type]
        wake_transcriber=transcriber,  # type: ignore[arg-type]
        wake=None,
    )
    pipeline.start()
    zegar = {"teraz": 1_000.0}
    monkeypatch.setattr("audio.pipeline.time.monotonic", lambda: zegar["teraz"])
    pipeline._mark_activity()
    zegar["teraz"] += 60.0
    assert pipeline._maybe_unload_idle() is False
    assert transcriber.is_loaded


def test_petla_nasluchu_nie_odpytuje_aktywnie(idle_settings: Settings) -> None:
    """Cisza kończy się limitem czasu, a nie tysiącami obrotów pętli.

    ``Microphone.read`` czeka blokująco na kolejce (200 ms), więc sekunda ciszy
    to około pięciu obrotów — a nie tyle, ile zdąży procesor.
    """
    pipeline, _, _ = build(idle_settings, [])
    mic: FakeMic = pipeline.microphone  # type: ignore[assignment]
    pipeline.start()
    assert pipeline.listen_once(timeout_s=0.5) is None
    assert mic.reads < 50


def test_domyslny_prog_jest_wlaczony_i_liczony_w_minutach() -> None:
    settings = Settings(_env_file=None)
    assert settings.whisper_idle_unload_s >= 60.0
