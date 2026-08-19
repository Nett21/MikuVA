"""Wykrywanie aktywności głosowej (VAD) i segmentacja wypowiedzi.

Dostępne są dwa silniki:

* ``webrtc``  — ``webrtcvad`` (model z WebRTC, bardzo szybki, odporny na szum),
* ``energy``  — wbudowany detektor energetyczny z adaptacyjnym progiem szumu.

Dlaczego dwa, a nie od razu Silero: Silero VAD wymaga ``torch`` albo
``onnxruntime`` plus pobrania modelu przy pierwszym uruchomieniu, więc na
świeżej maszynie bez internetu potok by nie wystartował. ``webrtcvad`` jest
maleńki, ale bywa problematyczny do zbudowania na najnowszych wersjach Pythona.
Dlatego domyślny tryb ``auto`` używa ``webrtcvad``, jeśli jest zainstalowany,
a w przeciwnym razie schodzi na detektor energetyczny — czysty NumPy, zawsze
dostępny, bez pobierania czegokolwiek.

VAD odpowiada tylko na pytanie „czy w tej ramce jest mowa?". Zamianę strumienia
ramek na całe wypowiedzi robi :class:`UtteranceSegmenter`, dzięki czemu Whisper
dostaje wyłącznie fragmenty z mową i nie transkrybuje ciszy.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

import numpy as np

from audio.microphone import AudioFrame
from audio.resample import rms_dbfs
from config import Settings, get_settings

logger = logging.getLogger(__name__)

WEBRTC_SUPPORTED_RATES: Final[frozenset[int]] = frozenset({8_000, 16_000, 32_000, 48_000})
WEBRTC_SUPPORTED_FRAME_MS: Final[frozenset[int]] = frozenset({10, 20, 30})


class VADError(RuntimeError):
    """Nie udało się zbudować detektora mowy."""


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Wspólny kontrakt detektorów mowy."""

    @property
    def name(self) -> str: ...

    def is_speech(self, frame: AudioFrame) -> bool: ...

    def reset(self) -> None: ...


class EnergyVAD:
    """Detektor energetyczny z estymacją tła metodą statystyki minimum.

    Poziom tła to niski percentyl poziomów z ostatnich kilku sekund. Mowa jest
    z natury przerywana, więc percentyl trafia w pauzy i opisuje szum
    pomieszczenia, a nie głos.

    Trzy reguły, każda wynika z konkretnej awarii:

    * historia jest wstępnie wypełniona wartością początkową, żeby zdanie
      wypowiedziane w pierwszej sekundzie po starcie zostało wykryte,
    * podczas wykrytej mowy tło się nie uczy — inaczej dłuższa wypowiedź sama
      podniosłaby próg i urwałaby się w połowie,
    * ...ale gdy „mowa" trwa nieprzerwanie dłużej niż :attr:`max_active_s`,
      uczenie wraca. Bez tego mikrofon ze stałym głośnym szumem (wentylator,
      za wysoki gain) zostawałby w stanie mowy już na zawsze i VAD przestałby
      pełnić swoją jedyną funkcję. Najgorszy przypadek powrotu do normy to
      ``max_active_s + window_s``.
    """

    def __init__(
        self,
        *,
        threshold_db: float = 8.0,
        release_db: float = 3.0,
        initial_floor_dbfs: float = -50.0,
        min_floor_dbfs: float = -75.0,
        max_floor_dbfs: float = -20.0,
        window_s: float = 3.0,
        max_active_s: float = 5.0,
        frame_ms: int = 20,
        percentile: float = 10.0,
    ) -> None:
        self._threshold_db = threshold_db
        self._release_db = min(release_db, threshold_db)
        self._initial_floor = initial_floor_dbfs
        self._min_floor = min_floor_dbfs
        self._max_floor = max_floor_dbfs
        self._percentile = percentile
        self._window_frames = max(1, int(window_s * 1000 / frame_ms))
        self._max_active_frames = max(1, int(max_active_s * 1000 / frame_ms))
        self._levels: deque[float] = deque(maxlen=self._window_frames)
        self._active = False
        self._active_frames = 0
        self.reset()

    @property
    def name(self) -> str:
        return "energy"

    @property
    def noise_floor_dbfs(self) -> float:
        """Bieżące oszacowanie poziomu tła w dBFS."""
        floor = float(np.percentile(np.fromiter(self._levels, dtype=np.float64), self._percentile))
        return min(max(floor, self._min_floor), self._max_floor)

    def reset(self) -> None:
        self._levels.clear()
        self._levels.extend([self._initial_floor] * self._window_frames)
        self._active = False
        self._active_frames = 0

    def is_speech(self, frame: AudioFrame) -> bool:
        level = rms_dbfs(frame.samples)
        if level == -float("inf"):
            level = self._min_floor

        # Podczas mowy nie uczymy tła — chyba że „mowa" trwa podejrzanie długo.
        learning = not self._active or self._active_frames >= self._max_active_frames
        if learning:
            self._levels.append(level)

        floor = self.noise_floor_dbfs
        margin = self._threshold_db - self._release_db if self._active else self._threshold_db
        speech = level > floor + margin

        self._active_frames = self._active_frames + 1 if speech else 0
        self._active = speech
        return speech


class WebRTCVAD:
    """Adapter na ``webrtcvad`` — wymaga 8/16/32/48 kHz i ramek 10/20/30 ms."""

    def __init__(self, *, aggressiveness: int = 2, sample_rate: int = 16_000, frame_ms: int = 20) -> None:
        if sample_rate not in WEBRTC_SUPPORTED_RATES:
            raise VADError(
                f"webrtcvad obsługuje wyłącznie {sorted(WEBRTC_SUPPORTED_RATES)} Hz, "
                f"a ustawiono {sample_rate} Hz"
            )
        if frame_ms not in WEBRTC_SUPPORTED_FRAME_MS:
            raise VADError(
                f"webrtcvad obsługuje ramki {sorted(WEBRTC_SUPPORTED_FRAME_MS)} ms, "
                f"a ustawiono {frame_ms} ms"
            )
        try:
            import webrtcvad  # noqa: PLC0415 - import celowo leniwy
        except ImportError as exc:
            raise VADError("pakiet 'webrtcvad' nie jest zainstalowany") from exc

        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._expected_samples = int(sample_rate * frame_ms / 1000)
        self._vad = webrtcvad.Vad(aggressiveness)

    @property
    def name(self) -> str:
        return "webrtc"

    def reset(self) -> None:
        """webrtcvad nie trzyma stanu między ramkami — nie ma czego zerować."""

    def is_speech(self, frame: AudioFrame) -> bool:
        samples = frame.samples
        if samples.size != self._expected_samples:
            # Ramka niepełna (koniec strumienia) — dopełniamy ciszą, żeby nie
            # rzucać wyjątkiem w środku nasłuchu.
            padded = np.zeros(self._expected_samples, dtype=np.int16)
            usable = min(samples.size, self._expected_samples)
            padded[:usable] = samples[:usable]
            samples = padded
        return bool(self._vad.is_speech(samples.astype(np.int16).tobytes(), self._sample_rate))


def create_vad(settings: Settings | None = None) -> VoiceActivityDetector:
    """Zbuduj detektor zgodnie z ``VAD_ENGINE`` (``auto`` wybiera najlepszy dostępny)."""
    active = settings or get_settings()
    engine = active.vad_engine

    if engine in ("auto", "webrtc"):
        try:
            detector = WebRTCVAD(
                aggressiveness=active.vad_aggressiveness,
                sample_rate=active.audio_sample_rate,
                frame_ms=active.audio_frame_ms,
            )
        except VADError as exc:
            if engine == "webrtc":
                raise
            logger.info("webrtcvad niedostępny (%s) — używam VAD energetycznego.", exc)
        else:
            logger.info("VAD: webrtcvad (agresywność %s)", active.vad_aggressiveness)
            return detector

    detector = EnergyVAD(
        threshold_db=active.vad_energy_threshold_db,
        frame_ms=active.audio_frame_ms,
    )
    logger.info("VAD: energetyczny (próg %.1f dB ponad poziom tła)", active.vad_energy_threshold_db)
    return detector


@dataclass(frozen=True, slots=True)
class Utterance:
    """Wycięty fragment nagrania zawierający jedną wypowiedź."""

    samples: np.ndarray
    sample_rate: int
    started_at: float
    ended_at: float
    truncated: bool = False

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sample_rate if self.sample_rate else 0.0


@dataclass
class _SegmenterState:
    recording: bool = False
    speech_frames: int = 0
    silence_ms: int = 0
    buffer: list[np.ndarray] = field(default_factory=list)
    started_at: float = 0.0
    last_timestamp: float = 0.0


class UtteranceSegmenter:
    """Zamienia strumień ramek w pojedyncze wypowiedzi.

    Reguły: mowa musi trwać co najmniej ``VAD_MIN_SPEECH_MS``, żeby uznać ją za
    początek wypowiedzi (odsiewa trzaski i kliknięcia myszą); wypowiedź kończy
    ``VAD_MIN_SILENCE_MS`` ciszy; ``VAD_MAX_UTTERANCE_S`` jest twardym
    ogranicznikiem, żeby ciągły hałas nie zapełnił pamięci. Bufor wstępny
    (``VAD_PREROLL_MS``) dokleja dźwięk sprzed detekcji, dzięki czemu Whisper
    nie gubi pierwszej sylaby.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        vad: VoiceActivityDetector | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vad = vad if vad is not None else create_vad(self._settings)
        self._frame_ms = self._settings.audio_frame_ms
        self._sample_rate = self._settings.audio_sample_rate

        preroll_frames = max(0, self._settings.vad_preroll_ms // self._frame_ms)
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames or 1)
        self._state = _SegmenterState()

    @property
    def vad_name(self) -> str:
        return self._vad.name

    @property
    def is_recording(self) -> bool:
        return self._state.recording

    def reset(self) -> None:
        self._preroll.clear()
        self._state = _SegmenterState()
        self._vad.reset()

    def push(self, frame: AudioFrame) -> Utterance | None:
        """Dodaj ramkę; zwróć wypowiedź, jeśli właśnie się zakończyła."""
        speech = self._vad.is_speech(frame)
        state = self._state
        state.last_timestamp = frame.timestamp

        if not state.recording:
            self._preroll.append(frame.samples)
            if speech:
                state.speech_frames += 1
                if state.speech_frames * self._frame_ms >= self._settings.vad_min_speech_ms:
                    state.recording = True
                    state.started_at = frame.timestamp
                    state.silence_ms = 0
                    state.buffer = list(self._preroll)
                    self._preroll.clear()
            else:
                state.speech_frames = 0
            return None

        state.buffer.append(frame.samples)
        state.silence_ms = 0 if speech else state.silence_ms + self._frame_ms

        if state.silence_ms >= self._settings.vad_min_silence_ms:
            return self._finish(truncated=False)

        buffered_samples = sum(chunk.size for chunk in state.buffer)
        if buffered_samples >= self._settings.vad_max_utterance_s * self._sample_rate:
            logger.info("Wypowiedź przycięta po %.1f s", self._settings.vad_max_utterance_s)
            return self._finish(truncated=True)

        return None

    def flush(self) -> Utterance | None:
        """Zamknij trwające nagranie (np. przy zatrzymywaniu nasłuchu)."""
        if not self._state.recording:
            return None
        return self._finish(truncated=True)

    def _finish(self, *, truncated: bool) -> Utterance | None:
        state = self._state
        chunks = state.buffer
        started_at = state.started_at
        ended_at = state.last_timestamp

        self._state = _SegmenterState()
        self._preroll.clear()

        if not chunks:
            return None

        samples = np.concatenate(chunks).astype(np.int16, copy=False)
        return Utterance(
            samples=samples,
            sample_rate=self._sample_rate,
            started_at=started_at,
            ended_at=ended_at,
            truncated=truncated,
        )


__all__ = [
    "EnergyVAD",
    "UtteranceSegmenter",
    "Utterance",
    "VADError",
    "VoiceActivityDetector",
    "WebRTCVAD",
    "create_vad",
]
