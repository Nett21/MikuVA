"""Wspólne atrapy dla testów warstwy audio.

Żaden test nie dotyka prawdziwego mikrofonu, karty dźwiękowej ani modelu
Whisper. Biblioteki natywne są podmieniane w ``sys.modules`` — działa to,
ponieważ moduły projektu importują je leniwie, dopiero w chwili użycia.
"""

from __future__ import annotations

import dataclasses
import importlib.machinery
import json
import re
import sys
import types
import zlib
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings  # noqa: E402 - po ustawieniu sys.path

# --------------------------------------------------------------------------- #
# Ustawienia testowe
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pusty katalog modeli zamiast prawdziwego ``models/whisper``.

    Bez tego wynik testów zależałby od tego, czy deweloper ma akurat pobrany
    model — a od jego obecności zależy rozstrzygnięcie trybu ``OFFLINE_MODE=auto``.
    """
    import config

    cache = tmp_path / "models" / "whisper"
    cache.mkdir(parents=True)
    monkeypatch.setattr(config, "WHISPER_CACHE_DIR", cache)
    monkeypatch.setattr(config, "WHEELHOUSE_DIR", tmp_path / "vendor" / "wheels")

    for module_name in ("audio.whisper", "audio.dependencies"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "WHISPER_CACHE_DIR"):
            monkeypatch.setattr(module, "WHISPER_CACHE_DIR", cache)
    return cache


@pytest.fixture(autouse=True)
def isolated_voice_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Odetnij testy od głosów Pipera zainstalowanych w systemie.

    Bez tego wynik zależałby od tego, czy deweloper ma akurat rozpakowany
    katalog ``~/.local/share/piper`` — a testy mają sprawdzać kod, nie maszynę.
    """
    import config

    voices = tmp_path / "models" / "piper"
    voices.mkdir(parents=True)
    monkeypatch.setattr(config, "PIPER_DIR", voices)
    # Katalogi systemowe (XDG, LOCALAPPDATA, /usr/share) znikają z listy —
    # zostaje wyłącznie to, co test sam położy na dysku.
    monkeypatch.setattr(config, "user_data_directories", lambda *args, **kwargs: [])
    return voices


@pytest.fixture(autouse=True)
def isolated_embedding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pusty katalog modeli embeddingów zamiast prawdziwego ``models/embeddings``.

    Bez tego wynik testów zależałby od tego, czy deweloper ma akurat pobrany
    model — a testy mają sprawdzać kod, nie zawartość jego dysku.
    """
    import config

    cache = tmp_path / "models" / "embeddings"
    cache.mkdir(parents=True)
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", cache)
    for module_name in ("brain.embeddings", "brain.dependencies"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "EMBEDDINGS_DIR"):
            monkeypatch.setattr(module, "EMBEDDINGS_DIR", cache)
    return cache


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Katalog danych (baza SQLite) w tmp_path zamiast prawdziwego katalogu użytkownika.

    Bezpiecznik: żaden test nie może dopisać się do bazy, której używa asystent
    na maszynie dewelopera — nawet gdyby ktoś zapomniał podać ścieżki wprost.
    """
    import config

    data = tmp_path / "dane"
    monkeypatch.setattr(config, "app_data_directory", lambda *args, **kwargs: data)
    monkeypatch.setenv("MIKU_DATA_DIR", str(data))
    return data


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ustawienia użytkownika z ``tmp_path``, nie z ``config/user_settings.json``.

    Bezpiecznik dopisany po tym, jak dwa testy zaczęły zależeć od tego, co
    deweloper ma akurat w swoim pliku: wpisanie tam ``"speech_language": "en"``
    wywracało testy języka transkrypcji. Testy mają sprawdzać kod, nie maszynę.
    """
    import config

    target = tmp_path / "config" / "user_settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "USER_SETTINGS_FILE", target)
    # Pamięć podręczna trzyma to, co wczytano WCZEŚNIEJ — razem ze ścieżką.
    monkeypatch.setattr(config, "_user_settings_cache", None)
    monkeypatch.setattr(config, "_user_settings_mtime", None)
    return target


@pytest.fixture(autouse=True)
def default_ui_language() -> Iterator[None]:
    """Każdy test zaczyna z domyślnym (angielskim) językiem interfejsu.

    Bez tego test, który przełączy język na polski, zmieniałby wynik kolejnych —
    ``set_ui_language`` jest ustawieniem procesu, nie obiektu.
    """
    import i18n

    i18n.set_ui_language(i18n.DEFAULT_UI_LANGUAGE)
    yield
    i18n.set_ui_language(i18n.DEFAULT_UI_LANGUAGE)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Ustawienia niezależne od pliku .env użytkownika i od maszyny."""
    return Settings(
        _env_file=None,
        # Katalog głosów wskazany wprost: testy nie mogą zależeć od tego, czy
        # ktoś ma zainstalowanego Pipera w systemie.
        piper_voices_dir=str(tmp_path / "voices"),
        audio_output_device="",
        audio_sample_rate=16_000,
        audio_frame_ms=20,
        audio_queue_seconds=5.0,
        audio_suppress_device_warnings=False,
        vad_engine="energy",
        vad_min_speech_ms=100,
        vad_min_silence_ms=200,
        vad_preroll_ms=100,
        vad_max_utterance_s=2.0,
        vad_listen_timeout_s=1.0,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_compute_type="int8",
        whisper_min_duration_s=0.0,
        mic_enabled=True,
        audio_input_device="",
    )


# --------------------------------------------------------------------------- #
# Generatory sygnału
# --------------------------------------------------------------------------- #


def make_silence(samples: int) -> np.ndarray:
    """Cisza z minimalnym szumem tła (realniejsza niż idealne zera)."""
    generator = np.random.default_rng(1234)
    return (generator.normal(0.0, 12.0, samples)).astype(np.int16)


def make_tone(samples: int, *, amplitude: int = 9000, frequency: float = 220.0) -> np.ndarray:
    """Ton sinusoidalny udający mowę (wysoka energia względem tła)."""
    time_axis = np.arange(samples, dtype=np.float64) / 16_000.0
    return (amplitude * np.sin(2 * np.pi * frequency * time_axis)).astype(np.int16)


# --------------------------------------------------------------------------- #
# Atrapa sounddevice
# --------------------------------------------------------------------------- #


class FakeInputStream:
    """Atrapa ``sounddevice.InputStream`` sterowana ręcznie z testu."""

    def __init__(
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        blocksize: int,
        device: int | None,
        callback: Any,
        module: FakeSoundDevice,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.started = False
        self.closed = False
        self._module = module

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def feed(self, samples: np.ndarray) -> None:
        """Wepchnij próbki tak, jak zrobiłby to wątek PortAudio."""
        block = samples.reshape(-1, 1) if samples.ndim == 1 else samples
        self.callback(block, block.shape[0], None, None)


class FakeOutputStream:
    """Atrapa ``sounddevice.OutputStream`` — dźwięk odbiera test, nie karta.

    ``pull()`` udaje jeden obrót wątku PortAudio: prosi callback o ramki i
    zapamiętuje, co dostał. Dzięki temu da się sprawdzić, CO i KIEDY poszło do
    głośnika, bez żadnego sprzętu.
    """

    def __init__(
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        blocksize: int,
        device: int | None,
        callback: Any,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.started = False
        self.closed = False
        self.played: list[np.ndarray] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def pull(self, frames: int = 256) -> np.ndarray:
        """Poproś o kolejny blok próbek tak, jak zrobiłoby to PortAudio."""
        outdata = np.zeros((frames, self.channels), dtype=np.int16)
        self.callback(outdata, frames, None, None)
        block = outdata[:, 0].copy()
        self.played.append(block)
        return block

    def pull_until_silent(self, frames: int = 256, limit: int = 500) -> np.ndarray:
        """Odtwarzaj, dopóki coś jeszcze wychodzi (albo do bezpiecznika)."""
        collected: list[np.ndarray] = []
        for _ in range(limit):
            block = self.pull(frames)
            collected.append(block)
            if not np.any(block):
                break
        return np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)


class FakeSoundDevice(types.ModuleType):
    """Minimalna atrapa modułu ``sounddevice``."""

    def __init__(
        self,
        *,
        devices: Sequence[dict[str, Any]] | None = None,
        fail_rates: Sequence[int] = (),
        raise_on_query: Exception | None = None,
    ) -> None:
        super().__init__("sounddevice")
        # Bez ``__spec__`` moduł wygląda dla importlib.util.find_spec na
        # niezainstalowany — a właśnie tak sprawdza go detekcja zależności.
        self.__spec__ = importlib.machinery.ModuleSpec("sounddevice", loader=None)
        self._devices = list(
            devices
            if devices is not None
            else [
                {
                    "name": "Atrapa mikrofonu",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "hostapi": 0,
                    "default_samplerate": 48_000.0,
                },
                {
                    "name": "Atrapa głośnika",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "hostapi": 0,
                    "default_samplerate": 48_000.0,
                },
            ]
        )
        self._fail_rates = set(fail_rates)
        self._raise_on_query = raise_on_query
        self.streams: list[FakeInputStream] = []
        self.output_streams: list[FakeOutputStream] = []

        class PortAudioError(Exception):
            pass

        self.PortAudioError = PortAudioError

    def query_devices(self) -> list[dict[str, Any]]:
        if self._raise_on_query is not None:
            raise self._raise_on_query
        return list(self._devices)

    def query_hostapis(self) -> list[dict[str, Any]]:
        return [{"name": "AtrapaAPI"}]

    def InputStream(  # noqa: N802 - nazwa z API sounddevice
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        blocksize: int,
        device: int | None,
        callback: Any,
    ) -> FakeInputStream:
        if samplerate in self._fail_rates:
            raise self.PortAudioError(f"nieobsługiwana częstotliwość {samplerate}")
        stream = FakeInputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=blocksize,
            device=device,
            callback=callback,
            module=self,
        )
        self.streams.append(stream)
        return stream

    def OutputStream(  # noqa: N802 - nazwa z API sounddevice
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        blocksize: int,
        device: int | None,
        callback: Any,
    ) -> FakeOutputStream:
        if samplerate in self._fail_rates:
            raise self.PortAudioError(f"nieobsługiwana częstotliwość {samplerate}")
        stream = FakeOutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=blocksize,
            device=device,
            callback=callback,
        )
        self.output_streams.append(stream)
        return stream


@pytest.fixture
def fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSoundDevice]:
    module = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    yield module


# --------------------------------------------------------------------------- #
# Atrapa faster_whisper
# --------------------------------------------------------------------------- #


class FakeSegment:
    def __init__(self, text: str, *, start: float = 0.0, end: float = 1.0, no_speech_prob: float = 0.0) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.no_speech_prob = no_speech_prob


class FakeTranscriptionInfo:
    def __init__(self, language: str = "pl", language_probability: float = 0.98) -> None:
        self.language = language
        self.language_probability = language_probability


class FakeWhisperModel:
    """Atrapa ``faster_whisper.WhisperModel`` zapisująca, z czym ją wywołano."""

    instances: list[FakeWhisperModel] = []

    def __init__(
        self,
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
        download_root: str,
        local_files_only: bool,
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.local_files_only = local_files_only
        self.calls: list[dict[str, Any]] = []
        self.segments: list[FakeSegment] = [FakeSegment("Cześć, tu atrapa.")]
        self.info = FakeTranscriptionInfo()
        self.inference_error: Exception | None = None
        # Rozpoznawanie języka: co „słyszy" atrapa i ile razy ją o to pytano.
        self.detected_languages: list[tuple[str, float]] = [("en", 0.99)]
        self.detection_error: Exception | None = None
        self.language_detections = 0
        FakeWhisperModel.instances.append(self)

    def detect_language(self, audio: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        self.language_detections += 1
        if self.detection_error is not None:
            raise self.detection_error
        best, probability = self.detected_languages[0]
        return best, probability, list(self.detected_languages)

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[FakeSegment], FakeTranscriptionInfo]:
        self.calls.append({"samples": audio.size, "kwargs": kwargs})
        if self.inference_error is not None:
            raise self.inference_error
        return list(self.segments), self.info


class FakeFasterWhisper(types.ModuleType):
    """Atrapa modułu ``faster_whisper``.

    ``fail_on_devices``          — konstruktor modelu rzuca wyjątkiem,
    ``fail_inference_on_devices`` — konstruktor przechodzi, ale inferencja pada.
    Ten drugi wariant odtwarza realne zachowanie CTranslate2: biblioteki CUDA
    ładują się dopiero przy pierwszym liczeniu, więc brak cuBLAS ujawnia się
    długo po utworzeniu modelu.
    """

    def __init__(
        self,
        *,
        fail_on_devices: Sequence[str] = (),
        fail_inference_on_devices: Sequence[str] = (),
    ) -> None:
        super().__init__("faster_whisper")
        self._fail_on_devices = set(fail_on_devices)
        self._fail_inference_on_devices = set(fail_inference_on_devices)

    def WhisperModel(self, *args: Any, **kwargs: Any) -> FakeWhisperModel:  # noqa: N802
        device = kwargs.get("device")
        if device in self._fail_on_devices:
            raise RuntimeError(f"atrapa: brak obsługi urządzenia {device}")
        model = FakeWhisperModel(*args, **kwargs)
        if device in self._fail_inference_on_devices:
            model.inference_error = RuntimeError(
                "Library libcublas.so.12 is not found or cannot be loaded"
            )
        return model


@pytest.fixture
def fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeFasterWhisper]:
    FakeWhisperModel.instances.clear()
    module = FakeFasterWhisper()
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    yield module
    FakeWhisperModel.instances.clear()


# --------------------------------------------------------------------------- #
# Atrapa Pipera (Faza 4)
# --------------------------------------------------------------------------- #


def make_voice_files(
    directory: Path,
    name: str = "pl_PL-testowy-medium",
    *,
    sample_rate: int = 22_050,
    language: str | None = None,
    speakers: int = 1,
    with_config: bool = True,
) -> Path:
    """Utwórz parę plików udających głos Pipera (``.onnx`` + ``.onnx.json``)."""
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"{name}.onnx"
    model.write_bytes(b"nie-jest-to-prawdziwy-model")
    if with_config:
        code = language if language is not None else name.split("-", 1)[0]
        (directory / f"{name}.onnx.json").write_text(
            json.dumps(
                {
                    "audio": {"sample_rate": sample_rate},
                    "language": {"code": code},
                    "num_speakers": speakers,
                }
            ),
            encoding="utf-8",
        )
    return model


class FakePiperVoice:
    """Atrapa ``piper.PiperVoice`` w wariancie STARSZEGO API (surowe bajty)."""

    instances: ClassVar[list[FakePiperVoice]] = []

    def __init__(self, model_path: str, config_path: str | None = None) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.calls: list[dict[str, Any]] = []
        self.config = types.SimpleNamespace(sample_rate=22_050)
        self.chunks_per_call = 3
        self.error: Exception | None = None
        FakePiperVoice.instances.append(self)

    @classmethod
    def load(cls, model_path: str, config_path: str | None = None, **kwargs: Any) -> FakePiperVoice:
        return cls(model_path, config_path)

    def synthesize_stream_raw(
        self, text: str, speaker_id: int | None = None, length_scale: float | None = None
    ) -> Iterator[bytes]:
        self.calls.append({"text": text, "speaker_id": speaker_id, "length_scale": length_scale})
        if self.error is not None:
            raise self.error
        # Ton o długości proporcjonalnej do tekstu — dzięki temu test widzi,
        # że wypowiedziano właśnie ten fragment, a nie inny.
        for index in range(self.chunks_per_call):
            samples = make_tone(160 * (len(text) + index + 1))
            yield samples.tobytes()


class FakeAudioChunk:
    """Atrapa obiektu ``AudioChunk`` z NOWSZEGO API pakietu piper."""

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self.audio_int16_bytes = samples.tobytes()
        self.sample_rate = sample_rate


class FakeSynthesisConfig:
    """Atrapa ``piper.SynthesisConfig`` — tak wygląda API od wydania 1.3."""

    def __init__(
        self, length_scale: float | None = None, speaker_id: int | None = None
    ) -> None:
        self.length_scale = length_scale
        self.speaker_id = speaker_id


class FakeModernPiperVoice(FakePiperVoice):
    """Atrapa ``piper.PiperVoice`` w wariancie NOWSZEGO API (obiekty AudioChunk).

    Nowsze wydania nie mają ``synthesize_stream_raw``, a parametry przyjmują
    w ``syn_config`` — to wariant, który realnie instaluje dziś ``pip``.
    """

    def __init__(self, model_path: str, config_path: str | None = None) -> None:
        super().__init__(model_path, config_path)
        self.sample_rate = 24_000

    synthesize_stream_raw = None  # type: ignore[assignment]

    def synthesize(
        self, text: str, syn_config: FakeSynthesisConfig | None = None
    ) -> Iterator[FakeAudioChunk]:
        self.calls.append(
            {
                "text": text,
                "length_scale": getattr(syn_config, "length_scale", None),
                "speaker_id": getattr(syn_config, "speaker_id", None),
            }
        )
        if self.error is not None:
            raise self.error
        for index in range(self.chunks_per_call):
            yield FakeAudioChunk(make_tone(160 * (index + 1)), self.sample_rate)


class FakePiper(types.ModuleType):
    """Atrapa modułu ``piper``.

    ``__spec__`` nie jest ozdobą: ``config.detect_dependencies`` sprawdza
    obecność pakietów przez ``importlib.util.find_spec``, a moduł bez specyfikacji
    wygląda dla niego na niezainstalowany.
    """

    def __init__(
        self,
        voice_class: type[FakePiperVoice] = FakePiperVoice,
        *,
        synthesis_config: type[FakeSynthesisConfig] | None = None,
    ) -> None:
        super().__init__("piper")
        self.__spec__ = importlib.machinery.ModuleSpec("piper", loader=None)
        self.PiperVoice = voice_class
        if synthesis_config is not None:
            self.SynthesisConfig = synthesis_config


@pytest.fixture
def fake_piper(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakePiper]:
    FakePiperVoice.instances.clear()
    module = FakePiper()
    monkeypatch.setitem(sys.modules, "piper", module)
    yield module
    FakePiperVoice.instances.clear()


@pytest.fixture
def fake_modern_piper(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakePiper]:
    FakePiperVoice.instances.clear()
    module = FakePiper(FakeModernPiperVoice, synthesis_config=FakeSynthesisConfig)
    monkeypatch.setitem(sys.modules, "piper", module)
    yield module
    FakePiperVoice.instances.clear()


# --------------------------------------------------------------------------- #
# Atrapy embeddingów (Faza 6)
#
# Testy NIGDY nie pobierają prawdziwego modelu: to setki megabajtów i sieć.
# Zamiast tego liczymy wektor deterministycznie z samego tekstu — wspólne słowa
# dają wysokie podobieństwo, więc da się sprawdzić realne zachowanie
# wyszukiwania, a wynik jest identyczny na każdej maszynie.
# --------------------------------------------------------------------------- #

FAKE_EMBEDDING_DIM = 64

# Prawdziwy tokenizator też gubi interpunkcję — „imie:" i „imie" to to samo słowo.
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def fake_vector(text: str, dimension: int = FAKE_EMBEDDING_DIM) -> list[float]:
    """Wektor „worka słów": każde słowo trafia we własny wymiar, potem normalizacja.

    Podobieństwo dwóch takich wektorów to po prostu udział wspólnych słów —
    zachowanie przewidywalne i wystarczające, żeby sprawdzić CAŁĄ mechanikę
    wyszukiwania (indeks, próg, ranking) bez ładowania jakiegokolwiek modelu.
    Prawdziwy model rozpoznaje jeszcze synonimy; atrapa nie musi, bo testy
    sprawdzają kod, a nie jakość modelu.

    ``hash()`` byłby tu pułapką: Python losuje jego ziarno przy każdym starcie
    procesu (PYTHONHASHSEED), więc test bywałby raz zielony, raz czerwony.
    ``zlib.crc32`` daje tę samą liczbę zawsze, wszędzie i w każdej wersji Pythona.
    """
    values = [0.0] * dimension
    for word in _WORDS.findall(text.lower()):
        values[zlib.crc32(word.encode("utf-8")) % dimension] += 1.0
    length = sum(value * value for value in values) ** 0.5
    if length == 0.0:
        # Pusty tekst: wektor „gdzieś", byle nie zerowy — zerowy nie ma kierunku.
        values[0] = 1.0
        return values
    return [value / length for value in values]


class FakeEmbeddingProvider:
    """Atrapa dostawcy embeddingów zliczająca wywołania."""

    def __init__(
        self, name: str = "atrapa-embeddingow", dimension: int = FAKE_EMBEDDING_DIM
    ) -> None:
        self._name = name
        self._dimension = dimension
        self.calls: list[list[str]] = []
        self.error: Exception | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def describe(self) -> str:
        return f"atrapa: {self._name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [fake_vector(text, self._dimension) for text in texts]


class FakeSentenceTransformer:
    """Atrapa klasy ``SentenceTransformer`` — zapisuje, z czym ją wywołano."""

    instances: ClassVar[list[FakeSentenceTransformer]] = []

    def __init__(
        self,
        model_name_or_path: str,
        *,
        cache_folder: str | None = None,
        device: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.cache_folder = cache_folder
        self.device = device
        self.local_files_only = local_files_only
        self.encoded: list[list[str]] = []
        FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return FAKE_EMBEDDING_DIM

    def encode(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        self.encoded.append(list(texts))
        # Celowo BEZ normalizacji: prawdziwy model też jej nie gwarantuje,
        # a kod ma normalizować wynik sam.
        return [[value * 3.0 for value in fake_vector(text)] for text in texts]


class FakeSentenceTransformersModule(types.ModuleType):
    """Atrapa pakietu ``sentence_transformers`` (z ``__spec__`` dla find_spec)."""

    def __init__(self, model_class: type[FakeSentenceTransformer] | None = None) -> None:
        super().__init__("sentence_transformers")
        self.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", loader=None)
        self.SentenceTransformer = model_class or FakeSentenceTransformer


@pytest.fixture
def fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[FakeSentenceTransformersModule]:
    FakeSentenceTransformer.instances.clear()
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    yield module
    FakeSentenceTransformer.instances.clear()


@pytest.fixture
def no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Maszyna bez żadnego silnika embeddingów."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    import brain.embeddings

    monkeypatch.setattr(brain.embeddings, "available_embedding_engines", list)


def make_embedding_model_dir(root: Path, name: str = "atrapa-modelu") -> Path:
    """Utwórz katalog wyglądający na kompletny model sentence-transformers."""
    directory = root / f"models--sentence-transformers--{name}" / "snapshots" / "abc123"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "modules.json").write_text("[]", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"nie-jest-to-prawdziwy-model")
    reference = root / f"models--sentence-transformers--{name}" / "refs"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "main").write_text("abc123", encoding="utf-8")
    return directory


# --------------------------------------------------------------------------- #
# Narzędzia, uprawnienia i tool calling (Faza 7)
#
# Testy nie dotykają systemu: narzędzia są atrapami, potwierdzenia odpowiada
# atrapa brokera (nikt nie pisze na klawiaturze), a „model" to scenariusz
# odpowiedzi. Zegar jest zamrożony — inaczej test formatowania daty zależałby od
# tego, która jest godzina na maszynie CI.
# --------------------------------------------------------------------------- #

FROZEN_MOMENT = datetime(2026, 8, 17, 13, 42, 30, tzinfo=timezone.utc)


def frozen_clock(moment: datetime = FROZEN_MOMENT) -> Callable[[], datetime]:
    """Zegar zwracający zawsze tę samą chwilę."""
    return lambda: moment


class SpyBroker:
    """Kanał potwierdzeń, który odpowiada z góry ustaloną decyzją i notuje pytania."""

    def __init__(
        self,
        *,
        approve: bool = True,
        reason: str = "",
        available: bool = True,
        answers: Sequence[bool] | None = None,
    ) -> None:
        self._approve = approve
        self._reason = reason
        self._available = available
        self._answers = list(answers or [])
        self.requests: list[Any] = []

    @property
    def channel(self) -> str:
        return "atrapa"

    @property
    def available(self) -> bool:
        return self._available

    async def ask(self, request: Any) -> Any:
        from security.confirm import ConfirmationOutcome

        self.requests.append(request)
        approve = self._answers.pop(0) if self._answers else self._approve
        if approve:
            return ConfirmationOutcome.approve(channel=self.channel, reason=self._reason)
        return ConfirmationOutcome.deny(
            channel=self.channel, reason=self._reason or "odmowa w teście"
        )


def make_fake_tool(
    *,
    name: str = "test.echo",
    risk: Any = None,
    result: Any = None,
    error: Exception | None = None,
    delay_s: float = 0.0,
    args_model: type[Any] | None = None,
    dynamic: Any = None,
    available: tuple[bool, str] = (True, ""),
    timeout_s: float = 5.0,
) -> Any:
    """Narzędzie-atrapa: zapisuje wywołania, może zwlekać albo rzucić wyjątkiem."""
    import asyncio

    from pydantic import Field

    from security.risk import RiskLevel
    from tools.base import ToolArgs, ToolResult, make_tool

    class EchoArgs(ToolArgs):
        text: str = Field(default="cokolwiek", max_length=100)

    model = args_model or EchoArgs
    calls: list[Any] = []

    async def run(args: Any, ctx: Any) -> Any:
        calls.append(args)
        if delay_s:
            await asyncio.sleep(delay_s)
        if error is not None:
            raise error
        if result is not None:
            return result
        return ToolResult.success({"echo": getattr(args, "text", "")}, display="echo")

    tool = make_tool(
        name=name,
        description=f"atrapa narzędzia {name}",
        args_model=model,
        risk=risk or RiskLevel.SAFE,
        function=run,
        timeout_s=timeout_s,
        risk_hook=(lambda _args: dynamic) if dynamic is not None else None,
        availability_hook=(lambda: available) if available != (True, "") else None,
    )
    tool.calls = calls  # type: ignore[attr-defined]
    return tool


@dataclasses.dataclass
class LLMStep:
    """Jedno przejście „modelu": co napisze i o jakie narzędzia poprosi."""

    chunks: Sequence[str] = ()
    tool_calls: Sequence[dict[str, Any]] = ()


class FakeToolLLM:
    """Atrapa klienta Ollamy odgrywająca scenariusz przejść (Faza 7).

    Przy każdym wywołaniu ``stream_chat`` bierze kolejny krok scenariusza; ostatni
    powtarza się, gdyby model był pytany więcej razy niż przewidziano.
    """

    def __init__(self, steps: Sequence[LLMStep]) -> None:
        self.steps = list(steps) or [LLMStep(chunks=("...",))]
        self.index = 0
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(
        self,
        messages: Sequence[Any],
        *,
        system: str | None = None,
        on_thinking: Any = None,
        tools: Sequence[dict[str, Any]] | None = None,
        collect: Any = None,
        context: str | None = None,
    ) -> Any:
        step = self.steps[min(self.index, len(self.steps) - 1)]
        self.index += 1
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": list(tools or []),
                "context": context,
                "roles": [message.role for message in messages],
            }
        )
        if collect is not None:
            collect.tool_calls.extend(dict(call) for call in step.tool_calls)
        for chunk in step.chunks:
            if collect is not None:
                collect.content += chunk
            yield chunk

    async def chat(self, messages: Sequence[Any], *, system: str | None = None) -> str:
        chunks = [chunk async for chunk in self.stream_chat(messages, system=system)]
        return "".join(chunks)


@pytest.fixture
def no_piper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Maszyna bez Pipera: ani pakietu, ani programu.

    Podmieniamy funkcję we WSZYSTKICH modułach, które zaimportowały ją do
    własnej przestrzeni nazw — inaczej test przechodziłby albo nie zależnie od
    tego, czy deweloper ma akurat zainstalowany pakiet ``piper-tts``.
    """
    import audio.dependencies
    import audio.tts
    import config

    monkeypatch.setitem(sys.modules, "piper", None)
    for module in (config, audio.tts, audio.dependencies):
        monkeypatch.setattr(module, "find_piper_binary", lambda *args, **kwargs: None)
