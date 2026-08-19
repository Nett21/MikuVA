"""Synteza mowy (Faza 4): wspólny interfejs silników + implementacja Pipera.

Moduł ma dwie warstwy i to rozdzielenie jest tu najważniejsze:

* :class:`TTSProvider` — abstrakcyjny kontrakt „tekst → strumień PCM". Nie wie
  nic o Piperze, o plikach na dysku ani o karcie dźwiękowej. Kolejne silniki
  (XTTS, konwersja głosu RVC, docelowy głos Miku) dopisują własną klasę i
  rejestrują ją przez :func:`register_tts_provider` — reszta programu nie
  zmienia ani jednej linijki.
* :class:`PiperTTSProvider` — jedyna dziś istniejąca implementacja. Liczy
  lokalnie, na CPU, bez sieci.

Czego ten moduł NIE robi:

* nie odtwarza dźwięku — od tego jest ``audio/output.py`` (inne urządzenie,
  inny cykl życia, a w przyszłości barge-in i ducking),
* nie wie, na jakim systemie działa i gdzie leżą pliki — katalogi głosów oraz
  lokalizację binarki wyznacza ``config.py`` (:func:`config.piper_voice_directories`,
  :func:`config.find_piper_binary`). Dzięki temu w całym pliku nie ma ani jednej
  ścieżki zapisanej na sztywno.

Wybór głosu należy do użytkownika i mieszka w ``config/user_settings.json``:

* ``piper_model``  — nazwa modelu (``pl_PL-darkman-medium``) albo ścieżka do
  pliku ``.onnx``; puste = wybierz automatycznie głos pasujący do języka,
* ``piper_voices`` — opcjonalnie osobny głos per język, np.
  ``{"pl": "pl_PL-gosia-medium", "en": "en_US-amy-medium"}``,
* ``voice_speed``, ``voice_volume``, ``piper_speaker``.

Podmiana głosu na dowolny inny model Pipera to edycja pliku ustawień — nigdy
zmiana kodu. Plik jest wczytywany przy każdej wypowiedzi, więc działa bez
restartu programu.

Dwa sposoby liczenia (wybierane automatycznie, oba strumieniowe):

1. pakiet Pythona ``piper-tts`` (``pip install piper-tts``) — najszybszy start,
2. program ``piper`` w ``PATH`` — dla instalacji z gotowego archiwum.

Brak jednego i drugiego nie jest błędem: mowa się nie włącza, a asystent
odpowiada tekstem.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import subprocess  # nosec B404 - uruchamiamy wyłącznie binarkę Pipera, bez powłoki
import threading
import time
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from config import (
    Settings,
    UserSettings,
    find_piper_binary,
    get_settings,
    get_user_settings,
    pip_install_hint,
    piper_voice_directories,
    resolve_speech_language,
    subprocess_no_window_kwargs,
)
from i18n import t

logger = logging.getLogger(__name__)

# Częstotliwość używana, gdy model nie mówi, w czym liczy. Wartość domyślna
# Pipera dla głosów „medium"; prawdziwa wartość jest czytana z pliku
# ``<model>.onnx.json`` i tylko jego brak sprowadza nas tutaj.
DEFAULT_PIPER_SAMPLE_RATE: Final[int] = 22_050

# Ile bajtów czytamy naraz ze strumienia binarki (0,1 s dźwięku 16-bitowego).
_PROCESS_READ_BYTES: Final[int] = 4_096


class TTSError(RuntimeError):
    """Błąd syntezy mowy z komunikatem gotowym dla użytkownika."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


class TTSUnavailableError(TTSError):
    """Silnik mowy nie da się uruchomić na tej maszynie (brak pakietu, głosu, sprzętu)."""


# --------------------------------------------------------------------------- #
# Dane
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    """Kawałek zsyntezowanego dźwięku: mono, int16, własna częstotliwość.

    Każdy fragment niesie swoją częstotliwość próbkowania, bo różne głosy (i
    różne silniki) liczą w różnych: 16 000, 22 050 albo 24 000 Hz. Przeliczeniem
    na to, co przyjmie karta dźwiękowa, zajmuje się dopiero warstwa odtwarzania.
    """

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.samples.size / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return self.samples.size == 0


@dataclass(frozen=True, slots=True)
class VoiceModel:
    """Głos Pipera znaleziony na dysku (plik ``.onnx`` + opis ``.onnx.json``)."""

    path: Path
    config_path: Path | None
    name: str
    language: str
    sample_rate: int
    quality: str = ""
    speakers: int = 1

    @property
    def is_multispeaker(self) -> bool:
        return self.speakers > 1

    def describe(self) -> str:
        parts = [self.name]
        if self.language:
            parts.append(self.language)
        if self.quality:
            parts.append(self.quality)
        parts.append(f"{self.sample_rate} Hz")
        if self.is_multispeaker:
            parts.append(f"{self.speakers} głosów")
        return f"{' / '.join(parts)}"


# --------------------------------------------------------------------------- #
# Kontrakt silnika mowy
# --------------------------------------------------------------------------- #


class TTSProvider(ABC):
    """Wspólny interfejs wszystkich silników mowy.

    Implementacja musi dostarczyć wyłącznie :meth:`synthesize`. Reszta metod ma
    sensowne zachowanie domyślne, więc nowy silnik to zwykle kilkadziesiąt
    linijek.

    Kontrakt:

    * :meth:`synthesize` **strumieniuje** — oddaje pierwszy fragment tak
      szybko, jak potrafi, zamiast czekać z całym zdaniem. Na tym stoi
      odtwarzanie zaczynające się, zanim model językowy skończy odpowiedź,
    * fragmenty są zawsze mono ``int16``; częstotliwość niesie każdy z osobna,
    * metoda nie odtwarza dźwięku i nie dotyka urządzeń,
    * błąd możliwy do pokazania człowiekowi to :class:`TTSError`; brak silnika
      na tej maszynie to :class:`TTSUnavailableError`,
    * :meth:`cancel` może przyjść z innego wątku i ma przerwać trwającą syntezę.

    Planowane implementacje (Faza 15 i dalej) — to dla nich powstał ten
    interfejs, dziś NIE są zaimplementowane:

    * ``XttsTTSProvider`` — klonowanie głosu z próbki, GPU opcjonalne,
    * ``RvcVoiceProvider`` — konwersja barwy: opakowuje **inny** dostawcę
      (``base: TTSProvider``), przepuszcza jego fragmenty przez model RVC i
      oddaje dalej. Dlatego fragment niesie własną częstotliwość, a interfejs
      jest strumieniowy — inaczej takie opakowanie byłoby niemożliwe,
    * docelowy głos postaci — najpewniej RVC nałożone na Pipera.
    """

    #: Nazwa silnika używana w ``voice_engine`` i w rejestrze dostawców.
    name: str = "abstract"

    # --- cykl życia ------------------------------------------------------- #

    def load(self) -> None:  # noqa: B027 - hak opcjonalny: nie każdy silnik ma co ładować
        """Przygotuj silnik (wczytaj model). Domyślnie nic nie robi."""

    def unload(self) -> None:  # noqa: B027 - jw.: brak zasobów do zwolnienia jest poprawny
        """Zwolnij zasoby. Domyślnie nic nie robi."""

    def close(self) -> None:
        self.unload()

    @property
    def is_loaded(self) -> bool:
        return True

    def __enter__(self) -> TTSProvider:
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- opis ------------------------------------------------------------- #

    @property
    def sample_rate(self) -> int:
        """Częstotliwość, w której silnik zwykle liczy (0 = zmienna/nieznana)."""
        return 0

    def voice_name(self) -> str:
        """Nazwa aktualnie używanego głosu (do ``/status`` i GUI)."""
        return ""

    def describe(self) -> str:
        voice = self.voice_name()
        return f"{self.name} ({voice})" if voice else self.name

    def supports_language(self, language: str | None) -> bool:
        """Czy silnik ma czym powiedzieć tekst w tym języku? Domyślnie tak."""
        del language  # domyślna odpowiedź nie zależy od języka
        return True

    @property
    def is_speaking_enabled(self) -> bool:
        """Czy ten dostawca w ogóle wydaje dźwięk (``NullTTSProvider`` nie)."""
        return True

    # --- synteza ---------------------------------------------------------- #

    @abstractmethod
    def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
        """Zamień tekst na kolejne fragmenty dźwięku (leniwie, strumieniowo)."""

    def synthesize_all(
        self, text: str, *, language: str | None = None
    ) -> SpeechChunk | None:
        """Cała wypowiedź w jednym kawałku — wygodne w testach i przy zapisie do pliku."""
        collected = [
            chunk for chunk in self.synthesize(text, language=language) if not chunk.is_empty
        ]
        if not collected:
            return None
        rate = collected[0].sample_rate
        if any(chunk.sample_rate != rate for chunk in collected):
            raise TTSError(
                t("tts.rate_mismatch"),
                hint=t("tts.rate_mismatch_hint"),
            )
        return SpeechChunk(
            samples=np.concatenate([chunk.samples for chunk in collected]), sample_rate=rate
        )

    def warmup(self, text: str = "…") -> bool:
        """Wymuś załadowanie modelu krótką syntezą. Nigdy nie rzuca."""
        try:
            for _ in self.synthesize(text):
                pass
        except Exception as exc:  # pragma: no cover - zależne od silnika
            logger.debug("Rozgrzewanie silnika mowy nie powiodło się: %s", exc)
            return False
        return True

    def cancel(self) -> None:  # noqa: B027 - hak opcjonalny: nie każdy silnik da się przerwać
        """Przerwij trwającą syntezę (wywoływane z innego wątku)."""


class NullTTSProvider(TTSProvider):
    """Silnik, który milczy — dla ``voice_engine: "none"`` i dla testów."""

    name = "none"

    def __init__(self, reason: str = "mowa wyłączona w ustawieniach") -> None:
        self._reason = reason

    @property
    def is_speaking_enabled(self) -> bool:
        return False

    def describe(self) -> str:
        return f"brak ({self._reason})"

    def synthesize(
        self, text: str, *, language: str | None = None
    ) -> Iterator[SpeechChunk]:
        del text, language  # milczenie nie zależy od tego, co miało zostać powiedziane
        return iter(())


# --------------------------------------------------------------------------- #
# Wyszukiwanie głosów Pipera
# --------------------------------------------------------------------------- #


def _read_voice_config(config_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug("Nie udało się odczytać opisu głosu %s: %s", config_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _language_from_name(name: str) -> str:
    """``pl_PL-darkman-medium`` -> ``pl``; nierozpoznane nazwy dają pusty łańcuch."""
    head = name.split("-", 1)[0]
    code = head.replace("_", "-").split("-", 1)[0].lower()
    return code if code.isalpha() and len(code) == 2 else ""


def _quality_from_name(name: str) -> str:
    parts = name.split("-")
    tail = parts[-1].lower() if len(parts) > 2 else ""
    return tail if tail in ("x_low", "x-low", "low", "medium", "high") else ""


def describe_voice_file(path: Path) -> VoiceModel:
    """Zbuduj opis głosu na podstawie pliku ``.onnx`` i towarzyszącego JSON-a."""
    # Piper trzyma opis obok modelu jako <model>.onnx.json; starsze paczki
    # bywają zapisane jako <model>.json — akceptujemy oba, bo to nic nie kosztuje.
    candidates = (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json"))
    config_path = next((item for item in candidates if item.is_file()), None)
    data = _read_voice_config(config_path) if config_path is not None else {}

    name = path.name
    if name.lower().endswith(".onnx"):
        name = name[: -len(".onnx")]

    def block(key: str) -> dict[str, Any]:
        """Zagnieżdżony obiekt z opisu głosu — zawsze słownik, choćby pusty."""
        value = data.get(key)
        return value if isinstance(value, dict) else {}

    raw_rate = block("audio").get("sample_rate")
    try:
        sample_rate = int(raw_rate) if raw_rate else 0
    except (TypeError, ValueError):
        sample_rate = 0

    language_block = block("language")
    espeak_block = block("espeak")
    raw_language = ""
    sources = (
        language_block.get("code"),
        language_block.get("family"),
        espeak_block.get("voice"),
    )
    for source in sources:
        if isinstance(source, str) and source.strip():
            raw_language = source.strip()
            break
    language = raw_language.replace("_", "-").split("-", 1)[0].lower()[:2]
    language = language or _language_from_name(name)

    try:
        speakers = int(data.get("num_speakers", 1) or 1)
    except (TypeError, ValueError):
        speakers = 1

    return VoiceModel(
        path=path,
        config_path=config_path,
        name=name,
        language=language,
        sample_rate=sample_rate or DEFAULT_PIPER_SAMPLE_RATE,
        quality=_quality_from_name(name),
        speakers=max(1, speakers),
    )


def iter_piper_voices(settings: Settings | None = None) -> list[VoiceModel]:
    """Wszystkie głosy Pipera widoczne na tej maszynie (bez duplikatów nazw).

    Katalogi do przeszukania wyznacza ``config.piper_voice_directories`` —
    ten moduł nie wie, gdzie one są. Przeszukiwanie jest płytkie plus jeden
    poziom podkatalogów, bo oficjalne paczki głosów rozpakowują się do
    ``<język>/<nazwa>/`` i nie ma sensu schodzić głębiej.
    """
    found: dict[str, VoiceModel] = {}
    for directory in piper_voice_directories(settings):
        try:
            if not directory.is_dir():
                continue
            paths = sorted(directory.glob("*.onnx")) + sorted(directory.glob("*/**/*.onnx"))
        except OSError as exc:  # pragma: no cover - zależne od uprawnień
            logger.debug("Nie udało się przeszukać %s: %s", directory, exc)
            continue
        for path in paths:
            try:
                if not path.is_file():
                    continue
            except OSError:  # pragma: no cover - dowiązanie do nieistniejącego pliku
                continue
            voice = describe_voice_file(path)
            found.setdefault(voice.name.lower(), voice)
    return list(found.values())


def _voice_from_path(reference: str, settings: Settings | None = None) -> VoiceModel | None:
    """Spróbuj potraktować wpis użytkownika jako ścieżkę do pliku ``.onnx``."""
    raw = reference.strip()
    if not raw:
        return None

    expanded = Path(os.path.expandvars(raw)).expanduser()
    candidates: list[Path] = [expanded]
    if not expanded.is_absolute():
        # Ścieżka względna liczy się od katalogów z głosami, a dopiero potem od
        # katalogu roboczego — ten ostatni zależy od tego, skąd program odpalono.
        candidates = [directory / expanded for directory in piper_voice_directories(settings)]
        candidates.append(expanded)

    for candidate in candidates:
        try:
            if candidate.is_file():
                return describe_voice_file(candidate)
        except OSError:  # pragma: no cover - zależne od uprawnień
            continue
    return None


def find_piper_voice(
    reference: str, settings: Settings | None = None, *, voices: Sequence[VoiceModel] | None = None
) -> VoiceModel | None:
    """Znajdź głos po nazwie albo ścieżce. ``None`` = nie ma takiego głosu.

    Porównanie nazw ignoruje wielkość liter, bo ta sama paczka głosów na
    Linuksie i na Windowsie bywa rozpakowana z inną wielkością znaków.
    """
    raw = reference.strip()
    if not raw:
        return None

    if raw.lower().endswith(".onnx") or any(separator in raw for separator in ("/", "\\")):
        from_path = _voice_from_path(raw, settings)
        if from_path is not None:
            return from_path

    catalogue = list(voices) if voices is not None else iter_piper_voices(settings)
    wanted = raw.lower()
    for voice in catalogue:
        if voice.name.lower() == wanted:
            return voice
    for voice in catalogue:
        if voice.name.lower().startswith(wanted):
            return voice
    return _voice_from_path(raw, settings)


def select_piper_voice(
    settings: Settings | None = None,
    user_settings: UserSettings | None = None,
    *,
    language: str | None = None,
    voices: Sequence[VoiceModel] | None = None,
) -> VoiceModel | None:
    """Wybierz głos dla danego języka zgodnie z ustawieniami użytkownika.

    Kolejność:

    1. ``piper_voices[<język>]`` — jawny wybór dla tego języka,
    2. ``piper_model`` — jeden głos na wszystko,
    3. automat: dowolny zainstalowany głos w tym języku,
    4. automat: cokolwiek, co jest (lepiej powiedzieć z obcym akcentem niż milczeć).
    """
    active = settings or get_settings()
    user = user_settings if user_settings is not None else get_user_settings()
    catalogue = list(voices) if voices is not None else iter_piper_voices(active)

    preferred = language or resolve_speech_language(active, user) or active.language or ""
    wanted_language = preferred.lower()[:2]

    configured = user.voice_for_language(wanted_language)
    if configured:
        match = find_piper_voice(configured, active, voices=catalogue)
        if match is not None:
            return match
        logger.warning(
            "Nie znaleziono głosu %r — sprawdź piper_model w config/user_settings.json "
            "albo dołóż plik .onnx do jednego z katalogów: %s",
            configured,
            ", ".join(str(item) for item in piper_voice_directories(active)[:3]),
        )

    if wanted_language:
        for voice in catalogue:
            if voice.language == wanted_language:
                return voice

    return catalogue[0] if catalogue else None


# --------------------------------------------------------------------------- #
# Sposoby liczenia Pipera
# --------------------------------------------------------------------------- #


class PiperBackend(ABC):
    """Sposób, w jaki wołamy Pipera: pakiet Pythona albo osobny program."""

    name: str = "piper"

    @abstractmethod
    def stream(
        self,
        text: str,
        voice: VoiceModel,
        *,
        speaker: int = 0,
        length_scale: float = 1.0,
        timeout_s: float = 60.0,
    ) -> Iterator[SpeechChunk]:
        """Zsyntezuj tekst wskazanym głosem, oddając fragmenty na bieżąco."""

    def describe(self) -> str:
        return self.name

    def cancel(self) -> None:  # noqa: B027 - hak opcjonalny, patrz TTSProvider.cancel
        """Przerwij trwającą syntezę."""

    def close(self) -> None:  # noqa: B027 - jw.
        """Zwolnij zasoby (procesy, załadowane modele)."""


def _load_piper_module() -> ModuleType:
    try:
        import piper  # noqa: PLC0415 - import celowo leniwy (ciężka biblioteka)
    except ImportError as exc:
        raise TTSUnavailableError(
            "Pakiet 'piper-tts' nie jest zainstalowany.", hint=pip_install_hint()
        ) from exc
    except OSError as exc:  # onnxruntime potrafi nie znaleźć bibliotek natywnych
        raise TTSUnavailableError(
            t("tts.package_failed", error=exc),
            hint=t("tts.package_hint"),
        ) from exc
    return piper


def _as_int16(samples: np.ndarray) -> np.ndarray:
    """Sprowadź próbki do PCM int16 (float w zakresie [-1, 1] albo już int16)."""
    if samples.dtype == np.int16:
        return samples
    if np.issubdtype(samples.dtype, np.floating):
        return (np.clip(samples, -1.0, 1.0) * 32_767.0).astype(np.int16)
    return samples.astype(np.int16)


def _samples_from_piper_item(item: Any) -> tuple[np.ndarray, int]:
    """Sprowadź to, co oddał Piper, do pary (próbki int16, częstotliwość).

    API pakietu zmieniało się między wydaniami: starsze zwracały surowe bajty,
    nowsze obiekt ``AudioChunk``. Zamiast wymuszać jedną wersję, przyjmujemy
    każdą postać, którą da się jednoznacznie zinterpretować — inaczej aktualizacja
    ``pip install -U piper-tts`` psułaby mowę bez zmiany w naszym kodzie.
    """
    rate = 0
    raw_rate = getattr(item, "sample_rate", None)
    if isinstance(raw_rate, int) and raw_rate > 0:
        rate = raw_rate

    if isinstance(item, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(item), dtype=np.int16).copy(), rate

    if isinstance(item, np.ndarray):
        return _as_int16(item), rate

    for attribute in ("audio_int16_bytes", "audio_int16_array", "audio_float_array"):
        payload = getattr(item, attribute, None)
        if payload is None:
            continue
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return np.frombuffer(bytes(payload), dtype=np.int16).copy(), rate
        return _as_int16(np.asarray(payload)), rate

    raise TTSError(
        t("tts.unknown_chunk", kind=type(item).__name__),
        hint=t("tts.unknown_chunk_hint"),
    )


class PiperPythonBackend(PiperBackend):
    """Synteza przez pakiet ``piper-tts`` — model żyje w naszym procesie."""

    name = "piper-tts (pakiet Pythona)"

    def __init__(self, *, module: ModuleType | None = None) -> None:
        self._module = module
        self._voices: dict[str, Any] = {}
        self._cancelled = threading.Event()

    def _piper(self) -> ModuleType:
        if self._module is None:
            self._module = _load_piper_module()
        return self._module

    def _voice_object(self, voice: VoiceModel) -> Any:
        key = str(voice.path)
        loaded = self._voices.get(key)
        if loaded is not None:
            return loaded

        piper = self._piper()
        voice_class = getattr(piper, "PiperVoice", None)
        if voice_class is None:  # pragma: no cover - nieoczekiwana wersja pakietu
            raise TTSUnavailableError(
                t("tts.no_voice_class"),
                hint=t("tts.update_package"),
            )
        try:
            loaded = voice_class.load(
                str(voice.path),
                config_path=str(voice.config_path) if voice.config_path else None,
            )
        except Exception as exc:
            raise TTSError(
                t("tts.voice_load_failed", name=voice.name, error=exc),
                hint=t("tts.voice_load_hint", path=voice.path),
            ) from exc

        self._voices[key] = loaded
        return loaded

    def _synthesis_config(self, *, speaker: int, length_scale: float) -> Any | None:
        """Obiekt ``SynthesisConfig``, jeśli zainstalowana wersja pakietu go zna."""
        config_class = getattr(self._piper(), "SynthesisConfig", None)
        if config_class is None:
            return None
        try:
            return config_class(
                length_scale=length_scale,
                speaker_id=speaker if speaker > 0 else None,
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - zmiana pól w pakiecie
            logger.debug("Nie udało się zbudować SynthesisConfig: %s", exc)
            return None

    def _raw_stream(self, loaded: Any, text: str, *, speaker: int, length_scale: float) -> Any:
        """Wywołaj metodę syntezy właściwą dla zainstalowanej wersji pakietu.

        Zwracany typ jest świadomie ``Any``: każde wydanie pakietu oddaje coś
        innego (bajty albo obiekty ``AudioChunk``), a ujednolica to dopiero
        :func:`_samples_from_piper_item`.
        """
        legacy = getattr(loaded, "synthesize_stream_raw", None)
        if callable(legacy):
            try:
                return legacy(text, speaker_id=speaker or None, length_scale=length_scale)
            except TypeError:  # starsze wydania bez tych parametrów
                return legacy(text)

        modern = getattr(loaded, "synthesize", None)
        if callable(modern):
            # Nowsze wydania przyjmują parametry w obiekcie konfiguracji zamiast
            # w argumentach. Bez niego tempo mowy i wybór mówcy byłyby po cichu
            # ignorowane — a to najgorszy rodzaj awarii: wszystko „działa".
            config = self._synthesis_config(speaker=speaker, length_scale=length_scale)
            if config is not None:
                try:
                    return modern(text, syn_config=config)
                except TypeError:  # pragma: no cover - jeszcze inna wersja API
                    logger.debug(
                        "piper-tts nie przyjmuje syn_config — mówię z ustawieniami domyślnymi"
                    )
            try:
                return modern(text)
            except TypeError as exc:  # pragma: no cover - kolejna zmiana API
                raise TTSError(
                    t("tts.unsupported_api", error=exc),
                    hint=t("tts.update_assistant"),
                ) from exc

        raise TTSUnavailableError(  # pragma: no cover - nieoczekiwana wersja pakietu
            "Zainstalowany pakiet piper-tts nie ma metody syntezy.",
            hint="zaktualizuj pakiet: pip install -U piper-tts",
        )

    def stream(
        self,
        text: str,
        voice: VoiceModel,
        *,
        speaker: int = 0,
        length_scale: float = 1.0,
        # Limit czasu dotyczy tylko binarki: liczenie w procesie nie ma czego ubić.
        timeout_s: float = 60.0,
    ) -> Iterator[SpeechChunk]:
        self._cancelled.clear()
        loaded = self._voice_object(voice)
        config = getattr(loaded, "config", None)
        fallback_rate = int(getattr(config, "sample_rate", 0) or voice.sample_rate)

        try:
            for item in self._raw_stream(loaded, text, speaker=speaker, length_scale=length_scale):
                if self._cancelled.is_set():
                    return
                samples, rate = _samples_from_piper_item(item)
                if samples.size:
                    yield SpeechChunk(samples=samples, sample_rate=rate or fallback_rate)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(
                t("tts.synthesis_failed", error=exc), hint=t("tts.details_in_log")
            ) from exc

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        self._voices.clear()


@dataclass(frozen=True, slots=True)
class PiperCliFlags:
    """Nazwy opcji binarki — różnią się między wydaniami Pipera.

    Starsze wydania używają podkreśleń (``--output_raw``), nowsze myślników
    (``--output-raw``). Zamiast zgadywać, czytamy ``--help`` raz i zapamiętujemy
    wynik; gdy pomoc jest niedostępna, próbujemy obu wariantów po kolei.
    """

    output_raw: str = "--output-raw"
    length_scale: str | None = None
    speaker: str | None = None
    quiet: str | None = None


def _detect_cli_flags(binary: Path) -> PiperCliFlags:
    try:
        completed = subprocess.run(  # nosec B603 - stała lista argumentów, bez powłoki
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            check=False,
            **subprocess_no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("Nie udało się odczytać pomocy programu piper: %s", exc)
        return PiperCliFlags()

    help_text = f"{completed.stdout}\n{completed.stderr}"

    def pick(*names: str) -> str | None:
        return next((name for name in names if name in help_text), None)

    return PiperCliFlags(
        output_raw=pick("--output-raw", "--output_raw") or "--output-raw",
        length_scale=pick("--length-scale", "--length_scale"),
        speaker=pick("--speaker"),
        quiet=pick("--quiet"),
    )


class PiperProcessBackend(PiperBackend):
    """Synteza przez program ``piper`` — tekst na wejściu, surowy PCM na wyjściu.

    Proces żyje tyle, co jedno zdanie: czytamy jego wyjście w miarę napływania,
    więc odtwarzanie rusza, zanim binarka skończy liczyć. Limit czasu pilnuje
    osobny timer, bo odczyt z potoku nie da się przerwać inaczej niż zamykając
    proces — a to działa tak samo na każdym systemie.
    """

    name = "piper (program)"

    def __init__(self, binary: Path, *, flags: PiperCliFlags | None = None) -> None:
        self._binary = binary
        self._flags = flags
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def describe(self) -> str:
        return f"{self.name}: {self._binary}"

    @property
    def binary(self) -> Path:
        return self._binary

    def _resolved_flags(self) -> PiperCliFlags:
        if self._flags is None:
            self._flags = _detect_cli_flags(self._binary)
            logger.debug("Opcje binarki Pipera: %s", self._flags)
        return self._flags

    def _build_command(
        self, voice: VoiceModel, *, speaker: int, length_scale: float
    ) -> list[str]:
        flags = self._resolved_flags()
        command = [str(self._binary), "--model", str(voice.path), flags.output_raw]
        if voice.config_path is not None:
            command += ["--config", str(voice.config_path)]
        if flags.length_scale and abs(length_scale - 1.0) > 0.01:
            command += [flags.length_scale, f"{length_scale:.3f}"]
        if flags.speaker and speaker > 0:
            command += [flags.speaker, str(speaker)]
        if flags.quiet:
            command.append(flags.quiet)
        return command

    def stream(
        self,
        text: str,
        voice: VoiceModel,
        *,
        speaker: int = 0,
        length_scale: float = 1.0,
        timeout_s: float = 60.0,
    ) -> Iterator[SpeechChunk]:
        self._cancelled.clear()
        command = self._build_command(voice, speaker=speaker, length_scale=length_scale)

        try:
            process = subprocess.Popen(  # nosec B603 - lista argumentów, bez powłoki
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **subprocess_no_window_kwargs(),
            )
        except OSError as exc:
            raise TTSUnavailableError(
                t("tts.spawn_failed", path=self._binary, error=exc),
                hint=t("tts.spawn_hint"),
            ) from exc

        with self._lock:
            self._process = process

        errors: list[bytes] = []
        stderr_reader = threading.Thread(
            target=self._drain_stderr, args=(process, errors), daemon=True
        )
        stderr_reader.start()

        watchdog = threading.Timer(timeout_s, self._kill, args=(process,))
        watchdog.daemon = True
        watchdog.start()

        rate = voice.sample_rate or DEFAULT_PIPER_SAMPLE_RATE
        remainder = b""
        try:
            if process.stdin is not None:
                # Kodowanie jawnie: domyślne kodowanie konsoli na Windowsie nie
                # zapisze polskich znaków, a Piper oczekuje UTF-8.
                with contextlib.suppress(OSError, ValueError):
                    process.stdin.write(text.encode("utf-8"))
                    process.stdin.flush()
                    process.stdin.close()

            stdout = process.stdout
            # read1() oddaje to, co już przyszło, zamiast czekać na pełny bufor —
            # dzięki temu odtwarzanie rusza od pierwszych milisekund dźwięku.
            read = getattr(stdout, "read1", None) or getattr(stdout, "read", None)
            while stdout is not None and read is not None:
                data = read(_PROCESS_READ_BYTES)
                if not data:
                    break
                if self._cancelled.is_set():
                    break
                buffer = remainder + data
                # Próbka to 2 bajty — nieparzysta końcówka czeka na dalszy ciąg.
                usable = len(buffer) - (len(buffer) % 2)
                remainder = buffer[usable:]
                if usable:
                    yield SpeechChunk(
                        samples=np.frombuffer(buffer[:usable], dtype=np.int16).copy(),
                        sample_rate=rate,
                    )
        finally:
            watchdog.cancel()
            self._finish(process, errors, stderr_reader)
            with self._lock:
                self._process = None

    @staticmethod
    def _drain_stderr(process: subprocess.Popen[bytes], sink: list[bytes]) -> None:
        """Czytaj stderr na bieżąco — pełny bufor potoku zablokowałby Pipera."""
        stream = process.stderr
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            for line in stream:
                if len(sink) < 50:
                    sink.append(line.rstrip())

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            logger.warning("Piper nie odpowiedział w wyznaczonym czasie — zamykam proces.")
            with contextlib.suppress(OSError):
                process.kill()

    def _finish(
        self,
        process: subprocess.Popen[bytes],
        errors: list[bytes],
        reader: threading.Thread,
    ) -> None:
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        try:
            code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - proces nie do ubicia
            self._kill(process)
            code = -1
        reader.join(timeout=1.0)
        if process.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                process.stderr.close()

        if code not in (0, None) and not self._cancelled.is_set():
            detail = b" ".join(errors[-5:]).decode("utf-8", errors="replace").strip()
            raise TTSError(
                t(
                    "tts.piper_exit",
                    code=code,
                    detail=f": {detail[:300]}" if detail else ".",
                ),
                hint=t("tts.model_mismatch_hint"),
            )

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None:
            self._kill(process)

    def close(self) -> None:
        self.cancel()


def create_piper_backend(
    settings: Settings | None = None, *, prefer: str = "auto"
) -> PiperBackend:
    """Wybierz sposób liczenia: pakiet Pythona, a gdy go nie ma — program ``piper``.

    ``prefer`` przyjmuje ``auto`` (domyślnie), ``python`` albo ``process`` —
    przydaje się w testach i przy diagnozowaniu instalacji.
    """
    active = settings or get_settings()

    if prefer in ("auto", "python"):
        try:
            module = _load_piper_module()
        except TTSUnavailableError as exc:
            if prefer == "python":
                raise
            logger.debug("Pakiet piper-tts niedostępny: %s", exc.message)
        else:
            return PiperPythonBackend(module=module)

    binary = find_piper_binary(active)
    if binary is None:
        raise TTSUnavailableError(
            t("tts.nothing_found"),
            hint=t("tts.nothing_found_hint"),
        )
    return PiperProcessBackend(binary)


# --------------------------------------------------------------------------- #
# Dostawca: Piper
# --------------------------------------------------------------------------- #


class PiperTTSProvider(TTSProvider):
    """Lokalna synteza mowy Piperem — polski, angielski i każdy inny głos.

    Głos jest wybierany przy każdej wypowiedzi na podstawie ustawień użytkownika
    i języka odpowiedzi, więc asystent może odpowiedzieć po polsku głosem
    polskim, a po angielsku angielskim — o ile oba modele są na dysku. Gdy
    modelu dla danego języka nie ma, mówi tym, który jest (raz ostrzegając),
    zamiast milczeć.
    """

    name = "piper"

    def __init__(
        self,
        settings: Settings | None = None,
        user_settings: UserSettings | None = None,
        *,
        backend: PiperBackend | None = None,
        voices: Sequence[VoiceModel] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._user_settings = user_settings
        self._backend = backend
        self._voices = list(voices) if voices is not None else None
        self._voice: VoiceModel | None = None
        self._warned_languages: set[str] = set()

    # --- ustawienia i katalog głosów -------------------------------------- #

    def _user(self) -> UserSettings:
        """Ustawienia użytkownika czytane na bieżąco (chyba że wstrzyknięto własne).

        Plik jest przeładowywany, gdy zmieni się na dysku — dzięki temu podmiana
        ``piper_model`` działa bez restartu, tak samo jak imię asystenta.
        """
        return self._user_settings if self._user_settings is not None else get_user_settings()

    def _catalogue(self) -> list[VoiceModel]:
        if self._voices is None:
            self._voices = iter_piper_voices(self._settings)
        return self._voices

    def refresh_voices(self) -> list[VoiceModel]:
        """Przeszukaj katalogi jeszcze raz (po dograniu nowego głosu)."""
        self._voices = iter_piper_voices(self._settings)
        self._voice = None
        return self._voices

    @property
    def available_voices(self) -> list[VoiceModel]:
        return list(self._catalogue())

    # --- cykl życia -------------------------------------------------------- #

    def _ensure_backend(self) -> PiperBackend:
        if self._backend is None:
            self._backend = create_piper_backend(self._settings)
        return self._backend

    def load(self) -> None:
        """Sprawdź, że jest czym i co powiedzieć. Rzuca, gdy się nie da."""
        voices = self._catalogue()
        if not voices:
            raise TTSUnavailableError(
                t("tts.no_voices"),
                hint=t("tts.no_voices_hint"),
            )
        self._ensure_backend()
        if self._voice is None:
            self._voice = self._resolve_voice(None)
        if self._voice is None:  # pragma: no cover - katalog nie jest pusty
            raise TTSUnavailableError(t("tts.no_voice_selected"))

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None and self._voice is not None

    def unload(self) -> None:
        if self._backend is not None:
            with contextlib.suppress(Exception):
                self._backend.close()
        self._backend = None
        self._voice = None

    # --- opis --------------------------------------------------------------- #

    @property
    def voice(self) -> VoiceModel | None:
        return self._voice

    @property
    def sample_rate(self) -> int:
        return self._voice.sample_rate if self._voice else DEFAULT_PIPER_SAMPLE_RATE

    def voice_name(self) -> str:
        return self._voice.name if self._voice else ""

    def describe(self) -> str:
        parts: list[str] = []
        if self._voice is not None:
            parts.append(self._voice.describe())
        else:
            parts.append("głos nie został jeszcze wybrany")
        if self._backend is not None:
            parts.append(self._backend.describe())
        speed = self._user().voice_speed
        if abs(speed - 1.0) > 0.01:
            parts.append(f"tempo {speed:.2f}x")
        return " · ".join(parts)

    def supports_language(self, language: str | None) -> bool:
        if not language:
            return True
        code = language.strip().lower()[:2]
        return any(voice.language == code for voice in self._catalogue())

    # --- wybór głosu --------------------------------------------------------- #

    def _resolve_voice(self, language: str | None) -> VoiceModel | None:
        return select_piper_voice(
            self._settings, self._user(), language=language, voices=self._catalogue()
        )

    def voice_for(self, language: str | None) -> VoiceModel | None:
        """Głos, którym zostanie powiedziany tekst w tym języku."""
        voice = self._resolve_voice(language)
        if voice is None:
            return None
        code = (language or "").strip().lower()[:2]
        unknown = code and voice.language and voice.language != code
        if unknown and code not in self._warned_languages:
            self._warned_languages.add(code)
            logger.warning(
                "Brak głosu dla języka %r — mówię głosem %s (%s). "
                "Dołóż model .onnx dla tego języka albo wpisz go w piper_voices "
                "w config/user_settings.json.",
                code,
                voice.name,
                voice.language or "?",
            )
        return voice

    # --- synteza -------------------------------------------------------------- #

    def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
        spoken = text.strip()
        if not spoken:
            return

        voice = self.voice_for(language)
        if voice is None:
            raise TTSUnavailableError(
                t("tts.no_voice_to_speak"),
                hint=t("tts.no_voice_to_speak_hint"),
            )
        self._voice = voice

        user = self._user()
        # W Piperze „length_scale" to długość, nie tempo: im większa, tym wolniej.
        length_scale = 1.0 / max(0.1, user.voice_speed)

        backend = self._ensure_backend()
        yield from backend.stream(
            spoken,
            voice,
            speaker=user.piper_speaker if voice.is_multispeaker else 0,
            length_scale=length_scale,
            timeout_s=self._settings.tts_timeout_s,
        )

    def cancel(self) -> None:
        if self._backend is not None:
            self._backend.cancel()


# --------------------------------------------------------------------------- #
# Rejestr dostawców
# --------------------------------------------------------------------------- #

TTSFactory = Callable[[Settings, UserSettings], TTSProvider]

_PROVIDERS: dict[str, TTSFactory] = {}


def register_tts_provider(name: str, factory: TTSFactory) -> None:
    """Dopisz silnik mowy do rejestru (używane przez kolejne fazy: XTTS, RVC...).

    Nazwa jest tą samą, którą użytkownik wpisuje w ``voice_engine``
    w ``config/user_settings.json``.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError(t("tts.empty_provider_name"))
    _PROVIDERS[key] = factory


def available_tts_engines() -> list[str]:
    """Nazwy zarejestrowanych silników mowy."""
    return sorted(_PROVIDERS)


register_tts_provider("piper", PiperTTSProvider)
register_tts_provider("none", lambda settings, user: NullTTSProvider())


def create_tts_provider(
    settings: Settings | None = None,
    user_settings: UserSettings | None = None,
    *,
    engine: str | None = None,
) -> TTSProvider:
    """Zbuduj silnik mowy wskazany w ustawieniach (albo ``NullTTSProvider``).

    Nie ładuje modelu — od tego jest :meth:`TTSProvider.load`, żeby wywołujący
    mógł zdecydować, kiedy zapłacić za wczytanie i jak obsłużyć błąd.
    """
    active = settings or get_settings()
    user = user_settings if user_settings is not None else get_user_settings()

    if not active.tts_enabled:
        return NullTTSProvider("wyłączona ustawieniem TTS_ENABLED=false")

    name = (engine or user.voice_engine).strip().lower()
    if name in ("", "none", "off", "brak"):
        return NullTTSProvider("voice_engine: \"none\" w config/user_settings.json")

    factory = _PROVIDERS.get(name)
    if factory is None:
        logger.warning(
            "Nieznany silnik mowy %r — dostępne: %s. Mowa pozostaje wyłączona.",
            name,
            ", ".join(available_tts_engines()),
        )
        return NullTTSProvider(f"nieznany silnik „{name}”")
    return factory(active, user)


# --------------------------------------------------------------------------- #
# Przygotowanie tekstu do wypowiedzenia
# --------------------------------------------------------------------------- #

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN_MARKS = re.compile(r"[*_`~]{1,3}")
_HEADER_OR_BULLET = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)
# Znaki graficzne, których żaden silnik mowy nie przeczyta sensownie: emoji,
# strzałki, symbole techniczne, selektory wariantu i łącznik ZWJ (te ostatnie
# sklejają emoji w rodziny i zostałyby po usunięciu samych obrazków).
_PICTOGRAMS = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoji i symbole uzupełniające
    "\u2190-\u21ff"  # strzałki
    "\u2300-\u23ff"  # symbole techniczne
    "\u2600-\u27bf"  # symbole różne i dingbaty
    "\u2b00-\u2bff"  # strzałki uzupełniające
    "\ufe0f\u200d"  # selektor wariantu (VS16) i łącznik ZWJ
    "]"
)
_WHITESPACE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Usuń z tekstu to, czego nie da się przeczytać na głos.

    Model bywa poproszony o unikanie formatowania, ale nie zawsze słucha —
    a przeczytana na głos gwiazdka albo adres URL brzmi absurdalnie. Bloki kodu
    są pomijane w całości (jeśli po ich usunięciu nie zostaje nic, wracamy do
    tekstu bez samych znaczników — lepiej powiedzieć cokolwiek niż nic).
    """
    if not text.strip():
        return ""

    stripped = _CODE_FENCE.sub(" ", text)
    if not stripped.strip():
        stripped = text.replace("```", " ")

    stripped = _MARKDOWN_LINK.sub(r"\1", stripped)
    stripped = _URL.sub("link", stripped)
    stripped = _HEADER_OR_BULLET.sub("", stripped)
    stripped = _MARKDOWN_MARKS.sub("", stripped)
    stripped = _PICTOGRAMS.sub(" ", stripped)
    return _WHITESPACE.sub(" ", stripped).strip()


# Skróty, po których kropka NIE kończy zdania. Lista jest krótka z rozmysłem:
# obejmuje to, co realnie pada w rozmowie, a nie kompletny słownik.
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "np", "itd", "itp", "tzn", "tj", "ok", "ur", "zm", "godz", "ul", "al",
        "dr", "prof", "inż", "mgr", "hab", "św", "nr", "str", "mln", "mld",
        "tys", "por", "zob", "ws", "wg", "cd", "jw", "min", "maks",
        "mr", "mrs", "ms", "vs", "etc", "eg", "ie", "fig", "approx", "st",
    }
)

_SENTENCE_END_CHARS: Final[str] = ".!?…"
# Cudzysłowy i nawiasy zamykające zdanie po znaku interpunkcyjnym.
_CLOSING_CHARS: Final[str] = "\"'”’)]»"  # noqa: RUF001 - typografia polska, celowo


class SentenceBuffer:
    """Skleja napływające fragmenty tekstu w zdania gotowe do wypowiedzenia.

    Model językowy przysyła odpowiedź po kilka znaków. Żeby mowa ruszyła, zanim
    skończy pisać, potrzebna jest granica zdania — ale nie każda kropka nią jest
    (``np.``, ``3.5``, ``dr.``). Zbyt krótkie fragmenty czekają na dalszy ciąg,
    bo pojedyncze słowa syntezowane osobno brzmią rwanie i kosztują więcej niż
    jedno dłuższe zdanie.
    """

    def __init__(self, *, min_chars: int = 24, max_chars: int = 320) -> None:
        self._min_chars = max(0, min_chars)
        self._max_chars = max(20, max_chars)
        self._buffer = ""

    @property
    def pending(self) -> str:
        return self._buffer

    def reset(self) -> None:
        self._buffer = ""

    def append(self, text: str) -> None:
        """Dołóż fragment BEZ dzielenia (tryb bez strumieniowania zdań)."""
        self._buffer += text

    def push(self, text: str) -> list[str]:
        """Dołóż fragment i zwróć zdania, które można już wypowiedzieć."""
        if not text:
            return []
        self._buffer += text
        ready: list[str] = []
        while True:
            piece = self._take_next()
            if piece is None:
                break
            if piece.strip():
                ready.append(piece.strip())
        return ready

    def flush(self) -> list[str]:
        """Oddaj resztę bufora (koniec odpowiedzi — nie ma na co czekać)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if not remaining:
            return []
        pieces: list[str] = []
        while len(remaining) > self._max_chars:
            cut = self._hard_split_index(remaining)
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            pieces.append(remaining)
        return [piece for piece in pieces if piece]

    # --- środek ------------------------------------------------------------ #

    def _take_next(self) -> str | None:
        index = self._sentence_end_index()
        if index is not None:
            piece, self._buffer = self._buffer[:index], self._buffer[index:].lstrip()
            return piece
        if len(self._buffer) > self._max_chars:
            cut = self._hard_split_index(self._buffer)
            piece, self._buffer = self._buffer[:cut], self._buffer[cut:].lstrip()
            return piece
        return None

    def _sentence_end_index(self) -> int | None:
        """Pozycja tuż za końcem pierwszego pełnego zdania (albo ``None``)."""
        buffer = self._buffer
        for position, character in enumerate(buffer):
            if character == "\n":
                if position + 1 >= self._min_chars:
                    return position + 1
                continue
            if character not in _SENTENCE_END_CHARS:
                continue

            end = position + 1
            while end < len(buffer) and buffer[end] in _CLOSING_CHARS:
                end += 1
            # Bez znaku po interpunkcji nie wiadomo, czy zdanie się skończyło —
            # „3." może być początkiem liczby „3.5", a „np." skrótem.
            if end >= len(buffer):
                return None
            if not buffer[end].isspace():
                continue
            if end < self._min_chars:
                continue
            if character == "." and self._is_abbreviation(buffer, position):
                continue
            return end
        return None

    @staticmethod
    def _is_abbreviation(buffer: str, dot_position: int) -> bool:
        start = dot_position
        while start > 0 and (buffer[start - 1].isalpha() or buffer[start - 1] == "."):
            start -= 1
        word = buffer[start:dot_position].strip(".").lower()
        if not word:
            return False
        if len(word) == 1 and word.isalpha():
            return True  # inicjał: „J. Kowalski"
        return word in _ABBREVIATIONS

    def _hard_split_index(self, text: str) -> int:
        """Miejsce cięcia zbyt długiego fragmentu — po spacji, nie w środku słowa."""
        window = text[: self._max_chars]
        for separator in (", ", "; ", " — ", " - ", " "):
            cut = window.rfind(separator)
            if cut > self._max_chars // 3:
                return cut + len(separator)
        return self._max_chars


def split_sentences(text: str, *, min_chars: int = 24, max_chars: int = 320) -> list[str]:
    """Podziel gotowy tekst na fragmenty do wypowiedzenia (wygodna otoczka)."""
    buffer = SentenceBuffer(min_chars=min_chars, max_chars=max_chars)
    pieces = buffer.push(text)
    pieces.extend(buffer.flush())
    return pieces


# --------------------------------------------------------------------------- #
# Odtwarzanie: kontrakt ujścia dźwięku
# --------------------------------------------------------------------------- #


@runtime_checkable
class AudioSink(Protocol):
    """Coś, co przyjmuje dźwięk: głośnik, plik, atrapa w teście.

    ``audio/output.py`` dostarcza implementację na karcie dźwiękowej. Ten moduł
    celowo zna tylko ten wąski kontrakt — dzięki temu synteza jest testowalna
    bez żadnego sprzętu.
    """

    def open(self, sample_rate: int) -> None: ...

    def write(self, chunk: SpeechChunk) -> None: ...

    def drain(self, timeout: float | None = None) -> None: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class CollectingSink:
    """Ujście zbierające fragmenty w pamięci — do testów, zapisu i diagnostyki."""

    def __init__(self) -> None:
        self.chunks: list[SpeechChunk] = []
        self.opened_rates: list[int] = []
        self.cancelled = False
        self.closed = False

    def open(self, sample_rate: int) -> None:
        self.opened_rates.append(sample_rate)

    def write(self, chunk: SpeechChunk) -> None:
        self.chunks.append(chunk)

    def drain(self, timeout: float | None = None) -> None:
        return None

    def cancel(self) -> None:
        self.cancelled = True
        self.chunks.clear()

    def close(self) -> None:
        self.closed = True


def write_wav(path: Path, chunks: Iterable[SpeechChunk]) -> Path:
    """Zapisz fragmenty jako plik WAV (mono, 16 bitów).

    Przydaje się tam, gdzie nie ma głośnika: na serwerze, w kontenerze albo przy
    sprawdzaniu, czy głos w ogóle działa. Format WAV jest wybrany świadomie —
    nie wymaga żadnej biblioteki poza standardową.
    """
    collected = [chunk for chunk in chunks if not chunk.is_empty]
    if not collected:
        raise TTSError(t("tts.nothing_to_save"))

    rate = collected[0].sample_rate
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            for chunk in collected:
                handle.writeframes(chunk.samples.astype(np.int16).tobytes())
    except OSError as exc:
        raise TTSError(
            t("tts.save_failed", path=target, error=exc),
            hint=t("tts.save_hint"),
        ) from exc
    return target


# --------------------------------------------------------------------------- #
# Mówienie odpowiedzi w miarę jej powstawania
# --------------------------------------------------------------------------- #

ErrorCallback = Callable[[TTSError], None]


class SpeechOutput:
    """Spina silnik mowy z odtwarzaniem: tekst wchodzi, dźwięk wychodzi.

    Sedno jest w strumieniowaniu. Model językowy oddaje odpowiedź po kawałku;
    :meth:`feed` skleja te kawałki w zdania, a wątek roboczy syntezuje zdanie
    N-te w chwili, gdy model pisze N+1. Pierwszy dźwięk pada więc na długo przed
    końcem generowania.

    Klasa nigdy nie wywraca programu: każdy błąd syntezy albo odtwarzania kończy
    się jednorazowym zgłoszeniem przez ``on_error`` i wyłączeniem mowy do końca
    sesji — rozmowa toczy się dalej tekstem.
    """

    def __init__(
        self,
        provider: TTSProvider,
        sink: AudioSink,
        *,
        settings: Settings | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider
        self._sink = sink
        self._on_error = on_error

        self._buffer = SentenceBuffer(
            min_chars=self._settings.tts_min_sentence_chars,
            max_chars=self._settings.tts_max_sentence_chars,
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._speaking = threading.Event()
        self._failed = False
        self._language: str | None = None
        self._open_rate: int | None = None
        self._spoken_chunks = 0

    # --- stan -------------------------------------------------------------- #

    @property
    def provider(self) -> TTSProvider:
        return self._provider

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def failed(self) -> bool:
        """Czy mowa została wyłączona po błędzie?"""
        return self._failed

    @property
    def enabled(self) -> bool:
        return self._provider.is_speaking_enabled and not self._failed

    def describe(self) -> str:
        return self._provider.describe()

    # --- mówienie ----------------------------------------------------------- #

    def begin(self, language: str | None = None) -> None:
        """Zacznij nową wypowiedź (jedna odpowiedź asystenta)."""
        if not self.enabled:
            return
        self.cancel()  # domknij poprzednią, gdyby coś jeszcze grało
        self._cancelled.clear()
        # Po anulowaniu w kolejce może zostać niezjedzony znacznik końca —
        # nowy wątek zakończyłby się na nim, zanim cokolwiek powie.
        self._drain_queue()
        self._buffer.reset()
        self._language = language
        self._spoken_chunks = 0
        self._speaking.set()
        self._worker = threading.Thread(
            target=self._run, name="tts-worker", daemon=True
        )
        self._worker.start()

    def feed(self, text: str) -> None:
        """Dołóż kawałek tekstu; pełne zdania idą od razu do syntezy."""
        if not self.enabled or self._worker is None or self._cancelled.is_set():
            return
        if self._settings.tts_stream_sentences:
            for sentence in self._buffer.push(text):
                self._queue.put(sentence)
        else:
            # TTS_STREAM_SENTENCES=false: cała odpowiedź idzie do syntezy dopiero
            # w end() — prościej, ale pierwszy dźwięk pada znacznie później.
            self._buffer.append(text)

    def end(self, *, wait: bool = True) -> None:
        """Domknij wypowiedź: dokończ resztę tekstu i (opcjonalnie) poczekaj."""
        if self._worker is None:
            return
        if not self._cancelled.is_set():
            for sentence in self._buffer.flush():
                self._queue.put(sentence)
        self._queue.put(None)
        if wait:
            self.wait()

    def wait(self, timeout: float | None = None) -> None:
        """Poczekaj, aż wszystko zostanie zsyntezowane i odtworzone."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
            if not worker.is_alive():
                self._worker = None
        if not self._cancelled.is_set():
            with contextlib.suppress(Exception):
                self._sink.drain(timeout)
        self._speaking.clear()

    def speak(self, text: str, *, language: str | None = None, wait: bool = True) -> None:
        """Wypowiedz gotowy tekst (bez strumieniowania z modelu)."""
        if not self.enabled:
            return
        self.begin(language)
        self.feed(text)
        self.end(wait=wait)

    def cancel(self) -> None:
        """Przerwij mówienie natychmiast (Ctrl+C, nowa wypowiedź użytkownika)."""
        worker = self._worker
        if worker is None:
            return
        self._cancelled.set()
        with contextlib.suppress(Exception):
            self._provider.cancel()
        self._drain_queue()
        self._queue.put(None)
        with contextlib.suppress(Exception):
            self._sink.cancel()
        worker.join(timeout=5.0)
        self._worker = None
        self._speaking.clear()
        self._open_rate = None

    def close(self) -> None:
        self.cancel()
        with contextlib.suppress(Exception):
            self._sink.close()
        with contextlib.suppress(Exception):
            self._provider.close()

    def __enter__(self) -> SpeechOutput:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- wątek roboczy -------------------------------------------------------- #

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            sentence = self._queue.get()
            if sentence is None:
                break
            if self._cancelled.is_set() or self._failed:
                continue
            try:
                self._speak_sentence(sentence)
            except TTSError as exc:
                self._fail(exc)
            except Exception as exc:  # ostatnia linia obrony — wątek nie może paść
                logger.exception("Nieoczekiwany błąd syntezy mowy")
                self._fail(
                    TTSError(
                        t("tts.synthesis_failed", error=exc),
                        hint=t("tts.details_in_log"),
                    )
                )
        self._speaking.clear()

    def _speak_sentence(self, sentence: str) -> None:
        spoken = clean_for_speech(sentence)
        if not spoken:
            return
        started = time.monotonic()
        for chunk in self._provider.synthesize(spoken, language=self._language):
            if self._cancelled.is_set():
                return
            if chunk.is_empty:
                continue
            if self._open_rate != chunk.sample_rate:
                self._sink.open(chunk.sample_rate)
                self._open_rate = chunk.sample_rate
            self._sink.write(chunk)
            self._spoken_chunks += 1
        logger.debug(
            "Wypowiedziano %d znaków w %.2f s: %r",
            len(spoken),
            time.monotonic() - started,
            spoken[:80],
        )

    def _fail(self, error: TTSError) -> None:
        if self._failed:
            return
        self._failed = True
        logger.warning("Mowa wyłączona po błędzie: %s", error.message)
        # Kolejki NIE opróżniamy: leży w niej znacznik końca wypowiedzi, na
        # który czeka wątek główny. Pozostałe zdania i tak są pomijane w pętli.
        if self._on_error is not None:
            with contextlib.suppress(Exception):
                self._on_error(error)


__all__ = [
    "DEFAULT_PIPER_SAMPLE_RATE",
    "AudioSink",
    "CollectingSink",
    "NullTTSProvider",
    "PiperBackend",
    "PiperProcessBackend",
    "PiperPythonBackend",
    "PiperTTSProvider",
    "SentenceBuffer",
    "SpeechChunk",
    "SpeechOutput",
    "TTSError",
    "TTSProvider",
    "TTSUnavailableError",
    "VoiceModel",
    "available_tts_engines",
    "clean_for_speech",
    "create_piper_backend",
    "create_tts_provider",
    "describe_voice_file",
    "find_piper_voice",
    "iter_piper_voices",
    "register_tts_provider",
    "select_piper_voice",
    "split_sentences",
    "write_wav",
]
