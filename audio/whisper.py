"""Transkrypcja mowy przez Faster-Whisper.

Urządzenie i typ obliczeń pochodzą z ``config.py``: ``auto`` oznacza CUDA, gdy
wykryto działającą kartę NVIDIA, a w przeciwnym razie CPU. Jeśli inicjalizacja
na GPU zawiedzie już po starcie (najczęstszy przypadek: sterownik jest, ale
brakuje cuDNN albo wersja CTranslate2 nie pasuje), moduł automatycznie
przechodzi na CPU zamiast przerywać program.

Model jest pobierany do katalogu ``models/whisper`` wewnątrz projektu, a nie do
domyślnego cache'u HuggingFace w katalogu domowym — dzięki temu nic nie zależy
od układu katalogów użytkownika i całość da się przenieść razem z projektem.

Praca bez internetu: jeśli model jest już na dysku, ładujemy go **ze ścieżki**,
a nie po nazwie. Nazwa oznacza dla ``faster-whisper`` identyfikator repozytorium
HuggingFace, więc biblioteka i tak odpytałaby serwer o aktualność migawki —
bez sieci kończy się to kilkunastosekundowym czekaniem na timeout przy każdym
starcie. Ścieżka omija ten mechanizm w całości.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Final

import numpy as np

from audio.resample import int16_to_float32, resample_int16
from audio.vad import Utterance
from config import (
    WHISPER_CACHE_DIR,
    GPUInfo,
    Settings,
    apply_offline_environment,
    detect_cuda,
    find_local_whisper_model,
    get_settings,
    is_offline,
    pip_install_hint,
    resolve_speech_languages,
)
from i18n import t

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE: Final[int] = 16_000

# Cisza doklejana przed wypowiedzią i za nią. VAD wycina fragment tuż przy
# pierwszej głosce, a Whisper regularnie gubi wtedy pierwsze słowo — najczęściej
# to samo, które jest frazą wybudzającą („Hej Miku, jaka pogoda" → „jaka
# pogoda"). Pomiar na korpusie 480 transkrypcji mowy syntetycznej: samo
# dopełnienie ciszą podniosło wykrywanie frazy z 27% na 45%, a WER zwykłych
# zdań spadł z 27,5% na 16,3% (model small). Kosztuje 0,6 s dźwięku, czyli
# ułamek sekundy liczenia i ani jednej dodatkowej zależności.
PAD_LEAD_S: Final[float] = 0.4
PAD_TAIL_S: Final[float] = 0.2

# Whisper na samej ciszy potrafi „usłyszeć" napisy końcowe z materiałów
# treningowych. To najczęstsze halucynacje w polskim i angielskim.
_HALLUCINATION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^napisy (stworzone|wykonane) przez.*$",
        r"^napisy: .*$",
        r"^subtitles by.*$",
        r"^subs by.*$",
        r"^dziękuję za (uwagę|obejrzenie).*$",
        r"^thanks? for watching.*$",
        r"^thank you\.?$",
        r"^\.+$",
        r"^…+$",
    )
)


class TranscriptionError(RuntimeError):
    """Błąd transkrypcji z komunikatem gotowym dla użytkownika."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float


@dataclass(frozen=True, slots=True)
class Transcript:
    """Wynik transkrypcji pojedynczej wypowiedzi."""

    text: str
    language: str
    language_probability: float
    audio_duration_s: float
    processing_s: float
    segments: tuple[TranscriptSegment, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def realtime_factor(self) -> float:
        """Ile razy szybciej niż w czasie rzeczywistym poszła transkrypcja."""
        if self.processing_s <= 0.0:
            return 0.0
        return self.audio_duration_s / self.processing_s


def _load_faster_whisper() -> ModuleType:
    try:
        import faster_whisper  # noqa: PLC0415 - import celowo leniwy (ciężka biblioteka)
    except ImportError as exc:
        raise TranscriptionError(
            t("stt.no_package"),
            hint=pip_install_hint(),
        ) from exc
    except OSError as exc:
        raise TranscriptionError(
            t("stt.ctranslate_failed", error=exc),
            hint=t("stt.ctranslate_hint"),
        ) from exc
    return faster_whisper


def supported_languages() -> frozenset[str]:
    """Kody języków znane zainstalowanej wersji Whispera.

    Pusty zbiór oznacza „nie wiadomo" (inna wersja biblioteki, zmieniona nazwa
    pola) — wtedy nie blokujemy niczego i przepuszczamy kod dalej.
    """
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES  # noqa: PLC0415 - import leniwy
    except Exception:  # pragma: no cover - zależne od wersji faster-whisper
        return frozenset()
    try:
        return frozenset(str(code) for code in _LANGUAGE_CODES)
    except TypeError:  # pragma: no cover - nieoczekiwany typ w bibliotece
        return frozenset()


def resolve_device(settings: Settings, gpu: GPUInfo | None = None) -> tuple[str, str]:
    """Ustal parę (urządzenie, typ obliczeń) na podstawie konfiguracji i sprzętu."""
    info = gpu if gpu is not None else detect_cuda()

    device = settings.whisper_device
    if device == "auto":
        device = "cuda" if info.cuda_available else "cpu"
    elif device == "cuda" and not info.cuda_available:
        logger.warning(
            "WHISPER_DEVICE=cuda, ale nie wykryto działającego CUDA — przechodzę na CPU."
        )
        device = "cpu"

    compute_type = settings.whisper_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def _is_repetition_loop(text: str, *, min_words: int = 6, ratio: float = 0.6) -> bool:
    """Czy tekst to zapętlone powtórzenie jednego słowa?

    Whisper karmiony szumem wpada w pętlę i produkuje ciągi w rodzaju
    „No, no, no, no, no...". Taki tekst przechodzi przez filtr wzorców i przez
    ``no_speech_prob``, więc potrzebna jest osobna reguła.
    """
    words = [word for word in re.findall(r"\w+", text.lower()) if word]
    if len(words) < min_words:
        return False
    most_common = max(set(words), key=words.count)
    return words.count(most_common) / len(words) >= ratio


def is_hallucination(text: str) -> bool:
    """Czy tekst wygląda na halucynację Whispera z fragmentu ciszy lub szumu?"""
    stripped = text.strip()
    if not stripped:
        return True
    if any(pattern.match(stripped) for pattern in _HALLUCINATION_PATTERNS):
        return True
    return _is_repetition_loop(stripped)


def _supports_hotwords(model: Any) -> bool:
    """Czy ta wersja ``faster-whisper`` przyjmuje ``hotwords``?

    Sprawdzamy sygnaturę zamiast wersji: numer wersji nic nie mówi o wariantach
    pakietu, a nieznany argument kończyłby się wyjątkiem przy KAŻDEJ wypowiedzi.
    """
    try:
        parameters = inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):  # pragma: no cover - egzotyczna atrapa
        return False
    if "hotwords" in parameters:
        return True
    # Opakowania (i atrapy w testach) przyjmują wszystko przez ``**kwargs`` —
    # taka sygnatura też uniesie podpowiedź.
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _pad_with_silence(samples: np.ndarray) -> np.ndarray:
    """Dołóż ciszę na początku i końcu — patrz PAD_LEAD_S."""
    if samples.size == 0:
        return samples
    lead = np.zeros(int(WHISPER_SAMPLE_RATE * PAD_LEAD_S), dtype=samples.dtype)
    tail = np.zeros(int(WHISPER_SAMPLE_RATE * PAD_TAIL_S), dtype=samples.dtype)
    return np.concatenate([lead, samples, tail])


class WhisperTranscriber:
    """Otoczka na ``faster_whisper.WhisperModel`` z leniwym ładowaniem."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        gpu: GPUInfo | None = None,
        model: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gpu = gpu
        self._model: Any | None = model
        self._device, self._compute_type = resolve_device(self._settings, gpu)
        self._loaded_from_argument = model is not None
        self._reported_bad_languages: set[str] = set()

    # --- właściwości ----------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def model_name(self) -> str:
        return self._settings.whisper_model

    def describe(self) -> str:
        return f"{self.model_name} ({self._device}/{self._compute_type})"

    # --- ładowanie -------------------------------------------------------- #

    @property
    def offline(self) -> bool:
        """Czy w tej sesji obowiązuje zakaz sięgania do sieci?"""
        return is_offline(self._settings)

    def _download_allowed(self) -> bool:
        return self._settings.whisper_allow_download and not self.offline

    def _model_source(self) -> str:
        """Ścieżka do modelu lokalnego, a w ostateczności nazwa do pobrania."""
        raw = self._settings.whisper_model
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return str(candidate)

        local = find_local_whisper_model(raw, WHISPER_CACHE_DIR)
        if local is not None:
            logger.debug("Model Whisper '%s' znaleziony lokalnie: %s", raw, local)
            return str(local)
        return raw

    def load(self) -> None:
        """Załaduj model. Przy niepowodzeniu na GPU spada automatycznie na CPU."""
        if self._model is not None:
            return

        # Zanim zaimportujemy faster-whisper (a przez niego huggingface_hub),
        # który czyta zmienne trybu offline w czasie importu.
        apply_offline_environment(self._settings)

        faster_whisper = _load_faster_whisper()
        source = self._model_source()

        try:
            WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TranscriptionError(
                t("stt.models_dir_readonly", path=WHISPER_CACHE_DIR, error=exc),
                hint=t("stt.models_dir_hint"),
            ) from exc

        attempts: list[tuple[str, str]] = [(self._device, self._compute_type)]
        if self._device == "cuda":
            attempts.append(("cpu", "int8"))

        last_error: Exception | None = None
        for device, compute_type in attempts:
            started = time.monotonic()
            try:
                model = faster_whisper.WhisperModel(
                    source,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(WHISPER_CACHE_DIR),
                    local_files_only=not self._download_allowed(),
                )
                if device == "cuda":
                    # Konstruktor NIE dotyka bibliotek CUDA — cuBLAS i cuDNN
                    # ładują się dopiero przy pierwszej inferencji. Bez tego
                    # rozgrzania brak libcublas ujawniłby się dopiero przy
                    # pierwszym zdaniu użytkownika, gdy jest już za późno na
                    # zejście na CPU.
                    self._warmup(model)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Nie udało się załadować modelu Whisper na %s/%s: %s",
                    device,
                    compute_type,
                    exc,
                )
                continue

            self._model = model
            self._device = device
            self._compute_type = compute_type
            logger.info(
                "Whisper załadowany: %s na %s/%s w %.1f s",
                source,
                device,
                compute_type,
                time.monotonic() - started,
            )
            return

        hint = t("stt.load_hint_download")
        if not self._settings.whisper_allow_download:
            hint = t("stt.load_hint_no_download")
        elif self.offline:
            hint = t("stt.load_hint_offline")
        raise TranscriptionError(
            t("stt.load_failed", model=source, error=last_error),
            hint=hint,
        )

    @staticmethod
    def _warmup(model: Any) -> None:
        """Wymuś załadowanie bibliotek obliczeniowych krótką inferencją."""
        silence = np.zeros(WHISPER_SAMPLE_RATE // 10, dtype=np.float32)
        segments, _ = model.transcribe(silence, language="en", beam_size=1, vad_filter=False)
        list(segments)  # generator — bez konsumpcji nic się nie policzy

    def _fallback_to_cpu(self, error: Exception) -> bool:
        """Przeładuj model na CPU po awarii GPU w trakcie pracy."""
        if self._device != "cuda" or self._loaded_from_argument:
            return False

        logger.warning(
            "Awaria GPU podczas transkrypcji (%s) — przechodzę na CPU do końca sesji.", error
        )
        self._model = None
        self._device = "cpu"
        self._compute_type = "int8"
        try:
            self.load()
        except TranscriptionError as exc:
            logger.error("Nie udało się przeładować modelu na CPU: %s", exc.message)
            return False
        return True

    def unload(self) -> None:
        """Zwolnij model (i pamięć GPU, jeśli była używana)."""
        if self._loaded_from_argument:
            return
        self._model = None

    def __enter__(self) -> WhisperTranscriber:
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.unload()

    # --- transkrypcja ------------------------------------------------------ #

    def _prepare_audio(self, audio: Utterance | np.ndarray, sample_rate: int) -> np.ndarray:
        """Sygnał gotowy dla modelu: 16 kHz, float32, z ciszą na brzegach."""
        if isinstance(audio, Utterance):
            samples = audio.samples
            sample_rate = audio.sample_rate
        else:
            samples = audio

        if samples.dtype == np.int16 and sample_rate != WHISPER_SAMPLE_RATE:
            samples = resample_int16(samples, sample_rate, WHISPER_SAMPLE_RATE)
        return int16_to_float32(samples)

    def _language_for(self, samples: np.ndarray) -> str:
        """Język tej wypowiedzi zgodnie z ustawieniami użytkownika.

        Trzy przypadki, wszystkie zmierzone na tym projekcie (mowa syntetyczna,
        model ``small``, WER):

        * **jeden kod** — wymuszamy go. Najlepszy wynik, gdy język się zgadza
          (29,2% dla polskiej mowy), i najgorszy, gdy się nie zgadza (114,0%
          dla polskiej mowy przy wymuszonym angielskim — tekst do wyrzucenia),
        * **kilka kodów** („pl,en") — rozpoznajemy, ale wybieramy WYŁĄCZNIE
          spośród podanych. Użytkownik mówiący w dwóch językach dostaje 41,7%
          i 5,0% zamiast 114,0% na jednym z nich,
        * **brak** — pełne rozpoznawanie Whispera, z ryzykiem, że polskie zdanie
          wróci zapisane cyrylicą albo pismem arabskim.
        """
        allowed = resolve_speech_languages(self._settings)
        if len(allowed) == 1:
            return allowed[0]
        if not allowed:
            return ""

        detector = getattr(self._model, "detect_language", None)
        if detector is None:  # starsza wersja pakietu — zostaje pełne rozpoznawanie
            return ""
        try:
            _, _, probabilities = detector(samples)
        except Exception as exc:  # rozpoznawanie nie może wywalić transkrypcji
            logger.debug("Rozpoznawanie języka nie powiodło się: %s", exc)
            return ""

        ranking = {str(code): float(value) for code, value in probabilities}
        best = max(allowed, key=lambda code: ranking.get(code, 0.0))
        logger.debug(
            "Język wypowiedzi: %s (spośród %s, pewność %.2f)",
            best,
            ", ".join(allowed),
            ranking.get(best, 0.0),
        )
        return best

    def _validate_language(self, code: str | None) -> str | None:
        """Odrzuć kod języka, którego model nie zna — inaczej padłaby transkrypcja.

        Kod pochodzi z pliku ustawień użytkownika, więc może być literówką albo
        językiem spoza zestawu Whispera. Zamiast błędu przy każdej wypowiedzi
        wracamy do automatycznego rozpoznawania i mówimy o tym raz.
        """
        if code is None:
            return None
        known = supported_languages()
        if not known or code in known:
            return code
        if code not in self._reported_bad_languages:
            self._reported_bad_languages.add(code)
            logger.warning(
                "Whisper nie zna języka %r — wracam do automatycznego rozpoznawania. "
                "Popraw speech_language w config/user_settings.json (albo WHISPER_LANGUAGE).",
                code,
            )
        return None

    def transcribe(
        self,
        audio: Utterance | np.ndarray,
        *,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: str | None = None,
        hotwords: str = "",
    ) -> Transcript:
        """Przetranskrybuj wypowiedź.

        Język bierze się z preferencji użytkownika (``speech_language`` w
        ``config/user_settings.json``), a gdy ta jest ustawiona na ``auto`` —
        z ``WHISPER_LANGUAGE``. Brak obu = Whisper rozpoznaje język sam.
        Argument ``language`` przebija jedno i drugie (używa go np. rozgrzewka).
        Wartość jest czytana przy każdym wywołaniu, więc zmiana pliku ustawień
        działa bez restartu — tak samo jak imię asystenta.

        ``hotwords`` to słowa, których model ma się spodziewać. Używa tego
        detektor frazy: nazwa własna („Miku") jest dla modelu obcym słowem i bez
        podpowiedzi wychodzi z niego „tymiku" albo „micu". Parametr jest
        przekazywany tylko wtedy, gdy zainstalowana wersja ``faster-whisper``
        go zna — starsza działa dalej, po prostu bez podpowiedzi.
        """
        self.load()

        samples = self._prepare_audio(audio, sample_rate)
        audio_duration = samples.size / WHISPER_SAMPLE_RATE

        if audio_duration < self._settings.whisper_min_duration_s:
            logger.debug("Pominięto fragment krótszy niż próg: %.2f s", audio_duration)
            return Transcript(
                text="",
                language="",
                language_probability=0.0,
                audio_duration_s=audio_duration,
                processing_s=0.0,
            )

        # Cisza na brzegach dokładana TERAZ: pomiar długości powyżej dotyczy mowy,
        # a nie tego, co sami dołożyliśmy.
        padded = _pad_with_silence(samples)
        forced_language = self._validate_language(
            language or self._language_for(padded) or None
        )
        started = time.monotonic()

        extra: dict[str, Any] = {}
        if hotwords and _supports_hotwords(self._model):
            extra["hotwords"] = hotwords

        def run() -> tuple[list[Any], Any]:
            segments, info = self._model.transcribe(  # type: ignore[union-attr]
                padded,
                language=forced_language,
                beam_size=self._settings.whisper_beam_size,
                # Własny VAD już wyciął ciszę — powtórna filtracja tylko szkodzi.
                vad_filter=False,
                condition_on_previous_text=False,
                **extra,
            )
            return list(segments), info

        try:
            collected, info = run()
        except Exception as exc:
            # Druga linia obrony: gdyby awaria GPU ominęła rozgrzewanie przy
            # ładowaniu, schodzimy na CPU i próbujemy jeszcze raz zamiast
            # zwracać użytkownikowi błąd.
            if self._fallback_to_cpu(exc):
                try:
                    collected, info = run()
                except Exception as retry_error:
                    logger.exception("Transkrypcja nie powiodła się także na CPU")
                    raise TranscriptionError(
                        t("stt.transcribe_failed", error=retry_error),
                        hint=t("stt.details_in_log"),
                    ) from retry_error
            else:
                logger.exception("Transkrypcja nie powiodła się")
                raise TranscriptionError(
                    t("stt.transcribe_failed", error=exc),
                    hint=t("stt.details_in_log"),
                ) from exc

        processing = time.monotonic() - started
        parsed: list[TranscriptSegment] = []
        for segment in collected:
            no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            if no_speech > self._settings.whisper_max_no_speech_prob:
                logger.debug("Odrzucono segment (no_speech_prob=%.2f): %r", no_speech, text)
                continue
            if is_hallucination(text):
                logger.debug("Odrzucono prawdopodobną halucynację: %r", text)
                continue
            parsed.append(
                TranscriptSegment(
                    start=float(getattr(segment, "start", 0.0) or 0.0),
                    end=float(getattr(segment, "end", 0.0) or 0.0),
                    text=text,
                    no_speech_prob=no_speech,
                )
            )

        text = " ".join(segment.text for segment in parsed).strip()
        language_code = str(getattr(info, "language", "") or forced_language or "")
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        logger.info(
            "Transkrypcja: %.2f s audio w %.2f s (%s, p=%.2f): %r",
            audio_duration,
            processing,
            language_code or "?",
            probability,
            text[:120],
        )

        return Transcript(
            text=text,
            language=language_code,
            language_probability=probability,
            audio_duration_s=audio_duration,
            processing_s=processing,
            segments=tuple(parsed),
        )

    async def transcribe_async(
        self,
        audio: Utterance | np.ndarray,
        *,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: str | None = None,
    ) -> Transcript:
        """Wersja nieblokująca — transkrypcja w wątku roboczym.

        CTranslate2 zwalnia GIL na czas obliczeń, więc wątek wystarczy i nie
        trzeba przeładowywać modelu w osobnym procesie.
        """
        return await asyncio.to_thread(
            self.transcribe, audio, sample_rate=sample_rate, language=language
        )


__all__ = [
    "WHISPER_SAMPLE_RATE",
    "Transcript",
    "TranscriptSegment",
    "TranscriptionError",
    "WhisperTranscriber",
    "is_hallucination",
    "resolve_device",
    "supported_languages",
]
