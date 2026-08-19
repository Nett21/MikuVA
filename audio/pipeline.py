"""Potok wejścia głosowego: MIKROFON → VAD → [WAKE WORD] → WHISPER → TEKST.

Potok jest w całości wstrzykiwalny: mikrofon, VAD, segmenter, detektor frazy i
transkryber można podmienić na atrapy, dlatego testy nie potrzebują sprzętu.

Bramka słowa aktywującego (Faza 3) stoi przed głównym modelem: dopóki fraza nie
padnie, żadna wypowiedź nie trafia do dużego Whispera ani dalej do modelu
językowego. Po wykryciu frazy otwiera się okno rozmowy
(``WAKE_WINDOW_S``), w którym mówi się normalnie, bez powtarzania zawołania.

Klasa jest świadomie synchroniczna. Nasłuch to pętla czekająca na ramki z
kolejki wypełnianej przez wątek PortAudio, a Whisper i tak liczy w kodzie
natywnym — asyncio nie dałoby tu nic poza komplikacją. Do użycia z GUI
(Faza 10) służy :meth:`WhisperTranscriber.transcribe_async`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from types import TracebackType

from audio.microphone import Microphone, MicrophoneError, is_microphone_available
from audio.vad import Utterance, UtteranceSegmenter, VoiceActivityDetector
from audio.wakeword import (
    WakeMatch,
    WakeWordEngine,
    WakeWordError,
    create_wake_word_engine,
)
from audio.whisper import Transcript, TranscriptionError, WhisperTranscriber
from config import (
    Settings,
    find_local_whisper_model,
    get_settings,
    is_offline,
    resolve_speech_language,
)
from i18n import t

logger = logging.getLogger(__name__)

# Ile razy z rzędu wolno dostać pustą transkrypcję, zanim oddamy sterowanie.
# Bez tego limitu hałaśliwe otoczenie (VAD wyzwala się na szumie, Whisper nie
# rozpoznaje treści) trzymałoby nasłuch w nieskończoność, bo każda pusta próba
# odnawiała limit czasu.
MAX_EMPTY_TRANSCRIPTS: int = 3


class PipelineEvent(StrEnum):
    """Zdarzenia potoku — interfejs wypisuje je jako komunikaty ``[MIC]``/``[WAKE]``."""

    STARTED = "started"
    LISTENING = "listening"
    WAITING_FOR_WAKE = "waiting_for_wake"
    WAKE_DETECTED = "wake_detected"
    IGNORED = "ignored"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class PipelineMessage:
    event: PipelineEvent
    text: str
    detail: str = ""


EventCallback = Callable[[PipelineMessage], None]


class SpeechPipelineError(RuntimeError):
    """Potok nie może działać na tej maszynie."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n       Podpowiedź: {self.hint}"
        return self.message


class SpeechToTextPipeline:
    """Nasłuchuje mikrofonu i zwraca transkrypcje kolejnych wypowiedzi."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        microphone: Microphone | None = None,
        vad: VoiceActivityDetector | None = None,
        segmenter: UtteranceSegmenter | None = None,
        transcriber: WhisperTranscriber | None = None,
        wake: WakeWordEngine | None = None,
        wake_transcriber: WhisperTranscriber | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._microphone = microphone if microphone is not None else Microphone(self._settings)
        self._segmenter = (
            segmenter if segmenter is not None else UtteranceSegmenter(self._settings, vad=vad)
        )
        self._transcriber = (
            transcriber if transcriber is not None else WhisperTranscriber(self._settings)
        )
        self._on_event = on_event
        self._started = False

        # --- bramka słowa aktywującego ---
        self._wake_model = wake_transcriber
        self._wake = wake if wake is not None else self._build_wake_engine()
        self._awake_until = 0.0
        self._armed_turn = False

        # --- oszczędzanie pamięci w ciszy ---
        # Czas ostatniej rzeczywistej pracy (mowa albo transkrypcja). Model
        # Whispera trzymany „na wszelki wypadek" nie zużywa cykli, ale zajmuje
        # kilkaset MB RAM-u albo VRAM-u — na maszynie z jednym GPU to jest
        # różnica między działającą grą a swapem.
        self._last_activity = time.monotonic()
        self._idle_unloaded = False

    # --- właściwości ----------------------------------------------------- #

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def microphone(self) -> Microphone:
        return self._microphone

    @property
    def transcriber(self) -> WhisperTranscriber:
        return self._transcriber

    @property
    def vad_name(self) -> str:
        return self._segmenter.vad_name

    @property
    def wake_engine(self) -> WakeWordEngine | None:
        return self._wake

    @property
    def wake_phrase(self) -> str:
        return self._wake.phrase if self._wake is not None else ""

    @property
    def wake_name(self) -> str:
        return self._wake.name if self._wake is not None else t("pipe.wake_disabled")

    @property
    def is_awake(self) -> bool:
        """Czy fraza już padła i wolno mówić bez powtarzania zawołania?"""
        return self._armed_turn or time.monotonic() < self._awake_until

    def requires_wake(self) -> bool:
        """Czy w tej chwili wypowiedzi są odrzucane bez frazy?"""
        return self._wake is not None and not self.is_awake

    def wake_up(self) -> None:
        """Otwórz okno rozmowy ręcznie (np. skrótem klawiszowym albo z GUI)."""
        self._mark_activity()
        self._armed_turn = True
        self._awake_until = time.monotonic() + self._settings.wake_window_s

    def sleep(self) -> None:
        """Zamknij okno rozmowy — kolejna wypowiedź znów wymaga frazy."""
        self._armed_turn = False
        self._awake_until = 0.0
        if self._wake is not None:
            self._wake.reset()

    def describe(self) -> str:
        device = self._microphone.device
        device_name = device.name if device else t("pipe.default_device")
        stages = [device_name, f"VAD {self.vad_name}"]
        if self._wake is not None:
            stages.append(
                t("pipe.describe_wake", phrase=self.wake_phrase, engine=self.wake_name)
            )
        stages.append(f"Whisper {self._transcriber.describe()}")
        return " → ".join(stages)

    # --- budowa detektora frazy ------------------------------------------- #

    def _wake_model_for_detection(self) -> WhisperTranscriber:
        """Model używany do wykrywania frazy — mały i osobny, chyba że ten sam.

        Domyślnie ``tiny``: detektor ma być tani, a nie dokładny. Gdy
        ``WAKE_WHISPER_MODEL`` jest puste albo równe modelowi głównemu,
        korzystamy z tego samego obiektu i nie zajmujemy pamięci dwa razy.
        """
        if self._wake_model is not None:
            return self._wake_model

        wanted = self._settings.wake_whisper_model.strip()
        if not wanted or wanted == self._settings.whisper_model:
            self._wake_model = self._transcriber
            return self._wake_model

        # Bez sieci nie ma jak dociągnąć osobnego modelu. Zamiast wywalać tryb
        # głosowy, wykrywamy frazę modelem głównym — wolniej, ale działa.
        if is_offline(self._settings) and find_local_whisper_model(wanted) is None:
            logger.info(
                "Model frazy %r nie jest pobrany, a tryb offline zabrania pobierania — "
                "używam modelu głównego.",
                wanted,
            )
            self._wake_model = self._transcriber
            return self._wake_model

        self._wake_model = WhisperTranscriber(
            self._settings.model_copy(update={"whisper_model": wanted})
        )
        return self._wake_model

    def _transcribe_for_wake(self, utterance: Utterance) -> str:
        """Transkrypcja NA POTRZEBY detekcji frazy — inne ustawienia niż zwykła.

        Dwie różnice, obie zmierzone na korpusie 480 transkrypcji:

        * **podpowiedź frazy** (``hotwords``): nazwa asystenta jest dla modelu
          obcym słowem i bez niej wychodzi „tymiku" albo „micu" — z podpowiedzią
          wykrywanie rośnie z 45% na 95% (model ``base``),
        * **wymuszony język**: fraza jest zapisana w ustawieniach użytkownika, a
          więc jej język jest znany. Zgadywanie na jednosekundowym fragmencie
          kończyło się japońskimi znakami zamiast „hej miku".
        """
        phrase = self._wake.phrase if self._wake is not None else ""
        return (
            self._wake_model_for_detection()
            .transcribe(
                utterance,
                language=resolve_speech_language(self._settings) or None,
                hotwords=phrase,
            )
            .text
        )

    def _build_wake_engine(self) -> WakeWordEngine | None:
        try:
            return create_wake_word_engine(self._settings, transcribe=self._transcribe_for_wake)
        except WakeWordError as exc:
            # Brak detektora nie może odebrać użytkownikowi trybu głosowego —
            # nasłuch działa dalej, tyle że bez bramki.
            logger.warning("Detektor frazy niedostępny: %s", exc.message)
            return None

    @staticmethod
    def is_available(settings: Settings | None = None) -> bool:
        """Czy tryb głosowy ma szansę zadziałać? Nie rzuca wyjątków."""
        return is_microphone_available(settings)

    # --- cykl życia ------------------------------------------------------- #

    def start(self) -> None:
        """Uruchom mikrofon i załaduj model transkrypcji."""
        if self._started:
            return
        try:
            self._microphone.start()
        except MicrophoneError as exc:
            raise SpeechPipelineError(exc.message, hint=exc.hint) from exc

        try:
            self._transcriber.load()
        except TranscriptionError as exc:
            self._microphone.stop()
            raise SpeechPipelineError(exc.message, hint=exc.hint) from exc

        self._load_wake_model()

        self._segmenter.reset()
        self.sleep()
        self._started = True
        self._mark_activity()
        self._emit(
            PipelineEvent.STARTED,
            t("pipe.started"),
            self.describe(),
        )

    def _load_wake_model(self) -> None:
        """Załaduj model detektora frazy z góry, żeby pierwsze zawołanie nie czekało.

        Awaria nie może zablokować trybu głosowego: bramka jest wtedy zdejmowana,
        a użytkownik dostaje jasny komunikat, że nasłuch idzie bez frazy.
        """
        if self._wake is None or self._wake.mode != "utterance":
            return
        model = self._wake_model_for_detection()
        if model is self._transcriber:
            return  # główny model już jest załadowany
        try:
            model.load()
        except TranscriptionError as exc:
            logger.warning("Nie udało się załadować modelu frazy: %s", exc.message)
            self._emit(
                PipelineEvent.ERROR,
                t("pipe.wake_failed", error=exc.message),
                exc.hint,
            )
            self._wake = None

    # --- oszczędzanie pamięci w ciszy -------------------------------------- #

    @property
    def idle_unload_after_s(self) -> float:
        """Po ilu sekundach ciszy zwalniamy model. ``0`` = nigdy."""
        return self._settings.whisper_idle_unload_s

    @property
    def is_idle_unloaded(self) -> bool:
        """Czy główny model został zwolniony z powodu ciszy."""
        return self._idle_unloaded

    def _mark_activity(self) -> None:
        self._last_activity = time.monotonic()
        self._idle_unloaded = False

    def _maybe_unload_idle(self) -> bool:
        """Zwolnij główny model po dostatecznie długiej ciszy.

        Zwalniany jest WYŁĄCZNIE model główny (``small``/``medium`` — setki MB,
        często na GPU). Model detektora frazy (``tiny``, ok. 39 MB) zostaje:
        to on decyduje, czy w ogóle się obudzić, więc jego przeładowanie
        opóźniałoby każde zawołanie. Model główny przeładuje się sam przy
        pierwszej transkrypcji (``WhisperTranscriber.transcribe`` woła
        ``load()``), co kosztuje jednorazowo ok. 1–3 s.
        """
        limit = self._settings.whisper_idle_unload_s
        if limit <= 0 or self._idle_unloaded:
            return False
        if self._segmenter.is_recording or self.is_awake:
            return False
        if time.monotonic() - self._last_activity < limit:
            return False
        if not self._transcriber.is_loaded:
            # Nic do zwolnienia, ale znacznik stawiamy — inaczej sprawdzenie
            # powtarzałoby się przy każdej ramce ciszy.
            self._idle_unloaded = True
            return False
        if self._wake_model is not None and self._wake_model is self._transcriber:
            # Detektor frazy używa TEGO SAMEGO obiektu (WAKE_WHISPER_MODEL puste
            # albo równe głównemu). Zwolnienie kosztowałoby przeładowanie przy
            # każdym zawołaniu, więc zostawiamy model w pamięci.
            self._idle_unloaded = True
            return False
        self._transcriber.unload()
        self._idle_unloaded = True
        logger.info(
            "Cisza przez %.0f s — zwolniono model Whisper '%s' (wróci przy pierwszej wypowiedzi)",
            limit,
            self._transcriber.model_name,
        )
        return True

    def stop(self) -> None:
        """Zatrzymaj mikrofon; model zostaje w pamięci na wypadek ponownego startu."""
        if not self._started:
            return
        self._segmenter.flush()
        self._microphone.stop()
        self.sleep()
        self._started = False
        self._emit(PipelineEvent.STOPPED, t("pipe.stopped"))

    def close(self) -> None:
        """Zatrzymaj potok i zwolnij modele."""
        self.stop()
        self._transcriber.unload()
        if self._wake_model is not None and self._wake_model is not self._transcriber:
            self._wake_model.unload()

    def __enter__(self) -> SpeechToTextPipeline:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- nasłuch ----------------------------------------------------------- #

    def _emit(self, event: PipelineEvent, text: str, detail: str = "") -> None:
        message = PipelineMessage(event=event, text=text, detail=detail)
        logger.debug("Potok audio: %s — %s %s", event, text, detail)
        if self._on_event is not None:
            try:
                self._on_event(message)
            except Exception as exc:  # błąd interfejsu nie może zabić nasłuchu
                logger.warning("Callback zdarzeń potoku zgłosił błąd: %s", exc)

    def _announce_listening(self) -> None:
        if self.requires_wake():
            self._emit(
                PipelineEvent.WAITING_FOR_WAKE,
                t("pipe.waiting_for_wake", phrase=self.wake_phrase),
                self.wake_name,
            )
        else:
            self._emit(PipelineEvent.LISTENING, t("pipe.listening"))

    def _register_wake(self, match: WakeMatch) -> None:
        self._armed_turn = True
        self._awake_until = time.monotonic() + self._settings.wake_window_s
        score = f"{match.score:.2f}"
        detail = (
            t("pipe.wake_heard", score=score, heard=repr(match.heard))
            if match.heard
            else t("pipe.wake_score", score=score)
        )
        self._emit(
            PipelineEvent.WAKE_DETECTED, t("pipe.wake_detected", phrase=match.phrase), detail
        )

    def _check_wake_utterance(self, utterance: Utterance) -> WakeMatch | None:
        """Sprawdź wypowiedź detektorem frazy (silniki segmentowe)."""
        if self._wake is None or self._wake.mode != "utterance":
            return None
        return self._wake.process_utterance(utterance)

    def listen_once(self, timeout_s: float | None = None) -> Transcript | None:
        """Czekaj na jedną wypowiedź i zwróć jej transkrypcję.

        Zwraca ``None``, gdy w czasie ``timeout_s`` nikt nic nie powiedział.
        Limit czasu liczy się wyłącznie do ciszy — rozpoczętej wypowiedzi
        nigdy nie przerywa w połowie.

        Gdy bramka frazy jest aktywna, wypowiedzi bez słowa aktywującego są
        odrzucane i **nie trafiają do głównego modelu** — kończą się zdarzeniem
        ``IGNORED``.
        """
        self.start()

        limit = self._settings.vad_listen_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + limit if limit > 0 else None
        empty_results = 0
        self._armed_turn = False
        self._microphone.clear()
        self._segmenter.reset()
        if self._wake is not None:
            self._wake.reset()
        self._announce_listening()

        while True:
            frame = self._microphone.read(timeout=0.2)

            if frame is None:
                # Kolejka jest pusta — mikrofon nic nie oddał przez 200 ms. To
                # jedyne miejsce w pętli, w którym na pewno nic się nie dzieje,
                # więc tu sprawdzamy, czy nie zwolnić modelu (patrz
                # `_maybe_unload_idle`). Czekanie jest blokujące na kolejce, bez
                # aktywnego odpytywania — pętla sama z siebie nie zużywa CPU.
                self._maybe_unload_idle()
                if (
                    deadline is not None
                    and not self._segmenter.is_recording
                    and time.monotonic() > deadline
                ):
                    self._emit(PipelineEvent.TIMEOUT, t("pipe.timeout", seconds=f"{limit:.0f}"))
                    return None
                continue

            # Silniki strumieniowe (openWakeWord) oglądają każdą ramkę —
            # fraza może paść w środku wypowiedzi, którą segmenter właśnie nagrywa.
            streaming_wake = self._wake is not None and self._wake.mode == "stream"
            if streaming_wake and self._wake is not None and not self.is_awake:
                streamed = self._wake.process_frame(frame)
                if streamed is not None:
                    self._register_wake(streamed)

            was_recording = self._segmenter.is_recording
            utterance = self._segmenter.push(frame)

            if not was_recording and self._segmenter.is_recording:
                self._mark_activity()
                self._emit(PipelineEvent.SPEECH_START, t("pipe.speech_start"))

            if utterance is None:
                if not self._segmenter.is_recording:
                    self._maybe_unload_idle()
                if (
                    deadline is not None
                    and not self._segmenter.is_recording
                    and time.monotonic() > deadline
                ):
                    self._emit(PipelineEvent.TIMEOUT, t("pipe.timeout", seconds=f"{limit:.0f}"))
                    return None
                continue

            if utterance.truncated:
                # Wypowiedź ucięta twardym limitem prawie zawsze znaczy, że VAD
                # nie widzi ciszy — czyli próg jest za niski dla tego mikrofonu.
                self._emit(
                    PipelineEvent.SPEECH_END,
                    t("pipe.speech_end_truncated", seconds=f"{utterance.duration_s:.1f}"),
                    t("pipe.speech_end_truncated_hint"),
                )
            else:
                self._emit(
                    PipelineEvent.SPEECH_END,
                    t("pipe.speech_end", seconds=f"{utterance.duration_s:.1f}"),
                )
            if self.requires_wake():
                match = self._check_wake_utterance(utterance)
                if match is None:
                    # Rozmowa w tle: główny model ani model językowy nigdy jej
                    # nie zobaczą. Limit ciszy celowo NIE jest odnawiany —
                    # inaczej gadające radio trzymałoby nasłuch w nieskończoność.
                    self._emit(
                        PipelineEvent.IGNORED,
                        t("pipe.ignored", phrase=self.wake_phrase),
                        t("pipe.ignored_detail", seconds=f"{utterance.duration_s:.1f}"),
                    )
                    continue

                self._register_wake(match)
                if not match.has_command:
                    # Samo zawołanie — teraz czekamy na właściwe polecenie.
                    self._emit(PipelineEvent.LISTENING, t("pipe.listening"))
                    if deadline is not None:
                        deadline = time.monotonic() + limit
                    continue

            self._emit(PipelineEvent.TRANSCRIBING, t("pipe.transcribing"))
            self._mark_activity()

            try:
                transcript = self._transcriber.transcribe(utterance)
            except TranscriptionError as exc:
                self._emit(PipelineEvent.ERROR, exc.message, exc.hint)
                return None

            if transcript.is_empty:
                empty_results += 1
                if empty_results >= MAX_EMPTY_TRANSCRIPTS:
                    self._emit(
                        PipelineEvent.EMPTY,
                        t("pipe.empty_giving_up", count=empty_results),
                    )
                    return None
                self._emit(PipelineEvent.EMPTY, t("pipe.empty"))
                if deadline is not None:
                    deadline = time.monotonic() + limit
                continue

            transcript = self._without_wake_phrase(transcript)
            if transcript.is_empty:
                # Cała wypowiedź była samym zawołaniem („hej miku") — polecenie
                # dopiero nadejdzie.
                self._emit(PipelineEvent.LISTENING, t("pipe.listening"))
                if deadline is not None:
                    deadline = time.monotonic() + limit
                continue

            self._refresh_wake_window()
            self._emit(
                PipelineEvent.TRANSCRIBED,
                transcript.text,
                f"{transcript.language or '?'}, {transcript.audio_duration_s:.1f} s audio "
                f"w {transcript.processing_s:.1f} s",
            )
            return transcript

    def _without_wake_phrase(self, transcript: Transcript) -> Transcript:
        """Odetnij frazę z początku polecenia („hej miku, jaka pogoda")."""
        if self._wake is None:
            return transcript
        stripped = self._wake.strip_phrase(transcript.text).strip()
        if stripped == transcript.text.strip():
            return transcript
        return replace(transcript, text=stripped)

    def _refresh_wake_window(self) -> None:
        """Po udanym poleceniu okno rozmowy liczy się od nowa."""
        if self._wake is None:
            return
        self._armed_turn = False
        self._awake_until = time.monotonic() + self._settings.wake_window_s

    def transcripts(self, timeout_s: float | None = None) -> Iterator[Transcript]:
        """Nieskończony strumień transkrypcji; kończy się po ciszy dłuższej niż limit."""
        while True:
            transcript = self.listen_once(timeout_s)
            if transcript is None:
                return
            yield transcript


__all__ = [
    "EventCallback",
    "PipelineEvent",
    "PipelineMessage",
    "SpeechPipelineError",
    "SpeechToTextPipeline",
]
