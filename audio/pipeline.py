"""Voice input pipeline: MICROPHONE → VAD → [WAKE WORD] → WHISPER → TEXT.

The pipeline is injectable end to end: the microphone, the VAD, the segmenter,
the phrase detector and the transcriber can all be swapped for fakes, which is
why the tests need no hardware.

The wake word gate (Phase 3) stands in front of the main model: until the phrase
is spoken, no utterance reaches the large Whisper, let alone the language model.
Once the phrase is detected, a conversation window opens (``WAKE_WINDOW_S``) in
which you speak normally, without repeating the call.

The class is deliberately synchronous. Listening is a loop waiting for frames
from a queue filled by the PortAudio thread, and Whisper computes in native code
anyway — asyncio would add nothing here but complication. For use with the GUI
(Phase 10) there is :meth:`WhisperTranscriber.transcribe_async`.
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

# How many empty transcripts in a row we tolerate before handing control back.
# Without this limit a noisy room (the VAD triggers on noise, Whisper recognises
# nothing) would hold the listen forever, because every empty attempt renewed
# the time limit.
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
    """The pipeline cannot run on this machine."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


class SpeechToTextPipeline:
    """Listens to the microphone and returns transcripts of successive utterances."""

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

        # --- the wake word gate ---
        self._wake_model = wake_transcriber
        self._wake = wake if wake is not None else self._build_wake_engine()
        self._awake_until = 0.0
        self._armed_turn = False

        # --- saving memory during silence ---
        # The time of the last real work (speech or transcription). A Whisper
        # model kept loaded "just in case" consumes no cycles but occupies a few
        # hundred MB of RAM or VRAM — on a machine with a single GPU that is the
        # difference between a game that runs and swapping.
        self._last_activity = time.monotonic()
        self._idle_unloaded = False

    # --- properties ------------------------------------------------------ #

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
        """Has the phrase already been spoken, so one may speak without repeating it?"""
        return self._armed_turn or time.monotonic() < self._awake_until

    def requires_wake(self) -> bool:
        """Are utterances currently rejected without the phrase?"""
        return self._wake is not None and not self.is_awake

    def wake_up(self) -> None:
        """Open the conversation window manually (e.g. by a shortcut or from the GUI)."""
        self._mark_activity()
        self._armed_turn = True
        self._awake_until = time.monotonic() + self._settings.wake_window_s

    def sleep(self) -> None:
        """Close the conversation window — the next utterance needs the phrase again."""
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
        """The model used for phrase detection — small and separate, unless it is the same one.

        ``tiny`` by default: the detector is meant to be cheap, not accurate. When
        ``WAKE_WHISPER_MODEL`` is empty or equal to the main model, we use the
        same object and do not occupy memory twice.
        """
        if self._wake_model is not None:
            return self._wake_model

        wanted = self._settings.wake_whisper_model.strip()
        if not wanted or wanted == self._settings.whisper_model:
            self._wake_model = self._transcriber
            return self._wake_model

        # Without a network there is no way to fetch a separate model. Rather
        # than breaking voice mode, we detect the phrase with the main model —
        # slower, but it works.
        if is_offline(self._settings) and find_local_whisper_model(wanted) is None:
            logger.info(
                "The phrase model %r is not downloaded and offline mode forbids "
                "downloading — using the main model.",
                wanted,
            )
            self._wake_model = self._transcriber
            return self._wake_model

        self._wake_model = WhisperTranscriber(
            self._settings.model_copy(update={"whisper_model": wanted})
        )
        return self._wake_model

    def _transcribe_for_wake(self, utterance: Utterance) -> str:
        """Transcription FOR THE PURPOSE of phrase detection — different settings from the usual.

        Two differences, both measured on a corpus of 480 transcriptions:

        * **the phrase hint** (``hotwords``): the assistant's name is a foreign
          word to the model, and without it the result is "tymiku" or "micu" —
          with the hint, detection rises from 45% to 95% (the ``base`` model),
        * **a forced language**: the phrase is written in the user's settings, so
          its language is known. Guessing on a one-second fragment ended in
          Japanese characters instead of "hey miku".
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
            # A missing detector must not take voice mode away from the user —
            # listening carries on, only without the gate.
            logger.warning("Phrase detector unavailable: %s", exc.message)
            return None

    @staticmethod
    def is_available(settings: Settings | None = None) -> bool:
        """Does voice mode stand a chance of working? Raises nothing."""
        return is_microphone_available(settings)

    # --- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        """Start the microphone and load the transcription model."""
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
        """Load the phrase detector model up front, so the first call does not wait.

        A failure must not block voice mode: the gate is then removed and the
        user gets a clear message that listening proceeds without the phrase.
        """
        if self._wake is None or self._wake.mode != "utterance":
            return
        model = self._wake_model_for_detection()
        if model is self._transcriber:
            return  # the main model is already loaded
        try:
            model.load()
        except TranscriptionError as exc:
            logger.warning("Could not load the phrase model: %s", exc.message)
            self._emit(
                PipelineEvent.ERROR,
                t("pipe.wake_failed", error=exc.message),
                exc.hint,
            )
            self._wake = None

    # --- saving memory during silence -------------------------------------- #

    @property
    def idle_unload_after_s(self) -> float:
        """Po ilu sekundach ciszy zwalniamy model. ``0`` = nigdy."""
        return self._settings.whisper_idle_unload_s

    @property
    def is_idle_unloaded(self) -> bool:
        """Has the main model been released because of silence?"""
        return self._idle_unloaded

    def _mark_activity(self) -> None:
        self._last_activity = time.monotonic()
        self._idle_unloaded = False

    def _maybe_unload_idle(self) -> bool:
        """Release the main model after sufficiently long silence.

        ONLY the main model is released (``small``/``medium`` — hundreds of MB,
        often on the GPU). The phrase detector model (``tiny``, about 39 MB)
        stays: it is the one that decides whether to wake up at all, so reloading
        it would delay every call. The main model reloads itself at the first
        transcription (``WhisperTranscriber.transcribe`` calls ``load()``), which
        costs a one-off 1–3 s.
        """
        limit = self._settings.whisper_idle_unload_s
        if limit <= 0 or self._idle_unloaded:
            return False
        if self._segmenter.is_recording or self.is_awake:
            return False
        if time.monotonic() - self._last_activity < limit:
            return False
        if not self._transcriber.is_loaded:
            # Nothing to release, but we set the marker anyway — otherwise the
            # check would repeat on every frame of silence.
            self._idle_unloaded = True
            return False
        if self._wake_model is not None and self._wake_model is self._transcriber:
            # The phrase detector uses THE SAME object (WAKE_WHISPER_MODEL is
            # empty or equal to the main one). Releasing it would cost a reload
            # on every call, so we leave the model in memory.
            self._idle_unloaded = True
            return False
        self._transcriber.unload()
        self._idle_unloaded = True
        logger.info(
            "Silence for %.0f s — released the Whisper model '%s' (it returns at the first utterance)",
            limit,
            self._transcriber.model_name,
        )
        return True

    def stop(self) -> None:
        """Stop the microphone; the model stays in memory in case of a restart."""
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

    # --- listening --------------------------------------------------------- #

    def _emit(self, event: PipelineEvent, text: str, detail: str = "") -> None:
        message = PipelineMessage(event=event, text=text, detail=detail)
        logger.debug("Potok audio: %s — %s %s", event, text, detail)
        if self._on_event is not None:
            try:
                self._on_event(message)
            except Exception as exc:  # an interface error must not kill the listen
                logger.warning("The pipeline event callback raised: %s", exc)

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
        """Check the utterance with the phrase detector (utterance-mode engines)."""
        if self._wake is None or self._wake.mode != "utterance":
            return None
        return self._wake.process_utterance(utterance)

    def listen_once(self, timeout_s: float | None = None) -> Transcript | None:
        """Wait for one utterance and return its transcript.

        Returns ``None`` when nobody said anything within ``timeout_s``. The time
        limit counts silence only — it never interrupts an utterance already in
        progress.

        When the phrase gate is active, utterances without the wake word are
        rejected and **do not reach the main model** — they end with an
        ``IGNORED`` event.
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
                # The queue is empty — the microphone returned nothing for
                # 200 ms. This is the only place in the loop where we know for
                # certain that nothing is happening, so this is where we check
                # whether to release the model (see `_maybe_unload_idle`). The
                # wait blocks on the queue rather than polling — the loop
                # consumes no CPU by itself.
                self._maybe_unload_idle()
                if (
                    deadline is not None
                    and not self._segmenter.is_recording
                    and time.monotonic() > deadline
                ):
                    self._emit(PipelineEvent.TIMEOUT, t("pipe.timeout", seconds=f"{limit:.0f}"))
                    return None
                continue

            # Streaming engines (openWakeWord) inspect every frame — the phrase
            # may fall in the middle of an utterance the segmenter is recording.
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
                # An utterance cut by the hard limit almost always means the VAD
                # does not see silence — that is, the threshold is too low for
                # this microphone.
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
                    # Background conversation: neither the main model nor the
                    # language model will ever see it. The silence limit is
                    # deliberately NOT renewed — otherwise a talking radio would
                    # hold the listen forever.
                    self._emit(
                        PipelineEvent.IGNORED,
                        t("pipe.ignored", phrase=self.wake_phrase),
                        t("pipe.ignored_detail", seconds=f"{utterance.duration_s:.1f}"),
                    )
                    continue

                self._register_wake(match)
                if not match.has_command:
                    # Just the call — now we wait for the actual command.
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
                # The whole utterance was the call itself ("hey miku") — the
                # command is yet to come.
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
        """Strip the phrase from the start of a command ("hey miku, what is the weather")."""
        if self._wake is None:
            return transcript
        stripped = self._wake.strip_phrase(transcript.text).strip()
        if stripped == transcript.text.strip():
            return transcript
        return replace(transcript, text=stripped)

    def _refresh_wake_window(self) -> None:
        """After a successful command the conversation window starts counting again."""
        if self._wake is None:
            return
        self._armed_turn = False
        self._awake_until = time.monotonic() + self._settings.wake_window_s

    def transcripts(self, timeout_s: float | None = None) -> Iterator[Transcript]:
        """An endless stream of transcripts; ends after silence longer than the limit."""
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
