"""Testy potoku MIKROFON → VAD → NAGRYWANIE → WHISPER → TEKST.

Wszystkie elementy są wstrzykiwane jako atrapy — test nie dotyka sprzętu ani
modelu i wykonuje się w ułamku sekundy.
"""

from __future__ import annotations

import time

import pytest
from conftest import make_silence, make_tone

from audio.microphone import AudioFrame, MicrophoneUnavailableError
from audio.pipeline import (
    PipelineEvent,
    PipelineMessage,
    SpeechPipelineError,
    SpeechToTextPipeline,
)
from audio.vad import Utterance, UtteranceSegmenter
from audio.wakeword import PhraseMatcher, WakeMatch
from audio.whisper import Transcript
from config import Settings


class FakeMicrophone:
    """Mikrofon oddający wcześniej przygotowaną listę ramek."""

    def __init__(self, frames: list[AudioFrame], *, fail_on_start: Exception | None = None) -> None:
        self._frames = list(frames)
        self._fail_on_start = fail_on_start
        self.started = False
        self.stopped = False
        self.cleared = 0
        self.device = None

    def start(self) -> None:
        if self._fail_on_start is not None:
            raise self._fail_on_start
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.started = False

    def clear(self) -> None:
        self.cleared += 1

    def read(self, timeout: float | None = 0.2) -> AudioFrame | None:
        if self._frames:
            return self._frames.pop(0)
        return None


class FakeTranscriber:
    """Transkryber zwracający ustalony tekst i liczący wywołania."""

    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts if texts is not None else ["rozpoznany tekst"])
        self.calls: list[Utterance] = []
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def describe(self) -> str:
        return "atrapa"

    def transcribe(self, utterance: Utterance) -> Transcript:
        self.calls.append(utterance)
        text = self._texts.pop(0) if self._texts else ""
        return Transcript(
            text=text,
            language="pl",
            language_probability=0.99,
            audio_duration_s=utterance.duration_s,
            processing_s=0.01,
        )


class ScriptedVAD:
    def __init__(self, script: list[bool]) -> None:
        self._script = list(script)
        self._index = 0

    @property
    def name(self) -> str:
        return "scripted"

    def reset(self) -> None:
        self._index = 0

    def is_speech(self, frame: AudioFrame) -> bool:
        value = self._script[self._index] if self._index < len(self._script) else False
        self._index += 1
        return value


def frames_from(script: list[bool]) -> list[AudioFrame]:
    base = time.monotonic()
    return [
        AudioFrame(
            samples=make_tone(320) if speech else make_silence(320),
            sample_rate=16_000,
            timestamp=base + index * 0.02,
        )
        for index, speech in enumerate(script)
    ]


def build_pipeline(
    settings: Settings,
    script: list[bool],
    *,
    texts: list[str] | None = None,
    events: list[PipelineMessage] | None = None,
    wake: object | None = None,
) -> tuple[SpeechToTextPipeline, FakeMicrophone, FakeTranscriber]:
    """Potok na atrapach. Bez ``wake=`` bramka frazy jest wyłączona.

    Testy z Fazy 2 opisują zachowanie samego potoku, a nie bramki — dlatego
    domyślnie ją zdejmujemy. Bramka ma własne testy niżej.
    """
    active = settings if wake is not None else settings.model_copy(update={"wake_enabled": False})
    # Detektor frazy ma w tych testach korzystać z WSTRZYKNIĘTEJ atrapy modelu.
    # Bez tego potok zbudowałby drugi, prawdziwy WhisperTranscriber dla modelu
    # z WAKE_WHISPER_MODEL i próbował go pobrać — test sięgałby do sieci.
    active = active.model_copy(update={"wake_whisper_model": active.whisper_model})
    microphone = FakeMicrophone(frames_from(script))
    transcriber = FakeTranscriber(texts)
    segmenter = UtteranceSegmenter(active, vad=ScriptedVAD(script))
    pipeline = SpeechToTextPipeline(
        active,
        microphone=microphone,  # type: ignore[arg-type]
        segmenter=segmenter,
        transcriber=transcriber,  # type: ignore[arg-type]
        wake=wake,  # type: ignore[arg-type]
        on_event=events.append if events is not None else None,
    )
    return pipeline, microphone, transcriber


# --------------------------------------------------------------------------- #


def test_pelny_przeplyw_zwraca_tekst(settings: Settings) -> None:
    script = [False] * 3 + [True] * 15 + [False] * 12
    events: list[PipelineMessage] = []
    pipeline, microphone, transcriber = build_pipeline(
        settings, script, texts=["jaka jest pogoda"], events=events
    )

    transcript = pipeline.listen_once()

    assert transcript is not None
    assert transcript.text == "jaka jest pogoda"
    assert microphone.started is True
    assert transcriber.loaded is True
    assert len(transcriber.calls) == 1

    kinds = [message.event for message in events]
    assert PipelineEvent.STARTED in kinds
    assert PipelineEvent.LISTENING in kinds
    assert PipelineEvent.SPEECH_START in kinds
    assert PipelineEvent.SPEECH_END in kinds
    assert PipelineEvent.TRANSCRIBING in kinds
    assert PipelineEvent.TRANSCRIBED in kinds


def test_sama_cisza_konczy_sie_timeoutem_bez_transkrypcji(settings: Settings) -> None:
    fast = settings.model_copy(update={"vad_listen_timeout_s": 0.05})
    events: list[PipelineMessage] = []
    pipeline, _, transcriber = build_pipeline(fast, [False] * 5, events=events)

    assert pipeline.listen_once() is None
    assert transcriber.calls == []  # Whisper nie dostał ani jednej próbki ciszy
    assert PipelineEvent.TIMEOUT in [message.event for message in events]


def test_pusta_transkrypcja_nie_konczy_nasluchu(settings: Settings) -> None:
    """Halucynacja odsiana przez transkryber → potok słucha dalej, nie zwraca pustki."""
    script = [False, True, True, True, True, True, True, True, True, True] + [False] * 12
    fast = settings.model_copy(update={"vad_listen_timeout_s": 0.05})
    events: list[PipelineMessage] = []
    pipeline, _, _ = build_pipeline(fast, script, texts=[""], events=events)

    assert pipeline.listen_once() is None
    kinds = [message.event for message in events]
    assert PipelineEvent.EMPTY in kinds
    assert PipelineEvent.TIMEOUT in kinds


def test_halasliwe_otoczenie_nie_blokuje_nasluchu_na_zawsze(settings: Settings) -> None:
    """Regresja: puste transkrypcje odnawiały limit czasu i pętla nie kończyła się.

    VAD wyzwala się na szumie, Whisper nie rozpoznaje treści — po ustalonej
    liczbie takich prób sterowanie musi wrócić do rozmówcy.
    """
    from audio.pipeline import MAX_EMPTY_TRANSCRIPTS

    # Ciąg krótkich „wypowiedzi" szumu: mowa, cisza, mowa, cisza...
    burst = [True] * 10 + [False] * 12
    script = burst * (MAX_EMPTY_TRANSCRIPTS + 3)
    events: list[PipelineMessage] = []
    pipeline, _, transcriber = build_pipeline(
        settings, script, texts=[""] * (MAX_EMPTY_TRANSCRIPTS + 3), events=events
    )

    assert pipeline.listen_once() is None
    assert len(transcriber.calls) == MAX_EMPTY_TRANSCRIPTS
    # Sprawdzamy RODZAJ zdarzenia, nie jego treść: napis idzie przez i18n
    # i zależy od UI_LANGUAGE, a warunek, o który tu chodzi — nie.
    assert any(message.event is PipelineEvent.EMPTY for message in events)
    ostatnie = [m for m in events if m.event is PipelineEvent.EMPTY][-1]
    assert str(MAX_EMPTY_TRANSCRIPTS) in ostatnie.text


def test_blad_mikrofonu_daje_wyjatek_potoku(settings: Settings) -> None:
    microphone = FakeMicrophone(
        [], fail_on_start=MicrophoneUnavailableError("brak mikrofonu", hint="podłącz sprzęt")
    )
    pipeline = SpeechToTextPipeline(
        settings,
        microphone=microphone,  # type: ignore[arg-type]
        segmenter=UtteranceSegmenter(settings, vad=ScriptedVAD([])),
        transcriber=FakeTranscriber(),  # type: ignore[arg-type]
    )

    with pytest.raises(SpeechPipelineError) as error:
        pipeline.start()
    assert "brak mikrofonu" in error.value.message
    assert "podłącz sprzęt" in error.value.user_message


def test_stop_i_close_zwalniaja_zasoby(settings: Settings) -> None:
    pipeline, microphone, transcriber = build_pipeline(settings, [False] * 2)
    pipeline.start()
    assert pipeline.is_started is True

    pipeline.close()
    assert microphone.stopped is True
    assert transcriber.unloaded is True
    assert pipeline.is_started is False


def test_menedzer_kontekstu_sprzata_po_sobie(settings: Settings) -> None:
    pipeline, microphone, transcriber = build_pipeline(settings, [False] * 2)
    with pipeline:
        assert pipeline.is_started is True
    assert microphone.stopped is True
    assert transcriber.unloaded is True


def test_kolejka_jest_czyszczona_przed_kazdym_nasluchem(settings: Settings) -> None:
    fast = settings.model_copy(update={"vad_listen_timeout_s": 0.05})
    pipeline, microphone, _ = build_pipeline(fast, [False] * 3)

    pipeline.listen_once()
    pipeline.listen_once()

    # Stare ramki nie mogą wpaść do nowej wypowiedzi.
    assert microphone.cleared >= 2


def test_dostepnosc_bez_mikrofonu_jest_falszywa(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert SpeechToTextPipeline.is_available(settings) is False


# --------------------------------------------------------------------------- #
# Bramka słowa aktywującego (Faza 3)
# --------------------------------------------------------------------------- #


class ScriptedWake:
    """Detektor frazy sterowany z testu: kolejne wypowiedzi → kolejne wyniki."""

    def __init__(self, results: list[WakeMatch | None], *, mode: str = "utterance") -> None:
        self._results = list(results)
        self._mode = mode
        self.utterances: list[Utterance] = []
        self.frames = 0
        self.resets = 0

    @property
    def name(self) -> str:
        return "atrapa"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def phrase(self) -> str:
        return "hej miku"

    def reset(self) -> None:
        self.resets += 1

    def process_frame(self, frame: AudioFrame) -> WakeMatch | None:
        self.frames += 1
        if self._mode != "stream":
            return None
        return self._results.pop(0) if self._results else None

    def process_utterance(self, utterance: Utterance) -> WakeMatch | None:
        self.utterances.append(utterance)
        return self._results.pop(0) if self._results else None

    def strip_phrase(self, text: str) -> str:
        return PhraseMatcher(self.phrase).strip_phrase(text)


def wake_match(command: str = "") -> WakeMatch:
    return WakeMatch(phrase="hej miku", score=0.95, heard="hej miku", command=command)


def test_mowa_bez_frazy_nie_dociera_do_glownego_modelu(settings: Settings) -> None:
    """Sedno Fazy 3: tło jest odrzucane, zanim ruszy duży Whisper."""
    script = [False, True, True, True, True, True, True, True, True, True] + [False] * 12
    fast = settings.model_copy(update={"vad_listen_timeout_s": 0.05})
    events: list[PipelineMessage] = []
    wake = ScriptedWake([None])
    pipeline, _, transcriber = build_pipeline(fast, script, events=events, wake=wake)

    assert pipeline.listen_once() is None
    assert transcriber.calls == []  # główny model nie dostał ani jednej próbki
    assert len(wake.utterances) == 1  # ale bramka fragment obejrzała
    kinds = [message.event for message in events]
    assert PipelineEvent.WAITING_FOR_WAKE in kinds
    assert PipelineEvent.IGNORED in kinds


def test_po_wykryciu_frazy_polecenie_trafia_do_modelu(settings: Settings) -> None:
    """„Hej Miku" a potem osobno polecenie — dwie wypowiedzi, jedna transkrypcja."""
    burst = [True] * 10 + [False] * 12
    script = [False, *burst * 2]
    events: list[PipelineMessage] = []
    wake = ScriptedWake([wake_match(), None])
    pipeline, _, transcriber = build_pipeline(
        settings, script, texts=["jaka jest pogoda"], events=events, wake=wake
    )

    transcript = pipeline.listen_once()

    assert transcript is not None
    assert transcript.text == "jaka jest pogoda"
    assert len(transcriber.calls) == 1
    assert PipelineEvent.WAKE_DETECTED in [message.event for message in events]


def test_fraza_i_polecenie_jednym_tchem_sa_transkrybowane_raz(settings: Settings) -> None:
    script = [False] + [True] * 10 + [False] * 12
    wake = ScriptedWake([wake_match(command="jaka jest pogoda")])
    pipeline, _, transcriber = build_pipeline(
        settings, script, texts=["Hej Miku, jaka jest pogoda"], wake=wake
    )

    transcript = pipeline.listen_once()

    assert transcript is not None
    # Fraza jest odcięta z tekstu oddawanego modelowi językowemu.
    assert transcript.text == "jaka jest pogoda"
    assert len(transcriber.calls) == 1


def test_okno_rozmowy_pozwala_mowic_bez_powtarzania_frazy(settings: Settings) -> None:
    burst = [True] * 10 + [False] * 12
    with_window = settings.model_copy(update={"wake_window_s": 60.0})
    wake = ScriptedWake([wake_match(command="pierwsze polecenie")])
    pipeline, microphone, transcriber = build_pipeline(
        with_window, [False, *burst], texts=["pierwsze polecenie"], wake=wake
    )

    assert pipeline.listen_once() is not None
    assert pipeline.is_awake is True

    # Druga tura: nowe ramki, bramka już nie pyta o frazę.
    microphone._frames = frames_from([False, *burst])  # atrapa udostępnia te pola wprost
    transcriber._texts = ["drugie polecenie"]  # atrapa udostępnia te pola wprost
    second = pipeline.listen_once()

    assert second is not None
    assert second.text == "drugie polecenie"
    assert len(wake.utterances) == 1  # bramka nie była już pytana


def test_po_wygasnieciu_okna_znowu_potrzebna_jest_fraza(settings: Settings) -> None:
    script = [False] + [True] * 10 + [False] * 12
    fast = settings.model_copy(update={"wake_window_s": 0.0, "vad_listen_timeout_s": 0.05})
    wake = ScriptedWake([wake_match(command="polecenie"), None])
    pipeline, microphone, transcriber = build_pipeline(
        fast, script, texts=["polecenie"], wake=wake
    )

    assert pipeline.listen_once() is not None
    assert pipeline.is_awake is False  # okno zerowe = jedno polecenie na zawołanie

    microphone._frames = frames_from(script)  # atrapa udostępnia te pola wprost
    transcriber._texts = ["cokolwiek"]  # atrapa udostępnia te pola wprost
    assert pipeline.listen_once() is None
    assert len(wake.utterances) == 2


def test_reczne_wybudzenie_omija_fraze(settings: Settings) -> None:
    script = [False] + [True] * 10 + [False] * 12
    wake = ScriptedWake([None])
    pipeline, _, _ = build_pipeline(settings, script, texts=["polecenie"], wake=wake)

    pipeline.start()
    pipeline.wake_up()
    transcript = pipeline.listen_once()

    assert transcript is not None
    assert transcript.text == "polecenie"
    assert wake.utterances == []


def test_silnik_strumieniowy_wybudza_w_trakcie_wypowiedzi(settings: Settings) -> None:
    """openWakeWord melduje trafienie z ramek — wypowiedź leci dalej do modelu."""
    script = [False] + [True] * 10 + [False] * 12
    wake = ScriptedWake([None, wake_match()], mode="stream")
    pipeline, _, _ = build_pipeline(
        settings, script, texts=["Hej Miku, włącz światło"], wake=wake
    )

    transcript = pipeline.listen_once()

    assert transcript is not None
    assert transcript.text == "włącz światło"
    assert wake.frames > 0
    assert wake.utterances == []  # silnik strumieniowy nie ogląda całych wypowiedzi


def test_opis_potoku_pokazuje_fraze(settings: Settings) -> None:
    pipeline, _, _ = build_pipeline(settings, [False], wake=ScriptedWake([]))
    assert "hej miku" in pipeline.describe()
    assert pipeline.requires_wake() is True
