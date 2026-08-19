"""Wątek roboczy asystenta dla GUI — cała praca poza wątkiem interfejsu (Faza 10).

Wymaganie Fazy 10 brzmi: **okno nie może zamarzać**. Realizuje to podział:

* wątek interfejsu (tkinter) rysuje i zbiera kliknięcia — nic więcej,
* jeden wątek roboczy z własną pętlą ``asyncio`` prowadzi rozmowę: mikrofon,
  Whisper, model językowy, narzędzia i synteza mowy,
* komunikacja idzie w jedną stronę **komendami** (kolejka do wątku roboczego), a
  w drugą **zdarzeniami** (kolejka do interfejsu).

Dlaczego JEDEN wątek roboczy, a nie kilka: pamięć rozmowy to połączenie SQLite,
router narzędzi trzyma stan tury (budżet wywołań, ślad danych niezaufanych), a
Whisper i Piper i tak liczą po kolei. Jeden wątek znaczy zero blokad wokół tych
obiektów i zero pytań w rodzaju „czy dwie tury mogą się przepleść". Rzeczy, które
naprawdę czekają na sprzęt (odtwarzanie dźwięku, dokańczanie wypowiedzi),
i tak mają własne wątki wewnątrz warstwy audio.

Ten moduł **nie importuje tkintera ani CustomTkintera** i nie wie, co jest po
drugiej stronie kolejki — dzięki temu cały przepływ (tura, potwierdzenia,
przełączanie modelu, przeładowanie mowy) daje się przetestować bez ekranu.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import time
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from brain.memory import ConversationMemory
from brain.personality import (
    build_context_message,
    build_system_prompt,
    greeting,
    is_auto_language,
    normalize_language,
    resolve_reply_language,
)
from brain.remember import MemoryCurator, detect_memory_intent
from brain.request_kind import classify as classify_request
from brain.request_kind import prompt_hint as request_prompt_hint
from brain.request_kind import user_notice as request_notice
from brain.tool_router import tool_system_rules
from brain.turn import run_turn
from config import (
    Settings,
    configured_reply_language,
    describe_offline_mode,
    get_user_settings,
    reload_user_settings,
)
from gui.state import (
    ChatRole,
    ListeningState,
    ServiceState,
    ServiceStatus,
    StatusSnapshot,
    services_from_report,
)
from i18n import t
from security.confirm import CallbackBroker, ConfirmationOutcome, ConfirmationRequest

if TYPE_CHECKING:  # importy tylko dla typów — GUI działa też bez warstwy audio
    from audio.pipeline import PipelineMessage, SpeechToTextPipeline
    from audio.tts import SpeechOutput
    from brain.llm import OllamaClient
    from brain.tool_router import ToolOutcome, ToolRouter
    from tools.base import ToolContext

logger = logging.getLogger(__name__)

# Nasłuch mikrofonu jest dzielony na krótkie odcinki, żeby wątek roboczy mógł
# między nimi zajrzeć do kolejki komend. Rozpoczętej wypowiedzi to nie przerywa —
# potok nigdy nie ucina mowy w połowie, limit dotyczy wyłącznie ciszy.
LISTEN_SLICE_S: float = 2.0

# Jak długo czekamy na komendę, gdy nic się nie dzieje. Wartość widać jako
# opóźnienie reakcji na klik, więc jest mała; wątek i tak śpi.
IDLE_POLL_S: float = 0.2


# --------------------------------------------------------------------------- #
# Zdarzenia do interfejsu
# --------------------------------------------------------------------------- #


class EventKind(StrEnum):
    """Co wątek roboczy ma do powiedzenia interfejsowi."""

    READY = "ready"
    STATUS = "status"
    MESSAGE = "message"
    REPLY_START = "reply_start"
    REPLY_CHUNK = "reply_chunk"
    REPLY_END = "reply_end"
    THINKING = "thinking"
    TOOL = "tool"
    CONFIRM = "confirm"
    CONFIRM_CLOSED = "confirm_closed"
    MODELS = "models"
    VOICES = "voices"
    SETTINGS = "settings"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Jedno zdarzenie. Niezmienne, bo przechodzi między wątkami."""

    kind: EventKind
    text: str = ""
    detail: str = ""
    role: ChatRole | None = None
    snapshot: StatusSnapshot | None = None
    data: Any = None


Publisher = Callable[[RuntimeEvent], None]


class CommandKind(StrEnum):
    """Czego interfejs chce od wątku roboczego."""

    SEND = "send"
    RELOAD_SETTINGS = "reload_settings"
    SET_MODEL = "set_model"
    RELOAD_SPEECH = "reload_speech"
    SAY_SAMPLE = "say_sample"
    NEW_CONVERSATION = "new_conversation"
    REFRESH = "refresh"
    LIST_MODELS = "list_models"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class Command:
    kind: CommandKind
    text: str = ""
    data: Any = None


# --------------------------------------------------------------------------- #
# Mowa: to samo co w terminalu, tylko bez print()
# --------------------------------------------------------------------------- #


class GuiSpeaker:
    """Wyjście mowy dla GUI (Faza 4 widziana z okna).

    To nie jest kopia ``main.VoiceOutput`` przez niedbałość: tamta klasa opisuje
    stan **komunikatami w terminalu** i obsługuje komendy tekstowe (``/glos``).
    Tutaj kanałem jest kolejka zdarzeń, a interfejsem — przełącznik w oknie.
    Wspólna jest cała warstwa niżej (``audio/tts.py``, ``audio/output.py``), więc
    nie dubluje się nic, co dotyczy samej syntezy.
    """

    def __init__(self, settings: Settings, publish: Publisher) -> None:
        self._settings = settings
        self._publish = publish
        self._speech: SpeechOutput | None = None
        self._active = False
        self._muted = False
        self._reason = ""

    @property
    def enabled(self) -> bool:
        return self._active and not self._muted

    @property
    def muted(self) -> bool:
        return self._muted

    def describe(self) -> str:
        if self._muted:
            return t("runtime.speech_muted")
        if self._active and self._speech is not None:
            return self._speech.describe()
        return self._reason or t("runtime.speech_off")

    def state(self) -> ServiceState:
        if self._active and not self._muted:
            return ServiceState.OK
        if self._muted:
            return ServiceState.OFF
        return ServiceState.OFF if not self._reason else ServiceState.ERROR

    def _on_error(self, error: Any) -> None:
        """Błąd z wątku syntezy: mówimy o tym raz i schodzimy do samego tekstu."""
        self._active = False
        self._reason = getattr(error, "message", str(error))
        self._publish(
            RuntimeEvent(
                kind=EventKind.ERROR,
                text=getattr(error, "user_message", str(error)),
                detail=t("runtime.speech_error_hint"),
            )
        )

    def load(self) -> bool:
        """Zbuduj silnik mowy. ``False`` = zostajemy przy tekście (to nie awaria)."""
        self._muted = False
        if self._active and self._speech is not None:
            return True
        try:
            from audio.output import AudioOutput, AudioOutputError
            from audio.tts import SpeechOutput as SpeechOutputImpl
            from audio.tts import TTSError, create_tts_provider
        except ImportError as exc:
            self._reason = t("runtime.mic_unavailable", reason=exc)
            return False

        provider = create_tts_provider(self._settings)
        if not provider.is_speaking_enabled:
            self._reason = provider.describe()
            return False
        try:
            provider.load()
            sink = AudioOutput(self._settings)
        except (TTSError, AudioOutputError) as exc:
            self._reason = getattr(exc, "message", str(exc))
            with contextlib.suppress(Exception):
                provider.close()
            return False
        except Exception as exc:  # pragma: no cover - zależne od sprzętu
            logger.exception("Nieoczekiwany błąd uruchamiania mowy w GUI")
            self._reason = str(exc)
            return False

        self._speech = SpeechOutputImpl(
            provider, sink, settings=self._settings, on_error=self._on_error
        )
        self._active = True
        self._reason = ""
        return True

    def reload(self, settings: Settings | None = None) -> bool:
        """Zbuduj silnik od nowa — po zmianie głosu albo ścieżek RVC."""
        if settings is not None:
            self._settings = settings
        self.close()
        return self.load()

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self._muted:
            self.cancel()

    # --- interfejs oczekiwany przez brain.turn.TurnSpeaker ------------------ #

    def begin(self, language: str | None = None) -> None:
        if self.enabled and self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.begin(language)

    def feed(self, text: str) -> None:
        if self.enabled and self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.feed(text)

    def end(self) -> None:
        if self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.end(wait=True)

    def cancel(self) -> None:
        if self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.cancel()

    def say(self, text: str, language: str | None = None) -> None:
        """Wypowiedz jedno zdanie (powitanie, komunikat, próbka głosu)."""
        if not self.enabled or self._speech is None or not text.strip():
            return
        with contextlib.suppress(Exception):
            self._speech.speak(text)

    def close(self) -> None:
        self._active = False
        if self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.close()
            self._speech = None


# --------------------------------------------------------------------------- #
# Mikrofon
# --------------------------------------------------------------------------- #


class GuiListener:
    """Wejście głosowe dla GUI: potok mowy z Fazy 2/3 opakowany w zdarzenia.

    Nasłuch jest **cięty na odcinki** (:data:`LISTEN_SLICE_S`), więc wątek
    roboczy nigdy nie utyka na kilkadziesiąt sekund w jednym wywołaniu i reaguje
    na „przestań słuchać" bez ubijania czegokolwiek. Okno rozmowy słowa
    aktywującego liczy się czasem monotonicznym, więc cięcie mu nie szkodzi.
    """

    def __init__(
        self,
        settings: Settings,
        publish: Publisher,
        *,
        on_state: Callable[[ListeningState], None] | None = None,
    ) -> None:
        self._settings = settings
        self._publish = publish
        self._on_state = on_state
        self._pipeline: SpeechToTextPipeline | None = None
        self._active = False
        self._reason = ""

    @property
    def active(self) -> bool:
        return self._active

    def describe(self) -> str:
        if self._active and self._pipeline is not None:
            return self._pipeline.describe()
        return self._reason or t("runtime.mic_dummy")

    def wake_phrase(self) -> str:
        if self._pipeline is not None:
            phrase = self._pipeline.wake_phrase
            if phrase:
                return phrase
        return get_user_settings().effective_wake_word

    def wake_detail(self) -> str:
        if not self._settings.wake_enabled or self._settings.wake_engine == "none":
            return t("runtime.wake_disabled", phrase=self.wake_phrase())
        if self._pipeline is None:
            return t("runtime.wake_pending", phrase=self.wake_phrase())
        if self._pipeline.wake_engine is None:
            return t("runtime.wake_no_gate", phrase=self.wake_phrase())
        return t(
            "runtime.wake_active",
            phrase=self._pipeline.wake_phrase,
            engine=self._pipeline.wake_name,
        )

    def _state(self, state: ListeningState) -> None:
        if self._on_state is not None:
            self._on_state(state)

    def _on_event(self, message: PipelineMessage) -> None:
        """Zdarzenie potoku → stan wskaźnika (albo komunikat o błędzie)."""
        from audio.pipeline import PipelineEvent

        mapping = {
            PipelineEvent.WAITING_FOR_WAKE: ListeningState.WAITING_WAKE,
            PipelineEvent.WAKE_DETECTED: ListeningState.LISTENING,
            PipelineEvent.LISTENING: ListeningState.LISTENING,
            PipelineEvent.SPEECH_START: ListeningState.LISTENING,
            PipelineEvent.SPEECH_END: ListeningState.TRANSCRIBING,
            PipelineEvent.TRANSCRIBING: ListeningState.TRANSCRIBING,
        }
        state = mapping.get(message.event)
        if state is not None:
            self._state(state)
            return
        if message.event is PipelineEvent.ERROR:
            self._publish(
                RuntimeEvent(kind=EventKind.ERROR, text=message.text, detail=message.detail)
            )
            return
        if message.event is PipelineEvent.IGNORED:
            # Odrzucona wypowiedź MUSI być widoczna. Wcześniej lądowała tylko w logu
            # i z perspektywy użytkownika asystent po prostu nie reagował na mowę —
            # „nie rozpoznaje mowy tak jak powinien”. Teraz mówimy, co się stało.
            logger.info("Pominięto mowę bez frazy: %s", message.detail)
            self._publish(
                RuntimeEvent(
                    kind=EventKind.MESSAGE,
                    role=ChatRole.SYSTEM,
                    text=t("runtime.wake_ignored", phrase=self.wake_phrase()),
                )
            )

    def open_window(self) -> None:
        """Otwórz okno rozmowy bez wypowiadania frazy wybudzającej.

        Kliknięcie „Słuchaj” JEST jawnym zawołaniem — użytkownik właśnie powiedział
        interfejsem, że mówi do asystenta. Wymaganie po tym jeszcze frazy to pytanie
        o to samo dwa razy, a przy detektorze na modelu ``tiny`` bywa, że fraza nie
        zostaje rozpoznana i wypowiedź przepada bez śladu.
        """
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.wake_up()

    def start(self) -> bool:
        """Uruchom mikrofon i modele. ``False`` = zostaje pisanie z klawiatury."""
        if self._active:
            return True
        try:
            from audio.pipeline import SpeechPipelineError, SpeechToTextPipeline
        except ImportError as exc:
            self._reason = str(exc)
            self._publish(
                RuntimeEvent(
                    kind=EventKind.ERROR,
                    text=t("runtime.mic_unavailable", reason=exc),
                    detail=t("runtime.mic_unavailable_hint"),
                )
            )
            return False

        if self._pipeline is None:
            self._pipeline = SpeechToTextPipeline(self._settings, on_event=self._on_event)
        try:
            self._pipeline.start()
        except SpeechPipelineError as exc:
            self._reason = exc.message
            self._publish(RuntimeEvent(kind=EventKind.ERROR, text=exc.message, detail=exc.hint))
            self._pipeline = None
            return False
        except Exception as exc:  # pragma: no cover - zależne od sprzętu
            logger.exception("Nie udało się uruchomić trybu głosowego w GUI")
            self._reason = str(exc)
            self._publish(
                RuntimeEvent(
                    kind=EventKind.ERROR,
                    text=t("runtime.mic_failed", error=exc),
                    detail=t("runtime.mic_failed_hint"),
                )
            )
            self._pipeline = None
            return False

        self._active = True
        self._reason = ""
        return True

    def listen_slice(self) -> str | None:
        """Nasłuchuj krótką chwilę. ``None`` = cisza (albo błąd) — wracamy po komendy."""
        if not self._active or self._pipeline is None:
            return None
        limit = self._settings.vad_listen_timeout_s
        slice_s = LISTEN_SLICE_S if limit <= 0 else min(limit, LISTEN_SLICE_S)
        try:
            transcript = self._pipeline.listen_once(slice_s)
        except Exception as exc:
            logger.exception("Błąd nasłuchu w GUI")
            self._reason = str(exc)
            self._publish(
                RuntimeEvent(
                    kind=EventKind.ERROR,
                    text=t("runtime.mic_error", error=exc),
                    detail=t("runtime.mic_error_hint"),
                )
            )
            self.stop()
            return None
        if transcript is None:
            return None
        return transcript.text

    def stop(self) -> None:
        self._active = False
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.stop()
        self._state(ListeningState.OFF)

    def close(self) -> None:
        self._active = False
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.close()
            self._pipeline = None


# --------------------------------------------------------------------------- #
# Widok tury: strumień modelu → zdarzenia
# --------------------------------------------------------------------------- #


class _RuntimeView:
    """:class:`brain.turn.TurnView` publikujący zdarzenia dla okna."""

    def __init__(self, runtime: AssistantRuntime) -> None:
        self._runtime = runtime

    def on_thinking(self) -> None:
        self._runtime._set_listening(ListeningState.THINKING)
        self._runtime._publish(RuntimeEvent(kind=EventKind.THINKING, text=t("runtime.thinking")))

    def on_reply_start(self) -> None:
        self._runtime._publish(RuntimeEvent(kind=EventKind.REPLY_START))
        self._runtime._set_listening(
            ListeningState.SPEAKING if self._runtime.speaking else ListeningState.THINKING
        )

    def on_chunk(self, text: str) -> None:
        self._runtime._publish(RuntimeEvent(kind=EventKind.REPLY_CHUNK, text=text))

    def on_reply_end(self, text: str) -> None:
        self._runtime._publish(RuntimeEvent(kind=EventKind.REPLY_END, text=text))

    def on_tool(self, outcome: ToolOutcome) -> None:
        self._runtime._publish(
            RuntimeEvent(
                kind=EventKind.TOOL,
                text=outcome.line_for_user(),
                detail=outcome.call.name,
                data=outcome,
            )
        )

    def on_notice(self, text: str) -> None:
        self._runtime._publish(
            RuntimeEvent(kind=EventKind.MESSAGE, text=text, role=ChatRole.SYSTEM)
        )


# --------------------------------------------------------------------------- #
# Wątek roboczy
# --------------------------------------------------------------------------- #


@dataclass
class _PendingConfirm:
    """Żądanie zgody czekające na kliknięcie w oknie."""

    request: ConfirmationRequest
    future: asyncio.Future[ConfirmationOutcome] = field(repr=False)


class AssistantRuntime:
    """Asystent w wątku roboczym. Metody publiczne wolno wołać z wątku interfejsu.

    Każda metoda publiczna kończy się natychmiast: wkłada komendę do kolejki albo
    ustawia flagę. Nic w tej klasie nie czeka na model, na dysk ani na mikrofon w
    wątku wołającego — to jest cała odpowiedź na wymaganie „GUI nie może blokować
    głównego wątku".
    """

    def __init__(
        self,
        settings: Settings,
        *,
        publish: Publisher,
        report: object | None = None,
        speech_enabled: bool = True,
        memory_source: str = "gui",
    ) -> None:
        self._settings = settings
        self._publish_raw = publish
        self._report = report
        self._speech_enabled = speech_enabled
        self._memory_source = memory_source

        self._commands: queue.Queue[Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()
        self._listen_wanted = threading.Event()
        self._lock = threading.RLock()

        self._memory: ConversationMemory | None = None
        self._client: OllamaClient | None = None
        self._curator: MemoryCurator | None = None
        self._router: ToolRouter | None = None
        self._tool_ctx: ToolContext | None = None
        self._speaker: GuiSpeaker | None = None
        self._listener: GuiListener | None = None
        # Pluginy (Faza 11): jeden menedżer na proces i znacznik ostatniego
        # sprawdzenia, żeby pytać je co ``REMINDERS_POLL_S``, a nie co obrót pętli.
        self._plugins: Any | None = None
        self._last_plugin_poll: float = 0.0
        self._view = _RuntimeView(self)
        self._task: asyncio.Task[Any] | None = None
        self._pending: dict[str, _PendingConfirm] = {}
        self._status = StatusSnapshot()

    # --- podstawowe informacje -------------------------------------------- #

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def status(self) -> StatusSnapshot:
        return self._status

    @property
    def is_listening(self) -> bool:
        """Czy użytkownik chce teraz nasłuchu (flaga ustawiana natychmiast)."""
        return self._listen_wanted.is_set()

    @property
    def speaking(self) -> bool:
        return self._speaker is not None and self._speaker.enabled

    @property
    def muted(self) -> bool:
        return self._speaker is None or self._speaker.muted

    # --- API dla wątku interfejsu ----------------------------------------- #

    def start(self) -> None:
        """Uruchom wątek roboczy. Wraca od razu — reszta dzieje się w tle."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="miku-runtime", daemon=True)
        self._thread.start()

    def send(self, text: str) -> None:
        if text.strip():
            self._commands.put(Command(kind=CommandKind.SEND, text=text.strip()))

    def start_listening(self) -> None:
        self._listen_wanted.set()

    def stop_listening(self) -> None:
        self._listen_wanted.clear()

    def toggle_listening(self) -> bool:
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
        return self.is_listening

    def cancel(self) -> None:
        """Przerwij trwające generowanie (i mowę). Wołane z wątku interfejsu."""
        with self._lock:
            loop, task = self._loop, self._task
        if self._speaker is not None:
            self._speaker.cancel()
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)

    def set_muted(self, muted: bool) -> None:
        """Wycisz albo odcisz mowę. Działa natychmiast, bez kolejki."""
        if self._speaker is not None:
            self._speaker.set_muted(muted)
        self._commands.put(Command(kind=CommandKind.REFRESH))

    def reload_settings(self) -> None:
        self._commands.put(Command(kind=CommandKind.RELOAD_SETTINGS))

    def set_model(self, name: str) -> None:
        if name.strip():
            self._commands.put(Command(kind=CommandKind.SET_MODEL, text=name.strip()))

    def reload_speech(self) -> None:
        self._commands.put(Command(kind=CommandKind.RELOAD_SPEECH))

    def say_sample(self, text: str = "") -> None:
        self._commands.put(Command(kind=CommandKind.SAY_SAMPLE, text=text))

    def new_conversation(self) -> None:
        self._commands.put(Command(kind=CommandKind.NEW_CONVERSATION))

    def refresh(self) -> None:
        self._commands.put(Command(kind=CommandKind.REFRESH))

    def list_models(self) -> None:
        self._commands.put(Command(kind=CommandKind.LIST_MODELS))

    def answer_confirmation(self, request_id: str, approved: bool, reason: str = "") -> None:
        """Odpowiedz na pytanie o zgodę (kliknięcie w oknie).

        Odpowiedź jest przekazywana do pętli wątku roboczego, a nie ustawiana tu
        wprost: ``asyncio.Future`` wolno ruszać tylko z jego pętli.
        """
        with self._lock:
            loop = self._loop
        if loop is None:
            return
        outcome = (
            ConfirmationOutcome.approve(channel="gui")
            if approved
            else ConfirmationOutcome.deny(channel="gui", reason=reason or t("runtime.user_denied"))
        )
        loop.call_soon_threadsafe(self._resolve_confirm, request_id, outcome)

    def close(self, *, timeout: float = 8.0) -> None:
        """Zakończ pracę: przerwij turę, zamknij mikrofon, mowę, bazę i model."""
        self._stopping.set()
        self._listen_wanted.clear()
        self.cancel()
        self._commands.put(Command(kind=CommandKind.SHUTDOWN))
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # --- publikowanie ------------------------------------------------------ #

    def _publish(self, event: RuntimeEvent) -> None:
        try:
            self._publish_raw(event)
        except Exception:  # pragma: no cover - awaria kolejki interfejsu
            logger.debug("Nie udało się opublikować zdarzenia %s", event.kind, exc_info=True)

    def _say(self, text: str, role: ChatRole = ChatRole.SYSTEM, detail: str = "") -> None:
        self._publish(RuntimeEvent(kind=EventKind.MESSAGE, text=text, role=role, detail=detail))

    def _set_listening(self, state: ListeningState) -> None:
        if self._status.listening is state:
            return
        self._status = self._status.with_listening(state)
        self._publish(RuntimeEvent(kind=EventKind.STATUS, snapshot=self._status))

    def _publish_status(self) -> None:
        self._status = self._build_status()
        self._publish(RuntimeEvent(kind=EventKind.STATUS, snapshot=self._status))

    # --- budowa stanu ------------------------------------------------------ #

    def _language(self) -> str:
        """Język ODPOWIEDZI — patrz :func:`config.configured_reply_language`.

        Nie jest to język, w którym użytkownik MÓWI: ten opisuje
        ``speech_language`` i może być listą („pl,en").
        """
        return configured_reply_language(self._settings)

    def _build_status(self) -> StatusSnapshot:
        user = get_user_settings()
        preferred = self._language()
        listening = self._status.listening
        microphone_off = self._listener is None or not self._listener.active
        if microphone_off and listening.is_microphone_on:
            # Mikrofon zniknął w trakcie — wskaźnik nie może dalej pokazywać
            # „słucham”, bo nikt już nie słucha.
            listening = ListeningState.IDLE
        snapshot = StatusSnapshot(
            assistant_name=user.assistant_name,
            model=self._settings.ollama_model,
            host=self._settings.ollama_host,
            language=("auto" if is_auto_language(preferred) else normalize_language(preferred)),
            language_forced=not is_auto_language(preferred),
            listening=listening,
            wake_phrase=(
                self._listener.wake_phrase() if self._listener else user.effective_wake_word
            ),
            busy=self._task is not None and not self._task.done(),
        )
        if self._report is not None:
            snapshot = snapshot.with_services(services_from_report(self._report))

        items: list[ServiceStatus] = []
        if self._listener is not None:
            items.append(
                ServiceStatus(
                    key="mic",
                    label="Mikrofon",
                    state=ServiceState.OK if self._listener.active else ServiceState.OFF,
                    detail=self._listener.describe(),
                )
            )
            items.append(
                ServiceStatus(
                    key="wake",
                    label="Słowo aktywujące",
                    state=ServiceState.OK if self._listener.active else ServiceState.OFF,
                    detail=self._listener.wake_detail(),
                )
            )
        if self._speaker is not None:
            items.append(
                ServiceStatus(
                    key="speech",
                    label="Mowa",
                    state=self._speaker.state(),
                    detail=self._speaker.describe(),
                )
            )
        if self._memory is not None:
            items.append(
                ServiceStatus(
                    key="memory",
                    label="Pamięć",
                    state=ServiceState.OK if self._memory.persistent else ServiceState.OFF,
                    detail=self._memory.status_text,
                )
            )
        items.append(
            ServiceStatus(
                key="tools",
                label="Narzędzia",
                state=(
                    ServiceState.OK
                    if self._router is not None and self._router.enabled
                    else ServiceState.OFF
                ),
                detail=(
                    self._router.describe() if self._router is not None else t("runtime.tools_off")
                ),
            )
        )
        return snapshot.with_services(items)

    # --- pętla wątku roboczego -------------------------------------------- #

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self._loop = loop
        try:
            self._setup()
            self._loop_commands()
        except Exception as exc:  # pragma: no cover - awaria startu
            logger.exception("Wątek roboczy asystenta zakończył się błędem")
            self._publish(
                RuntimeEvent(
                    kind=EventKind.ERROR,
                    text=t("runtime.stopped_working", error=exc),
                    detail=t("runtime.mic_failed_hint"),
                )
            )
        finally:
            self._teardown(loop)
            with self._lock:
                self._loop = None
            asyncio.set_event_loop(None)
            self._publish(RuntimeEvent(kind=EventKind.CLOSED))

    def _setup(self) -> None:
        """Zbuduj wszystko, czego potrzebuje rozmowa. Każdy brak jest komunikatem."""
        from brain.llm import OllamaClient

        self._memory = ConversationMemory(self._settings, source=self._memory_source)
        if self._settings.memory_enabled and not self._memory.persistent:
            self._say(t("runtime.memory_unavailable", reason=self._memory.error))
        self._plugins = self._build_plugins()
        self._client = OllamaClient(self._settings)
        self._curator = MemoryCurator(self._memory, self._settings)
        self._router, self._tool_ctx = self._build_tools()

        self._speaker = GuiSpeaker(self._settings, self._publish)
        if self._speech_enabled and not self._speaker.load():
            logger.info("Mowa nie wystartowała: %s", self._speaker.describe())

        self._listener = GuiListener(self._settings, self._publish, on_state=self._set_listening)

        user = get_user_settings()
        self._publish_status()
        self._publish(
            RuntimeEvent(
                kind=EventKind.READY,
                text=greeting(user, language=normalize_language(self._language())),
                detail=describe_offline_mode(self._settings),
                data=user,
            )
        )
        self._refresh_voices()

    def _build_tools(self) -> tuple[ToolRouter | None, ToolContext | None]:
        """Router narzędzi z kanałem potwierdzeń wprowadzonym do okna.

        Potwierdzenia HIGH/CRITICAL idą przez :class:`CallbackBroker` — kanał
        przygotowany na to w Fazie 7. Brak okna (albo zamknięcie go w trakcie
        pytania) kończy się ODMOWĄ, nigdy zgodą.
        """
        if not self._settings.tools_enabled:
            return None, None
        try:
            from brain.tool_router import build_router
            from tools.base import ToolContext
        except Exception as exc:  # pragma: no cover - zależne od instalacji
            logger.warning("Warstwa narzędzi jest niedostępna: %s", exc)
            self._say(t("runtime.tools_unavailable", reason=exc))
            return None, None
        try:
            broker = CallbackBroker(
                self._ask_confirmation,
                channel="gui",
                timeout_s=self._settings.security_confirm_timeout_s,
            )
            router = build_router(
                self._settings,
                database=self._memory.database if self._memory is not None else None,
                conversation_id=self._memory.conversation_id if self._memory is not None else None,
                memory=self._memory,
                broker=broker,
                plugins=self._plugins,
            )
        except Exception as exc:
            logger.exception("Nie udało się zbudować routera narzędzi")
            self._say(t("runtime.tools_failed", error=exc))
            return None, None
        return router, ToolContext(settings=self._settings, dry_run=self._settings.security_dry_run)

    def _build_plugins(self) -> Any:
        """Menedżer pluginów dla tego uruchomienia (Faza 11). Nigdy nie rzuca."""
        try:
            from plugins.manager import PluginContext, PluginManager

            manager = PluginManager(
                self._settings,
                context=PluginContext(
                    settings=self._settings,
                    database=self._memory.database if self._memory is not None else None,
                    memory=self._memory,
                ),
            )
            manager.load()
            logger.info("Pluginy: %s", manager.describe())
            return manager
        except Exception as exc:  # pragma: no cover - awaria warstwy pluginów
            logger.warning("Warstwa pluginów niedostępna: %s", exc)
            return None

    def _poll_plugins(self) -> None:
        """Zapytaj pluginy, czy coś ma się odezwać (przypomnienia, budziki).

        Sprawdzanie idzie w pętli komend, a nie w osobnym wątku: pętla i tak
        budzi się co ``IDLE_POLL_S``, a jeden wątek mniej to jedno miejsce mniej,
        w którym coś może zawisnąć przy zamykaniu okna.
        """
        if self._plugins is None:
            return
        interval = max(1.0, float(self._settings.reminders_poll_s))
        moment = time.monotonic()
        if moment - self._last_plugin_poll < interval:
            return
        self._last_plugin_poll = moment

        for notice in self._plugins.poll():
            self._publish(
                RuntimeEvent(kind=EventKind.MESSAGE, text=notice.text, role=ChatRole.SYSTEM)
            )
            if notice.speak and self._speaker is not None:
                self._speaker.say(notice.text)

    def _loop_commands(self) -> None:
        """Główna pętla: komendy z interfejsu i (gdy włączony) nasłuch mikrofonu."""
        while not self._stopping.is_set():
            if self._drain_commands():
                break
            if self._stopping.is_set():
                break
            self._poll_plugins()

            if self._listen_wanted.is_set():
                if not self._ensure_listening():
                    continue
                text = self._listener.listen_slice() if self._listener else None
                if text and text.strip():
                    self._handle_text(text.strip(), spoken=True)
                continue

            self._release_listening()
            try:
                command = self._commands.get(timeout=IDLE_POLL_S)
            except queue.Empty:
                continue
            if self._handle_command(command):
                break

    def _drain_commands(self) -> bool:
        """Obsłuż komendy, które już czekają. ``True`` = kończymy pracę."""
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return False
            if self._handle_command(command):
                return True

    def _ensure_listening(self) -> bool:
        if self._listener is None:
            self._listen_wanted.clear()
            return False
        if self._listener.active:
            return True
        if self._listener.start():
            # Nasłuch włączony ręcznie = rozmowa zaczyna się teraz, bez frazy.
            self._listener.open_window()
            self._publish_status()
            return True
        # Mikrofonu nie ma — nie próbujemy w pętli co dwie sekundy.
        self._listen_wanted.clear()
        self._publish_status()
        return False

    def _release_listening(self) -> None:
        if self._listener is not None and self._listener.active:
            self._listener.stop()
            self._publish_status()

    def _handle_command(self, command: Command) -> bool:
        """Wykonaj jedną komendę. ``True`` = zamykamy wątek."""
        if command.kind is CommandKind.SHUTDOWN:
            return True
        try:
            if command.kind is CommandKind.SEND:
                self._handle_text(command.text, spoken=False)
            elif command.kind is CommandKind.RELOAD_SETTINGS:
                self._handle_reload_settings()
            elif command.kind is CommandKind.SET_MODEL:
                self._handle_set_model(command.text)
            elif command.kind is CommandKind.RELOAD_SPEECH:
                self._handle_reload_speech()
            elif command.kind is CommandKind.SAY_SAMPLE:
                self._handle_say_sample(command.text)
            elif command.kind is CommandKind.NEW_CONVERSATION:
                self._handle_new_conversation()
            elif command.kind is CommandKind.LIST_MODELS:
                self._handle_list_models()
            elif command.kind is CommandKind.REFRESH:
                self._publish_status()
        except Exception as exc:  # żadna komenda nie może zabić wątku roboczego
            logger.exception("Komenda %s nie powiodła się", command.kind)
            self._publish(
                RuntimeEvent(kind=EventKind.ERROR, text=t("runtime.command_failed", error=exc))
            )
        return False

    # --- tura rozmowy ------------------------------------------------------ #

    def _handle_text(self, text: str, *, spoken: bool) -> None:
        """Jedna tura: dokładnie to samo, co robi pętla terminala."""
        memory, client = self._memory, self._client
        if memory is None or client is None:  # pragma: no cover - broniony start
            return

        self._publish(
            RuntimeEvent(
                kind=EventKind.MESSAGE,
                text=text,
                role=ChatRole.USER,
                detail=t("gui.detail.microphone") if spoken else "",
            )
        )

        preferred = self._language()
        language = resolve_reply_language(preferred, text)
        lock_language = not is_auto_language(preferred)
        memory.add_user(text, language=language)

        # „Zapamiętaj, że…" / „Zapomnij, że…" (Faza 6) — rozpoznanie tekstowe,
        # więc działa też przy niedostępnym modelu.
        intent = detect_memory_intent(text)
        if intent is not None and self._curator is not None:
            self._set_listening(ListeningState.THINKING)
            try:
                outcome = self._run_async(self._curator.handle(intent, client, language=language))
            except asyncio.CancelledError:
                self._say(t("runtime.memory_interrupted"))
                self._after_turn()
                return
            except Exception as exc:
                logger.exception("Obsługa polecenia pamięciowego nie powiodła się")
                self._publish(
                    RuntimeEvent(
                        kind=EventKind.ERROR,
                        text=t("runtime.memory_failed", error=exc),
                    )
                )
                self._after_turn()
                return
            self._say(outcome.message, role=ChatRole.ASSISTANT)
            memory.add_assistant(outcome.message, language=language)
            if self._speaker is not None:
                self._speaker.say(outcome.message, language)
            self._after_turn()
            return

        if memory.needs_compaction:
            self._say(t("runtime.compacting"))
            with contextlib.suppress(Exception):
                self._run_async(memory.compact(client, language=language))

        if self._router is not None:
            self._router.reset_turn(conversation_id=memory.conversation_id)

        # LOCAL czy WEB (Faza 9). Ocena jest czysto tekstowa — nie kosztuje tury.
        assessment = classify_request(text)
        web_ready = self._web_tools_ready()
        notice = request_notice(assessment, language=language, web_available=web_ready)
        if notice:
            self._say(notice)
            if self._speaker is not None:
                self._speaker.say(notice, language)

        # Prompt systemowy STAŁY, reszta osobną wiadomością — patrz brain/llm.py.
        system_prompt = build_system_prompt(
            get_user_settings(),
            language=language,
            lock_language=lock_language,
            tool_rules=(
                tool_system_rules(language)
                if self._router is not None and self._router.enabled
                else ""
            ),
        )
        turn_context = build_context_message(
            language=language,
            extra_context=memory.context_block(language, query=text),
            request_hint=request_prompt_hint(
                assessment, language=language, web_available=web_ready
            ),
        )

        self._set_listening(ListeningState.THINKING)
        try:
            answer = self._run_async(
                run_turn(
                    client,
                    memory,
                    self._router,
                    self._tool_ctx.localized(language) if self._tool_ctx is not None else None,
                    system_prompt,
                    view=self._view,
                    speaker=self._speaker,
                    language=language,
                    context=turn_context,
                ),
                track=True,
            )
        except asyncio.CancelledError:
            self._publish(RuntimeEvent(kind=EventKind.REPLY_END, text="", detail="cancelled"))
            self._say(t("runtime.generation_interrupted"))
            self._after_turn()
            return
        except Exception as exc:
            self._publish(RuntimeEvent(kind=EventKind.REPLY_END, text="", detail="error"))
            message = getattr(exc, "user_message", None) or t("runtime.unexpected_error", error=exc)
            if not isinstance(getattr(exc, "user_message", None), str):
                logger.exception("Nieoczekiwany błąd podczas rozmowy w GUI")
            self._publish(RuntimeEvent(kind=EventKind.ERROR, text=str(message)))
            self._after_turn()
            return

        if answer.strip():
            memory.add_assistant(answer, language=language)
        else:
            self._say(t("runtime.empty_reply"))
        self._after_turn()

    def _after_turn(self) -> None:
        with self._lock:
            self._task = None
        self._set_listening(
            ListeningState.WAITING_WAKE
            if self._listener is not None and self._listener.active
            else ListeningState.IDLE
        )
        self._publish_status()

    def _web_tools_ready(self) -> bool:
        """Czy model ma czym sięgnąć po świeże dane (Faza 9)."""
        if self._router is None:
            return False
        return any(
            "." in name and name.split(".")[0] in ("web", "weather", "news", "youtube")
            for name in self._router.visible_names()
        )

    def _run_async(self, coroutine: Any, *, track: bool = False) -> Any:
        """Uruchom korutynę w pętli tego wątku, z możliwością przerwania z GUI."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(coroutine)
        if track:
            with self._lock:
                self._task = task
        try:
            return loop.run_until_complete(task)
        finally:
            if track:
                with self._lock:
                    self._task = None

    # --- potwierdzenia ----------------------------------------------------- #

    async def _ask_confirmation(self, request: ConfirmationRequest) -> ConfirmationOutcome:
        """Zapytaj okno o zgodę i poczekaj na kliknięcie.

        Czekamy w pętli zdarzeń wątku roboczego, więc interfejs jest w tym czasie
        w pełni sprawny — użytkownik może przewijać rozmowę i anulować. Limit czasu
        pilnuje :class:`CallbackBroker`; spóźniona zgoda jest odmową.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ConfirmationOutcome] = loop.create_future()
        self._pending[request.request_id] = _PendingConfirm(request=request, future=future)
        self._publish(
            RuntimeEvent(
                kind=EventKind.CONFIRM,
                text=request.summary,
                detail=request.tool,
                data=request,
            )
        )
        try:
            return await future
        finally:
            self._pending.pop(request.request_id, None)
            self._publish(RuntimeEvent(kind=EventKind.CONFIRM_CLOSED, detail=request.request_id))

    def _resolve_confirm(self, request_id: str, outcome: ConfirmationOutcome) -> None:
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return
        pending.future.set_result(outcome)

    def _deny_pending_confirmations(self, reason: str) -> None:
        """Zamknięcie okna = odmowa dla wszystkiego, co czekało na zgodę."""
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_result(ConfirmationOutcome.deny(channel="gui", reason=reason))

    # --- pozostałe komendy ------------------------------------------------- #

    def _handle_reload_settings(self) -> None:
        user = reload_user_settings()
        self._publish(
            RuntimeEvent(
                kind=EventKind.SETTINGS,
                text=user.assistant_name,
                detail=user.ui_accent_color,
                data=user,
            )
        )
        self._publish_status()

    def _handle_set_model(self, name: str) -> None:
        """Przełącz model językowy na czas tej sesji.

        Świadomie **nie** zapisujemy tego do ``user_settings.json``: model to
        ustawienie infrastruktury (``OLLAMA_MODEL`` w ``.env``), a nie preferencja
        użytkownika. Okno mówi to wprost, żeby nikt nie szukał później, dlaczego
        po restarcie wrócił poprzedni model.
        """
        from brain.llm import OllamaClient

        if name == self._settings.ollama_model:
            return
        previous = self._client
        self._settings = self._settings.model_copy(update={"ollama_model": name})
        self._client = OllamaClient(self._settings)
        if previous is not None:
            with contextlib.suppress(Exception):
                self._run_async(previous.aclose())
        if self._curator is not None and self._memory is not None:
            self._curator = MemoryCurator(self._memory, self._settings)
        self._say(t("gui.session_model", model=name))
        self._publish_status()

    def _handle_reload_speech(self) -> None:
        if self._speaker is None:
            return
        if self._speaker.reload(self._settings):
            self._say(t("runtime.speech_reloaded", detail=self._speaker.describe()))
        else:
            self._say(t("runtime.speech_still_off", detail=self._speaker.describe()))
        self._refresh_voices()
        self._publish_status()

    def _handle_say_sample(self, text: str) -> None:
        if self._speaker is None:
            return
        if not self._speaker.enabled and not self._speaker.load():
            self._say(t("runtime.speech_no_sample", detail=self._speaker.describe()))
            self._publish_status()
            return
        user = get_user_settings()
        language = normalize_language(self._language())
        # Próbka jest MÓWIONA, więc idzie w języku odpowiedzi (a nie interfejsu):
        # ma zabrzmieć tym głosem, którego użytkownik właśnie słucha.
        sample = text.strip() or t("runtime.voice_sample", _lang=language, name=user.assistant_name)
        self._say(sample, role=ChatRole.ASSISTANT, detail=t("gui.detail.voice_sample"))
        self._speaker.say(sample, language)

    def _handle_new_conversation(self) -> None:
        if self._memory is None:
            return
        self._memory.new_session()
        if self._router is not None:
            self._router.reset_turn(conversation_id=self._memory.conversation_id)
        self._publish_status()

    def _handle_list_models(self) -> None:
        """Lista modeli z Ollamy — do listy wyboru w oknie."""
        if self._client is None:
            return
        try:
            models = self._run_async(self._client.list_models())
        except Exception as exc:
            logger.info("Nie udało się pobrać listy modeli: %s", exc)
            self._publish(
                RuntimeEvent(
                    kind=EventKind.MODELS,
                    data=[self._settings.ollama_model],
                    detail=t("runtime.ollama_no_models", error=exc),
                )
            )
            return
        names = [str(item) for item in models] or [self._settings.ollama_model]
        self._publish(RuntimeEvent(kind=EventKind.MODELS, data=names))

    def _refresh_voices(self) -> None:
        """Lista głosów Pipera widocznych na TEJ maszynie (do panelu ustawień)."""
        try:
            from audio.tts import available_tts_engines, iter_piper_voices
        except ImportError as exc:
            logger.info("Nie sprawdzono głosów — brak warstwy audio (%s)", exc)
            self._publish(RuntimeEvent(kind=EventKind.VOICES, data=([], ["piper", "none"])))
            return
        try:
            voices = [voice.name for voice in iter_piper_voices(self._settings)]
        except Exception as exc:  # pragma: no cover - zależne od dysku
            logger.info("Nie udało się wypisać głosów: %s", exc)
            voices = []
        engines = list(available_tts_engines())
        if "none" not in engines:
            engines.append("none")
        self._publish(RuntimeEvent(kind=EventKind.VOICES, data=(voices, engines)))

    # --- zamykanie --------------------------------------------------------- #

    def _teardown(self, loop: asyncio.AbstractEventLoop) -> None:
        self._deny_pending_confirmations(t("runtime.confirm_window_closed"))
        if self._listener is not None:
            self._listener.close()
        if self._speaker is not None:
            self._speaker.close()
        if self._memory is not None:
            with contextlib.suppress(Exception):
                self._memory.close()
        if self._client is not None:
            with contextlib.suppress(Exception):
                loop.run_until_complete(self._client.aclose())
        # Kolejność jak w ``asyncio.run()``: dokończ generatory, zamknij executor,
        # potem pętlę. Bez tego niedokończony strumień odpowiedzi zostawia
        # ostrzeżenie „Task was destroyed but it is pending!".
        for step in (loop.shutdown_asyncgens(), loop.shutdown_default_executor()):
            with contextlib.suppress(Exception):
                loop.run_until_complete(step)
        with contextlib.suppress(Exception):
            loop.close()


__all__ = [
    "IDLE_POLL_S",
    "LISTEN_SLICE_S",
    "AssistantRuntime",
    "Command",
    "CommandKind",
    "EventKind",
    "GuiListener",
    "GuiSpeaker",
    "Publisher",
    "RuntimeEvent",
]
