"""Stan pokazywany w oknie: historia rozmowy i stan usług (Faza 10).

Ten moduł jest **czysty** — bez tkintera, bez wątków, bez sieci. Powód: to, co
użytkownik widzi, ma dać się sprawdzić testem bez ekranu i bez sprzętu. Widgety
(``gui/chat.py``, ``gui/status.py``) tylko rysują te obiekty; cała logika „co
właściwie pokazać" siedzi tutaj.

Obiekty stanu są **niezmienne** (albo zmieniane tylko w jednym wątku). Migawki
stanu podróżują z wątku roboczego do wątku interfejsu, a przekazywanie między
wątkami obiektu, który ktoś w tle jeszcze modyfikuje, to najprostszy sposób na
interfejs pokazujący nieistniejący stan.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Final

from gui.theme import Palette
from i18n import SUPPORTED_UI_LANGUAGES, t

# Ile wiadomości trzyma okno. Rozmowa i tak jest zapisywana w bazie (Faza 5) —
# tutaj chodzi tylko o to, żeby po kilku godzinach gadania okno nie zjadło
# pamięci i nie zaczęło się przewijać w sekundach.
DEFAULT_MAX_MESSAGES: Final[int] = 400


class ChatRole(StrEnum):
    """Kto mówi. Każda rola ma własny kolor bąbelka policzony z akcentu."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"


@dataclass(slots=True)
class ChatMessage:
    """Jeden bąbelek w rozmowie.

    Świadomie **zmienny**: odpowiedź modelu przyrasta fragment po fragmencie i
    widget dopisuje tekst do istniejącego bąbelka, zamiast tworzyć nowy przy
    każdym słowie. Modyfikuje go wyłącznie wątek interfejsu.
    """

    role: ChatRole
    text: str = ""
    created_at: datetime | None = None
    detail: str = ""
    streaming: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def time_label(self) -> str:
        """Godzina w czasie lokalnym maszyny albo ``""``, gdy jej nie znamy."""
        if self.created_at is None:
            return ""
        return self.created_at.astimezone().strftime("%H:%M")

    def bubble_colors(self, palette: Palette) -> tuple[str, str]:
        """(tło, kolor tekstu) dla tej roli — zawsze z palety, nigdy na sztywno."""
        mapping = {
            ChatRole.USER: (palette.user_bubble, palette.user_text),
            ChatRole.ASSISTANT: (palette.assistant_bubble, palette.assistant_text),
            ChatRole.TOOL: (palette.tool_bubble, palette.tool_text),
            ChatRole.SYSTEM: (palette.system_bubble, palette.system_text),
            ChatRole.ERROR: (palette.error_bubble, palette.error_text),
        }
        return mapping[self.role]


class ChatLog:
    """Historia rozmowy widoczna w oknie, razem z obsługą strumienia odpowiedzi.

    Nie jest to pamięć asystenta (tą zajmuje się :mod:`brain.memory`) — tylko to,
    co widać na ekranie. Dwie rzeczy pilnowane tutaj: limit długości oraz
    poprawne domykanie strumienia, żeby przerwana odpowiedź nie została „w
    trakcie pisania" na zawsze.
    """

    def __init__(self, *, max_messages: int = DEFAULT_MAX_MESSAGES) -> None:
        self._messages: list[ChatMessage] = []
        self._max_messages = max(10, int(max_messages))
        self._streaming: ChatMessage | None = None

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[ChatMessage]:
        return iter(tuple(self._messages))

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        return tuple(self._messages)

    @property
    def is_streaming(self) -> bool:
        return self._streaming is not None

    @property
    def streaming_message(self) -> ChatMessage | None:
        return self._streaming

    def add(
        self,
        role: ChatRole,
        text: str,
        *,
        detail: str = "",
        created_at: datetime | None = None,
    ) -> ChatMessage:
        """Dodaj gotową wiadomość i zwróć ją (widget dostaje obiekt do narysowania)."""
        message = ChatMessage(
            role=role,
            text=text,
            detail=detail,
            created_at=created_at or datetime.now().astimezone(),
        )
        self._messages.append(message)
        self._trim()
        return message

    def start_assistant(self, *, created_at: datetime | None = None) -> ChatMessage:
        """Otwórz pusty bąbelek asystenta, do którego dopisze się strumień."""
        self.finish()
        message = ChatMessage(
            role=ChatRole.ASSISTANT,
            text="",
            created_at=created_at or datetime.now().astimezone(),
            streaming=True,
        )
        self._messages.append(message)
        self._streaming = message
        self._trim()
        return message

    def append_chunk(self, text: str) -> ChatMessage:
        """Dopisz fragment odpowiedzi; gdy nie ma otwartego bąbelka — otwórz go."""
        message = self._streaming or self.start_assistant()
        message.text += text
        return message

    def finish(self, text: str | None = None) -> ChatMessage | None:
        """Zamknij strumień. ``text`` nadpisuje treść (np. pełną odpowiedzią modelu).

        Bąbelek, w którym nic nie zdążyło się pojawić, jest usuwany: puste dymki
        po przerwanej albo pustej odpowiedzi tylko śmiecą w rozmowie.
        """
        message = self._streaming
        self._streaming = None
        if message is None:
            return None
        if text is not None:
            message.text = text
        message.streaming = False
        if message.is_empty:
            try:
                self._messages.remove(message)
            except ValueError:  # pragma: no cover - już wypadła przez limit
                pass
            return None
        return message

    def clear(self) -> None:
        self._messages.clear()
        self._streaming = None

    def _trim(self) -> None:
        while len(self._messages) > self._max_messages:
            dropped = self._messages.pop(0)
            if dropped is self._streaming:  # pragma: no cover - skrajnie długi strumień
                self._streaming = None


# --------------------------------------------------------------------------- #
# Stan usług
# --------------------------------------------------------------------------- #


class ServiceState(StrEnum):
    """Stan jednej usługi. ``UNKNOWN`` znaczy „jeszcze nie sprawdzone", nie „ok"."""

    OK = "ok"
    BUSY = "busy"
    OFF = "off"
    ERROR = "error"
    UNKNOWN = "unknown"

    def color(self, palette: Palette) -> str:
        return {
            ServiceState.OK: palette.state_ok,
            ServiceState.BUSY: palette.state_busy,
            ServiceState.OFF: palette.state_off,
            ServiceState.ERROR: palette.state_error,
            ServiceState.UNKNOWN: palette.state_off,
        }[self]


class ListeningState(StrEnum):
    """Co asystent teraz robi — źródło wskaźnika nasłuchiwania."""

    OFF = "off"
    IDLE = "idle"
    WAITING_WAKE = "waiting_wake"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"

    @property
    def is_active(self) -> bool:
        """Czy dzieje się coś, co ma pulsować (a nie stać w miejscu)?"""
        return self in (
            ListeningState.LISTENING,
            ListeningState.TRANSCRIBING,
            ListeningState.THINKING,
            ListeningState.SPEAKING,
        )

    @property
    def is_microphone_on(self) -> bool:
        return self in (
            ListeningState.WAITING_WAKE,
            ListeningState.LISTENING,
            ListeningState.TRANSCRIBING,
        )

    def caption(self, *, wake_phrase: str = "") -> str:
        """Podpis pod wskaźnikiem — jedno krótkie zdanie w języku interfejsu."""
        if self is ListeningState.WAITING_WAKE:
            if wake_phrase:
                return t("listening.waiting_wake", phrase=wake_phrase)
            return t("listening.waiting_wake_generic")
        return t(f"listening.{self.value}")

    def color(self, palette: Palette) -> str:
        if self in (ListeningState.OFF, ListeningState.IDLE):
            return palette.listening_idle
        if self in (ListeningState.TRANSCRIBING, ListeningState.THINKING):
            return palette.listening_busy
        return palette.listening_active


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Jedna pozycja panelu statusu: etykieta, stan, szczegół."""

    key: str
    label: str
    state: ServiceState = ServiceState.UNKNOWN
    detail: str = ""

    def line(self) -> str:
        return f"{self.label}: {self.detail}" if self.detail else self.label


# Kolejność pozycji statusu. Klucze są TECHNICZNE i nie zmieniają się z językiem —
# etykietę do pokazania daje katalog tekstów (``service.<klucz>``), więc panel i
# testy mówią o tym samym niezależnie od ustawionego języka interfejsu.
SERVICE_KEYS: Final[tuple[str, ...]] = (
    "mic",
    "wake",
    "whisper",
    "ollama",
    "speech",
    "memory",
    "tools",
)


def service_label(key: str) -> str:
    """Nazwa pozycji statusu w języku interfejsu."""
    return t(f"service.{key}") if key in SERVICE_KEYS else key


def default_services() -> tuple[ServiceStatus, ...]:
    """Pozycje statusu w stanie „jeszcze nie wiem" — tak wygląda okno przy starcie."""
    return tuple(
        ServiceStatus(
            key=key,
            label=service_label(key),
            state=ServiceState.UNKNOWN,
            detail=t("common.checking"),
        )
        for key in SERVICE_KEYS
    )


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Migawka stanu przesyłana z wątku roboczego do interfejsu.

    Niezmienna z rozmysłem: wątek roboczy buduje nową migawkę, a interfejs rysuje
    tę, którą właśnie dostał. Nie ma tu wspólnego obiektu, który jeden wątek
    czyta, a drugi w tym samym czasie zmienia.
    """

    assistant_name: str = "Asystent"
    model: str = ""
    host: str = ""
    language: str = ""
    language_forced: bool = True
    listening: ListeningState = ListeningState.OFF
    wake_phrase: str = ""
    busy: bool = False
    services: tuple[ServiceStatus, ...] = field(default_factory=default_services)

    def language_label(self) -> str:
        """Język odpowiedzi w postaci dla panelu: kod + skąd się wziął."""
        if not self.language or self.language == "auto":
            return t("gui.status.language_auto")
        if self.language_forced:
            return t("gui.status.language_forced", code=self.language)
        return self.language

    def service(self, key: str) -> ServiceStatus | None:
        for item in self.services:
            if item.key == key:
                return item
        return None

    def with_service(
        self, key: str, state: ServiceState, detail: str = ""
    ) -> StatusSnapshot:
        """Migawka z jedną zmienioną pozycją (pozostałe zostają bez zmian)."""
        label = service_label(key)
        updated: list[ServiceStatus] = []
        seen = False
        for item in self.services:
            if item.key == key:
                updated.append(replace(item, state=state, detail=detail))
                seen = True
            else:
                updated.append(item)
        if not seen:
            updated.append(ServiceStatus(key=key, label=label, state=state, detail=detail))
        return replace(self, services=tuple(updated))

    def with_services(self, items: Iterable[ServiceStatus]) -> StatusSnapshot:
        snapshot = self
        for item in items:
            snapshot = snapshot.with_service(item.key, item.state, item.detail)
        return snapshot

    def with_listening(self, state: ListeningState) -> StatusSnapshot:
        return replace(self, listening=state)

    def summary_lines(self) -> tuple[str, ...]:
        """Stan jako tekst — do logu i do testów (to samo, co widzi użytkownik)."""
        head = (
            f"{self.assistant_name}: {self.model or t('common.none')}"
            f" @ {self.host or t('common.none')}",
            t("gui.status.language", language=self.language_label()),
            self.listening.caption(wake_phrase=self.wake_phrase),
        )
        return head + tuple(item.line() for item in self.services)


def services_from_report(report: object) -> tuple[ServiceStatus, ...]:
    """Przełóż raport zależności (Faza 1) na pozycje panelu statusu.

    Raport jest liczony PRZED zbudowaniem interfejsu, więc okno od pierwszej
    sekundy pokazuje prawdę o maszynie: brak mikrofonu, brak głosu, nieuruchomioną
    Ollamę. Funkcja jest defensywna — raport bywa niepełny (część sprawdzeń
    rejestrują moduły, których na tej maszynie nie da się zaimportować).
    """
    checks: Sequence[object] = tuple(getattr(report, "checks", ()) or ())
    by_name: dict[str, object] = {}
    for check in checks:
        name = str(getattr(check, "name", "")).strip().lower()
        if name:
            by_name.setdefault(name, check)

    def state_for(name_key: str) -> tuple[ServiceState, str]:
        """Znajdź pozycję raportu po nazwie — w DOWOLNYM języku interfejsu.

        Raport powstaje w języku obowiązującym w chwili jego liczenia, a
        ``config/dependency_status.json`` bywa starszy niż bieżące ustawienie —
        dlatego szukamy nazwy we wszystkich katalogach. Dopasowanie po przedrostku
        obsługuje nazwy uszczegółowione („Whisper model (cache)").
        """
        wanted = [
            t(name_key, _lang=code).strip().lower() for code in SUPPORTED_UI_LANGUAGES
        ]
        check = next((by_name[name] for name in wanted if name in by_name), None)
        if check is None:
            check = next(
                (
                    value
                    for key, value in by_name.items()
                    if any(key.startswith(name) for name in wanted)
                ),
                None,
            )
        if check is None:
            return ServiceState.UNKNOWN, t("common.not_checked")
        ok = bool(getattr(check, "ok", False))
        detail = str(getattr(check, "detail", "") or "")
        required = bool(getattr(check, "required", False))
        if ok:
            return ServiceState.OK, detail
        return (ServiceState.ERROR if required else ServiceState.OFF), detail

    items: list[ServiceStatus] = []
    for key, source in (("mic", "deps.mic.name"), ("whisper", "deps.whisper.cache_name")):
        state, detail = state_for(source)
        items.append(
            ServiceStatus(key=key, label=service_label(key), state=state, detail=detail)
        )

    ollama = getattr(report, "ollama", None)
    if ollama is not None:
        reachable = bool(getattr(ollama, "reachable", False))
        model_present = bool(getattr(ollama, "model_present", False))
        if reachable and model_present:
            state = ServiceState.OK
            detail = str(getattr(ollama, "detail", "") or t("common.available"))
        elif reachable:
            state, detail = ServiceState.ERROR, t("deps.ollama.no_model")
        else:
            state, detail = ServiceState.ERROR, t("deps.ollama.unreachable")
        items.append(
            ServiceStatus(
                key="ollama", label=service_label("ollama"), state=state, detail=detail
            )
        )
    return tuple(items)


__all__ = [
    "DEFAULT_MAX_MESSAGES",
    "SERVICE_KEYS",
    "ChatLog",
    "ChatMessage",
    "ChatRole",
    "ListeningState",
    "ServiceState",
    "ServiceStatus",
    "StatusSnapshot",
    "default_services",
    "service_label",
    "services_from_report",
]
