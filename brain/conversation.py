"""Okno rozmowy: to, co model widzi w bieżącej turze.

Historia robocza jest trzymana w pamięci; trwały zapis i przypominanie należą do
``brain/memory.py`` (Faza 5). Prompt systemowy NIE jest częścią historii — jest
budowany na nowo przy każdej turze, dzięki czemu zmiana ``assistant_name`` czy
języka działa od razu.

Wiadomości wypadające poza limit nie znikają po cichu: :meth:`ConversationHistory.trim`
je ZWRACA i podaje do ``on_evict``. Dzięki temu Faza 5 może je streścić i zapisać
zamiast po prostu wyrzucić — Faza 1 nie miała tego dokąd oddać, więc obcinała.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from i18n import t

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant", "tool"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Message(BaseModel):
    """Pojedyncza wiadomość w rozmowie."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_ollama(self) -> dict[str, Any]:
        """Postać wymagana przez ``/api/chat`` Ollamy.

        Wiadomość asystenta może nieść ``tool_calls``, a wynik narzędzia — nazwę
        narzędzia. To nie jest ozdoba protokołu: bez wywołania po stronie
        asystenta model dostaje wynik „znikąd" i albo powtarza to samo wywołanie
        (użytkownik jest pytany o zgodę raz za razem), albo opowiada, że akcja
        się udała, choć została odrzucona. Oba objawy zgłoszone z prawdziwej
        rozmowy i odtworzone w testach.
        """
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        calls = self.metadata.get("tool_calls")
        if self.role == "assistant" and calls:
            payload["tool_calls"] = list(calls)
        if self.role == "tool":
            name = str(self.metadata.get("tool") or "")
            if name:
                payload["tool_name"] = name
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def select_for_model(
    messages: Sequence[Message],
    *,
    max_messages: int = 0,
    max_chars: int = 0,
) -> tuple[Message, ...]:
    """Ostatni fragment rozmowy, który ma trafić do modelu w tej turze.

    To NIE jest przycinanie okna (:meth:`ConversationHistory.trim`). Okno jest
    świadomie większe: z niego powstają streszczenia (Faza 5) i to ono opisuje
    rozmowę dla człowieka. Model dostaje mniej, bo każdy tysiąc tokenów promptu
    to na słabszej maszynie sekundy czekania, a starsze tury wracają do niego
    streszczeniem i przypomnieniem semantycznym w bloku kontekstu.

    Limit ``0`` (albo jego brak) znaczy „bez dodatkowego ograniczenia".

    Dwie reguły, które nie są kosmetyką:

    * **ostatnia wiadomość zostaje zawsze** — bieżące pytanie nie może wypaść,
      choćby samo przekraczało limit znaków,
    * **wynik narzędzia nigdy nie zostaje bez swojego wywołania**. Wiadomość
      ``tool`` bez poprzedzającej ją wiadomości asystenta z ``tool_calls`` to
      dla modelu wynik „znikąd": albo powtarza wywołanie (użytkownik jest
      pytany o zgodę raz za razem), albo opowiada, że akcja się udała, choć
      została odrzucona. Dlatego cięcie przesuwa się w przód, dopóki pierwsza
      wybrana wiadomość jest wynikiem narzędzia.
    """
    total = len(messages)
    if total == 0:
        return ()
    if max_messages <= 0 and max_chars <= 0:
        return tuple(messages)

    start = total - 1  # bieżąca tura zostaje bezwarunkowo
    chars = len(messages[start].content)
    while start > 0:
        candidate = messages[start - 1]
        if max_messages > 0 and (total - (start - 1)) > max_messages:
            break
        if max_chars > 0 and chars + len(candidate.content) > max_chars:
            break
        start -= 1
        chars += len(candidate.content)

    # Osierocone wyniki narzędzi na początku fragmentu: odcinamy je razem z
    # ogonem, którego i tak nie ma jak zrekonstruować.
    while start < total - 1 and messages[start].role == "tool":
        start += 1

    return tuple(messages[start:])


class ConversationHistory:
    """Okno rozmowy ograniczone liczbą wiadomości i liczbą znaków.

    Przy przekroczeniu któregokolwiek limitu z okna wypadają NAJSTARSZE
    wiadomości — bieżąca tura zawsze zostaje w kontekście.

    Przycinanie schodzi PONIŻEJ limitu (do ``trim_ratio`` jego wartości), a nie
    dokładnie do niego. Powód jest praktyczny: przy przycinaniu „co do sztuki"
    każda kolejna tura wypychałaby jedną wiadomość, więc streszczanie z Fazy 5
    odpalałoby się przy każdej wypowiedzi. Z zapasem dzieje się to rzadko i od
    razu dla większej porcji tekstu.

    ``on_evict`` dostaje wiadomości wypadające z okna. Bez tej funkcji zachowanie
    jest takie jak w Fazie 1 — wiadomości po prostu przepadają.
    """

    def __init__(
        self,
        *,
        max_messages: int = 40,
        max_chars: int = 12_000,
        trim_ratio: float = 0.75,
        on_evict: Callable[[Sequence[Message]], None] | None = None,
    ) -> None:
        if max_messages < 2:
            raise ValueError(t("conv.min_messages"))
        if max_chars < 100:
            raise ValueError(t("conv.min_chars"))
        if not 0.25 <= trim_ratio <= 1.0:
            raise ValueError(t("conv.trim_ratio_range"))
        self._max_messages = max_messages
        self._max_chars = max_chars
        self._trim_ratio = trim_ratio
        self._on_evict = on_evict
        self._messages: list[Message] = []

    # --- właściwości ---------------------------------------------------- #

    @property
    def max_messages(self) -> int:
        return self._max_messages

    @property
    def max_chars(self) -> int:
        return self._max_chars

    @property
    def trim_ratio(self) -> float:
        return self._trim_ratio

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def char_count(self) -> int:
        return sum(len(message.content) for message in self._messages)

    # --- modyfikacje ---------------------------------------------------- #

    def add(self, role: Role, content: str, **metadata: Any) -> Message:
        message = Message(role=role, content=content, metadata=dict(metadata))
        self._messages.append(message)
        self.trim()
        return message

    def set_evict_handler(self, handler: Callable[[Sequence[Message]], None] | None) -> None:
        """Podłącz odbiorcę wiadomości wypadających z okna (Faza 5: streszczanie)."""
        self._on_evict = handler

    def add_user(self, content: str, **metadata: Any) -> Message:
        return self.add("user", content, **metadata)

    def add_assistant(self, content: str, **metadata: Any) -> Message:
        return self.add("assistant", content, **metadata)

    def add_tool(self, content: str, **metadata: Any) -> Message:
        return self.add("tool", content, **metadata)

    def extend(self, messages: Iterable[Message]) -> None:
        self._messages.extend(messages)
        self.trim()

    def trim(self) -> list[Message]:
        """Przytnij okno rozmowy. Zwraca wiadomości, które z niego wypadły.

        Przycinanie rusza dopiero po PRZEKROCZENIU limitu i schodzi poniżej niego
        (``trim_ratio``), więc dzieje się rzadko i porcjami.
        """
        if len(self._messages) <= self._max_messages and self.char_count <= self._max_chars:
            return []

        message_target = max(1, int(self._max_messages * self._trim_ratio))
        char_target = max(1, int(self._max_chars * self._trim_ratio))

        evicted: list[Message] = []
        while len(self._messages) > message_target:
            evicted.append(self._messages.pop(0))
        # Ostatnia wiadomość zostaje zawsze: bieżąca tura nie może wypaść z okna,
        # choćby sama przekraczała limit znaków.
        while len(self._messages) > 1 and self.char_count > char_target:
            evicted.append(self._messages.pop(0))

        if evicted and self._on_evict is not None:
            try:
                self._on_evict(tuple(evicted))
            except Exception:  # odbiorca nie może zablokować rozmowy
                logger.exception("Obsługa wiadomości wypadających z okna zgłosiła błąd")
        return evicted

    def clear(self) -> None:
        self._messages.clear()

    # --- odczyt ---------------------------------------------------------- #

    def for_llm(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Historia w formacie Ollamy, opcjonalnie z promptem systemowym na początku."""
        payload: list[dict[str, str]] = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(message.to_ollama() for message in self._messages)
        return payload

    def last(self, role: Role | None = None) -> Message | None:
        for message in reversed(self._messages):
            if role is None or message.role == role:
                return message
        return None

    def to_dicts(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self._messages]

    # --- protokoły ------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationHistory(messages={len(self._messages)}, "
            f"chars={self.char_count}, max_messages={self._max_messages}, "
            f"max_chars={self._max_chars})"
        )


__all__ = ["ConversationHistory", "Message", "Role", "select_for_model"]
