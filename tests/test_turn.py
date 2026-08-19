"""Testy protokołu tury — czyli tego, co model naprawdę widzi po narzędziu.

Zgłoszone z prawdziwej rozmowy dwa objawy, które wyglądają na dwa różne błędy:

* asystent twierdzi, że utworzył plik, choć nic nie powstało („gaslightuje mnie"),
* asystent pyta o zgodę raz za razem — po trzech potwierdzeniach pyta czwarty raz.

Przyczyna jest jedna i leży w protokole: do modelu jechał **wynik narzędzia bez
wywołania**, które go wywołało. Historia wyglądała tak::

    user:      usuń plik
    tool:      <<TOOL_RESULT ...>> {"ok": false, "error": "the user declined"}

Z perspektywy modelu wynik pojawia się znikąd — nie ma go z czym powiązać. Model
albo powtarza wywołanie (i użytkownik dostaje kolejne pytanie o zgodę), albo
opowiada wynik, który wymyślił. Poprawna historia ma trzy wiadomości::

    user:      usuń plik
    assistant: (tool_calls: fs.delete{path: ...})
    tool:      <<TOOL_RESULT ...>> {"ok": false, ...}

Te testy pilnują obu połówek: że wywołanie asystenta trafia do historii i że
do interfejsu idzie zdanie o niewykonanej akcji, którego model nie może
przekręcić — bo powstaje w kodzie, nie w modelu.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from brain.conversation import Message
from brain.tool_router import ToolCall, ToolOutcome
from brain.turn import failure_notice, run_turn
from tools.base import RiskLevel, ToolResult

# --------------------------------------------------------------------------- #
# Atrapy: tyle, ile ogląda `run_turn`
# --------------------------------------------------------------------------- #


class FakeHistory:
    def __init__(self) -> None:
        self.messages: list[Message] = []


class FakeMemory:
    """Historia rozmowy bez bazy — zapamiętuje to samo, co prawdziwa."""

    def __init__(self) -> None:
        self.history = FakeHistory()

    def add_assistant(self, content: str, **metadata: Any) -> Message:
        message = Message(role="assistant", content=content, metadata=dict(metadata))
        self.history.messages.append(message)
        return message

    def add_tool(self, content: str, **metadata: Any) -> Message:
        message = Message(role="tool", content=content, metadata=dict(metadata))
        self.history.messages.append(message)
        return message


class FakeClient:
    """Model, który w pierwszym przejściu woła narzędzie, a w drugim odpowiada."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    async def stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> AsyncIterator[str]:
        self.seen.append([message.to_ollama() for message in messages])
        collect = kwargs.get("collect")
        text = self._replies.pop(0) if self._replies else ""
        if collect is not None and self._calls_pending:
            collect.tool_calls = [{"function": {"name": "fs.delete", "arguments": {}}}]
            self._calls_pending = False
        yield text

    _calls_pending = True


def outcome(*, ok: bool, error: str = "") -> ToolOutcome:
    result = (
        ToolResult.success({"deleted": True}, display="usunięto")
        if ok
        else ToolResult.failure(error)
    )
    return ToolOutcome(
        call=ToolCall(name="fs.delete", arguments={"path": "a.txt"}),
        result=result,
        risk=RiskLevel.HIGH,
        decision="confirm",
    )


class FakeRouter:
    """Router, który wykonuje dokładnie jedno wywołanie o zadanym wyniku."""

    enabled = True

    def __init__(self, produced: ToolOutcome) -> None:
        self._produced = produced
        self._parsed = False
        self.calls_run = 0

    def schemas_for_llm(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "fs.delete"}}]

    def parse(self, *, native: Any = None, text: str = "") -> list[ToolCall]:
        if self._parsed or not native:
            return []
        self._parsed = True
        return [self._produced.call]

    async def run_calls(self, calls: Sequence[ToolCall], _ctx: Any) -> list[ToolOutcome]:
        self.calls_run += len(calls)
        return [self._produced]

    def budget_left(self) -> int:
        return 5


class RecordingView:
    def __init__(self) -> None:
        self.notices: list[str] = []
        self.tools: list[ToolOutcome] = []

    def on_tool(self, result: ToolOutcome) -> None:
        self.tools.append(result)

    def on_notice(self, text: str) -> None:
        self.notices.append(text)


async def turn(router: FakeRouter, view: RecordingView, memory: FakeMemory) -> str:
    client = FakeClient(["", "gotowe"])
    return await run_turn(
        client,  # type: ignore[arg-type]
        memory,  # type: ignore[arg-type]
        router,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        "PROMPT",
        view=view,
        language="pl",
    )


# --------------------------------------------------------------------------- #
# Wiadomość z wywołaniem wraca do modelu
# --------------------------------------------------------------------------- #


def test_wywolanie_asystenta_trafia_do_historii() -> None:
    """Bez tej wiadomości wynik narzędzia wisi w historii bez pytania."""
    memory = FakeMemory()

    asyncio.run(turn(FakeRouter(outcome(ok=True)), RecordingView(), memory))

    roles = [message.role for message in memory.history.messages]
    assert roles == ["assistant", "tool"], "wynik narzędzia bez poprzedzającego wywołania"

    call_message = memory.history.messages[0].to_ollama()
    assert call_message["tool_calls"][0]["function"]["name"] == "fs.delete"
    # Wynik musi być podpisany nazwą narzędzia — inaczej przy dwóch wywołaniach
    # naraz model nie wie, który wynik należy do którego.
    assert memory.history.messages[1].to_ollama()["tool_name"] == "fs.delete"


def test_zwykla_wiadomosc_nie_dostaje_pustego_pola() -> None:
    """Wiadomość bez wywołań jedzie tak jak dotąd — sam ``role`` i ``content``."""
    plain = Message(role="assistant", content="cześć")

    assert plain.to_ollama() == {"role": "assistant", "content": "cześć"}


def test_tylko_asystent_niesie_wywolania() -> None:
    """``tool_calls`` przy roli ``user`` byłoby błędem protokołu."""
    odd = Message(role="user", content="x", metadata={"tool_calls": [{"a": 1}]})

    assert "tool_calls" not in odd.to_ollama()


# --------------------------------------------------------------------------- #
# Odmowa nie może wyglądać jak sukces
# --------------------------------------------------------------------------- #


def test_odmowa_daje_ostrzezenie_niezalezne_od_modelu() -> None:
    """To zdanie powstaje w kodzie — model nie ma jak go przekręcić."""
    view = RecordingView()

    asyncio.run(
        turn(
            FakeRouter(outcome(ok=False, error="użytkownik nie zgodził się")),
            view,
            FakeMemory(),
        )
    )

    assert view.notices, "odmowa przeszła bez śladu w interfejsie"
    notice = view.notices[0]
    assert "fs.delete" in notice and "NIE" in notice


def test_udane_wywolanie_nie_straszy_ostrzezeniem() -> None:
    view = RecordingView()

    asyncio.run(turn(FakeRouter(outcome(ok=True)), view, FakeMemory()))

    assert view.notices == []


@pytest.mark.parametrize(
    ("language", "fragment"),
    [("pl", "NIE zostało wykonane"), ("en", "was NOT carried out")],
)
def test_ostrzezenie_mowi_jezykiem_uzytkownika(language: str, fragment: str) -> None:
    assert fragment in failure_notice("fs.delete", "odmowa", language)


def test_ostrzezenie_bez_powodu_ma_sensowna_postac() -> None:
    """Narzędzie może zawieść bez opisu błędu — komunikat i tak musi się złożyć."""
    assert "fs.delete" in failure_notice("fs.delete", "", "pl")
