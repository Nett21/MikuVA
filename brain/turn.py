"""Jedna tura rozmowy: model → narzędzia → model → odpowiedź.

Ten moduł powstał w Fazie 10 i **nie dodaje nowego zachowania** — wyjmuje na
zewnątrz to, co od Fazy 7 stało w ``main.py``. Powód jest prosty: GUI prowadzi
dokładnie taką samą turę co terminal, a pętla narzędziowa pilnuje budżetu
wywołań i śladu danych niezaufanych. Dwie kopie takiej pętli rozjechałyby się
przy pierwszej zmianie, a rozjazd w tym miejscu znaczy „w jednym interfejsie
limit działa, w drugim nie".

Różnicą między interfejsami jest **kanał wyjściowy**, nie logika. Dlatego tura
pisze do :class:`TurnView`: terminal wypisuje tekst przez ``print``, GUI wysyła
zdarzenia do kolejki wątku interfejsu. Mowa (Faza 4) jest karmiona tym samym
strumieniem w obu przypadkach, więc jej obsługa siedzi tutaj, a nie w widoku.

Wszystko jest asynchroniczne, ale **nie zakłada żadnej pętli zdarzeń**: terminal
uruchamia turę przez ``loop.run_until_complete``, GUI — w wątku roboczym z własną
pętlą. Interfejs graficzny nie jest tu w ogóle widoczny.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # importy tylko dla typów — moduł nie wymaga httpx ani narzędzi
    from brain.conversation import ConversationHistory
    from brain.llm import OllamaClient, StreamedReply
    from brain.memory import ConversationMemory
    from brain.tool_router import ToolOutcome, ToolRouter
    from tools.base import ToolContext

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Kontrakty: gdzie leci tekst i czym się go mówi
# --------------------------------------------------------------------------- #


@runtime_checkable
class TurnSpeaker(Protocol):
    """Wyjście mowy widziane przez turę.

    Celowo minimalne: tura nie wie, czy po drugiej stronie jest Piper, atrapa w
    teście, czy nic. ``enabled`` bywa ``False`` w trakcie tury (mowa potrafi się
    wyłączyć po błędzie karty dźwiękowej), więc jest sprawdzane, a nie zapamiętane.
    """

    @property
    def enabled(self) -> bool: ...

    def begin(self, language: str | None = None) -> None: ...

    def feed(self, text: str) -> None: ...

    def end(self) -> None: ...

    def cancel(self) -> None: ...


@runtime_checkable
class TurnView(Protocol):
    """Kanał, którym tura mówi do człowieka.

    Metody są wołane z wątku, w którym pracuje tura — implementacja dla GUI musi
    więc tylko przekazać zdarzenie dalej (kolejka), a nie rysować.
    """

    def on_thinking(self) -> None:
        """Model milczy i liczy (modele rozumujące potrafią kilkanaście sekund)."""

    def on_reply_start(self) -> None:
        """Zaraz poleci pierwszy fragment odpowiedzi."""

    def on_chunk(self, text: str) -> None:
        """Kolejny fragment odpowiedzi — dokładnie taki, jaki przyszedł z modelu."""

    def on_reply_end(self, text: str) -> None:
        """Koniec strumienia; ``text`` to cała odpowiedź tego przejścia."""

    def on_tool(self, outcome: ToolOutcome) -> None:
        """Narzędzie zostało wywołane (albo odrzucone) — jest wynik dla człowieka."""

    def on_notice(self, text: str) -> None:
        """Krótki komunikat systemowy w trakcie tury (np. wyczerpany budżet)."""


class SilentView:
    """Widok, który nic nie robi — dla testów i dla użycia bez interfejsu."""

    def on_thinking(self) -> None:
        return None

    def on_reply_start(self) -> None:
        return None

    def on_chunk(self, text: str) -> None:
        return None

    def on_reply_end(self, text: str) -> None:
        return None

    def on_tool(self, outcome: ToolOutcome) -> None:
        return None

    def on_notice(self, text: str) -> None:
        return None


def _safe(view: TurnView, method: str, *args: Any) -> None:
    """Zawołaj widok, ale nie pozwól mu wywrócić tury.

    Interfejs jest dodatkiem do rozmowy, a nie jej warunkiem: błąd rysowania
    (zamknięte okno, zapełniona kolejka) nie może przerwać generowania ani
    zostawić historii rozmowy w połowie zapisanej.
    """
    function = getattr(view, method, None)
    if function is None:
        return
    try:
        function(*args)
    except Exception:  # pragma: no cover - zależne od implementacji widoku
        logger.debug("Widok tury zgłosił błąd w %s", method, exc_info=True)


# --------------------------------------------------------------------------- #
# Jedno przejście: strumień odpowiedzi modelu
# --------------------------------------------------------------------------- #


async def stream_reply(
    client: OllamaClient,
    history: ConversationHistory,
    system_prompt: str,
    *,
    view: TurnView | None = None,
    speaker: TurnSpeaker | None = None,
    language: str | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    collect: StreamedReply | None = None,
    context: str = "",
) -> str:
    """Przepisz odpowiedź modelu do widoku (i do mowy) w miarę jej napływania.

    Tekst leci do interfejsu fragment po fragmencie, a równolegle trafia do
    syntezy mowy: pierwsze pełne zdanie jest wypowiadane, gdy model pisze dopiero
    kolejne. Dzięki temu odtwarzanie nie czeka na koniec generowania.

    ``tools`` i ``collect`` obsługują tool calling (Faza 7): model dostaje listę
    narzędzi, a jego wywołania trafiają do ``collect`` — strumień tekstu jest
    wtedy zwykle pusty, bo odpowiedź powstaje w drugim przejściu, już z wynikami.
    """
    sink = view if view is not None else SilentView()
    chunks: list[str] = []
    started = False
    thinking = False
    speaking = speaker is not None and speaker.enabled
    if speaking and speaker is not None:
        speaker.begin(language)

    try:

        def on_thinking(_: str) -> None:
            # Sygnał „model liczy" ma sens tylko przed pierwszym fragmentem —
            # potem tekst na ekranie sam pokazuje, że coś się dzieje.
            nonlocal thinking
            if not started and not thinking:
                thinking = True
                _safe(sink, "on_thinking")

        # Argumenty Fazy 7 dokładamy tylko wtedy, gdy są potrzebne: klient bez
        # obsługi tool-callingu (starsza wersja, atrapa w teście, inny backend)
        # ma dalej działać z tą samą sygnaturą co w Fazie 1.
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        if collect is not None:
            extra["collect"] = collect
        if context:
            extra["context"] = context

        async for chunk in client.stream_chat(
            history.messages,
            system=system_prompt,
            on_thinking=on_thinking,
            **extra,
        ):
            if not started:
                started = True
                _safe(sink, "on_reply_start")
            chunks.append(chunk)
            _safe(sink, "on_chunk", chunk)
            if speaking and speaker is not None:
                speaker.feed(chunk)

        answer = "".join(chunks)
        if started:
            _safe(sink, "on_reply_end", answer)
    except BaseException:
        # Przerwanie albo błąd modelu ucina też mowę — inaczej asystent
        # dokańczałby zdanie, którego w interfejsie już nie ma.
        if speaking and speaker is not None:
            speaker.cancel()
        raise
    else:
        if speaking and speaker is not None:
            # Dokończenie wypowiedzi czeka na głośnik, więc idzie do wątku —
            # pętla zdarzeń zostaje wolna i przerwanie działa natychmiast.
            await asyncio.to_thread(speaker.end)
    return "".join(chunks)


# --------------------------------------------------------------------------- #
# Cała tura: model → narzędzia → model
# --------------------------------------------------------------------------- #

# Komunikat o wyczerpanym budżecie — jedno zdanie, bo bywa też wypowiadane.
BUDGET_NOTICE_PL: str = "Limit wywołań narzędzi w tej turze wyczerpany."
BUDGET_NOTICE_EN: str = "The tool call limit for this turn has been used up."


def budget_notice(language: str = "en") -> str:
    return BUDGET_NOTICE_PL if language == "pl" else BUDGET_NOTICE_EN


# Ostrzeżenie o NIEWYKONANEJ akcji. Świadomie budowane w kodzie, nie przez model:
# to jedyne zdanie w turze, o którym wiadomo na pewno, że jest prawdziwe.
FAILURE_NOTICE_PL: str = "Uwaga: {tool} NIE zostało wykonane ({error})."
FAILURE_NOTICE_EN: str = "Note: {tool} was NOT carried out ({error})."


def failure_notice(tool: str, error: str, language: str = "en") -> str:
    template = FAILURE_NOTICE_PL if language == "pl" else FAILURE_NOTICE_EN
    return template.format(tool=tool, error=error or "—")


async def run_turn(
    client: OllamaClient,
    memory: ConversationMemory,
    router: ToolRouter | None,
    ctx: ToolContext | None,
    system_prompt: str,
    *,
    view: TurnView | None = None,
    speaker: TurnSpeaker | None = None,
    language: str = "en",
    context: str = "",
) -> str:
    """Pełny przepływ tury: model → narzędzia → model → odpowiedź (Faza 7).

    Bez narzędzi (albo gdy model o żadne nie poprosi) jest to dokładnie to samo
    strumieniowanie co w Fazie 1 — jedno przejście, bez dodatkowego kosztu.
    Gdy narzędzie zostaje wywołane, jego wynik ląduje w historii jako wiadomość
    roli ``tool`` i model dostaje drugą szansę na odpowiedź, tym razem z danymi.
    """
    # Import lokalny, jak reszta rzeczy z brain.llm: bez httpx interfejs nadal
    # pokazuje czytelny komunikat zamiast wywalać się na imporcie.
    from brain.llm import StreamedReply

    sink = view if view is not None else SilentView()
    tools_available = router is not None and ctx is not None and router.enabled
    schemas = router.schemas_for_llm() if tools_available and router is not None else None

    while True:
        reply = StreamedReply()
        answer = await stream_reply(
            client,
            memory.history,
            system_prompt,
            view=sink,
            speaker=speaker,
            language=language,
            tools=schemas,
            collect=reply if tools_available else None,
            context=context,
        )

        if not tools_available or router is None or ctx is None:
            return answer

        calls = router.parse(
            native=reply.tool_calls,
            # Wariant tekstowy tylko wtedy, gdy model nie zgłosił wywołań
            # natywnie — inaczej ten sam zamiar policzylibyśmy dwa razy.
            text="" if reply.tool_calls else answer,
        )
        if not calls:
            return answer

        # Wiadomość asystenta z JEGO wywołaniami — razem z tekstem, który zdążył
        # napisać („sprawdzę godzinę"). Bez tej wiadomości model widzi wynik
        # narzędzia bez poprzedzającego wywołania i traktuje go jak szum: pyta
        # o to samo drugi raz albo twierdzi, że zrobił coś, czego nie zrobił.
        memory.add_assistant(
            answer,
            language=language,
            tool_calls=[
                {"function": {"name": call.name, "arguments": dict(call.arguments)}}
                for call in calls
            ],
        )

        refused: list[ToolOutcome] = []
        for outcome in await router.run_calls(calls, ctx):
            _safe(sink, "on_tool", outcome)
            memory.add_tool(
                outcome.message_for_llm(), tool=outcome.call.name, language=language
            )
            if not outcome.ok:
                refused.append(outcome)

        # Deterministyczne ostrzeżenie: model bywa przekonany, że akcja się udała,
        # mimo odmowy. Ta linia nie zależy od modelu — pochodzi z wyniku narzędzia.
        for outcome in refused:
            _safe(
                sink,
                "on_notice",
                failure_notice(outcome.call.name, outcome.result.error, language),
            )

        if router.budget_left() <= 0:
            # Budżet wyczerpany: ostatnie przejście BEZ narzędzi, żeby model musiał
            # odpowiedzieć tym, co ma, zamiast wołać kolejne narzędzie w pętli.
            _safe(sink, "on_notice", budget_notice(language))
            return await stream_reply(
                client,
                memory.history,
                system_prompt,
                view=sink,
                speaker=speaker,
                language=language,
                context=context,
            )


__all__ = [
    "BUDGET_NOTICE_EN",
    "BUDGET_NOTICE_PL",
    "FAILURE_NOTICE_EN",
    "FAILURE_NOTICE_PL",
    "SilentView",
    "TurnSpeaker",
    "TurnView",
    "budget_notice",
    "failure_notice",
    "run_turn",
    "stream_reply",
]
