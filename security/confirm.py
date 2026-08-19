"""Potwierdzenia człowieka dla narzędzi HIGH i CRITICAL (Faza 7).

Zasada: **nigdy automatycznej zgody.** Brak kanału potwierdzeń (praca wsadowa,
przekierowane stdin, usługa w tle) oznacza ODMOWĘ, a nie „przepuść".

Treść pytania buduje NARZĘDZIE ze zwalidowanych argumentów — model językowy nie
ma na nią wpływu. Inaczej mógłby dla `rm -rf ~` napisać „wykonać nieszkodliwą
operację porządkową?".

Jedno potwierdzenie = jedno wywołanie: żądanie ma identyfikator (nonce) i termin
ważności. Odpowiedź spóźniona jest odmową.

Kanały potwierdzeń są wymienne (:class:`ConfirmationBroker`): terminal tutaj,
modal GUI w Fazie 10, głos w dalszych. Testy podstawiają atrapę i nie dotykają
klawiatury.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol, TextIO, runtime_checkable

from security.risk import RiskLevel, describe_risk

logger = logging.getLogger(__name__)

# Ile żądanie potwierdzenia jest ważne, jeśli nikt nie powiedział inaczej.
DEFAULT_CONFIRM_TTL_S: Final[float] = 60.0

# Odpowiedzi uznawane za zgodę dla HIGH. Świadomie krótka lista — „może",
# „chyba tak" czy „ok, ale…" nie są zgodą.
_AFFIRMATIVE: Final[frozenset[str]] = frozenset(
    {"t", "tak", "y", "yes", "j", "ja", "potwierdzam", "zgoda", "confirm", "ok"}
)

# CRITICAL wymaga pełnej frazy, nie jednej litery: przy pomyłce („t" zamiast „r")
# koszt jest nieodwracalny. Frazy są w obu językach, bo asystent bywa dwujęzyczny.
CRITICAL_PHRASES: Final[tuple[str, ...]] = ("tak, potwierdzam", "yes, i confirm")


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """Pytanie do człowieka o zgodę na jedno wywołanie narzędzia."""

    tool: str
    risk: RiskLevel
    summary: str
    details: tuple[str, ...] = ()
    preview: str | None = None
    # Ostrzeżenie doklejane przez router, np. gdy wywołanie wynika z treści
    # pobranej z sieci (patrz twarda bariera przeciw prompt injection).
    warning: str = ""
    language: str = "en"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime | None = None

    @classmethod
    def build(
        cls,
        *,
        tool: str,
        risk: RiskLevel,
        summary: str,
        details: Sequence[str] = (),
        preview: str | None = None,
        warning: str = "",
        language: str = "en",
        ttl_s: float = DEFAULT_CONFIRM_TTL_S,
        now: datetime | None = None,
    ) -> ConfirmationRequest:
        moment = now or _utc_now()
        return cls(
            tool=tool,
            risk=risk,
            summary=summary,
            details=tuple(details),
            preview=preview,
            warning=warning,
            language=language,
            created_at=moment,
            expires_at=moment + timedelta(seconds=ttl_s) if ttl_s > 0 else None,
        )

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or _utc_now()) >= self.expires_at

    @property
    def requires_phrase(self) -> bool:
        """Czy do zgody potrzebna jest pełna fraza, a nie samo „tak"?"""
        return self.risk is RiskLevel.CRITICAL

    def render_lines(self) -> list[str]:
        """Pytanie w postaci gotowej do wypisania — wspólne dla terminala i GUI."""
        polish = self.language == "pl"
        header = "Narzędzie prosi o zgodę" if polish else "A tool is asking for permission"
        lines = [
            f"{header}: {self.tool}  [{describe_risk(self.risk, language=self.language)}]",
            f"  {self.summary}",
        ]
        lines.extend(f"    - {detail}" for detail in self.details)
        if self.preview:
            label = "podgląd" if polish else "preview"
            lines.append(f"    {label}:")
            lines.extend(f"      {line}" for line in self.preview.splitlines()[:20])
        if self.warning:
            lines.append(f"    ! {self.warning}")
        return lines

    def prompt(self) -> str:
        """Zachęta do wpisania odpowiedzi (bez znaku nowej linii)."""
        polish = self.language == "pl"
        if self.requires_phrase:
            phrase = CRITICAL_PHRASES[0] if polish else CRITICAL_PHRASES[1]
            if polish:
                return f"Wpisz dokładnie: {phrase} — cokolwiek innego anuluje: "
            return f"Type exactly: {phrase} — anything else cancels: "
        return "Wykonać? [t/N]: " if polish else "Proceed? [y/N]: "


@dataclass(frozen=True, slots=True)
class ConfirmationOutcome:
    """Odpowiedź człowieka (albo jej brak)."""

    approved: bool
    reason: str = ""
    channel: str = ""

    @classmethod
    def approve(cls, *, channel: str, reason: str = "") -> ConfirmationOutcome:
        return cls(approved=True, reason=reason, channel=channel)

    @classmethod
    def deny(cls, *, channel: str, reason: str) -> ConfirmationOutcome:
        return cls(approved=False, reason=reason, channel=channel)


@runtime_checkable
class ConfirmationBroker(Protocol):
    """Kanał, którym pytamy człowieka o zgodę."""

    @property
    def channel(self) -> str:
        """Nazwa kanału — trafia do logu audytu."""

    @property
    def available(self) -> bool:
        """Czy tym kanałem da się teraz o cokolwiek zapytać?"""

    async def ask(self, request: ConfirmationRequest) -> ConfirmationOutcome:
        """Zapytaj i zwróć odpowiedź. Nie rzuca — brak odpowiedzi to odmowa."""


class AutoDenyBroker:
    """Brak kanału potwierdzeń: każde żądanie jest odrzucane.

    Używany, gdy program nie ma z kim rozmawiać (skrypt, usługa, przekierowane
    stdin). Świadomie nie ma tu wariantu „auto-zgoda" — nie istnieje ustawienie,
    które by go włączyło.
    """

    def __init__(self, *, reason: str = "") -> None:
        self._reason = reason or "brak kanału potwierdzeń (praca bez interaktywnego terminala)"

    @property
    def channel(self) -> str:
        return "auto-deny"

    @property
    def available(self) -> bool:
        return False

    async def ask(self, request: ConfirmationRequest) -> ConfirmationOutcome:
        logger.info("Odmowa automatyczna dla %s: %s", request.tool, self._reason)
        return ConfirmationOutcome.deny(channel=self.channel, reason=self._reason)


class CallbackBroker:
    """Potwierdzenia przez dowolną funkcję — dla GUI (Faza 10) i dla testów.

    Funkcja może być zwykła albo asynchroniczna; zwraca ``bool`` albo gotowy
    :class:`ConfirmationOutcome`. Wyjątek w funkcji jest traktowany jak odmowa —
    awaria kanału potwierdzeń nie może skutkować wykonaniem akcji.
    """

    def __init__(
        self,
        callback: Callable[[ConfirmationRequest], Any],
        *,
        channel: str = "callback",
        timeout_s: float = DEFAULT_CONFIRM_TTL_S,
    ) -> None:
        self._callback = callback
        self._channel = channel
        self._timeout_s = max(0.0, timeout_s)

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def available(self) -> bool:
        return True

    async def ask(self, request: ConfirmationRequest) -> ConfirmationOutcome:
        try:
            answer = await self._invoke(request)
        except TimeoutError:
            return ConfirmationOutcome.deny(
                channel=self._channel, reason="brak odpowiedzi w wyznaczonym czasie"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Kanał potwierdzeń zgłosił błąd: %s", exc)
            logger.debug("Szczegóły błędu kanału potwierdzeń", exc_info=True)
            return ConfirmationOutcome.deny(
                channel=self._channel, reason=f"kanał potwierdzeń zawiódł ({exc})"
            )

        if isinstance(answer, ConfirmationOutcome):
            return answer
        if answer is True:
            return ConfirmationOutcome.approve(channel=self._channel)
        return ConfirmationOutcome.deny(channel=self._channel, reason="użytkownik odmówił")

    async def _invoke(self, request: ConfirmationRequest) -> Any:
        result = self._callback(request)
        if asyncio.iscoroutine(result):
            if self._timeout_s > 0:
                return await asyncio.wait_for(result, timeout=self._timeout_s)
            return await result
        return result


class TerminalConfirmationBroker:
    """Potwierdzenia wpisywane w terminalu.

    Czytanie odpowiedzi idzie do osobnego wątku (``asyncio.to_thread``), żeby nie
    blokować pętli zdarzeń — dzięki temu strumień odpowiedzi modelu i odtwarzanie
    mowy nie zamierają na czas pytania.

    **Czego tu świadomie nie ma: przerwania czekania po czasie.** Blokującego
    ``input()`` nie da się przenośnie anulować — ``select()`` na stdin działa na
    Uniksie, ale nie na uchwytach konsoli Windows, a ``msvcrt`` jest tylko na
    Windowsie. Zamiast pisać dwie różne implementacje jednego pytania, pytamy i
    czekamy, a termin ważności żądania sprawdzamy PO otrzymaniu odpowiedzi:
    spóźniona zgoda jest odmową. Anulowanie zawsze jest pod ręką — Enter, „n"
    albo Ctrl+C.
    """

    def __init__(
        self,
        *,
        reader: Callable[[str], str] | None = None,
        stream: TextIO | None = None,
        interactive: bool | None = None,
    ) -> None:
        self._reader = reader or input
        self._stream = stream
        self._interactive = interactive

    @property
    def channel(self) -> str:
        return "terminal"

    @property
    def available(self) -> bool:
        if self._interactive is not None:
            return self._interactive
        return _stdin_is_interactive()

    async def ask(self, request: ConfirmationRequest) -> ConfirmationOutcome:
        if not self.available:
            return ConfirmationOutcome.deny(
                channel=self.channel,
                reason="stdin nie jest terminalem — nie ma kogo zapytać",
            )

        for line in request.render_lines():
            self._write(line)

        try:
            answer = await asyncio.to_thread(self._reader, request.prompt())
        except asyncio.CancelledError:
            raise
        except (EOFError, KeyboardInterrupt):
            self._write("")
            return ConfirmationOutcome.deny(
                channel=self.channel, reason="anulowane przez użytkownika"
            )
        except Exception as exc:  # pragma: no cover - awaria stdin
            logger.warning("Nie udało się odczytać potwierdzenia: %s", exc)
            return ConfirmationOutcome.deny(
                channel=self.channel, reason=f"nie udało się odczytać odpowiedzi ({exc})"
            )

        if request.is_expired():
            return ConfirmationOutcome.deny(
                channel=self.channel, reason="żądanie straciło ważność przed odpowiedzią"
            )
        return interpret_answer(answer, request, channel=self.channel)

    def _write(self, text: str) -> None:
        if self._stream is not None:
            print(text, file=self._stream)
        else:
            print(text)


def interpret_answer(
    answer: str, request: ConfirmationRequest, *, channel: str = "terminal"
) -> ConfirmationOutcome:
    """Zamień odpowiedź człowieka na decyzję. Wszystko poza zgodą = odmowa."""
    cleaned = " ".join(str(answer or "").split()).strip().lower().rstrip(".!")
    if request.requires_phrase:
        if cleaned in CRITICAL_PHRASES:
            return ConfirmationOutcome.approve(channel=channel, reason="pełna fraza potwierdzenia")
        return ConfirmationOutcome.deny(
            channel=channel,
            reason=("anulowane" if cleaned else "brak wymaganej frazy potwierdzenia"),
        )
    if cleaned in _AFFIRMATIVE:
        return ConfirmationOutcome.approve(channel=channel)
    return ConfirmationOutcome.deny(channel=channel, reason="użytkownik odmówił")


def _stdin_is_interactive() -> bool:
    """Czy da się o cokolwiek zapytać na stdin?

    Nie zakładamy istnienia stdin: pod pythonw.exe na Windowsie i w usługach
    ``sys.stdin`` bywa ``None``, a ``isatty()`` potrafi rzucić przy zamkniętym
    uchwycie. Każda wątpliwość znaczy „nie ma kogo pytać".
    """
    import sys

    stream = getattr(sys, "stdin", None)
    if stream is None:
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):  # pragma: no cover - zależne od systemu
        return False


def default_broker(*, force_terminal: bool | None = None) -> ConfirmationBroker:
    """Kanał właściwy dla tego uruchomienia: terminal, gdy jest; inaczej odmowa."""
    broker = TerminalConfirmationBroker(interactive=force_terminal)
    if broker.available:
        return broker
    return AutoDenyBroker()


__all__ = [
    "CRITICAL_PHRASES",
    "DEFAULT_CONFIRM_TTL_S",
    "AutoDenyBroker",
    "CallbackBroker",
    "ConfirmationBroker",
    "ConfirmationOutcome",
    "ConfirmationRequest",
    "TerminalConfirmationBroker",
    "default_broker",
    "interpret_answer",
]
