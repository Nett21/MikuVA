"""Kontrakt narzędzia: opis, argumenty, ryzyko, wynik (Faza 7).

Narzędzie to jedyny sposób, w jaki model językowy może cokolwiek zrobić poza
mówieniem. Dlatego kontrakt jest wąski i jawny:

* **nazwa** w postaci ``obszar.czynność`` (``time.now``, ``fs.read``) — model
  widzi ją dosłownie,
* **opis** dla modelu, w jego języku rozmowy,
* **argumenty** jako model Pydantic z ``extra="forbid"`` — nieznane pole to błąd,
  a nie cicha ignorancja; halucynacja argumentu nie ma jak przejść,
* **poziom ryzyka** deklarowany w kodzie (patrz ``security/risk.py``),
* **wykonanie** przez ``run()``, które dostaje już ZWALIDOWANE argumenty i
  :class:`ToolContext` — a w kontekście nie ma ``subprocess``, ``os`` ani
  ``eval``. Narzędzie może zrobić tylko to, co samo w sobie zawiera.

Narzędzie nigdy nie pyta użytkownika o zgodę samo z siebie — buduje tylko treść
pytania (:meth:`Tool.confirmation`), a pyta router przez kanał potwierdzeń.
Dzięki temu ten sam kod działa w terminalu, w GUI i w testach.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import Settings, get_settings
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel, escalate

logger = logging.getLogger(__name__)

# Domyślny limit czasu jednego wywołania. Narzędzie może podać własny, ale nie ma
# wariantu „bez limitu" — zawieszone narzędzie zablokowałoby całą rozmowę.
DEFAULT_TOOL_TIMEOUT_S: Final[float] = 15.0

# Ile znaków wyniku wolno wpuścić do promptu. Reszta jest obcinana — wynik
# narzędzia nie może wypchnąć rozmowy z okna kontekstu.
DEFAULT_RESULT_MAX_CHARS: Final[int] = 4_000


class ToolError(RuntimeError):
    """Błąd narzędzia przewidziany przez autora — wraca do modelu jako treść.

    Nie jest awarią programu: model dostaje komunikat i może spróbować inaczej.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class ToolArgs(BaseModel):
    """Baza dla argumentów narzędzia: nic poza zadeklarowanymi polami.

    ``extra="forbid"`` jest tu najważniejszą linią całego pliku. Model, który
    wymyśli parametr ``force=true`` albo ``path`` dla narzędzia od godziny,
    dostanie błąd walidacji, a nie ciche wykonanie z pominiętym argumentem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        """``null`` w argumencie znaczy „nie podaję go", a nie „ustaw na nic".

        Modele nagminnie wysyłają ``{"limit": null}`` dla argumentów, których nie
        chcą ustawiać — zwłaszcza po pierwszym nieudanym wywołaniu. Odrzucanie
        tego jako błędu typu kosztowało jedną turę rozmowy i nic nie chroniło:
        pominięcie klucza znaczy dokładnie to samo, a wtedy wchodzi wartość
        domyślna. Nieznane pola i złe TYPY nadal są błędem — to jest rozluźnienie
        wyłącznie dla „braku wartości".
        """
        if isinstance(data, Mapping):
            return {key: value for key, value in data.items() if value is not None}
        return data


class ToolResult(BaseModel):
    """Znormalizowany wynik: dane dla modelu, tekst dla człowieka."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    # Dane strukturalne — to trafia do modelu (po obcięciu i oczyszczeniu).
    data: dict[str, Any] = Field(default_factory=dict)
    # Tekst dla człowieka: terminal teraz, GUI w Fazie 10.
    display: str = ""
    error: str = ""
    # Czy wynik zawiera treść z zewnątrz (sieć, plik, cudze dane). Router używa
    # tego do postawienia twardej bariery przed kolejnym wywołaniem.
    untrusted: bool = False

    @classmethod
    def success(
        cls,
        data: Mapping[str, Any] | None = None,
        *,
        display: str = "",
        untrusted: bool = False,
    ) -> ToolResult:
        return cls(ok=True, data=dict(data or {}), display=display, untrusted=untrusted)

    @classmethod
    def failure(cls, error: str, *, display: str = "") -> ToolResult:
        return cls(ok=False, error=error, display=display or error)

    def to_json(self) -> str:
        """Wynik w postaci, w jakiej widzi go model."""
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["data"] = self.data
        else:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)


class ToolSpec(BaseModel):
    """Metryczka narzędzia — wszystko, co router i model muszą o nim wiedzieć."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: type[BaseModel] = ToolArgs
    # Domyślnie CRITICAL, a nie SAFE: narzędzie, którego autor zapomniał opisać
    # ryzyka, ma być zablokowane, nie przepuszczone.
    risk: RiskLevel = RiskLevel.CRITICAL
    timeout_s: float = Field(default=DEFAULT_TOOL_TIMEOUT_S, gt=0.0, le=600.0)
    requires_network: bool = False
    idempotent: bool = True
    # Krótki opis dla człowieka (``/narzedzia``); puste = użyj ``description``.
    summary: str = ""

    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema argumentów — dokładnie to, co dostaje model."""
        schema = self.args_model.model_json_schema()
        # „title" z nazwy klasy Pydantic nic modelowi nie mówi, a zajmuje kontekst.
        schema.pop("title", None)
        schema.setdefault("type", "object")
        return schema

    def llm_schema(self) -> dict[str, Any]:
        """Deklaracja narzędzia w formacie ``tools`` Ollamy/OpenAI."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


@dataclass(slots=True)
class ToolContext:
    """Wszystko, co narzędzie dostaje od świata — i nic więcej.

    Nie ma tu ``os``, ``subprocess``, ``open`` ani klienta HTTP „na wszystko":
    kolejne fazy dodadzą wąskie, kontrolowane obiekty (klient HTTP z limitem
    rozmiaru, repozytoria bazy), a nie dostęp do systemu.

    ``now`` jest wstrzykiwanym zegarem: narzędzia pytają o czas przez niego, więc
    testy nie zależą od zegara maszyny ani od jej strefy czasowej.
    """

    settings: Settings = field(default_factory=get_settings)
    language: str = "en"
    dry_run: bool = False
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    logger: logging.Logger = field(default=logger)
    # Miejsce dla Fazy 8+: repozytoria bazy, klient HTTP z limitami, ścieżki.
    extras: dict[str, Any] = field(default_factory=dict)

    def localized(self, language: str) -> ToolContext:
        """Kopia kontekstu dla innego języka odpowiedzi."""
        return ToolContext(
            settings=self.settings,
            language=language,
            dry_run=self.dry_run,
            now=self.now,
            logger=self.logger,
            extras=dict(self.extras),
        )


@runtime_checkable
class Tool[ArgsT: BaseModel](Protocol):
    """Minimum, którego router wymaga od narzędzia.

    Parametr typu to model argumentów danego narzędzia. Dzięki niemu ciało
    ``run()`` widzi konkretne pola (``args.path``, ``args.pid``), a nie gołe
    ``BaseModel`` — router i tak waliduje argumenty modelem z ``spec.args_model``,
    więc do metody trafia dokładnie ten typ. Router pracuje na ``Tool[Any]``:
    nie musi wiedzieć, jakie argumenty ma konkretne narzędzie.
    """

    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, args: ArgsT, ctx: ToolContext) -> ToolResult: ...

    def dynamic_risk(self, args: ArgsT) -> RiskLevel:
        """Ryzyko po zajrzeniu w argumenty. Wolno tylko podnieść."""

    def normalize(self, args: ArgsT) -> ArgsT:
        """Kanonizacja argumentów (ścieżki, adresy) przed oceną polityki."""

    def confirmation(self, args: ArgsT, *, language: str = "en") -> ConfirmationRequest | None:
        """Treść pytania o zgodę — budowana przez NARZĘDZIE, nie przez model."""

    async def preview(self, args: ArgsT, ctx: ToolContext) -> str:
        """Co by się stało, gdyby narzędzie zadziałało (tryb próbny)."""

    def available(self) -> tuple[bool, str]:
        """Czy narzędzie da się uruchomić na TEJ maszynie (i dlaczego nie)."""


class BaseTool[ArgsT: BaseModel]:
    """Wygodna baza: sensowne domyślne zachowania wszystkich metod kontraktu.

    Podklasy podają swój model argumentów jako parametr typu
    (``class ListTool(BaseTool[ListArgs])``), więc metody widzą konkretne pola.
    """

    spec: ToolSpec

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    async def run(self, args: ArgsT, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def dynamic_risk(self, args: ArgsT) -> RiskLevel:
        """Domyślnie ryzyko statyczne. Nadpisując, wolno tylko eskalować."""
        return self.spec.risk

    def effective_risk(self, args: ArgsT) -> RiskLevel:
        """Poziom użyty przez router: eskalacja jest jednokierunkowa.

        Gdyby ``dynamic_risk`` zwróciło coś niższego niż deklaracja (błąd autora
        albo pomysł na obniżenie rygoru), wynik jest ignorowany.
        """
        return escalate(self.spec.risk, self.dynamic_risk(args))

    def normalize(self, args: ArgsT) -> ArgsT:
        return args

    def confirmation(self, args: ArgsT, *, language: str = "en") -> ConfirmationRequest | None:
        """Domyślne pytanie: nazwa narzędzia i argumenty wypisane wprost.

        Nie jest ozdobne, ale jest PRAWDZIWE — pokazuje dokładnie to, co pójdzie
        do wykonania. Narzędzia o wyższym ryzyku powinny to nadpisać własnym,
        zrozumiałym opisem skutków.
        """
        risk = self.effective_risk(args)
        summary = (
            f"Wywołanie narzędzia {self.spec.name}"
            if language == "pl"
            else f"Call the tool {self.spec.name}"
        )
        details = [f"{key} = {value!r}" for key, value in sorted(args.model_dump().items())]
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=risk,
            summary=summary,
            details=details or [("bez argumentów" if language == "pl" else "no arguments")],
            language=language,
        )

    async def preview(self, args: ArgsT, ctx: ToolContext) -> str:
        arguments = ", ".join(
            f"{key}={value!r}" for key, value in sorted(args.model_dump().items())
        )
        return f"{self.spec.name}({arguments})"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.spec.name!r}, risk={self.spec.risk.value})"


class FunctionTool(BaseTool[BaseModel]):
    """Narzędzie zbudowane z funkcji — zwykłej albo asynchronicznej.

    Funkcja synchroniczna jest uruchamiana w wątku roboczym, żeby nie blokowała
    pętli zdarzeń (a więc strumienia odpowiedzi i odtwarzania mowy).
    """

    def __init__(
        self,
        spec: ToolSpec,
        function: Callable[..., Any],
        *,
        risk_hook: Callable[[BaseModel], RiskLevel] | None = None,
        confirmation_hook: Callable[[BaseModel, str], ConfirmationRequest | None] | None = None,
        availability_hook: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(spec)
        self._function = function
        self._risk_hook = risk_hook
        self._confirmation_hook = confirmation_hook
        self._availability_hook = availability_hook
        self._is_async = inspect.iscoroutinefunction(function)

    async def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if self._is_async:
            outcome = await self._function(args, ctx)
        else:
            outcome = await asyncio.to_thread(self._function, args, ctx)
        if isinstance(outcome, ToolResult):
            return outcome
        if isinstance(outcome, Mapping):
            return ToolResult.success(outcome)
        # Zwykły tekst też jest poprawnym wynikiem — najczęstszy przypadek.
        return ToolResult.success({"value": outcome}, display=str(outcome))

    def dynamic_risk(self, args: BaseModel) -> RiskLevel:
        if self._risk_hook is None:
            return self.spec.risk
        return escalate(self.spec.risk, self._risk_hook(args))

    def confirmation(self, args: BaseModel, *, language: str = "en") -> ConfirmationRequest | None:
        if self._confirmation_hook is None:
            return super().confirmation(args, language=language)
        return self._confirmation_hook(args, language)

    def available(self) -> tuple[bool, str]:
        if self._availability_hook is None:
            return True, ""
        return self._availability_hook()


def make_tool(
    *,
    name: str,
    description: str,
    args_model: type[BaseModel],
    risk: RiskLevel,
    function: Callable[..., Any],
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    requires_network: bool = False,
    idempotent: bool = True,
    summary: str = "",
    risk_hook: Callable[[BaseModel], RiskLevel] | None = None,
    confirmation_hook: Callable[[BaseModel, str], ConfirmationRequest | None] | None = None,
    availability_hook: Callable[[], tuple[bool, str]] | None = None,
) -> FunctionTool:
    """Zbuduj narzędzie z funkcji ``(args, ctx) -> ToolResult | dict | str``.

    Ryzyko jest argumentem OBOWIĄZKOWYM i nazwanym — żeby nie dało się go podać
    przez przypadek na złej pozycji ani pominąć „bo na razie to tylko odczyt".
    """
    spec = ToolSpec(
        name=name,
        description=description,
        args_model=args_model,
        risk=risk,
        timeout_s=timeout_s,
        requires_network=requires_network,
        idempotent=idempotent,
        summary=summary,
    )
    return FunctionTool(
        spec,
        function,
        risk_hook=risk_hook,
        confirmation_hook=confirmation_hook,
        availability_hook=availability_hook,
    )


def describe_tools(tools: Sequence[Tool[Any]], *, language: str = "en") -> list[str]:
    """Lista narzędzi w jednej linijce każde — do ``/narzedzia`` i do logu."""
    lines: list[str] = []
    for item in tools:
        spec = item.spec
        note = spec.summary or spec.description
        lines.append(f"{spec.name}  [{spec.risk.value}]  {note}")
    return lines


__all__ = [
    "DEFAULT_RESULT_MAX_CHARS",
    "DEFAULT_TOOL_TIMEOUT_S",
    "BaseTool",
    "FunctionTool",
    "Tool",
    "ToolArgs",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "describe_tools",
    "make_tool",
]
