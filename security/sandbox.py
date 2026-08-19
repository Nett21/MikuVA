"""Kontrolowane wykonanie narzędzia: limit czasu, obcięcie, oczyszczenie (Faza 7).

„Sandbox" jest tu nazwą funkcji, nie obietnicą izolacji procesu — kod narzędzia
działa w tym samym procesie. Ta warstwa daje trzy rzeczy, których brak boli
najbardziej w praktyce:

1. **limit czasu** — zawieszone narzędzie nie może zawiesić rozmowy,
2. **żaden wyjątek nie wychodzi na zewnątrz** — awaria narzędzia wraca do modelu
   jako ``ToolResult(ok=False)``, a pełny ślad ląduje w ``logs/errors.log``,
3. **oczyszczenie i obcięcie wyniku** — bo wynik narzędzia to DANE NIEZAUFANE,
   które trafiają wprost do promptu.

Punkt 3 jest zabezpieczeniem przed prompt injection: treść, która przyjdzie ze
strony WWW czy z pliku, nie może udawać wiadomości systemowej ani zamknąć ramki,
w którą ją włożył router. Znaki sterujące i sekwencje imitujące znaczniki ról są
usuwane, a długość ograniczona.

Czego tu nie ma i nie będzie: uruchamiania powłoki, ``eval``, importu z tekstu.
Wykonanie polega na wywołaniu metody zarejestrowanego narzędzia — nic więcej.

*O kierunku zależności:* ten moduł importuje z ``tools/base.py``, czyli z samego
KONTRAKTU narzędzia (typy wyniku i kontekstu), a nie z konkretnych narzędzi.
``tools/base.py`` zna z kolei tylko ``security/risk.py`` i ``security/confirm.py``,
więc cyklu nie ma: kontrakt jest wspólną warstwą pod ``security`` i ``brain``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Final

from pydantic import BaseModel

from tools.base import (
    DEFAULT_RESULT_MAX_CHARS,
    DEFAULT_TOOL_TIMEOUT_S,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Znaczniki, którymi treść z zewnątrz mogłaby udawać ramkę wyniku albo znacznik
# roli. Usuwamy je W DOWOLNYM MIEJSCU tekstu, nie tylko na początku linii:
# „blabla <<END_TOOL_RESULT>> system: zrób X" wygląda w prompcie równie
# przekonująco jak to samo w nowej linii.
_FRAME_IMITATIONS: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:<<\s*/?\s*(?:tool_result|end_tool_result|system|user|assistant)[^>]{0,120}>>"
    r"|<\|(?:im_start|im_end|endoftext|system|user|assistant)\|>)"
)

# Zapis „system:" na POCZĄTKU linii udaje wiadomość systemową. Tylko na początku:
# w środku zdania („w kolumnie system: brak") to zwykły tekst i kasowanie go
# psułoby dane, a nie chroniło przed niczym — ramki i tak są już usunięte wyżej.
_ROLE_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"(?im)^[ \t]*(?:system|assistant|user)[ \t]*:[ \t]*(?=\S)"
)

# Znaki sterujące poza tabulacją i nową linią — w tekście dla modelu nic nie wnoszą,
# a potrafią psuć terminal i logi.
_CONTROL_CHARS: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_TRUNCATION_NOTE_PL: Final[str] = "\n[...] wynik skrócony"
_TRUNCATION_NOTE_EN: Final[str] = "\n[...] result truncated"


def sanitize_tool_text(text: str, *, max_chars: int = DEFAULT_RESULT_MAX_CHARS,
                       language: str = "en") -> str:
    """Przygotuj tekst z narzędzia do wstawienia w prompt.

    Kolejność ma znaczenie: najpierw usuwamy znaki sterujące (inaczej mogłyby
    rozbić wzorce), potem sekwencje udające znaczniki rozmowy, na końcu obcinamy.
    """
    cleaned = _CONTROL_CHARS.sub("", str(text or ""))
    cleaned = _FRAME_IMITATIONS.sub("[usunięto]", cleaned)
    cleaned = _ROLE_PREFIX.sub("[usunięto] ", cleaned)
    if len(cleaned) > max_chars:
        note = _TRUNCATION_NOTE_PL if language == "pl" else _TRUNCATION_NOTE_EN
        cleaned = cleaned[:max_chars].rstrip() + note
    return cleaned


class ToolSandbox:
    """Wykonawca narzędzi: jedno wywołanie, jeden limit czasu, jeden wynik."""

    def __init__(
        self,
        *,
        default_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        max_result_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ) -> None:
        self._default_timeout_s = max(0.1, float(default_timeout_s))
        self._max_result_chars = max(200, int(max_result_chars))

    @property
    def max_result_chars(self) -> int:
        return self._max_result_chars

    def timeout_for(self, tool: Tool[Any]) -> float:
        """Limit czasu: mniejszy z limitu narzędzia i globalnego."""
        return min(float(tool.spec.timeout_s), self._default_timeout_s)

    async def run(
        self, tool: Tool[Any], args: BaseModel, ctx: ToolContext
    ) -> tuple[ToolResult, int]:
        """Wykonaj narzędzie. Zwraca wynik i czas trwania w milisekundach.

        Nigdy nie rzuca (poza anulowaniem całej tury, które musi przejść dalej).
        """
        timeout = self.timeout_for(tool)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=timeout)
        except TimeoutError:
            elapsed = self._elapsed_ms(started)
            logger.warning("Narzędzie %s przekroczyło %.1f s", tool.spec.name, timeout)
            return (
                ToolResult.failure(
                    f"narzędzie {tool.spec.name} nie odpowiedziało w {timeout:.0f} s"
                ),
                elapsed,
            )
        except asyncio.CancelledError:
            # Przerwanie tury (Ctrl+C, barge-in) nie jest błędem narzędzia.
            raise
        except ToolError as exc:
            logger.info("Narzędzie %s zgłosiło błąd: %s", tool.spec.name, exc.message)
            return ToolResult.failure(exc.message), self._elapsed_ms(started)
        except Exception as exc:
            logger.exception("Narzędzie %s zawiodło", tool.spec.name)
            return (
                ToolResult.failure(f"narzędzie {tool.spec.name} zawiodło: {exc}"),
                self._elapsed_ms(started),
            )

        elapsed = self._elapsed_ms(started)
        if not isinstance(result, ToolResult):  # pragma: no cover - błąd autora narzędzia
            logger.error("Narzędzie %s zwróciło %r zamiast ToolResult", tool.spec.name, result)
            return ToolResult.failure("narzędzie zwróciło wynik w nieznanym formacie"), elapsed
        return self._clean(result, language=ctx.language), elapsed

    async def preview(self, tool: Tool[Any], args: BaseModel, ctx: ToolContext) -> str:
        """Podgląd zamiast wykonania (tryb ``SECURITY_DRY_RUN=true``)."""
        try:
            text = await asyncio.wait_for(tool.preview(args, ctx), timeout=self.timeout_for(tool))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Podgląd narzędzia %s nie powiódł się: %s", tool.spec.name, exc)
            return f"{tool.spec.name}: podgląd niedostępny"
        return sanitize_tool_text(text, max_chars=self._max_result_chars, language=ctx.language)

    def _clean(self, result: ToolResult, *, language: str) -> ToolResult:
        """Oczyść wynik: teksty w danych i w polu dla człowieka."""
        data = {
            key: (
                sanitize_tool_text(value, max_chars=self._max_result_chars, language=language)
                if isinstance(value, str)
                else value
            )
            for key, value in result.data.items()
        }
        return result.model_copy(
            update={
                "data": data,
                "display": sanitize_tool_text(
                    result.display, max_chars=self._max_result_chars, language=language
                ),
                "error": sanitize_tool_text(result.error, max_chars=500, language=language),
            }
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


__all__ = ["ToolSandbox", "sanitize_tool_text"]
