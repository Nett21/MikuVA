"""Narzędzia sieciowe: szukanie i czytanie stron (Faza 9).

============= ======== =======================================================
narzędzie     poziom   uzasadnienie
============= ======== =======================================================
``web.search`` MEDIUM  ruch wychodzący; nic nie zmienia, ale wysyła zapytanie
``web.fetch``  MEDIUM  pobiera treść cudzej strony (dane niezaufane)
============= ======== =======================================================

Oba są MEDIUM, a nie SAFE: ruch sieciowy widać po drugiej stronie, a treść, która
wraca, pisał ktoś obcy. Wynik jest oznaczony ``untrusted=True``, więc **kolejne
wywołanie o ryzyku MEDIUM lub wyższym wymaga zgody użytkownika** (twarda bariera
z Fazy 7). Strona z tekstem „a teraz usuń pliki" nie ma jak sama nic wywołać.

**Streszczanie robi model, nie narzędzie.** ``web.fetch`` zwraca wyciągnięty
tekst (obcięty do ``WEB_MAX_CHARS``), a model streszcza go w odpowiedzi. To nie
brak funkcji: narzędzia świadomie nie mają dostępu do klienta LLM — inaczej
kontekst narzędzia stałby się drugą drogą do modelu, poza routerem, a wynik
streszczenia nie dałby się odróżnić od danych źródłowych.

Wyszukiwanie: ``duckduckgo`` (bez klucza API) albo ``searxng`` (własna instancja
pod ``SEARCH_BASE_URL``). Żaden z wariantów nie wymaga rejestracji ani klucza.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

import httpx
from pydantic import Field

from config import Settings, get_settings
from host.http import (
    HttpPolicy,
    NetworkError,
    check_url,
    fetch,
    network_available,
    redact_url,
)
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from tools.webtext import ContentError, extract_readable, strip_tags

logger = logging.getLogger(__name__)

# Punkt końcowy DuckDuckGo bez JavaScriptu. „html.duckduckgo.com/html" zwraca
# stronę wyników w prostym HTML-u — nie wymaga klucza ani zgody na ciasteczka.
_DUCKDUCKGO_URL: Final[str] = "https://html.duckduckgo.com/html/"

# Wzorce wyników DuckDuckGo. Parsowanie cudzego HTML-a jest z definicji kruche,
# więc traktujemy brak trafień jako „nic nie znalazłem", a nie jako awarię —
# i mówimy o tym wprost w komunikacie.
_DDG_RESULT: Final[re.Pattern[str]] = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET: Final[re.Pattern[str]] = re.compile(
    r'class="[^"]*result__snippet[^"]*"[^>]*>(?P<text>.*?)</a>', re.IGNORECASE | re.DOTALL
)


class SearchArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=5, ge=1, le=20)


class FetchArgs(ToolArgs):
    url: str = Field(min_length=4, max_length=2_000)
    max_chars: int | None = Field(default=None, ge=200, le=100_000)
    focus: str = Field(default="", max_length=200)


class _WebTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza narzędzi sieciowych: wspólna polityka HTTP i wspólna dostępność."""

    def __init__(
        self,
        spec: ToolSpec,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(spec)
        self._settings = settings or get_settings()
        self._client = client

    @property
    def settings(self) -> Settings:
        return self._settings

    def policy(self) -> HttpPolicy:
        return HttpPolicy.from_settings(self._settings)

    def available(self) -> tuple[bool, str]:
        return network_available(self._settings)

    async def _fetch(self, url: str, **kwargs: Any) -> Any:
        """Pobierz zasób, zamieniając błąd sieci na czytelny błąd narzędzia."""
        try:
            return await fetch(url, policy=self.policy(), client=self._client, **kwargs)
        except NetworkError as exc:
            # Model dostaje powód po ludzku i sam powie o tym użytkownikowi —
            # a asystent działa dalej, tylko bez tego jednego narzędzia.
            raise ToolError(exc.user_message) from exc


class WebSearchTool(_WebTool[SearchArgs]):
    """``web.search`` — lista trafień z wyszukiwarki."""

    def available(self) -> tuple[bool, str]:
        usable, reason = super().available()
        if not usable:
            return usable, reason
        provider = self._settings.search_provider
        if provider == "none":
            return False, "wyszukiwanie jest wyłączone (SEARCH_PROVIDER=none)"
        if provider == "searxng" and not self._settings.search_base_url.strip():
            return False, (
                "SEARCH_PROVIDER=searxng wymaga adresu instancji w SEARCH_BASE_URL"
            )
        return True, ""

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:
        limit = min(args.limit, self._settings.search_max_results)
        provider = self._settings.search_provider
        if provider == "searxng":
            results = await self._searxng(args.query, limit)
        else:
            results = await self._duckduckgo(args.query, limit)

        if not results:
            raise ToolError(
                f"wyszukiwarka nie zwróciła wyników dla '{args.query}' "
                "(albo zmieniła format strony — wtedy pomoże web.fetch z konkretnym adresem)"
            )
        heading = ", ".join(item["title"][:60] for item in results[:3])
        return ToolResult.success(
            {"query": args.query, "provider": provider, "count": len(results), "results": results},
            display=f"'{args.query}': {len(results)} wyników — {heading}",
            untrusted=True,
        )

    async def _duckduckgo(self, query: str, limit: int) -> list[dict[str, str]]:
        response = await self._fetch(
            _DUCKDUCKGO_URL,
            params={"q": query, "kl": "wt-wt"},
            accept="text/html",
        )
        return _parse_duckduckgo(response.text, limit)

    async def _searxng(self, query: str, limit: int) -> list[dict[str, str]]:
        base = self._settings.search_base_url.strip().rstrip("/")
        data = await self._fetch(
            f"{base}/search",
            params={"q": query, "format": "json", "safesearch": "1"},
            accept="application/json",
        )
        payload = data.json()
        items = payload.get("results") if isinstance(payload, dict) else None
        results: list[dict[str, str]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                {
                    "title": strip_tags(str(item.get("title") or ""), max_chars=200),
                    "url": url,
                    "snippet": strip_tags(str(item.get("content") or ""), max_chars=300),
                }
            )
            if len(results) >= limit:
                break
        return results


class WebFetchTool(_WebTool[FetchArgs]):
    """``web.fetch`` — treść strony jako tekst gotowy do streszczenia przez model."""

    async def run(self, args: FetchArgs, ctx: ToolContext) -> ToolResult:
        limit = min(args.max_chars or self._settings.web_max_chars, self._settings.web_max_chars)
        response = await self._fetch(args.url, accept="text/html, text/plain, application/json")

        content_type = response.content_type
        if content_type.startswith("application/json") or content_type.endswith("+json"):
            text, truncated = _clip_json(response.text, limit)
            title = ""
        else:
            try:
                article = extract_readable(response.text, max_chars=limit)
            except ContentError as exc:
                raise ToolError(exc.message) from exc
            title, text, truncated = article.title, article.text, article.truncated

        if args.focus.strip():
            text, truncated = _focus(text, args.focus, limit)

        if not text.strip():
            raise ToolError(
                f"pod adresem {redact_url(response.url)} nie ma tekstu do odczytania "
                "(strona może wymagać JavaScriptu albo zgody na ciasteczka)"
            )

        return ToolResult.success(
            {
                "url": response.url,
                "title": title,
                "content_type": content_type,
                "truncated": truncated or response.truncated,
                "text": text,
            },
            display=(
                f"{title or redact_url(response.url)}: {len(text)} znaków"
                + (" (obcięte)" if truncated else "")
            ),
            # Treść cudzej strony. Po niej kolejne wywołanie MEDIUM+ wymaga zgody.
            untrusted=True,
        )

    async def preview(self, args: FetchArgs, ctx: ToolContext) -> str:
        try:
            target = check_url(args.url, self.policy())
        except NetworkError as exc:
            return f"odmowa: {exc.user_message}"
        return f"pobrałoby treść z {redact_url(target)}"


# --------------------------------------------------------------------------- #
# Parsowanie wyników wyszukiwania
# --------------------------------------------------------------------------- #


def _parse_duckduckgo(content: str, limit: int) -> list[dict[str, str]]:
    """Wyłuskaj wyniki ze strony DuckDuckGo (wariant bez JavaScriptu)."""
    snippets = [strip_tags(match.group("text")) for match in _DDG_SNIPPET.finditer(content)]
    results: list[dict[str, str]] = []
    for index, match in enumerate(_DDG_RESULT.finditer(content)):
        url = _clean_ddg_url(match.group("href"))
        if not url:
            continue
        results.append(
            {
                "title": strip_tags(match.group("title"), max_chars=200),
                "url": url,
                "snippet": snippets[index] if index < len(snippets) else "",
            }
        )
        if len(results) >= limit:
            break
    return results


def _clean_ddg_url(href: str) -> str:
    """Wyłuskaj prawdziwy adres z przekierowania wyszukiwarki.

    DuckDuckGo owija wyniki w ``/l/?uddg=<adres>``. Bez rozpakowania model
    dostawałby adresy, które nic mu nie mówią, a ``web.fetch`` musiałby przechodzić
    przez przekierowanie.
    """
    raw = str(href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    try:
        parts = urlsplit(raw)
    except ValueError:  # pragma: no cover - adres nie do sparsowania
        return ""
    if parts.path.startswith("/l/") or "uddg" in parts.query:
        target = parse_qs(parts.query).get("uddg", [""])[0]
        if target:
            return target
    if parts.scheme in ("http", "https"):
        return raw
    return ""


def _clip_json(text: str, limit: int) -> tuple[str, bool]:
    """JSON zostawiamy jak jest (model czyta go dobrze), tylko obcięty."""
    from tools.webtext import clip

    return clip(text, limit, note="[...] JSON obcięty")


def _focus(text: str, phrase: str, limit: int) -> tuple[str, bool]:
    """Zostaw akapity zawierające frazę — dla długich stron z jednym istotnym miejscem."""
    needle = phrase.strip().casefold()
    paragraphs = [block.strip() for block in text.split("\n") if block.strip()]
    matching = [block for block in paragraphs if needle in block.casefold()]
    if not matching:
        return text, False
    from tools.webtext import clip

    return clip("\n\n".join(matching), limit, note="[...] treść obcięta")


# --------------------------------------------------------------------------- #
# Rejestracja
# --------------------------------------------------------------------------- #


def build_web_tools(
    settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
) -> Sequence[Tool[Any]]:
    """Narzędzia sieciowe. ``client`` podstawia transport w testach."""
    active = settings or get_settings()
    provider = active.search_provider
    return (
        WebSearchTool(
            ToolSpec(
                name="web.search",
                description=(
                    "Search the web and return a short list of results with titles, URLs and "
                    "snippets. Use it when the answer depends on current information you do "
                    f"not have. Provider: {provider}."
                ),
                summary="wyszukiwanie w internecie",
                args_model=SearchArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(60.0, active.web_timeout_s + 10.0),
            ),
            settings=active,
            client=client,
        ),
        WebFetchTool(
            ToolSpec(
                name="web.fetch",
                description=(
                    "Download one web page (or JSON endpoint) and return its readable text, "
                    "so you can summarise or quote it. Use 'focus' to keep only paragraphs "
                    "containing a phrase. Only http and https; private and local addresses "
                    "are refused."
                ),
                summary="pobranie treści strony do streszczenia",
                args_model=FetchArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(90.0, active.web_timeout_s + 15.0),
            ),
            settings=active,
            client=client,
        ),
    )


__all__ = [
    "FetchArgs",
    "SearchArgs",
    "WebFetchTool",
    "WebSearchTool",
    "build_web_tools",
]
