"""Wiadomości — kanały RSS/Atom, bez klucza API (Faza 9).

============== ======== ====================================================
narzędzie      poziom   uzasadnienie
============== ======== ====================================================
``news.headlines`` MEDIUM  ruch wychodzący, treść pisana przez kogoś innego
``news.search``    MEDIUM  to samo, tylko z zapytaniem
============== ======== ====================================================

**RSS, a nie API z kluczem** — z trzech powodów: działa bez rejestracji, nie ma
limitu zapytań na darmowym planie i nie wiąże asystenta z jednym dostawcą.
Kanały podaje się w ``NEWS_FEEDS``; szukanie idzie przez ``NEWS_SEARCH_URL``,
w którym podstawiamy ``{query}`` i ``{language}``. Domyślnie to kanał Google News
— **jeden wpis w ``.env`` wymienia go na dowolny inny serwis**, bez zmiany kodu.

``NEWS_API_KEY`` jest w konfiguracji dla dostawców, którzy go wymagają, ale żaden
z domyślnych wariantów go nie potrzebuje. Klucz nigdy nie trafia do logu ani do
wyniku narzędzia.

Nie streszczamy tu nagłówków: narzędzie zwraca tytuły, źródła, daty i krótkie
opisy, a wnioski wyciąga model. Wynik jest oznaczony jako niezaufany — nagłówek
też jest cudzym tekstem.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote_plus

import httpx
from pydantic import Field

from config import Settings, get_settings
from host.http import HttpPolicy, NetworkError, fetch, network_available, public_hosts
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from tools.webtext import ContentError, FeedItem, parse_feed, split_list
from i18n import t

logger = logging.getLogger(__name__)


class HeadlinesArgs(ToolArgs):
    limit: int = Field(default=8, ge=1, le=30)
    feed: str = Field(default="", max_length=300)


class NewsSearchArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=200)
    limit: int = Field(default=8, ge=1, le=30)


class _NewsTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza narzędzi wiadomości: pobranie kanału i zamiana błędów na komunikaty."""

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

    def available(self) -> tuple[bool, str]:
        return network_available(self._settings)

    @property
    def feeds(self) -> list[str]:
        return split_list(self._settings.news_feeds)

    async def _items(self, url: str, limit: int) -> list[FeedItem]:
        try:
            response = await fetch(
                url,
                policy=HttpPolicy.from_settings(self._settings),
                accept="application/rss+xml, application/atom+xml, application/xml, text/xml",
                client=self._client,
            )
        except NetworkError as exc:
            raise ToolError(exc.user_message) from exc

        source = ", ".join(public_hosts([url])) or url
        try:
            return parse_feed(response.text, limit=limit, source=source)
        except ContentError as exc:
            raise ToolError(t("news.feed_error", name=source, error=exc.message)) from exc


class HeadlinesTool(_NewsTool[HeadlinesArgs]):
    """``news.headlines`` — najnowsze wpisy z kanałów użytkownika."""

    def available(self) -> tuple[bool, str]:
        usable, reason = super().available()
        if not usable:
            return usable, reason
        if not self.feeds:
            return False, (
                "nie skonfigurowano żadnego kanału wiadomości — dodaj adresy RSS do "
                "NEWS_FEEDS w .env (news.search działa bez tego)"
            )
        return True, ""

    async def run(self, args: HeadlinesArgs, ctx: ToolContext) -> ToolResult:
        chosen = [args.feed.strip()] if args.feed.strip() else self.feeds
        if not chosen:
            raise ToolError(t("news.no_feeds"))

        limit = min(args.limit, self._settings.news_max_items)
        per_feed = max(1, limit // max(1, len(chosen)))
        collected: list[FeedItem] = []
        problems: list[str] = []

        for url in chosen:
            try:
                collected.extend(await self._items(url, per_feed))
            except ToolError as exc:
                # Jeden padnięty kanał nie może zabrać wszystkich pozostałych.
                problems.append(exc.message)

        if not collected:
            raise ToolError(
                "nie udało się pobrać żadnego kanału: " + "; ".join(problems[:3])
                if problems
                else "kanały nie zwróciły wpisów"
            )

        items = [item.to_dict() for item in collected[:limit]]
        headline = "; ".join(item["title"][:70] for item in items[:3])
        data: dict[str, Any] = {
            "count": len(items),
            "items": items,
            "sources": public_hosts(chosen),
        }
        if problems:
            data["problems"] = problems[:3]
        return ToolResult.success(
            data,
            display=t("news.headlines", count=len(items), sources=headline),
            untrusted=True,
        )


class NewsSearchTool(_NewsTool[NewsSearchArgs]):
    """``news.search`` — wiadomości na wskazany temat."""

    def available(self) -> tuple[bool, str]:
        usable, reason = super().available()
        if not usable:
            return usable, reason
        if not self._settings.news_search_url.strip():
            return False, "nie ustawiono adresu szukania wiadomości (NEWS_SEARCH_URL)"
        return True, ""

    def _url(self, query: str, language: str) -> str:
        template = self._settings.news_search_url.strip()
        # ``format`` na cudzym szablonie mogłoby wywalić się na nawiasach w adresie,
        # więc podstawiamy dosłownie i tylko te dwa pola.
        return template.replace("{query}", quote_plus(query)).replace(
            "{language}", "pl" if language == "pl" else "en"
        )

    async def run(self, args: NewsSearchArgs, ctx: ToolContext) -> ToolResult:
        limit = min(args.limit, self._settings.news_max_items)
        url = self._url(args.query, ctx.language)
        items = [item.to_dict() for item in await self._items(url, limit)]
        headline = "; ".join(item["title"][:70] for item in items[:3])
        return ToolResult.success(
            {
                "query": args.query,
                "count": len(items),
                "items": items,
                "sources": public_hosts([url]),
            },
            display=t("news.search_results", query=args.query, count=len(items), sources=headline),
            untrusted=True,
        )


def build_news_tools(
    settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
) -> Sequence[Tool[Any]]:
    """Narzędzia wiadomości (kanały RSS, bez klucza API)."""
    active = settings or get_settings()
    feeds = split_list(active.news_feeds)
    hint = (
        f" Configured feeds: {', '.join(public_hosts(feeds))}."
        if feeds
        else " No feeds configured yet; news.search still works."
    )
    return (
        HeadlinesTool(
            ToolSpec(
                name="news.headlines",
                description=(
                    "Latest headlines from the user's configured news feeds, with titles, "
                    "sources, dates and short summaries." + hint
                ),
                summary=t("spec.news_headlines"),
                args_model=HeadlinesArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(90.0, active.web_timeout_s + 20.0),
            ),
            settings=active,
            client=client,
        ),
        NewsSearchTool(
            ToolSpec(
                name="news.search",
                description=(
                    "Search recent news for a topic and return titles, sources, dates and "
                    "short summaries. Use it when the user asks what happened or what is new."
                ),
                summary=t("spec.news_search"),
                args_model=NewsSearchArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(90.0, active.web_timeout_s + 20.0),
            ),
            settings=active,
            client=client,
        ),
    )


__all__ = [
    "HeadlinesArgs",
    "HeadlinesTool",
    "NewsSearchArgs",
    "NewsSearchTool",
    "build_news_tools",
]
