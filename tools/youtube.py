"""YouTube: szukanie, transkrypcje, odtwarzanie (Faza 9).

================== ======== =================================================
narzędzie          poziom   uzasadnienie
================== ======== =================================================
``youtube.search``  MEDIUM  zapytanie do API, nic nie zmienia
``youtube.transcript`` MEDIUM  pobiera tekst napisów (dane niezaufane)
``youtube.play``    HIGH    otwiera odtwarzacz na ekranie użytkownika
================== ======== =================================================

``youtube.play`` jest HIGH, a nie MEDIUM, i to jest świadome: to jedyne narzędzie
tej fazy, które **coś robi na ekranie** — przerywa to, co użytkownik miał otwarte,
i zaczyna odtwarzać dźwięk. Zgoda człowieka jest tu na miejscu, a pytanie pokazuje
tytuł i adres, więc widać, co się włączy.

**Szukanie wymaga klucza** (``YOUTUBE_API_KEY``, Data API v3) — nie ma sensownego
wariantu bez klucza: strona wyników jest aplikacją JavaScriptu, a jej „skrobanie"
psuje się przy każdej zmianie układu. Bez klucza narzędzie jest niedostępne, a
model dostaje w komunikacie podpowiedź, żeby użyć ``web.search``.

**Transkrypcje działają bez klucza**: publiczny punkt końcowy napisów
(``timedtext``) zwraca XML z tekstem. Gdy film nie ma napisów albo są wyłączone,
mówimy to wprost, zamiast udawać, że coś się nie udało.
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
from host.apps import LaunchError, has_graphical_session, open_target, session_label
from host.http import (
    HttpPolicy,
    NetworkError,
    fetch,
    fetch_json,
    network_available,
    secret_value,
)
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from tools.webtext import clip, normalize_date, strip_tags
from i18n import t

logger = logging.getLogger(__name__)

_SEARCH_URL: Final[str] = "https://www.googleapis.com/youtube/v3/search"
_TIMEDTEXT_URL: Final[str] = "https://www.youtube.com/api/timedtext"
_WATCH_URL: Final[str] = "https://www.youtube.com/watch?v="

# Identyfikator filmu: 11 znaków z bezpiecznego alfabetu base64url.
_VIDEO_ID: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Hosty, z których przyjmujemy adresy filmów.
_YOUTUBE_HOSTS: Final[frozenset[str]] = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
)


class SearchArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=200)
    limit: int = Field(default=5, ge=1, le=25)


class TranscriptArgs(ToolArgs):
    video: str = Field(min_length=5, max_length=200)
    language: str = Field(default="", max_length=10)
    max_chars: int | None = Field(default=None, ge=200, le=100_000)


class PlayArgs(ToolArgs):
    video: str = Field(min_length=5, max_length=200)


def video_id(value: str) -> str:
    """Wyłuskaj identyfikator filmu z adresu albo przyjmij gotowy.

    Obsługiwane zapisy: ``dQw4w9WgXcQ``, ``youtube.com/watch?v=…``,
    ``youtu.be/…``, ``youtube.com/shorts/…``, ``/embed/…``. Adres z innego hosta
    jest odrzucany — to narzędzie jest do YouTube'a, a nie do dowolnych stron.
    """
    raw = str(value or "").strip()
    if not raw:
        raise ToolError("nie podano filmu")
    if _VIDEO_ID.match(raw):
        return raw

    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise ToolError(f"'{raw}' nie jest adresem filmu") from exc

    host = (parts.hostname or "").lower().removeprefix("www.")
    if host and host not in {item.removeprefix("www.") for item in _YOUTUBE_HOSTS}:
        raise ToolError(f"'{host}' to nie YouTube — podaj adres filmu z YouTube'a")

    if host == "youtu.be":
        found = parts.path.strip("/").split("/")[0]
    elif "v" in parse_qs(parts.query):
        found = parse_qs(parts.query)["v"][0]
    else:
        segments = [item for item in parts.path.split("/") if item]
        found = segments[-1] if segments else ""

    if not _VIDEO_ID.match(found):
        raise ToolError(t("yt.no_video_id", value=raw))
    return found


def watch_url(identifier: str) -> str:
    return f"{_WATCH_URL}{identifier}"


class _YouTubeTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza: sieć, klucz API i zamiana błędów na komunikaty dla modelu."""

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
    def api_key(self) -> str:
        """Klucz API. Zwracany wyłącznie do wysłania w żądaniu — nigdy do logu."""
        return secret_value(self._settings.youtube_api_key).strip()

    def policy(self) -> HttpPolicy:
        return HttpPolicy.from_settings(self._settings)


class YouTubeSearchTool(_YouTubeTool[SearchArgs]):
    """``youtube.search`` — filmy pasujące do zapytania (wymaga klucza API)."""

    def available(self) -> tuple[bool, str]:
        usable, reason = super().available()
        if not usable:
            return usable, reason
        if not self.api_key:
            return False, (
                "szukanie na YouTube wymaga klucza Data API v3 w YOUTUBE_API_KEY "
                "(bez niego zostaje web.search)"
            )
        return True, ""

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:
        limit = min(args.limit, self._settings.youtube_max_results)
        try:
            payload = await fetch_json(
                _SEARCH_URL,
                policy=self.policy(),
                params={
                    "part": "snippet",
                    "type": "video",
                    "maxResults": limit,
                    "q": args.query,
                    # Klucz idzie w parametrze „key" — a ``redact_url`` z host/http.py
                    # zamazuje właśnie takie parametry przed zapisem do logu.
                    "key": self.api_key,
                },
                client=self._client,
            )
        except NetworkError as exc:
            raise ToolError(exc.user_message) from exc

        entries = payload.get("items") if isinstance(payload, dict) else None
        videos: list[dict[str, str]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            identifier = str((entry.get("id") or {}).get("videoId") or "")
            snippet = entry.get("snippet") or {}
            if not identifier:
                continue
            videos.append(
                {
                    "id": identifier,
                    "title": strip_tags(str(snippet.get("title") or ""), max_chars=200),
                    "channel": strip_tags(str(snippet.get("channelTitle") or ""), max_chars=100),
                    "published": normalize_date(str(snippet.get("publishedAt") or "")),
                    "url": watch_url(identifier),
                }
            )
            if len(videos) >= limit:
                break

        if not videos:
            raise ToolError(t("yt.no_results", query=args.query))
        listing = "; ".join(f"{item['title'][:60]} ({item['channel']})" for item in videos[:3])
        return ToolResult.success(
            {"query": args.query, "count": len(videos), "videos": videos},
            display=t("yt.results", query=args.query, count=len(videos), names=listing),
            untrusted=True,
        )


class YouTubeTranscriptTool(_YouTubeTool[TranscriptArgs]):
    """``youtube.transcript`` — tekst napisów filmu (bez klucza API)."""

    async def run(self, args: TranscriptArgs, ctx: ToolContext) -> ToolResult:
        identifier = video_id(args.video)
        language = (args.language or ("pl" if ctx.language == "pl" else "en")).strip()
        limit = min(args.max_chars or self._settings.web_max_chars, self._settings.web_max_chars)

        text = await self._timedtext(identifier, language)
        if not text and language != "en":
            # Napisy w języku rozmowy bywają nieobecne — angielskie zwykle są.
            text = await self._timedtext(identifier, "en")
        if not text:
            raise ToolError(
                t("yt.no_transcript", id=identifier)
            )

        clipped, truncated = clip(text, limit, note="[...] transkrypcja obcięta")
        return ToolResult.success(
            {
                "video": identifier,
                "url": watch_url(identifier),
                "language": language,
                "truncated": truncated,
                "text": clipped,
            },
            display=t("yt.transcript", id=identifier, chars=len(clipped)),
            untrusted=True,
        )

    async def _timedtext(self, identifier: str, language: str) -> str:
        """Napisy z publicznego punktu końcowego. Pusta odpowiedź = brak napisów."""
        try:
            response = await fetch(
                _TIMEDTEXT_URL,
                policy=self.policy(),
                params={"v": identifier, "lang": language},
                accept="text/xml, application/xml",
                client=self._client,
            )
        except NetworkError as exc:
            raise ToolError(exc.user_message) from exc
        return _timedtext_to_plain(response.text)


class YouTubePlayTool(_YouTubeTool[PlayArgs]):
    """``youtube.play`` — otwórz film w przeglądarce. Wymaga zgody użytkownika."""

    def __init__(
        self,
        spec: ToolSpec,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        runner: Any | None = None,
        opener: Any | None = None,
    ) -> None:
        super().__init__(spec, settings=settings, client=client)
        self._runner = runner
        self._opener = opener

    def available(self) -> tuple[bool, str]:
        usable, reason = super().available()
        if not usable:
            return usable, reason
        if not has_graphical_session():
            return False, f"brak sesji graficznej ({session_label()}) — nie ma gdzie odtworzyć"
        return True, ""

    def confirmation(self, args: PlayArgs, *, language: str = "en") -> ConfirmationRequest | None:
        polish = language == "pl"
        try:
            identifier = video_id(args.video)
        except ToolError:
            return super().confirmation(args, language=language)
        summary = (
            f"Włączy film w przeglądarce: {watch_url(identifier)}"
            if polish
            else f"Open a video in the browser: {watch_url(identifier)}"
        )
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=RiskLevel.HIGH,
            summary=summary,
            details=[
                (
                    "przerwie to, co jest teraz na ekranie, i zacznie odtwarzać dźwięk"
                    if polish
                    else "this takes over the screen and starts audio"
                )
            ],
            language=language,
        )

    async def run(self, args: PlayArgs, ctx: ToolContext) -> ToolResult:
        identifier = video_id(args.video)
        url = watch_url(identifier)
        try:
            note = open_target(url, runner=self._runner, opener=self._opener)
        except LaunchError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success({"video": identifier, "url": url}, display=note)

    async def preview(self, args: PlayArgs, ctx: ToolContext) -> str:
        try:
            return f"otworzyłoby {watch_url(video_id(args.video))}"
        except ToolError as exc:
            return f"odmowa: {exc.message}"


def _timedtext_to_plain(content: str) -> str:
    """XML napisów → zwykły tekst.

    Świadomie regexem, nie parserem XML: to jeden ustalony format o jednym
    znaczniku (``<text start=… dur=…>``), a plik z internetu i tak przechodzi
    przez limit rozmiaru w ``host/http.py``. Parser XML dałby tu tylko większą
    powierzchnię na dokumenty złośliwie zagnieżdżone.
    """
    raw = str(content or "")
    if "<text" not in raw:
        return ""
    fragments = re.findall(r"<text[^>]*>(.*?)</text>", raw, re.DOTALL)
    lines = [strip_tags(fragment, max_chars=500) for fragment in fragments]
    return "\n".join(line for line in lines if line).strip()


def build_youtube_tools(
    settings: Settings | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    runner: Any | None = None,
    opener: Any | None = None,
) -> Sequence[Tool[Any]]:
    """Narzędzia YouTube. ``runner``/``opener`` podstawiają atrapy w testach."""
    active = settings or get_settings()
    return (
        YouTubeSearchTool(
            ToolSpec(
                name="youtube.search",
                description=(
                    "Search YouTube for videos and return titles, channels and links. "
                    "Requires a YouTube Data API key."
                ),
                summary=t("spec.yt_search"),
                args_model=SearchArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(60.0, active.web_timeout_s + 10.0),
            ),
            settings=active,
            client=client,
        ),
        YouTubeTranscriptTool(
            ToolSpec(
                name="youtube.transcript",
                description=(
                    "Read the subtitles of a YouTube video as text, so you can summarise or "
                    "quote it. Accepts a video id or any YouTube link. No API key needed."
                ),
                summary="transkrypcja filmu z YouTube",
                args_model=TranscriptArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=min(90.0, active.web_timeout_s + 20.0),
            ),
            settings=active,
            client=client,
        ),
        YouTubePlayTool(
            ToolSpec(
                name="youtube.play",
                description=(
                    "Open a YouTube video in the user's browser. This takes over the screen "
                    "and starts audio, so it requires the user's confirmation."
                ),
                summary="odtworzenie filmu (wymaga zgody)",
                args_model=PlayArgs,
                risk=RiskLevel.HIGH,
                requires_network=True,
                timeout_s=20.0,
                idempotent=False,
            ),
            settings=active,
            client=client,
            runner=runner,
            opener=opener,
        ),
    )


__all__ = [
    "PlayArgs",
    "SearchArgs",
    "TranscriptArgs",
    "YouTubePlayTool",
    "YouTubeSearchTool",
    "YouTubeTranscriptTool",
    "build_youtube_tools",
    "video_id",
    "watch_url",
]
