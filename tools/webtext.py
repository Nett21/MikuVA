"""Treść z sieci → tekst dla modelu: HTML i kanały RSS/Atom (Faza 9).

Czysto tekstowe przetwarzanie, bez dostępu do sieci i bez zależności od systemu —
dlatego jest tutaj, a nie w ``host/``. I bez nowych pakietów: ``html.parser``
oraz ``xml.etree`` z biblioteki standardowej wystarczają, a każdy dodatkowy
pakiet (``beautifulsoup4``, ``readability``, ``feedparser``) to kolejna rzecz,
która może się nie zainstalować na cudzej maszynie.

Dwie rzeczy, które ten moduł traktuje poważnie:

* **treść ze strony to dane niezaufane.** Wyciągamy tekst, a nie wykonujemy
  niczego: skrypty, style i atrybuty zdarzeń są wyrzucane razem z zawartością.
  Ostateczne oczyszczenie (znaczniki udające ramkę wyniku) robi ``security/sandbox.py``
  na wyniku narzędzia,
* **XML z internetu bywa bombą.** ``xml.etree`` nie rozwija encji zewnętrznych,
  ale rozwija wewnętrzne — więc dokument z zagnieżdżonymi encjami („billion
  laughs") potrafi zjeść pamięć. Dlatego przed parsowaniem odrzucamy dokumenty
  z ``<!DOCTYPE`` i ``<!ENTITY``: prawdziwy kanał RSS ich nie potrzebuje.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Final
from i18n import t

logger = logging.getLogger(__name__)

# Znaczniki, których zawartość NIE jest treścią strony.
_SKIP_CONTENT: Final[frozenset[str]] = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "object"}
)

# Znaczniki, które kończą akapit (żeby tekst nie zlał się w jedną linię).
_BLOCK_TAGS: Final[frozenset[str]] = frozenset(
    {
        "p", "div", "section", "article", "header", "footer", "nav", "aside",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "hr", "blockquote",
        "pre", "table", "ul", "ol", "dl", "dd", "dt", "figure", "figcaption", "main",
    }
)

# Elementy, których treść zwykle jest nawigacją albo stopką — pomijamy je, o ile
# strona nie składa się wyłącznie z nich (patrz :func:`extract_readable`).
_BOILERPLATE_TAGS: Final[frozenset[str]] = frozenset({"nav", "footer", "aside", "form"})

# Poniżej tylu znaków uznajemy stronę za „bez treści" i dokładamy nawigację ze
# stopką — lepiej dać modelowi cokolwiek niż pustkę. Progu nie stawiamy wyżej,
# bo krótki wpis na blogu ma kilkadziesiąt znaków i nie chcemy mu doklejać menu.
_EMPTY_PAGE_CHARS: Final[int] = 80

# Zapisy, którymi dokument XML mógłby próbować rozwinąć encje.
_XML_DANGEROUS: Final[re.Pattern[str]] = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)", re.IGNORECASE)

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"[ \t ]+")
_BLANK_LINES: Final[re.Pattern[str]] = re.compile(r"\n{3,}")

_ATOM_NS: Final[str] = "{http://www.w3.org/2005/Atom}"


class ContentError(Exception):
    """Treści nie da się przetworzyć — komunikat jest do pokazania modelowi."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# --------------------------------------------------------------------------- #
# HTML → tekst
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Article:
    """To, co da się wyciągnąć ze strony."""

    title: str
    text: str
    links: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def words(self) -> int:
        return len(self.text.split())


class _TextExtractor(HTMLParser):
    """Zbiera tekst widoczny na stronie, pomijając kod i nawigację."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.links: list[str] = []
        self._skip_depth = 0
        self._boilerplate_depth = 0
        self._in_title = False
        self._boilerplate_parts: list[str] = []

    # --- znaczniki -------------------------------------------------------- #

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in _SKIP_CONTENT:
            self._skip_depth += 1
            return
        if name == "title":
            self._in_title = True
        if name in _BOILERPLATE_TAGS:
            self._boilerplate_depth += 1
        if name == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self.links.append(value.strip())
        if name in _BLOCK_TAGS:
            self._append("\n")
        if name == "meta":
            self._maybe_description(attrs)

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_CONTENT:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if name == "title":
            self._in_title = False
        if name in _BOILERPLATE_TAGS:
            self._boilerplate_depth = max(0, self._boilerplate_depth - 1)
        if name in _BLOCK_TAGS:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        self._append(data)

    # --- pomocnicze ------------------------------------------------------- #

    def _append(self, text: str) -> None:
        if self._boilerplate_depth:
            self._boilerplate_parts.append(text)
        else:
            self.parts.append(text)

    def _maybe_description(self, attrs: list[tuple[str, str | None]]) -> None:
        """Opis ze ``<meta name="description">`` — ratunek dla stron bez treści."""
        values = {key.lower(): (value or "") for key, value in attrs}
        name = values.get("name", "").lower() or values.get("property", "").lower()
        if name in ("description", "og:description") and values.get("content"):
            self.parts.append("\n" + values["content"] + "\n")

    def result(self) -> tuple[str, str]:
        main = "".join(self.parts).strip()
        if len(main) < _EMPTY_PAGE_CHARS and self._boilerplate_parts:
            # Strona zbudowana wyłącznie z nawigacji i stopki (albo aplikacja
            # jednostronicowa) — lepiej dać cokolwiek niż pustkę.
            main = "".join(self.parts + self._boilerplate_parts).strip()
        return " ".join(self.title.split()), main


def clean_text(text: str) -> str:
    """Uporządkuj tekst: pojedyncze spacje, najwyżej jedna pusta linia."""
    without_tabs = _WHITESPACE.sub(" ", str(text or ""))
    # Puste linie ZOSTAJĄ (to granice akapitów) — usuwamy tylko ich nadmiar niżej.
    lines = [line.strip() for line in without_tabs.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def clip(text: str, max_chars: int, *, note: str = "[...] treść obcięta") -> tuple[str, bool]:
    """Obetnij tekst do limitu na granicy zdania, gdy się da."""
    value = str(text or "")
    if len(value) <= max_chars:
        return value, False
    cut = value[:max_chars]
    for separator in (". ", "\n", " "):
        position = cut.rfind(separator)
        if position > max_chars * 0.6:
            cut = cut[: position + 1]
            break
    return cut.rstrip() + f"\n{note}", True


def extract_readable(content: str, *, max_chars: int = 6_000) -> Article:
    """Wyciągnij tytuł i czytelny tekst ze strony HTML.

    Nie jest to „readability" z pełnym rankingiem bloków — to prosty ekstraktor,
    który w praktyce radzi sobie z artykułami, dokumentacją i wpisami na blogach,
    a nie wymaga ani jednej dodatkowej biblioteki. Strony będące aplikacjami
    JavaScriptu zwrócą niewiele; wtedy zostaje ``<meta description>``.
    """
    parser = _TextExtractor()
    try:
        parser.feed(str(content or ""))
        parser.close()
    except Exception as exc:  # pragma: no cover - uszkodzony HTML
        logger.debug("Nie udało się rozłożyć HTML-a: %s", exc)
        raise ContentError(t("content.unreadable")) from exc

    title, body = parser.result()
    text, truncated = clip(clean_text(body), max_chars)
    return Article(
        title=title,
        text=text,
        links=tuple(dict.fromkeys(parser.links))[:50],
        truncated=truncated,
    )


def strip_tags(value: str, *, max_chars: int = 400) -> str:
    """Zdejmij znaczniki z krótkiego fragmentu (opis w kanale RSS, wynik szukania)."""
    without_tags = re.sub(r"<[^>]{0,200}>", " ", str(value or ""))
    text, _ = clip(clean_text(html.unescape(without_tags)), max_chars, note="…")
    return text


# --------------------------------------------------------------------------- #
# Kanały RSS/Atom
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeedItem:
    """Jeden wpis z kanału."""

    title: str
    link: str
    published: str = ""
    summary: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "link": self.link,
            "published": self.published,
            "summary": self.summary,
            "source": self.source,
        }


def parse_feed(content: str, *, limit: int = 10, source: str = "") -> list[FeedItem]:
    """Rozłóż kanał RSS albo Atom na wpisy.

    Odrzucamy dokumenty z deklaracją encji: prawdziwy kanał ich nie potrzebuje, a
    „billion laughs" z internetu potrafi zjeść całą pamięć procesu.
    """
    text = str(content or "").strip()
    if not text:
        raise ContentError(t("content.empty_feed"))
    if _XML_DANGEROUS.search(text[:4_000]):
        raise ContentError(
            t("content.xml_entities")
        )

    from xml.etree import ElementTree  # noqa: PLC0415 - import lokalny, tylko tutaj potrzebny

    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ContentError(t("content.bad_xml", error=exc)) from exc

    items: list[FeedItem] = []
    for element in list(root.iter("item")) + list(root.iter(f"{_ATOM_NS}entry")):
        item = _feed_item(element, source=source)
        if item is not None:
            items.append(item)
        if len(items) >= max(1, limit):
            break
    if not items:
        raise ContentError(t("content.no_entries"))
    return items


def _feed_item(element: object, *, source: str) -> FeedItem | None:
    find = getattr(element, "find", None)
    findall = getattr(element, "findall", None)
    if find is None or findall is None:  # pragma: no cover - nie-element
        return None

    title = _text(find("title")) or _text(find(f"{_ATOM_NS}title"))
    link = _text(find("link")) or _text(find(f"{_ATOM_NS}id"))
    if not link:
        # Atom trzyma adres w atrybucie ``href``.
        for candidate in findall(f"{_ATOM_NS}link"):
            href = candidate.get("href")
            if href:
                link = href
                break
    published = (
        _text(find("pubDate"))
        or _text(find(f"{_ATOM_NS}published"))
        or _text(find(f"{_ATOM_NS}updated"))
    )
    summary = (
        _text(find("description"))
        or _text(find(f"{_ATOM_NS}summary"))
        or _text(find(f"{_ATOM_NS}content"))
    )
    if not title and not link:
        return None
    return FeedItem(
        title=strip_tags(title, max_chars=200) or "(bez tytułu)",
        link=link.strip(),
        published=normalize_date(published),
        summary=strip_tags(summary, max_chars=300),
        source=source,
    )


def _text(element: object) -> str:
    value = getattr(element, "text", None)
    return str(value).strip() if value else ""


def normalize_date(value: str) -> str:
    """Data w postaci ISO, gdy da się ją rozpoznać; inaczej tekst jak przyszedł.

    Kanały używają RFC 822 („Mon, 17 Aug 2026 15:42:00 GMT") albo ISO 8601.
    Nie zgadujemy strefy ani formatu lokalnego — po rozpoznaniu zapisujemy ISO,
    czyli zapis jednoznaczny na każdej maszynie.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    from email.utils import parsedate_to_datetime  # noqa: PLC0415 - tylko tutaj

    try:
        return parsedate_to_datetime(raw).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat(timespec="seconds")
    except ValueError:
        return raw[:40]


def split_list(raw: str) -> list[str]:
    """Lista z ``.env`` (przecinki albo średniki) → lista niepustych wpisów."""
    items = [raw]
    for separator in (";", ","):
        items = [part for item in items for part in item.split(separator)]
    return [item.strip() for item in items if item.strip()]


__all__ = [
    "Article",
    "ContentError",
    "FeedItem",
    "clean_text",
    "clip",
    "extract_readable",
    "normalize_date",
    "parse_feed",
    "split_list",
    "strip_tags",
]
