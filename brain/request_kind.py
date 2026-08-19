"""LOCAL REQUEST czy WEB REQUEST — czy pytanie wymaga świeżych danych (Faza 9).

Model sam z siebie chętnie odpowiada z pamięci na pytania, na które **nie ma jak**
znać odpowiedzi: „jaka jest pogoda?", „co się dzieje w wiadomościach?", „ile
kosztuje X?". Wynik brzmi wiarygodnie i jest nieprawdziwy. Ten moduł rozpoznaje
takie pytania **przed** zapytaniem modelu i dokłada do promptu jedno zdanie:
skorzystaj z narzędzia albo powiedz, że nie wiesz — nie zgaduj.

Rozpoznawanie jest **czysto tekstowe** (bez modelu i bez sieci), bo ma działać
także wtedy, gdy Ollama nie odpowiada, i nie może kosztować dodatkowej tury.
Trzy kategorie:

``LOCAL``
    pytanie o dane z tej maszyny albo o wiedzę ogólną — nic nie dopisujemy,
``WEB``
    pytanie wymaga aktualnych danych z internetu (pogoda, wiadomości, kursy,
    „dzisiaj", „najnowszy"),
``MIXED``
    pytanie dotyczy jednocześnie plików/pamięci i świeżych danych — prompt mówi
    o obu drogach.

Klasyfikacja jest **podpowiedzią, nie wyrokiem**: nie blokuje odpowiedzi i nie
wymusza wywołania narzędzia. Gdy narzędzia sieciowe są niedostępne (tryb offline,
brak internetu), prompt dostaje inne zdanie: powiedz wprost, że nie masz dostępu
do świeżych danych — i wtedy asystent działa dalej, tylko lokalnie.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

logger = logging.getLogger(__name__)


class RequestKind(StrEnum):
    """Czego pytanie wymaga."""

    LOCAL = "LOCAL"
    WEB = "WEB"
    MIXED = "MIXED"


# Słowa, po których widać, że chodzi o świeże dane. Zapisane BEZ polskich znaków
# diakrytycznych, bo tekst jest przed porównaniem składany tą samą funkcją co
# w brain/memory.py — z rozpoznawania mowy wychodzi i „pogoda", i „pogode".
_WEB_MARKERS: Final[tuple[str, ...]] = (
    # Uwaga na odmianę: wpisujemy RDZENIE, nie formy podstawowe. „pogoda",
    # „pogodę", „pogodzie" to jedno słowo dla użytkownika, a trzy dla porównania
    # tekstu — dopasowanie po rdzeniu („pogod") łapie wszystkie.
    # pogoda i otoczenie
    "pogod", "temperatur", "deszcz", "snieg", "prognoz",
    "weather", "forecast", "rain", "snow", "wind",
    # wiadomości i wydarzenia
    "wiadomosc", "newsy", "news", "headlines", "naglow", "co sie dzieje",
    "co nowego", "wydarzen", "wydarzyl", "stalo sie", "happening", "happened",
    # ceny, kursy, wyniki
    "kurs", "cena", "ceny", "kosztuje", "price", "cost",
    "stock", "akcje", "bitcoin", "kryptowalut", "exchange rate", "wynik",
    "score", "mecz", "match",
    # rozkłady i dostępność
    "rozklad", "godziny otwarcia", "opening hours", "otwarte", "open now",
    # świeżość
    "dzisiaj", "dzis", "wczoraj", "jutro", "teraz", "aktualn", "najnowsz",
    "ostatnio", "biezac", "today", "yesterday", "tomorrow", "current", "latest",
    "recent", "right now", "this week", "w tym tygodniu",
    # wprost o internecie
    "internet", "sieci", "google", "wyszukaj", "poszukaj", "sprawdz w",
    "search for", "look up", "youtube",
)

# Słowa wskazujące na dane z tej maszyny — pamięć, pliki, procesy, aplikacje.
_LOCAL_MARKERS: Final[tuple[str, ...]] = (
    # Tu również rdzenie: „notatce", „notatek", „notatkami" to ten sam „notat".
    "plik", "katalog", "folder", "workspace", "notat", "zapisz", "zapisa",
    "zapamie", "zapomn", "pamie", "proces", "aplikac", "program", "uruchom",
    "otworz", "uslug", "dysk",
    "file", "directory", "note", "remember", "launch", "service", "process",
)

# Zapisy, po których widać, że użytkownik pyta o coś sprawdzalnego w sieci mimo
# braku słów kluczowych: adres strony w treści pytania.
_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://|\bwww\.\w|\b\w+\.(?:pl|com|org|net)\b")

_FOLD_EXTRA: Final[dict[int, str]] = str.maketrans({"ł": "l", "Ł": "l"})


def _fold(text: str) -> str:
    """Tekst do porównań: małe litery, bez diakrytyków, pojedyncze spacje.

    Ta sama zasada co w ``brain/memory.py``: ``unicodedata``, nie ``locale`` —
    klasyfikacja tego samego pytania nie może zależeć od ustawień regionalnych.
    """
    lowered = str(text or "").strip().lower().translate(_FOLD_EXTRA)
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.split())


@dataclass(frozen=True, slots=True)
class RequestAssessment:
    """Wynik rozpoznania: czego pytanie wymaga i dlaczego tak sądzimy."""

    kind: RequestKind
    web_hits: tuple[str, ...] = ()
    local_hits: tuple[str, ...] = ()

    @property
    def needs_web(self) -> bool:
        return self.kind in (RequestKind.WEB, RequestKind.MIXED)

    @property
    def needs_local(self) -> bool:
        return self.kind in (RequestKind.LOCAL, RequestKind.MIXED)

    def describe(self) -> str:
        if self.kind is RequestKind.LOCAL:
            return "pytanie lokalne"
        reason = ", ".join(self.web_hits[:3])
        label = "wymaga świeżych danych" if self.kind is RequestKind.WEB else "lokalne + sieć"
        return f"{label} ({reason})" if reason else label


def classify(text: str) -> RequestAssessment:
    """Rozpoznaj, czy pytanie wymaga danych z internetu.

    Zasada rozstrzygania: **przy wątpliwości LOCAL**. Fałszywe „WEB" dokłada do
    promptu zdanie o narzędziach tam, gdzie nie są potrzebne — kosztuje kontekst i
    zachęca model do zbędnych wywołań. Fałszywe „LOCAL" jest tańsze: model i tak
    ma listę narzędzi i regułę „nie wymyślaj", więc nadal może z nich skorzystać.
    """
    folded = _fold(text)
    if not folded:
        return RequestAssessment(kind=RequestKind.LOCAL)

    web_hits = tuple(marker for marker in _WEB_MARKERS if marker in folded)
    local_hits = tuple(marker for marker in _LOCAL_MARKERS if marker in folded)
    if not web_hits and _URL_PATTERN.search(folded):
        web_hits = ("adres strony",)

    if web_hits and local_hits:
        return RequestAssessment(
            kind=RequestKind.MIXED, web_hits=web_hits, local_hits=local_hits
        )
    if web_hits:
        return RequestAssessment(kind=RequestKind.WEB, web_hits=web_hits)
    return RequestAssessment(kind=RequestKind.LOCAL, local_hits=local_hits)


# --------------------------------------------------------------------------- #
# Zdania doklejane do promptu systemowego
# --------------------------------------------------------------------------- #

_WEB_HINT_PL: Final[str] = """\
TO PYTANIE WYMAGA AKTUALNYCH DANYCH (weryfikowalnych w internecie). Nie odpowiadaj
z pamięci: użyj narzędzia sieciowego (pogoda, wiadomości, wyszukiwanie) i oprzyj
odpowiedź na tym, co zwróci. Jeśli narzędzie zawiedzie albo go nie ma, powiedz
wprost, że nie masz dostępu do świeżych danych — nie zgaduj i nie podawaj wartości
„z grubsza"."""

_WEB_HINT_EN: Final[str] = """\
THIS REQUEST NEEDS CURRENT DATA (something verifiable on the internet). Do not
answer from memory: call a web tool (weather, news, search) and base your answer on
what it returns. If the tool fails or is unavailable, say plainly that you have no
access to current data — do not guess and do not give approximate values."""

_OFFLINE_HINT_PL: Final[str] = """\
TO PYTANIE WYMAGA AKTUALNYCH DANYCH, ale narzędzia sieciowe są teraz niedostępne
(tryb offline albo brak internetu). Powiedz to użytkownikowi wprost w pierwszym
zdaniu, a potem — jeśli to ma sens — podziel się tym, co wiesz, wyraźnie oznaczając,
że nie jest to bieżąca informacja. Nie podawaj wymyślonych liczb ani dat."""

_OFFLINE_HINT_EN: Final[str] = """\
THIS REQUEST NEEDS CURRENT DATA, but the web tools are unavailable right now
(offline mode or no internet). Say so plainly in your first sentence, then — if it
helps — share what you know while making clear it is not current. Never invent
numbers or dates."""

_MIXED_SUFFIX_PL: Final[str] = (
    "\nCzęść pytania dotyczy danych z tego komputera (pliki, notatki, procesy) — "
    "do tego użyj narzędzi lokalnych."
)
_MIXED_SUFFIX_EN: Final[str] = (
    "\nPart of the request is about this computer (files, notes, processes) — use the "
    "local tools for that part."
)


def prompt_hint(
    assessment: RequestAssessment, *, language: str = "en", web_available: bool = True
) -> str:
    """Zdanie do promptu systemowego dla tej tury. Puste = nie dokładamy nic."""
    if not assessment.needs_web:
        return ""
    polish = language == "pl"
    if web_available:
        hint = _WEB_HINT_PL if polish else _WEB_HINT_EN
    else:
        hint = _OFFLINE_HINT_PL if polish else _OFFLINE_HINT_EN
    if assessment.kind is RequestKind.MIXED:
        hint += _MIXED_SUFFIX_PL if polish else _MIXED_SUFFIX_EN
    return hint


def user_notice(
    assessment: RequestAssessment, *, language: str = "en", web_available: bool = True
) -> str:
    """Komunikat dla użytkownika, gdy pytanie wymaga sieci, a sieci nie ma.

    Pusty łańcuch = nie ma o czym mówić. Ten tekst jest też **wypowiadany**
    (Faza 4), więc jest jednym zdaniem bez nawiasów i skrótów.
    """
    if not assessment.needs_web or web_available:
        return ""
    if language == "pl":
        return (
            "To pytanie wymaga danych z internetu, a pracuję teraz bez dostępu do sieci. "
            "Odpowiem z tego, co wiem, ale to nie będą bieżące informacje."
        )
    return (
        "This needs data from the internet and I have no network access right now. "
        "I will answer from what I know, but it will not be current information."
    )


__all__ = [
    "RequestAssessment",
    "RequestKind",
    "classify",
    "prompt_hint",
    "user_notice",
]
