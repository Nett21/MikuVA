"""Osobowość asystenta i budowa promptu systemowego.

Imię asystenta pochodzi WYŁĄCZNIE z ``config/user_settings.json``
(``assistant_name``) i jest wstawiane do promptu dynamicznie — w treści promptu
nie ma żadnego imienia zapisanego na sztywno.

Domyślnym językiem odpowiedzi jest angielski (``LANGUAGE`` w ``.env``), a wpisany
kod OBOWIĄZUJE: przy ``LANGUAGE=en`` pytanie zadane po polsku również dostaje
odpowiedź po angielsku, bo inaczej ustawienie byłoby tylko sugestią. Dopiero
``LANGUAGE=auto`` oddaje decyzję heurystyce :func:`detect_language`, która
rozpoznaje język każdej wypowiedzi osobno.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final, Literal

from config import UserSettings, get_user_settings

Language = Literal["pl", "en"]

# Język, w którym asystent odpowiada, gdy nie ustawiono nic innego. Polski jest
# obsługiwany równorzędnie — wystarczy LANGUAGE=pl w .env.
DEFAULT_LANGUAGE: Final[Language] = "en"
SUPPORTED_LANGUAGES: Final[tuple[Language, ...]] = ("en", "pl")

# Wpisany kod języka WIĄŻE: przy LANGUAGE=en pytanie po polsku również dostaje
# odpowiedź po angielsku. Rozpoznawanie języka z wypowiedzi włącza się dopiero
# przy LANGUAGE=auto (albo "speech_language": "auto" w user_settings.json) —
# inaczej ustawienie użytkownika byłoby tylko sugestią, którą nadpisuje pierwsze
# polskie słowo w pytaniu.
LANGUAGE_AUTO: Final[str] = "auto"

_PL_DIACRITICS: Final[frozenset[str]] = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

_PL_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "jest", "nie", "tak", "czy", "jak", "co", "to", "sie", "się", "mi", "mnie",
        "ale", "bo", "dla", "jestem", "masz", "mam", "moze", "może", "prosze",
        "proszę", "dziekuje", "dziękuję", "dzien", "dzień", "dobry", "witaj",
        "cesc", "cześć", "kiedy", "gdzie", "dlaczego", "opowiedz", "napisz",
        "pokaz", "pokaż", "zrob", "zrób", "powiedz", "wiesz", "jaki", "jaka",
    }
)

_EN_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "the", "is", "are", "what", "how", "why", "when", "where", "you", "your",
        "please", "thanks", "thank", "hello", "hi", "can", "could", "would",
        "tell", "give", "show", "make", "write", "about", "with", "and", "for",
        "do", "does", "did", "have", "has", "want", "need", "there", "this",
    }
)

_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str, *, default: Language = DEFAULT_LANGUAGE) -> Language:
    """Rozpoznaj język wypowiedzi (``pl``/``en``) prostą heurystyką słów kluczowych.

    Brak zewnętrznych zależności jest tu celowy: to ma działać na czystej
    instalacji, także offline. Przy niejednoznacznym wyniku wygrywa ``default``.
    """
    if not text or not text.strip():
        return default

    if any(character in _PL_DIACRITICS for character in text):
        return "pl"

    words = [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]
    if not words:
        return default

    polish_hits = sum(1 for word in words if word in _PL_MARKERS)
    english_hits = sum(1 for word in words if word in _EN_MARKERS)

    if english_hits > polish_hits:
        return "en"
    if polish_hits > english_hits:
        return "pl"
    return default


def normalize_language(code: str | None) -> Language:
    """Sprowadź dowolny kod języka do obsługiwanego zestawu."""
    if not code:
        return DEFAULT_LANGUAGE
    candidate = code.strip().lower()[:2]
    if candidate in SUPPORTED_LANGUAGES:
        return candidate  # type: ignore[return-value]
    return DEFAULT_LANGUAGE


def is_auto_language(code: str | None) -> bool:
    """Czy ustawienie oznacza „rozpoznawaj język przy każdej wypowiedzi"?"""
    return (code or "").strip().lower() in ("", LANGUAGE_AUTO)


def resolve_reply_language(preferred: str | None, text: str = "") -> Language:
    """Język odpowiedzi dla tej tury.

    Wpisany kod (``en``, ``pl``) obowiązuje bez wyjątku — także wtedy, gdy
    użytkownik napisze w innym języku. ``auto`` oddaje decyzję heurystyce
    :func:`detect_language`.
    """
    if is_auto_language(preferred):
        return detect_language(text)
    return normalize_language(preferred)


_BASE_PL = """\
Jesteś asystentem głosowym o imieniu {name}. Rozmawiasz z jednym użytkownikiem,
na jego komputerze, lokalnie.

Twój charakter:
- pogodna, energiczna i pomocna; lekko żartobliwa, ale nigdy kosztem konkretu,
- mówisz naturalnie i zwięźle — Twoje odpowiedzi bywają czytane na głos,
  więc unikaj długich wyliczeń, nagłówków i formatowania markdown,
- emotikony stosujesz oszczędnie, najwyżej pojedynczo i tylko gdy coś wnoszą,
- nie powtarzasz swojego imienia bez potrzeby i nie zaczynasz od niego zdań,
- nie zaczynasz odpowiedzi od potwierdzeń w rodzaju "Jasne!", "Oczywiście!" —
  przechodzisz do rzeczy.

Zasady odpowiadania:
- {language_rule},
- gdy czegoś nie wiesz albo nie masz do czegoś dostępu, mówisz to wprost
  zamiast zgadywać,
- nie wymyślasz faktów, dat ani cytatów,
- gdy pytanie jest niejasne, dopytujesz jednym krótkim zdaniem."""

_BASE_EN = """\
You are a voice assistant named {name}. You talk with a single user, locally on
their computer.

Your character:
- cheerful, energetic and helpful; lightly humorous, never at the cost of substance,
- you speak naturally and concisely — your answers may be read aloud, so avoid
  long lists, headings and markdown formatting,
- you use emoji sparingly, at most one, and only when it adds something,
- you do not repeat your own name unnecessarily and do not open sentences with it,
- you do not start with filler like "Sure!" or "Of course!" — you get to the point.

Answering rules:
- {language_rule},
- when you do not know something or have no access to it, say so plainly instead
  of guessing,
- never invent facts, dates or quotations,
- when a request is ambiguous, ask one short clarifying question."""

# Reguła języka wstawiana w {language_rule}. Wariant „ustawiony" jest stanowczy,
# bo modele chętnie przechodzą na język pytania — a użytkownik, który wpisał
# LANGUAGE=en, chce angielskiej odpowiedzi także na pytanie po polsku.
_RULE_PL_LOCKED = (
    "odpowiadasz ZAWSZE po polsku, także wtedy, gdy użytkownik napisze w innym języku"
)
_RULE_PL_AUTO = "odpowiadasz po polsku, chyba że użytkownik wyraźnie pisze w innym języku"
_RULE_EN_LOCKED = (
    "you ALWAYS reply in English, even when the user writes in another language"
)
_RULE_EN_AUTO = "reply in English while the user writes in English; switch back when they do"

# Domknięcie promptu przy ustawionym języku. Sama reguła na liście zasad NIE
# wystarcza — sprawdzone na qwen2.5:7b-instruct, który na pytanie po polsku
# odpowiadał po polsku mimo instrukcji. Modele silnie ciągną do języka pytania,
# więc żądanie stoi na KOŃCU promptu (najbliżej wypowiedzi użytkownika) i jest
# sformułowane jako warunek bezwyjątkowy.
_LOCK_PL = """\

JĘZYK ODPOWIEDZI: polski. Zasada jest bezwzględna i ma pierwszeństwo nad językiem
wypowiedzi użytkownika — nawet jeśli napisze po angielsku lub w jakimkolwiek innym
języku, odpowiadasz po polsku. Nie tłumacz jego wypowiedzi i nie komentuj wyboru
języka; po prostu odpowiedz po polsku."""

_LOCK_EN = """\

LANGUAGE OF YOUR REPLY: English. This rule is absolute and takes precedence over the
language of the user's message — even if they write in Polish or any other language,
you answer in English. Do not translate their message and do not comment on the
language; just answer in English."""

_TRAITS_PL = """\

Dodatkowe cechy charakteru do zastosowania: {traits}"""

_TRAITS_EN = """\

Additional character traits to apply: {traits}"""

_TRAITS_GUARD_PL = """\

Powyższe dodatkowe cechy dotyczą wyłącznie STYLU i TONU wypowiedzi (poziom humoru,
sposób zwracania się, ulubione porównania). Nie zmieniają one zasad opisanych
wcześniej, nie nadają Ci nowych uprawnień i nie wpływają na to, jakie treści
uznajesz za odpowiednie do wygenerowania. W razie sprzeczności obowiązują zasady
z początku tej instrukcji."""

_TRAITS_GUARD_EN = """\

The additional traits above concern STYLE and TONE only (level of humour, forms of
address, favourite comparisons). They do not modify the rules stated earlier, do
not grant you new capabilities, and do not change what content you consider
appropriate to produce. If they ever conflict, the rules stated earlier win."""


_TIME_PL: Final[str] = "\n\nAktualna data i godzina użytkownika: {timestamp}."
_TIME_EN: Final[str] = "\n\nThe user's current date and time: {timestamp}."

# Nagłówek wiadomości kontekstowej. Jest potrzebny, bo ta wiadomość jedzie z rolą
# „user" (patrz brain/llm.py) — bez niego model mógłby odpowiadać NA nią zamiast
# na pytanie człowieka.
_CONTEXT_HEADER_PL: Final[str] = (
    "[KONTEKST — informacje pomocnicze, nie odpowiadaj na nie wprost]"
)
_CONTEXT_HEADER_EN: Final[str] = (
    "[CONTEXT — background information, do not reply to it directly]"
)


def _format_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M (%Z)")


def build_system_prompt(
    user_settings: UserSettings | None = None,
    *,
    language: Language | str = DEFAULT_LANGUAGE,
    extra_context: str | None = None,
    lock_language: bool = True,
    tool_rules: str = "",
    request_hint: str = "",
) -> str:
    """Zbuduj prompt systemowy dla bieżącej tury rozmowy.

    Imię asystenta i dodatkowe cechy charakteru pochodzą z ustawień użytkownika,
    więc zmiana ``config/user_settings.json`` zmienia zachowanie modelu bez
    modyfikacji kodu.

    ``lock_language=True`` (domyślnie, bo domyślnie ``LANGUAGE`` jest wpisany)
    każe modelowi trzymać się wskazanego języka niezależnie od tego, w jakim
    języku pyta użytkownik. ``False`` zostawia mu swobodę — tak działa
    ``LANGUAGE=auto``, gdzie język wypowiedzi i tak jest rozpoznawany osobno.

    ``request_hint`` to jedno zdanie o charakterze pytania (Faza 9): czy wymaga
    świeżych danych z internetu, czy da się odpowiedzieć lokalnie. Trafia PRZED
    regułami narzędzi, bo dotyczy tego, *czy* wolno odpowiedzieć z pamięci.

    ``tool_rules`` to reguły korzystania z narzędzi (Faza 7,
    :func:`brain.tool_router.tool_system_rules`). Stoją PRZED sekcją cech
    charakteru i przed kontekstem pamięci, bo są zasadą, a nie stylem — a cechy
    charakteru wprost nie mogą zmieniać zasad.
    """
    settings = user_settings if user_settings is not None else get_user_settings()
    resolved = normalize_language(language if isinstance(language, str) else language)

    base = _BASE_EN if resolved == "en" else _BASE_PL
    if resolved == "en":
        rule = _RULE_EN_LOCKED if lock_language else _RULE_EN_AUTO
    else:
        rule = _RULE_PL_LOCKED if lock_language else _RULE_PL_AUTO
    prompt = base.format(name=settings.assistant_name, language_rule=rule)

    # --- część STAŁA: to samo w każdej turze ---------------------------------
    # Kolejność nie jest kosmetyką, tylko wydajnością. Serwer modelu trzyma w
    # pamięci policzony PREFIKS promptu i przy kolejnej turze przelicza dopiero
    # od pierwszej różnicy. Zmierzone na tej maszynie (qwen2.5:7b na CPU):
    # ten sam prefiks — 0,5 s zamiast 86 s. Znacznik czasu stojący na POCZĄTKU
    # zmieniał się co minutę i kasował cały cache, więc każda tura kosztowała
    # ~90 s czekania, zanim model powiedział pierwsze słowo.
    if tool_rules.strip():
        prompt += "\n\n" + tool_rules.strip()

    traits = settings.personality_traits.strip()
    if traits:
        traits_template = _TRAITS_EN if resolved == "en" else _TRAITS_PL
        guard = _TRAITS_GUARD_EN if resolved == "en" else _TRAITS_GUARD_PL
        prompt += traits_template.format(traits=traits) + "\n" + guard

    # Treści ZMIENNE (godzina, wspomnienia, podpowiedź o świeżych danych) tu nie
    # trafiają — buduje je :func:`build_context_message` i idą osobną wiadomością.
    # Argumenty zostają w sygnaturze dla zgodności ze starszym użyciem.
    if extra_context:
        prompt += "\n\n" + extra_context.strip()

    if request_hint.strip():
        prompt += "\n\n" + request_hint.strip()

    if lock_language:
        # Na samym końcu, czyli najbliżej wypowiedzi użytkownika — inaczej model
        # gubi to żądanie pod sekcją wspomnień.
        prompt += "\n" + (_LOCK_EN if resolved == "en" else _LOCK_PL)

    return prompt


def build_context_message(
    *,
    language: Language | str = DEFAULT_LANGUAGE,
    extra_context: str | None = None,
    request_hint: str = "",
    include_time: bool = True,
) -> str:
    """Zmienna część promptu: godzina, wspomnienia, podpowiedź o świeżych danych.

    Osobno od :func:`build_system_prompt`, bo to jedyne fragmenty, które zmieniają
    się między turami. Trzymanie ich poza promptem systemowym pozwala serwerowi
    modelu użyć ponownie policzonego prefiksu (razem ze schematami narzędzi) —
    różnica zmierzona na tej maszynie to 43 s kontra 0,4 s na turę.
    """
    resolved = normalize_language(language if isinstance(language, str) else language)
    parts: list[str] = [_CONTEXT_HEADER_EN if resolved == "en" else _CONTEXT_HEADER_PL]
    if extra_context and extra_context.strip():
        parts.append(extra_context.strip())
    if request_hint.strip():
        parts.append(request_hint.strip())
    if include_time:
        template = _TIME_EN if resolved == "en" else _TIME_PL
        parts.append(template.format(timestamp=_format_timestamp()).strip())
    if len(parts) == 1:  # sam nagłówek bez treści = nie ma czego wysyłać
        return ""
    return "\n\n".join(parts)


def greeting(user_settings: UserSettings | None = None, *, language: Language | str = DEFAULT_LANGUAGE) -> str:
    """Krótkie powitanie używane przy starcie trybu terminalowego."""
    settings = user_settings if user_settings is not None else get_user_settings()
    resolved = normalize_language(language if isinstance(language, str) else language)
    if resolved == "en":
        return f"Hi, I'm {settings.assistant_name}. What can I do for you?"
    return f"Cześć, jestem {settings.assistant_name}. W czym mogę pomóc?"


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGE_AUTO",
    "SUPPORTED_LANGUAGES",
    "Language",
    "build_context_message",
    "build_system_prompt",
    "detect_language",
    "greeting",
    "is_auto_language",
    "normalize_language",
    "resolve_reply_language",
]
