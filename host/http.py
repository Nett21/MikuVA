"""Kontrolowany dostęp do sieci — jedyne wyjście narzędzi na zewnątrz (Faza 9).

Do Fazy 8 asystent nie ruszał sieci: Ollama stoi na ``127.0.0.1``, modele leżą na
dysku. Faza 9 to zmienia, więc dostęp do sieci ma jedno przejście i twarde reguły:

* **``OFFLINE_MODE=on`` wyłącza sieć całkowicie.** Nie „próbuje i czeka na
  timeout" — narzędzi sieciowych po prostu nie ma, a reszta asystenta działa dalej.
  Świadomie NIE patrzymy tu na ``is_offline()``, czyli na tryb ``auto``: ten
  rozstrzyga, czy *pobierać modele* (Fazy 2–6), a nie czy użytkownikowi wolno
  zapytać o pogodę. Gdyby ``auto`` wyłączał narzędzia sieciowe, to samo posiadanie
  modelu Whispera na dysku odbierałoby asystentowi dostęp do internetu —
  zachowanie zaskakujące i niemożliwe do odgadnięcia z nazwy ustawienia,
* **tylko ``http`` i ``https``** — żadnych ``file://``, ``ftp://``, ``gopher://``,
* **żadnych adresów prywatnych** (loopback, sieci lokalne, link-local, metadane
  chmury). To nie teoria: bez tej blokady model namówiony treścią strony mógłby
  wywołać ``http://127.0.0.1:11434`` (własna Ollama), ``http://169.254.169.254``
  (tokeny maszyny w chmurze) albo router pod ``192.168.0.1``,
* **limity**: czas, rozmiar odpowiedzi, liczba przekierowań, typ treści,
* **sekrety nigdy nie trafiają do logu**. Adresy są przed zalogowaniem
  przepuszczane przez :func:`redact`, a klucze API żyją w ``SecretStr``,
* **brak ciasteczek i brak uwierzytelniania** — każde żądanie jest anonimowe,
  a przekierowania są sprawdzane od nowa (przekierowanie to nowy adres).

Sieć jest jedynym miejscem w projekcie, gdzie **honorujemy proxy z otoczenia**
(``HTTP_PROXY``/``HTTPS_PROXY``): w firmowej sieci to jedyna droga na zewnątrz.
Ollama jest odwrotnie — tam proxy jest jawnie wypięte, bo stoi lokalnie.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from config import (
    APP_NAME,
    APP_VERSION,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

# Schematy, którymi wolno cokolwiek pobrać.
ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# Nazwy parametrów i nagłówków, których wartości nigdy nie pokazujemy — ani w
# logu, ani w komunikacie błędu, ani w wyniku narzędzia.
SECRET_HINTS: Final[tuple[str, ...]] = (
    "key", "apikey", "api_key", "token", "access_token", "secret", "password",
    "passwd", "auth", "signature", "sig", "session",
)

_REDACTED: Final[str] = "***"

# Typy treści, które umiemy przetworzyć. Wszystko inne (obrazy, archiwa, PDF-y
# z sieci) odrzucamy przed pobraniem całości — narzędzie tekstowe nie ma co z tym
# zrobić, a pobieranie kosztuje pasmo użytkownika.
TEXTUAL_CONTENT: Final[tuple[str, ...]] = (
    "text/", "application/json", "application/xml", "application/rss+xml",
    "application/atom+xml", "application/xhtml+xml", "application/ld+json",
    "application/javascript",
)

_HOST_LABEL: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$", re.I)

# Zapis schematu na początku adresu (``https:``, ``javascript:``, ``data:``).
_SCHEME_PREFIX: Final[re.Pattern[str]] = re.compile(r"^([a-z][a-z0-9+.\-]*):", re.IGNORECASE)


class NetworkError(Exception):
    """Błąd sieci nadający się do pokazania człowiekowi (i modelowi).

    ``hint`` mówi, co z tym zrobić. Treść NIGDY nie zawiera sekretów — adresy
    przechodzą przez :func:`redact`.
    """

    def __init__(self, message: str, *, hint: str = "", offline: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.offline = offline

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message} ({self.hint})"
        return self.message


class UrlRefusedError(NetworkError):
    """Adres odrzucony przed wysłaniem żądania (schemat, adres prywatny, port)."""


# --------------------------------------------------------------------------- #
# Ukrywanie sekretów
# --------------------------------------------------------------------------- #


def redact(text: str) -> str:
    """Usuń wartości wyglądające na sekrety z tekstu przeznaczonego do logu.

    Działa na dwóch poziomach: parametry zapytania o „podejrzanych" nazwach
    (``?key=``, ``&token=``) i pary ``nazwa=wartość`` w zwykłym tekście. Lepiej
    zamazać za dużo niż wpuścić klucz API do ``logs/assistant.log``, który
    użytkownik potem wkleja do zgłoszenia błędu.
    """
    value = str(text or "")
    if not value:
        return value

    # 1. Adresy: przepisujemy parametry zapytania.
    def _clean_url(match: re.Match[str]) -> str:
        return redact_url(match.group(0))

    value = re.sub(r"https?://[^\s\"'<>]+", _clean_url, value)

    # 2. Nagłówki uwierzytelniające: „Authorization: Bearer <token>".
    value = re.sub(
        r"(?i)\b(bearer|basic|token)\s+[A-Za-z0-9._~+/=-]{8,}", rf"\1 {_REDACTED}", value
    )

    # 3. Gołe pary nazwa=wartość (np. w komunikacie biblioteki). Nazwa może być
    # złożona („X-Api-Key", „apiKey", „refresh_token"), więc dopuszczamy otoczkę.
    pattern = "|".join(re.escape(name) for name in SECRET_HINTS)
    value = re.sub(
        rf"(?i)\b([\w-]*(?:{pattern})[\w-]*)\s*[=:]\s*([^\s,;&\"']+)",
        rf"\1={_REDACTED}",
        value,
    )
    return value


def redact_url(url: str) -> str:
    """Adres z zamazanymi wartościami parametrów wyglądających na sekrety."""
    try:
        parts = urlsplit(str(url))
    except ValueError:  # pragma: no cover - adres nie do sparsowania
        return _REDACTED
    if not parts.query:
        return _strip_userinfo(parts)
    cleaned: list[tuple[str, str]] = []
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        secret = any(hint in name.lower() for hint in SECRET_HINTS)
        cleaned.append((name, _REDACTED if secret else value))
    # ``safe="*"`` żeby zamazanie zostało czytelnym „***", a nie „%2A%2A%2A":
    # ten adres trafia do komunikatu dla użytkownika i do logu.
    rebuilt = parts._replace(query=urlencode(cleaned, safe="*"))
    return _strip_userinfo(rebuilt)


def _strip_userinfo(parts: Any) -> str:
    """Usuń ``użytkownik:hasło@`` z adresu — to też sekret."""
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{_REDACTED}@{netloc.rsplit('@', 1)[1]}"
    return str(urlunsplit(parts._replace(netloc=netloc)))


def secret_value(value: Any) -> str:
    """Wyłuskaj treść sekretu z ``SecretStr`` (albo z gołego łańcucha).

    Wartość jest zwracana wyłącznie do wysłania w żądaniu. Nigdzie jej nie
    logujemy i nie umieszczamy w wyniku narzędzia.
    """
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        return str(getter() or "")
    return str(value or "")


# --------------------------------------------------------------------------- #
# Sprawdzanie adresu
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HttpPolicy:
    """Limity i uprawnienia jednego zapytania sieciowego."""

    timeout_s: float = 15.0
    max_bytes: int = 2_000_000
    max_redirects: int = 3
    allow_private_hosts: bool = False
    user_agent: str = ""
    offline: bool = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> HttpPolicy:
        active = settings or get_settings()
        agent = active.web_user_agent.strip() or f"{APP_NAME}/{APP_VERSION} (local assistant)"
        return cls(
            timeout_s=active.web_timeout_s,
            max_bytes=active.web_max_bytes,
            max_redirects=active.web_max_redirects,
            # Wyjątek dla adresów prywatnych jest potrzebny w jednym realnym
            # przypadku: własna instancja SearXNG w sieci domowej.
            allow_private_hosts=active.web_allow_private_hosts,
            user_agent=agent,
            offline=hard_offline(active),
        )

    def describe(self) -> str:
        parts = [f"limit {self.timeout_s:.0f} s", f"{self.max_bytes // 1000} kB"]
        if self.allow_private_hosts:
            parts.append("adresy prywatne dozwolone")
        if self.offline:
            parts.append("TRYB OFFLINE — sieć wyłączona")
        return ", ".join(parts)


def check_url(url: str, policy: HttpPolicy | None = None) -> str:
    """Sprawdź adres przed wysłaniem żądania. Zwraca adres znormalizowany.

    Rzuca :class:`UrlRefusedError` z powodem, który wraca do modelu jako zwykły
    błąd narzędzia.
    """
    active = policy or HttpPolicy()
    raw = str(url or "").strip().strip('"').strip("'")
    if not raw:
        raise UrlRefusedError("nie podano adresu")

    declared = _SCHEME_PREFIX.match(raw)
    if declared and declared.group(1).lower() not in ALLOWED_SCHEMES:
        # ``javascript:``, ``data:``, ``file:`` — odrzucamy po nazwie schematu,
        # zanim spróbujemy cokolwiek z tego zrobić. Dopisanie „https://" do
        # takiego zapisu dałoby adres bez sensu i błąd nie na temat.
        raise UrlRefusedError(
            f"schemat '{declared.group(1).lower()}' nie jest obsługiwany — "
            "wolno tylko http i https"
        )
    if "://" not in raw:
        # Model często podaje „example.org/artykuł" — domyślamy się https, ale
        # NIE domyślamy się schematu innego niż bezpieczny.
        raw = f"https://{raw}"

    try:
        parts = urlsplit(raw)
        scheme = (parts.scheme or "").lower()
        port = parts.port
    except ValueError as exc:
        # ``parts.port`` rzuca dopiero przy odczycie, gdy port nie jest liczbą.
        raise UrlRefusedError(f"adres '{redact(raw)}' jest nieprawidłowy") from exc

    if scheme not in ALLOWED_SCHEMES:
        raise UrlRefusedError(
            f"schemat '{scheme}' nie jest obsługiwany — wolno tylko http i https"
        )
    host = (parts.hostname or "").strip()
    if not host:
        raise UrlRefusedError(f"adres '{redact(raw)}' nie ma nazwy hosta")
    if parts.username or parts.password:
        raise UrlRefusedError("adres z loginem i hasłem nie jest obsługiwany")

    _check_host(host, active)
    _check_port(port)
    return str(urlunsplit(parts._replace(scheme=scheme, fragment="")))


def _check_port(port: int | None) -> None:
    """Odrzuć porty usług, które nie są ruchem WWW."""
    if port is None:
        return
    if port in (22, 23, 25, 445, 465, 587, 3306, 5432, 6379, 11211, 27017):
        raise UrlRefusedError(
            f"port {port} nie jest portem WWW — nie łączę się z usługami innego rodzaju"
        )


def _check_host(host: str, policy: HttpPolicy) -> None:
    """Odrzuć adresy prywatne, lokalne i metadane chmury.

    Rozstrzygamy dwa razy: po zapisie adresu (``127.0.0.1``, ``[::1]``,
    ``localhost``) i po rozwiązaniu nazwy (bo ``moja-domena.pl`` może wskazywać
    na ``192.168.1.1``). Bez drugiego sprawdzenia blokada byłaby ozdobą.
    """
    if policy.allow_private_hosts:
        return

    lowered = host.lower().rstrip(".")
    if lowered in ("localhost",) or lowered.endswith((".localhost", ".local", ".internal")):
        raise UrlRefusedError(
            f"'{host}' to adres lokalny — narzędzia sieciowe nie łączą się z tą maszyną "
            "ani z siecią lokalną"
        )

    literal = _as_ip(lowered)
    if literal is not None:
        _reject_private_ip(literal, host)
        return

    if not all(_HOST_LABEL.match(label) for label in lowered.split(".") if label):
        raise UrlRefusedError(f"'{host}' nie wygląda na poprawną nazwę hosta")

    for address in _resolve(lowered):
        _reject_private_ip(address, host)


def _as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    candidate = host.strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return None


def _reject_private_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, host: str
) -> None:
    """Odrzuć wszystko, co nie jest publicznym internetem."""
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise UrlRefusedError(
            f"'{host}' wskazuje na adres {address}, który nie jest publicznym adresem "
            "internetowym (sieć lokalna, ta maszyna albo metadane usługi chmurowej)"
        )


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Adresy IP nazwy. Brak odpowiedzi DNS to błąd sieci, nie błąd adresu."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise NetworkError(
            f"nie udało się rozwiązać nazwy '{host}'",
            hint="sprawdź połączenie z internetem albo poprawność adresu",
        ) from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        candidate = _as_ip(str(info[4][0]))
        if candidate is not None:
            addresses.append(candidate)
    return addresses


# --------------------------------------------------------------------------- #
# Pobieranie
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Odpowiedź serwera po sprawdzeniach i obcięciu."""

    url: str
    status: int
    content_type: str
    text: str
    truncated: bool = False
    elapsed_ms: int = 0
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        import json

        try:
            return json.loads(self.text)
        except ValueError as exc:
            raise NetworkError(
                f"odpowiedź z {redact_url(self.url)} nie jest poprawnym JSON-em",
                hint="serwer mógł zwrócić stronę błędu zamiast danych",
            ) from exc


def build_headers(policy: HttpPolicy, *, accept: str = "") -> dict[str, str]:
    """Nagłówki żądania. Bez ciasteczek, bez uwierzytelniania, bez śledzenia."""
    headers = {
        "User-Agent": policy.user_agent or f"{APP_NAME}/{APP_VERSION}",
        "Accept-Language": "*",
        "Accept-Encoding": "gzip, deflate",
    }
    if accept:
        headers["Accept"] = accept
    return headers


def _ensure_online(policy: HttpPolicy) -> None:
    if policy.offline:
        raise NetworkError(
            "asystent pracuje w trybie offline, więc nie sięga do internetu",
            hint="ustaw OFFLINE_MODE=off, jeśli chcesz pozwolić na dostęp do sieci",
            offline=True,
        )


async def fetch(
    url: str,
    *,
    policy: HttpPolicy | None = None,
    params: Mapping[str, Any] | None = None,
    accept: str = "",
    client: httpx.AsyncClient | None = None,
    require_textual: bool = True,
) -> HttpResponse:
    """Pobierz zasób z sieci po wszystkich sprawdzeniach.

    ``client`` pozwala podstawić transport w testach (``httpx.MockTransport``) —
    dzięki temu żaden test nie wychodzi do internetu.
    """
    active = policy or HttpPolicy.from_settings()
    _ensure_online(active)
    target = check_url(url, active)

    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=httpx.Timeout(active.timeout_s),
        follow_redirects=False,
        # Sieć jest jedynym miejscem, gdzie proxy z otoczenia MA sens.
        trust_env=True,
    )

    started = time.perf_counter()
    try:
        response = await _follow(
            http, target, active, params=params, accept=accept, require_textual=require_textual
        )
    finally:
        if owns_client:
            await http.aclose()

    return HttpResponse(
        url=response.url,
        status=response.status,
        content_type=response.content_type,
        text=response.text,
        truncated=response.truncated,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        headers=response.headers,
    )


async def _follow(
    http: httpx.AsyncClient,
    url: str,
    policy: HttpPolicy,
    *,
    params: Mapping[str, Any] | None,
    accept: str,
    require_textual: bool,
) -> HttpResponse:
    """Wyślij żądanie, podążając za przekierowaniami RĘCZNIE.

    Ręcznie, bo każde przekierowanie to nowy adres — i musi przejść przez te same
    sprawdzenia. Automatyczne ``follow_redirects`` biblioteki pozwoliłoby serwerowi
    przekierować nas na ``http://169.254.169.254`` po zaakceptowaniu adresu
    publicznego.
    """
    current = url
    current_params = dict(params or {})
    headers = build_headers(policy, accept=accept)

    for hop in range(policy.max_redirects + 1):
        try:
            response = await http.get(current, params=current_params or None, headers=headers)
        except httpx.TimeoutException as exc:
            raise NetworkError(
                f"serwer {redact_url(current)} nie odpowiedział w {policy.timeout_s:.0f} s",
                hint="spróbuj ponownie albo sprawdź połączenie",
            ) from exc
        except httpx.TooManyRedirects as exc:  # pragma: no cover - obsługujemy ręcznie
            raise NetworkError("zbyt wiele przekierowań") from exc
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"nie udało się połączyć z {redact_url(current)}",
                hint="sprawdź połączenie z internetem",
            ) from exc

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location", "")
            if not location:
                break
            current = check_url(str(httpx.URL(current).join(location)), policy)
            current_params = {}
            logger.debug("Przekierowanie %s → %s", hop + 1, redact_url(current))
            continue

        return _read_response(response, policy, require_textual=require_textual)

    raise NetworkError(
        f"przekroczono limit {policy.max_redirects} przekierowań",
        hint="strona przekierowuje w kółko albo prosi o zgodę na ciasteczka",
    )


def _read_response(
    response: httpx.Response, policy: HttpPolicy, *, require_textual: bool
) -> HttpResponse:
    """Sprawdź nagłówki, przeczytaj treść z limitem, zdekoduj."""
    content_type = str(response.headers.get("content-type", "")).split(";")[0].strip().lower()
    if require_textual and content_type and not content_type.startswith(TEXTUAL_CONTENT):
        raise NetworkError(
            f"treść typu '{content_type}' nie jest tekstem — nie pobieram jej",
            hint="to narzędzie czyta strony i dane tekstowe, nie pliki binarne",
        )

    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > policy.max_bytes:
        raise NetworkError(
            f"zasób ma {int(declared) // 1000} kB, a limit to {policy.max_bytes // 1000} kB",
            hint="podaj adres konkretnej podstrony zamiast całego archiwum",
        )

    raw = response.content[: policy.max_bytes]
    truncated = len(response.content) > policy.max_bytes
    text = _decode(raw, response)

    if response.status_code >= 400:
        raise NetworkError(
            f"serwer {redact_url(str(response.url))} odpowiedział kodem {response.status_code}",
            hint=_status_hint(response.status_code),
        )

    return HttpResponse(
        url=str(response.url),
        status=response.status_code,
        content_type=content_type,
        text=text,
        truncated=truncated,
        headers={key.lower(): value for key, value in response.headers.items()},
    )


def _decode(raw: bytes, response: httpx.Response) -> str:
    """Zamień bajty na tekst.

    Kolejność: kodowanie z nagłówka, potem UTF-8, na końcu UTF-8 z zamianą złych
    bajtów. **Nie** używamy kodowania domyślnego dla systemu — ta sama strona
    czytana na dwóch maszynach musi dać ten sam tekst.
    """
    declared = response.charset_encoding
    for encoding in (declared, "utf-8"):
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _status_hint(status: int) -> str:
    if status == 404:
        return "strony nie ma pod tym adresem"
    if status in (401, 403):
        return "serwer odmówił dostępu (może wymagać logowania)"
    if status == 429:
        return "zbyt wiele zapytań — serwis prosi o przerwę"
    if status >= 500:
        return "problem po stronie serwera, nie u nas"
    return "szczegóły w logs/assistant.log"


async def fetch_json(
    url: str,
    *,
    policy: HttpPolicy | None = None,
    params: Mapping[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    """Pobierz i odczytaj JSON — najczęstszy przypadek przy API."""
    response = await fetch(
        url, policy=policy, params=params, accept="application/json", client=client
    )
    return response.json()


# --------------------------------------------------------------------------- #
# Diagnostyka
# --------------------------------------------------------------------------- #


def hard_offline(settings: Settings | None = None) -> bool:
    """Czy dostęp do sieci jest ZAKAZANY, a nie tylko „niepotrzebny".

    Tylko ``OFFLINE_MODE=on``. Tryb ``auto`` (patrz ``config.is_offline``) mówi
    „nie pobieraj modeli, bo są na dysku" — to inne pytanie niż „czy użytkownikowi
    wolno zapytać o pogodę". Rozdzielenie tych dwóch rzeczy jest tu celowe.
    """
    active = settings or get_settings()
    return str(active.offline_mode).strip().lower() == "on"


def network_available(settings: Settings | None = None) -> tuple[bool, str]:
    """Czy narzędzia sieciowe mają w ogóle sens na tej maszynie i w tym trybie."""
    active = settings or get_settings()
    if not active.web_enabled:
        return False, "narzędzia sieciowe są wyłączone (WEB_ENABLED=false)"
    if hard_offline(active):
        return False, (
            "asystent pracuje w trybie offline (OFFLINE_MODE=on) — narzędzia sieciowe "
            "są wyłączone; wszystko lokalne działa dalej"
        )
    return True, ""


def describe_backend(settings: Settings | None = None) -> str:
    """Jedna linijka do raportu zależności i do ``/narzedzia``."""
    active = settings or get_settings()
    usable, reason = network_available(active)
    if not usable:
        return reason
    policy = HttpPolicy.from_settings(active)
    proxy = "proxy z otoczenia" if _proxy_configured() else "bez proxy"
    return f"{policy.describe()}, {proxy}"


def _proxy_configured() -> bool:
    import os

    return any(
        os.environ.get(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY")
    )


def public_hosts(urls: Sequence[str]) -> list[str]:
    """Nazwy hostów z listy adresów — do opisów i podpowiedzi."""
    hosts: list[str] = []
    for url in urls:
        try:
            host = urlsplit(url).hostname or ""
        except ValueError:  # pragma: no cover - adres nie do sparsowania
            continue
        if host and host not in hosts:
            hosts.append(host)
    return hosts


__all__ = [
    "ALLOWED_SCHEMES",
    "SECRET_HINTS",
    "TEXTUAL_CONTENT",
    "HttpPolicy",
    "HttpResponse",
    "NetworkError",
    "UrlRefusedError",
    "build_headers",
    "check_url",
    "describe_backend",
    "fetch",
    "fetch_json",
    "hard_offline",
    "network_available",
    "public_hosts",
    "redact",
    "redact_url",
    "secret_value",
]
