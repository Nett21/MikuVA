"""Klient REST API Home Assistanta (Faza 11).

Adres i token pochodzą **wyłącznie z konfiguracji** (``HOME_ASSISTANT_URL`` i
``HOME_ASSISTANT_TOKEN`` w ``.env``). W kodzie nie ma ani jednego adresu IP, ani
jednej nazwy hosta i ani jednego tokenu — instalacja domowa każdego użytkownika
jest inna, a token jest sekretem tak samo jak hasło.

Jak traktujemy token
--------------------

* leży w ustawieniach jako ``SecretStr``, więc nie pokaże się w ``repr()``
  obiektu ustawień ani w raporcie zależności,
* jedyne miejsce, w którym jest odpakowywany, to nagłówek ``Authorization``
  budowany tuż przed wysłaniem,
* **żaden komunikat błędu i żaden log nie zawiera nagłówków** — funkcja
  :func:`redact` czyści to, co mimo wszystko mogłoby się prześlizgnąć.

Dlaczego własny klient, a nie ``host.http.fetch``
-------------------------------------------------

``host.http`` służy do czytania stron z internetu: nie umie POST-a ani własnych
nagłówków i domyślnie blokuje adresy prywatne (słusznie — chroni przed tym, żeby
model kazał asystentowi pukać do routera). Home Assistant jest przypadkiem
odwrotnym: to **własny serwer użytkownika w jego sieci**, wskazany przez niego
świadomie w pliku konfiguracyjnym. Dlatego jest osobny, wąski klient, który umie
dokładnie dwie rzeczy: odczytać stan encji i wywołać usługę.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import httpx

from config import Settings, get_settings
from i18n import t

logger = logging.getLogger(__name__)

# Ile treści odpowiedzi wolno wczytać. Instalacja z setkami encji potrafi zwrócić
# kilka megabajtów JSON-a, a do niczego nam tyle nie potrzeba.
MAX_RESPONSE_BYTES: Final[int] = 4_000_000

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Usuń token z tekstu, zanim trafi do logu albo do modelu."""
    return _TOKEN_PATTERN.sub(r"\1<ukryty>", text)


class HomeAssistantError(RuntimeError):
    """Błąd rozmowy z Home Assistantem — z podpowiedzią, co zrobić."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = redact(message)
        self.hint = hint

    @property
    def user_message(self) -> str:
        return f"{self.message} ({self.hint})" if self.hint else self.message


@dataclass(frozen=True, slots=True)
class EntityState:
    """Stan jednej encji: to, co Home Assistant nazywa ``state`` plus atrybuty."""

    entity_id: str
    state: str
    attributes: dict[str, Any]
    changed_at: str = ""

    @property
    def domain(self) -> str:
        """Część przed kropką: ``light``, ``switch``, ``lock``, ``cover``…

        Domena decyduje o poziomie ryzyka — patrz ``plugins/home_assistant/tools.py``.
        """
        return self.entity_id.split(".", 1)[0].lower()

    @property
    def friendly_name(self) -> str:
        name = self.attributes.get("friendly_name")
        return str(name) if name else self.entity_id

    def describe(self) -> str:
        return f"{self.friendly_name} ({self.entity_id}): {self.state}"


def normalize_base_url(raw: str) -> str:
    """Adres instancji sprowadzony do postaci ``schemat://host:port``.

    Użytkownik wpisuje adres ręcznie, więc trafiają się końcowe ukośniki,
    doklejone ``/api`` i brak schematu. Naprawiamy to tutaj, raz.
    """
    text = raw.strip()
    if not text:
        return ""
    if "://" not in text:
        # Bez schematu zakładamy http: Home Assistant w sieci domowej najczęściej
        # stoi bez certyfikatu, a zgadnięcie https dałoby błąd nie do zrozumienia.
        text = f"http://{text}"
    parts = urlsplit(text)
    if not parts.hostname:
        raise HomeAssistantError(
            t("ha.bad_url", url=repr(raw)),
            hint=t("ha.bad_url_hint"),
        )
    path = parts.path.rstrip("/")
    if path.endswith("/api"):
        path = path[: -len("/api")]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True, slots=True)
class HomeAssistantConfig:
    """Konfiguracja połączenia — zbudowana z ustawień, nigdy z kodu."""

    base_url: str
    token: str
    timeout_s: float = 10.0
    max_entities: int = 60

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> HomeAssistantConfig:
        active = settings or get_settings()
        return cls(
            base_url=normalize_base_url(active.home_assistant_url),
            token=active.home_assistant_token.get_secret_value().strip(),
            timeout_s=active.home_assistant_timeout_s,
            max_entities=active.home_assistant_max_entities,
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def describe(self) -> str:
        """Opis do raportu zależności. Adres tak, token — nigdy."""
        if not self.base_url:
            return "brak adresu (HOME_ASSISTANT_URL)"
        if not self.token:
            return f"{self.base_url}, brak tokenu (HOME_ASSISTANT_TOKEN)"
        return f"{self.base_url}, token ustawiony"


class HomeAssistantClient:
    """Dwie operacje: odczyt stanu i wywołanie usługi. Nic więcej.

    ``client`` można podstawić (``httpx.AsyncClient`` z ``MockTransport``), więc
    testy nie potrzebują ani serwera Home Assistanta, ani sieci.
    """

    def __init__(
        self,
        config: HomeAssistantConfig | None = None,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or HomeAssistantConfig.from_settings(settings)
        self._client = client

    @property
    def config(self) -> HomeAssistantConfig:
        return self._config

    def _headers(self) -> dict[str, str]:
        """Nagłówki żądania. JEDYNE miejsce, w którym token jest odpakowywany."""
        return {
            "Authorization": f"Bearer {self._config.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}/api/{path.lstrip('/')}"

    async def _request(self, method: str, path: str, *, json: Any | None = None) -> Any:
        if not self._config.configured:
            raise HomeAssistantError(
                "Home Assistant nie jest skonfigurowany",
                hint="ustaw HOME_ASSISTANT_URL i HOME_ASSISTANT_TOKEN w .env",
            )

        owns = self._client is None
        http = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout_s),
            follow_redirects=False,
            # Proxy z otoczenia jest tu wręcz szkodliwe: serwer stoi w sieci
            # lokalnej, a proxy firmowe nie ma jak do niego trafić.
            trust_env=False,
        )
        try:
            response = await http.request(
                method, self._url(path), headers=self._headers(), json=json
            )
        except httpx.TimeoutException as exc:
            raise HomeAssistantError(
                t("ha.timeout", seconds=f"{self._config.timeout_s:.0f}"),
                hint=t("ha.timeout_hint"),
            ) from exc
        except httpx.HTTPError as exc:
            raise HomeAssistantError(
                t("ha.connect_failed", error=redact(str(exc))),
                hint=t("ha.connect_hint", url=self._config.base_url),
            ) from exc
        finally:
            if owns:
                await http.aclose()

        return self._parse(response)

    def _parse(self, response: httpx.Response) -> Any:
        status = response.status_code
        if status in (401, 403):
            raise HomeAssistantError(
                t("ha.bad_token"),
                hint=t("ha.bad_token_hint"),
            )
        if status == 404:
            raise HomeAssistantError(
                t("ha.not_found"),
                hint=t("ha.not_found_hint"),
            )
        if status >= 500:
            raise HomeAssistantError(
                t("ha.server_error", status=status),
                hint=t("ha.server_error_hint"),
            )
        if status >= 400:
            raise HomeAssistantError(t("ha.rejected", status=status))

        if len(response.content) > MAX_RESPONSE_BYTES:
            raise HomeAssistantError(
                t("ha.too_large"),
                hint=t("ha.too_large_hint"),
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise HomeAssistantError(
                t("ha.bad_response"),
                hint=t("ha.bad_response_hint"),
            ) from exc

    # --- operacje ---------------------------------------------------------- #

    async def ping(self) -> str:
        """Sprawdź połączenie. Zwraca komunikat powitalny API."""
        payload = await self._request("GET", "/")
        if isinstance(payload, dict):
            return str(payload.get("message") or "API odpowiada")
        return "API odpowiada"

    async def state(self, entity_id: str) -> EntityState:
        payload = await self._request("GET", f"/states/{entity_id}")
        if not isinstance(payload, dict):
            raise HomeAssistantError(t("ha.unexpected_entity", entity=entity_id))
        return _entity_from_payload(payload)

    async def states(self, *, domain: str = "", limit: int | None = None) -> list[EntityState]:
        payload = await self._request("GET", "/states")
        if not isinstance(payload, list):
            raise HomeAssistantError(t("ha.unexpected_list"))

        entities = [_entity_from_payload(item) for item in payload if isinstance(item, dict)]
        wanted = domain.strip().lower()
        if wanted:
            entities = [item for item in entities if item.domain == wanted]
        entities.sort(key=lambda item: item.entity_id)
        cap = self._config.max_entities if limit is None else limit
        return entities[:cap]

    async def call_service(
        self, domain: str, service: str, *, entity_id: str = "", data: dict[str, Any] | None = None
    ) -> list[EntityState]:
        """Wywołaj usługę (``light.turn_on``). Zwraca encje, które się zmieniły."""
        body: dict[str, Any] = dict(data or {})
        if entity_id:
            body["entity_id"] = entity_id
        payload = await self._request(
            "POST", f"/services/{domain.strip().lower()}/{service.strip().lower()}", json=body
        )
        if not isinstance(payload, list):
            return []
        return [_entity_from_payload(item) for item in payload if isinstance(item, dict)]


def _entity_from_payload(payload: dict[str, Any]) -> EntityState:
    attributes = payload.get("attributes")
    return EntityState(
        entity_id=str(payload.get("entity_id") or ""),
        state=str(payload.get("state") or "unknown"),
        attributes=dict(attributes) if isinstance(attributes, dict) else {},
        changed_at=str(payload.get("last_changed") or ""),
    )


__all__ = [
    "MAX_RESPONSE_BYTES",
    "EntityState",
    "HomeAssistantClient",
    "HomeAssistantConfig",
    "HomeAssistantError",
    "normalize_base_url",
    "redact",
]
