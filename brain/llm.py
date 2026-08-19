"""Klient Ollamy: streaming odpowiedzi, prompt systemowy, czytelne błędy.

Wszystkie błędy sieciowe są tłumaczone na wyjątki :class:`LLMError`, które mają
gotowy komunikat dla użytkownika (``user_message``). Do terminala nigdy nie
trafia surowy stack trace — pełny ślad wyjątku ląduje w ``logs/errors.log``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final

import httpx

from brain.conversation import Message, select_for_model
from config import Settings, get_settings

logger = logging.getLogger(__name__)

_HTTP_OK: Final[int] = 200
_HTTP_NOT_FOUND: Final[int] = 404


class LLMError(RuntimeError):
    """Błąd komunikacji z modelem językowym nadający się do pokazania użytkownikowi."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n       Podpowiedź: {self.hint}"
        return self.message


class LLMConnectionError(LLMError):
    """Nie udało się połączyć z serwerem Ollamy."""


class LLMTimeoutError(LLMError):
    """Serwer nie odpowiedział w wyznaczonym czasie."""


class LLMModelNotFoundError(LLMError):
    """Wybrany model nie jest pobrany na tej maszynie."""


class LLMResponseError(LLMError):
    """Serwer odpowiedział, ale treść odpowiedzi jest nieprawidłowa."""


@dataclass(slots=True)
class StreamedReply:
    """Co przyszło w strumieniu poza samym tekstem (Faza 7).

    Model z obsługą tool-callingu zgłasza wywołania narzędzi w polu
    ``message.tool_calls`` — zwykle w ostatniej ramce strumienia i zwykle bez
    treści tekstowej. Zbieramy je tutaj, żeby wywołujący (``main.py``) mógł
    najpierw wykonać narzędzia, a potem poprosić model o odpowiedź końcową.

    ``content`` gromadzi tekst, który już poszedł na ekran — dzięki temu pętla
    narzędzi wie, czy model powiedział cokolwiek przed wywołaniem.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class OllamaClient:
    """Asynchroniczny klient ``/api/chat`` Ollamy.

    Użycie::

        async with OllamaClient() as client:
            async for chunk in client.stream_chat(history.messages, system=prompt):
                print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self._build_timeout(),
            headers={"Accept": "application/json"},
            trust_env=False,
        )

    # --- cykl życia ------------------------------------------------------ #

    def _build_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self._settings.ollama_connect_timeout,
            read=self._settings.ollama_read_timeout,
            write=self._settings.ollama_read_timeout,
            pool=self._settings.ollama_connect_timeout,
        )

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    # --- pomocnicze ------------------------------------------------------ #

    @property
    def model(self) -> str:
        return self._settings.ollama_model

    @property
    def host(self) -> str:
        return self._settings.ollama_host

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        stream: bool,
        system: str | None,
        tools: Sequence[dict[str, Any]] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        # Do modelu idzie OSTATNI fragment okna rozmowy, nie całe okno (patrz
        # `llm_history_max_*` w config.py). Starsze tury nie znikają: wracają
        # streszczeniem i przypomnieniem semantycznym w bloku `context`.
        sent = select_for_model(
            messages,
            max_messages=self._settings.llm_history_max_messages,
            max_chars=self._settings.llm_history_max_chars,
        )
        if len(sent) < len(messages):
            logger.debug(
                "Do modelu idzie %d z %d wiadomości okna (limit historii promptu)",
                len(sent),
                len(messages),
            )
        chat.extend(message.to_ollama() for message in sent)
        if context:
            # Rzeczy ZMIENNE co turę (godzina, wspomnienia, podpowiedź o świeżych
            # danych) idą osobną wiadomością na końcu — nie do promptu systemowego.
            #
            # Rola „user", a nie „system", i to nie jest szczegół. Szablony modeli
            # (m.in. qwen2.5) skalają WSZYSTKIE wiadomości systemowe w jeden blok
            # na początku promptu, razem z deklaracjami narzędzi. Zmienna treść
            # trafiałaby więc przed ~3400 tokenów schematów i unieważniała je przy
            # każdej turze. Zmierzone na tej maszynie (qwen2.5:7b, CPU), trzy tury:
            #   kontekst jako „system" na końcu:  40,5 s / 43,7 s / 43,3 s
            #   kontekst jako „user" na końcu:     1,1 s /  0,9 s /  1,0 s
            chat.append({"role": "user", "content": context})

        options: dict[str, Any] = {
            "temperature": self._settings.ollama_temperature,
            "num_ctx": self._settings.ollama_num_ctx,
        }
        if self._settings.ollama_max_tokens > 0:
            options["num_predict"] = self._settings.ollama_max_tokens

        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "messages": chat,
            "stream": stream,
            "keep_alive": self._settings.ollama_keep_alive,
            "options": options,
        }
        if tools:
            # Pole „tools" wysyłamy tylko wtedy, gdy naprawdę są narzędzia:
            # starsze wydania Ollamy i modele bez obsługi tool-callingu radzą
            # sobie z jego brakiem lepiej niż z pustą listą.
            payload["tools"] = list(tools)
        return payload

    def _connection_error(self, exc: Exception) -> LLMConnectionError:
        logger.error("Brak połączenia z Ollamą pod %s: %s", self.host, exc, exc_info=True)
        return LLMConnectionError(
            f"Nie mogę połączyć się z Ollamą pod adresem {self.host}.",
            hint="sprawdź, czy usługa działa (`ollama serve`) i czy OLLAMA_HOST w .env jest poprawny",
        )

    def _timeout_error(self, exc: Exception) -> LLMTimeoutError:
        logger.error("Przekroczono limit czasu Ollamy (%s): %s", self.host, exc, exc_info=True)
        return LLMTimeoutError(
            "Model nie odpowiedział w wyznaczonym czasie "
            f"({self._settings.ollama_read_timeout:.0f} s).",
            hint="zwiększ OLLAMA_READ_TIMEOUT w .env albo wybierz mniejszy model",
        )

    def _status_error(self, status_code: int, body: str) -> LLMError:
        detail = body.strip()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("error"):
                detail = str(parsed["error"])
        except json.JSONDecodeError:
            pass
        detail = detail[:500] or "brak szczegółów"

        if status_code == _HTTP_NOT_FOUND:
            logger.error("Model %s nie został znaleziony: %s", self.model, detail)
            return LLMModelNotFoundError(
                f"Model '{self.model}' nie jest dostępny w Ollamie.",
                hint=f"pobierz go poleceniem: ollama pull {self.model}",
            )

        logger.error("Ollama zwróciła HTTP %s: %s", status_code, detail)
        return LLMResponseError(
            f"Ollama zwróciła błąd HTTP {status_code}: {detail}",
            hint="szczegóły w logs/errors.log",
        )

    # --- operacje -------------------------------------------------------- #

    async def is_available(self) -> bool:
        """Czy serwer odpowiada? Nie rzuca wyjątków."""
        try:
            response = await self._client.get(
                self._settings.api_url("/api/version"),
                timeout=self._settings.ollama_connect_timeout,
            )
        except httpx.HTTPError as exc:
            logger.debug("Ollama niedostępna: %s", exc)
            return False
        return response.status_code == _HTTP_OK

    async def list_models(self) -> list[str]:
        """Lista modeli pobranych na tej maszynie."""
        try:
            response = await self._client.get(self._settings.api_url("/api/tags"))
        except httpx.ConnectError as exc:
            raise self._connection_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise self._timeout_error(exc) from exc
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc

        if response.status_code != _HTTP_OK:
            raise self._status_error(response.status_code, response.text)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "Ollama zwróciła odpowiedź, której nie da się odczytać jako JSON.",
                hint="szczegóły w logs/errors.log",
            ) from exc

        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        return [
            str(entry["name"])
            for entry in models
            if isinstance(entry, dict) and entry.get("name")
        ]

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        on_thinking: Callable[[str], None] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        collect: StreamedReply | None = None,
        context: str | None = None,
    ) -> AsyncIterator[str]:
        """Strumieniuj odpowiedź modelu fragment po fragmencie.

        Modele rozumujące (np. gemma4, qwen3, deepseek-r1) wysyłają najpierw pole
        ``message.thinking``. Nie jest ono częścią odpowiedzi — nie chcemy go
        wypowiadać ani zapisywać w historii — ale bez sygnału o nim interfejs
        wygląda na zawieszony, dlatego trafia do ``on_thinking``.

        ``tools`` to deklaracje narzędzi (Faza 7). Gdy model postanowi któreś
        wywołać, wywołania trafiają do ``collect`` — strumień tekstu jest wtedy
        zwykle pusty, a odpowiedź powstaje w drugim przejściu, już z wynikami
        narzędzi w historii.

        ``context`` to treść ZMIENNA w każdej turze (godzina, wspomnienia,
        podpowiedź o świeżych danych). Trafia osobną wiadomością na koniec, żeby
        prompt systemowy — a razem z nim kosztowne schematy narzędzi — pozostał
        identyczny i nadawał się do ponownego użycia przez serwer modelu.
        """
        url = self._settings.api_url("/api/chat")
        payload = self._payload(
            messages, stream=True, system=system, tools=tools, context=context
        )

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code != _HTTP_OK:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise self._status_error(response.status_code, body)

                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError:
                        logger.warning("Pominięto nieprawidłową linię strumienia: %r", stripped[:200])
                        continue

                    if not isinstance(data, dict):
                        continue

                    error = data.get("error")
                    if error:
                        raise self._status_error(_HTTP_OK, json.dumps({"error": error}))

                    message = data.get("message")
                    if isinstance(message, dict):
                        if on_thinking is not None:
                            thinking = message.get("thinking")
                            if isinstance(thinking, str) and thinking:
                                on_thinking(thinking)
                        if collect is not None:
                            calls = message.get("tool_calls")
                            if isinstance(calls, list):
                                collect.tool_calls.extend(
                                    item for item in calls if isinstance(item, dict)
                                )
                        chunk = message.get("content")
                        if isinstance(chunk, str) and chunk:
                            if collect is not None:
                                collect.content += chunk
                            yield chunk

                    if data.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise self._connection_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise self._timeout_error(exc) from exc
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        collect: StreamedReply | None = None,
    ) -> str:
        """Zbierz całą odpowiedź modelu w jeden łańcuch znaków."""
        chunks: list[str] = []
        async for chunk in self.stream_chat(
            messages, system=system, tools=tools, collect=collect
        ):
            chunks.append(chunk)
        return "".join(chunks)

    async def warmup(self) -> bool:
        """Wymuś załadowanie modelu do pamięci (pierwsze ładowanie potrafi trwać minutę)."""
        try:
            await self._client.post(
                self._settings.api_url("/api/chat"),
                json={
                    "model": self._settings.ollama_model,
                    "messages": [],
                    "stream": False,
                    "keep_alive": self._settings.ollama_keep_alive,
                },
            )
        except httpx.HTTPError as exc:
            logger.debug("Rozgrzewanie modelu nie powiodło się: %s", exc)
            return False
        return True


__all__ = [
    "LLMConnectionError",
    "LLMError",
    "LLMModelNotFoundError",
    "LLMResponseError",
    "LLMTimeoutError",
    "OllamaClient",
    "StreamedReply",
]
