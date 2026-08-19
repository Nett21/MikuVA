"""Embeddingi liczone LOKALNIE — zamiana tekstu na wektor znaczenia (Faza 6).

Zasada nadrzędna tego modułu: **treść rozmów nigdy nie opuszcza tej maszyny.**
Nie ma tu — i nie będzie — dostawcy chmurowego. Embeddingi to nie „anonimowe
liczby": z wektora da się odtworzyć sens zdania, a często i samo zdanie, więc
wysłanie ich do zewnętrznego API byłoby wysłaniem prywatnych rozmów. Dostępne
warianty liczą wszystko u siebie:

* ``sentence-transformers`` — model uruchamiany w tym procesie, na CPU albo GPU,
* ``ollama`` — model w usłudze Ollama, która i tak stoi na ``127.0.0.1``
  (wariant dla osób, które nie chcą instalować PyTorcha).

Wybór silnika nie jest wpisany w kod: ``EMBEDDING_ENGINE=auto`` bierze
sentence-transformers, jeśli pakiet jest zainstalowany, a w przeciwnym razie
Ollamę. Brak obu = pamięć semantyczna wyłączona, reszta asystenta działa
normalnie (pamięć z Fazy 5 zostaje).

Interfejs to :class:`EmbeddingProvider` — dzięki niemu testy podstawiają atrapę
i nie pobierają żadnego modelu.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from config import (
    EMBEDDINGS_DIR,
    Settings,
    find_local_embedding_model,
    get_settings,
    is_local_host,
    is_offline,
    resolve_compute_device,
)
from i18n import t

logger = logging.getLogger(__name__)

# Nazwy silników. Zwykłe łańcuchy, bo kolejny dostawca ma się dodawać funkcją
# fabrykującą, a nie zmianą typu w pół projektu.
ENGINE_SENTENCE_TRANSFORMERS: Final[str] = "sentence-transformers"
ENGINE_OLLAMA: Final[str] = "ollama"

# Dłuższe teksty i tak nie mieszczą się w oknie modelu (typowo 128–512 tokenów),
# a samo kodowanie kosztuje. Ucinamy zawczasu i mówimy o tym w logu.
MAX_TEXT_CHARS: Final[int] = 4_000


class EmbeddingError(RuntimeError):
    """Błąd liczenia embeddingów nadający się do pokazania użytkownikowi."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


class EmbeddingUnavailableError(EmbeddingError):
    """Na tej maszynie nie da się liczyć embeddingów (brak pakietu, modelu, usługi)."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimum, którego pamięć semantyczna potrzebuje od modelu embeddingów."""

    @property
    def name(self) -> str:
        """Nazwa modelu — trafia do bazy przy każdym wektorze."""

    @property
    def dimension(self) -> int:
        """Liczba wymiarów wektora (0 = jeszcze nie wiadomo, model nieładowany)."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Policz wektory dla listy tekstów (znormalizowane do długości 1)."""


def normalize_vector(values: Sequence[float]) -> list[float]:
    """Sprowadź wektor do długości 1.

    Po normalizacji podobieństwo kosinusowe to zwykły iloczyn skalarny — dzięki
    temu ani indeks, ani zapytania nie muszą liczyć długości wektorów.
    Wektor zerowy (pusty tekst, zdegenerowany model) zwracamy bez zmian.
    """
    total = math.fsum(float(item) * float(item) for item in values)
    if total <= 0.0:
        return [float(item) for item in values]
    length = math.sqrt(total)
    return [float(item) / length for item in values]


def _clean_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > MAX_TEXT_CHARS:
        logger.debug(
            "Tekst do embeddingu skrócony z %s do %s znaków.", len(cleaned), MAX_TEXT_CHARS
        )
        return cleaned[:MAX_TEXT_CHARS]
    return cleaned


# --------------------------------------------------------------------------- #
# sentence-transformers — model w tym procesie
# --------------------------------------------------------------------------- #


class SentenceTransformerProvider:
    """Model ``sentence-transformers`` ładowany do pamięci procesu.

    Model jest ładowany LENIWIE, przy pierwszym użyciu: import PyTorcha i wczytanie
    wag to kilka sekund, a nie każda sesja asystenta w ogóle sięga do pamięci
    semantycznej. Ładowanie jest chronione blokadą — do pamięci sięga też wątek
    syntezatora mowy.
    """

    def __init__(
        self,
        model: str,
        *,
        cache_dir: Path | None = None,
        device: str = "auto",
        batch_size: int = 16,
        allow_download: bool = True,
    ) -> None:
        self._model_name = model.strip()
        self._cache_dir = cache_dir or EMBEDDINGS_DIR
        self._device = device
        self._batch_size = max(1, batch_size)
        self._allow_download = allow_download
        self._model: Any | None = None
        self._dimension = 0
        self._lock = threading.RLock()

    # --- właściwości ----------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> str:
        where = "załadowany" if self.loaded else "zostanie załadowany przy pierwszym użyciu"
        return f"sentence-transformers: {self._model_name} ({where})"

    # --- ładowanie ------------------------------------------------------- #

    def load(self) -> None:
        """Wczytaj model. Powtórne wywołanie nic nie robi."""
        with self._lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    t("emb.no_package"),
                    hint=t("emb.no_package_hint"),
                ) from exc

            local = find_local_embedding_model(self._model_name, self._cache_dir)
            if local is None and not self._allow_download:
                raise EmbeddingUnavailableError(
                    t("emb.model_missing", model=self._model_name),
                    hint=t("emb.download_hint"),
                )

            # Ścieżka lokalna ma pierwszeństwo: gdy model już jest, biblioteka nie
            # pyta o nic sieci, niezależnie od zmiennych HF_*.
            target = str(local) if local is not None else self._model_name
            device = self._resolve_device()

            try:
                model = SentenceTransformer(
                    target,
                    cache_folder=str(self._cache_dir),
                    device=device,
                    local_files_only=local is not None or not self._allow_download,
                )
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    t("emb.load_failed", model=self._model_name, error=exc),
                    hint=t("emb.load_hint"),
                ) from exc

            self._model = model
            self._dimension = int(self._probe_dimension(model))
            logger.info(
                "Model embeddingów %s gotowy (%s wymiarów, urządzenie %s).",
                self._model_name,
                self._dimension,
                device or "wybrane przez bibliotekę",
            )

    def _resolve_device(self) -> str | None:
        """Urządzenie dla modelu. ``None`` = niech zdecyduje sama biblioteka.

        Przy ``EMBEDDING_DEVICE=auto`` świadomie NIE narzucamy urządzenia i nie
        pytamy o nie ``resolve_compute_device()``. Powód jest praktyczny: tamta
        funkcja wykrywa GPU po obecności ``nvidia-smi``, a to mówi tylko o
        sterowniku — na maszynie z kartą NVIDII i PyTorchem w wersji CPU-only
        ``device="cuda"`` wywala ładowanie modelu i wyłącza pamięć semantyczną.
        Biblioteka pyta samego PyTorcha, więc wybierze to, co naprawdę działa
        (także ``mps`` na układach Apple i ROCm na kartach AMD).

        Wpisana wprost wartość jest przekazywana bez zmian — kto ustawił
        ``EMBEDDING_DEVICE``, ten wie, co ma w komputerze.
        """
        if self._device.strip().lower() in ("", "auto"):
            return None
        return resolve_compute_device(self._device)

    @staticmethod
    def _probe_dimension(model: Any) -> int:
        """Wymiar wektora — z API modelu, a w ostateczności z próbnego zdania."""
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return int(value)
            except Exception:  # pragma: no cover - nietypowa implementacja modelu
                pass
        probe = model.encode(["test"], convert_to_numpy=False)
        return len(list(probe[0]))

    # --- liczenie -------------------------------------------------------- #

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [_clean_text(text) for text in texts]
        if not prepared:
            return []
        self.load()
        model = self._model
        if model is None:  # pragma: no cover - load() rzuca wcześniej
            raise EmbeddingUnavailableError(t("emb.not_loaded"))

        try:
            raw = model.encode(
                prepared,
                batch_size=self._batch_size,
                convert_to_numpy=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                t("emb.compute_failed", error=exc),
                hint=t("emb.details_in_log"),
            ) from exc

        return [normalize_vector([float(value) for value in vector]) for vector in raw]


# --------------------------------------------------------------------------- #
# Ollama — model w usłudze na 127.0.0.1
# --------------------------------------------------------------------------- #


class OllamaEmbeddingProvider:
    """Embeddingi z lokalnej usługi Ollama (``/api/embed``).

    Wariant dla maszyn, na których nie chcemy PyTorcha: Ollama musi działać i
    tak, więc to zero dodatkowych zależności. Ruch idzie wyłącznie na adres z
    ``OLLAMA_HOST`` — domyślnie ``127.0.0.1``, czyli nigdzie poza tę maszynę.
    """

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.embedding_ollama_model.strip()
        # Obietnica „nic nie opuszcza tej maszyny" trzyma się tylko dopóki
        # OLLAMA_HOST wskazuje na tę maszynę. Kto przestawi go na inny komputer,
        # ma o tym usłyszeć wprost — z wektora da się odtworzyć treść zdania.
        self._local = is_local_host(self._settings.ollama_host)
        if not self._local:
            logger.warning(
                "OLLAMA_HOST=%s nie jest adresem tej maszyny — treść wspomnień będzie "
                "wysyłana do policzenia embeddingów na inny komputer. Jeśli to nie było "
                "zamierzone, ustaw OLLAMA_HOST=http://127.0.0.1:11434 albo "
                "EMBEDDING_ENGINE=sentence-transformers.",
                self._settings.ollama_host,
            )
        self._client = client
        self._owns_client = client is None
        self._dimension = 0
        # Nowsze wydania Ollamy mają /api/embed (wsadowe), starsze tylko
        # /api/embeddings (jeden tekst na raz). Ustalamy to raz, przy pierwszym
        # użyciu, zamiast zakładać wersję usługi.
        self._batch_endpoint: bool | None = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return f"ollama/{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def describe(self) -> str:
        where = "" if self._local else "  UWAGA: to nie jest adres tej maszyny!"
        return f"Ollama: {self._model_name} @ {self._settings.ollama_host}{where}"

    def _http(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx jest wymaganym pakietem
            raise EmbeddingUnavailableError(
                t("emb.no_httpx", error=exc)
            ) from exc
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=self._settings.ollama_connect_timeout,
                read=self._settings.ollama_read_timeout,
                write=self._settings.ollama_read_timeout,
                pool=self._settings.ollama_connect_timeout,
            ),
            # Ollama stoi lokalnie — proxy z otoczenia tylko by przeszkadzało.
            trust_env=False,
        )
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [_clean_text(text) for text in texts]
        if not prepared:
            return []

        with self._lock:
            if self._batch_endpoint is None:
                self._batch_endpoint = True
            vectors = (
                self._embed_batch(prepared)
                if self._batch_endpoint
                else [self._embed_single(text) for text in prepared]
            )

        if vectors:
            self._dimension = len(vectors[0])
        return [normalize_vector(vector) for vector in vectors]

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            return self._http().post(self._settings.api_url(path), json=payload)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                t("emb.ollama_failed", host=self._settings.ollama_host),
                hint=t("emb.ollama_hint", model=self._model_name),
            ) from exc

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._post("/api/embed", {"model": self._model_name, "input": list(texts)})
        if response.status_code == 404:
            # Starsza Ollama nie zna /api/embed — przechodzimy na wariant po jednym.
            logger.info("Ollama bez /api/embed — używam starszego /api/embeddings.")
            self._batch_endpoint = False
            return [self._embed_single(text) for text in texts]
        self._raise_for_status(response)

        data = response.json()
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError(
                t("emb.no_embeddings_field"),
                hint=t("emb.no_embeddings_hint", model=self._model_name),
            )
        return [[float(value) for value in vector] for vector in vectors]

    def _embed_single(self, text: str) -> list[float]:
        response = self._post("/api/embeddings", {"model": self._model_name, "prompt": text})
        self._raise_for_status(response)
        data = response.json()
        vector = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError(t("emb.no_embedding_field"))
        return [float(value) for value in vector]

    def _raise_for_status(self, response: Any) -> None:
        if response.status_code == 200:
            return
        detail = str(getattr(response, "text", ""))[:300]
        if response.status_code == 404:
            raise EmbeddingUnavailableError(
                t("emb.ollama_model_missing", model=self._model_name),
                hint=t("emb.pull_hint", model=self._model_name),
            )
        raise EmbeddingError(
            t("emb.ollama_http", status=response.status_code, body=detail)
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - klient bywa już zamknięty
                pass
            self._client = None


# --------------------------------------------------------------------------- #
# Wybór silnika
# --------------------------------------------------------------------------- #


def _module_available(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def available_embedding_engines() -> list[str]:
    """Silniki, które da się uruchomić na TEJ maszynie (kolejność = preferencja)."""
    engines: list[str] = []
    if _module_available("sentence_transformers"):
        engines.append(ENGINE_SENTENCE_TRANSFORMERS)
    # Ollamy nie odpytujemy tutaj: sprawdzenie kosztuje żądanie HTTP, a i tak
    # dowiemy się przy pierwszym użyciu. Wystarczy, że jest czym z nią rozmawiać.
    if _module_available("httpx"):
        engines.append(ENGINE_OLLAMA)
    return engines


def create_embedding_provider(
    settings: Settings | None = None,
) -> EmbeddingProvider | None:
    """Zbuduj dostawcę embeddingów zgodnie z konfiguracją.

    ``None`` = pamięć semantyczna jest na tej maszynie niedostępna albo wyłączona.
    Funkcja NIE ładuje modelu — to dzieje się dopiero przy pierwszym liczeniu,
    żeby start asystenta nie czekał kilku sekund na PyTorcha.
    """
    active = settings or get_settings()
    if not active.embeddings_enabled:
        return None

    requested = active.embedding_engine
    if requested == "none":
        return None

    available = available_embedding_engines()

    if requested == ENGINE_SENTENCE_TRANSFORMERS:
        if ENGINE_SENTENCE_TRANSFORMERS not in available:
            logger.warning(
                "EMBEDDING_ENGINE=sentence-transformers, ale pakietu nie ma — "
                "pamięć semantyczna wyłączona."
            )
            return None
        return _make_sentence_transformer(active)

    if requested == ENGINE_OLLAMA:
        return OllamaEmbeddingProvider(active)

    # auto: własny model, jeśli jest czym go uruchomić; inaczej Ollama.
    if ENGINE_SENTENCE_TRANSFORMERS in available:
        return _make_sentence_transformer(active)
    if ENGINE_OLLAMA in available:
        logger.info(
            "Brak pakietu sentence-transformers — embeddingi policzy Ollama (%s).",
            active.embedding_ollama_model,
        )
        return OllamaEmbeddingProvider(active)

    logger.info("Żaden silnik embeddingów nie jest dostępny — pamięć semantyczna wyłączona.")
    return None


def _make_sentence_transformer(settings: Settings) -> SentenceTransformerProvider:
    return SentenceTransformerProvider(
        settings.embedding_model,
        cache_dir=EMBEDDINGS_DIR,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        # W trybie offline nie pobieramy niczego, niezależnie od ustawienia.
        allow_download=settings.embedding_allow_download and not is_offline(settings),
    )


def describe_provider(provider: EmbeddingProvider | None) -> str:
    """Jedna linijka do ``/status`` i do raportu zależności."""
    if provider is None:
        return "wyłączona"
    describe = getattr(provider, "describe", None)
    if callable(describe):
        return str(describe())
    return provider.name


__all__ = [
    "ENGINE_OLLAMA",
    "ENGINE_SENTENCE_TRANSFORMERS",
    "MAX_TEXT_CHARS",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingUnavailableError",
    "OllamaEmbeddingProvider",
    "SentenceTransformerProvider",
    "available_embedding_engines",
    "create_embedding_provider",
    "describe_provider",
    "normalize_vector",
]
