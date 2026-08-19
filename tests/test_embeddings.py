"""Testy warstwy embeddingów (Faza 6).

Żaden test nie pobiera modelu i nie wychodzi do sieci: pakiet
``sentence_transformers`` jest podmieniany atrapą w ``sys.modules``, a klient
HTTP Ollamy — obiektem, który zwraca ustalone odpowiedzi. Testy sprawdzają kod,
nie to, co akurat jest zainstalowane na maszynie.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FAKE_EMBEDDING_DIM,
    FakeSentenceTransformer,
    FakeSentenceTransformersModule,
    make_embedding_model_dir,
)

from brain.embeddings import (
    ENGINE_OLLAMA,
    ENGINE_SENTENCE_TRANSFORMERS,
    MAX_TEXT_CHARS,
    EmbeddingError,
    EmbeddingUnavailableError,
    OllamaEmbeddingProvider,
    SentenceTransformerProvider,
    available_embedding_engines,
    create_embedding_provider,
    describe_provider,
    normalize_vector,
)
from config import Settings, find_local_embedding_model


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# Normalizacja
# --------------------------------------------------------------------------- #


def test_wektor_po_normalizacji_ma_dlugosc_jeden() -> None:
    wynik = normalize_vector([3.0, 4.0])
    assert math.isclose(math.hypot(*wynik), 1.0, rel_tol=1e-9)
    assert math.isclose(wynik[0], 0.6) and math.isclose(wynik[1], 0.8)


def test_wektor_zerowy_nie_wywraca_normalizacji() -> None:
    assert normalize_vector([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    assert normalize_vector([]) == []


# --------------------------------------------------------------------------- #
# sentence-transformers
# --------------------------------------------------------------------------- #


def test_model_jest_ladowany_dopiero_przy_pierwszym_uzyciu(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path)
    assert not provider.loaded
    assert FakeSentenceTransformer.instances == []

    vectors = provider.embed(["pierwsze zdanie"])
    assert provider.loaded
    assert len(vectors) == 1 and len(vectors[0]) == FAKE_EMBEDDING_DIM
    assert provider.dimension == FAKE_EMBEDDING_DIM


def test_wynik_modelu_jest_normalizowany(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    """Atrapa zwraca wektory o długości 3 — kod ma je sprowadzić do 1."""
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path)
    vector = provider.embed(["kawa z mlekiem"])[0]
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-6)


def test_model_z_dysku_wymusza_prace_bez_sieci(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    make_embedding_model_dir(tmp_path)
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path)
    provider.embed(["cokolwiek"])

    instance = FakeSentenceTransformer.instances[0]
    assert instance.local_files_only is True
    assert "snapshots" in instance.model_name_or_path


def test_brak_modelu_i_zakaz_pobierania_daje_czytelny_blad(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    provider = SentenceTransformerProvider(
        "atrapa-modelu", cache_dir=tmp_path, allow_download=False
    )
    with pytest.raises(EmbeddingUnavailableError) as blad:
        provider.embed(["cokolwiek"])
    assert "prepare_offline" in blad.value.user_message


def test_brak_pakietu_daje_czytelny_blad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path)
    with pytest.raises(EmbeddingUnavailableError) as blad:
        provider.embed(["cokolwiek"])
    assert "EMBEDDING_ENGINE=ollama" in blad.value.user_message


def test_dlugi_tekst_jest_przycinany(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path)
    provider.embed(["a" * (MAX_TEXT_CHARS * 2)])
    wyslany = FakeSentenceTransformer.instances[0].encoded[0][0]
    assert len(wyslany) == MAX_TEXT_CHARS


def test_pusta_lista_nie_laduje_modelu(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path)
    assert provider.embed([]) == []
    assert not provider.loaded


def test_model_na_dysku_jest_odnajdywany(tmp_path: Path) -> None:
    make_embedding_model_dir(tmp_path, "wielojezyczny-maly")
    znaleziony = find_local_embedding_model("wielojezyczny-maly", tmp_path)
    assert znaleziony is not None and (znaleziony / "modules.json").is_file()
    # Nazwa z organizacją wskazuje ten sam model.
    assert find_local_embedding_model("sentence-transformers/wielojezyczny-maly", tmp_path)
    assert find_local_embedding_model("zupelnie-inny", tmp_path) is None


def test_niekompletny_katalog_nie_uchodzi_za_model(tmp_path: Path) -> None:
    """Przerwane pobieranie zostawia strukturę katalogów bez wag."""
    directory = tmp_path / "models--sentence-transformers--polowiczny" / "snapshots" / "abc"
    directory.mkdir(parents=True)
    (directory / "modules.json").write_text("[]", encoding="utf-8")
    assert find_local_embedding_model("polowiczny", tmp_path) is None


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload


class FakeHttpClient:
    """Atrapa ``httpx.Client`` — zapisuje żądania i oddaje ustalone odpowiedzi."""

    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.requests.append((url, json))
        for path, response in self.responses.items():
            if url.endswith(path):
                return response
        raise AssertionError(f"nieoczekiwany adres: {url}")

    def close(self) -> None:
        self.closed = True


def test_ollama_liczy_embeddingi_wsadowo() -> None:
    client = FakeHttpClient(
        {"/api/embed": FakeResponse(200, {"embeddings": [[3.0, 4.0], [1.0, 0.0]]})}
    )
    provider = OllamaEmbeddingProvider(make_settings(), client=client)

    vectors = provider.embed(["pierwszy", "drugi"])
    assert len(vectors) == 2
    assert math.isclose(math.hypot(*vectors[0]), 1.0, rel_tol=1e-6)
    assert provider.dimension == 2
    # Jedno żądanie na całą partię, nie jedno na tekst.
    assert len(client.requests) == 1
    assert client.requests[0][1]["input"] == ["pierwszy", "drugi"]


def test_starsza_ollama_bez_api_embed_dziala_dalej() -> None:
    """Brak ``/api/embed`` ma przełączyć na starszy endpoint, a nie wywalić błąd."""
    client = FakeHttpClient(
        {
            "/api/embed": FakeResponse(404, text="not found"),
            "/api/embeddings": FakeResponse(200, {"embedding": [0.0, 2.0]}),
        }
    )
    provider = OllamaEmbeddingProvider(make_settings(), client=client)

    vectors = provider.embed(["pierwszy", "drugi"])
    assert len(vectors) == 2
    # Drugie wywołanie nie próbuje już wsadowego endpointu.
    provider.embed(["trzeci"])
    adresy = [url for url, _ in client.requests]
    assert sum(1 for url in adresy if url.endswith("/api/embed")) == 1


def test_brak_modelu_w_ollamie_daje_podpowiedz() -> None:
    client = FakeHttpClient(
        {
            "/api/embed": FakeResponse(404, text="model not found"),
            "/api/embeddings": FakeResponse(404, text="model not found"),
        }
    )
    provider = OllamaEmbeddingProvider(make_settings(), client=client)
    with pytest.raises(EmbeddingUnavailableError) as blad:
        provider.embed(["cokolwiek"])
    assert "ollama pull" in blad.value.user_message


def test_odpowiedz_bez_embeddingow_jest_bledem() -> None:
    client = FakeHttpClient({"/api/embed": FakeResponse(200, {"co": "innego"})})
    provider = OllamaEmbeddingProvider(make_settings(), client=client)
    with pytest.raises(EmbeddingError):
        provider.embed(["cokolwiek"])


def test_padnieta_ollama_daje_czytelny_blad() -> None:
    class ZrywajacyClient:
        def post(self, url: str, json: dict[str, Any]) -> Any:
            raise OSError("connection refused")

    provider = OllamaEmbeddingProvider(make_settings(), client=ZrywajacyClient())
    with pytest.raises(EmbeddingUnavailableError) as blad:
        provider.embed(["cokolwiek"])
    assert "ollama serve" in blad.value.user_message


def test_zdalny_host_ollamy_jest_zglaszany_jako_wyciek(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Obietnica „nic nie opuszcza maszyny" znika, gdy ktoś wskaże inny komputer."""
    import logging

    with caplog.at_level(logging.WARNING, logger="brain.embeddings"):
        provider = OllamaEmbeddingProvider(make_settings(ollama_host="http://192.168.0.50:11434"))

    assert "nie jest adresem tej maszyny" in caplog.text
    assert "UWAGA" in provider.describe()


def test_lokalny_host_nie_wywoluje_ostrzezenia(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="brain.embeddings"):
        provider = OllamaEmbeddingProvider(make_settings(ollama_host="http://localhost:11434"))

    assert caplog.text == ""
    assert "UWAGA" not in provider.describe()


def test_urzadzenie_obliczeniowe_jest_przekazywane_modelowi(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    """Wpisana wprost wartość idzie do modelu bez zmian („mps" na układach Apple)."""
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path, device="mps")
    provider.embed(["cokolwiek"])
    assert FakeSentenceTransformer.instances[0].device == "mps"

    settings = make_settings(embedding_engine="sentence-transformers", embedding_device="mps")
    zbudowany = create_embedding_provider(settings)
    assert isinstance(zbudowany, SentenceTransformerProvider)


def test_auto_zostawia_wybor_urzadzenia_bibliotece(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    """``auto`` nie może narzucać CUDA: nvidia-smi mówi o sterowniku, nie o PyTorchu.

    Na maszynie z kartą NVIDII i PyTorchem zbudowanym bez CUDA wymuszone
    ``device="cuda"`` wywala ładowanie modelu — a wtedy pamięć semantyczna
    wyłącza się na komputerze, na którym miała działać najlepiej.
    """
    provider = SentenceTransformerProvider("atrapa-modelu", cache_dir=tmp_path, device="auto")
    provider.embed(["cokolwiek"])
    assert FakeSentenceTransformer.instances[0].device is None


def test_ollama_pyta_wylacznie_wskazany_host() -> None:
    """Prywatność: embeddingi nie mogą iść nigdzie poza skonfigurowany adres."""
    client = FakeHttpClient({"/api/embed": FakeResponse(200, {"embeddings": [[1.0]]})})
    settings = make_settings(ollama_host="http://127.0.0.1:11434")
    OllamaEmbeddingProvider(settings, client=client).embed(["tajna treść"])

    url = client.requests[0][0]
    assert url.startswith("http://127.0.0.1:11434")


# --------------------------------------------------------------------------- #
# Wybór silnika
# --------------------------------------------------------------------------- #


def test_auto_wybiera_sentence_transformers_gdy_jest(
    fake_sentence_transformers: FakeSentenceTransformersModule,
) -> None:
    assert ENGINE_SENTENCE_TRANSFORMERS in available_embedding_engines()
    provider = create_embedding_provider(make_settings(embedding_engine="auto"))
    assert isinstance(provider, SentenceTransformerProvider)


def test_auto_spada_na_ollame_bez_pakietu(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    provider = create_embedding_provider(make_settings(embedding_engine="auto"))
    assert isinstance(provider, OllamaEmbeddingProvider)


def test_brak_silnikow_wylacza_pamiec_semantyczna(monkeypatch: pytest.MonkeyPatch) -> None:
    import brain.embeddings

    monkeypatch.setattr(brain.embeddings, "available_embedding_engines", list)
    assert create_embedding_provider(make_settings(embedding_engine="auto")) is None


def test_wylaczenie_w_ustawieniach_jest_respektowane(
    fake_sentence_transformers: FakeSentenceTransformersModule,
) -> None:
    assert create_embedding_provider(make_settings(embeddings_enabled=False)) is None
    assert create_embedding_provider(make_settings(embedding_engine="none")) is None


def test_wymuszony_silnik_bez_pakietu_nie_podmienia_po_cichu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skoro użytkownik wskazał silnik wprost, nie wybieramy za niego innego."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    settings = make_settings(embedding_engine="sentence-transformers")
    assert create_embedding_provider(settings) is None


def test_wymuszona_ollama_dziala_bez_sentence_transformers() -> None:
    provider = create_embedding_provider(make_settings(embedding_engine=ENGINE_OLLAMA))
    assert isinstance(provider, OllamaEmbeddingProvider)


def test_tryb_offline_zabrania_pobierania(
    fake_sentence_transformers: FakeSentenceTransformersModule, tmp_path: Path
) -> None:
    settings = make_settings(embedding_engine="sentence-transformers", offline_mode="on")
    provider = create_embedding_provider(settings)
    assert isinstance(provider, SentenceTransformerProvider)
    with pytest.raises(EmbeddingUnavailableError):
        provider.embed(["cokolwiek"])


def test_opis_dostawcy_nadaje_sie_do_statusu(
    fake_sentence_transformers: FakeSentenceTransformersModule,
) -> None:
    provider = create_embedding_provider(make_settings(embedding_engine="auto"))
    assert "sentence-transformers" in describe_provider(provider)
    assert describe_provider(None) == "wyłączona"
