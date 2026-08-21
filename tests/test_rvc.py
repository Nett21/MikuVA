"""Konwersja głosu RVC (Faza 15) — bez modelu, bez GPU, bez sieci.

Prawdziwego modelu RVC nie ma tu ani jednego bajtu i nigdy nie będzie: to
setki megabajtów cudzej własności (patrz sekcja Ograniczeń w README). Backend
jest więc atrapą, a testy pilnują tego, co i tak jest w tej fazie najważniejsze
— **że awaria RVC kończy się głosem Pipera, a nie ciszą**.

Każdy sposób, na jaki RVC może zawieść, ma tu swój test: wyłączone w
ustawieniach, brak pliku modelu, brak backendu, wyjątek w środku wypowiedzi,
przekroczenie limitu czasu. We wszystkich pięciu przypadkach z dostawcy MUSI
wyjść komplet dźwięku.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from conftest import make_tone

import audio.dependencies
from audio import rvc as rvc_module
from audio.rvc import RvcUnavailableError, read_wav, resample, resolve_rvc_device, write_wav
from audio.tts import SpeechChunk, TTSProvider, create_tts_provider
from audio.tts_rvc import RvcVoiceProvider
from config import (
    DependencyContext,
    GPUInfo,
    OllamaStatus,
    RVCSettings,
    Settings,
    UserSettings,
    detect_platform,
)
from i18n import t

BASE_RATE = 22_050
RVC_RATE = 40_000


# --------------------------------------------------------------------------- #
# Atrapy
# --------------------------------------------------------------------------- #


class FakeBaseProvider(TTSProvider):
    """Dostawca „Pipera": oddaje policzalne fragmenty, notuje wywołania."""

    name = "fake-piper"

    def __init__(self, *, chunks: int = 8, samples_per_chunk: int = 441) -> None:
        self.chunks = chunks
        self.samples_per_chunk = samples_per_chunk
        self.calls: list[str] = []
        self.loaded = False
        self.cancelled = False
        self.error: Exception | None = None

    @property
    def sample_rate(self) -> int:
        return BASE_RATE

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def voice_name(self) -> str:
        return "fake-voice"

    def cancel(self) -> None:
        self.cancelled = True

    def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        for index in range(self.chunks):
            yield SpeechChunk(
                samples=make_tone(self.samples_per_chunk, frequency=110.0 * (index + 1)),
                sample_rate=BASE_RATE,
            )

    def total_samples(self) -> int:
        return self.chunks * self.samples_per_chunk


class FakeRvcBackend:
    """Backend RVC, który da się zepsuć na życzenie."""

    name = "fake-rvc"

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.params: list[tuple[int, float]] = []
        self.closed = False
        self.fail_after: int | None = None
        self.error: Exception = RuntimeError("model zniknął")

    def convert(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        pitch_shift: int,
        index_rate: float,
    ) -> tuple[np.ndarray, int]:
        if self.fail_after is not None and len(self.blocks) >= self.fail_after:
            raise self.error
        self.blocks.append(np.asarray(samples))
        self.params.append((pitch_shift, index_rate))
        # Konwersja udaje zmianę częstotliwości — tak jak robi to prawdziwe RVC.
        converted = resample(samples, sample_rate, RVC_RATE)
        return (converted // 2).astype(np.int16), RVC_RATE

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeRvcBackend:
    """Podstaw atrapę pod backend wskazywany przez ``RVC_BACKEND``.

    Rejestrujemy prawdziwy moduł w ``sys.modules``, bo dostawca ładuje backend
    przez ``importlib.import_module`` — testujemy realną drogę ładowania,
    a nie obejście.
    """
    backend = FakeRvcBackend()
    module = types.ModuleType("atrapa_rvc")
    module.__spec__ = importlib.machinery.ModuleSpec("atrapa_rvc", loader=None)
    module.create_backend = lambda **kwargs: backend  # type: ignore[attr-defined]
    module.last_kwargs = {}  # type: ignore[attr-defined]

    def create_backend(**kwargs: object) -> FakeRvcBackend:
        module.last_kwargs = kwargs  # type: ignore[attr-defined]
        return backend

    module.create_backend = create_backend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atrapa_rvc", module)
    backend.module = module  # type: ignore[attr-defined]
    return backend


def gpu_info(*, available: bool, name: str = "") -> GPUInfo:
    """GPUInfo bez wartości domyślnych — testy nie mogą pytać o prawdziwą kartę."""
    return GPUInfo(
        cuda_available=available,
        source="nvidia-smi" if available else "none",
        device_name=name or None,
        driver_version="999.99" if available else None,
        detail="atrapa",
    )


def make_model_files(root: Path) -> tuple[Path, Path]:
    """Puste pliki o właściwych nazwach — kod sprawdza istnienie, nie zawartość."""
    model = root / "miku.pth"
    index = root / "miku.index"
    model.write_bytes(b"to nie jest model")
    index.write_bytes(b"to nie jest indeks")
    return model, index


def rvc_settings(root: Path, **overrides: object) -> UserSettings:
    model, index = make_model_files(root)
    data: dict[str, object] = {
        "enabled": True,
        "model_path": str(model),
        "index_path": str(index),
        "pitch_shift": 12,
        "index_rate": 0.5,
    }
    data.update(overrides)
    return UserSettings(voice_engine="rvc_miku", rvc=RVCSettings(**data))  # type: ignore[arg-type]


def rvc_config(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "piper_voices_dir": str(tmp_path / "voices"),
        "rvc_backend": "atrapa_rvc",
        "rvc_device": "cpu",
        "rvc_chunk_min_ms": 100,
        "rvc_chunk_max_ms": 1_000,
        "rvc_latency_target_ms": 60_000,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def collect(provider: TTSProvider, text: str = "Cześć, tu Miku.") -> list[SpeechChunk]:
    return [chunk for chunk in provider.synthesize(text) if not chunk.is_empty]


# --------------------------------------------------------------------------- #
# Narzędzia sygnałowe
# --------------------------------------------------------------------------- #


def test_resample_zmienia_dlugosc_proporcjonalnie() -> None:
    signal = make_tone(1_000)
    converted = resample(signal, 22_050, 44_100)
    assert converted.dtype == np.int16
    assert abs(converted.size - 2_000) <= 2


def test_resample_bez_zmiany_czestotliwosci_nie_rusza_probek() -> None:
    signal = make_tone(500)
    assert np.array_equal(resample(signal, 16_000, 16_000), signal)


def test_zapis_i_odczyt_wav_zachowuje_sygnal(tmp_path: Path) -> None:
    signal = make_tone(1_600)
    path = tmp_path / "probka.wav"
    write_wav(path, signal, 16_000)
    odczytane, rate = read_wav(path)
    assert rate == 16_000
    assert np.array_equal(odczytane, signal)


# --------------------------------------------------------------------------- #
# Wybór urządzenia
# --------------------------------------------------------------------------- #


def test_bez_gpu_liczymy_na_cpu(tmp_path: Path) -> None:
    device = resolve_rvc_device(
        rvc_config(tmp_path, rvc_device="auto"),
        gpu_info(available=False),
    )
    assert device.is_cpu
    assert device.name == "cpu"


def test_z_gpu_wybieramy_karte_z_numerem(tmp_path: Path) -> None:
    """Backendy oparte o torch potrzebują indeksu karty — samo „cuda" bywa ignorowane."""
    device = resolve_rvc_device(
        rvc_config(tmp_path, rvc_device="auto"),
        gpu_info(available=True, name="RTX 4070"),
    )
    assert device.name == "cuda:0"
    assert not device.is_cpu
    assert "RTX 4070" in device.describe()


def test_ustawienie_wymusza_cpu_mimo_dostepnego_gpu(tmp_path: Path) -> None:
    device = resolve_rvc_device(
        rvc_config(tmp_path, rvc_device="cpu"),
        gpu_info(available=True, name="RTX 4070"),
    )
    assert device.name == "cpu"


# --------------------------------------------------------------------------- #
# Ścieżka szczęśliwa
# --------------------------------------------------------------------------- #


def test_dzwiek_przechodzi_przez_rvc(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    base = FakeBaseProvider()
    provider = RvcVoiceProvider(rvc_config(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    chunks = collect(provider)

    assert provider.is_active()
    assert fake_backend.blocks, "backend nie dostał ani jednego fragmentu"
    assert all(chunk.sample_rate == RVC_RATE for chunk in chunks)


def test_parametry_ida_z_ustawien_uzytkownika(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    """Pitch i index rate NIE są zaszyte — pochodzą z config/user_settings.json."""
    base = FakeBaseProvider()
    user = rvc_settings(tmp_path, pitch_shift=-7, index_rate=0.25)
    provider = RvcVoiceProvider(rvc_config(tmp_path), user, base=base)
    provider.load()
    collect(provider)

    assert fake_backend.params
    assert set(fake_backend.params) == {(-7, 0.25)}


def test_sciezki_modelu_trafiaja_do_backendu(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    base = FakeBaseProvider()
    user = rvc_settings(tmp_path)
    provider = RvcVoiceProvider(rvc_config(tmp_path), user, base=base)
    provider.load()

    kwargs = fake_backend.module.last_kwargs  # type: ignore[attr-defined]
    assert kwargs["model_path"] == user.rvc.resolved_model_path
    assert kwargs["index_path"] == user.rvc.resolved_index_path
    assert kwargs["device"] == "cpu"


def test_cala_wypowiedz_ma_jedna_czestotliwosc(
    tmp_path: Path, fake_backend: FakeRvcBackend
) -> None:
    """Zmiana częstotliwości w trakcie zdania oznaczałaby ponowne otwarcie strumienia."""
    base = FakeBaseProvider(chunks=12)
    provider = RvcVoiceProvider(rvc_config(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    rates = {chunk.sample_rate for chunk in collect(provider)}
    assert len(rates) == 1


def test_nazwa_glosu_nie_zdradza_sciezki(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    """Ścieżka do modelu zawiera katalog domowy — do /status idzie sama nazwa pliku."""
    provider = RvcVoiceProvider(
        rvc_config(tmp_path), rvc_settings(tmp_path), base=FakeBaseProvider()
    )
    provider.load()

    opis = provider.describe()
    assert "miku" in opis
    assert str(tmp_path) not in opis


# --------------------------------------------------------------------------- #
# Buforowanie i opóźnienie
# --------------------------------------------------------------------------- #


def test_fragmenty_sa_sklejane_do_minimalnej_dlugosci(
    tmp_path: Path, fake_backend: FakeRvcBackend
) -> None:
    """Konwersja 20-milisekundowych ramek brzmi jak bulgot — zbieramy je najpierw."""
    settings = rvc_config(tmp_path, rvc_chunk_min_ms=200)
    base = FakeBaseProvider(chunks=20, samples_per_chunk=441)  # 20 ms na fragment
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)
    provider.load()
    collect(provider)

    minimum = BASE_RATE * 200 // 1000
    # Ostatni blok to reszta po zakończeniu zdania — ten może być krótszy.
    assert all(block.size >= minimum for block in fake_backend.blocks[:-1])
    assert len(fake_backend.blocks) < base.chunks


def test_pierwszy_fragment_leci_przed_koncem_zdania(
    tmp_path: Path, fake_backend: FakeRvcBackend
) -> None:
    """Sedno fazy: nie czekamy na całą wypowiedź, żeby zacząć grać."""
    settings = rvc_config(tmp_path, rvc_chunk_min_ms=100)
    base = FakeBaseProvider(chunks=20, samples_per_chunk=441)
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)
    provider.load()

    strumien = provider.synthesize("Zdanie testowe.")
    pierwszy = next(strumien)

    assert not pierwszy.is_empty
    # Piper wciąż ma co generować — gdybyśmy czekali na koniec, ta liczba
    # byłaby równa całości.
    assert sum(block.size for block in fake_backend.blocks) < base.total_samples()
    strumien.close()


def test_dlugi_fragment_jest_dzielony_na_kawalki(
    tmp_path: Path, fake_backend: FakeRvcBackend
) -> None:
    settings = rvc_config(tmp_path, rvc_chunk_min_ms=100, rvc_chunk_max_ms=200)
    base = FakeBaseProvider(chunks=1, samples_per_chunk=BASE_RATE)  # całą sekundę naraz
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)
    provider.load()
    collect(provider)

    maksimum = BASE_RATE * 200 // 1000
    assert len(fake_backend.blocks) > 1
    assert all(block.size <= maksimum for block in fake_backend.blocks)


def test_opoznienie_trafia_do_logu(
    tmp_path: Path, fake_backend: FakeRvcBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """„Około sekundy" ma być liczbą w logu, a nie wrażeniem."""
    provider = RvcVoiceProvider(
        rvc_config(tmp_path), rvc_settings(tmp_path), base=FakeBaseProvider()
    )
    provider.load()
    with caplog.at_level(logging.INFO, logger="audio.tts_rvc"):
        collect(provider)

    wpisy = [record.getMessage() for record in caplog.records]
    assert any("ms" in wpis for wpis in wpisy), wpisy


def test_przekroczony_cel_opoznienia_jest_ostrzezeniem(
    tmp_path: Path, fake_backend: FakeRvcBackend, caplog: pytest.LogCaptureFixture
) -> None:
    settings = rvc_config(tmp_path, rvc_latency_target_ms=100)
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=FakeBaseProvider())
    provider.load()
    with caplog.at_level(logging.INFO, logger="audio.tts_rvc"):
        collect(provider)

    # Cel 100 ms z zerowym opóźnieniem atrapy nie zostanie przekroczony —
    # sprawdzamy więc samą ścieżkę, wymuszając cel niemożliwy do dotrzymania.
    settings_zero = rvc_config(tmp_path, rvc_latency_target_ms=100)
    object.__setattr__(settings_zero, "rvc_latency_target_ms", 0)
    caplog.clear()
    drugi = RvcVoiceProvider(settings_zero, rvc_settings(tmp_path), base=FakeBaseProvider())
    drugi.load()
    with caplog.at_level(logging.INFO, logger="audio.tts_rvc"):
        collect(drugi)

    assert any(record.levelno == logging.WARNING for record in caplog.records)


# --------------------------------------------------------------------------- #
# Pięć sposobów na awarię — i jeden wynik: głos Pipera
# --------------------------------------------------------------------------- #


def test_wylaczone_rvc_oddaje_czystego_pipera(tmp_path: Path) -> None:
    base = FakeBaseProvider()
    user = rvc_settings(tmp_path, enabled=False)
    provider = RvcVoiceProvider(rvc_config(tmp_path), user, base=base)
    provider.load()

    chunks = collect(provider)

    assert not provider.is_active()
    assert sum(chunk.samples.size for chunk in chunks) == base.total_samples()
    assert all(chunk.sample_rate == BASE_RATE for chunk in chunks)


def test_brak_pliku_modelu_nie_wywraca_mowy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = FakeBaseProvider()
    user = UserSettings(
        voice_engine="rvc_miku",
        rvc=RVCSettings(enabled=True, model_path=str(tmp_path / "nie-ma-takiego.pth")),
    )
    provider = RvcVoiceProvider(rvc_config(tmp_path), user, base=base)

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        provider.load()
        chunks = collect(provider)

    assert not provider.is_active()
    assert sum(chunk.samples.size for chunk in chunks) == base.total_samples()
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_brak_sciezki_do_modelu_konczy_sie_komunikatem(tmp_path: Path) -> None:
    user = UserSettings(voice_engine="rvc_miku", rvc=RVCSettings(enabled=True, model_path=""))
    with pytest.raises(RvcUnavailableError) as blad:
        rvc_module.create_rvc_backend(rvc_config(tmp_path), user.rvc, device="cpu")
    assert "user_settings" in blad.value.user_message


def test_brak_backendu_nie_wywraca_mowy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Żadna implementacja RVC nie jest zainstalowana — najczęstszy przypadek."""
    base = FakeBaseProvider()
    monkeypatch.setattr(rvc_module, "PROJECT_ROOT", tmp_path)
    settings = rvc_config(tmp_path, rvc_backend="", rvc_worker_python="")
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        provider.load()
        chunks = collect(provider)

    assert not provider.is_active()
    assert sum(chunk.samples.size for chunk in chunks) == base.total_samples()


def test_awaria_w_srodku_zdania_nie_gubi_reszty(
    tmp_path: Path, fake_backend: FakeRvcBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """Model pada po pierwszym bloku — dalszy ciąg zdania ma wyjść głosem Pipera."""
    fake_backend.fail_after = 1
    base = FakeBaseProvider(chunks=20, samples_per_chunk=441)
    settings = rvc_config(tmp_path, rvc_chunk_min_ms=100)
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)
    provider.load()

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        chunks = collect(provider)

    assert not provider.is_active()
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert chunks, "po awarii nie wyszedł żaden dźwięk"
    # Częstotliwość zostaje ta ustalona przez pierwszy fragment: dalsze
    # fragmenty Pipera są do niej przeliczane, zamiast przełączać strumień.
    assert len({chunk.sample_rate for chunk in chunks}) == 1


def test_po_awarii_nie_probujemy_ponownie(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    """Ponawianie kosztowałoby sekundy ciszy przed każdym kolejnym zdaniem."""
    fake_backend.fail_after = 1
    base = FakeBaseProvider(chunks=6, samples_per_chunk=441)
    provider = RvcVoiceProvider(
        rvc_config(tmp_path, rvc_chunk_min_ms=80), rvc_settings(tmp_path), base=base
    )
    provider.load()
    collect(provider, "Pierwsze zdanie.")
    po_pierwszym = len(fake_backend.blocks)
    collect(provider, "Drugie zdanie.")

    assert len(fake_backend.blocks) == po_pierwszym
    assert base.calls == ["Pierwsze zdanie.", "Drugie zdanie."]


def test_przekroczenie_limitu_czasu_konczy_sie_piperem(
    tmp_path: Path, fake_backend: FakeRvcBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """Zawieszonego backendu nie da się zabić — ma nie zabrać ze sobą mowy."""
    zwolnij = threading.Event()

    def zawies(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
        zwolnij.wait(timeout=10)
        return np.zeros(10, dtype=np.int16), RVC_RATE

    fake_backend.convert = zawies  # type: ignore[method-assign]
    base = FakeBaseProvider(chunks=10, samples_per_chunk=441)
    settings = rvc_config(tmp_path, rvc_chunk_min_ms=100, rvc_timeout_s=0.05)
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)
    provider.load()

    try:
        with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
            chunks = collect(provider)
    finally:
        zwolnij.set()

    assert not provider.is_active()
    assert chunks, "po przekroczeniu limitu czasu asystent zamilkł"
    assert any(record.levelno == logging.ERROR for record in caplog.records)


# --------------------------------------------------------------------------- #
# Wpięcie w mechanizm z Fazy 4
# --------------------------------------------------------------------------- #


def test_voice_engine_wybiera_dostawce_rvc(tmp_path: Path) -> None:
    provider = create_tts_provider(rvc_config(tmp_path), UserSettings(voice_engine="rvc_miku"))
    assert isinstance(provider, RvcVoiceProvider)
    assert provider.name == "rvc_miku"


def test_piper_pozostaje_domyslny(tmp_path: Path) -> None:
    provider = create_tts_provider(rvc_config(tmp_path), UserSettings())
    assert not isinstance(provider, RvcVoiceProvider)


def test_wylaczona_mowa_wygrywa_nad_rvc(tmp_path: Path) -> None:
    """`voice_engine` na „none" ma milczeć niezależnie od sekcji rvc."""
    provider = create_tts_provider(rvc_config(tmp_path), UserSettings(voice_engine="none"))
    assert not provider.is_speaking_enabled


def test_anulowanie_idzie_do_dostawcy_bazowego(tmp_path: Path) -> None:
    base = FakeBaseProvider()
    provider = RvcVoiceProvider(rvc_config(tmp_path), rvc_settings(tmp_path), base=base)
    provider.cancel()
    assert base.cancelled


def test_zamkniecie_zwalnia_backend(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    provider = RvcVoiceProvider(
        rvc_config(tmp_path), rvc_settings(tmp_path), base=FakeBaseProvider()
    )
    provider.load()
    provider.close()
    assert fake_backend.closed


def test_pusty_tekst_nie_uruchamia_niczego(tmp_path: Path, fake_backend: FakeRvcBackend) -> None:
    base = FakeBaseProvider()
    provider = RvcVoiceProvider(rvc_config(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    assert collect(provider, "   ") == []
    assert base.calls == []


# --------------------------------------------------------------------------- #
# Raport --check-deps
# --------------------------------------------------------------------------- #


def make_context(settings: Settings, user: UserSettings):  # type: ignore[no-untyped-def]
    return DependencyContext(
        settings=settings,
        platform_info=detect_platform(),
        gpu=gpu_info(available=False),
        ollama=OllamaStatus("http://x", False, None, (), "model", False, None, None),
        user_settings=user,
        offline=True,
    )


def rvc_checks(settings: Settings, user: UserSettings) -> dict[str, object]:
    return {
        check.name: check for check in audio.dependencies._check_rvc(make_context(settings, user))
    }


def test_raport_milczy_o_rvc_gdy_nikt_o_nie_prosil(tmp_path: Path) -> None:
    """Użytkownik samego Pipera nie ogląda pozycji o funkcji, której nie włączał."""
    assert rvc_checks(rvc_config(tmp_path), UserSettings()) == {}


def test_raport_pokazuje_brakujacy_model(tmp_path: Path) -> None:

    user = UserSettings(
        voice_engine="rvc_miku",
        rvc=RVCSettings(enabled=True, model_path=str(tmp_path / "nie-ma.pth")),
    )
    checks = rvc_checks(rvc_config(tmp_path), user)
    model = checks[t("deps.rvc.model_name")]
    assert model.ok is False  # type: ignore[attr-defined]
    assert "rvc.model_path" in model.hint  # type: ignore[attr-defined]


def test_raport_podpowiada_skrypt_instalacyjny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Podpowiedź ma prowadzić do skryptu, nie do PyPI.

    Applio nie jest pakietem, tylko repozytorium z wagami — `pip install` nie
    postawi go w żaden sposób, więc rada w tej formie byłaby ślepym zaułkiem.

    PROJECT_ROOT przestawiamy na katalog tymczasowy, żeby wynik nie zależał od
    tego, czy na maszynie testowej stoi `.venv-applio`. Wcześniejsza wersja
    tego testu pomijała się sama, gdy backend był zainstalowany — czyli
    przestawała cokolwiek sprawdzać dokładnie tam, gdzie RVC było używane
    naprawdę.
    """
    monkeypatch.setattr(rvc_module, "PROJECT_ROOT", tmp_path)
    settings = rvc_config(
        tmp_path, rvc_backend="", rvc_worker_python="", rvc_applio_path="", rvc_applio_python=""
    )
    checks = rvc_checks(settings, rvc_settings(tmp_path))
    silnik = checks[t("deps.rvc.backend_name")]
    assert silnik.ok is False  # type: ignore[attr-defined]
    assert "install-applio.sh" in silnik.hint  # type: ignore[attr-defined]


def test_raport_ostrzega_o_pracy_na_cpu(tmp_path: Path) -> None:

    checks = rvc_checks(rvc_config(tmp_path, rvc_backend="atrapa_rvc"), rvc_settings(tmp_path))
    silnik = checks[t("deps.rvc.backend_name")]
    assert silnik.ok is True  # type: ignore[attr-defined]
    assert t("deps.rvc.cpu_hint") == silnik.hint  # type: ignore[attr-defined]


def test_zawieszony_backend_nie_blokuje_wyjscia_z_programu(tmp_path: Path) -> None:
    """Wątek konwersji MUSI być daemonem — inaczej limit czasu jest fikcją.

    `ThreadPoolExecutor` byłby tu naturalnym wyborem i byłby błędem: rejestruje
    hak `atexit`, który przy wyjściu czeka na swoje wątki nawet po
    `shutdown(wait=False)`. Zawieszony backend zatrzymywałby wtedy zamykanie
    asystenta — a pod systemd kończyłoby się to SIGKILL-em.

    Test uruchamia osobny proces, bo tego zachowania nie da się sprawdzić
    wewnątrz procesu, który sam ma się zamknąć.
    """
    program = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(Path.cwd())!r})
        import numpy as np
        from audio.rvc import RvcConverter, RvcDevice
        from config import GPUInfo, Settings

        class Zawieszony:
            name = "zawieszony"
            def convert(self, samples, sample_rate, *, pitch_shift, index_rate):
                time.sleep(120)
            def close(self):
                pass

        gpu = GPUInfo(False, "none", None, None, "atrapa")
        converter = RvcConverter(
            Zawieszony(),
            settings=Settings(_env_file=None, rvc_timeout_s=0.2),
            device=RvcDevice(name="cpu", gpu=gpu),
        )
        try:
            converter.convert(np.zeros(64, dtype=np.int16), 22050, pitch_shift=0, index_rate=0.5)
        except Exception:
            pass
        print("koniec")
        """
    )
    wynik = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert "koniec" in wynik.stdout, wynik.stderr
    assert wynik.returncode == 0, wynik.stderr


# --------------------------------------------------------------------------- #
# Backend w osobnym procesie
#
# Biblioteki RVC nie działają na Pythonie 3.11+, a asystent wymaga 3.12+ —
# więc RVC liczy w drugim środowisku, w osobnym procesie, a rozmawiamy z nim
# przez potok. Testy używają ATRAPY pracownika: prawdziwy wymagałby modelu
# i drugiego venva, których w CI nie ma. Sprawdzamy protokół i — przede
# wszystkim — czy każda jego awaria kończy się głosem Pipera.
# --------------------------------------------------------------------------- #

WORKER_DZIALAJACY = """
import json, sys, wave
sys.stdout.write(json.dumps({"ready": True, "device": "cpu:0"}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    with wave.open(req["in"], "rb") as src:
        params, raw = src.getparams(), src.readframes(src.getnframes())
    with wave.open(req["out"], "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(params.framerate)
        dst.writeframes(raw)
    sys.stdout.write(json.dumps({"ok": True, "pitch": req["pitch"]}) + "\\n")
    sys.stdout.flush()
"""

WORKER_NIE_WCZYTA_MODELU = """
import json, sys
sys.stdout.write(json.dumps({"ready": False, "error": "plik nie jest modelem"}) + "\\n")
sys.stdout.flush()
"""

WORKER_MILCZY = """
import time
time.sleep(60)
"""

WORKER_ODMAWIA = """
import json, sys
sys.stdout.write(json.dumps({"ready": True, "device": "cpu:0"}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    if line.strip():
        sys.stdout.write(json.dumps({"ok": False, "error": "brak pamięci na karcie"}) + "\\n")
        sys.stdout.flush()
"""

WORKER_UMIERA = """
import json, sys
sys.stdout.write(json.dumps({"ready": True, "device": "cpu:0"}) + "\\n")
sys.stdout.flush()
raise SystemExit(3)
"""


@pytest.fixture
def fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Podstaw atrapę skryptu pracownika, zachowując prawdziwe uruchamianie procesu."""

    def ustaw(kod: str) -> Path:
        script = tmp_path / "atrapa_worker.py"
        script.write_text(kod, encoding="utf-8")
        monkeypatch.setattr(rvc_module, "rvc_worker_script", lambda: script)
        return script

    return ustaw


def worker_settings(tmp_path: Path, **overrides: object) -> Settings:
    ustawienia: dict[str, object] = {
        "rvc_backend": "subprocess",
        "rvc_worker_python": sys.executable,
        "rvc_worker_start_s": 20.0,
    }
    ustawienia.update(overrides)
    return rvc_config(tmp_path, **ustawienia)


def test_znajduje_interpreter_wskazany_w_ustawieniach(tmp_path: Path) -> None:
    settings = worker_settings(tmp_path)
    assert rvc_module.find_rvc_worker_python(settings) == Path(sys.executable)


def test_brak_interpretera_to_brak_backendu_a_nie_wyjatek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Najczęstszy stan świata: nikt nie uruchomił scripts/install-rvc.sh."""
    monkeypatch.setattr(rvc_module, "PROJECT_ROOT", tmp_path)
    settings = rvc_config(tmp_path, rvc_backend="", rvc_worker_python="")
    assert rvc_module.find_rvc_worker_python(settings) is None
    assert rvc_module.BACKEND_SUBPROCESS not in rvc_module.available_rvc_backends(settings)


def test_wskazany_interpreter_ktorego_nie_ma_nie_wywraca_niczego(tmp_path: Path) -> None:
    settings = rvc_config(tmp_path, rvc_worker_python=str(tmp_path / "nie-ma-mnie"))
    assert rvc_module.find_rvc_worker_python(settings) is None


def test_osobny_proces_konwertuje_dzwiek(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    fake_worker(WORKER_DZIALAJACY)
    base = FakeBaseProvider(chunks=10, samples_per_chunk=441)
    provider = RvcVoiceProvider(worker_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    chunks = collect(provider)

    assert provider.is_active(), "konwersja przez osobny proces nie ruszyła"
    assert sum(chunk.samples.size for chunk in chunks) > 0
    provider.close()


def test_parametry_docieraja_do_osobnego_procesu(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Pitch z config/user_settings.json ma przejść przez potok bez zgubienia."""
    fake_worker(WORKER_DZIALAJACY)
    user = rvc_settings(tmp_path, pitch_shift=-5, index_rate=0.3)
    backend = rvc_module.SubprocessRvcBackend(
        user.rvc.resolved_model_path,  # type: ignore[arg-type]
        user.rvc.resolved_index_path,
        "cpu:0",
        settings=worker_settings(tmp_path),
    )
    try:
        wynik, rate = backend.convert(make_tone(882), BASE_RATE, pitch_shift=-5, index_rate=0.3)
        assert rate == BASE_RATE
        assert wynik.size == 882
    finally:
        backend.close()


def test_proces_ktory_nie_wczyta_modelu_konczy_sie_piperem(
    tmp_path: Path, fake_worker, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    fake_worker(WORKER_NIE_WCZYTA_MODELU)
    base = FakeBaseProvider()
    provider = RvcVoiceProvider(worker_settings(tmp_path), rvc_settings(tmp_path), base=base)

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        provider.load()
        chunks = collect(provider)

    assert not provider.is_active()
    assert sum(chunk.samples.size for chunk in chunks) == base.total_samples()
    assert any("plik nie jest modelem" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_proces_ktory_nie_odpowiada_na_starcie_jest_ubijany(
    tmp_path: Path, fake_worker, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """Pracownik, który milczy, nie ma prawa zatrzymać startu asystenta."""
    fake_worker(WORKER_MILCZY)
    base = FakeBaseProvider()
    settings = worker_settings(tmp_path, rvc_worker_start_s=1.0)
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        provider.load()
        chunks = collect(provider)

    assert not provider.is_active()
    assert sum(chunk.samples.size for chunk in chunks) == base.total_samples()


def test_odmowa_konwersji_konczy_sie_piperem(
    tmp_path: Path, fake_worker, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """Typowy przypadek: model się wczytał, ale zabrakło pamięci na karcie."""
    fake_worker(WORKER_ODMAWIA)
    base = FakeBaseProvider(chunks=10, samples_per_chunk=441)
    provider = RvcVoiceProvider(worker_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        chunks = collect(provider)

    assert not provider.is_active()
    assert chunks, "po odmowie konwersji asystent zamilkł"
    assert len({chunk.sample_rate for chunk in chunks}) == 1
    provider.close()


def test_smierc_procesu_w_trakcie_konczy_sie_piperem(
    tmp_path: Path, fake_worker, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    fake_worker(WORKER_UMIERA)
    base = FakeBaseProvider(chunks=10, samples_per_chunk=441)
    provider = RvcVoiceProvider(worker_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        chunks = collect(provider)

    assert not provider.is_active()
    assert chunks, "po śmierci procesu asystent zamilkł"
    provider.close()


def test_zamkniecie_konczy_proces(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    fake_worker(WORKER_DZIALAJACY)
    user = rvc_settings(tmp_path)
    backend = rvc_module.SubprocessRvcBackend(
        user.rvc.resolved_model_path,  # type: ignore[arg-type]
        user.rvc.resolved_index_path,
        "cpu:0",
        settings=worker_settings(tmp_path),
    )
    backend.close()
    assert backend._process.poll() is not None, "proces pracownika został po zamknięciu"


# --------------------------------------------------------------------------- #
# Sam skrypt pracownika: czytanie checkpointu HuBERT-a
# --------------------------------------------------------------------------- #
#
# Testy wyżej podstawiają atrapę skryptu, więc prawdziwy `scripts/rvc_worker.py`
# nie jest przez nie w ogóle wykonywany. Poniższe biorą go na warsztat wprost,
# bo siedzi w nim obejście, którego zepsucie NIE objawia się czerwonym testem
# ani wyjątkiem: `rvc_python` łyka wtedy `UnpicklingError` jako ostrzeżenie
# i oddaje krotkę zamiast dźwięku, a asystent po cichu wraca do Pipera.
# Dokładnie tego błędu szukało się długo, więc ma tu zostać przygwożdżony.

WORKER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "rvc_worker.py"


def zaladuj_worker(
    monkeypatch: pytest.MonkeyPatch, torch_stub: types.ModuleType
) -> types.ModuleType:
    """Wczytaj skrypt pracownika ze ścieżki, z atrapą ``torch`` w ``sys.modules``.

    Skrypt nie leży w pakiecie i nie ma prawa importować niczego z projektu,
    więc zwykłe ``import`` go nie sięgnie. Atrapa ``torch`` jest tu podwójnie
    na miejscu: prawdziwy import trwa sekundy, a łatka zakłada trwałą podmianę
    ``torch.load`` — na prawdziwym module wyciekłaby do pozostałych testów.
    """
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    spec = importlib.util.spec_from_file_location("mikuva_rvc_worker_pod_testem", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atrapa_torcha(load: object) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.load = load  # type: ignore[attr-defined]
    return module


def nowoczesny_torch() -> tuple[types.ModuleType, list[dict[str, object]]]:
    """Atrapa PyTorcha 2.6+: zapamiętuje argumenty, którymi ją zawołano."""
    wywolania: list[dict[str, object]] = []

    def load(path: object, **kwargs: object) -> str:
        wywolania.append(dict(kwargs))
        return f"checkpoint z {path}"

    return atrapa_torcha(load), wywolania


def test_checkpoint_hubert_czytamy_z_wylaczonym_weights_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_stub, wywolania = nowoczesny_torch()
    worker = zaladuj_worker(monkeypatch, torch_stub)

    worker.zaufaj_lokalnym_checkpointom()
    torch_stub.load("hubert_base.pt")

    assert wywolania == [{"weights_only": False}], (
        "bez tego fairseq nie odtworzy Dictionary z checkpointu, a konwersja "
        "wywróci się dopiero linijkę dalej na 'tuple' object has no attribute 'dtype'"
    )


def test_jawne_weights_only_wygrywa_z_domyslnym(monkeypatch: pytest.MonkeyPatch) -> None:
    """Podmieniamy wartość domyślną, nie decyzję wołającego."""
    torch_stub, wywolania = nowoczesny_torch()
    worker = zaladuj_worker(monkeypatch, torch_stub)

    worker.zaufaj_lokalnym_checkpointom()
    torch_stub.load("cudzy.pt", weights_only=True)

    assert wywolania == [{"weights_only": True}]


def test_dwa_wywolania_nie_owijaja_torch_load_dwa_razy(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_stub, _ = nowoczesny_torch()
    worker = zaladuj_worker(monkeypatch, torch_stub)

    worker.zaufaj_lokalnym_checkpointom()
    po_pierwszym = torch_stub.load
    worker.zaufaj_lokalnym_checkpointom()

    assert torch_stub.load is po_pierwszym, "łatka nałożyła się drugi raz na samą siebie"


def test_stary_torch_bez_weights_only_dalej_dziala(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyTorch starszy niż 1.13 nie zna tego argumentu — i nie musi."""
    wywolania: list[dict[str, object]] = []

    def load(path: object, **kwargs: object) -> str:
        if "weights_only" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'weights_only'")
        wywolania.append(dict(kwargs))
        return f"checkpoint z {path}"

    torch_stub = atrapa_torcha(load)
    worker = zaladuj_worker(monkeypatch, torch_stub)

    worker.zaufaj_lokalnym_checkpointom()

    assert torch_stub.load("hubert_base.pt") == "checkpoint z hubert_base.pt"
    assert wywolania == [{}], "argument nie został odrzucony przy ponownej próbie"


def test_import_pracownika_nie_rusza_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Podmiana ``sys.stdout`` należy do :func:`main`, nie do importu.

    Gdyby przejęcie strumienia było efektem ubocznym importu, ten test — i cały
    pytest razem z nim — pisałby od tego miejsca na ``stderr``.
    """
    przed = sys.stdout
    torch_stub, _ = nowoczesny_torch()
    zaladuj_worker(monkeypatch, torch_stub)

    assert sys.stdout is przed


# --------------------------------------------------------------------------- #
# Applio: szybszy backend w jeszcze jednym środowisku
# --------------------------------------------------------------------------- #
#
# Applio jedzie tym samym protokołem i tym samym skryptem pracownika, więc
# całą mechanikę potoku sprawdzają testy wyżej. Tutaj pilnujemy trzech rzeczy,
# które są dla Applio SPECYFICZNE i których pomyłka nie objawia się od razu:
#
# 1. katalog roboczy procesu — Applio robi przy imporcie `now_dir = os.getcwd()`
#    i po tym katalogu szuka wag; uruchomione skądinąd zaimportuje się bez
#    słowa skargi i wywróci dopiero na pierwszej konwersji,
# 2. argumenty silnika — bez `--engine applio` pracownik wystartuje stary
#    `rvc_python`, czyli dokładnie to, od czego uciekamy,
# 3. brak Applio ma kończyć się głosem Pipera, a nie wyjątkiem.


def zrob_applio(root: Path) -> Path:
    """Atrapa katalogu Applio: liczy się obecność podkatalogu ``rvc``."""
    korzen = root / "Applio"
    (korzen / "rvc").mkdir(parents=True)
    return korzen


def applio_settings(tmp_path: Path, **overrides: object) -> Settings:
    korzen = overrides.pop("korzen", None) or zrob_applio(tmp_path)
    ustawienia: dict[str, object] = {
        "rvc_backend": "applio",
        "rvc_applio_path": str(korzen),
        "rvc_applio_python": sys.executable,
        "rvc_worker_start_s": 20.0,
    }
    ustawienia.update(overrides)
    return rvc_config(tmp_path, **ustawienia)


def kod_pracownika_applio(raport: Path) -> str:
    """Atrapa pracownika, która donosi, JAK została uruchomiona."""
    return textwrap.dedent(
        f"""
        import json, os, sys
        from pathlib import Path

        Path(r"{raport}").write_text(
            json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}), encoding="utf-8"
        )
        sys.stdout.write(json.dumps({{"ready": True, "device": "cuda:0"}}) + "\\n")
        sys.stdout.flush()
        for line in sys.stdin:
            if not line.strip():
                continue
            if json.loads(line).get("cmd") == "quit":
                break
            sys.stdout.write(json.dumps({{"ok": True}}) + "\\n")
            sys.stdout.flush()
        """
    )


# --- odnajdywanie ---------------------------------------------------------- #


def test_wskazany_katalog_applio_jest_uzywany(tmp_path: Path) -> None:
    korzen = zrob_applio(tmp_path)
    settings = applio_settings(tmp_path, korzen=korzen)
    assert rvc_module.find_applio_root(settings) == korzen


def test_katalog_bez_podkatalogu_rvc_to_nie_applio(tmp_path: Path) -> None:
    """Przerwany w połowie klon istnieje na dysku, ale nic z niego nie zaimportujemy."""
    pusty = tmp_path / "niby-applio"
    pusty.mkdir()
    settings = rvc_config(tmp_path, rvc_applio_path=str(pusty))
    assert rvc_module.find_applio_root(settings) is None


def test_brak_katalogu_applio_to_brak_backendu_a_nie_wyjatek(tmp_path: Path) -> None:
    settings = rvc_config(tmp_path, rvc_applio_path=str(tmp_path / "nie-ma-mnie"))
    assert rvc_module.find_applio_root(settings) is None


def test_wskazany_interpreter_applio_jest_uzywany(tmp_path: Path) -> None:
    settings = rvc_config(tmp_path, rvc_applio_python=sys.executable)
    assert rvc_module.find_applio_python(settings) == Path(sys.executable)


def test_interpreter_applio_ktorego_nie_ma_nie_wywraca_niczego(tmp_path: Path) -> None:
    settings = rvc_config(tmp_path, rvc_applio_python=str(tmp_path / "nie-ma-mnie"))
    assert rvc_module.find_applio_python(settings) is None


def test_applio_i_rvc_python_maja_osobne_interpretery(tmp_path: Path) -> None:
    """Jedno środowisko ich nie pomieści: fairseq chce 3.10, Applio wymaga 3.12+."""
    settings = rvc_config(
        tmp_path,
        rvc_worker_python=sys.executable,
        rvc_applio_python=str(tmp_path / "nie-ma-mnie"),
    )
    assert rvc_module.find_rvc_worker_python(settings) == Path(sys.executable)
    assert rvc_module.find_applio_python(settings) is None


def test_applio_pojawia_sie_na_liscie_dostepnych(tmp_path: Path) -> None:
    settings = applio_settings(tmp_path)
    assert rvc_module.BACKEND_APPLIO in rvc_module.available_rvc_backends(settings)


def test_bez_kodu_applio_nie_ma_go_na_liscie(tmp_path: Path) -> None:
    settings = rvc_config(
        tmp_path, rvc_applio_python=sys.executable, rvc_applio_path=str(tmp_path / "pusto")
    )
    assert rvc_module.BACKEND_APPLIO not in rvc_module.available_rvc_backends(settings)


# --- uruchamianie ---------------------------------------------------------- #


def test_applio_startuje_z_wlasnego_katalogu(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Najważniejszy test w tej sekcji — patrz punkt 1 w komentarzu wyżej."""
    raport = tmp_path / "raport.json"
    fake_worker(kod_pracownika_applio(raport))
    korzen = zrob_applio(tmp_path)
    user = rvc_settings(tmp_path)
    backend = rvc_module.ApplioBackend(
        user.rvc.resolved_model_path,  # type: ignore[arg-type]
        user.rvc.resolved_index_path,
        "cuda:0",
        settings=applio_settings(tmp_path, korzen=korzen),
    )
    backend.close()

    dane = json.loads(raport.read_text(encoding="utf-8"))
    assert Path(dane["cwd"]).resolve() == korzen.resolve(), (
        "pracownik wystartował spoza katalogu Applio — wagi znajdzie dopiero przypadkiem"
    )


def test_applio_dostaje_swoj_silnik_i_parametry(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    raport = tmp_path / "raport.json"
    fake_worker(kod_pracownika_applio(raport))
    korzen = zrob_applio(tmp_path)
    user = rvc_settings(tmp_path)
    backend = rvc_module.ApplioBackend(
        user.rvc.resolved_model_path,  # type: ignore[arg-type]
        user.rvc.resolved_index_path,
        "cuda:0",
        settings=applio_settings(tmp_path, korzen=korzen, rvc_f0_method="fcpe"),
    )
    backend.close()

    argv = json.loads(raport.read_text(encoding="utf-8"))["argv"]
    assert "--engine" in argv and argv[argv.index("--engine") + 1] == "applio", (
        "bez tego pracownik uruchomiłby stary rvc_python"
    )
    assert argv[argv.index("--applio-root") + 1] == str(korzen)
    assert argv[argv.index("--f0-method") + 1] == "fcpe"
    assert argv[argv.index("--embedder") + 1] == "contentvec"


def test_backend_applio_ma_wlasna_nazwe(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Nazwa trafia do logu i do `--check-deps`; „subprocess" wprowadzałoby w błąd."""
    fake_worker(kod_pracownika_applio(tmp_path / "raport.json"))
    user = rvc_settings(tmp_path)
    backend = rvc_module.ApplioBackend(
        user.rvc.resolved_model_path,  # type: ignore[arg-type]
        user.rvc.resolved_index_path,
        "cuda:0",
        settings=applio_settings(tmp_path),
    )
    assert backend.name == "applio"
    backend.close()


def test_wybor_applio_w_ustawieniach_daje_backend_applio(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    fake_worker(kod_pracownika_applio(tmp_path / "raport.json"))
    user = rvc_settings(tmp_path)
    backend = rvc_module.create_rvc_backend(
        applio_settings(tmp_path), user.rvc, device="cuda:0"
    )
    assert backend.name == "applio"
    backend.close()


def test_automat_woli_applio_od_rvc_python(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Puste RVC_BACKEND i oba zainstalowane — wygrywa szybsze."""
    fake_worker(kod_pracownika_applio(tmp_path / "raport.json"))
    user = rvc_settings(tmp_path)
    settings = applio_settings(tmp_path, rvc_backend="", rvc_worker_python=sys.executable)
    backend = rvc_module.create_rvc_backend(settings, user.rvc, device="cuda:0")
    assert backend.name == "applio", "automat zszedł do wolniejszego backendu"
    backend.close()


# --- awarie ---------------------------------------------------------------- #


def test_brak_applio_konczy_sie_piperem(tmp_path: Path) -> None:
    """Wybrane w ustawieniach, ale niezainstalowane — ma być cichy powrót do Pipera."""
    base = FakeBaseProvider(chunks=6, samples_per_chunk=441)
    settings = rvc_config(
        tmp_path,
        rvc_backend="applio",
        rvc_applio_path=str(tmp_path / "nie-ma-mnie"),
        rvc_applio_python=str(tmp_path / "tez-nie-ma"),
    )
    provider = RvcVoiceProvider(settings, rvc_settings(tmp_path), base=base)
    provider.load()

    chunks = collect(provider)

    assert not provider.is_active()
    assert chunks, "brak Applio uciszył asystenta"
    assert len({chunk.sample_rate for chunk in chunks}) == 1
    provider.close()


def test_sam_interpreter_bez_kodu_applio_to_za_malo(tmp_path: Path) -> None:
    settings = rvc_config(
        tmp_path,
        rvc_backend="applio",
        rvc_applio_python=sys.executable,
        rvc_applio_path=str(tmp_path / "pusto"),
    )
    user = rvc_settings(tmp_path)
    with pytest.raises(RvcUnavailableError):
        rvc_module.create_rvc_backend(settings, user.rvc, device="cpu:0")


# --- pracownik: wybór silnika ---------------------------------------------- #


def test_pracownik_odmawia_katalogu_ktory_nie_jest_applio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Zła ścieżka ma paść w linii powitalnej, nie przy pierwszym zdaniu.

    Przechwytujemy `capfd`, a nie `capsys`: protokół idzie do `sys.__stdout__`,
    czyli prosto do deskryptora, i podmiana `sys.stdout` go nie widzi.
    """
    torch_stub, _ = nowoczesny_torch()
    worker = zaladuj_worker(monkeypatch, torch_stub)
    monkeypatch.setattr(worker, "przejmij_stdout", lambda: None)

    kod = worker.main(
        [
            "--engine",
            "applio",
            "--applio-root",
            str(tmp_path / "pusto"),
            "--model",
            str(tmp_path / "miku.pth"),
        ]
    )

    assert kod == 1
    odpowiedz = json.loads(capfd.readouterr().out.strip().splitlines()[-1])
    assert odpowiedz["ready"] is False
    assert "Applio" in odpowiedz["error"]


@pytest.mark.parametrize(
    ("device", "oczekiwane"),
    [("cpu:0", ""), ("cuda:0", "0"), ("cuda:1", "1")],
)
def test_wybor_karty_idzie_przez_zmienna_srodowiskowa(
    monkeypatch: pytest.MonkeyPatch, device: str, oczekiwane: str
) -> None:
    """Applio nie przyjmuje urządzenia w argumencie — czyta je z torcha."""
    torch_stub, _ = nowoczesny_torch()
    worker = zaladuj_worker(monkeypatch, torch_stub)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    worker.ustaw_widoczna_karte(device)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == oczekiwane


# --------------------------------------------------------------------------- #
# Łańcuch backendów: po awarii Applio schodzimy do rvc-python
# --------------------------------------------------------------------------- #
#
# Awaria Applio kończyła się dotąd zwykłym Piperem do końca sesji, mimo że na
# maszynie stał sprawny drugi backend. Teraz jest kolejka — ale z dwoma
# ograniczeniami, które są równie ważne jak sama przesiadka:
#
# * przesiadka NIE dzieje się w połowie zdania (kosztuje sekundy na wczytanie
#   modelu, czyli dokładnie tę ciszę, której ta faza ma nie dopuścić),
# * każde ogniwo próbujemy RAZ; wyczerpana kolejka to Piper do końca sesji.
#
# Atrapa pracownika rozróżnia backendy po `--engine`: to ten sam skrypt dla
# obu, więc inaczej nie da się kazać jednemu paść, a drugiemu działać.

WORKER_WYBIORCZY = '''
import json, sys, wave

engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "rvc_python"
psuj_start = "PSUJ_START" == "{tryb}"

if engine == "applio" and psuj_start:
    sys.stdout.write(json.dumps({{"ready": False, "error": "brak pamieci na karcie"}}) + "\\n")
    sys.stdout.flush()
    raise SystemExit(1)

sys.stdout.write(json.dumps({{"ready": True, "device": "cpu:0", "engine": engine}}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    if engine == "applio":
        sys.stdout.write(json.dumps({{"ok": False, "error": "konwersja odmowila"}}) + "\\n")
        sys.stdout.flush()
        continue
    with wave.open(req["in"], "rb") as src:
        params, raw = src.getparams(), src.readframes(src.getnframes())
    with wave.open(req["out"], "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(params.framerate)
        dst.writeframes(raw)
    sys.stdout.write(json.dumps({{"ok": True}}) + "\\n")
    sys.stdout.flush()
'''


def worker_wybiorczy(tryb: str = "") -> str:
    """Atrapa, w której Applio zawodzi, a `rvc-python` działa."""
    return WORKER_WYBIORCZY.format(tryb=tryb)


def lancuch_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Obie implementacje dostępne, wybór zostawiony automatowi."""
    ustawienia: dict[str, object] = {"rvc_backend": "", "rvc_worker_python": sys.executable}
    ustawienia.update(overrides)
    return applio_settings(tmp_path, **ustawienia)


def test_kolejka_ma_applio_przed_rvc_python(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    fake_worker(worker_wybiorczy())
    assert rvc_module.rvc_backend_chain(lancuch_settings(tmp_path)) == ["applio", "subprocess"]


def test_wskazany_backend_nie_jest_podmieniany(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Kto wpisał `applio`, prosił o Applio — nie o cokolwiek, co ruszy."""
    fake_worker(worker_wybiorczy())
    settings = lancuch_settings(tmp_path, rvc_backend="applio")
    assert rvc_module.rvc_backend_chain(settings) == ["applio"]


def test_applio_ktore_nie_wstaje_oddaje_pole_rvc_python(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Awaria przy starcie — drugi backend wchodzi od razu, przed pierwszym zdaniem."""
    fake_worker(worker_wybiorczy("PSUJ_START"))
    base = FakeBaseProvider(chunks=6, samples_per_chunk=441)
    provider = RvcVoiceProvider(lancuch_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    assert provider.is_active(), "asystent zszedł do Pipera, choć drugi backend był sprawny"
    chunks = collect(provider)
    assert chunks
    provider.close()


def test_po_awarii_applio_nastepne_zdanie_idzie_przez_rvc_python(
    tmp_path: Path, fake_worker, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """Sedno tej zmiany — i granica, na której przesiadka wolno się odbyć."""
    fake_worker(worker_wybiorczy())
    base = FakeBaseProvider(chunks=8, samples_per_chunk=441)
    provider = RvcVoiceProvider(lancuch_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()
    assert provider.is_active()

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        pierwsze = collect(provider)

    assert pierwsze, "awaria w środku wypowiedzi uciszyła asystenta"
    assert not provider.is_active(), "przesiadka nastąpiła w połowie zdania"

    drugie = collect(provider)

    assert drugie, "po przesiadce asystent zamilkł"
    assert provider.is_active(), "drugi backend nie wszedł przy następnej wypowiedzi"
    assert provider._converter is not None
    assert provider._converter.backend_name == "subprocess"
    provider.close()


def test_wyczerpana_kolejka_to_piper_do_konca_sesji(tmp_path: Path, fake_worker) -> None:  # type: ignore[no-untyped-def]
    """Stara gwarancja zostaje: nie ponawiamy w nieskończoność.

    Gdy oba backendy odmówią, kolejne wypowiedzi nie mają prawa próbować
    niczego budować — sekundy ciszy przed każdym zdaniem byłyby gorsze od
    zwykłego głosu Pipera.
    """
    fake_worker(WORKER_ODMAWIA)
    base = FakeBaseProvider(chunks=6, samples_per_chunk=441)
    provider = RvcVoiceProvider(lancuch_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    for _ in range(3):
        assert collect(provider), "asystent zamilkł po wyczerpaniu kolejki"

    assert not provider.is_active()
    assert provider._chain == [], "kolejka nie została wyczerpana"
    provider.close()


def test_przesiadka_zostawia_slad_w_logu(
    tmp_path: Path, fake_worker, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    """Cicha zmiana silnika jest gorsza od głośnej — barwa głosu się zmienia."""
    fake_worker(worker_wybiorczy())
    base = FakeBaseProvider(chunks=6, samples_per_chunk=441)
    provider = RvcVoiceProvider(lancuch_settings(tmp_path), rvc_settings(tmp_path), base=base)
    provider.load()

    with caplog.at_level(logging.ERROR, logger="audio.tts_rvc"):
        collect(provider)

    assert any("subprocess" in zapis.getMessage() for zapis in caplog.records), (
        "log nie mówi, na który backend przechodzimy"
    )
    provider.close()
