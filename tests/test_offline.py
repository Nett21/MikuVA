"""Testy pracy bez internetu.

Sprawdzają trzy rzeczy, od których zależy, czy asystent wystartuje na maszynie
odciętej od sieci:

* rozpoznawanie modelu Whisper leżącego już na dysku (i odrzucanie migawek
  niekompletnych — przerwane pobieranie zostawia strukturę katalogów),
* rozstrzyganie trybu ``OFFLINE_MODE`` (``auto``/``on``/``off``),
* ładowanie modelu ZE ŚCIEŻKI, żeby ``faster-whisper`` nie odpytywał
  HuggingFace o aktualność migawki.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from conftest import FakeFasterWhisper, FakeWhisperModel

import config
from audio.whisper import TranscriptionError, WhisperTranscriber
from config import (
    GPUInfo,
    Settings,
    apply_offline_environment,
    describe_offline_mode,
    find_local_whisper_model,
    is_offline,
    iter_local_whisper_models,
    pip_install_hint,
    wheelhouse_packages,
)

CUDA_ABSENT = GPUInfo(
    cuda_available=False,
    source="none",
    device_name=None,
    driver_version=None,
    detail="brak GPU",
)


def empty_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Podmień ``os.environ`` na pusty słownik — test nie może brudzić procesu."""
    environment: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", environment)
    return environment


def make_hf_model(
    cache: Path,
    repo: str = "Systran/faster-whisper-small",
    *,
    revision: str = "abc123",
    complete: bool = True,
) -> Path:
    """Odtwórz układ cache'u HuggingFace tak, jak robi to huggingface_hub."""
    model_dir = cache / ("models--" + repo.replace("/", "--"))
    snapshot = model_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    if complete:
        (snapshot / "model.bin").write_bytes(b"ctranslate2")
    return snapshot


# --------------------------------------------------------------------------- #
# Wyszukiwanie modelu na dysku
# --------------------------------------------------------------------------- #


def test_model_z_cache_huggingface_jest_znajdowany_po_krotkiej_nazwie(
    isolated_model_cache: Path,
) -> None:
    snapshot = make_hf_model(isolated_model_cache)

    assert find_local_whisper_model("small") == snapshot
    assert find_local_whisper_model("Systran/faster-whisper-small") == snapshot
    assert sorted(iter_local_whisper_models()) == ["small"]


def test_inny_rozmiar_modelu_nie_jest_podstawiany(isolated_model_cache: Path) -> None:
    """Pobrane 'tiny' nie może uchodzić za 'small' — to inna jakość rozpoznawania."""
    make_hf_model(isolated_model_cache, "Systran/faster-whisper-tiny")

    assert find_local_whisper_model("tiny") is not None
    assert find_local_whisper_model("small") is None


def test_przerwane_pobieranie_nie_liczy_sie_jako_model(isolated_model_cache: Path) -> None:
    make_hf_model(isolated_model_cache, complete=False)

    assert find_local_whisper_model("small") is None
    assert iter_local_whisper_models() == {}


def test_wlasny_katalog_z_modelem_ma_pierwszenstwo(tmp_path: Path) -> None:
    local = tmp_path / "moj-model"
    local.mkdir()
    (local / "model.bin").write_bytes(b"ctranslate2")

    assert find_local_whisper_model(str(local)) == local


def test_wariant_distil_jest_rozpoznawany(isolated_model_cache: Path) -> None:
    make_hf_model(isolated_model_cache, "Systran/faster-distil-whisper-large-v3")

    assert find_local_whisper_model("distil-large-v3") is not None


# --------------------------------------------------------------------------- #
# Rozstrzyganie trybu
# --------------------------------------------------------------------------- #


def test_zapis_logiczny_w_env_jest_tlumaczony_na_on_off() -> None:
    assert Settings(_env_file=None, offline_mode="true").offline_mode == "on"
    assert Settings(_env_file=None, offline_mode="0").offline_mode == "off"
    assert Settings(_env_file=None, offline_mode="TAK").offline_mode == "on"
    assert Settings(_env_file=None).offline_mode == "auto"


def test_tryb_auto_wlacza_offline_dopiero_gdy_model_jest_na_dysku(
    isolated_model_cache: Path, settings: Settings
) -> None:
    auto = settings.model_copy(update={"offline_mode": "auto", "whisper_model": "small"})
    assert is_offline(auto) is False

    make_hf_model(isolated_model_cache)
    assert is_offline(auto) is True


def test_tryb_wymuszony_nie_zaleza_od_zawartosci_dysku(
    isolated_model_cache: Path, settings: Settings
) -> None:
    forced_on = settings.model_copy(update={"offline_mode": "on"})
    forced_off = settings.model_copy(update={"offline_mode": "off"})

    assert is_offline(forced_on) is True
    assert is_offline(forced_off) is False

    make_hf_model(isolated_model_cache)
    assert is_offline(forced_on) is True
    assert is_offline(forced_off) is False


def test_opis_trybu_mowi_czego_brakuje(settings: Settings) -> None:
    auto = settings.model_copy(update={"offline_mode": "auto", "whisper_model": "small"})
    description = describe_offline_mode(auto)

    assert "online" in description
    assert "small" in description


# --------------------------------------------------------------------------- #
# Zmienne środowiskowe
# --------------------------------------------------------------------------- #


def test_tryb_offline_blokuje_siec_bibliotekom_huggingface(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    environment = empty_environment(monkeypatch)

    assert apply_offline_environment(settings.model_copy(update={"offline_mode": "on"})) is True

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    # Cache modeli zawsze w katalogu projektu, nie w ~/.cache.
    assert environment["HF_HUB_CACHE"] == str(config.WHISPER_CACHE_DIR)
    # Ollama jest lokalna — proxy nigdy nie może jej dotyczyć.
    assert "127.0.0.1" in environment["no_proxy"]


def test_tryb_online_nie_ustawia_blokady_sieci(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    environment = empty_environment(monkeypatch)

    assert apply_offline_environment(settings.model_copy(update={"offline_mode": "off"})) is False

    assert "HF_HUB_OFFLINE" not in environment
    # Katalog cache i wyłączenie proxy dla adresów lokalnych obowiązują zawsze.
    assert environment["HF_HUB_CACHE"] == str(config.WHISPER_CACHE_DIR)


# --------------------------------------------------------------------------- #
# Ładowanie modelu bez sieci
# --------------------------------------------------------------------------- #


def test_model_z_dysku_laduje_sie_ze_sciezki_a_nie_po_nazwie(
    isolated_model_cache: Path, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Nazwa = repozytorium HuggingFace, ścieżka = pewność, że nikt nie dzwoni do sieci."""
    snapshot = make_hf_model(isolated_model_cache)
    offline = settings.model_copy(update={"whisper_model": "small", "offline_mode": "auto"})

    transcriber = WhisperTranscriber(offline, gpu=CUDA_ABSENT)
    transcriber.load()

    model = FakeWhisperModel.instances[-1]
    assert model.model_size_or_path == str(snapshot)
    assert model.local_files_only is True


def test_bez_modelu_i_z_wymuszonym_offline_blad_odsyla_do_skryptu(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    module = FakeFasterWhisper(fail_on_devices=("cpu",))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    offline = settings.model_copy(update={"offline_mode": "on"})
    transcriber = WhisperTranscriber(offline, gpu=CUDA_ABSENT)

    with pytest.raises(TranscriptionError) as error:
        transcriber.load()
    assert "prepare_offline.py" in error.value.hint


# --------------------------------------------------------------------------- #
# Instalacja zależności bez internetu
# --------------------------------------------------------------------------- #


def test_podpowiedz_instalacji_uzywa_magazynu_kol(tmp_path: Path) -> None:
    wheelhouse = config.WHEELHOUSE_DIR
    wheelhouse.mkdir(parents=True)
    (wheelhouse / "numpy-2.0.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"")

    assert wheelhouse_packages() == 1
    hint = pip_install_hint(offline=True)
    assert "--no-index" in hint
    assert str(wheelhouse) in hint


def test_bez_magazynu_kol_podpowiedz_kieruje_do_przygotowania() -> None:
    assert wheelhouse_packages() == 0
    assert "prepare_offline.py" in pip_install_hint(offline=True)
    assert pip_install_hint(offline=False) == "python -m pip install -r requirements.txt"
