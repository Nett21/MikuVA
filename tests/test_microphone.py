"""Testy mikrofonu — wyłącznie na atrapie PortAudio, bez sprzętu."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from conftest import FakeSoundDevice, make_tone

from audio.microphone import (
    Microphone,
    MicrophoneError,
    MicrophoneUnavailableError,
    find_input_device,
    is_microphone_available,
    list_input_devices,
)
from config import Settings


def test_lista_urzadzen_pomija_wyjscia(fake_sounddevice: FakeSoundDevice, settings: Settings) -> None:
    devices = list_input_devices(settings)
    assert [device.name for device in devices] == ["Atrapa mikrofonu"]
    assert devices[0].host_api == "AtrapaAPI"


def test_brak_pakietu_sounddevice_nie_wywala_programu(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert is_microphone_available(settings) is False


def test_blad_portaudio_daje_czytelny_wyjatek(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    module = FakeSoundDevice(raise_on_query=RuntimeError("serwer dźwięku nie działa"))
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    with pytest.raises(MicrophoneUnavailableError) as error:
        list_input_devices(settings)
    assert "serwer dźwięku nie działa" in error.value.message
    assert is_microphone_available(settings) is False


def test_mic_enabled_false_wylacza_mikrofon(fake_sounddevice: FakeSoundDevice) -> None:
    disabled = Settings(_env_file=None, mic_enabled=False)
    assert is_microphone_available(disabled) is False


def test_wyszukiwanie_po_fragmencie_nazwy(fake_sounddevice: FakeSoundDevice, settings: Settings) -> None:
    assert find_input_device("atrapa mikro", settings) is not None
    assert find_input_device("nieistniejace", settings) is None


def test_nieznane_urzadzenie_z_konfiguracji_konczy_sie_wyjatkiem(
    fake_sounddevice: FakeSoundDevice, settings: Settings
) -> None:
    configured = settings.model_copy(update={"audio_input_device": "Nie ma takiego"})
    microphone = Microphone(configured)

    with pytest.raises(MicrophoneUnavailableError) as error:
        microphone.start()
    assert "Atrapa mikrofonu" in error.value.hint  # podpowiada, co jest dostępne


def test_indeks_urzadzenia_w_konfiguracji_jest_odrzucany() -> None:
    with pytest.raises(ValueError, match="fragment of the device NAME"):
        Settings(_env_file=None, audio_input_device="3")


def test_ramki_maja_stala_dlugosc_niezaleznie_od_bloku(
    fake_sounddevice: FakeSoundDevice, settings: Settings
) -> None:
    microphone = Microphone(settings)
    microphone.start()
    stream = fake_sounddevice.streams[-1]

    # Blok o długości 1,5 ramki — reszta musi poczekać na kolejny blok.
    microphone._emit(make_tone(480))
    first = microphone.read(timeout=0.1)
    assert first is not None
    assert first.samples.size == settings.frame_samples == 320
    assert microphone.read(timeout=0.05) is None

    microphone._emit(make_tone(160))
    second = microphone.read(timeout=0.1)
    assert second is not None
    assert second.samples.size == 320

    microphone.stop()
    assert stream.closed is True


def test_przepelnienie_kolejki_odrzuca_najstarsze_ramki(
    fake_sounddevice: FakeSoundDevice, settings: Settings
) -> None:
    tiny = settings.model_copy(update={"audio_queue_seconds": 1.0})  # 50 ramek po 20 ms
    microphone = Microphone(tiny)
    microphone.start()

    for _ in range(60):
        microphone._emit(make_tone(320))

    assert microphone.dropped_frames > 0
    frame = microphone.read(timeout=0.1)
    assert frame is not None
    microphone.stop()


def test_fallback_na_natywna_czestotliwosc_i_przeprobkowanie(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    # Urządzenie odmawia 16 kHz (typowe dla WASAPI w trybie współdzielonym).
    module = FakeSoundDevice(fail_rates=(16_000,))
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    microphone = Microphone(settings)
    microphone.start()

    stream = module.streams[-1]
    assert stream.samplerate == 48_000

    # 20 ms przy 48 kHz to 960 próbek; po przepróbkowaniu ma wyjść 320.
    stream.feed(make_tone(960, frequency=200.0))
    frame = microphone.read(timeout=0.2)
    assert frame is not None
    assert frame.samples.size == 320
    assert frame.sample_rate == 16_000
    microphone.stop()


def test_stereo_jest_sprowadzane_do_mono(fake_sounddevice: FakeSoundDevice, settings: Settings) -> None:
    microphone = Microphone(settings)
    microphone.start()
    stream = fake_sounddevice.streams[-1]

    left = make_tone(320, amplitude=8000)
    right = make_tone(320, amplitude=4000)
    stream.feed(np.stack([left, right], axis=1))

    frame = microphone.read(timeout=0.2)
    assert frame is not None
    assert frame.samples.ndim == 1
    assert frame.samples.size == 320
    microphone.stop()


def test_czytanie_przed_startem_daje_czytelny_blad(
    fake_sounddevice: FakeSoundDevice, settings: Settings
) -> None:
    microphone = Microphone(settings)
    # Treść komunikatu idzie przez i18n i zależy od UI_LANGUAGE, więc test
    # sprawdza RODZAJ błędu i to, że niesie podpowiedź — a nie brzmienie zdania.
    with pytest.raises(MicrophoneError) as blad:
        microphone.read(timeout=0.01)
    assert blad.value.message
    assert "start()" in blad.value.hint
