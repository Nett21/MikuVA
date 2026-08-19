"""Testy odtwarzania dźwięku (Faza 4).

Karta dźwiękowa jest atrapą: ``FakeOutputStream`` wywołuje ten sam callback,
który wywoływałoby PortAudio, więc testy sprawdzają realną ścieżkę danych
(kolejka → callback → próbki) bez żadnego sprzętu i bez emitowania dźwięku.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeSoundDevice, make_tone

from audio.output import (
    AudioOutput,
    AudioOutputUnavailableError,
    find_output_device,
    is_speaker_available,
    list_output_devices,
    play_chunks,
)
from audio.tts import SpeechChunk
from config import Settings, UserSettings


def chunk(samples: int = 1_000, *, rate: int = 22_050) -> SpeechChunk:
    return SpeechChunk(samples=make_tone(samples), sample_rate=rate)


# --------------------------------------------------------------------------- #
# Wykrywanie urządzeń
# --------------------------------------------------------------------------- #


def test_lista_zawiera_tylko_urzadzenia_wyjsciowe(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    devices = list_output_devices(settings)

    assert [device.name for device in devices] == ["Atrapa głośnika"]
    assert devices[0].max_output_channels == 2
    assert "Atrapa głośnika" in devices[0].describe()


def test_urzadzenie_wybiera_sie_po_fragmencie_nazwy(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    """Indeks urządzenia oznacza inny sprzęt na każdym komputerze — stąd nazwa."""
    assert find_output_device("głośnik", settings) is not None
    assert find_output_device("GŁOŚ", settings) is not None
    assert find_output_device("nie ma takiego", settings) is None


def test_brak_pakietu_sounddevice_nie_wywraca_programu(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)

    assert is_speaker_available(settings) is False


def test_awaria_sterownika_konczy_sie_czytelnym_bledem(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    module = FakeSoundDevice(raise_on_query=RuntimeError("serwer dźwięku nie działa"))
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    with pytest.raises(AudioOutputUnavailableError) as info:
        list_output_devices(settings)

    assert "serwer dźwięku" in info.value.user_message
    assert is_speaker_available(settings) is False


def test_brak_glosnika_daje_podpowiedz(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    only_input = FakeSoundDevice(
        devices=[
            {
                "name": "Sam mikrofon",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "hostapi": 0,
                "default_samplerate": 48_000.0,
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", only_input)
    output = AudioOutput(settings)

    with pytest.raises(AudioOutputUnavailableError) as info:
        output.open(22_050)

    assert "głośnika" in info.value.message


def test_nieznane_urzadzenie_z_konfiguracji_wymienia_dostepne(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings.model_copy(update={"audio_output_device": "Yeti"}))

    with pytest.raises(AudioOutputUnavailableError) as info:
        output.open(22_050)

    assert "Atrapa głośnika" in info.value.hint


# --------------------------------------------------------------------------- #
# Odtwarzanie
# --------------------------------------------------------------------------- #


def test_probki_trafiaja_do_karty_dzwiekowej(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings, volume=1.0)
    output.open(22_050)
    output.write(chunk(1_000))

    stream = fake_sounddevice.output_streams[0]
    played = stream.pull_until_silent()

    assert stream.started is True
    assert int(np.count_nonzero(played)) >= 900
    output.close()
    assert stream.closed is True


def test_niedopasowana_czestotliwosc_jest_przeprobkowana(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WASAPI potrafi odrzucić 22 050 Hz — wtedy liczymy sami, zamiast milczeć."""
    import sys

    module = FakeSoundDevice(fail_rates=(22_050,))
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    output = AudioOutput(settings, volume=1.0)

    output.open(22_050)
    output.write(chunk(1_000, rate=22_050))

    stream = module.output_streams[-1]
    assert stream.samplerate == 48_000
    assert output.sample_rate == 48_000
    played = stream.pull_until_silent()
    # Sygnał wydłużył się proporcjonalnie do stosunku częstotliwości.
    assert int(np.count_nonzero(played)) > 1_500
    output.close()


def test_mono_jest_rozkopiowane_na_kanaly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    # Urządzenie przyjmujące wyłącznie stereo (jednokanałowy strumień odrzuca).
    class StereoOnly(FakeSoundDevice):
        def OutputStream(self, **kwargs: object) -> object:  # noqa: N802 - API sounddevice
            if kwargs.get("channels") == 1:
                raise self.PortAudioError("urządzenie wymaga stereo")
            return super().OutputStream(**kwargs)  # type: ignore[arg-type]

    module = StereoOnly()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    output = AudioOutput(settings, volume=1.0)

    output.open(22_050)
    output.write(chunk(500))

    stream = module.output_streams[-1]
    assert stream.channels == 2
    outdata = np.zeros((256, 2), dtype=np.int16)
    stream.callback(outdata, 256, None, None)
    assert np.array_equal(outdata[:, 0], outdata[:, 1])
    output.close()


def test_glosnosc_z_ustawien_uzytkownika_skaluje_sygnal(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    louder = AudioOutput(settings, volume=1.0)
    louder.open(22_050)
    louder.write(chunk(500))
    loud = fake_sounddevice.output_streams[0].pull_until_silent()
    louder.close()

    quieter = AudioOutput(settings, volume=0.2)
    quieter.open(22_050)
    quieter.write(chunk(500))
    quiet = fake_sounddevice.output_streams[1].pull_until_silent()
    quieter.close()

    assert int(np.abs(quiet).max()) < int(np.abs(loud).max()) // 2


def test_glosnosc_domyslnie_pochodzi_z_user_settings(
    settings: Settings, fake_sounddevice: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    import audio.output

    monkeypatch.setattr(
        audio.output, "get_user_settings", lambda: UserSettings(voice_volume=0.0)
    )
    output = AudioOutput(settings)
    output.open(22_050)
    output.write(chunk(500))

    played = fake_sounddevice.output_streams[0].pull_until_silent()

    assert int(np.abs(played).max()) == 0
    output.close()


def test_anulowanie_wyrzuca_to_co_jeszcze_nie_zagralo(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings, volume=1.0)
    output.open(22_050)
    output.write(chunk(10_000))
    output.cancel()

    played = fake_sounddevice.output_streams[0].pull(256)

    assert int(np.count_nonzero(played)) == 0
    output.close()


def test_anulowanie_w_trakcie_callbacku_nie_zostawia_ogona(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    """Wyścig z wątkiem PortAudio: resztka przerwanego zdania nie może wrócić.

    Callback pobiera blok z kolejki, a w tej samej chwili z innego wątku pada
    ``cancel()``. Bez licznika pokoleń reszta bloku wróciłaby do bufora i
    zagrała po przerwaniu — czyli asystent mówiłby mimo „cicho".
    """
    output = AudioOutput(settings, volume=1.0)
    output.open(22_050)
    stream = fake_sounddevice.output_streams[0]
    output.write(chunk(5_000))

    original_get = output._queue.get_nowait

    def get_and_cancel() -> np.ndarray:
        block = original_get()
        output.cancel()  # dokładnie w środku pracy callbacku
        return block

    output._queue.get_nowait = get_and_cancel  # type: ignore[method-assign]
    stream.pull(256)
    output._queue.get_nowait = original_get  # type: ignore[method-assign]

    assert int(np.count_nonzero(stream.pull(256))) == 0
    output.close()


def test_ponowne_otwarcie_z_ta_sama_czestotliwoscia_nic_nie_zmienia(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings)
    output.open(22_050)
    output.open(22_050)

    assert len(fake_sounddevice.output_streams) == 1
    output.close()


def test_zmiana_czestotliwosci_otwiera_nowy_strumien(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings)
    output.open(22_050)
    output.open(16_000)

    assert len(fake_sounddevice.output_streams) == 2
    assert fake_sounddevice.output_streams[0].closed is True
    output.close()


def test_write_otwiera_wyjscie_gdy_trzeba(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings)
    output.write(chunk(200, rate=16_000))

    assert output.is_open is True
    assert fake_sounddevice.output_streams[0].samplerate == 16_000
    output.close()


def test_pusty_fragment_niczego_nie_otwiera(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    output = AudioOutput(settings)
    output.write(SpeechChunk(samples=np.zeros(0, dtype=np.int16), sample_rate=22_050))

    assert output.is_open is False
    assert fake_sounddevice.output_streams == []


def test_wyjatek_w_callbacku_nie_zabija_strumienia(
    settings: Settings, fake_sounddevice: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wyjątek w wątku PortAudio zamknąłby urządzenie — musi zostać złapany."""
    output = AudioOutput(settings)
    output.open(22_050)
    stream = fake_sounddevice.output_streams[0]

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("awaria w środku callbacku")

    monkeypatch.setattr(np, "concatenate", broken)
    output.write(chunk(500))

    played = stream.pull(256)  # nie może rzucić

    assert int(np.count_nonzero(played)) == 0
    output.close()


def test_play_chunks_odtwarza_i_zamyka(
    settings: Settings, fake_sounddevice: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    import audio.output

    monkeypatch.setattr(audio.output, "get_settings", lambda: settings)
    # Bez wątku PortAudio nic nie „zagra" samo — drain ma nie wisieć w nieskończoność.
    monkeypatch.setattr(audio.output.AudioOutput, "drain", lambda self, timeout=None: None)

    play_chunks([chunk(300), chunk(300)], settings)

    assert fake_sounddevice.output_streams[0].closed is True
