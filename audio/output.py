"""Odtwarzanie dźwięku (Faza 4): kolejka próbek → karta dźwiękowa.

Wyjście audio jest osobnym modułem od syntezy z tego samego powodu, dla którego
mikrofon jest osobny od Whispera: to inne urządzenie, inny cykl życia i inne
tryby awarii. Głośnika może w ogóle nie być (serwer, kontener, WSL bez serwera
dźwięku) — i wtedy asystent ma dalej działać, tyle że po cichu.

Ten moduł nie zna Pipera ani żadnego silnika mowy. Przyjmuje
:class:`audio.tts.SpeechChunk` i realizuje kontrakt :class:`audio.tts.AudioSink`.

Uwagi o przenośności — nic tutaj nie zakłada konkretnego sprzętu:

* urządzenie wybieramy po **fragmencie nazwy** (``AUDIO_OUTPUT_DEVICE``), nigdy
  po indeksie: indeks 3 to inny sprzęt na każdym komputerze,
* jeśli karta nie przyjmie częstotliwości głosu (22 050 Hz bywa odrzucane przez
  WASAPI w trybie współdzielonym), próbujemy kolejnych i przepróbkowujemy
  sygnał w Pythonie zamiast zgłaszać błąd,
* mono jest rozkopiowywane na tyle kanałów, ile urządzenie faktycznie ma.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Any, Final

import numpy as np

from audio.microphone import suppressed_native_stderr
from audio.resample import resample_int16
from audio.tts import SpeechChunk
from config import Settings, get_settings, get_user_settings, pip_install_hint
from i18n import t

logger = logging.getLogger(__name__)

# Kolejne częstotliwości do wypróbowania, gdy urządzenie odrzuci natywną.
_FALLBACK_RATES: Final[tuple[int, ...]] = (48_000, 44_100, 22_050, 16_000)


class AudioOutputError(RuntimeError):
    """Błąd wyjścia audio z komunikatem gotowym dla użytkownika."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


class AudioOutputUnavailableError(AudioOutputError):
    """Na tej maszynie nie ma czego (albo czym) odtwarzać."""


@dataclass(frozen=True, slots=True)
class AudioOutputDevice:
    """Urządzenie wyjściowe zgłoszone przez PortAudio."""

    index: int
    name: str
    host_api: str
    max_output_channels: int
    default_samplerate: float

    def describe(self) -> str:
        return (
            f"{self.name} [{self.host_api}] "
            f"({self.max_output_channels} kan., {self.default_samplerate:.0f} Hz)"
        )


def _load_sounddevice() -> ModuleType:
    """Załaduj ``sounddevice`` zamieniając każdy problem na czytelny wyjątek."""
    try:
        import sounddevice  # noqa: PLC0415 - import celowo leniwy
    except ImportError as exc:
        raise AudioOutputUnavailableError(
            t("out.no_package"),
            hint=pip_install_hint(),
        ) from exc
    except OSError as exc:
        raise AudioOutputUnavailableError(
            t("out.portaudio_failed", error=exc),
            hint=t("mic.portaudio_hint"),
        ) from exc
    return sounddevice


def list_output_devices(settings: Settings | None = None) -> list[AudioOutputDevice]:
    """Urządzenia wyjściowe widoczne dla PortAudio."""
    active = settings or get_settings()
    sounddevice = _load_sounddevice()

    try:
        with suppressed_native_stderr(active.audio_suppress_device_warnings):
            raw_devices = sounddevice.query_devices()
            host_apis = sounddevice.query_hostapis()
    except Exception as exc:  # PortAudioError i wszystko, co rzuci sterownik
        raise AudioOutputUnavailableError(
            t("out.devices_failed", error=exc),
            hint=t("out.sound_server_hint"),
        ) from exc

    devices: list[AudioOutputDevice] = []
    for index, entry in enumerate(raw_devices):
        channels = int(entry.get("max_output_channels", 0))
        if channels <= 0:
            continue
        host_index = int(entry.get("hostapi", -1))
        host_name = "?"
        if 0 <= host_index < len(host_apis):
            host_name = str(host_apis[host_index].get("name", "?"))
        devices.append(
            AudioOutputDevice(
                index=index,
                name=str(entry.get("name", f"urządzenie {index}")),
                host_api=host_name,
                max_output_channels=channels,
                default_samplerate=float(entry.get("default_samplerate", 0.0) or 0.0),
            )
        )
    return devices


def find_output_device(
    name_fragment: str, settings: Settings | None = None
) -> AudioOutputDevice | None:
    """Znajdź głośnik po fragmencie nazwy (bez rozróżniania wielkości liter)."""
    fragment = name_fragment.strip().lower()
    if not fragment:
        return None
    for device in list_output_devices(settings):
        if fragment in device.name.lower():
            return device
    return None


def is_speaker_available(settings: Settings | None = None) -> bool:
    """Czy da się cokolwiek odtworzyć na tej maszynie? Nigdy nie rzuca."""
    try:
        return bool(list_output_devices(settings))
    except AudioOutputError as exc:
        logger.info("Odtwarzanie niedostępne: %s", exc.message)
        return False
    except Exception as exc:  # pragma: no cover - nietypowe błędy sterownika
        logger.warning("Nieoczekiwany błąd przy sprawdzaniu głośnika: %s", exc)
        return False


class AudioOutput:
    """Odtwarzanie strumienia mowy przez PortAudio.

    Realizuje kontrakt ``audio.tts.AudioSink``. Próbki trafiają do kolejki, a
    wątek PortAudio bierze je własnym tempem — dokładnie odwrotnie niż przy
    nagrywaniu. Dzięki temu synteza może wyprzedzać odtwarzanie i nie ma
    przerw między zdaniami.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        device_name: str | None = None,
        volume: float | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._device_name = (
            device_name if device_name is not None else self._settings.audio_output_device
        )
        self._volume = volume
        self._lock = threading.RLock()
        self._stream: Any = None
        self._device: AudioOutputDevice | None = None
        self._stream_rate = 0
        self._source_rate = 0
        self._channels = 1
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._pending = np.zeros(0, dtype=np.int16)
        self._queued_samples = 0
        self._generation = 0
        self._idle = threading.Event()
        self._idle.set()
        self._underruns = 0

    # --- właściwości ------------------------------------------------------- #

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def device(self) -> AudioOutputDevice | None:
        return self._device

    @property
    def sample_rate(self) -> int:
        """Częstotliwość otwartego strumienia (0 = zamknięty)."""
        return self._stream_rate

    @property
    def underruns(self) -> int:
        """Ile razy zabrakło danych w trakcie odtwarzania (zacięcia)."""
        return self._underruns

    def describe(self) -> str:
        base = "urządzenie domyślne" if self._device is None else self._device.describe()
        if self._stream_rate and self._source_rate and self._stream_rate != self._source_rate:
            return f"{base}, {self._source_rate} → {self._stream_rate} Hz"
        if self._stream_rate:
            return f"{base}, {self._stream_rate} Hz"
        return base

    def _effective_volume(self) -> float:
        """Głośność z ustawień użytkownika (czytana na bieżąco, jak reszta)."""
        if self._volume is not None:
            return max(0.0, min(1.0, self._volume))
        return max(0.0, min(1.0, get_user_settings().voice_volume))

    # --- otwieranie strumienia --------------------------------------------- #

    def _resolve_device(self) -> AudioOutputDevice | None:
        devices = list_output_devices(self._settings)
        if not devices:
            raise AudioOutputUnavailableError(
                t("out.none_reported"),
                hint=t("out.none_reported_hint"),
            )
        if not self._device_name:
            return None  # domyślne urządzenie systemowe
        match = find_output_device(self._device_name, self._settings)
        if match is None:
            available = ", ".join(device.name for device in devices[:8])
            raise AudioOutputUnavailableError(
                t("out.not_matched", name=self._device_name),
                hint=t("out.available_devices", devices=available),
            )
        return match

    def _candidate_configurations(
        self, device: AudioOutputDevice | None, wanted_rate: int
    ) -> list[tuple[int, int]]:
        """Warianty (częstotliwość, kanały) do wypróbowania — od najlepszego."""
        max_channels = device.max_output_channels if device else 2
        channels = [1] if max_channels <= 1 else [1, min(2, max_channels)]

        rates: list[int] = [wanted_rate]
        if device is not None and device.default_samplerate:
            rates.append(int(device.default_samplerate))
        rates.extend(_FALLBACK_RATES)

        seen: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int]] = []
        for rate in rates:
            for channel_count in channels:
                key = (rate, channel_count)
                if rate <= 0 or key in seen:
                    continue
                seen.add(key)
                candidates.append(key)
        return candidates

    def open(self, sample_rate: int) -> None:
        """Otwórz wyjście dla dźwięku o tej częstotliwości (idempotentne)."""
        with self._lock:
            if self._stream is not None and self._source_rate == sample_rate:
                return
            if self._stream is not None:
                self._close_stream()

            sounddevice = _load_sounddevice()
            device = self._resolve_device()
            self._device = device
            self._source_rate = sample_rate
            last_error: Exception | None = None

            for rate, channels in self._candidate_configurations(device, sample_rate):
                try:
                    with suppressed_native_stderr(
                        self._settings.audio_suppress_device_warnings
                    ):
                        stream = sounddevice.OutputStream(
                            samplerate=rate,
                            channels=channels,
                            dtype="int16",
                            blocksize=0,  # 0 = niech PortAudio dobierze sam
                            device=device.index if device else None,
                            callback=self._callback,
                        )
                        stream.start()
                except Exception as exc:  # PortAudioError zależny od sterownika
                    last_error = exc
                    logger.debug(
                        "Nie udało się otworzyć wyjścia %s Hz / %s kan.: %s", rate, channels, exc
                    )
                    continue

                self._stream = stream
                self._stream_rate = rate
                self._channels = channels
                logger.info(
                    "Wyjście audio otwarte: %s, %s Hz, %s kan.%s",
                    device.name if device else "urządzenie domyślne",
                    rate,
                    channels,
                    "" if rate == sample_rate else f" (przepróbkowanie z {sample_rate} Hz)",
                )
                return

            raise AudioOutputUnavailableError(
                t(
                    "out.open_failed",
                    error=f" ({last_error})." if last_error else ".",
                ),
                hint=t("out.open_failed_hint"),
            )

    # --- ścieżka danych ----------------------------------------------------- #

    def write(self, chunk: SpeechChunk) -> None:
        """Dołóż fragment do odtworzenia (nie blokuje, chyba że kolejka jest pełna)."""
        if chunk.is_empty:
            return
        with self._lock:
            if self._stream is None:
                self.open(chunk.sample_rate)
            samples = chunk.samples
            if chunk.sample_rate != self._stream_rate:
                samples = resample_int16(samples, chunk.sample_rate, self._stream_rate)

            volume = self._effective_volume()
            if volume < 0.999:
                samples = (samples.astype(np.float32) * volume).astype(np.int16)

            limit = int(self._stream_rate * self._settings.audio_output_queue_seconds)
            queued = self._queued_samples
            self._queued_samples = queued + samples.size
            self._idle.clear()
            self._queue.put(np.ascontiguousarray(samples, dtype=np.int16))

        if queued > limit:
            # Synteza wyprzedziła odtwarzanie o więcej niż pozwala bufor —
            # czekamy zamiast puchnąć w nieskończoność. Limit czasu chroni
            # przed zawieszeniem, gdyby strumień padł w międzyczasie.
            self._wait_below(limit, timeout=self._settings.audio_output_queue_seconds * 2)

    def _wait_below(self, limit: int, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self._queued_samples > limit and time.monotonic() < deadline:
            if self._stream is None:
                return
            time.sleep(0.02)

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Wywoływane przez PortAudio w jego wątku — musi być krótkie i bez wyjątków."""
        del time_info  # sygnatura narzucona przez PortAudio
        if status:
            self._underruns += 1
            logger.debug("Status strumienia wyjściowego: %s", status)

        try:
            needed = frames
            # Numer „pokolenia" bufora: cancel() z innego wątku podnosi go, a my
            # sprawdzamy na końcu, czy w międzyczasie nas nie unieważniono.
            # Inaczej resztka odrzuconego zdania wróciłaby do _pending i zagrała
            # mimo przerwania — dokładnie tego, czego barge-in ma nie robić.
            generation = self._generation
            block = self._pending
            while block.size < needed:
                try:
                    block = np.concatenate((block, self._queue.get_nowait()))
                except queue.Empty:
                    break

            available = min(needed, block.size)
            if available:
                outdata[:available] = block[:available].reshape(-1, 1).repeat(
                    self._channels, axis=1
                )
                self._queued_samples = max(0, self._queued_samples - available)
            if available < needed:
                outdata[available:] = 0
                if self._queue.empty():
                    self._idle.set()
            if generation == self._generation:
                self._pending = block[available:]
        except Exception as exc:  # wyjątek w callbacku zabiłby strumień
            logger.warning("Błąd w callbacku odtwarzania: %s", exc)
            with contextlib.suppress(Exception):
                outdata[:] = 0

    # --- sterowanie ---------------------------------------------------------- #

    def drain(self, timeout: float | None = None) -> None:
        """Poczekaj, aż wszystko z kolejki zostanie odtworzone."""
        if self._stream is None:
            return
        self._idle.wait(timeout if timeout is not None else 120.0)
        # Bufor PortAudio ma jeszcze kilkadziesiąt milisekund dźwięku po tym,
        # jak oddaliśmy ostatnią próbkę — bez tej chwili koniec zdania bywa ucięty.
        time.sleep(0.12)

    def cancel(self) -> None:
        """Przerwij odtwarzanie natychmiast (barge-in, Ctrl+C)."""
        with self._lock:
            self._generation += 1
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._pending = np.zeros(0, dtype=np.int16)
            self._queued_samples = 0
            self._idle.set()

    def close(self) -> None:
        """Zamknij urządzenie (można otworzyć ponownie przez :meth:`open`)."""
        with self._lock:
            self.cancel()
            self._close_stream()

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        self._stream_rate = 0
        self._source_rate = 0
        if stream is None:
            return
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()
        logger.debug("Wyjście audio zamknięte (zacięcia: %s)", self._underruns)

    def __enter__(self) -> AudioOutput:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def play_chunks(
    chunks: Sequence[SpeechChunk], settings: Settings | None = None
) -> None:
    """Odtwórz gotowe fragmenty i poczekaj na koniec (diagnostyka, testy sprzętu)."""
    if not chunks:
        return
    output = AudioOutput(settings)
    try:
        output.open(chunks[0].sample_rate)
        for chunk in chunks:
            output.write(chunk)
        output.drain()
    finally:
        output.close()


__all__ = [
    "AudioOutput",
    "AudioOutputDevice",
    "AudioOutputError",
    "AudioOutputUnavailableError",
    "find_output_device",
    "is_speaker_available",
    "list_output_devices",
    "play_chunks",
]
