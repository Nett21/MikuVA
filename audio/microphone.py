"""Ciągłe nasłuchiwanie mikrofonu.

Wybór biblioteki: **sounddevice** (PortAudio), nie PyAudio. Powody:

* koła (wheels) ``sounddevice`` niosą ze sobą skompilowaną bibliotekę PortAudio
  dla Linuksa, Windowsa i macOS-a — ``pip install sounddevice`` działa bez
  kompilatora i bez pakietów systemowych. PyAudio na Linuksie wymaga nagłówków
  ``portaudio19-dev``/``portaudio``, czyli kroku poza pip-em, innego na każdej
  dystrybucji,
* callback dostaje gotową tablicę NumPy — nie ma ręcznego rozpakowywania
  bajtów, więc ten sam kod obsługuje int16 na każdej platformie,
* jednolita enumeracja urządzeń niezależnie od API systemowego (ALSA/PipeWire
  przez PortAudio na Linuksie, WASAPI/MME na Windowsie),
* biblioteka jest aktywnie utrzymywana, a jej API nie zmienia się między
  systemami — nie ma potrzeby rozgałęziania kodu na Windows/Linux.

Brak mikrofonu, brak PortAudio albo brak samego pakietu nie przerywa programu:
wszystkie te sytuacje kończą się wyjątkiem :class:`MicrophoneUnavailableError`
z czytelnym komunikatem, który wywołujący łapie i przełącza się na tryb tekstowy.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Any, Final

import numpy as np

from audio.resample import resample_int16, to_mono
from config import Settings, get_settings, pip_install_hint
from i18n import t

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: Final[int] = 16_000


class MicrophoneError(RuntimeError):
    """Błąd wejścia audio z komunikatem gotowym do pokazania użytkownikowi."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n       Podpowiedź: {self.hint}"
        return self.message


class MicrophoneUnavailableError(MicrophoneError):
    """Mikrofon (albo cała warstwa audio) jest niedostępny na tej maszynie."""


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """Urządzenie wejściowe zgłoszone przez PortAudio."""

    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float

    def describe(self) -> str:
        return (
            f"{self.name} [{self.host_api}] "
            f"({self.max_input_channels} kan., {self.default_samplerate:.0f} Hz)"
        )


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """Pojedyncza ramka PCM: mono, int16, docelowa częstotliwość próbkowania."""

    samples: np.ndarray
    sample_rate: int
    timestamp: float

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sample_rate if self.sample_rate else 0.0


def _load_sounddevice() -> ModuleType:
    """Załaduj ``sounddevice`` zamieniając każdy problem na czytelny wyjątek.

    ``import sounddevice`` potrafi rzucić ``OSError`` (brak biblioteki PortAudio
    w systemie), a nie tylko ``ImportError`` — dlatego łapiemy oba.
    """
    try:
        import sounddevice  # noqa: PLC0415 - import celowo leniwy
    except ImportError as exc:
        raise MicrophoneUnavailableError(
            t("mic.no_package"),
            hint=pip_install_hint(),
        ) from exc
    except OSError as exc:
        raise MicrophoneUnavailableError(
            t("mic.portaudio_failed", error=exc),
            hint=t("mic.portaudio_hint"),
        ) from exc
    return sounddevice


@contextlib.contextmanager
def _suppressed_native_stderr(enabled: bool) -> Iterator[None]:
    """Wycisz komunikaty bibliotek natywnych pisane wprost na deskryptor 2.

    ALSA i JACK potrafią zasypać terminal ostrzeżeniami przy enumeracji
    urządzeń. Przekierowanie deskryptora jest neutralne platformowo
    (``os.dup2`` i ``os.devnull`` działają tak samo na Windowsie) i obejmuje
    wyłącznie wywołanie PortAudio — wyjątki Pythona lecą normalnie.
    """
    if not enabled:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    try:
        saved_fd = os.dup(stderr_fd)
    except OSError:
        yield
        return

    devnull_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            os.dup2(saved_fd, stderr_fd)
        with contextlib.suppress(OSError):
            os.close(saved_fd)
        if devnull_fd is not None:
            with contextlib.suppress(OSError):
                os.close(devnull_fd)


# Wyjście audio (Faza 4) enumeruje urządzenia tym samym PortAudio i musi wyciszać
# te same ostrzeżenia sterowników — stąd publiczna nazwa zamiast kopii kodu.
suppressed_native_stderr = _suppressed_native_stderr


def list_input_devices(settings: Settings | None = None) -> list[AudioDevice]:
    """Zwróć urządzenia wejściowe widoczne dla PortAudio."""
    active = settings or get_settings()
    sounddevice = _load_sounddevice()

    try:
        with _suppressed_native_stderr(active.audio_suppress_device_warnings):
            raw_devices = sounddevice.query_devices()
            host_apis = sounddevice.query_hostapis()
    except Exception as exc:  # PortAudioError i wszystko, co rzuci sterownik
        raise MicrophoneUnavailableError(
            t("mic.devices_failed", error=exc),
            hint=t("mic.sound_server_hint"),
        ) from exc

    devices: list[AudioDevice] = []
    for index, entry in enumerate(raw_devices):
        channels = int(entry.get("max_input_channels", 0))
        if channels <= 0:
            continue
        host_index = int(entry.get("hostapi", -1))
        host_name = "?"
        if 0 <= host_index < len(host_apis):
            host_name = str(host_apis[host_index].get("name", "?"))
        devices.append(
            AudioDevice(
                index=index,
                name=str(entry.get("name", f"urządzenie {index}")),
                host_api=host_name,
                max_input_channels=channels,
                default_samplerate=float(entry.get("default_samplerate", 0.0) or 0.0),
            )
        )
    return devices


def find_input_device(
    name_fragment: str, settings: Settings | None = None
) -> AudioDevice | None:
    """Znajdź urządzenie po fragmencie nazwy (bez rozróżniania wielkości liter)."""
    fragment = name_fragment.strip().lower()
    if not fragment:
        return None
    for device in list_input_devices(settings):
        if fragment in device.name.lower():
            return device
    return None


def is_microphone_available(settings: Settings | None = None) -> bool:
    """Czy da się nagrywać na tej maszynie? Nigdy nie rzuca wyjątku."""
    active = settings or get_settings()
    if not active.mic_enabled:
        return False
    try:
        return bool(list_input_devices(active))
    except MicrophoneError as exc:
        logger.info("Mikrofon niedostępny: %s", exc.message)
        return False
    except Exception as exc:  # pragma: no cover - nietypowe błędy sterownika
        logger.warning("Nieoczekiwany błąd przy sprawdzaniu mikrofonu: %s", exc)
        return False


class Microphone:
    """Ciągły nasłuch mikrofonu z kolejką ramek o stałej długości.

    Dane z PortAudio przychodzą w osobnym wątku (callback) i trafiają do
    kolejki. Konsument (VAD/potok) czyta ramki własnym tempem; przy zatorze
    najstarsze ramki są odrzucane, żeby opóźnienie nie rosło w nieskończoność.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        device_name: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._device_name = (
            device_name if device_name is not None else self._settings.audio_input_device
        )
        self._target_rate = self._settings.audio_sample_rate
        self._frame_samples = self._settings.frame_samples

        max_frames = max(
            1,
            int(self._settings.audio_queue_seconds * 1000 / self._settings.audio_frame_ms),
        )
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=max_frames)
        self._lock = threading.Lock()
        self._stream: Any = None
        self._device: AudioDevice | None = None
        self._stream_rate: int = self._target_rate
        self._stream_channels: int = 1
        self._dropped_frames = 0
        self._overflow_events = 0
        self._pending: np.ndarray = np.zeros(0, dtype=np.int16)

    # --- właściwości ----------------------------------------------------- #

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    @property
    def device(self) -> AudioDevice | None:
        return self._device

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def dropped_frames(self) -> int:
        """Ile ramek odrzucono z powodu zbyt wolnego konsumenta."""
        return self._dropped_frames

    @property
    def overflow_events(self) -> int:
        """Ile razy PortAudio zgłosiło przepełnienie bufora wejściowego."""
        return self._overflow_events

    # --- uruchamianie ---------------------------------------------------- #

    def _resolve_device(self) -> AudioDevice | None:
        devices = list_input_devices(self._settings)
        if not devices:
            raise MicrophoneUnavailableError(
                t("mic.none_reported"),
                hint=t("mic.none_reported_hint"),
            )

        if not self._device_name:
            return None  # domyślne urządzenie systemowe

        match = find_input_device(self._device_name, self._settings)
        if match is None:
            available = ", ".join(device.name for device in devices[:8])
            raise MicrophoneUnavailableError(
                t("mic.not_matched", name=self._device_name),
                hint=t("mic.available_devices", devices=available),
            )
        return match

    def _candidate_configurations(
        self, device: AudioDevice | None
    ) -> list[tuple[int, int]]:
        """Kolejne warianty (częstotliwość, liczba kanałów) do wypróbowania.

        Nie każde urządzenie przyjmie 16 kHz — WASAPI w trybie współdzielonym
        często wymusza częstotliwość miksera. Wtedy otwieramy strumień na
        natywnej częstotliwości i przepróbkowujemy w Pythonie.
        """
        native_rate = int(device.default_samplerate) if device else 0
        max_channels = device.max_input_channels if device else 1

        candidates: list[tuple[int, int]] = [(self._target_rate, 1)]
        if native_rate and native_rate != self._target_rate:
            candidates.append((native_rate, 1))
        for fallback_rate in (48_000, 44_100):
            if fallback_rate != self._target_rate:
                candidates.append((fallback_rate, 1))
        if max_channels > 1:
            extra = [(rate, min(2, max_channels)) for rate, _ in list(candidates)]
            candidates.extend(extra)
        return candidates

    def start(self) -> None:
        """Otwórz strumień wejściowy i zacznij zbierać ramki."""
        with self._lock:
            if self._stream is not None:
                return

            sounddevice = _load_sounddevice()
            device = self._resolve_device()
            self._device = device
            last_error: Exception | None = None

            for rate, channels in self._candidate_configurations(device):
                block_samples = max(1, int(rate * self._settings.audio_frame_ms / 1000))
                try:
                    with _suppressed_native_stderr(
                        self._settings.audio_suppress_device_warnings
                    ):
                        stream = sounddevice.InputStream(
                            samplerate=rate,
                            channels=channels,
                            dtype="int16",
                            blocksize=block_samples,
                            device=device.index if device else None,
                            callback=self._callback,
                        )
                        stream.start()
                except Exception as exc:  # PortAudioError zależny od sterownika
                    last_error = exc
                    logger.debug(
                        "Nie udało się otworzyć strumienia %s Hz / %s kan.: %s",
                        rate,
                        channels,
                        exc,
                    )
                    continue

                self._stream = stream
                self._stream_rate = rate
                self._stream_channels = channels
                self._pending = np.zeros(0, dtype=np.int16)
                logger.info(
                    "Mikrofon uruchomiony: %s, %s Hz, %s kan.%s",
                    device.name if device else "urządzenie domyślne",
                    rate,
                    channels,
                    "" if rate == self._target_rate else f" (przepróbkowanie do {self._target_rate} Hz)",
                )
                return

            raise MicrophoneUnavailableError(
                t(
                    "mic.stream_failed",
                    error=f" ({last_error})" if last_error else ".",
                ),
                hint=t("mic.stream_failed_hint"),
            )

    def stop(self) -> None:
        """Zatrzymaj nasłuch i zwolnij urządzenie."""
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()
        self.clear()
        logger.info(
            "Mikrofon zatrzymany (odrzucone ramki: %s, przepełnienia: %s)",
            self._dropped_frames,
            self._overflow_events,
        )

    def clear(self) -> None:
        """Opróżnij kolejkę ramek (np. przed nowym nasłuchem)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._pending = np.zeros(0, dtype=np.int16)

    def __enter__(self) -> Microphone:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # --- ścieżka danych --------------------------------------------------- #

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002 - sygnatura narzucona przez PortAudio
        time_info: Any,  # noqa: ARG002 - sygnatura narzucona przez PortAudio
        status: Any,
    ) -> None:
        """Wywoływane przez PortAudio w jego własnym wątku — musi być krótkie."""
        if status:
            self._overflow_events += 1
            logger.debug("Status strumienia wejściowego: %s", status)

        try:
            mono = to_mono(np.asarray(indata))
            if self._stream_rate != self._target_rate:
                mono = resample_int16(mono, self._stream_rate, self._target_rate)
            self._emit(mono.astype(np.int16, copy=True))
        except Exception as exc:  # wyjątek w callbacku zabiłby strumień
            logger.warning("Błąd w callbacku audio: %s", exc)

    def _emit(self, samples: np.ndarray) -> None:
        """Sklej próbki w ramki o dokładnej długości i wrzuć je do kolejki."""
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))

        frame_size = self._frame_samples
        complete = (samples.size // frame_size) * frame_size
        now = time.monotonic()

        for start in range(0, complete, frame_size):
            frame = AudioFrame(
                samples=samples[start : start + frame_size],
                sample_rate=self._target_rate,
                timestamp=now,
            )
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
                    self._dropped_frames += 1
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(frame)

        self._pending = samples[complete:]

    def read(self, timeout: float | None = 0.5) -> AudioFrame | None:
        """Pobierz kolejną ramkę albo ``None``, gdy w zadanym czasie nic nie przyszło."""
        if self._stream is None:
            raise MicrophoneError(
                t("mic.not_started"),
                hint=t("mic.not_started_hint"),
            )
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def frames(self, timeout: float | None = 0.5) -> Iterator[AudioFrame]:
        """Strumień ramek jako generator; pomija okresy bez danych."""
        while self._stream is not None:
            frame = self.read(timeout=timeout)
            if frame is not None:
                yield frame


__all__ = [
    "TARGET_SAMPLE_RATE",
    "AudioDevice",
    "AudioFrame",
    "Microphone",
    "MicrophoneError",
    "MicrophoneUnavailableError",
    "find_input_device",
    "is_microphone_available",
    "list_input_devices",
    "suppressed_native_stderr",
]
