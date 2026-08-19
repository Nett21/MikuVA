"""Konwersje formatu i częstotliwości próbkowania.

Whisper wymaga 16 kHz mono, a karty dźwiękowe potrafią narzucić własną
częstotliwość (na Windowsie typowo 44,1 lub 48 kHz). Te funkcje są czystym
NumPy — bez zależności natywnych, bez założeń o systemie.
"""

from __future__ import annotations

import math

import numpy as np

INT16_SCALE: float = 32768.0


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Sprowadź blok audio do jednego kanału (uśrednienie kanałów)."""
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return samples[:, 0]
    if np.issubdtype(samples.dtype, np.integer):
        return samples.mean(axis=1).astype(samples.dtype)
    return samples.mean(axis=1)


def int16_to_float32(samples: np.ndarray) -> np.ndarray:
    """Zamień PCM int16 na float32 w zakresie [-1, 1] (format wejściowy Whispera)."""
    if samples.dtype == np.float32:
        return samples
    return (samples.astype(np.float32) / INT16_SCALE).clip(-1.0, 1.0)


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    """Zamień float32 [-1, 1] na PCM int16."""
    if samples.dtype == np.int16:
        return samples
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (INT16_SCALE - 1)).astype(np.int16)


def resample_int16(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Przepróbkuj sygnał int16 do docelowej częstotliwości.

    Dla całkowitej krotności (48000 → 16000, 32000 → 16000) używamy uśredniania
    kolejnych próbek: działa jak prosty filtr dolnoprzepustowy i ogranicza
    aliasing. W pozostałych przypadkach interpolacja liniowa.

    To celowo proste rozwiązanie bez ``scipy``/``soxr`` — dla mowy przed VAD-em
    i Whisperem jest wystarczające, a nie dokłada zależności natywnej, która
    musiałaby się skompilować na każdej platformie.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("częstotliwość próbkowania musi być dodatnia")
    if source_rate == target_rate or samples.size == 0:
        return samples

    mono = to_mono(samples)

    if source_rate % target_rate == 0:
        factor = source_rate // target_rate
        usable = (mono.size // factor) * factor
        if usable == 0:
            return np.zeros(0, dtype=np.int16)
        blocks = mono[:usable].astype(np.float32).reshape(-1, factor)
        return blocks.mean(axis=1).astype(np.int16)

    target_length = max(1, int(math.floor(mono.size * target_rate / source_rate)))
    source_positions = np.arange(mono.size, dtype=np.float64)
    target_positions = np.linspace(0, mono.size - 1, target_length, dtype=np.float64)
    resampled = np.interp(target_positions, source_positions, mono.astype(np.float64))
    return resampled.astype(np.int16)


def rms_dbfs(samples: np.ndarray) -> float:
    """Poziom skuteczny sygnału w dBFS (−inf dla ciszy absolutnej)."""
    if samples.size == 0:
        return -float("inf")
    as_float = int16_to_float32(samples) if samples.dtype == np.int16 else samples
    mean_square = float(np.mean(np.square(as_float, dtype=np.float64)))
    if mean_square <= 0.0:
        return -float("inf")
    return 10.0 * math.log10(mean_square)


__all__ = [
    "INT16_SCALE",
    "float32_to_int16",
    "int16_to_float32",
    "resample_int16",
    "rms_dbfs",
    "to_mono",
]
