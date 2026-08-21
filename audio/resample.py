"""Format and sample-rate conversions.

Whisper requires 16 kHz mono, and sound cards can impose their own rate (on
Windows typically 44.1 or 48 kHz). These functions are pure NumPy — no native
dependencies, no assumptions about the system.
"""

from __future__ import annotations

import math

import numpy as np

from i18n import t

INT16_SCALE: float = 32768.0


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Reduce an audio block to a single channel (averaging the channels)."""
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return samples[:, 0]
    if np.issubdtype(samples.dtype, np.integer):
        return samples.mean(axis=1).astype(samples.dtype)
    return samples.mean(axis=1)


def int16_to_float32(samples: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in the range [-1, 1] (Whisper's input format)."""
    if samples.dtype == np.float32:
        return samples
    return (samples.astype(np.float32) / INT16_SCALE).clip(-1.0, 1.0)


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM."""
    if samples.dtype == np.int16:
        return samples
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (INT16_SCALE - 1)).astype(np.int16)


def resample_int16(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample an int16 signal to the target rate.

    For an integer ratio (48000 → 16000, 32000 → 16000) we average consecutive
    samples: it acts as a simple low-pass filter and limits aliasing. In the
    remaining cases, linear interpolation.

    This is deliberately simple, without ``scipy``/``soxr`` — for speech ahead of
    the VAD and Whisper it is sufficient, and it adds no native dependency that
    would have to compile on every platform.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(t("cfg.sample_rate_positive"))
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
    """The signal's RMS level in dBFS (−inf for absolute silence)."""
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
