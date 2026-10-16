
"""
Audio utilities for multilingual ASR experiments.

Responsibilities:
- Load audio from disk.
- Convert stereo audio to mono when requested.
- Resample waveforms to the model's required sample rate.
- Validate waveform values and duration.
- Provide reusable audio metadata.

Waveforms are returned as float32 NumPy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


LOGGER = logging.getLogger(__name__)


class AudioValidationError(ValueError):
    """Raised when an audio file or waveform is invalid."""


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata describing a loaded audio waveform."""

    path: str
    sample_rate: int
    num_samples: int
    duration_seconds: float
    num_channels: int
    peak_amplitude: float


@dataclass(frozen=True)
class AudioData:
    """A validated mono audio waveform and its metadata."""

    waveform: np.ndarray
    sample_rate: int
    metadata: AudioMetadata


def _validate_sample_rate(sample_rate: int) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer")

    if sample_rate <= 0:
        raise AudioValidationError("sample_rate must be positive")


def validate_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    min_duration_seconds: float = 0.0,
    max_duration_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Validate a mono waveform and return it as float32.

    The waveform must be a one-dimensional, non-empty array containing
    only finite values.
    """
    _validate_sample_rate(sample_rate)

    array = np.asarray(waveform)

    if array.ndim != 1:
        raise AudioValidationError(
            f"Expected mono 1-D waveform, got shape {array.shape}"
        )

    if array.size == 0:
        raise AudioValidationError("Waveform is empty")

    if not np.issubdtype(array.dtype, np.number):
        raise AudioValidationError("Waveform must contain numeric values")

    if not np.isfinite(array).all():
        raise AudioValidationError("Waveform contains NaN or infinite values")

    array = array.astype(np.float32, copy=False)

    duration = len(array) / sample_rate

    if duration < min_duration_seconds:
        raise AudioValidationError(
            f"Audio duration {duration:.3f}s is below the minimum "
            f"{min_duration_seconds:.3f}s"
        )

    if (
        max_duration_seconds is not None
        and duration > max_duration_seconds
    ):
        raise AudioValidationError(
            f"Audio duration {duration:.3f}s exceeds the maximum "
            f"{max_duration_seconds:.3f}s"
        )

    return array


def resample_waveform(
    waveform: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """
    Resample a waveform using polyphase filtering.

    If the sample rates match, return a float32 representation without
    performing unnecessary resampling.
    """
    _validate_sample_rate(original_sample_rate)
    _validate_sample_rate(target_sample_rate)

    array = np.asarray(waveform)

    if array.ndim != 1:
        raise AudioValidationError(
            "resample_waveform expects a mono 1-D waveform"
        )

    if array.size == 0:
        raise AudioValidationError("Cannot resample an empty waveform")

    if not np.isfinite(array).all():
        raise AudioValidationError("Waveform contains NaN or infinite values")

    array = array.astype(np.float32, copy=False)

    if original_sample_rate == target_sample_rate:
        return array

    # Reduce the rate ratio to avoid unnecessary computation.
    from math import gcd

    divisor = gcd(original_sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = original_sample_rate // divisor

    resampled = resample_poly(array, up=up, down=down)

    return np.asarray(resampled, dtype=np.float32)


def load_audio(
    audio_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
    min_duration_seconds: float = 0.0,
    max_duration_seconds: Optional[float] = None,
) -> AudioData:
    """
    Load audio from disk and optionally convert it to mono and resample it.

    Args:
        audio_path: Path to a supported audio file.
        target_sample_rate: Desired output sample rate in Hz.
        mono: Average channels to produce mono audio when True.
        min_duration_seconds: Optional minimum permitted duration.
        max_duration_seconds: Optional maximum permitted duration.

    Returns:
        AudioData containing a float32 waveform and metadata.
    """
    path = Path(audio_path)

    _validate_sample_rate(target_sample_rate)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, original_sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError) as exc:
        raise AudioValidationError(
            f"Could not read audio file '{path}': {exc}"
        ) from exc

    if waveform.size == 0:
        raise AudioValidationError(f"Audio file is empty: {path}")

    if not np.isfinite(waveform).all():
        raise AudioValidationError(
            f"Audio file contains non-finite values: {path}"
        )

    original_channels = waveform.shape[1]

    if mono:
        # Averaging channels is a simple, reproducible mono conversion.
        waveform = waveform.mean(axis=1)
    elif original_channels == 1:
        waveform = waveform[:, 0]
    else:
        raise AudioValidationError(
            "This pipeline currently requires mono audio. "
            "Set mono=True to combine channels."
        )

    waveform = resample_waveform(
        waveform,
        original_sample_rate=int(original_sample_rate),
        target_sample_rate=target_sample_rate,
    )

    waveform = validate_waveform(
        waveform,
        sample_rate=target_sample_rate,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )

    metadata = AudioMetadata(
        path=str(path.resolve()),
        sample_rate=target_sample_rate,
        num_samples=len(waveform),
        duration_seconds=len(waveform) / target_sample_rate,
        num_channels=original_channels,
        peak_amplitude=float(np.max(np.abs(waveform))),
    )

    LOGGER.debug(
        "Loaded %s: %.2fs, %d Hz, %d channel(s)",
        path,
        metadata.duration_seconds,
        metadata.sample_rate,
        metadata.num_channels,
    )

    return AudioData(
        waveform=waveform,
        sample_rate=target_sample_rate,
        metadata=metadata,
    )
