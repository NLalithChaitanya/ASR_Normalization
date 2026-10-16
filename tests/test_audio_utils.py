
"""Tests for audio loading and preprocessing."""

import numpy as np
import pytest
import soundfile as sf

from src.data.audio_utils import (
    AudioValidationError,
    load_audio,
    resample_waveform,
    validate_waveform,
)


def test_validate_waveform_accepts_valid_audio():
    waveform = np.zeros(16000, dtype=np.float32)

    result = validate_waveform(waveform, sample_rate=16000)

    assert result.dtype == np.float32
    assert len(result) == 16000


def test_validate_waveform_rejects_empty_audio():
    with pytest.raises(AudioValidationError, match="empty"):
        validate_waveform(np.array([]), sample_rate=16000)


def test_validate_waveform_rejects_nan():
    waveform = np.array([0.0, np.nan], dtype=np.float32)

    with pytest.raises(AudioValidationError, match="NaN or infinite"):
        validate_waveform(waveform, sample_rate=16000)


def test_validate_waveform_rejects_invalid_duration():
    waveform = np.zeros(100, dtype=np.float32)

    with pytest.raises(AudioValidationError, match="below the minimum"):
        validate_waveform(
            waveform,
            sample_rate=16000,
            min_duration_seconds=1.0,
        )


def test_resample_waveform_changes_sample_count():
    waveform = np.zeros(8000, dtype=np.float32)

    result = resample_waveform(
        waveform,
        original_sample_rate=8000,
        target_sample_rate=16000,
    )

    assert len(result) == 16000
    assert result.dtype == np.float32


def test_resample_same_rate_preserves_waveform():
    waveform = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    result = resample_waveform(
        waveform,
        original_sample_rate=16000,
        target_sample_rate=16000,
    )

    np.testing.assert_array_equal(result, waveform)


def test_load_audio_resamples_and_returns_metadata(tmp_path):
    audio_path = tmp_path / "sample.wav"
    waveform = np.zeros(8000, dtype=np.float32)

    sf.write(audio_path, waveform, 8000)

    result = load_audio(audio_path, target_sample_rate=16000)

    assert result.sample_rate == 16000
    assert len(result.waveform) == 16000
    assert result.metadata.duration_seconds == pytest.approx(1.0)
    assert result.metadata.num_channels == 1


def test_load_audio_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")
