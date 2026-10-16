
"""Tests for dataset manifest validation."""

import pandas as pd
import pytest

from src.data.prepare_dataset import (
    ManifestValidationError,
    validate_manifest,
)


@pytest.fixture
def valid_manifest():
    return pd.DataFrame(
        {
            "utt_id": ["utt001", "utt002"],
            "lang": ["hi", "hi"],
            "cond": ["clean", "noisy"],
            "audio_path": ["audio/a.wav", "audio/b.wav"],
            "reference": ["नमस्ते", "आप कैसे हैं"],
            "duration": [2.5, 3.0],
            "gender": ["female", "male"],
            "snr_db": [None, 10.0],
            "noise_file": [None, "noise.wav"],
            "noise_cat": [None, "environmental"],
            "pair_id": ["pair001", "pair001"],
        }
    )


def test_valid_manifest_passes(valid_manifest):
    result = validate_manifest(valid_manifest)

    assert len(result) == 2
    assert set(result["lang"]) == {"hi"}


def test_missing_columns_fail(valid_manifest):
    invalid = valid_manifest.drop(columns=["reference"])

    with pytest.raises(ManifestValidationError, match="Missing required"):
        validate_manifest(invalid)


def test_duplicate_keys_fail(valid_manifest):
    duplicate = pd.concat(
        [valid_manifest, valid_manifest.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ManifestValidationError, match="Duplicate"):
        validate_manifest(duplicate)


def test_unsupported_language_fails(valid_manifest):
    invalid = valid_manifest.copy()
    invalid.loc[0, "lang"] = "bn"

    with pytest.raises(ManifestValidationError, match="Unsupported languages"):
        validate_manifest(invalid)


def test_invalid_condition_fails(valid_manifest):
    invalid = valid_manifest.copy()
    invalid.loc[0, "cond"] = "corrupted"

    with pytest.raises(ManifestValidationError, match="Unsupported conditions"):
        validate_manifest(invalid)


def test_nonpositive_duration_fails(valid_manifest):
    invalid = valid_manifest.copy()
    invalid.loc[0, "duration"] = 0

    with pytest.raises(ManifestValidationError, match="invalid durations"):
        validate_manifest(invalid)


def test_group_count_validation(valid_manifest):
    with pytest.raises(ManifestValidationError, match="Unexpected records"):
        validate_manifest(valid_manifest, expected_per_group=500)
