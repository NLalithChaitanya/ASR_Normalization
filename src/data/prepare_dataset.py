
"""
Dataset preparation and validation for the ASR normalization project.

Expected manifest columns:
    utt_id, lang, cond, audio_path, reference, duration, gender,
    snr_db, noise_file, noise_cat, pair_id

The module validates metadata without modifying audio files or silently
dropping invalid records.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "utt_id",
    "lang",
    "cond",
    "audio_path",
    "reference",
    "duration",
    "gender",
    "snr_db",
    "noise_file",
    "noise_cat",
    "pair_id",
}

SUPPORTED_LANGUAGES = {"hi", "ta", "te"}
SUPPORTED_CONDITIONS = {"clean", "noisy"}


class ManifestValidationError(ValueError):
    """Raised when the dataset manifest fails validation."""


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or Parquet manifest based on its file extension.
    """
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        dataframe = pd.read_parquet(path)
    elif suffix == ".csv":
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported manifest format '{suffix}'. "
            "Expected .parquet or .csv."
        )

    LOGGER.info(
        "Loaded manifest: %s rows, %s columns",
        len(dataframe),
        len(dataframe.columns),
    )

    return dataframe


def validate_manifest(
    manifest: pd.DataFrame,
    check_audio_paths: bool = False,
    audio_root: Optional[str | Path] = None,
    expected_per_group: Optional[int] = None,
) -> pd.DataFrame:
    """
    Validate manifest structure and values.

    Args:
        manifest: Dataset metadata.
        check_audio_paths: Whether to check that audio files exist.
        audio_root: Optional root used to resolve relative audio paths.
        expected_per_group: Optional expected record count per language
            and condition. Leave None when the dataset is not balanced.

    Returns:
        A validated copy of the original manifest.

    Raises:
        ManifestValidationError: If validation fails.
    """
    if not isinstance(manifest, pd.DataFrame):
        raise TypeError("manifest must be a pandas DataFrame")

    dataframe = manifest.copy()

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ManifestValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ManifestValidationError("Manifest contains no records")

    # Required metadata must be present.
    required_non_null = [
        "utt_id",
        "lang",
        "cond",
        "audio_path",
        "reference",
        "duration",
        "pair_id",
    ]

    for column in required_non_null:
        missing_count = int(dataframe[column].isna().sum())
        if missing_count:
            raise ManifestValidationError(
                f"Column '{column}' contains {missing_count} null values"
            )

    # Identifiers should be strings and unique for each language-condition
    # record. A paired clean/noisy utterance may share pair_id.
    for column in ("utt_id", "lang", "cond", "audio_path", "pair_id"):
        dataframe[column] = dataframe[column].astype(str).str.strip()

        if dataframe[column].eq("").any():
            raise ManifestValidationError(
                f"Column '{column}' contains empty strings"
            )

    duplicate_keys = dataframe.duplicated(
        subset=["utt_id", "lang", "cond"],
        keep=False,
    )

    if duplicate_keys.any():
        examples = dataframe.loc[
            duplicate_keys, ["utt_id", "lang", "cond"]
        ].head(10)

        raise ManifestValidationError(
            "Duplicate utterance-language-condition records found. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    invalid_languages = (
        set(dataframe["lang"].unique()) - SUPPORTED_LANGUAGES
    )

    if invalid_languages:
        raise ManifestValidationError(
            f"Unsupported languages: {sorted(invalid_languages)}"
        )

    invalid_conditions = (
        set(dataframe["cond"].unique()) - SUPPORTED_CONDITIONS
    )

    if invalid_conditions:
        raise ManifestValidationError(
            f"Unsupported conditions: {sorted(invalid_conditions)}"
        )

    # References may legitimately be empty in some datasets, so flag them
    # rather than silently removing the associated records.
    empty_references = dataframe["reference"].fillna("").astype(str).str.strip().eq("")

    if empty_references.any():
        LOGGER.warning(
            "%d records have empty references",
            int(empty_references.sum()),
        )

    dataframe["reference"] = dataframe["reference"].fillna("").astype(str)

    dataframe["duration"] = pd.to_numeric(
        dataframe["duration"],
        errors="coerce",
    )

    invalid_duration = (
        dataframe["duration"].isna()
        | ~dataframe["duration"].map(
            lambda value: value == value and abs(value) != float("inf")
        )
        | (dataframe["duration"] <= 0)
    )

    if invalid_duration.any():
        raise ManifestValidationError(
            f"{int(invalid_duration.sum())} records have invalid durations"
        )

    if "snr_db" in dataframe.columns:
        dataframe["snr_db"] = pd.to_numeric(
            dataframe["snr_db"],
            errors="coerce",
        )

        # SNR can be missing for clean recordings; don't require it globally.

    if expected_per_group is not None:
        if expected_per_group <= 0:
            raise ValueError("expected_per_group must be positive")

        counts = dataframe.groupby(["lang", "cond"]).size()

        incorrect = counts[counts != expected_per_group]

        if not incorrect.empty:
            raise ManifestValidationError(
                "Unexpected records per language-condition group. "
                f"Expected {expected_per_group}; found:\n"
                f"{incorrect.to_string()}"
            )

    if check_audio_paths:
        root = Path(audio_root) if audio_root is not None else None
        missing_paths = []

        for audio_path in dataframe["audio_path"]:
            candidate = Path(audio_path)

            if not candidate.is_absolute() and root is not None:
                candidate = root / candidate

            if not candidate.is_file():
                missing_paths.append(str(candidate))

        if missing_paths:
            examples = "\n".join(missing_paths[:10])
            raise ManifestValidationError(
                f"{len(missing_paths)} audio files were not found. "
                f"First missing paths:\n{examples}"
            )

    dataframe = dataframe.sort_values(
        ["lang", "cond", "utt_id"],
        kind="stable",
    ).reset_index(drop=True)

    LOGGER.info(
        "Manifest validation passed: %d records",
        len(dataframe),
    )

    return dataframe


def save_manifest(
    manifest: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save a validated manifest to CSV or Parquet.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        manifest.to_parquet(path, index=False)
    elif suffix == ".csv":
        manifest.to_csv(path, index=False)
    else:
        raise ValueError(
            f"Unsupported output format '{suffix}'. "
            "Expected .parquet or .csv."
        )

    LOGGER.info("Saved manifest to %s", path)

    return path


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    check_audio_paths: bool = False,
    audio_root: Optional[str | Path] = None,
    expected_per_group: Optional[int] = None,
) -> pd.DataFrame:
    """Load, validate, and save a dataset manifest."""
    manifest = load_manifest(input_path)

    validated = validate_manifest(
        manifest,
        check_audio_paths=check_audio_paths,
        audio_root=audio_root,
        expected_per_group=expected_per_group,
    )

    save_manifest(validated, output_path)

    return validated


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Validate and prepare the ASR dataset manifest."
    )

    parser.add_argument(
        "--input",
        default="data/manifest.parquet",
        help="Input CSV or Parquet manifest.",
    )
    parser.add_argument(
        "--output",
        default="data/manifest_validated.parquet",
        help="Output CSV or Parquet manifest.",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Root directory for relative audio paths.",
    )
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Check that all referenced audio files exist.",
    )
    parser.add_argument(
        "--expected-per-group",
        type=int,
        default=None,
        help="Optional expected record count per language-condition group.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        prepared = prepare_dataset(
            input_path=args.input,
            output_path=args.output,
            check_audio_paths=args.check_audio,
            audio_root=args.audio_root,
            expected_per_group=args.expected_per_group,
        )
    except (OSError, ValueError, TypeError) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 1

    print("\nRecords per language and condition:")
    print(prepared.groupby(["lang", "cond"]).size().to_string())

    print(f"\nValidated manifest saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
