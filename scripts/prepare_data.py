
"""
Validate and prepare the ASR dataset manifest.

Examples:
    python scripts/prepare_data.py \
        --input data/manifest.parquet \
        --output data/manifest_validated.parquet \
        --audio-root data/audio \
        --expected-per-group 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_dataset import (  # noqa: E402
    load_manifest,
    prepare_dataset,
    validate_manifest,
)


LOGGER = logging.getLogger("asr_prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the ASR dataset manifest."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input manifest in CSV or Parquet format.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the validated manifest.",
    )
    parser.add_argument(
        "--audio-root",
        default=None,
        help=(
            "Optional root directory for resolving relative audio paths. "
            "When supplied, all referenced audio files must exist."
        ),
    )
    parser.add_argument(
        "--expected-per-group",
        type=int,
        default=None,
        help=(
            "Optional expected number of rows for every language/condition "
            "group, for example 500."
        ),
    )
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Verify that every referenced audio file exists.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.expected_per_group is not None and args.expected_per_group < 1:
        parser.error("--expected-per-group must be positive.")

    if args.check_audio and not args.audio_root:
        parser.error("--check-audio requires --audio-root.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    LOGGER.info("Loading manifest: %s", input_path)
    manifest = load_manifest(input_path)

    LOGGER.info("Loaded %d manifest rows.", len(manifest))

    # Resolve relative paths before validation if an audio root is supplied.
    if args.audio_root:
        audio_root = Path(args.audio_root).resolve()

        def resolve_audio_path(value: object) -> object:
            if pd.isna(value):
                return value

            path = Path(str(value))
            if path.is_absolute():
                return str(path)

            return str((audio_root / path).resolve())

        manifest = manifest.copy()
        manifest["audio_path"] = manifest["audio_path"].map(
            resolve_audio_path
        )

    validated = prepare_dataset(
        manifest,
        output_path=output_path,
        expected_per_group=args.expected_per_group,
        check_audio_exists=args.check_audio,
    )

    print("\nManifest validation successful.")
    print(f"Rows: {len(validated)}")
    print(f"Unique utterance IDs: {validated['utt_id'].nunique()}")

    print("\nRows by language and condition:")
    print(
        validated.groupby(["lang", "cond"])
        .size()
        .rename("rows")
        .to_string()
    )

    print(f"\nValidated manifest saved to: {output_path}")


if __name__ == "__main__":
    main()
