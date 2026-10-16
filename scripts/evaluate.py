
"""
Evaluate saved ASR predictions under matched normalization.

Expected input:
    CSV or Parquet containing:
        model, lang, cond, reference, hypothesis

Example:
    python scripts/evaluate.py \
        --input artifacts/predictions/all_predictions.parquet \
        --output-dir artifacts/metrics \
        --normalizers raw minimal whisper_style
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow execution as:
# python scripts/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    aggregate_metrics,
    evaluate_dataframe,
)


LOGGER = logging.getLogger("asr_evaluate")

REQUIRED_COLUMNS = {
    "model",
    "lang",
    "cond",
    "reference",
    "hypothesis",
}

SUPPORTED_NORMALIZERS = {
    "raw",
    "minimal",
    "whisper_style",
}


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet prediction file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        predictions = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        predictions = pd.read_parquet(path)
    else:
        raise ValueError(
            "Unsupported input format. Use .csv or .parquet."
        )

    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError("Prediction file contains no rows.")

    for column in ("model", "lang", "cond"):
        if predictions[column].isna().any():
            raise ValueError(
                f"Metadata column {column!r} contains missing values."
            )

    for column in ("reference", "hypothesis"):
        if predictions[column].isna().any():
            LOGGER.warning(
                "Column %s contains null values; treating them as "
                "empty strings for evaluation.",
                column,
            )
            predictions[column] = predictions[column].fillna("")

    return predictions


def evaluate_predictions(
    predictions: pd.DataFrame,
    normalizers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate utterance-level and aggregate metrics for each normalizer.

    Returns:
        utterance_results: One row per prediction per normalizer.
        aggregate_results: One row per model/language/condition/normalizer.
    """
    unknown = set(normalizers).difference(SUPPORTED_NORMALIZERS)
    if unknown:
        raise ValueError(
            f"Unsupported normalizers: {sorted(unknown)}"
        )

    utterance_results = []
    aggregate_results = []

    for normalizer in normalizers:
        LOGGER.info("Evaluating normalizer: %s", normalizer)

        evaluated = evaluate_dataframe(
            predictions,
            normalizer=normalizer,
            reference_column="reference",
            hypothesis_column="hypothesis",
        )

        evaluated = evaluated.copy()
        evaluated["normalizer"] = normalizer

        summary = aggregate_metrics(
            evaluated,
            group_columns=("model", "lang", "cond"),
        )
        summary["normalizer"] = normalizer

        utterance_results.append(evaluated)
        aggregate_results.append(summary)

    return (
        pd.concat(utterance_results, ignore_index=True),
        pd.concat(aggregate_results, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved ASR predictions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV or Parquet prediction file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/metrics",
        help="Directory in which evaluation CSVs are saved.",
    )
    parser.add_argument(
        "--normalizers",
        nargs="+",
        choices=sorted(SUPPORTED_NORMALIZERS),
        default=["raw", "minimal", "whisper_style"],
        help="Normalization methods to evaluate.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    predictions = load_predictions(args.input)

    LOGGER.info("Loaded %d predictions.", len(predictions))
    LOGGER.info(
        "Models: %s",
        sorted(predictions["model"].astype(str).unique()),
    )

    utterance_results, aggregate_results = evaluate_predictions(
        predictions,
        normalizers=args.normalizers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    utterance_path = output_dir / "utterance_metrics.csv"
    aggregate_path = output_dir / "model_comparison.csv"

    utterance_results.to_csv(utterance_path, index=False)
    aggregate_results.to_csv(aggregate_path, index=False)

    LOGGER.info("Saved utterance metrics to %s", utterance_path)
    LOGGER.info("Saved aggregate metrics to %s", aggregate_path)

    print("\nAggregate evaluation results:")
    display_columns = [
        column
        for column in (
            "model",
            "lang",
            "cond",
            "normalizer",
            "wer",
            "cer",
        )
        if column in aggregate_results.columns
    ]
    print(
        aggregate_results[display_columns]
        .sort_values(["lang", "cond", "model", "normalizer"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
