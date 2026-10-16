
"""
Error analysis for multilingual ASR predictions.

Features:
- Per-utterance edit counts
- Corpus-level error summaries
- Error breakdown by model, language, and condition
- Worst-performing utterances
- Error examples for manual inspection

All rates are calculated from aggregated counts rather than averaging
utterance-level error rates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.evaluation.metrics import evaluate_dataframe, aggregate_metrics


LOGGER = logging.getLogger(__name__)

DEFAULT_GROUP_COLUMNS = ("model", "lang", "cond")


def validate_prediction_schema(
    predictions: pd.DataFrame,
    reference_column: str = "reference",
    hypothesis_column: str = "hypothesis",
) -> None:
    """Validate that the prediction table contains required columns."""
    required = {reference_column, hypothesis_column}
    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction table is missing columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError("Prediction table is empty")


def analyze_predictions(
    predictions: pd.DataFrame,
    normalizer: str = "minimal",
    reference_column: str = "reference",
    hypothesis_column: str = "hypothesis",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate predictions and produce utterance-level and grouped results.

    Returns:
        utterance_metrics: Original records with computed error metrics.
        group_summary: Corpus-level WER/CER and edit counts per group.
    """
    validate_prediction_schema(
        predictions,
        reference_column=reference_column,
        hypothesis_column=hypothesis_column,
    )

    missing_groups = set(group_columns).difference(predictions.columns)
    if missing_groups:
        raise ValueError(
            f"Missing grouping columns: {sorted(missing_groups)}"
        )

    utterance_metrics = evaluate_dataframe(
        predictions,
        normalizer=normalizer,
        reference_column=reference_column,
        hypothesis_column=hypothesis_column,
    )

    group_summary = aggregate_metrics(
        utterance_metrics,
        group_columns=group_columns,
    )

    return utterance_metrics, group_summary


def summarize_error_types(
    evaluated: pd.DataFrame,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    """
    Summarize the proportion of word errors attributable to S, D, and I.

    Error proportions sum to 1 when the group has at least one error.
    They are set to zero when a group has no errors.
    """
    required = set(group_columns) | {
        "substitutions",
        "deletions",
        "insertions",
    }

    missing = required.difference(evaluated.columns)
    if missing:
        raise ValueError(
            f"Missing columns for error-type analysis: {sorted(missing)}"
        )

    summary = (
        evaluated.groupby(
            list(group_columns),
            dropna=False,
            sort=True,
        )[["substitutions", "deletions", "insertions"]]
        .sum()
        .reset_index()
    )

    summary["total_word_errors"] = (
        summary["substitutions"]
        + summary["deletions"]
        + summary["insertions"]
    )

    for error_type in ("substitutions", "deletions", "insertions"):
        summary[f"{error_type}_share"] = (
            summary[error_type]
            .div(summary["total_word_errors"].replace(0, pd.NA))
            .fillna(0.0)
        )

    return summary


def get_worst_utterances(
    evaluated: pd.DataFrame,
    top_k: int = 20,
    group_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return the utterances with the highest WER.

    If group_columns are supplied, return up to top_k utterances per group.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    required = {"wer", "reference", "hypothesis"}
    missing = required.difference(evaluated.columns)

    if missing:
        raise ValueError(
            f"Missing columns for worst-utterance analysis: {sorted(missing)}"
        )

    if group_columns:
        missing_groups = set(group_columns).difference(evaluated.columns)
        if missing_groups:
            raise ValueError(
                f"Missing grouping columns: {sorted(missing_groups)}"
            )

        result = (
            evaluated.sort_values(
                "wer",
                ascending=False,
                kind="stable",
            )
            .groupby(
                list(group_columns),
                dropna=False,
                sort=False,
                group_keys=False,
            )
            .head(top_k)
        )
    else:
        result = evaluated.sort_values(
            "wer",
            ascending=False,
            kind="stable",
        ).head(top_k)

    return result.reset_index(drop=True)


def export_error_analysis(
    utterance_metrics: pd.DataFrame,
    group_summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save error-analysis tables to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "utterance_metrics": output_dir / "utterance_metrics.csv",
        "group_summary": output_dir / "error_summary_by_group.csv",
        "error_types": output_dir / "error_type_breakdown.csv",
        "worst_utterances": output_dir / "worst_utterances.csv",
    }

    error_types = summarize_error_types(utterance_metrics)
    worst = get_worst_utterances(utterance_metrics, top_k=20)

    utterance_metrics.to_csv(paths["utterance_metrics"], index=False)
    group_summary.to_csv(paths["group_summary"], index=False)
    error_types.to_csv(paths["error_types"], index=False)
    worst.to_csv(paths["worst_utterances"], index=False)

    LOGGER.info("Exported error-analysis results to %s", output_dir)

    return paths
