
"""
Normalization audit for multilingual ASR evaluation.

Measures:
- How much normalized reference text differs from the raw reference.
- Changes in reference character and token counts.
- Combining-mark retention for Indic scripts.
- WER/CER under matched normalization settings.

Important:
WER/CER comparisons should use the SAME normalizer for references
and hypotheses. Cross-normalization results are diagnostic only.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.evaluation.normalization import (
    NORMALIZERS,
    normalize_text,
)
from src.evaluation.metrics import evaluate_dataframe


DEFAULT_GROUP_COLUMNS = ("model", "lang", "cond")


def _combining_mark_count(text: str) -> int:
    """Count Unicode characters categorized as combining marks."""
    return sum(
        unicodedata.category(char).startswith("M")
        for char in text
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio, using zero when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def audit_text_changes(
    references: pd.Series,
    normalizer: str,
) -> pd.DataFrame:
    """
    Audit how a normalizer transforms reference text.

    Returns one record per reference, including character counts,
    whitespace-token counts, combining marks, and a changed-text flag.
    """
    if normalizer not in NORMALIZERS:
        raise ValueError(
            f"Unknown normalizer {normalizer!r}. "
            f"Choose from {list(NORMALIZERS)}."
        )

    records = []

    for index, value in references.items():
        raw = "" if pd.isna(value) else str(value)
        normalized = normalize_text(raw, normalizer)

        records.append(
            {
                "index": index,
                "normalizer": normalizer,
                "raw_reference": raw,
                "normalized_reference": normalized,
                "text_changed": raw != normalized,
                "raw_characters": len(raw),
                "normalized_characters": len(normalized),
                "raw_tokens": len(raw.split()),
                "normalized_tokens": len(normalized.split()),
                "raw_combining_marks": _combining_mark_count(raw),
                "normalized_combining_marks": _combining_mark_count(
                    normalized
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_text_changes(
    audit: pd.DataFrame,
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate text-change statistics globally or by supplied groups."""
    required = {
        "normalizer",
        "text_changed",
        "raw_characters",
        "normalized_characters",
        "raw_tokens",
        "normalized_tokens",
        "raw_combining_marks",
        "normalized_combining_marks",
    }

    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(
            f"Missing audit columns: {sorted(missing)}"
        )

    missing_groups = set(group_columns).difference(audit.columns)
    if missing_groups:
        raise ValueError(
            f"Missing grouping columns: {sorted(missing_groups)}"
        )

    keys = list(group_columns) + ["normalizer"]

    summary = (
        audit.groupby(keys, dropna=False, sort=True)
        .agg(
            utterances=("text_changed", "size"),
            changed_utterances=("text_changed", "sum"),
            raw_characters=("raw_characters", "sum"),
            normalized_characters=("normalized_characters", "sum"),
            raw_tokens=("raw_tokens", "sum"),
            normalized_tokens=("normalized_tokens", "sum"),
            raw_combining_marks=("raw_combining_marks", "sum"),
            normalized_combining_marks=(
                "normalized_combining_marks",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["changed_utterance_percent"] = (
        100.0
        * summary["changed_utterances"]
        / summary["utterances"].replace(0, pd.NA)
    ).fillna(0.0)

    summary["token_count_ratio"] = (
        summary["normalized_tokens"]
        / summary["raw_tokens"].replace(0, pd.NA)
    ).fillna(0.0)

    summary["character_count_ratio"] = (
        summary["normalized_characters"]
        / summary["raw_characters"].replace(0, pd.NA)
    ).fillna(0.0)

    summary["combining_mark_retention"] = [
        _safe_ratio(int(normalized), int(raw))
        for raw, normalized in zip(
            summary["raw_combining_marks"],
            summary["normalized_combining_marks"],
        )
    ]

    return summary


def compare_normalization_metrics(
    predictions: pd.DataFrame,
    normalizers: Sequence[str] = (
        "raw",
        "minimal",
        "whisper_style",
    ),
    reference_column: str = "reference",
    hypothesis_column: str = "hypothesis",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    """
    Evaluate predictions separately under each matched normalizer.

    Each normalizer is applied to both reference and hypothesis.
    WER/CER are returned as percentages when the underlying metrics
    use fractional rates.
    """
    required = {reference_column, hypothesis_column, *group_columns}
    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction table is missing columns: {sorted(missing)}"
        )

    unknown = set(normalizers).difference(NORMALIZERS)
    if unknown:
        raise ValueError(
            f"Unknown normalizers: {sorted(unknown)}"
        )

    summaries = []

    for normalizer in normalizers:
        evaluated = evaluate_dataframe(
            predictions,
            normalizer=normalizer,
            reference_column=reference_column,
            hypothesis_column=hypothesis_column,
        )

        from src.evaluation.metrics import aggregate_metrics

        summary = aggregate_metrics(
            evaluated,
            group_columns=group_columns,
        )

        summary["normalizer"] = normalizer

        if "wer" in summary.columns:
            summary["wer_percent"] = 100.0 * summary["wer"]

        if "cer" in summary.columns:
            summary["cer_percent"] = 100.0 * summary["cer"]

        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


def run_normalization_audit(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    reference_column: str = "reference",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> dict[str, Path]:
    """Run text-change and matched-metric audits and save CSV reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_audits = []
    for normalizer in NORMALIZERS:
        audit = audit_text_changes(
            predictions[reference_column],
            normalizer,
        )

        # Retain metadata for grouping reference changes by language
        # and condition when those columns exist.
        audit["source_index"] = predictions.index.to_numpy()

        for column in group_columns:
            if column in predictions.columns:
                audit[column] = predictions[column].to_numpy()

        text_audits.append(audit)

    all_text_audits = pd.concat(text_audits, ignore_index=True)

    available_groups = [
        column for column in group_columns
        if column in all_text_audits.columns
    ]

    text_summary = summarize_text_changes(
        all_text_audits,
        group_columns=available_groups,
    )

    metric_summary = compare_normalization_metrics(
        predictions,
        group_columns=group_columns,
    )

    paths = {
        "text_audit": output_dir / "normalization_text_audit.csv",
        "text_summary": output_dir / "normalization_text_summary.csv",
        "metric_comparison": output_dir / "normalization_metric_comparison.csv",
    }

    all_text_audits.to_csv(paths["text_audit"], index=False)
    text_summary.to_csv(paths["text_summary"], index=False)
    metric_summary.to_csv(paths["metric_comparison"], index=False)

    return paths
