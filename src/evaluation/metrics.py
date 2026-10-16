
"""
Evaluation metrics for multilingual automatic speech recognition (ASR).

Provides:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Substitution, deletion, and insertion counts
- Per-utterance and grouped evaluation
- Validation of reference/hypothesis records

WER = (S + D + I) / N
CER = (S_char + D_char + I_char) / N_char

Reference and hypothesis text must use the same normalization method.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from src.evaluation.normalization import normalize_pair


@dataclass(frozen=True)
class EditCounts:
    """Edit operation counts for a reference-hypothesis pair."""

    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    hypothesis_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        """Return an error rate as a fraction, not a percentage."""
        if self.reference_units == 0:
            return 0.0 if self.hypothesis_units == 0 else float("inf")

        return self.errors / self.reference_units


def _edit_counts(
    reference_units: Sequence[str],
    hypothesis_units: Sequence[str],
) -> EditCounts:
    """
    Compute Levenshtein edit counts using dynamic programming.

    Ties are resolved deterministically in this order:
    correct match, substitution, deletion, insertion.
    """
    n = len(reference_units)
    m = len(hypothesis_units)

    # Each cell stores (edit distance, substitutions, deletions, insertions).
    dp: List[List[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)]
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)

    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference_units[i - 1] == hypothesis_units[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            previous = dp[i - 1][j - 1]
            substitution = (
                previous[0] + 1,
                previous[1] + 1,
                previous[2],
                previous[3],
            )

            previous = dp[i - 1][j]
            deletion = (
                previous[0] + 1,
                previous[1],
                previous[2] + 1,
                previous[3],
            )

            previous = dp[i][j - 1]
            insertion = (
                previous[0] + 1,
                previous[1],
                previous[2],
                previous[3] + 1,
            )

            dp[i][j] = min(
                (substitution, deletion, insertion),
                key=lambda item: item[0],
            )

    _, substitutions, deletions, insertions = dp[n][m]

    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=n,
        hypothesis_units=m,
    )


def word_error_counts(reference: str, hypothesis: str) -> EditCounts:
    """Compute word-level edit counts."""
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    return _edit_counts(reference_tokens, hypothesis_tokens)


def character_error_counts(reference: str, hypothesis: str) -> EditCounts:
    """Compute character-level edit counts, excluding spaces."""
    reference_chars = list("".join(reference.split()))
    hypothesis_chars = list("".join(hypothesis.split()))

    return _edit_counts(reference_chars, hypothesis_chars)


def _safe_rate(numerator: int, denominator: int) -> float:
    """
    Return a rate as a fraction.

    An empty reference and empty hypothesis have rate zero.
    A nonempty hypothesis against an empty reference has infinite rate.
    """
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")

    return numerator / denominator


def evaluate_utterance(
    reference: Optional[str],
    hypothesis: Optional[str],
    normalizer: str = "minimal",
) -> Dict[str, Any]:
    """
    Evaluate a single utterance using a shared normalization method.

    WER and CER are returned as fractions. Multiply by 100 for percentages.
    """
    reference = "" if reference is None else str(reference)
    hypothesis = "" if hypothesis is None else str(hypothesis)

    normalized_reference, normalized_hypothesis = normalize_pair(
        reference,
        hypothesis,
        method=normalizer,
    )

    word_counts = word_error_counts(
        normalized_reference,
        normalized_hypothesis,
    )
    char_counts = character_error_counts(
        normalized_reference,
        normalized_hypothesis,
    )

    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "normalizer": normalizer,
        "wer": _safe_rate(
            word_counts.errors,
            word_counts.reference_units,
        ),
        "cer": _safe_rate(
            char_counts.errors,
            char_counts.reference_units,
        ),
        "substitutions": word_counts.substitutions,
        "deletions": word_counts.deletions,
        "insertions": word_counts.insertions,
        "reference_words": word_counts.reference_units,
        "hypothesis_words": word_counts.hypothesis_units,
        "character_substitutions": char_counts.substitutions,
        "character_deletions": char_counts.deletions,
        "character_insertions": char_counts.insertions,
        "reference_characters": char_counts.reference_units,
        "hypothesis_characters": char_counts.hypothesis_units,
    }


def evaluate_dataframe(
    predictions: pd.DataFrame,
    normalizer: str = "minimal",
    reference_column: str = "reference",
    hypothesis_column: str = "hypothesis",
) -> pd.DataFrame:
    """
    Evaluate a prediction DataFrame row by row.

    The input must contain reference and hypothesis columns.
    Returns a copy with per-utterance metrics appended.
    """
    required = {reference_column, hypothesis_column}
    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing required prediction columns: {sorted(missing)}"
        )

    result = predictions.copy()

    evaluated = [
        evaluate_utterance(reference, hypothesis, normalizer)
        for reference, hypothesis in zip(
            result[reference_column],
            result[hypothesis_column],
        )
    ]

    metrics = pd.DataFrame(evaluated, index=result.index)

    # Avoid duplicate column names if the input already contains metrics.
    metrics = metrics.drop(
        columns=[
            column
            for column in metrics.columns
            if column in result.columns
        ]
    )

    return pd.concat([result, metrics], axis=1)


def aggregate_metrics(
    evaluated: pd.DataFrame,
    group_columns: Sequence[str] = ("model", "lang", "cond"),
) -> pd.DataFrame:
    """
    Aggregate utterance-level edit counts into corpus-level WER and CER.

    This computes micro-averaged corpus metrics:
        WER = total word errors / total reference words
        CER = total character errors / total reference characters

    It does not average utterance-level WER/CER values.
    """
    required = set(group_columns) | {
        "substitutions",
        "deletions",
        "insertions",
        "reference_words",
        "character_substitutions",
        "character_deletions",
        "character_insertions",
        "reference_characters",
    }

    missing = required.difference(evaluated.columns)
    if missing:
        raise ValueError(
            f"Missing columns required for aggregation: {sorted(missing)}"
        )

    grouped = evaluated.groupby(
        list(group_columns),
        dropna=False,
        sort=True,
    )

    result = grouped.agg(
        n_utterances=("reference_words", "size"),
        substitutions=("substitutions", "sum"),
        deletions=("deletions", "sum"),
        insertions=("insertions", "sum"),
        reference_words=("reference_words", "sum"),
        character_substitutions=("character_substitutions", "sum"),
        character_deletions=("character_deletions", "sum"),
        character_insertions=("character_insertions", "sum"),
        reference_characters=("reference_characters", "sum"),
    ).reset_index()

    result["word_errors"] = (
        result["substitutions"]
        + result["deletions"]
        + result["insertions"]
    )

    result["character_errors"] = (
        result["character_substitutions"]
        + result["character_deletions"]
        + result["character_insertions"]
    )

    result["wer"] = [
        _safe_rate(errors, count)
        for errors, count in zip(
            result["word_errors"],
            result["reference_words"],
        )
    ]

    result["cer"] = [
        _safe_rate(errors, count)
        for errors, count in zip(
            result["character_errors"],
            result["reference_characters"],
        )
    ]

    result["wer_percent"] = result["wer"] * 100.0
    result["cer_percent"] = result["cer"] * 100.0

    return result
