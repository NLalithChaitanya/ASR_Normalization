
"""
Visualization utilities for multilingual ASR evaluation.

Expected metric columns:
    model, lang, cond, wer, cer

WER and CER may be fractions (e.g. 0.1086) or percentages
(e.g. 10.86). Set rates_are_percent accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANGUAGE_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
}

CONDITION_ORDER = ["clean", "noisy"]


def _prepare_metrics(
    metrics: pd.DataFrame,
    rates_are_percent: bool = False,
) -> pd.DataFrame:
    """Validate and standardize WER/CER columns to percentages."""
    required = {"model", "lang", "cond", "wer", "cer"}
    missing = required.difference(metrics.columns)

    if missing:
        raise ValueError(
            f"Metrics dataframe is missing columns: {sorted(missing)}"
        )

    data = metrics.copy()

    if not rates_are_percent:
        data["wer"] = data["wer"] * 100.0
        data["cer"] = data["cer"] * 100.0

    if not set(data["lang"].dropna().unique()).issubset(LANGUAGE_NAMES):
        raise ValueError("Unexpected language code in metrics.")

    if not set(data["cond"].dropna().unique()).issubset(
        CONDITION_ORDER
    ):
        raise ValueError("Conditions must be 'clean' or 'noisy'.")

    for column in ("wer", "cer"):
        data[column] = pd.to_numeric(data[column], errors="raise")

        if not np.isfinite(data[column]).all():
            raise ValueError(f"Column {column!r} contains non-finite values.")

        if (data[column] < 0).any():
            raise ValueError(f"Column {column!r} contains negative values.")

    return data


def plot_metric_comparison(
    metrics: pd.DataFrame,
    language: str,
    metric: str = "wer",
    output_path: str | Path | None = None,
    rates_are_percent: bool = False,
) -> None:
    """
    Compare models on clean and noisy audio for one language.

    Args:
        metrics: Aggregated model metrics.
        language: One of 'hi', 'ta', or 'te'.
        metric: 'wer' or 'cer'.
        output_path: Optional path to save the figure.
        rates_are_percent: True if metric values are already percentages.
    """
    if metric not in {"wer", "cer"}:
        raise ValueError("metric must be 'wer' or 'cer'.")

    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language!r}")

    data = _prepare_metrics(metrics, rates_are_percent)
    data = data.loc[data["lang"] == language]

    if data.empty:
        raise ValueError(f"No metrics found for language {language!r}.")

    pivot = data.pivot_table(
        index="model",
        columns="cond",
        values=metric,
        aggfunc="mean",
    )

    models = pivot.index.tolist()
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    for offset, condition in zip((-width / 2, width / 2), CONDITION_ORDER):
        if condition not in pivot.columns:
            continue

        values = pivot[condition].to_numpy()
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=condition.capitalize(),
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_title(
        f"{metric.upper()} by ASR model — {LANGUAGE_NAMES[language]}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)
    else:
        plt.show()

    plt.close(fig)


def plot_noise_degradation(
    metrics: pd.DataFrame,
    metric: str = "wer",
    output_path: str | Path | None = None,
    rates_are_percent: bool = False,
) -> None:
    """
    Plot noisy-minus-clean degradation in percentage points.

    Positive values indicate a higher error rate under noisy conditions.
    Only models with both conditions are included.
    """
    if metric not in {"wer", "cer"}:
        raise ValueError("metric must be 'wer' or 'cer'.")

    data = _prepare_metrics(metrics, rates_are_percent)

    pivot = data.pivot_table(
        index=["lang", "model"],
        columns="cond",
        values=metric,
        aggfunc="mean",
    )

    if not {"clean", "noisy"}.issubset(pivot.columns):
        raise ValueError(
            "Both clean and noisy metrics are required."
        )

    degradation = (
        pivot["noisy"] - pivot["clean"]
    ).dropna().rename("degradation").reset_index()

    if degradation.empty:
        raise ValueError("No matched clean/noisy model groups found.")

    languages = [
        lang for lang in LANGUAGE_NAMES
        if lang in degradation["lang"].unique()
    ]

    fig, axes = plt.subplots(
        1,
        len(languages),
        figsize=(6 * len(languages), 5),
        squeeze=False,
    )

    for ax, language in zip(axes[0], languages):
        subset = degradation.loc[
            degradation["lang"] == language
        ].sort_values("degradation")

        bars = ax.barh(
            subset["model"],
            subset["degradation"],
        )
        ax.axvline(0, linewidth=1)
        ax.set_title(LANGUAGE_NAMES[language])
        ax.set_xlabel(f"Noisy − clean {metric.upper()} (percentage points)")
        ax.grid(axis="x", alpha=0.25)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    fig.suptitle(f"Noise-associated {metric.upper()} degradation")
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)
    else:
        plt.show()

    plt.close(fig)


def plot_error_distribution(
    error_summary: pd.DataFrame,
    language: str,
    condition: str,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot substitution, deletion, and insertion counts by model.

    Expected columns:
        model, lang, cond, substitutions, deletions, insertions
    """
    required = {
        "model",
        "lang",
        "cond",
        "substitutions",
        "deletions",
        "insertions",
    }
    missing = required.difference(error_summary.columns)

    if missing:
        raise ValueError(
            f"Error summary is missing columns: {sorted(missing)}"
        )

    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language!r}")

    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unsupported condition: {condition!r}")

    subset = error_summary.loc[
        (error_summary["lang"] == language)
        & (error_summary["cond"] == condition)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No error summaries for {language}/{condition}."
        )

    subset = subset.sort_values("model")

    x = np.arange(len(subset))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for offset, column, label in (
        (-width, "substitutions", "Substitutions"),
        (0, "deletions", "Deletions"),
        (width, "insertions", "Insertions"),
    ):
        values = pd.to_numeric(
            subset[column], errors="raise"
        ).to_numpy()

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
        )
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subset["model"], rotation=15, ha="right")
    ax.set_ylabel("Error count")
    ax.set_title(
        f"Error types — {LANGUAGE_NAMES[language]}, {condition}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        _save_figure(fig, output_path)
    else:
        plt.show()

    plt.close(fig)


def _save_figure(fig, output_path: str | Path) -> None:
    """Create parent directories and save a high-resolution figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
