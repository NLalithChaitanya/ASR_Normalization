
"""
Telugu ASR failure audit.

Classifies output-script patterns and identifies potentially degenerate
hypotheses. These are diagnostic flags, not definitive explanations
of why an ASR model failed.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd


SUPPORTED_LANGUAGES = {"hi", "ta", "te"}


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_mark(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def _script_of_char(char: str) -> str | None:
    """Return a coarse script label for a Unicode character."""
    if not (_is_letter(char) or _is_mark(char)):
        return None

    codepoint = ord(char)

    if 0x0C00 <= codepoint <= 0x0C7F:
        return "telugu"
    if 0x0900 <= codepoint <= 0x097F:
        return "devanagari"
    if 0x0B80 <= codepoint <= 0x0BFF:
        return "tamil"
    if 0x0A00 <= codepoint <= 0x0A7F:
        return "gurmukhi"
    if 0x0A80 <= codepoint <= 0x0AFF:
        return "gujarati"
    if 0x0B00 <= codepoint <= 0x0B7F:
        return "odia"
    if 0x0980 <= codepoint <= 0x09FF:
        return "bengali"
    if 0x0600 <= codepoint <= 0x06FF:
        return "arabic"
    if 0x0041 <= codepoint <= 0x005A or 0x0061 <= codepoint <= 0x007A:
        return "latin"

    return "other"


def classify_script(text: str) -> str:
    """
    Classify a hypothesis by the scripts represented among its letters.

    Possible labels:
    - no_letters
    - telugu, devanagari, tamil, latin, etc.
    - mixed
    - other
    """
    scripts = {
        script
        for char in str(text)
        if (script := _script_of_char(char)) is not None
    }

    if not scripts:
        return "no_letters"

    if len(scripts) > 1:
        return "mixed"

    return next(iter(scripts))


def inspect_hypothesis(text: str) -> dict:
    """Calculate text-quality indicators for one ASR hypothesis."""
    text = "" if pd.isna(text) else str(text)

    nonspace = [char for char in text if not char.isspace()]
    letters = [char for char in text if _is_letter(char)]
    marks = [char for char in text if _is_mark(char)]

    return {
        "script_class": classify_script(text),
        "is_empty": not text.strip(),
        "character_count": len(text),
        "nonspace_character_count": len(nonspace),
        "letter_count": len(letters),
        "combining_mark_count": len(marks),
        "token_count": len(text.split()),
        "has_letters": bool(letters),
        "symbol_only": bool(nonspace) and not bool(letters),
    }


def audit_predictions(
    predictions: pd.DataFrame,
    reference_column: str = "reference",
    hypothesis_column: str = "hypothesis",
    language_column: str = "lang",
    model_column: str = "model",
    condition_column: str = "cond",
    target_language: str = "te",
) -> pd.DataFrame:
    """
    Add diagnostic indicators to Telugu predictions.

    The function filters to the target language, then retains all
    original prediction columns and appends hypothesis-quality fields.
    """
    required = {
        reference_column,
        hypothesis_column,
        language_column,
        model_column,
        condition_column,
    }
    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction table is missing columns: {sorted(missing)}"
        )

    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {target_language!r}; "
            f"choose from {sorted(SUPPORTED_LANGUAGES)}."
        )

    result = predictions.loc[
        predictions[language_column] == target_language
    ].copy()

    if result.empty:
        raise ValueError(
            f"No predictions found for language {target_language!r}."
        )

    indicators = result[hypothesis_column].apply(inspect_hypothesis)
    indicators = pd.DataFrame(
        indicators.tolist(),
        index=result.index,
    )

    result = pd.concat([result, indicators], axis=1)

    result["reference_script_class"] = result[
        reference_column
    ].apply(classify_script)

    result["script_mismatch"] = (
        result["script_class"] != result["reference_script_class"]
    )

    return result


def summarize_failures(
    audited: pd.DataFrame,
    group_columns: tuple[str, ...] = ("model", "lang", "cond"),
) -> pd.DataFrame:
    """Summarize script patterns and potential output-quality issues."""
    required = {
        *group_columns,
        "script_class",
        "is_empty",
        "symbol_only",
        "has_letters",
        "script_mismatch",
        "character_count",
        "nonspace_character_count",
        "token_count",
    }
    missing = required.difference(audited.columns)

    if missing:
        raise ValueError(
            f"Audited table is missing columns: {sorted(missing)}"
        )

    summary = (
        audited.groupby(
            list(group_columns) + ["script_class"],
            dropna=False,
            sort=True,
        )
        .agg(
            utterances=("script_class", "size"),
            empty_outputs=("is_empty", "sum"),
            symbol_only_outputs=("symbol_only", "sum"),
            outputs_with_letters=("has_letters", "sum"),
            script_mismatches=("script_mismatch", "sum"),
            mean_characters=("character_count", "mean"),
            median_characters=("character_count", "median"),
            max_characters=("character_count", "max"),
            mean_nonspace_characters=(
                "nonspace_character_count",
                "mean",
            ),
            mean_tokens=("token_count", "mean"),
        )
        .reset_index()
    )

    summary["empty_output_percent"] = (
        100.0 * summary["empty_outputs"] / summary["utterances"]
    )
    summary["symbol_only_percent"] = (
        100.0 * summary["symbol_only_outputs"] / summary["utterances"]
    )
    summary["script_mismatch_percent"] = (
        100.0 * summary["script_mismatches"] / summary["utterances"]
    )

    return summary


def export_failure_audit(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    target_language: str = "te",
) -> dict[str, Path]:
    """Generate and export the detailed audit and grouped summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audited = audit_predictions(
        predictions,
        target_language=target_language,
    )
    summary = summarize_failures(audited)

    detailed_path = output_dir / f"{target_language}_failure_audit.csv"
    summary_path = output_dir / f"{target_language}_failure_summary.csv"

    audited.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "detailed_audit": detailed_path,
        "summary": summary_path,
    }
