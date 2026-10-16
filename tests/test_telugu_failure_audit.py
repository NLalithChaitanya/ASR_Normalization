
import pandas as pd
import pytest

from src.evaluation.telugu_failure_audit import (
    audit_predictions,
    classify_script,
    inspect_hypothesis,
    summarize_failures,
)


def test_classifies_telugu_script():
    assert classify_script("తెలుగు భాష") == "telugu"


def test_classifies_devanagari_script():
    assert classify_script("हिंदी भाषा") == "devanagari"


def test_classifies_latin_script():
    assert classify_script("hello world") == "latin"


def test_classifies_symbol_only_output():
    assert classify_script(",,, !!!") == "no_letters"

    result = inspect_hypothesis(",,, !!!")
    assert result["symbol_only"] is True
    assert result["has_letters"] is False


def test_empty_hypothesis():
    result = inspect_hypothesis("   ")

    assert result["is_empty"] is True
    assert result["character_count"] == 3
    assert result["token_count"] == 0


def test_mixed_scripts():
    assert classify_script("తెలుగు hello") == "mixed"


def test_audit_filters_target_language_and_flags_mismatch():
    predictions = pd.DataFrame(
        {
            "reference": ["తెలుగు భాష", "हिंदी भाषा"],
            "hypothesis": ["हिंदी भाषा", "हिंदी भाषा"],
            "lang": ["te", "hi"],
            "model": ["Whisper-small", "Whisper-small"],
            "cond": ["clean", "clean"],
        }
    )

    audited = audit_predictions(predictions, target_language="te")

    assert len(audited) == 1
    assert audited.iloc[0]["script_class"] == "devanagari"
    assert bool(audited.iloc[0]["script_mismatch"])


def test_summary_counts_all_outputs():
    predictions = pd.DataFrame(
        {
            "reference": ["తెలుగు", "తెలుగు", "తెలుగు"],
            "hypothesis": ["తెలుగు", "हिंदी", ",,,"],
            "lang": ["te"] * 3,
            "model": ["Whisper-small"] * 3,
            "cond": ["clean"] * 3,
        }
    )

    audited = audit_predictions(predictions)
    summary = summarize_failures(audited)

    assert summary["utterances"].sum() == 3
    assert summary["script_mismatches"].sum() == 2


def test_missing_columns_raise_error():
    with pytest.raises(ValueError, match="missing columns"):
        audit_predictions(pd.DataFrame({"hypothesis": ["తెలుగు"]}))
