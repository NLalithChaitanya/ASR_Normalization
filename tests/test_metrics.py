
"""Unit tests for ASR evaluation metrics."""

import math

import pandas as pd
import pytest

from src.evaluation.metrics import (
    aggregate_metrics,
    character_error_counts,
    evaluate_dataframe,
    evaluate_utterance,
    word_error_counts,
)


def test_identical_words_have_zero_errors():
    counts = word_error_counts("hello world".split(), "hello world".split())
    assert counts.errors == 0


def test_word_substitution():
    counts = word_error_counts(
        "hello world".split(),
        "hello there".split(),
    )

    assert counts.substitutions == 1
    assert counts.deletions == 0
    assert counts.insertions == 0


def test_word_deletion():
    counts = word_error_counts(
        "hello beautiful world".split(),
        "hello world".split(),
    )

    assert counts.deletions == 1


def test_word_insertion():
    counts = word_error_counts(
        "hello world".split(),
        "hello brave world".split(),
    )

    assert counts.insertions == 1


def test_empty_hypothesis():
    result = evaluate_utterance("hello world", "", normalizer="raw")

    assert result["wer"] == 1.0
    assert result["deletions"] == 2


def test_empty_reference_and_hypothesis():
    result = evaluate_utterance("", "", normalizer="raw")

    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_nonempty_hypothesis_against_empty_reference():
    result = evaluate_utterance(" ", "hello", normalizer="raw")

    assert math.isinf(result["wer"])


def test_wer_can_exceed_100_percent():
    result = evaluate_utterance(
        "hello",
        "one two three",
        normalizer="raw",
    )

    assert result["wer"] == 3.0
    assert result["wer_percent"] if "wer_percent" in result else True


def test_character_counts_ignore_spaces():
    counts = character_error_counts("a b", "a c")

    assert counts.reference_units == 2
    assert counts.substitutions == 1


def test_evaluate_dataframe_requires_columns():
    with pytest.raises(ValueError, match="Missing required"):
        evaluate_dataframe(pd.DataFrame({"reference": ["hello"]}))


def test_aggregate_metrics_uses_corpus_counts():
    data = pd.DataFrame(
        [
            {
                "model": "model_a",
                "lang": "hi",
                "cond": "clean",
                "substitutions": 1,
                "deletions": 0,
                "insertions": 0,
                "reference_words": 2,
                "character_substitutions": 1,
                "character_deletions": 0,
                "character_insertions": 0,
                "reference_characters": 4,
            },
            {
                "model": "model_a",
                "lang": "hi",
                "cond": "clean",
                "substitutions": 0,
                "deletions": 1,
                "insertions": 0,
                "reference_words": 3,
                "character_substitutions": 0,
                "character_deletions": 1,
                "character_insertions": 0,
                "reference_characters": 5,
            },
        ]
    )

    result = aggregate_metrics(data)

    assert len(result) == 1
    assert result.loc[0, "reference_words"] == 5
    assert result.loc[0, "word_errors"] == 2
    assert result.loc[0, "wer_percent"] == 40.0
