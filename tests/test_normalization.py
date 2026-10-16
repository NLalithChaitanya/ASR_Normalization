
"""Unit tests for multilingual ASR text normalization."""

import unicodedata

import pytest

from src.evaluation.normalization import (
    normalize_minimal,
    normalize_pair,
    normalize_raw,
    normalize_text,
)


def test_raw_collapses_whitespace():
    assert normalize_raw("  hello   world\n") == "hello world"


def test_raw_applies_unicode_nfc():
    decomposed = "e\u0301"
    assert normalize_raw(decomposed) == unicodedata.normalize("NFC", decomposed)


def test_minimal_separates_punctuation():
    assert normalize_minimal("hello,world!") == "hello world"


def test_minimal_preserves_indic_text():
    text = "सुखदेव विकास"
    assert normalize_minimal(text) == text


def test_minimal_preserves_digits():
    assert normalize_minimal("Order 123") == "Order 123"


def test_none_becomes_empty_string():
    assert normalize_raw(None) == ""
    assert normalize_minimal(None) == ""


def test_unknown_normalizer_raises_error():
    with pytest.raises(ValueError, match="Unknown normalizer"):
        normalize_text("hello", "unsupported")


def test_normalize_pair_uses_same_method():
    reference, hypothesis = normalize_pair(
        "hello, world!",
        "hello world",
        method="minimal",
    )

    assert reference == "hello world"
    assert hypothesis == "hello world"


def test_whisper_style_is_available_when_transformers_is_installed():
    pytest.importorskip("transformers")

    from src.evaluation.normalization import normalize_whisper_style

    result = normalize_whisper_style("Hello, WORLD!")
    assert isinstance(result, str)
