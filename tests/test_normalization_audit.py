
"""Tests for normalization audit utilities."""

import pandas as pd
import pytest

from src.evaluation.normalization_audit import (
    audit_text_changes,
    summarize_text_changes,
)


def test_audit_reports_changed_text():
    references = pd.Series(["hello, world", "தமிழ் மொழி"])

    audit = audit_text_changes(references, normalizer="minimal")

    assert len(audit) == 2
    assert audit["text_changed"].all()
    assert "raw_tokens" in audit.columns
    assert "normalized_tokens" in audit.columns


def test_raw_normalization_preserves_text_when_already_normalized():
    references = pd.Series(["hello world", "தமிழ் மொழி"])

    audit = audit_text_changes(references, normalizer="raw")

    assert not audit["text_changed"].any()


def test_summary_contains_token_and_character_ratios():
    references = pd.Series(["hello, world", "hello there"])

    audit = audit_text_changes(references, normalizer="minimal")
    summary = summarize_text_changes(audit)

    assert len(summary) == 1
    assert "token_count_ratio" in summary.columns
    assert "character_count_ratio" in summary.columns
    assert summary.iloc[0]["utterances"] == 2


def test_unknown_normalizer_fails():
    references = pd.Series(["hello"])

    with pytest.raises(ValueError, match="Unknown normalizer"):
        audit_text_changes(references, normalizer="invalid")
