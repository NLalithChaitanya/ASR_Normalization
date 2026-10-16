
"""
Text normalization utilities for multilingual ASR evaluation.

Supported normalizers
---------------------
raw:
    Unicode NFC normalization and whitespace cleanup.

minimal:
    Unicode NFC normalization, punctuation separation, and
    whitespace cleanup. Unicode letters and combining marks are preserved.

whisper_style:
    Hugging Face Whisper BasicTextNormalizer.

Keep reference and hypothesis normalization consistent when computing
WER and CER. Do not compare metrics computed with mismatched normalizers.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Callable, Dict, Optional


Normalizer = Callable[[str], str]


def _to_text(text: Optional[str]) -> str:
    """Convert a nullable text value to a safe string."""
    if text is None:
        return ""

    if not isinstance(text, str):
        return str(text)

    return text


def _collapse_whitespace(text: str) -> str:
    """Replace consecutive whitespace characters with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_raw(text: Optional[str]) -> str:
    """
    Apply conservative Unicode normalization.

    Preserves punctuation, digits, letters, and combining marks.
    """
    text = _to_text(text)
    text = unicodedata.normalize("NFC", text)
    return _collapse_whitespace(text)


def normalize_minimal(text: Optional[str]) -> str:
    """
    Apply NFC normalization and separate Unicode punctuation.

    Punctuation and symbols are otherwise preserved. This deliberately
    avoids removing Indic combining marks, which can affect orthography.
    """
    text = _to_text(text)
    text = unicodedata.normalize("NFC", text)

    output = []

    for character in text:
        category = unicodedata.category(character)

        if category.startswith("P"):
            output.append(" ")
        else:
            output.append(character)

    return _collapse_whitespace("".join(output))


@lru_cache(maxsize=1)
def _get_whisper_normalizer():
    """Load Whisper's text normalizer only when it is needed."""
    try:
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer,
        )
    except ImportError as exc:
        raise ImportError(
            "Whisper-style normalization requires transformers. "
            "Install it using: pip install transformers"
        ) from exc

    return BasicTextNormalizer()


def normalize_whisper_style(text: Optional[str]) -> str:
    """
    Apply Hugging Face Whisper BasicTextNormalizer.

    Important:
        This normalizer may alter punctuation, symbols, and script-specific
        characters. Audit its effects before using it for Indic languages.
    """
    text = _to_text(text)
    normalizer = _get_whisper_normalizer()

    normalized = normalizer(text)

    return _collapse_whitespace(normalized)


NORMALIZERS: Dict[str, Normalizer] = {
    "raw": normalize_raw,
    "minimal": normalize_minimal,
    "whisper_style": normalize_whisper_style,
}


def normalize_text(
    text: Optional[str],
    method: str = "minimal",
) -> str:
    """
    Normalize a string using a named normalization method.

    Args:
        text: Input text.
        method: One of 'raw', 'minimal', or 'whisper_style'.

    Returns:
        Normalized text.

    Raises:
        ValueError: If the normalization method is unsupported.
    """
    try:
        normalizer = NORMALIZERS[method]
    except KeyError as exc:
        supported = ", ".join(sorted(NORMALIZERS))
        raise ValueError(
            f"Unknown normalizer '{method}'. Supported methods: {supported}"
        ) from exc

    return normalizer(text)


def normalize_pair(
    reference: Optional[str],
    hypothesis: Optional[str],
    method: str = "minimal",
) -> tuple[str, str]:
    """
    Normalize a reference-hypothesis pair with the same method.
    """
    return (
        normalize_text(reference, method),
        normalize_text(hypothesis, method),
    )
