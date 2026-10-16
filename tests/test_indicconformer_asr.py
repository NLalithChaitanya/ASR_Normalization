
"""Unit tests for the IndicConformer wrapper."""

import pytest
import torch

from src.models.indicconformer_asr import (
    IndicConformerASR,
    IndicConformerPrediction,
)


def test_invalid_decoder_is_rejected():
    with pytest.raises(ValueError, match="Unsupported decoding"):
        IndicConformerASR(decoding="invalid")


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="device must"):
        IndicConformerASR(device="tpu")


def test_cuda_without_gpu_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        IndicConformerASR(device="cuda")


def test_invalid_sample_rate_is_rejected():
    with pytest.raises(ValueError, match="target_sample_rate"):
        IndicConformerASR(target_sample_rate=0)


def test_supported_decoders():
    assert IndicConformerASR(decoding="ctc").decoding == "ctc"
    assert IndicConformerASR(decoding="rnnt").decoding == "rnnt"


def test_transcription_output_extraction():
    assert (
        IndicConformerASR._extract_transcription(" నమస్తే ")
        == "నమస్తే"
    )


def test_single_item_output_extraction():
    assert IndicConformerASR._extract_transcription(["नमस्ते"]) == "नमस्ते"


def test_invalid_output_raises_error():
    with pytest.raises(RuntimeError, match="Unexpected"):
        IndicConformerASR._extract_transcription(object())


def test_prediction_serializes_to_dict():
    prediction = IndicConformerPrediction(
        audio_path="sample.wav",
        hypothesis="வணக்கம்",
        model="ai4bharat/indic-conformer-600m-multilingual",
        language="ta",
        decoding="ctc",
        sample_rate=16000,
        duration_seconds=2.0,
        inference_seconds=0.5,
        status="success",
    )

    result = prediction.to_dict()

    assert result["language"] == "ta"
    assert result["decoding"] == "ctc"
    assert result["status"] == "success"
