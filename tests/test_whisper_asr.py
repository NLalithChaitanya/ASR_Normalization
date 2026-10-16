
"""Unit tests for the Whisper inference wrapper."""

import pytest
import torch

from src.models.whisper_asr import WhisperASR, WhisperPrediction


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="device must"):
        WhisperASR(device="tpu")


def test_cuda_request_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        WhisperASR(device="cuda")


def test_invalid_task_is_rejected():
    with pytest.raises(ValueError, match="task must"):
        WhisperASR(task="summarize")


def test_invalid_sample_rate_is_rejected():
    with pytest.raises(ValueError, match="target_sample_rate"):
        WhisperASR(target_sample_rate=0)


def test_prediction_serializes_to_dict():
    prediction = WhisperPrediction(
        audio_path="sample.wav",
        hypothesis="नमस्ते",
        model="openai/whisper-small",
        sample_rate=16000,
        duration_seconds=1.0,
        inference_seconds=0.25,
        status="success",
    )

    result = prediction.to_dict()

    assert result["hypothesis"] == "नमस्ते"
    assert result["status"] == "success"
    assert result["error"] is None
