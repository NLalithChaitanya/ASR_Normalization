
"""
Whisper ASR inference wrapper.

Supports:
- Lazy model loading
- CPU or CUDA inference
- Audio-file transcription
- Batch processing through repeated single-file inference
- Structured prediction records

Model output is kept separate from text normalization and evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from transformers import pipeline

from src.data.audio_utils import load_audio


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperPrediction:
    """Prediction and inference metadata for one audio file."""

    audio_path: str
    hypothesis: str
    model: str
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    status: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WhisperASR:
    """Transcribe audio using a Hugging Face Whisper checkpoint."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str = "auto",
        target_sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_length_s: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.language = language
        self.task = task
        self.chunk_length_s = chunk_length_s

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False"
            )

        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")

        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")

        if chunk_length_s is not None and chunk_length_s <= 0:
            raise ValueError("chunk_length_s must be positive")

        self.device_name = device
        self._pipeline = None

    def load_model(self) -> None:
        """Load the ASR pipeline once."""
        if self._pipeline is not None:
            return

        pipeline_device = 0 if self.device_name == "cuda" else -1

        LOGGER.info(
            "Loading Whisper model '%s' on %s",
            self.model_name,
            self.device_name,
        )

        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model_name,
            device=pipeline_device,
        )

        LOGGER.info("Whisper model loaded")

    def _transcribe_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Run inference on a validated waveform."""
        self.load_model()

        assert self._pipeline is not None

        audio_input = {
            "raw": waveform,
            "sampling_rate": sample_rate,
        }

        generate_kwargs: dict[str, Any] = {
            "task": self.task,
        }

        if self.language:
            generate_kwargs["language"] = self.language

        call_kwargs: dict[str, Any] = {
            "generate_kwargs": generate_kwargs,
        }

        if self.chunk_length_s is not None:
            call_kwargs["chunk_length_s"] = self.chunk_length_s

        output = self._pipeline(audio_input, **call_kwargs)

        if isinstance(output, dict):
            text = output.get("text", "")
        else:
            raise RuntimeError(
                f"Unexpected Whisper pipeline output: {type(output).__name__}"
            )

        if not isinstance(text, str):
            raise RuntimeError("Whisper returned a non-string transcription")

        return text.strip()

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> WhisperPrediction:
        """Transcribe one audio file and return a structured result."""
        path = Path(audio_path)
        started = time.perf_counter()

        try:
            audio = load_audio(
                path,
                target_sample_rate=self.target_sample_rate,
                mono=True,
            )

            hypothesis = self._transcribe_waveform(
                audio.waveform,
                audio.sample_rate,
            )

            elapsed = time.perf_counter() - started

            return WhisperPrediction(
                audio_path=str(path),
                hypothesis=hypothesis,
                model=self.model_name,
                sample_rate=audio.sample_rate,
                duration_seconds=audio.metadata.duration_seconds,
                inference_seconds=elapsed,
                status="success",
            )

        except Exception as exc:
            elapsed = time.perf_counter() - started

            LOGGER.exception("Whisper inference failed for %s", path)

            return WhisperPrediction(
                audio_path=str(path),
                hypothesis="",
                model=self.model_name,
                sample_rate=self.target_sample_rate,
                duration_seconds=0.0,
                inference_seconds=elapsed,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def transcribe_many(
        self,
        audio_paths: Sequence[str | Path],
        fail_fast: bool = False,
    ) -> list[WhisperPrediction]:
        """
        Transcribe multiple files sequentially.

        The model is reused across files. This method favors predictable
        memory usage over batched inference.
        """
        predictions: list[WhisperPrediction] = []

        for index, audio_path in enumerate(audio_paths, start=1):
            LOGGER.info(
                "Transcribing file %d/%d: %s",
                index,
                len(audio_paths),
                audio_path,
            )

            prediction = self.transcribe(audio_path)
            predictions.append(prediction)

            if fail_fast and prediction.status != "success":
                raise RuntimeError(
                    f"Inference failed for {audio_path}: {prediction.error}"
                )

        return predictions
