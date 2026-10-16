
"""
IndicConformer ASR inference wrapper.

Supports the AI4Bharat multilingual IndicConformer checkpoint with
CTC and RNNT decoding.

Supported project languages:
    hi: Hindi
    ta: Tamil
    te: Telugu

Inference is separated from evaluation and normalization so that saved
transcriptions can be evaluated repeatedly without rerunning the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel

from src.data.audio_utils import load_audio


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"

SUPPORTED_LANGUAGES = {"hi", "ta", "te"}
SUPPORTED_DECODERS = {"ctc", "rnnt"}


@dataclass(frozen=True)
class IndicConformerPrediction:
    """Transcription and inference metadata for one audio file."""

    audio_path: str
    hypothesis: str
    model: str
    language: str
    decoding: str
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    status: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the prediction to a serializable dictionary."""
        return asdict(self)


class IndicConformerASR:
    """Run CTC or RNNT inference using IndicConformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        decoding: str = "ctc",
        device: str = "auto",
        target_sample_rate: int = 16000,
        trust_remote_code: bool = True,
    ) -> None:
        if decoding not in SUPPORTED_DECODERS:
            raise ValueError(
                f"Unsupported decoding '{decoding}'. "
                f"Choose from {sorted(SUPPORTED_DECODERS)}."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )

        if (
            isinstance(target_sample_rate, bool)
            or not isinstance(target_sample_rate, int)
            or target_sample_rate <= 0
        ):
            raise ValueError("target_sample_rate must be a positive integer")

        self.model_name = model_name
        self.decoding = decoding
        self.device_name = device
        self.device = torch.device(device)
        self.target_sample_rate = target_sample_rate
        self.trust_remote_code = trust_remote_code

        self._model = None

    def load_model(self) -> None:
        """Load the model once and reuse it for subsequent predictions."""
        if self._model is not None:
            return

        LOGGER.info(
            "Loading IndicConformer '%s' on %s",
            self.model_name,
            self.device_name,
        )

        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        model = model.to(self.device)
        model.eval()

        self._model = model

        LOGGER.info("IndicConformer loaded successfully")

    @staticmethod
    def _extract_transcription(output: Any) -> str:
        """
        Convert the model output into a transcription string.

        The documented checkpoint returns a transcription. Unexpected
        output types raise an error instead of being silently stringified.
        """
        if isinstance(output, str):
            return output.strip()

        if isinstance(output, (list, tuple)):
            if len(output) == 1:
                return IndicConformerASR._extract_transcription(output[0])

            raise RuntimeError(
                "Expected one transcription, but received multiple outputs."
            )

        if isinstance(output, dict):
            for key in ("text", "transcription"):
                value = output.get(key)
                if isinstance(value, str):
                    return value.strip()

        raise RuntimeError(
            "Unexpected IndicConformer output type: "
            f"{type(output).__name__}"
        )

    @torch.inference_mode()
    def _transcribe_waveform(
        self,
        waveform: np.ndarray,
        language: str,
    ) -> str:
        """Run inference on a mono waveform sampled at the target rate."""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported project language '{language}'. "
                f"Supported languages: {sorted(SUPPORTED_LANGUAGES)}"
            )

        self.load_model()

        if self._model is None:
            raise RuntimeError("IndicConformer failed to initialize")

        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Expected a non-empty mono waveform")

        if not np.isfinite(waveform).all():
            raise ValueError("Waveform contains NaN or infinite values")

        # Official example uses a [1, samples] waveform tensor.
        audio_tensor = torch.from_numpy(waveform).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        output = self._model(
            audio_tensor,
            language,
            self.decoding,
        )

        return self._extract_transcription(output)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str,
    ) -> IndicConformerPrediction:
        """Transcribe one audio file and return a structured prediction."""
        path = Path(audio_path)
        started = time.perf_counter()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported languages: {sorted(SUPPORTED_LANGUAGES)}"
            )

        try:
            audio = load_audio(
                path,
                target_sample_rate=self.target_sample_rate,
                mono=True,
            )

            hypothesis = self._transcribe_waveform(
                audio.waveform,
                language,
            )

            return IndicConformerPrediction(
                audio_path=str(path),
                hypothesis=hypothesis,
                model=self.model_name,
                language=language,
                decoding=self.decoding,
                sample_rate=audio.sample_rate,
                duration_seconds=audio.metadata.duration_seconds,
                inference_seconds=time.perf_counter() - started,
                status="success",
            )

        except Exception as exc:
            LOGGER.exception(
                "IndicConformer inference failed for %s (%s, %s)",
                path,
                language,
                self.decoding,
            )

            return IndicConformerPrediction(
                audio_path=str(path),
                hypothesis="",
                model=self.model_name,
                language=language,
                decoding=self.decoding,
                sample_rate=self.target_sample_rate,
                duration_seconds=0.0,
                inference_seconds=time.perf_counter() - started,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def transcribe_many(
        self,
        audio_paths: Sequence[str | Path],
        language: str,
        fail_fast: bool = False,
    ) -> list[IndicConformerPrediction]:
        """Transcribe multiple files sequentially using the loaded model."""
        predictions = []

        for index, audio_path in enumerate(audio_paths, start=1):
            LOGGER.info(
                "Transcribing %d/%d using %s: %s",
                index,
                len(audio_paths),
                self.decoding.upper(),
                audio_path,
            )

            prediction = self.transcribe(audio_path, language)
            predictions.append(prediction)

            if fail_fast and prediction.status != "success":
                raise RuntimeError(
                    f"Inference failed for {audio_path}: {prediction.error}"
                )

        return predictions
