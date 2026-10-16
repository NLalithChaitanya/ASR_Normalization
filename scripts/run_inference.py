
"""
Run ASR inference on a validated dataset manifest.

Examples:
    python scripts/run_inference.py \
        --manifest data/manifest_validated.parquet \
        --output-dir artifacts/predictions \
        --models whisper indicconformer_ctc indicconformer_rnnt \
        --languages hi ta te

Note:
    Check the IndicConformer wrapper's expected audio input format and
    decoder API before launching a full inference run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_dataset import load_manifest  # noqa: E402
from src.models.whisper_asr import WhisperASR  # noqa: E402
from src.models.indicconformer_asr import IndicConformerASR  # noqa: E402


LOGGER = logging.getLogger("asr_inference")

MODEL_NAMES = {
    "whisper",
    "indicconformer_ctc",
    "indicconformer_rnnt",
}

LANGUAGE_NAMES = {"hi", "ta", "te"}
CONDITIONS = {"clean", "noisy"}


def load_existing_predictions(path: Path) -> pd.DataFrame:
    """Load an existing prediction file when resuming inference."""
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported prediction file: {path}")


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write predictions using the selected file format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        predictions.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        predictions.to_csv(output_path, index=False)
    else:
        raise ValueError("Output format must be .parquet or .csv")


def _make_model(model_name: str):
    """Instantiate the requested model wrapper."""
    if model_name == "whisper":
        return WhisperASR(model_name="openai/whisper-small")

    if model_name == "indicconformer_ctc":
        return IndicConformerASR(decoding="ctc")

    if model_name == "indicconformer_rnnt":
        return IndicConformerASR(decoding="rnnt")

    raise ValueError(f"Unsupported model: {model_name}")


def _transcribe(model, model_name: str, audio_path: str, language: str):
    """Dispatch to the appropriate model wrapper."""
    if model_name == "whisper":
        return model.transcribe(
            audio_path,
            language=language,
        )

    return model.transcribe(
        audio_path,
        language=language,
    )


def run_inference(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    models: list[str],
    languages: list[str],
    conditions: list[str],
    output_format: str = "parquet",
    resume: bool = True,
    checkpoint_every: int = 25,
) -> list[Path]:
    """Run selected models and save per-model prediction files."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive.")

    unknown_models = set(models).difference(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unsupported models: {sorted(unknown_models)}")

    unknown_languages = set(languages).difference(LANGUAGE_NAMES)
    if unknown_languages:
        raise ValueError(
            f"Unsupported languages: {sorted(unknown_languages)}"
        )

    unknown_conditions = set(conditions).difference(CONDITIONS)
    if unknown_conditions:
        raise ValueError(
            f"Unsupported conditions: {sorted(unknown_conditions)}"
        )

    required = {
        "utt_id",
        "lang",
        "cond",
        "audio_path",
        "reference",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".parquet" if output_format == "parquet" else ".csv"
    output_paths = []

    for model_name in models:
        output_path = output_dir / f"{model_name}_predictions{suffix}"
        output_paths.append(output_path)

        selected = manifest.loc[
            manifest["lang"].isin(languages)
            & manifest["cond"].isin(conditions)
        ].copy()

        existing = (
            load_existing_predictions(output_path)
            if resume
            else pd.DataFrame()
        )

        if not existing.empty:
            completed_keys = set(
                zip(
                    existing["utt_id"].astype(str),
                    existing["lang"].astype(str),
                    existing["cond"].astype(str),
                )
            )
            selected = selected.loc[
                ~selected.apply(
                    lambda row: (
                        str(row["utt_id"]),
                        str(row["lang"]),
                        str(row["cond"]),
                    ) in completed_keys,
                    axis=1,
                )
            ]

        if selected.empty:
            LOGGER.info(
                "No pending rows for %s; skipping.",
                model_name,
            )
            continue

        LOGGER.info(
            "Starting %s for %d utterances.",
            model_name,
            len(selected),
        )

        model = _make_model(model_name)
        new_records = []

        try:
            for row in selected.itertuples(index=False):
                start = time.perf_counter()

                record = {
                    "utt_id": str(row.utt_id),
                    "lang": row.lang,
                    "cond": row.cond,
                    "reference": str(row.reference),
                    "hypothesis": "",
                    "model": model_name,
                    "status": "error",
                    "error": "",
                    "inference_seconds": None,
                }

                if not Path(str(row.audio_path)).is_file():
                    record["error"] = (
                        f"Audio file not found: {row.audio_path}"
                    )
                else:
                    try:
                        prediction = _transcribe(
                            model,
                            model_name,
                            str(row.audio_path),
                            str(row.lang),
                        )

                        record["hypothesis"] = getattr(
                            prediction,
                            "hypothesis",
                            "",
                        )
                        record["status"] = getattr(
                            prediction,
                            "status",
                            "ok",
                        )
                        record["error"] = getattr(
                            prediction,
                            "error",
                            "",
                        )

                    except Exception as exc:
                        LOGGER.exception(
                            "Inference failed for utterance %s",
                            row.utt_id,
                        )
                        record["error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                record["inference_seconds"] = (
                    time.perf_counter() - start
                )
                new_records.append(record)

                if len(new_records) % checkpoint_every == 0:
                    combined = pd.concat(
                        [existing, pd.DataFrame(new_records)],
                        ignore_index=True,
                    )
                    combined = combined.drop_duplicates(
                        subset=["utt_id", "lang", "cond"],
                        keep="last",
                    )
                    save_predictions(combined, output_path)
                    existing = combined
                    new_records = []

                    LOGGER.info(
                        "Checkpoint saved: %s",
                        output_path,
                    )

        finally:
            # Save completed work even if inference is interrupted.
            if new_records:
                combined = pd.concat(
                    [existing, pd.DataFrame(new_records)],
                    ignore_index=True,
                )
                combined = combined.drop_duplicates(
                    subset=["utt_id", "lang", "cond"],
                    keep="last",
                )
                save_predictions(combined, output_path)

            del model

        LOGGER.info("Finished %s.", model_name)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ASR predictions from a manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Validated manifest path (CSV or Parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/predictions",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_NAMES),
        default=["whisper", "indicconformer_ctc", "indicconformer_rnnt"],
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=sorted(LANGUAGE_NAMES),
        default=["hi", "ta", "te"],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=["clean", "noisy"],
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing prediction files and start fresh.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    manifest = load_manifest(args.manifest)

    paths = run_inference(
        manifest=manifest,
        output_dir=args.output_dir,
        models=args.models,
        languages=args.languages,
        conditions=args.conditions,
        output_format=args.format,
        resume=not args.no_resume,
        checkpoint_every=args.checkpoint_every,
    )

    print("\nPrediction files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
