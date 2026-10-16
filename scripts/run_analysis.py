
"""
Run post-evaluation analysis for the ASR normalization project.

Example:
    python scripts/run_analysis.py \
        --input artifacts/predictions/all_predictions.parquet \
        --output-dir artifacts/analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import (  # noqa: E402
    analyze_predictions,
    export_error_analysis,
    get_worst_utterances,
)
from src.evaluation.normalization_audit import (  # noqa: E402
    run_normalization_audit,
)
from src.evaluation.telugu_failure_audit import (  # noqa: E402
    export_failure_audit,
)
from scripts.evaluate import load_predictions  # noqa: E402


LOGGER = logging.getLogger("asr_analysis")


def run_analysis(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 100,
    include_telugu_audit: bool = True,
) -> dict[str, Path]:
    """Run available diagnostic analyses and export their results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # 1. General error analysis.
    LOGGER.info("Running general error analysis.")

    error_dir = output_dir / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)

    analyzed = analyze_predictions(predictions)
    exported = export_error_analysis(analyzed, error_dir)
    paths.update(
        {
            f"error_analysis_{name}": path
            for name, path in exported.items()
        }
    )

    # 2. Highest-WER utterances.
    LOGGER.info("Exporting worst-performing utterances.")

    worst = get_worst_utterances(
        analyzed,
        top_n=top_n,
    )
    worst_path = output_dir / "worst_utterances.csv"
    worst.to_csv(worst_path, index=False)
    paths["worst_utterances"] = worst_path

    # 3. Normalization behavior and matched-normalizer metrics.
    LOGGER.info("Running normalization audit.")

    normalization_paths = run_normalization_audit(
        predictions,
        output_dir=output_dir / "normalization_audit",
    )
    paths.update(
        {
            f"normalization_{name}": path
            for name, path in normalization_paths.items()
        }
    )

    # 4. Telugu-specific output diagnostics.
    if include_telugu_audit:
        LOGGER.info("Running Telugu output-failure audit.")

        if "lang" in predictions.columns and (
            predictions["lang"] == "te"
        ).any():
            failure_paths = export_failure_audit(
                predictions,
                output_dir=output_dir / "telugu_failure_audit",
                target_language="te",
            )
            paths.update(
                {
                    f"telugu_{name}": path
                    for name, path in failure_paths.items()
                }
            )
        else:
            LOGGER.warning(
                "No Telugu predictions found; skipping Telugu audit."
            )

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASR error and normalization analyses."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV or Parquet file containing saved predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis",
        help="Directory for generated analysis reports.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of high-WER utterances to export.",
    )
    parser.add_argument(
        "--skip-telugu-audit",
        action="store_true",
        help="Skip the Telugu-specific failure audit.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    args = parser.parse_args()

    if args.top_n < 1:
        parser.error("--top-n must be at least 1.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    predictions = load_predictions(args.input)
    LOGGER.info("Loaded %d prediction rows.", len(predictions))

    paths = run_analysis(
        predictions=predictions,
        output_dir=args.output_dir,
        top_n=args.top_n,
        include_telugu_audit=not args.skip_telugu_audit,
    )

    print("\nAnalysis complete. Generated files:")
    for name, path in sorted(paths.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
