#!/usr/bin/env python
"""
Run simple ablation experiments:

- Multimodal (image + tabular) via train_multimodal.py
- Tabular-only baseline (sklearn RandomForest)
- Image-only baseline (placeholder: currently proxied by multimodal model)

Results are written to:
  dataset/experiments/ablations.json

This script assumes REAL (or at least non-synthetic) data has been built in `dataset/`
via build_dataset.py in REAL mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from scripts.run_baselines import run_tabular_baseline


def run_multimodal_results(dataset_dir: Path) -> Dict[str, Any]:
    """
    Read multimodal test metrics from training/evaluation artifacts.
    Assumes that:
      - train_multimodal.py has been run
      - src.evaluate_test.py has been run
    """
    models_dir = dataset_dir / "models"
    history_path = models_dir / "history.json"
    rep_path = models_dir / "classification_report.txt"

    if not history_path.exists():
        raise FileNotFoundError(
            f"history.json not found at {history_path}. "
            "Run scripts/run_experiment.py first."
        )

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list) or not history:
        raise ValueError("history.json must contain a non-empty list of epoch records.")

    last = history[-1]
    result: Dict[str, Any] = {
        "model": "MultimodalNet",
        "input_type": "image+tabular",
        "train_acc_last": float(last.get("train_acc", float("nan"))),
        "val_acc_last": float(last.get("val_acc", float("nan"))),
        "train_macro_f1_last": float(last.get("train_macro_f1", float("nan"))),
        "val_macro_f1_last": float(last.get("val_macro_f1", float("nan"))),
    }

    if rep_path.exists():
        with rep_path.open("r", encoding="utf-8") as f:
            report_text = f.read()
        result["classification_report"] = report_text

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments (multimodal vs tabular-only vs image-only placeholder)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save ablation results (default: <dataset_dir>/experiments/).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations: Dict[str, Any] = {}

    # Multimodal results (from training + evaluation artifacts)
    ablations["multimodal"] = run_multimodal_results(dataset_dir)

    # Tabular-only baseline (from run_baselines logic)
    ablations["tabular_only"] = run_tabular_baseline(dataset_dir)

    # Image-only baseline is not yet implemented as a separate model in this repo.
    # We record a placeholder entry to keep the structure stable.
    ablations["image_only_placeholder"] = {
        "model": "ImageOnlyCNN",
        "input_type": "image",
        "note": "Not implemented in this version; image-only baseline would use a CNN trained on images only.",
    }

    out_path = out_dir / "ablations.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ablations, f, indent=2)

    print(f"[INFO] Ablation results written to {out_path}")


if __name__ == "__main__":
    main()

