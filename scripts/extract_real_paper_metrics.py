#!/usr/bin/env python
"""
Extract REAL-mode metrics and dataset summaries into CSVs suitable for papers.

Outputs (under --dataset_dir, default: dataset_real):
  - paper_dataset_summary.csv
  - paper_model_performance.csv
  - paper_weather_stats.csv

This script does NOT fabricate values; it only reads existing
REAL-mode artifacts (metadata.json, baselines.json, classification_report.txt,
tabular.csv, labels.csv).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_dataset_summary(dataset_dir: Path, meta: Dict[str, Any]) -> None:
    mode = meta.get("mode", {})
    split_sizes = meta.get("split_sizes", {})
    class_dist = meta.get("class_distribution", {})
    image_stats = meta.get("image_stats", {})
    tab_stats = meta.get("tabular_stats", {})
    preprocess = meta.get("preprocess_stats", {}).get("tabular", {})

    total_samples = int(meta.get("total_samples", 0))
    n_classes_present = len(class_dist)
    feature_cols: List[str] = preprocess.get("feature_columns", [])

    rows = [
        {"item": "data_mode", "value": mode.get("data_mode", "unknown")},
        {"item": "lst_source", "value": mode.get("lst_source", "unknown")},
        {"item": "imagery_source", "value": mode.get("image_source", "unknown")},
        {"item": "weather_source", "value": mode.get("weather_source", "unknown")},
        {"item": "urban_source", "value": mode.get("urban_source", "unknown")},
        {"item": "total_samples", "value": total_samples},
        {
            "item": "train_val_test_counts",
            "value": f"{split_sizes.get('train', 0)}/"
            f"{split_sizes.get('val', 0)}/"
            f"{split_sizes.get('test', 0)}",
        },
        {"item": "n_classes_present", "value": n_classes_present},
        {"item": "class_distribution", "value": json.dumps(class_dist)},
        {
            "item": "image_size",
            "value": "×".join(str(x) for x in image_stats.get("output_image_size", [])),
        },
        {
            "item": "tabular_numeric_features",
            "value": ", ".join(preprocess.get("numeric_columns", [])),
        },
        {
            "item": "tabular_feature_columns",
            "value": ", ".join(feature_cols),
        },
    ]

    # Include basic numeric summary for LST if available
    numeric_summary = tab_stats.get("numeric_summary", {})
    lst_stats = numeric_summary.get("lst")
    if lst_stats:
        rows.append(
            {
                "item": "lst_summary",
                "value": json.dumps(
                    {
                        "min": lst_stats.get("min"),
                        "max": lst_stats.get("max"),
                        "mean": lst_stats.get("mean"),
                        "std": lst_stats.get("std"),
                    }
                ),
            }
        )

    df = pd.DataFrame(rows)
    out_path = dataset_dir / "paper_dataset_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote dataset summary to {out_path}")


def parse_classification_report_text(rep_text: str) -> Dict[str, Any]:
    """
    Parse a scikit-learn style classification report text into aggregate metrics.
    Only extracts:
      - accuracy
      - macro F1
      - weighted F1
    """
    acc = None
    macro_f1 = None
    weighted_f1 = None

    for line in rep_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if stripped.startswith("accuracy"):
            # e.g. "accuracy                         0.0000       1.0"
            try:
                acc = float(tokens[-2])
            except Exception:
                continue
        elif stripped.startswith("macro avg"):
            # e.g. "macro avg     0.0000    0.0000    0.0000       1.0"
            if len(tokens) >= 4:
                try:
                    macro_f1 = float(tokens[-2])
                except Exception:
                    continue
        elif stripped.startswith("weighted avg"):
            if len(tokens) >= 4:
                try:
                    weighted_f1 = float(tokens[-2])
                except Exception:
                    continue

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def write_model_performance(dataset_dir: Path, meta: Dict[str, Any]) -> None:
    exp_dir = dataset_dir / "experiments"
    baselines_path = exp_dir / "baselines.json"

    rows: List[Dict[str, Any]] = []

    # Baselines (tabular-only and majority)
    if baselines_path.exists():
        with baselines_path.open("r", encoding="utf-8") as f:
            baselines = json.load(f)
        for key, entry in baselines.items():
            rows.append(
                {
                    "model": entry.get("model", key),
                    "input_type": entry.get("input_type", "unknown"),
                    "split_used": "train(pseudo-test)",  # tiny dataset fallback
                    "accuracy": entry.get("accuracy"),
                    "macro_f1": entry.get("macro_f1"),
                    "weighted_f1": None,
                    "note": "evaluated on training data reused as test (tiny dataset)",
                }
            )

    # Multimodal model metrics from classification_report.txt
    models_dir = dataset_dir / "models"
    rep_path = models_dir / "classification_report.txt"
    split_sizes = meta.get("split_sizes", {})
    split_used = "test"
    if split_sizes.get("test", 0) == 0:
        split_used = "val(pseudo-test)"

    if rep_path.exists():
        with rep_path.open("r", encoding="utf-8") as f:
            rep_text = f.read()
        agg = parse_classification_report_text(rep_text)
        rows.append(
            {
                "model": "MultimodalNet",
                "input_type": "image+tabular",
                "split_used": split_used,
                "accuracy": agg.get("accuracy"),
                "macro_f1": agg.get("macro_f1"),
                "weighted_f1": agg.get("weighted_f1"),
                "note": "evaluated on pseudo-test split (no true test samples)"
                if split_used != "test"
                else "",
            }
        )

    if not rows:
        print("[WARN] No baseline or multimodal metrics found; skipping model performance CSV.")
        return

    df = pd.DataFrame(rows)
    out_path = dataset_dir / "paper_model_performance.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote model performance to {out_path}")


def write_weather_stats(dataset_dir: Path) -> None:
    tab_path = dataset_dir / "tabular.csv"
    if not tab_path.exists():
        print(f"[WARN] tabular.csv not found at {tab_path}; skipping weather stats.")
        return

    tab = pd.read_csv(tab_path)
    # Select numeric weather columns (exclude identifiers and labels)
    exclude = {"image_id", "label_class"}
    weather_cols = [c for c in tab.columns if c not in exclude]
    if not weather_cols:
        print("[WARN] No numeric weather columns found in tabular.csv; skipping weather stats.")
        return

    desc = tab[weather_cols].describe().T  # index = feature, columns = count, mean, std, min, 25%, 50%, 75%, max
    # Keep only key stats
    cols_to_keep = ["count", "min", "max", "mean", "std"]
    desc = desc[cols_to_keep].reset_index().rename(columns={"index": "feature"})

    out_path = dataset_dir / "paper_weather_stats.csv"
    desc.to_csv(out_path, index=False)
    print(f"[INFO] Wrote weather feature stats to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract REAL-mode paper metrics into CSV files."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset_real",
        help="Path to REAL dataset directory (default: dataset_real).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    meta = load_metadata(dataset_dir)
    mode = meta.get("mode", {})
    if mode.get("data_mode") != "real":
        print(
            f"[WARN] metadata.json reports data_mode='{mode.get('data_mode')}'. "
            "This script is intended for REAL-mode datasets."
        )

    write_dataset_summary(dataset_dir, meta)
    write_model_performance(dataset_dir, meta)
    write_weather_stats(dataset_dir)


if __name__ == "__main__":
    main()

