#!/usr/bin/env python
"""
Run simple baseline models on an existing dataset:

- Tabular-only baseline (sklearn classifier on weather [+ NDVI] features)
- Majority-class baseline

Results are written to:
  dataset/experiments/baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_dataset(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    tabular = pd.read_csv(dataset_dir / "tabular.csv")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    splits_dir = dataset_dir / "splits"

    splits = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        splits[split] = df

    return {"tabular": tabular, "labels": labels, "splits": splits}


def detect_tabular_features(tabular: pd.DataFrame) -> list[str]:
    exclude = {"image_id", "label_class", "lst"}
    cols: list[str] = []
    for col in tabular.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(tabular[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric tabular feature columns found for baseline.")
    return cols


def build_split_frame(
    split_df: pd.DataFrame, tabular: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    # Ensure we have label_class and features for each image_id in the split
    merged = split_df[["image_id"]].merge(labels[["image_id", "label_class"]], on="image_id", how="inner")
    merged = merged.merge(tabular, on="image_id", how="inner")
    if merged.empty:
        raise ValueError("Split join produced zero rows for baseline.")
    return merged


def run_tabular_baseline(dataset_dir: Path) -> Dict[str, Any]:
    data = load_dataset(dataset_dir)
    tabular = data["tabular"]
    labels = data["labels"]
    splits = data["splits"]

    feature_cols = detect_tabular_features(tabular)

    train_df = build_split_frame(splits["train"], tabular, labels)
    test_df = build_split_frame(splits["test"], tabular, labels)

    X_train = train_df[feature_cols].values
    y_train = train_df["label_class"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label_class"].values

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(
        y_test, y_pred, digits=4, zero_division=0
    )

    return {
        "model": "TabularRandomForest",
        "input_type": "tabular",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "feature_columns": feature_cols,
    }


def run_majority_baseline(dataset_dir: Path) -> Dict[str, Any]:
    data = load_dataset(dataset_dir)
    labels = data["labels"]
    splits = data["splits"]

    train_df = splits["train"].merge(labels[["image_id", "label_class"]], on="image_id", how="inner")
    test_df = splits["test"].merge(labels[["image_id", "label_class"]], on="image_id", how="inner")
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test splits empty when computing majority baseline.")

    majority_class = int(train_df["label_class"].value_counts().idxmax())
    y_test = test_df["label_class"].values
    y_pred = np.full_like(y_test, fill_value=majority_class)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(
        y_test, y_pred, digits=4, zero_division=0
    )

    return {
        "model": "MajorityClass",
        "input_type": "none",
        "majority_class": majority_class,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple baseline models (tabular-only, majority) on an existing dataset."
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
        help="Directory to save baseline results (default: <dataset_dir>/experiments/).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["tabular_baseline"] = run_tabular_baseline(dataset_dir)
    results["majority_baseline"] = run_majority_baseline(dataset_dir)

    out_path = out_dir / "baselines.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Baseline results written to {out_path}")


if __name__ == "__main__":
    main()

