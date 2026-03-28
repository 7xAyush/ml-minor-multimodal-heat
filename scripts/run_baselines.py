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
    merged = split_df[["image_id"]].merge(
        labels[["image_id", "label_class"]], on="image_id", how="inner"
    )
    merged = merged.merge(tabular, on="image_id", how="inner")
    # Handle possible duplicate label_class columns from tabular.csv
    if "label_class" not in merged.columns:
        for candidate in ("label_class_x", "label_class_y"):
            if candidate in merged.columns:
                merged["label_class"] = merged[candidate]
                break
    if merged.empty:
        raise ValueError("Split join produced zero rows for baseline.")
    return merged


def run_tabular_baseline(dataset_dir: Path, allow_train_as_test: bool = False) -> Dict[str, Any]:
    data = load_dataset(dataset_dir)
    tabular = data["tabular"]
    labels = data["labels"]
    splits = data["splits"]

    feature_cols = detect_tabular_features(tabular)

    train_df = build_split_frame(splits["train"], tabular, labels)
    try:
        test_df = build_split_frame(splits["test"], tabular, labels)
    except ValueError as exc:
        if not allow_train_as_test:
            raise ValueError(
                "Test split is empty or join produced zero rows for tabular baseline. "
                "For strict evaluation, this is not allowed. Increase the dataset "
                "or adjust splits so that the test split is non-empty."
            ) from exc
        print(
            "[WARN] Test split is empty or join produced zero rows for tabular baseline; "
            "falling back to using the training split as pseudo-test (SMOKE TEST ONLY)."
        )
        test_df = train_df.copy()

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


def run_majority_baseline(dataset_dir: Path, allow_train_as_test: bool = False) -> Dict[str, Any]:
    data = load_dataset(dataset_dir)
    labels = data["labels"]
    splits = data["splits"]

    train_df = splits["train"].merge(
        labels[["image_id", "label_class"]], on="image_id", how="inner"
    )
    test_df = splits["test"].merge(
        labels[["image_id", "label_class"]], on="image_id", how="inner"
    )
    # Normalize label column name in case of x/y suffixes
    for df in (train_df, test_df):
        if "label_class" not in df.columns:
            for candidate in ("label_class_x", "label_class_y"):
                if candidate in df.columns:
                    df["label_class"] = df[candidate]
                    break
    if train_df.empty:
        raise ValueError("Training split empty when computing majority baseline.")
    if test_df.empty:
        if not allow_train_as_test:
            raise ValueError(
                "Test split is empty for majority baseline. For strict evaluation, "
                "a non-empty test split is required. Increase dataset size or adjust "
                "splits so that test has samples."
            )
        print(
            "[WARN] Test split is empty for majority baseline; "
            "falling back to using the training split as pseudo-test (SMOKE TEST ONLY)."
        )
        test_df = train_df.copy()

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
    parser.add_argument(
        "--allow_train_as_test",
        action="store_true",
        help=(
            "Allow reusing the training split as a pseudo-test when the true test "
            "split is empty. ONLY for smoke tests; must not be used for final "
            "reported metrics."
        ),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["tabular_baseline"] = run_tabular_baseline(
        dataset_dir, allow_train_as_test=args.allow_train_as_test
    )
    results["majority_baseline"] = run_majority_baseline(
        dataset_dir, allow_train_as_test=args.allow_train_as_test
    )

    out_path = out_dir / "baselines.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Baseline results written to {out_path}")


if __name__ == "__main__":
    main()
