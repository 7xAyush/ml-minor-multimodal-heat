#!/usr/bin/env python
"""
Generate simple figures for a one-city REAL experiment:

1. Training curves (accuracy and macro-F1 vs epoch) from history.json.
2. Copy or re-save confusion_matrix.png into experiments/figures/.
3. Optional: class distribution bar plot from metadata.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(models_dir: Path, figs_dir: Path) -> None:
    history_path = models_dir / "history.json"
    if not history_path.exists():
        print(f"[WARN] No history.json found at {history_path}; skipping training curves.")
        return

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list) or not history:
        print("[WARN] history.json is empty or malformed; skipping training curves.")
        return

    epochs = [h["epoch"] for h in history]
    train_acc = [h.get("train_acc", float("nan")) for h in history]
    val_acc = [h.get("val_acc", float("nan")) for h in history]
    train_f1 = [h.get("train_macro_f1", float("nan")) for h in history]
    val_f1 = [h.get("val_macro_f1", float("nan")) for h in history]

    figs_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = figs_dir / "train_val_accuracy.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved accuracy curves to {acc_path}")

    # Macro-F1
    plt.figure()
    plt.plot(epochs, train_f1, label="train_macro_f1")
    plt.plot(epochs, val_f1, label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Train vs Val Macro-F1")
    plt.legend()
    plt.grid(True)
    f1_path = figs_dir / "train_val_macro_f1.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved macro-F1 curves to {f1_path}")


def plot_class_distribution(dataset_dir: Path, figs_dir: Path) -> None:
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[WARN] metadata.json not found at {meta_path}; skipping class distribution plot.")
        return

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    dist = meta.get("class_distribution", None)
    if not isinstance(dist, dict) or not dist:
        print("[WARN] class_distribution missing or empty in metadata; skipping plot.")
        return

    classes = sorted(dist.keys(), key=lambda x: int(x))
    counts = [dist[c] for c in classes]

    figs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    cd_path = figs_dir / "class_distribution.png"
    plt.savefig(cd_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved class distribution plot to {cd_path}")


def copy_confusion_matrix(dataset_dir: Path, figs_dir: Path) -> None:
    models_dir = dataset_dir / "models"
    src = models_dir / "confusion_matrix.png"
    if not src.exists():
        print(f"[WARN] confusion_matrix.png not found at {src}; skipping copy.")
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    dst = figs_dir / "confusion_matrix.png"
    dst.write_bytes(src.read_bytes())
    print(f"[INFO] Copied confusion matrix figure to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate common experiment figures (training curves, confusion matrix, class distribution)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    models_dir = dataset_dir / "models"
    figs_dir = dataset_dir / "experiments" / "figures"

    plot_training_curves(models_dir, figs_dir)
    copy_confusion_matrix(dataset_dir, figs_dir)
    plot_class_distribution(dataset_dir, figs_dir)


if __name__ == "__main__":
    main()

