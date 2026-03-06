import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


logger = logging.getLogger("plot_history")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_history(history_path: Path) -> List[Dict]:
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found at {history_path}")
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError("history.json must contain a list of epoch records.")
    return history


def plot_curves(
    history: List[Dict],
    output_dir: Path,
) -> None:
    epochs = [h["epoch"] for h in history]

    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    train_f1 = [h["train_macro_f1"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    loss_path = output_dir / "history_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = output_dir / "history_accuracy.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Macro-F1
    plt.figure()
    plt.plot(epochs, train_f1, label="train_macro_f1")
    plt.plot(epochs, val_f1, label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Train vs Val Macro-F1")
    plt.legend()
    plt.grid(True)
    f1_path = output_dir / "history_macro_f1.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved loss plot to %s", loss_path)
    logger.info("Saved accuracy plot to %s", acc_path)
    logger.info("Saved macro-F1 plot to %s", f1_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training history curves (loss, accuracy, macro-F1)."
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    args = parser.parse_args()

    setup_logging()

    dataset_dir = Path(args.dataset_dir)
    models_dir = dataset_dir / "models"
    history_path = models_dir / "history.json"

    history = load_history(history_path)
    plot_curves(history, models_dir)


if __name__ == "__main__":
    main()
