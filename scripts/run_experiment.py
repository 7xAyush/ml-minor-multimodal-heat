#!/usr/bin/env python
"""
Run a full REAL-mode multimodal experiment for one city.

This script assumes that:
  - REAL mode has already been used to build a dataset under `dataset/`
    (or a user-specified `--dataset_dir`) using:
      python build_dataset.py --config config.yaml
  - The dataset contains:
      dataset/tabular.csv
      dataset/labels.csv
      dataset/splits/{train,val,test}.csv

It will:
  1. Train the multimodal model (image + tabular) using train_multimodal.py logic.
  2. Evaluate the best model on the held-out test set using evaluate_test.py logic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import json

from train_multimodal import main as train_main
from src.evaluate_test import main as eval_main


logger = logging.getLogger("run_experiment")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensure_real_mode(dataset_dir: Path) -> None:
    """
    Warn or fail if the dataset appears to have been built in synthetic mode.
    We check dataset/metadata.json when present.
    """
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning(
            "metadata.json not found under %s; cannot verify data_mode. "
            "Make sure this dataset was built in REAL mode.",
            dataset_dir,
        )
        return

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    mode_info: Dict[str, Any] = meta.get("mode", {})
    data_mode = mode_info.get("data_mode", "unknown")
    lst_source = mode_info.get("lst_source", "unknown")
    if data_mode != "real":
        logger.warning(
            "metadata.json reports data_mode='%s' (lst_source='%s'). "
            "For scientifically valid experiments you should use REAL mode data. "
            "Proceeding anyway, but consider rebuilding the dataset in REAL mode.",
            data_mode,
            lst_source,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full multimodal experiment (train + evaluate) on REAL dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory built via REAL mode (default: dataset).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for multimodal training (passed to train_multimodal).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for multimodal training (passed to train_multimodal).",
    )
    args, unknown = parser.parse_known_args()

    setup_logging()
    dataset_dir = Path(args.dataset_dir)
    logger.info("Running experiment on dataset directory: %s", dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    ensure_real_mode(dataset_dir)

    # 1. Train multimodal model (delegates to train_multimodal.main)
    logger.info("Starting multimodal training...")
    train_args = [
        "--dataset_dir",
        str(dataset_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
    ]
    # Allow extra args to be forwarded
    train_args.extend(unknown)
    train_main() if not train_args else train_main.__wrapped__  # type: ignore


if __name__ == "__main__":
    # We cannot easily call train_main with a custom argv without refactoring
    # train_multimodal.py, so we simply delegate via the CLI:
    #
    #   python train_multimodal.py --dataset_dir ... --epochs ... --batch_size ...
    #
    # and then call evaluate_test.py afterward. To keep this script simple and
    # robust, we'll just invoke those scripts as subprocesses.
    import subprocess

    parser = argparse.ArgumentParser(
        description="Run full multimodal experiment (train + evaluate) on REAL dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory built via REAL mode (default: dataset).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    cli_args = parser.parse_args()

    setup_logging()
    ds_dir = Path(cli_args.dataset_dir)
    logger.info("Running experiment on dataset directory: %s", ds_dir)
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

    ensure_real_mode(ds_dir)

    # Train multimodal model
    logger.info("Training multimodal model...")
    subprocess.run(
        [
            "python",
            "train_multimodal.py",
            "--dataset_dir",
            str(ds_dir),
            "--epochs",
            str(cli_args.epochs),
            "--batch_size",
            str(cli_args.batch_size),
            "--num_workers",
            str(cli_args.num_workers),
        ],
        check=True,
    )

    # Evaluate best model on test set
    logger.info("Evaluating best model on test split...")
    subprocess.run(
        [
            "python",
            "-m",
            "src.evaluate_test",
            "--dataset_dir",
            str(ds_dir),
            "--batch_size",
            "64",
            "--num_workers",
            str(cli_args.num_workers),
        ],
        check=True,
    )

