import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import MultimodalHeatDataset, multimodal_collate_fn
from src.model import MultimodalNet


logger = logging.getLogger("evaluate_test")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_base_tables(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tabular_path = dataset_dir / "tabular.csv"
    labels_path = dataset_dir / "labels.csv"

    if not tabular_path.exists():
        raise FileNotFoundError(f"tabular.csv not found at {tabular_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")

    tabular_df = pd.read_csv(tabular_path)
    labels_df = pd.read_csv(labels_path)

    if "image_id" not in tabular_df.columns:
        raise ValueError("tabular.csv must contain 'image_id' column.")
    if "image_id" not in labels_df.columns or "label_class" not in labels_df.columns:
        raise ValueError("labels.csv must contain 'image_id' and 'label_class' columns.")

    return tabular_df, labels_df


def detect_feature_columns(tabular_df: pd.DataFrame) -> List[str]:
    exclude_cols = {"image_id", "label_class", "lst"}
    feature_cols: List[str] = []
    for col in tabular_df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(tabular_df[col]):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in tabular.csv")
    logger.info("Using %d tabular feature columns.", len(feature_cols))
    return feature_cols


def load_test_split(dataset_dir: Path) -> pd.DataFrame:
    splits_dir = dataset_dir / "splits"
    test_path = splits_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Split file not found: {test_path}")
    df = pd.read_csv(test_path)
    if "image_id" not in df.columns:
        raise ValueError("test.csv must contain 'image_id' column.")
    return df


def build_split_dataframe(
    split_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: List[str],
    split_name: str,
) -> pd.DataFrame:
    if "label_class" in tabular_df.columns:
        base = tabular_df[["image_id", "label_class", *feature_cols]]
        merged = split_df[["image_id"]].merge(base, on="image_id", how="inner")
    else:
        base_labels = labels_df[["image_id", "label_class"]]
        merged = split_df[["image_id"]].merge(base_labels, on="image_id", how="inner")
        base_feats = tabular_df[["image_id", *feature_cols]]
        merged = merged.merge(base_feats, on="image_id", how="inner")

    if merged.empty:
        raise ValueError(f"{split_name} split join produced zero rows.")

    merged["image_id"] = merged["image_id"].astype(str)
    merged["label_class"] = merged["label_class"].astype(int)
    for col in feature_cols:
        merged[col] = merged[col].astype(float)

    class_counts = merged["label_class"].value_counts().to_dict()
    logger.info("%s split: %d rows after joins.", split_name, len(merged))
    logger.info("%s split class distribution: %s", split_name, class_counts)
    return merged


def create_test_loader(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, List[str]]:
    tabular_df, labels_df = load_base_tables(dataset_dir)
    feature_cols = detect_feature_columns(tabular_df)
    test_split = load_test_split(dataset_dir)

    images_dir = dataset_dir / "images"

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    merged_df = build_split_dataframe(
        split_df=test_split,
        tabular_df=tabular_df,
        labels_df=labels_df,
        feature_cols=feature_cols,
        split_name="test",
    )

    ds = MultimodalHeatDataset(
        df=merged_df,
        images_dir=images_dir,
        feature_cols=feature_cols,
        transform=eval_tf,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )
    return loader, feature_cols


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in loader:
        if batch is None:
            continue
        images, tabular, labels = batch
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        logits = model(images, tabular)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    return np.array(all_targets, dtype=np.int64), np.array(all_preds, dtype=np.int64)


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    models_dir: Path,
    class_labels: List[int],
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im, ax=ax)
    out_path = models_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix heatmap to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best multimodal model on test split."
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    dataset_dir = Path(args.dataset_dir)
    models_dir = dataset_dir / "models"
    best_model_path = models_dir / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")

    test_loader, feature_cols = create_test_loader(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    checkpoint = torch.load(best_model_path, map_location=device)
    tabular_input_dim = checkpoint.get("tabular_input_dim", len(feature_cols))
    num_classes = checkpoint.get("num_classes", 3)

    model = MultimodalNet(
        tabular_input_dim=tabular_input_dim,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    y_true, y_pred = run_inference(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    class_labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    report = classification_report(
        y_true, y_pred, digits=4, labels=class_labels, zero_division=0
    )

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report)

    cm_path = models_dir / "confusion_matrix.txt"
    rep_path = models_dir / "classification_report.txt"

    with cm_path.open("w", encoding="utf-8") as f:
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))

    with rep_path.open("w", encoding="utf-8") as f:
        f.write("Classification report:\n")
        f.write(report)

    logger.info("Saved confusion matrix to %s", cm_path)
    logger.info("Saved classification report to %s", rep_path)

    # Save heatmap figure
    save_confusion_matrix_heatmap(cm, models_dir, class_labels)


if __name__ == "__main__":
    main()
