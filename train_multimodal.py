import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import MultimodalHeatDataset, multimodal_collate_fn
from src.model import MultimodalNet


logger = logging.getLogger("multimodal_train")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_splits(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    splits_dir = dataset_dir / "splits"
    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        path = splits_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        df = pd.read_csv(path)
        if "image_id" not in df.columns:
            raise ValueError(f"{split_name}.csv must contain 'image_id' column.")
        splits[split_name] = df
    return splits


def build_split_dataframe(
    split_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: List[str],
    split_name: str,
) -> pd.DataFrame:
    # Ensure each split row has label_class and features.
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


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    tabular_df, labels_df = load_base_tables(dataset_dir)
    feature_cols = detect_feature_columns(tabular_df)
    splits = load_splits(dataset_dir)

    images_dir = dataset_dir / "images"

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    loaders: Dict[str, DataLoader] = {}
    for split_name in ["train", "val", "test"]:
        merged_df = build_split_dataframe(
            split_df=splits[split_name],
            tabular_df=tabular_df,
            labels_df=labels_df,
            feature_cols=feature_cols,
            split_name=split_name,
        )
        ds = MultimodalHeatDataset(
            df=merged_df,
            images_dir=images_dir,
            feature_cols=feature_cols,
            transform=train_tf if split_name == "train" else eval_tf,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            collate_fn=multimodal_collate_fn,
        )

    logger.info("Number of tabular feature columns: %d", len(feature_cols))
    return loaders["train"], loaders["val"], loaders["test"], len(feature_cols)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in loader:
        if batch is None:
            continue
        images, tabular, labels = batch
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, tabular)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    if not all_targets:
        return float("nan"), float("nan"), float("nan")

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
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
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    if not all_targets:
        return float("nan"), float("nan"), float("nan")

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multimodal (image + tabular) model for heat-risk classification."
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    dataset_dir = Path(args.dataset_dir)
    models_dir = dataset_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, input_dim = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = MultimodalNet(tabular_input_dim=input_dim, num_classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_model_path = models_dir / "best.pt"
    history: List[Dict[str, float]] = []

    logger.info("Starting training for %d epochs", args.epochs)
    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        logger.info(
            "Epoch %d: train_loss=%.4f train_acc=%.4f train_macro_f1=%.4f "
            "val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_macro_f1": float(train_f1),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_f1),
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tabular_input_dim": input_dim,
                    "num_classes": 3,
                    "epoch": epoch,
                },
                best_model_path,
            )
            logger.info(
                "Saved new best model to %s (val_macro_f1=%.4f)",
                best_model_path,
                val_f1,
            )

    history_path = models_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", history_path)

    logger.info("Evaluating best model on test set.")
    checkpoint = torch.load(best_model_path, map_location=device)
    best_model = MultimodalNet(
        tabular_input_dim=checkpoint["tabular_input_dim"],
        num_classes=checkpoint["num_classes"],
    )
    best_model.load_state_dict(checkpoint["model_state"])
    best_model.to(device)

    test_loss, test_acc, test_f1 = evaluate(
        best_model, test_loader, criterion, device
    )
    logger.info(
        "Test metrics using best model: loss=%.4f acc=%.4f macro_f1=%.4f",
        test_loss,
        test_acc,
        test_f1,
    )


if __name__ == "__main__":
    main()
