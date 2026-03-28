#!/usr/bin/env python
"""
Streamlit UI for exploring REAL urban heat-risk predictions.

Features:
  - Shows global test-set metrics (classification report, baselines).
  - Lets a user pick a tile image_id from the held-out test set.
  - Displays:
      * Original RGB tile (from dataset_real/images/)
      * Optional LST heat overlay (from docs/tile_heat_overlays/)
      * True vs predicted class and confidence for that tile.

This UI does NOT fabricate metrics or run new inference; it reads REAL
results already produced by:

  - build_dataset.py   (REAL mode)
  - train_multimodal.py
  - src.evaluate_test  (which writes test_predictions.csv)
  - scripts/run_baselines.py + scripts/aggregate_results.py

Run with:

  streamlit run scripts/ui_heat_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


CLASS_NAMES = {
    0: "Low heat risk",
    1: "Medium heat risk",
    2: "High heat risk",
}


def load_test_predictions(dataset_dir: Path) -> pd.DataFrame:
    pred_path = dataset_dir / "test_predictions.csv"
    if not pred_path.exists():
        st.error(
            f"test_predictions.csv not found at {pred_path}. "
            "Run `python -m src.evaluate_test --dataset_dir dataset_real` first."
        )
        st.stop()
    df = pd.read_csv(pred_path)
    return df


def load_labels(dataset_dir: Path) -> pd.DataFrame:
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.exists():
        st.error(f"labels.csv not found at {labels_path}.")
        st.stop()
    return pd.read_csv(labels_path)


def load_results_table(dataset_dir: Path) -> pd.DataFrame | None:
    rt_path = dataset_dir / "experiments" / "results_table.csv"
    if not rt_path.exists():
        return None
    return pd.read_csv(rt_path)


def load_classification_report(dataset_dir: Path) -> str | None:
    cr_path = dataset_dir / "models" / "classification_report.txt"
    if not cr_path.exists():
        return None
    return cr_path.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="Urban Heat Risk Explorer", layout="wide")
    st.title("Multimodal Urban Heat Risk Explorer (REAL data)")

    # Sidebar: dataset selection
    st.sidebar.header("Configuration")
    dataset_dir_str = st.sidebar.text_input("Dataset directory", "dataset_real")
    dataset_dir = Path(dataset_dir_str)
    if not dataset_dir.exists():
        st.sidebar.error(f"Dataset directory not found: {dataset_dir}")
        st.stop()

    st.sidebar.info(
        "Ensure you have already run:\n"
        "1. build_dataset.py (REAL mode)\n"
        "2. train_multimodal.py\n"
        "3. python -m src.evaluate_test\n"
        "4. scripts/run_baselines.py & scripts/aggregate_results.py\n"
        "5. scripts/overlay_tile_heatmap.py (for tiles you want overlays for)"
    )

    # Load data
    preds = load_test_predictions(dataset_dir)
    labels = load_labels(dataset_dir)
    results_table = load_results_table(dataset_dir)
    clf_report = load_classification_report(dataset_dir)

    # Global metrics section
    st.subheader("Global Test-Set Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Multimodal model (CNN + MLP) – test classification report**")
        if clf_report is None:
            st.write("classification_report.txt not found.")
        else:
            st.text(clf_report)

    with col2:
        st.markdown("**Baselines on the same REAL test set**")
        if results_table is not None:
            st.dataframe(results_table, use_container_width=True)
        else:
            st.write("results_table.csv not found; run run_baselines.py + aggregate_results.py.")

    st.markdown("---")
    st.subheader("Per-Tile Exploration (Held-out Test Set)")

    # Restrict to image_ids in test_predictions (held-out tiles)
    image_ids = sorted(preds["image_id"].astype(str).unique())
    default_idx = 0
    selected_id = st.selectbox("Select a test tile (image_id)", image_ids, index=default_idx)

    # Merge true label/LST for the selected tile
    info = labels[labels["image_id"].astype(str) == selected_id]
    if info.empty:
        st.error(f"No label information found for image_id={selected_id}")
        st.stop()
    info = info.iloc[0]
    tile_id = info.get("tile_id", "NA")
    date = info.get("date", "NA")
    true_label = int(info["label_class"])
    true_name = CLASS_NAMES.get(true_label, str(true_label))
    lst_val = float(info["lst"])

    pred_row = preds[preds["image_id"].astype(str) == selected_id]
    if pred_row.empty:
        st.error(f"No prediction found for image_id={selected_id}")
        st.stop()
    pred_row = pred_row.iloc[0]
    pred_label = int(pred_row["pred_label"])
    pred_name = CLASS_NAMES.get(pred_label, str(pred_label))
    pred_conf = float(pred_row.get("pred_confidence", float("nan")))

    # Layout for images and metrics
    col_img, col_overlay, col_metrics = st.columns([1, 1, 1])

    # Original RGB tile
    images_dir = dataset_dir / "images"
    img_path = images_dir / f"{selected_id}.png"
    with col_img:
        st.markdown("**Original RGB tile**")
        if img_path.exists():
            st.image(str(img_path), caption=f"{selected_id}\n{tile_id} @ {date}", use_column_width=True)
        else:
            st.write(f"Image file not found at {img_path}")

    # Heat overlay (if generated)
    overlay_path = Path("docs") / "tile_heat_overlays" / f"{selected_id}_heat_overlay.png"
    with col_overlay:
        st.markdown("**LST heat overlay (if available)**")
        if overlay_path.exists():
            st.image(str(overlay_path), caption="RGB + REAL LST overlay", use_column_width=True)
        else:
            st.info(
                f"No overlay found for {selected_id}. "
                "Generate one with:\n"
                "  python scripts/overlay_tile_heatmap.py "
                "--image_id {selected_id}"
            )

    # Metrics for this tile
    with col_metrics:
        st.markdown("**Tile details & prediction**")
        st.write(f"**Tile ID:** {tile_id}")
        st.write(f"**Date:** {date}")
        st.write(f"**True LST (°C):** {lst_val:.2f}")
        st.write(f"**True class:** {true_label} – {true_name}")
        st.write(f"**Predicted class:** {pred_label} – {pred_name}")
        if pred_conf == pred_conf:  # not NaN
            st.write(f"**Predicted confidence (max softmax):** {pred_conf:.3f}")

        correct = true_label == pred_label
        st.write(f"**Correct prediction?** {'✅ Yes' if correct else '❌ No'}")

    st.markdown(
        "> Note: This UI works on the REAL held-out test set. "
        "Feeding completely new landmarks/images would require matching them "
        "to tiles and dates, and providing corresponding REAL weather data."
    )


if __name__ == "__main__":
    main()

