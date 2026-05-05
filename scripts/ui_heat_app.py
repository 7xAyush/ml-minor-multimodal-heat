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
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


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


def generate_overlay_if_missing(
    image_id: str,
    ee_metadata_csv: Path,
    composite_tif: Path,
    out_dir: Path,
    alpha: float = 0.5,
) -> Path | None:
    """
    Generate an LST overlay PNG for a given image_id if it does not already exist.

    This uses the same assumptions as scripts/overlay_tile_heatmap.py:
      - composite_tif bands: [1=RED,2=GREEN,3=BLUE,4=NDVI,5=LST_C]
      - ee_metadata_csv: columns tile_id,date,image_id,min_lon,min_lat,max_lon,max_lat
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}_heat_overlay.png"
    if out_path.exists():
        return out_path

    if not ee_metadata_csv.exists() or not composite_tif.exists():
        return None

    df = pd.read_csv(ee_metadata_csv)
    required = ["tile_id", "date", "image_id", "min_lon", "min_lat", "max_lon", "max_lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None

    sub = df[df["image_id"].astype(str) == str(image_id)]
    if sub.empty:
        return None
    row = sub.iloc[0]

    min_lon = float(row["min_lon"])
    min_lat = float(row["min_lat"])
    max_lon = float(row["max_lon"])
    max_lat = float(row["max_lat"])
    tile_id = str(row["tile_id"])
    date = str(row["date"])

    # Helper to normalize RGB
    def _norm_rgb(patch: np.ndarray) -> np.ndarray:
        if patch.ndim != 3 or patch.shape[2] != 3:
            return np.zeros_like(patch, dtype=np.uint8)
        if not np.isfinite(patch).any() or np.all(patch == 0):
            return np.zeros_like(patch, dtype=np.uint8)
        out = np.zeros_like(patch, dtype=np.uint8)
        for c in range(3):
            band = patch[:, :, c].astype(np.float32)
            mask = np.isfinite(band)
            if not mask.any():
                continue
            vmin = np.percentile(band[mask], 2.0)
            vmax = np.percentile(band[mask], 98.0)
            if vmax <= vmin:
                vmax = vmin + 1.0
            band = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
            out[:, :, c] = (band * 255.0).astype(np.uint8)
        return out

    try:
        with rasterio.open(composite_tif) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326",
                src.crs,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                densify_pts=2,
            )
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            rgb = src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
            lst = src.read(5, window=window, boundless=True, fill_value=np.nan)
    except Exception:
        return None

    rgb = np.transpose(rgb, (1, 2, 0))
    lst = lst.squeeze()
    rgb_uint8 = _norm_rgb(rgb)

    lst_mask = np.isfinite(lst)
    if not lst_mask.any():
        return None

    vals = lst[lst_mask]
    p5, p95 = np.percentile(vals, [5, 95])
    if p95 <= p5:
        p95 = p5 + 1.0
    lst_norm = (lst - p5) / (p95 - p5)
    lst_norm = np.clip(lst_norm, 0.0, 1.0)

    cmap = plt.cm.inferno
    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.imshow(rgb_uint8, origin="upper")
    ax.imshow(cmap(lst_norm), origin="upper", alpha=alpha)
    ax.set_title(f"{tile_id} @ {date}", fontsize=7)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


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

    # Sidebar configuration for overlays
    st.sidebar.subheader("Overlay configuration")
    ee_meta_str = st.sidebar.text_input(
        "EE metadata CSV",
        "raw/landsat/chennai_tiles_metadata.csv",
        help="Used to auto-generate LST overlays",
    )
    comp_tif_str = st.sidebar.text_input(
        "Composite GeoTIFF",
        "raw/landsat/chennai_landsat_composite.tif",
        help="Composite with RGB+NDVI+LST_C bands",
    )
    overlay_alpha = st.sidebar.slider(
        "Overlay opacity (alpha)", min_value=0.2, max_value=0.9, value=0.5, step=0.05
    )
    ee_meta_path = Path(ee_meta_str)
    comp_tif_path = Path(comp_tif_str)
    overlay_dir = Path("docs") / "tile_heat_overlays"

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
    overlay_path = overlay_dir / f"{selected_id}_heat_overlay.png"
    with col_overlay:
        st.markdown("**LST heat overlay (if available)**")
        if not overlay_path.exists():
            # Try to generate overlay on the fly.
            gen_path = generate_overlay_if_missing(
                selected_id, ee_meta_path, comp_tif_path, overlay_dir, alpha=overlay_alpha
            )
            if gen_path is not None and gen_path.exists():
                overlay_path = gen_path
        if overlay_path.exists():
            st.image(str(overlay_path), caption="RGB + REAL LST overlay", use_column_width=True)
        else:
            st.info(
                "No overlay available and automatic generation failed. "
                "Check EE metadata / composite paths in the sidebar."
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

