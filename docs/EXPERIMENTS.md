# Experiment Workflow (One-City REAL Mode)

This document describes how to run a complete experiment using the REAL-mode
pipeline for a single city.

## Prerequisites

- A REAL dataset has been built under `dataset/` using:

  ```bash
  python build_dataset.py --config config.yaml
  ```

- `config.yaml` is configured with:

  ```yaml
  mode:
    data_mode: "real"
  ```

- REAL inputs (`raw/satellite_metadata.csv` and `raw/satellite_images/`) have
  been prepared, e.g. via the Earth Engine + local conversion workflow in
  `docs/REAL_CITY_RUN.md`.

## 1. Train and evaluate the multimodal model

Run:

```bash
python scripts/run_experiment.py --dataset_dir dataset
```

This will:

- Call `train_multimodal.py` to train the multimodal model (image + tabular)
  on `dataset/`.
- Call `src.evaluate_test` to evaluate the best model on the held-out test
  split.
- Save:
  - `dataset/models/best.pt`
  - `dataset/models/history.json`
  - `dataset/models/confusion_matrix.txt`
  - `dataset/models/confusion_matrix.png`
  - `dataset/models/classification_report.txt`

## 2. Run baselines

Run:

```bash
python scripts/run_baselines.py --dataset_dir dataset
```

This will train and evaluate:

- A **tabular-only** baseline (RandomForest) on weather [+ NDVI if present].
- A **majority-class** baseline.

Results are saved to:

- `dataset/experiments/baselines.json`

## 3. Run ablations

Run:

```bash
python scripts/run_ablations.py --dataset_dir dataset
```

This will:

- Record **multimodal** results from `history.json` and
  `classification_report.txt`.
- Reuse the **tabular-only** baseline metrics.
- Add a placeholder entry describing the not-yet-implemented image-only model.

Results are saved to:

- `dataset/experiments/ablations.json`

## 4. Aggregate metrics into a table

Run:

```bash
python scripts/aggregate_results.py --dataset_dir dataset
```

This will read `baselines.json` and `ablations.json` and produce:

- `dataset/experiments/results_table.csv`

Columns:

- `Model`
- `Input Type`
- `Accuracy`
- `Macro-F1`

## 5. Generate figures

Run:

```bash
python scripts/generate_figures.py --dataset_dir dataset
```

This will:

- Generate training curves from `history.json`:
  - `train_val_accuracy.png`
  - `train_val_macro_f1.png`
- Copy `models/confusion_matrix.png` into experiments figures.
- Plot class distribution from `metadata.json`.

All figures are saved under:

- `dataset/experiments/figures/`

## Notes on Safety

- These scripts assume the dataset was built in **REAL mode**. If
  `dataset/metadata.json` reports `data_mode: "synthetic"`, the experiment
  scripts will warn you. Synthetic-mode results are useful for debugging but
  **not** for scientific claims about real urban heat.

