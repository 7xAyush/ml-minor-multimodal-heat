# One-City REAL Mode Run Guide

This guide shows how to run a **single-city REAL experiment** end-to-end using
Landsat L2 surface temperature from Google Earth Engine (GEE) plus the local
conversion script `scripts/prepare_real_inputs.py`.

The goal is to produce:

- `raw/satellite_metadata.csv`
- `raw/satellite_images/<image_id>.png`

so that REAL mode (`mode.data_mode: "real"`) can build the dataset and train
the multimodal model.

## 1. In Google Earth Engine (Code Editor)

1. Open https://code.earthengine.google.com and create a new script.
2. Paste the Earth Engine JavaScript from the repository (see `docs/REAL_CITY_LANDSAT_EE.js` snippet below).
3. Set:
   - `aoi` to your city polygon or bounding box.
   - `startDate` / `endDate` to a short hot-season window (e.g., 4–8 weeks).
   - `cityName` (used in export names).
4. Hit **Run** to preview tiles and LST.
5. In the **Tasks** pane, start both exports:
   - `<cityName>_tiles_metadata` (CSV, to Google Drive)
   - `<cityName>_landsat_composite` (GeoTIFF, to Google Drive)
6. Wait for both exports to complete.

## 2. Download EE exports into the repo

From your Google Drive folder (e.g. `gee_urban_heat`):

1. Download the tile metadata CSV, e.g. `ExampleCity_tiles_metadata.csv`.
2. Download the composite TIFF, e.g. `ExampleCity_landsat_composite.tif`.
3. Place them in your repo:

   ```text
   raw/landsat/ExampleCity_tiles_metadata.csv
   raw/landsat/ExampleCity_landsat_composite.tif
   ```

## 3. Run local conversion to REAL inputs

Activate your environment and run:

```bash
python scripts/prepare_real_inputs.py \
  --ee_metadata_csv raw/landsat/ExampleCity_tiles_metadata.csv \
  --composite_tif raw/landsat/ExampleCity_landsat_composite.tif \
  --output_metadata_csv raw/satellite_metadata.csv \
  --output_images_dir raw/satellite_images \
  --tile_size_m 1000 \
  --image_size 224
```

This will:

- Crop one RGB chip per tile using EE-provided bounds.
- Normalize and resize chips to `224x224` and write:
  - `raw/satellite_images/<image_id>.png`
- Produce REAL metadata:
  - `raw/satellite_metadata.csv` with `image_id,date,tile_id,lst` (+ extras such as `lat,lon,NDVI_mean` if present).

The script will fail loudly if required columns (including `min_lon/min_lat/max_lon/max_lat`) are missing or invalid.

## 4. Configure REAL mode

In `config.yaml`, set:

```yaml
mode:
  data_mode: "real"
  use_kaggle_source: true        # ignored in REAL mode
  demo_labels_if_missing: true   # ignored in REAL mode

paths:
  dataset_root: "dataset"
  raw_satellite_dir: "raw/satellite_images"
  satellite_metadata_csv: "raw/satellite_metadata.csv"
  weather_csv: "raw/weather.csv"        # optional; otherwise Open-Meteo is used
  urban_proxy_csv: "raw/urban_proxy.csv"  # optional; not required for first run

sources:
  lst: "landsat_c2_st"
  imagery: "landsat_c2_sr"
  weather: "open_meteo"
  urban: "none"
```

If you do not provide `raw/weather.csv`, REAL mode will call Open-Meteo using
the grid defined in `grid`/`city`/`time` and align weather on `["date","tile_id"]`.

## 5. Build the REAL dataset

Run:

```bash
python build_dataset.py --config config.yaml
```

Expected outputs (under `dataset/`):

- `images/` – resized chips
- `tabular.csv` – weather (and any extra numeric features)
- `labels.csv` – `image_id,tile_id,date,label_class,lst`
- `splits/train.csv`, `val.csv`, `test.csv`
- `metadata.json` – includes:
  - `"mode": {"data_mode": "real", "lst_source": "landsat_c2_st", ...}`
  - dataset stats and `split_sizes`

## 6. Train and evaluate the multimodal model

Train:

```bash
python train_multimodal.py --dataset_dir dataset
```

Evaluate and plot (optional):

```bash
python -m src.evaluate_test --dataset_dir dataset
python -m src.plot_history --dataset_dir dataset
```

This will train the existing multimodal model (ResNet18 image encoder + tabular
MLP) on REAL, Landsat-derived LST labels and weather features for your city.

