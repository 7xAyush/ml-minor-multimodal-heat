# Testing / Smoke Tests

This repository does not ship an automated test suite, but you can run a small
set of **manual smoke tests** to verify that both synthetic and REAL modes work
as expected.

All commands are assumed to be run from the project root.

## 1. Synthetic-mode smoke test

Goal: exercise the end-to-end **synthetic Kaggle demo** pipeline.

1. Create a small Kaggle-like folder with a few images:

   ```bash
   mkdir -p raw/kaggle_source
   # Create or copy 3+ small jpg/png images into raw/kaggle_source/
   ```

2. Create a synthetic-config file (e.g. `config_synth_test.yaml`) with:

   ```yaml
   paths:
     dataset_root: "dataset_synth_test"
     raw_satellite_dir: "raw/satellite_images"
     satellite_metadata_csv: "raw/satellite_metadata.csv"
     weather_csv: "raw/weather.csv"
     urban_proxy_csv: "raw/urban_proxy.csv"
     weather_download_csv: "raw/weather_downloaded.csv"
     kaggle_source_dir: "raw/kaggle_source"

   sources:
     lst: "synthetic_demo"
     imagery: "kaggle"
     weather: "open_meteo"
     urban: "synthetic_constant"

   grid:
     tile_size_m: 1000
     max_tiles: 10

   city:
     name: "ExampleCity"
     bbox: [12.0, 77.0, 13.0, 78.0]

   time:
     start_date: "2022-01-01"
     end_date: "2022-01-10"

   heat_risk:
     low_threshold: 30.0
     high_threshold: 35.0

   images:
     size: [224, 224]
     augment: false

   preprocessing:
     join_keys: ["date", "tile_id"]
     image_id_column: "image_id"
     lst_column: "lst"
     label_column: "label_class"
     exclude_from_scaling: ["image_id", "tile_id", "date", "label_class", "lst"]

   mode:
     data_mode: "synthetic"
     use_kaggle_source: true
     demo_labels_if_missing: true

   splits:
     train_ratio: 0.7
     val_ratio: 0.15
     test_ratio: 0.15
     random_state: 42
   ```

3. Run the synthetic pipeline:

   ```bash
   python build_dataset.py --config config_synth_test.yaml
   ```

Expected behavior:

- Logs mention:
  - `Using data_mode='synthetic'`
  - `Using Kaggle source ingestion flow for satellite data.`
  - Warning: `LST is synthetic demo label from brightness; ...`
- `dataset_synth_test/` is created with:
  - `images/`, `tabular.csv`, `labels.csv`, `splits/train.csv`, `val.csv`, `test.csv`.
  - `metadata.json` whose `"mode"` has `"data_mode": "synthetic"`.

## 2. REAL-mode smoke test

Goal: exercise the **REAL mode** path with a tiny, structurally valid dataset.

1. Create a minimal real metadata file:

   ```bash
   mkdir -p raw/satellite_images
   ```

   Create `raw/satellite_metadata.csv`:

   ```csv
   image_id,date,tile_id,lst
   tile_r0_c0_2022-01-01,2022-01-01,tile_r0_c0,32.5
   ```

   And a dummy image at `raw/satellite_images/tile_r0_c0_2022-01-01.png` (32x32 RGB).

2. Create a matching weather file (optional; otherwise Open-Meteo will be used):

   ```csv
   # raw/weather.csv
   date,tile_id,temperature_2m,relativehumidity_2m,windspeed_10m
   2022-01-01,tile_r0_c0,30.0,50.0,2.0
   ```

3. In `config.yaml` (or a separate config), set:

   ```yaml
   mode:
     data_mode: "real"
     use_kaggle_source: true
     demo_labels_if_missing: true

   paths:
     dataset_root: "dataset_real_test"
     raw_satellite_dir: "raw/satellite_images"
     satellite_metadata_csv: "raw/satellite_metadata.csv"
     weather_csv: "raw/weather.csv"
   ```

4. Run:

   ```bash
   python build_dataset.py --config config.yaml
   ```

Expected behavior:

- Logs mention:
  - `Using data_mode='real'`
  - `Running REAL data pipeline (no synthetic label generation).`
  - `Loaded real satellite metadata with 1 rows`
  - `Aligned dataset has 1 rows after join`
- `dataset_real_test/metadata.json` has `"mode": {"data_mode": "real", ...}` and `split_sizes.train` reported as 1 with `val`/`test` possibly empty.

## 3. Failure: missing bounds in EE metadata

Goal: verify `scripts/prepare_real_inputs.py` enforces bounds columns.

1. Create `raw/landsat/bad_tiles_metadata_missing_bounds.csv`:

   ```csv
   tile_id,date,image_id,lst
   tile_r0_c0,2022-01-01,tile_r0_c0_2022-01-01,32.5
   ```

2. Create a dummy composite TIFF (any small GeoTIFF) at `raw/landsat/dummy_composite.tif`.

3. Run:

   ```bash
   python scripts/prepare_real_inputs.py \
     --ee_metadata_csv raw/landsat/bad_tiles_metadata_missing_bounds.csv \
     --composite_tif raw/landsat/dummy_composite.tif \
     --output_metadata_csv raw/satellite_metadata.csv \
     --output_images_dir raw/satellite_images
   ```

Expected behavior:

- Script aborts with a clear error:

  ```text
  [ERROR] EE metadata CSV is missing required columns (bounds are mandatory for robust cropping): ['min_lon', 'min_lat', 'max_lon', 'max_lat']
  ```

## 4. Failure: missing required REAL metadata columns

1. Create `raw/satellite_metadata_missing_lst.csv`:

   ```csv
   image_id,date,tile_id
   tile_r0_c0_2022-01-01,2022-01-01,tile_r0_c0
   ```

2. Point `paths.satellite_metadata_csv` in `config.yaml` to this file and set `mode.data_mode: "real"`.

3. Run:

   ```bash
   python build_dataset.py --config config.yaml
   ```

Expected behavior:

- REAL mode fails early with a `KeyError` similar to:

  ```text
  Satellite metadata CSV at 'raw/satellite_metadata_missing_lst.csv' is missing required columns: ['lst']
  ```

## 5. Failure: tile_id/date alignment mismatch

1. Create `raw/satellite_metadata.csv`:

   ```csv
   image_id,date,tile_id,lst
   tile_r0_c0_2022-01-01,2022-01-01,tile_r0_c0,32.0
   ```

2. Create `raw/weather.csv` with a different `tile_id`:

   ```csv
   date,tile_id,temperature_2m,relativehumidity_2m,windspeed_10m
   2022-01-01,tile_r0_c1,30.0,50.0,2.0
   ```

3. Set `mode.data_mode: "real"` and point `paths.satellite_metadata_csv` / `paths.weather_csv` accordingly.

4. Run:

   ```bash
   python build_dataset.py --config config.yaml
   ```

Expected behavior:

- REAL mode logs the alignment attempt, then fails with:

  ```text
  ValueError: Aligned dataset is empty after joining satellite metadata with weather on ['date', 'tile_id']. Check that your tile_id/date values match between the two sources.
  ```

