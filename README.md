# Climate-Aware Urban Planning: Multimodal Dataset Builder

This project builds an end-to-end **multimodal dataset** for urban heat risk prediction using:

- Satellite image tiles (CNN input)
- Daily weather features per tile (tabular)
- Urban proxy per tile (tabular)

Each sample corresponds to one `(date, tile_id, image_id)` triple, labelled into 3 heat-risk classes from Land Surface Temperature (LST).

## 1. Setup

### 1.1. Python environment

Requirements:

- Python 3.10+

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1.2. Quickstart (REAL mode, manual or EE-derived images)

1. Place satellite images under `raw/satellite_images/` with filenames matching `image_id` (e.g. `img_0001.png`).
2. Create `raw/satellite_metadata.csv` with columns: `image_id,date,lst,tile_id`.
3. (Optional) Provide `raw/weather.csv` and/or `raw/urban_proxy.csv`. If omitted, they will be generated/downloaded.
4. Ensure in `config.yaml`:
   - `mode.data_mode: "real"`
5. Run:

```bash
python build_dataset.py --config config.yaml
```

### 1.3. Quickstart (synthetic Kaggle demo mode)

1. Download a Kaggle dataset containing satellite images (and optionally a CSV with LST/temperature labels).
2. Unzip it under `raw/kaggle_source/` so you have images and CSVs inside that folder.
3. Ensure in `config.yaml`:
   - `mode.use_kaggle_source: true`
   - `paths.kaggle_source_dir: "raw/kaggle_source"`
4. Ensure in `config.yaml`:
   - `mode.data_mode: "synthetic"`
5. Run:

```bash
python build_dataset.py --config config.yaml
```

The pipeline will:

- Discover images recursively in `raw/kaggle_source/`.
- Discover a CSV with LST/temperature labels if present and join by filename.
- Auto-generate `raw/satellite_images/` and `raw/satellite_metadata.csv`.
- Then run the usual cleaning, alignment, labelling, preprocessing, and splitting.

If no LST labels are found, **demo labels** are generated from image pixel statistics (not scientifically valid) and a clear warning is logged.

## 2. Configuration

Edit `config.yaml` to match your city, date range, mode and file locations.

Key sections:

- `paths.dataset_root`: where the processed dataset will be written.
- `paths.raw_satellite_dir`: directory with raw satellite images (used after Kaggle ingestion or manual placement).
- `paths.satellite_metadata_csv`: CSV with satellite metadata and LST values. Must include:
  - `image_id`
  - `date` (YYYY-MM-DD)
  - `lst` (Land Surface Temperature in °C)
  - `tile_id` used in `preprocessing.join_keys`.
- `paths.weather_csv`: optional pre-downloaded weather CSV. If missing, the script will call the Open-Meteo API per tile using the configured bounding box and date range and save daily aggregates to `paths.weather_download_csv`.
- `paths.urban_proxy_csv`: CSV containing urban proxy features per `tile_id` (e.g. population density or building density). If missing, a synthetic proxy will be generated.
- `paths.kaggle_source_dir`: root of the unzipped Kaggle dataset (used when `mode.use_kaggle_source` is true).
- `grid`: tile generation parameters (`tile_size_m`, `max_tiles`).
- `city` / `time`: city name, bounding box and date range for weather download.
- `heat_risk`: LST thresholds defining the three heat-risk classes.
- `images.size`: target image size (e.g. `[224, 224]`).
- `preprocessing.join_keys`: columns used to join satellite and weather data (default: `["date", "tile_id"]`; urban joins on `tile_id` only).
- `mode.use_kaggle_source`: if true, use Kaggle ingestion instead of expecting manual `satellite_images` and `satellite_metadata.csv`.
- `mode.demo_labels_if_missing`: if true, generate demo LST labels when none are found.

## 3. Expected Inputs

### Manual mode

You provide:

- A folder of satellite images at `paths.raw_satellite_dir`.
- A `satellite_metadata.csv` at `paths.satellite_metadata_csv` describing each image, with at least:
  - `image_id` (matching the image filename without extension)
  - `date`
  - `lst` (ground-truth LST in °C)
  - `tile_id` key compatible with weather and urban proxy data.
- (Optional) A `urban_proxy.csv` at `paths.urban_proxy_csv` with:
  - `tile_id`
  - one or more urban proxy features (e.g. `population_density`, `building_density`).

### Kaggle mode

You provide:

- An unzipped Kaggle dataset under `raw/kaggle_source/` containing:
  - Image files (png/jpg/jpeg/tif) somewhere in the directory tree.
  - Optionally, one or more CSVs with LST/temperature labels and dates.

The pipeline will:

- Discover images and copy/convert them into `raw/satellite_images/` as PNG.
- Auto-generate `raw/satellite_metadata.csv` with `image_id,date,lst,tile_id`.

Weather data:

- If `paths.weather_csv` exists, it will be used (and aggregated to daily per `date,tile_id` if hourly).
- Otherwise, data is downloaded from the **Open-Meteo historical API** for each tile centroid and aggregated hourly → daily.

## 4. Running the Pipeline

From the project root:

```bash
python build_dataset.py --config config.yaml
```

This will:

1. Download/load weather and urban proxy data.
2. Load satellite metadata and images (from Kaggle or manual sources).
3. Clean data (missing values, outliers, corrupt/duplicate images).
4. Align data across modalities:
   - Satellite + weather joined on `["date", "tile_id"]`.
   - Urban proxy joined on `["tile_id"]` only (static per tile).
5. Label samples into three heat-risk classes based on LST:
   - class 0: `LST < 30°C`
   - class 1: `30°C ≤ LST < 35°C`
   - class 2: `LST ≥ 35°C`
6. Preprocess images (resize to 224x224; normalization is typically applied at training time) and tabular features (scale numeric features, encode categoricals).
7. Export:
   - `dataset/images/` — resized images.
   - `dataset/tabular.csv` — processed weather + urban features per sample.
   - `dataset/labels.csv` — `image_id`, `tile_id`, `date`, `label_class`, `lst`.
   - `dataset/splits/train.csv`, `dataset/splits/val.csv`, `dataset/splits/test.csv` — stratified (70/15/15) on `label_class` when the class distribution allows; otherwise random split with warning.
   - `dataset/metadata.json` — schema + basic stats + data quality report.

## 5. Outputs

After running, the `dataset/` folder will look like:

- `dataset/images/` — final image files (`<image_id>.png`).
- `dataset/tabular.csv` — one row per `(date, tile_id, image_id)` with aligned and preprocessed features.
- `dataset/labels.csv` — mapping from `(image_id, tile_id, date)` to `label_class` and `lst`.
- `dataset/splits/train.csv` — training split.
- `dataset/splits/val.csv` — validation split.
- `dataset/splits/test.csv` — test split.
- `dataset/metadata.json` — summary statistics and configuration snapshot.

## 6. Modes: REAL vs Synthetic

- **REAL mode (`mode.data_mode: "real"`)**
  - Expects `raw/satellite_metadata.csv` with real LST values and tile assignments.
  - Expects geospatially aligned image chips in `raw/satellite_images/` (e.g., from Landsat via Google Earth Engine + `scripts/prepare_real_inputs.py`).
  - Uses real weather (from `raw/weather.csv` or Open-Meteo) aligned by `["date","tile_id"]`.
  - Does **not** generate any synthetic brightness-based LST labels.

- **Synthetic mode (`mode.data_mode: "synthetic"`)**
  - Uses Kaggle ingestion (`mode.use_kaggle_source: true`) to auto-generate `satellite_metadata.csv` and images from a Kaggle dataset.
  - May generate **demo** LST labels from image brightness if no labels are present. These are explicitly non-scientific and for pipeline demos only.

## 7. Notes & Extensions

- You can plug in any satellite dataset (e.g., Landsat, Sentinel) as long as you provide:
  - Image files.
  - A metadata CSV with LST and `tile_id` join keys, or a Kaggle dataset that can be auto-ingested.
- For urban proxy data, any suitable proxy (population density, building footprint density, NDVI, etc.) can be used, provided it is tabular and linked by `tile_id`.
- You can train the multimodal model on the built dataset using:
  - `python train_multimodal.py --dataset_dir dataset`
- Synthetic mode is for demonstrations only; REAL mode with physically meaningful LST labels should be used for scientific evaluation.
