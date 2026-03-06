import argparse
from pathlib import Path

from src.utils import load_config, setup_logging, ensure_dir, save_metadata
from src import download, clean, label, preprocess, split


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    logger = setup_logging()

    logger.info("Loaded config from %s", config_path)

    # Prepare directories
    base_dataset_dir = Path(config["paths"]["dataset_root"])
    images_out_dir = base_dataset_dir / "images"
    splits_out_dir = base_dataset_dir / "splits"
    ensure_dir(base_dataset_dir)
    ensure_dir(images_out_dir)
    ensure_dir(splits_out_dir)

    # 1. Download / load raw data
    logger.info("Step 1/6: Downloading or loading raw data")
    weather_df = download.download_weather(config)
    urban_df = download.download_or_load_urban_proxy(config)
    sat_meta_df, raw_image_dir = download.load_satellite_metadata_and_images(config)

    # 2. Clean datasets
    logger.info("Step 2/6: Cleaning datasets")
    weather_df = clean.clean_weather(weather_df, config)
    urban_df = clean.clean_urban(urban_df, config)
    sat_meta_df, valid_image_ids, image_stats = clean.validate_and_filter_images(
        sat_meta_df, raw_image_dir, images_out_dir, config
    )

    # 3. Align data by location + date
    logger.info("Step 3/6: Aligning datasets by join keys")
    aligned_df = label.align_by_keys(sat_meta_df, weather_df, urban_df, config)

    # 4. Label into heat-risk classes
    logger.info("Step 4/6: Labelling samples based on LST")
    labelled_df = label.add_heat_risk_labels(aligned_df, config)

    # Filter to samples with valid images only
    image_id_col = config["preprocessing"]["image_id_column"]
    labelled_df = labelled_df[labelled_df[image_id_col].isin(valid_image_ids)].reset_index(drop=True)

    # 5. Preprocessing (images + tabular)
    logger.info("Step 5/6: Preprocessing images and tabular features")
    tabular_df, labels_df, preprocess_stats = preprocess.preprocess_all(
        labelled_df, images_out_dir, config
    )

    # 6. Train/val/test splits
    logger.info("Step 6/6: Creating dataset splits")
    split.create_splits(labels_df, config)

    # Save main tabular + labels
    tabular_path = base_dataset_dir / "tabular.csv"
    labels_path = base_dataset_dir / "labels.csv"
    tabular_df.to_csv(tabular_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    # Build metadata and report
    metadata = {
        "paths": {
            "dataset_root": str(base_dataset_dir),
            "images_dir": str(images_out_dir),
            "tabular_csv": str(tabular_path),
            "labels_csv": str(labels_path),
            "splits_dir": str(splits_out_dir),
        },
        "image_stats": image_stats,
        "preprocess_stats": preprocess_stats,
    }
    metadata.update(clean.compute_dataset_report(tabular_df, labels_df, image_stats))

    metadata_path = base_dataset_dir / "metadata.json"
    save_metadata(metadata, metadata_path)

    # Print short human-readable report
    clean.print_dataset_report(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build multimodal dataset for urban heat risk."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
