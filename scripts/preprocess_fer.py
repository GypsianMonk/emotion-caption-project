"""
FER-2013 Preprocessing Script
-------------------------------
Validates, summarizes, and prepares the FER-2013 dataset for training.

Usage:
    python scripts/preprocess_fer.py --data_dir data/raw/fer2013
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.fer2013_preprocessor import FER2013Preprocessor
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess FER-2013 dataset for emotion recognition training"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing fer2013.csv"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for test dataset building"
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    csv_path = Path(args.data_dir) / "fer2013.csv"
    if not csv_path.exists():
        logger.error(f"FER-2013 CSV not found at {csv_path}")
        logger.info(
            "Download from: https://www.kaggle.com/datasets/msambare/fer2013"
        )
        sys.exit(1)

    logger.info(f"Loading FER-2013 from {csv_path}")

    preprocessor = FER2013Preprocessor(
        csv_path=str(csv_path),
        augment=not args.no_augment,
    )

    # Print dataset summary
    preprocessor.summary()

    # Compute class weights
    class_weights = preprocessor.get_class_weights()
    logger.info(f"Computed class weights: {class_weights}")

    # Build datasets to validate everything works
    logger.info("Building tf.data datasets (validation run)...")
    train_ds, val_ds, test_ds = preprocessor.build_datasets(
        batch_size=args.batch_size
    )

    # Validate a single batch
    for batch_imgs, batch_labels in train_ds.take(1):
        logger.info(f"Train batch shape: images={batch_imgs.shape}, labels={batch_labels.shape}")
        logger.info(f"Image value range: [{batch_imgs.numpy().min():.3f}, {batch_imgs.numpy().max():.3f}]")

    logger.info("✅ FER-2013 preprocessing complete. Dataset is ready for training.")
    logger.info(f"   Run: python scripts/train_emotion.py --config configs/emotion_config.yaml")


if __name__ == "__main__":
    main()
