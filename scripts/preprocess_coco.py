"""
COCO Preprocessing Script
---------------------------
Builds vocabulary from COCO captions and extracts/caches InceptionV3 features.

Usage:
    python scripts/preprocess_coco.py --data_dir data/raw/coco --max_vocab 10000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.coco_preprocessor import COCOPreprocessor
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MS-COCO dataset for image captioning"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root COCO directory (with annotations/ and images/ subfolders)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="data/processed/coco_features",
        help="Directory to cache extracted InceptionV3 features"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="data/processed/tokenizer.pkl",
        help="Path to save/load vocabulary pickle"
    )
    parser.add_argument(
        "--max_vocab", type=int, default=10000,
        help="Maximum vocabulary size"
    )
    parser.add_argument(
        "--max_caption_length", type=int, default=50,
        help="Maximum caption token length"
    )
    parser.add_argument(
        "--skip_features", action="store_true",
        help="Skip feature extraction (use if already cached)"
    )
    parser.add_argument(
        "--image_size", type=int, default=299,
        help="Input image size for InceptionV3"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info(
            "Download MS-COCO from: https://cocodataset.org/#download\n"
            "Expected layout:\n"
            "  data/raw/coco/\n"
            "    ├── annotations/captions_train2017.json\n"
            "    ├── annotations/captions_val2017.json\n"
            "    └── images/train2017/ & images/val2017/"
        )
        sys.exit(1)

    preprocessor = COCOPreprocessor(
        data_dir=str(data_dir),
        cache_dir=args.cache_dir,
        tokenizer_path=args.tokenizer_path,
        max_vocab_size=args.max_vocab,
        max_caption_length=args.max_caption_length,
        image_size=args.image_size,
    )

    # Step 1: Build vocabulary
    logger.info("=" * 50)
    logger.info("Step 1: Building vocabulary...")
    vocab = preprocessor.build_vocabulary()
    logger.info(f"Vocabulary size: {len(vocab)} tokens")

    # Step 2: Extract features
    if not args.skip_features:
        logger.info("=" * 50)
        logger.info("Step 2: Extracting InceptionV3 features (may take a while)...")
        preprocessor.extract_features()
    else:
        logger.info("Step 2: Skipping feature extraction (using cached)")

    # Step 3: Validate dataset building
    logger.info("=" * 50)
    logger.info("Step 3: Validating tf.data pipeline...")
    train_ds, val_ds = preprocessor.build_datasets(batch_size=8)

    for (features, inp_tokens), tgt in train_ds.take(1):
        logger.info(f"Train batch: features={features.shape}, inp={inp_tokens.shape}, tgt={tgt.shape}")

    logger.info("✅ COCO preprocessing complete. Dataset is ready for training.")
    logger.info(f"   Run: python scripts/train_captioning.py --config configs/captioning_config.yaml")


if __name__ == "__main__":
    main()
