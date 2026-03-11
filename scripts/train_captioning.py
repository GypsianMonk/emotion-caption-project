"""
CLI Training Script — Image Captioning
----------------------------------------
python scripts/train_captioning.py --config configs/captioning_config.yaml

Steps:
  1. Build vocabulary from COCO train captions
  2. Extract & cache InceptionV3 features for all images (one-time)
  3. Build tf.data pipelines
  4. Train encoder projector + LSTM decoder with teacher forcing
  5. Evaluate BLEU-4 on validation set
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.coco_preprocessor import COCOPreprocessor
from models.captioning.trainer import CaptioningTrainer
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train InceptionV3+LSTM captioning model")
    parser.add_argument("--config", type=str, default="configs/captioning_config.yaml")
    parser.add_argument("--skip_feature_extraction", action="store_true",
                        help="Skip feature extraction (use if features already cached)")
    parser.add_argument("--eval_samples", type=int, default=500,
                        help="Number of samples for BLEU evaluation")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def main():
    args = parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    setup_logging(log_file=f"{paths['log_dir']}/train.log")
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {args.config}")

    # === Step 1: Preprocessing ===
    logger.info("Setting up COCO preprocessor...")
    preprocessor = COCOPreprocessor(
        data_dir=paths["data_dir"],
        cache_dir=paths["features_cache"],
        tokenizer_path=paths["tokenizer_path"],
        max_vocab_size=config["model"]["vocab"]["max_vocab_size"],
        max_caption_length=config["model"]["vocab"]["max_caption_length"],
        image_size=config["model"]["encoder"]["input_shape"][0],
    )

    # === Step 2: Build vocabulary ===
    logger.info("Building vocabulary...")
    vocab = preprocessor.build_vocabulary()
    logger.info(f"Vocabulary size: {len(vocab)}")

    # === Step 3: Extract features ===
    if not args.skip_feature_extraction:
        logger.info("Extracting InceptionV3 features (this runs once and is cached)...")
        preprocessor.extract_features()
    else:
        logger.info("Skipping feature extraction (using cached features)")

    # === Step 4: Build datasets ===
    logger.info("Building tf.data datasets...")
    train_ds, val_ds = preprocessor.build_datasets(
        batch_size=config["training"]["batch_size"]
    )

    # === Step 5: Train ===
    trainer = CaptioningTrainer.from_config(config, vocab)
    trainer.train_ds = train_ds
    trainer.val_ds = val_ds
    trainer.build_models()

    logger.info("Starting training...")
    history = trainer.train()

    # === Step 6: Evaluate BLEU ===
    logger.info(f"Evaluating BLEU on {args.eval_samples} validation samples...")
    bleu_scores = trainer.evaluate_bleu(num_samples=args.eval_samples)

    logger.info("=" * 50)
    logger.info("Training complete. Final metrics:")
    for k, v in bleu_scores.items():
        logger.info(f"  {k.upper()}: {v:.4f}")
    logger.info(f"  Target BLEU-4: 0.28")


if __name__ == "__main__":
    main()
