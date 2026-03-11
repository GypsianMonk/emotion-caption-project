"""
CLI Training Script — Emotion Recognition
------------------------------------------
python scripts/train_emotion.py --config configs/emotion_config.yaml [--resume]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.emotion.trainer import EmotionTrainer
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EmotionCNN on FER-2013 dataset"
    )
    parser.add_argument(
        "--config", type=str, default="configs/emotion_config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training, only evaluate best checkpoint"
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="GPU device ID(s) to use (e.g. '0' or '0,1')"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # GPU configuration
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup logging
    setup_logging(log_file=f"{config['paths']['log_dir']}/train.log")
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {args.config}")
    logger.info(f"GPUs: {[g.name for g in gpus]}")

    # Build trainer
    trainer = EmotionTrainer.from_config(config)

    # Prepare data
    logger.info("Preparing FER-2013 dataset...")
    trainer.prepare_data()

    # Build model
    trainer.build_model()

    if args.resume:
        ckpt_path = f"{config['paths']['checkpoint_dir']}/last_model.h5"
        if Path(ckpt_path).exists():
            trainer.model.load_weights(ckpt_path)
            logger.info(f"Resumed from: {ckpt_path}")

    # Train
    if not args.eval_only:
        logger.info("Starting training...")
        history = trainer.train()

    # Evaluate
    logger.info("Evaluating on test set...")
    results = trainer.evaluate()
    trainer.save_training_report(results)

    logger.info("=" * 50)
    logger.info(f"Final Test Accuracy: {results['test_accuracy']*100:.1f}%")
    logger.info(f"Final Test Top-2:    {results['test_top2_accuracy']*100:.1f}%")

    per_class = results["classification_report"]
    logger.info("\nPer-class F1 scores:")
    for emotion in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
        f1 = per_class.get(emotion, {}).get("f1-score", 0)
        logger.info(f"  {emotion:10s}: {f1:.3f}")


if __name__ == "__main__":
    main()
