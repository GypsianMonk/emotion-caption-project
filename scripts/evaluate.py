"""
Evaluation Script
------------------
Evaluate trained models on held-out test sets.

Usage:
    python scripts/evaluate.py --model emotion --config configs/emotion_config.yaml
    python scripts/evaluate.py --model captioning --config configs/captioning_config.yaml --num_samples 1000
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test sets")
    parser.add_argument(
        "--model", choices=["emotion", "captioning"], required=True,
        help="Which model to evaluate"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override checkpoint path (default: use config paths)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="Number of samples for BLEU evaluation (captioning only)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path for evaluation results"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID(s)")
    return parser.parse_args()


def evaluate_emotion(config: dict, checkpoint: str = None):
    """Evaluate emotion recognition model on FER-2013 test set."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import tensorflow as tf

    from models.emotion.trainer import EmotionTrainer

    logger = logging.getLogger(__name__)

    trainer = EmotionTrainer.from_config(config)
    trainer.prepare_data()
    trainer.build_model()

    # Load best checkpoint
    ckpt = checkpoint or f"{config['paths']['checkpoint_dir']}/best_model.h5"
    if not Path(ckpt).exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        logger.info("Train a model first: python scripts/train_emotion.py --config configs/emotion_config.yaml")
        return None

    trainer.model.load_weights(ckpt)
    logger.info(f"Loaded checkpoint: {ckpt}")

    results = trainer.evaluate()

    logger.info("=" * 50)
    logger.info(f"Test Accuracy:  {results['test_accuracy']*100:.1f}%")
    logger.info(f"Test Top-2:     {results['test_top2_accuracy']*100:.1f}%")

    per_class = results.get("classification_report", {})
    logger.info("\nPer-class F1 scores:")
    for emotion in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
        f1 = per_class.get(emotion, {}).get("f1-score", 0)
        logger.info(f"  {emotion:10s}: {f1:.3f}")

    return results


def evaluate_captioning(config: dict, num_samples: int = 500, checkpoint: str = None):
    """Evaluate captioning model BLEU scores on validation set."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import tensorflow as tf

    from data.preprocessing.coco_preprocessor import COCOPreprocessor
    from models.captioning.trainer import CaptioningTrainer

    logger = logging.getLogger(__name__)

    paths = config["paths"]

    # Build vocabulary
    preprocessor = COCOPreprocessor(
        data_dir=paths["data_dir"],
        cache_dir=paths["features_cache"],
        tokenizer_path=paths["tokenizer_path"],
        max_vocab_size=config["model"]["vocab"]["max_vocab_size"],
        max_caption_length=config["model"]["vocab"]["max_caption_length"],
        image_size=config["model"]["encoder"]["input_shape"][0],
    )
    vocab = preprocessor.build_vocabulary()
    _, val_ds = preprocessor.build_datasets(batch_size=config["training"]["batch_size"])

    # Build trainer and restore checkpoint
    trainer = CaptioningTrainer.from_config(config, vocab)
    trainer.val_ds = val_ds
    trainer.build_models()

    logger.info(f"Evaluating BLEU on {num_samples} validation samples...")
    bleu_scores = trainer.evaluate_bleu(num_samples=num_samples)

    logger.info("=" * 50)
    logger.info("BLEU Scores:")
    for k, v in bleu_scores.items():
        logger.info(f"  {k.upper()}: {v:.4f}")

    return bleu_scores


def main():
    args = parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_dir = config.get("paths", {}).get("log_dir", "logs")
    setup_logging(log_file=f"{log_dir}/evaluate.log")
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {args.model} model with config: {args.config}")

    if args.model == "emotion":
        results = evaluate_emotion(config, checkpoint=args.checkpoint)
    else:
        results = evaluate_captioning(
            config, num_samples=args.num_samples, checkpoint=args.checkpoint
        )

    if results and args.output:
        # Serialize results to JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        def make_serializable(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(make_serializable(results), f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
