"""
Emotion Model Trainer
----------------------
Production training loop for the FER-2013 emotion recognition CNN.
Includes callbacks, TensorBoard logging, checkpoint management, and
post-training evaluation with per-class metrics.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.emotion.cnn_model import build_emotion_cnn, compile_emotion_model
from data.preprocessing.fer2013_preprocessor import FER2013Preprocessor, EMOTION_LABELS

logger = logging.getLogger(__name__)


class EmotionTrainer:
    """
    Manages the full training lifecycle for the EmotionCNN model.

    Usage:
        trainer = EmotionTrainer.from_config(config)
        history = trainer.train()
        metrics = trainer.evaluate()
        trainer.save_training_report()
    """

    def __init__(
        self,
        config: Dict,
        checkpoint_dir: str = "checkpoints/emotion",
        log_dir: str = "logs/emotion",
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.train_ds = self.val_ds = self.test_ds = None
        self.class_weights: Dict[int, float] = {}
        self.label_names = list(EMOTION_LABELS.values())

    @classmethod
    def from_config(cls, config: Dict) -> "EmotionTrainer":
        paths = config.get("paths", {})
        return cls(
            config=config,
            checkpoint_dir=paths.get("checkpoint_dir", "checkpoints/emotion"),
            log_dir=paths.get("log_dir", "logs/emotion"),
        )

    def prepare_data(self):
        """Load and preprocess FER-2013 dataset."""
        data_cfg = self.config["paths"]
        train_cfg = self.config["training"]
        aug_cfg = self.config.get("augmentation", {})

        csv_path = os.path.join(data_cfg["data_dir"], "fer2013.csv")
        preprocessor = FER2013Preprocessor(
            csv_path=csv_path,
            augment=aug_cfg.get("horizontal_flip", True),
        )
        preprocessor.summary()

        self.train_ds, self.val_ds, self.test_ds = preprocessor.build_datasets(
            batch_size=train_cfg["batch_size"]
        )
        if train_cfg.get("use_class_weights", True):
            self.class_weights = preprocessor.get_class_weights()

    def build_model(self) -> tf.keras.Model:
        """Instantiate and compile the CNN."""
        model_cfg = self.config["model"]
        arch_cfg = model_cfg["architecture"]
        train_cfg = self.config["training"]

        self.model = build_emotion_cnn(
            input_shape=tuple(model_cfg["input_shape"]),
            num_classes=model_cfg["num_classes"],
            dense_units=arch_cfg["dense_units"],
            dense_dropout=arch_cfg["dense_dropout"],
        )
        compile_emotion_model(self.model, learning_rate=train_cfg["learning_rate"])
        self.model.summary(print_fn=logger.info)
        return self.model

    def _build_callbacks(self) -> list:
        """Construct training callbacks."""
        train_cfg = self.config["training"]
        es_cfg = train_cfg["early_stopping"]
        lr_cfg = train_cfg["reduce_lr"]

        best_ckpt = str(self.checkpoint_dir / "best_model.h5")
        last_ckpt = str(self.checkpoint_dir / "last_model.h5")

        callbacks = [
            # Save best model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_ckpt,
                monitor=es_cfg["monitor"],
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            # Save last checkpoint for resumption
            tf.keras.callbacks.ModelCheckpoint(
                filepath=last_ckpt,
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
            ),
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=es_cfg["patience"],
                restore_best_weights=es_cfg["restore_best_weights"],
                verbose=1,
            ),
            # Learning rate reduction on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_cfg["monitor"],
                factor=lr_cfg["factor"],
                patience=lr_cfg["patience"],
                min_lr=lr_cfg["min_lr"],
                verbose=1,
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
            ),
            # CSV logging
            tf.keras.callbacks.CSVLogger(
                str(self.log_dir / "training_log.csv"), append=False
            ),
            # Learning rate logger
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logs.update(
                    {"lr": float(self.model.optimizer.learning_rate.numpy()
                                 if hasattr(self.model.optimizer.learning_rate, "numpy")
                                 else self.model.optimizer.learning_rate)}
                )
            ),
        ]
        return callbacks

    def train(self) -> tf.keras.callbacks.History:
        """Run the full training loop."""
        assert self.model is not None, "Call build_model() first"
        assert self.train_ds is not None, "Call prepare_data() first"

        train_cfg = self.config["training"]
        logger.info("=" * 60)
        logger.info("Starting EmotionCNN training")
        logger.info(f"  Epochs:     {train_cfg['epochs']}")
        logger.info(f"  Batch size: {train_cfg['batch_size']}")
        logger.info(f"  LR:         {train_cfg['learning_rate']}")
        logger.info(f"  Class weights: {bool(self.class_weights)}")
        logger.info("=" * 60)

        start = time.time()
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=train_cfg["epochs"],
            callbacks=self._build_callbacks(),
            class_weight=self.class_weights if self.class_weights else None,
            verbose=1,
        )
        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed/60:.1f} min")

        best_val_acc = max(self.history.history.get("val_accuracy", [0]))
        logger.info(f"Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        return self.history

    def evaluate(self) -> Dict:
        """
        Evaluate on the held-out test set and compute per-class metrics.
        Returns dict with accuracy, per-class F1, confusion matrix.
        """
        assert self.model is not None and self.test_ds is not None

        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_top2 = self.model.evaluate(self.test_ds, verbose=1)
        logger.info(f"Test accuracy: {test_acc:.4f} | Top-2: {test_top2:.4f}")

        # Collect predictions
        y_true, y_pred = [], []
        for batch_imgs, batch_labels in self.test_ds:
            preds = self.model.predict(batch_imgs, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(batch_labels.numpy(), axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        report = classification_report(
            y_true, y_pred, target_names=self.label_names, output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_top2_accuracy": float(test_top2),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        self._plot_confusion_matrix(cm)
        self._plot_training_curves()
        return results

    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Save confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Emotion Recognition — Confusion Matrix", fontsize=14)
        plt.tight_layout()
        out_path = self.log_dir / "confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {out_path}")

    def _plot_training_curves(self):
        """Save accuracy and loss training curves."""
        if self.history is None:
            return
        hist = self.history.history
        epochs = range(1, len(hist["accuracy"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, hist["accuracy"], label="Train", linewidth=2)
        axes[0].plot(epochs, hist["val_accuracy"], label="Val", linewidth=2)
        axes[0].set_title("Accuracy", fontsize=13)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, hist["loss"], label="Train", linewidth=2)
        axes[1].plot(epochs, hist["val_loss"], label="Val", linewidth=2)
        axes[1].set_title("Loss", fontsize=13)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("EmotionCNN Training History", fontsize=15)
        plt.tight_layout()
        out_path = self.log_dir / "training_curves.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Training curves saved to {out_path}")

    def save_training_report(self, results: Optional[Dict] = None):
        """Persist training summary as JSON."""
        report = {
            "model": self.config["model"]["name"],
            "config": self.config,
            "history": {k: [float(v) for v in vals]
                        for k, vals in (self.history.history.items() if self.history else {})},
            "evaluation": results or {},
        }
        report_path = self.log_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report saved to {report_path}")
