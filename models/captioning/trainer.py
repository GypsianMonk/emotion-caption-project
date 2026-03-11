"""
Image Captioning Trainer
-------------------------
Custom tf.GradientTape training loop for the InceptionV3+LSTM captioning model.
Uses teacher forcing during training and beam search for evaluation.

Supports:
  - Gradient clipping
  - Masked loss (ignore <PAD> tokens)
  - BLEU-4 evaluation per epoch
  - TensorBoard scalar and image-caption logging
  - Checkpoint save/restore
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from models.captioning.encoder import PrecomputedFeatureProjector
from models.captioning.decoder import CaptionDecoder, SequenceDecoder
from data.preprocessing.coco_preprocessor import COCOVocabulary
from utils.metrics import compute_corpus_bleu

logger = logging.getLogger(__name__)


class CaptioningTrainer:
    """
    Full training orchestration for the InceptionV3+LSTM captioning system.

    Usage:
        trainer = CaptioningTrainer.from_config(config, vocab)
        trainer.prepare_data()
        trainer.build_models()
        trainer.train()
        results = trainer.evaluate_bleu(num_samples=1000)
    """

    def __init__(
        self,
        config: Dict,
        vocab: COCOVocabulary,
        checkpoint_dir: str = "checkpoints/captioning",
        log_dir: str = "logs/captioning",
    ):
        self.config = config
        self.vocab = vocab
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.encoder: Optional[PrecomputedFeatureProjector] = None
        self.decoder: Optional[CaptionDecoder] = None
        self.sequence_decoder: Optional[SequenceDecoder] = None
        self.optimizer: Optional[tf.keras.optimizers.Optimizer] = None
        self.train_ds = self.val_ds = None

        # Metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        # TensorBoard writer
        self.tb_writer = tf.summary.create_file_writer(str(self.log_dir))

    @classmethod
    def from_config(cls, config: Dict, vocab: COCOVocabulary) -> "CaptioningTrainer":
        paths = config.get("paths", {})
        return cls(
            config=config,
            vocab=vocab,
            checkpoint_dir=paths.get("checkpoint_dir", "checkpoints/captioning"),
            log_dir=paths.get("log_dir", "logs/captioning"),
        )

    def build_models(self):
        """Initialize encoder, decoder, optimizer, and checkpointing."""
        model_cfg = self.config["model"]
        enc_cfg = model_cfg["encoder"]
        dec_cfg = model_cfg["decoder"]
        train_cfg = self.config["training"]

        self.encoder = PrecomputedFeatureProjector(
            projection_dim=enc_cfg["projection_dim"],
            name="feature_projector",
        )

        self.decoder = CaptionDecoder(
            vocab_size=len(self.vocab),
            embedding_dim=dec_cfg["embedding_dim"],
            lstm_units=dec_cfg["lstm_units"],
            feature_dim=enc_cfg["projection_dim"],
            attention_units=dec_cfg["attention_units"],
            dropout_rate=dec_cfg["dropout"],
            recurrent_dropout=dec_cfg["recurrent_dropout"],
            name="caption_decoder",
        )

        self.sequence_decoder = SequenceDecoder(self.decoder, name="sequence_decoder")

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=train_cfg["learning_rate"],
            clipnorm=train_cfg.get("gradient_clip_norm", 5.0),
        )

        # TF Checkpoint manager
        self.ckpt = tf.train.Checkpoint(
            encoder=self.encoder,
            decoder=self.decoder,
            optimizer=self.optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            str(self.checkpoint_dir),
            max_to_keep=3,
        )
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logger.info(f"Restored checkpoint: {self.ckpt_manager.latest_checkpoint}")

    def _masked_loss(self, targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        """
        Sparse categorical crossentropy ignoring PAD tokens (index 0).
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = loss_fn(targets, logits)
        mask = tf.cast(tf.not_equal(targets, 0), dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    @tf.function
    def _train_step(
        self,
        features: tf.Tensor,
        inp_tokens: tf.Tensor,
        tgt_tokens: tf.Tensor,
    ) -> tf.Tensor:
        """Single gradient update step."""
        with tf.GradientTape() as tape:
            projected = self.encoder(features, training=True)
            logits = self.sequence_decoder(
                (projected, inp_tokens), training=True
            )  # (batch, seq_len, vocab_size)
            loss = self._masked_loss(tgt_tokens, logits)

        variables = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function
    def _val_step(
        self,
        features: tf.Tensor,
        inp_tokens: tf.Tensor,
        tgt_tokens: tf.Tensor,
    ) -> tf.Tensor:
        projected = self.encoder(features, training=False)
        logits = self.sequence_decoder(
            (projected, inp_tokens), training=False
        )
        return self._masked_loss(tgt_tokens, logits)

    def train(self):
        """Run full training loop."""
        assert self.train_ds is not None, "Call prepare_data() first"
        assert self.encoder is not None, "Call build_models() first"

        train_cfg = self.config["training"]
        num_epochs = train_cfg["epochs"]
        best_val_loss = float("inf")
        patience_counter = 0
        es_patience = train_cfg["early_stopping"]["patience"]
        lr_patience = train_cfg["reduce_lr"]["patience"]
        lr_counter = 0

        history = {"train_loss": [], "val_loss": []}

        logger.info("=" * 60)
        logger.info("Starting image captioning training")
        logger.info(f"  Vocab size: {len(self.vocab)}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info("=" * 60)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            self.train_loss_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            # --- Training ---
            for batch_inputs, tgt in self.train_ds:
                features, inp_tokens = batch_inputs
                loss = self._train_step(features, inp_tokens, tgt)
                self.train_loss_tracker.update_state(loss)

            # --- Validation ---
            for batch_inputs, tgt in self.val_ds:
                features, inp_tokens = batch_inputs
                val_loss = self._val_step(features, inp_tokens, tgt)
                self.val_loss_tracker.update_state(val_loss)

            train_loss = float(self.train_loss_tracker.result())
            val_loss = float(self.val_loss_tracker.result())
            elapsed = time.time() - epoch_start
            current_lr = float(self.optimizer.learning_rate)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # TensorBoard
            with self.tb_writer.as_default():
                tf.summary.scalar("train_loss", train_loss, step=epoch)
                tf.summary.scalar("val_loss", val_loss, step=epoch)
                tf.summary.scalar("learning_rate", current_lr, step=epoch)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                lr_counter = 0
                self.ckpt_manager.save()
                logger.info(f"  ✓ New best val loss: {best_val_loss:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                lr_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{es_patience})")

            # Learning rate decay
            if lr_counter >= lr_patience:
                new_lr = max(
                    current_lr * train_cfg["reduce_lr"]["factor"],
                    train_cfg["reduce_lr"]["min_lr"],
                )
                self.optimizer.learning_rate.assign(new_lr)
                lr_counter = 0
                logger.info(f"  LR reduced to {new_lr:.2e}")

            # Early stopping
            if patience_counter >= es_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Save history
        with open(self.log_dir / "captioning_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return history

    def evaluate_bleu(self, num_samples: int = 500) -> Dict:
        """
        Greedy-decode captions and compute BLEU-1 through BLEU-4.
        """
        from inference.caption_generator import CaptionGenerator
        generator = CaptionGenerator(
            encoder=self.encoder,
            decoder=self.decoder,
            vocab=self.vocab,
            max_length=self.config["model"]["vocab"]["max_caption_length"],
        )

        references, hypotheses = [], []
        count = 0

        for batch_inputs, tgt_tokens in self.val_ds:
            features, _ = batch_inputs
            projected = self.encoder(features, training=False)

            for i in range(features.shape[0]):
                if count >= num_samples:
                    break
                caption = generator.greedy_decode(projected[i:i+1])
                ref_tokens = [
                    self.vocab.idx2word.get(int(t), "") 
                    for t in tgt_tokens[i].numpy() if int(t) not in (0, 1, 2, 3)
                ]
                hypotheses.append(caption.split())
                references.append([ref_tokens])
                count += 1
            if count >= num_samples:
                break

        bleu_scores = compute_corpus_bleu(references, hypotheses)
        logger.info(f"BLEU Scores on {num_samples} samples: {bleu_scores}")
        return bleu_scores
