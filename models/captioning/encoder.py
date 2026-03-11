"""
InceptionV3 Image Encoder
---------------------------
Wraps InceptionV3 (pretrained on ImageNet) as a frozen feature extractor,
projecting 2048-dim pooled features to a configurable embedding dimension.

During training, features are pre-extracted and cached (see coco_preprocessor.py),
so this module is used for inference and fine-tuning scenarios.
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ImageEncoder(tf.keras.Model):
    """
    InceptionV3-based image encoder for captioning.

    Input:  Image tensor (batch, 299, 299, 3) — or pre-extracted features (batch, 2048)
    Output: Projected feature vector (batch, projection_dim)

    Args:
        projection_dim:    Output embedding dimension (default 512)
        fine_tune_from:    Layer name to unfreeze from (None = fully frozen)
        dropout_rate:      Dropout on projected features
    """

    def __init__(
        self,
        projection_dim: int = 512,
        fine_tune_from: Optional[str] = None,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.fine_tune_from = fine_tune_from

        # InceptionV3 backbone — remove top, apply global avg pooling
        self.inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        self.inception.trainable = False

        if fine_tune_from is not None:
            self._enable_fine_tuning(fine_tune_from)

        self.projection = layers.Dense(
            projection_dim,
            activation="relu",
            name="feature_projection",
        )
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.bn = layers.BatchNormalization(name="encoder_bn")

    def _enable_fine_tuning(self, from_layer: str):
        """Unfreeze InceptionV3 layers starting from `from_layer`."""
        layer_names = [l.name for l in self.inception.layers]
        if from_layer not in layer_names:
            raise ValueError(
                f"Layer '{from_layer}' not found. Available: {layer_names[-10:]}"
            )
        trainable = False
        for layer in self.inception.layers:
            if layer.name == from_layer:
                trainable = True
            layer.trainable = trainable
        trainable_count = sum(1 for l in self.inception.layers if l.trainable)
        logger.info(
            f"Fine-tuning enabled from layer '{from_layer}': "
            f"{trainable_count}/{len(self.inception.layers)} layers trainable"
        )

    def call(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.

        Args:
            images: (batch, 299, 299, 3) — preprocessed with inception_v3.preprocess_input
            training: bool

        Returns:
            features: (batch, projection_dim)
        """
        x = self.inception(images, training=training and self.inception.trainable)
        x = self.projection(x, training=training)
        x = self.bn(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self):
        return {
            "projection_dim": self.projection_dim,
            "fine_tune_from": self.fine_tune_from,
        }


class PrecomputedFeatureProjector(tf.keras.layers.Layer):
    """
    Lightweight projector for pre-extracted InceptionV3 features (2048-dim).
    Used during training when features are cached on disk — avoids running
    the full InceptionV3 forward pass every batch.

    Input:  (batch, 2048) — cached feature vectors
    Output: (batch, projection_dim)
    """

    def __init__(self, projection_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.dense = layers.Dense(projection_dim, activation="relu", name="feat_proj")
        self.bn = layers.BatchNormalization(name="feat_proj_bn")

    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense(features)
        x = self.bn(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config["projection_dim"] = self.projection_dim
        return config
