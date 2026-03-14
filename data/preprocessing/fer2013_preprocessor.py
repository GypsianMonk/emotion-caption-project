"""
FER-2013 Data Preprocessor
---------------------------
Handles loading, augmentation, and tf.data pipeline construction
for the FER-2013 facial emotion recognition dataset.

FER-2013 CSV format:
  columns: emotion (0-6), pixels (space-separated 48x48), Usage (train/val/test)
  35,887 grayscale 48x48 images
  Classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
"""

import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

IMG_SIZE = 48
NUM_CLASSES = 7


class FER2013Preprocessor:
    """
    Full preprocessing pipeline for FER-2013 dataset.

    Usage:
        preprocessor = FER2013Preprocessor("data/raw/fer2013/fer2013.csv")
        train_ds, val_ds, test_ds = preprocessor.build_datasets(batch_size=64)
        class_weights = preprocessor.get_class_weights()
    """

    def __init__(
        self,
        csv_path: str,
        img_size: int = IMG_SIZE,
        augment: bool = True,
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.img_size = img_size
        self.augment = augment
        self.seed = seed
        self._df: Optional[pd.DataFrame] = None

    def _load_csv(self) -> pd.DataFrame:
        """Load and parse the FER-2013 CSV file."""
        if self._df is not None:
            return self._df

        logger.info(f"Loading FER-2013 from {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        # Validate expected columns
        assert set(["emotion", "pixels", "Usage"]).issubset(
            df.columns
        ), "CSV missing required columns: emotion, pixels, Usage"

        self._df = df
        logger.info(
            f"Loaded {len(df)} samples | "
            f"Train: {(df.Usage=='Training').sum()} | "
            f"Val: {(df.Usage=='PublicTest').sum()} | "
            f"Test: {(df.Usage=='PrivateTest').sum()}"
        )
        return df

    def _parse_pixels(self, pixel_str: str) -> np.ndarray:
        """Convert space-separated pixel string to (48, 48, 1) float32 array."""
        pixels = np.array(pixel_str.split(), dtype=np.float32)
        img = pixels.reshape(self.img_size, self.img_size, 1) / 255.0
        return img

    def _split_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split into train/val/test numpy arrays."""
        df = self._load_csv()
        splits = {}

        usage_map = {
            "train": "Training",
            "val": "PublicTest",
            "test": "PrivateTest",
        }

        for split_name, usage_label in usage_map.items():
            subset = df[df["Usage"] == usage_label].reset_index(drop=True)
            X = np.stack(subset["pixels"].apply(self._parse_pixels).values)
            y = tf.keras.utils.to_categorical(subset["emotion"].values, NUM_CLASSES)
            splits[split_name] = (X.astype(np.float32), y.astype(np.float32))
            logger.info(
                f"{split_name}: {len(X)} samples | shape: {X.shape}"
            )

        return splits

    def _build_augmentation_layer(self) -> tf.keras.Sequential:
        """Keras augmentation pipeline applied only during training."""
        import tensorflow as tf
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),       # ±10 degrees
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomBrightness(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ],
            name="augmentation",
        )

    def build_datasets(
        self,
        batch_size: int = 64,
        cache: bool = True,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Build optimized tf.data.Dataset objects for train/val/test.

        Returns:
            train_ds, val_ds, test_ds — batched, prefetched tf.data.Datasets
        """
        splits = self._split_data()
        augmenter = self._build_augmentation_layer()

        def make_ds(X, y, training: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if training:
                ds = ds.shuffle(buffer_size=len(X), seed=self.seed, reshuffle_each_iteration=True)
            if cache:
                ds = ds.cache()
            if training and self.augment:
                ds = ds.batch(batch_size).map(
                    lambda imgs, labels: (augmenter(imgs, training=True), labels),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            else:
                ds = ds.batch(batch_size)
            return ds.prefetch(tf.data.AUTOTUNE)

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        train_ds = make_ds(X_train, y_train, training=True)
        val_ds = make_ds(X_val, y_val, training=False)
        test_ds = make_ds(X_test, y_test, training=False)

        return train_ds, val_ds, test_ds

    def get_class_weights(self) -> Dict[int, float]:
        """
        Compute balanced class weights to handle FER-2013 class imbalance.
        FER-2013 is heavily skewed toward Happy and Neutral.
        """
        df = self._load_csv()
        train_labels = df[df["Usage"] == "Training"]["emotion"].values
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(NUM_CLASSES),
            y=train_labels,
        )
        class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
        logger.info(f"Class weights: {class_weight_dict}")
        return class_weight_dict

    def get_label_names(self) -> Dict[int, str]:
        return EMOTION_LABELS.copy()

    def summary(self):
        """Print dataset statistics."""
        df = self._load_csv()
        print("\n=== FER-2013 Dataset Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"\nClass distribution (Training set):")
        train = df[df["Usage"] == "Training"]
        dist = train["emotion"].value_counts().sort_index()
        for idx, count in dist.items():
            bar = "█" * (count // 100)
            print(f"  {EMOTION_LABELS[idx]:10s} ({idx}): {count:5d}  {bar}")
        print()
