"""
MS-COCO Image Captioning Preprocessor
---------------------------------------
Handles COCO annotation parsing, image feature caching,
vocabulary building, and tf.data pipeline construction.

Expected directory layout:
  data/raw/coco/
    ├── annotations/
    │   ├── captions_train2017.json
    │   └── captions_val2017.json
    └── images/
        ├── train2017/
        └── val2017/
"""

import os
import json
import pickle
import hashlib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class COCOVocabulary:
    """
    Builds and manages the word-to-index vocabulary from COCO captions.

    Special tokens:
        <PAD> = 0
        <START> = 1
        <END> = 2
        <OOV> = 3
    """

    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    OOV_TOKEN = "<OOV>"
    RESERVED = [PAD_TOKEN, START_TOKEN, END_TOKEN, OOV_TOKEN]

    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._built = False

    def build(self, captions: List[str]):
        """Build vocabulary from list of caption strings."""
        from collections import Counter
        import re

        logger.info("Building vocabulary...")
        counter: Counter = Counter()
        for caption in tqdm(captions, desc="Tokenizing"):
            tokens = self._tokenize(caption)
            counter.update(tokens)

        # Reserve slots for special tokens
        vocab_words = [w for w, _ in counter.most_common(self.max_vocab_size - len(self.RESERVED))]
        all_words = self.RESERVED + vocab_words

        self.word2idx = {w: i for i, w in enumerate(all_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built = True

        logger.info(
            f"Vocabulary built: {len(self.word2idx)} tokens "
            f"(max: {self.max_vocab_size}, coverage cutoff at rank {len(vocab_words)})"
        )

    def _tokenize(self, caption: str) -> List[str]:
        """Simple whitespace tokenizer with lowercasing and punctuation removal."""
        import re
        caption = caption.lower().strip()
        caption = re.sub(r"[^a-z0-9\s]", "", caption)
        return caption.split()

    def encode(self, caption: str, add_special_tokens: bool = True) -> List[int]:
        """Convert caption string to list of token IDs."""
        tokens = self._tokenize(caption)
        ids = [self.word2idx.get(t, self.word2idx[self.OOV_TOKEN]) for t in tokens]
        if add_special_tokens:
            ids = [self.word2idx[self.START_TOKEN]] + ids + [self.word2idx[self.END_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert list of token IDs back to string."""
        special = {self.word2idx[t] for t in self.RESERVED}
        words = []
        for i in ids:
            if skip_special and i in special:
                continue
            words.append(self.idx2word.get(i, self.OOV_TOKEN))
        return " ".join(words)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)
        logger.info(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path: str) -> "COCOVocabulary":
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        vocab._built = True
        return vocab

    def __len__(self):
        return len(self.word2idx)


class COCOFeatureExtractor:
    """
    Extracts and caches InceptionV3 image features for the COCO dataset.
    Features are cached to disk to avoid re-extraction on each training run.
    """

    def __init__(
        self,
        cache_dir: str,
        image_size: int = 299,
        batch_size: int = 32,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.batch_size = batch_size
        self._model: Optional[tf.keras.Model] = None

    def _build_encoder(self) -> tf.keras.Model:
        """Build InceptionV3 feature extractor (no top, global avg pool)."""
        inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(self.image_size, self.image_size, 3),
        )
        output = tf.keras.layers.GlobalAveragePooling2D()(inception.output)
        model = tf.keras.Model(inputs=inception.input, outputs=output, name="inception_extractor")
        model.trainable = False
        logger.info(f"InceptionV3 encoder built: output shape {model.output_shape}")
        return model

    def _preprocess_image(self, image_path: str) -> tf.Tensor:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img

    def _cache_key(self, image_path: str) -> str:
        return hashlib.md5(image_path.encode()).hexdigest() + ".npy"

    def extract_and_cache(self, image_paths: List[str]) -> None:
        """Extract features for all images and save to cache directory."""
        if self._model is None:
            self._model = self._build_encoder()

        uncached = [p for p in image_paths if not (self.cache_dir / self._cache_key(p)).exists()]
        if not uncached:
            logger.info("All features already cached. Skipping extraction.")
            return

        logger.info(f"Extracting features for {len(uncached)} images...")
        for i in tqdm(range(0, len(uncached), self.batch_size), desc="Extracting"):
            batch_paths = uncached[i : i + self.batch_size]
            imgs = tf.stack([self._preprocess_image(p) for p in batch_paths])
            features = self._model(imgs, training=False).numpy()
            for path, feat in zip(batch_paths, features):
                np.save(self.cache_dir / self._cache_key(path), feat)

    def load_feature(self, image_path: str) -> np.ndarray:
        cache_file = self.cache_dir / self._cache_key(image_path)
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Feature cache not found for {image_path}. Run extract_and_cache() first."
            )
        return np.load(str(cache_file))


class COCOPreprocessor:
    """
    Full preprocessing pipeline for MS-COCO image captioning.

    Usage:
        prep = COCOPreprocessor(
            data_dir="data/raw/coco",
            cache_dir="data/processed/coco_features",
            tokenizer_path="data/processed/tokenizer.pkl"
        )
        vocab = prep.build_vocabulary(max_vocab_size=10000)
        prep.extract_features()
        train_ds, val_ds = prep.build_datasets(batch_size=64, max_length=50)
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        tokenizer_path: str,
        max_vocab_size: int = 10000,
        max_caption_length: int = 50,
        image_size: int = 299,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = cache_dir
        self.tokenizer_path = tokenizer_path
        self.max_vocab_size = max_vocab_size
        self.max_caption_length = max_caption_length
        self.image_size = image_size

        self._train_data: Optional[List] = None
        self._val_data: Optional[List] = None
        self.vocab: Optional[COCOVocabulary] = None
        self.feature_extractor = COCOFeatureExtractor(
            cache_dir=cache_dir, image_size=image_size
        )

    def _load_annotations(self, split: str) -> Tuple[List[str], List[str]]:
        """
        Load COCO captions annotations.
        Returns: (image_paths, captions) — one caption per entry (5 per image).
        """
        ann_file = self.data_dir / "annotations" / f"captions_{split}2017.json"
        img_dir = self.data_dir / "images" / f"{split}2017"

        with open(ann_file) as f:
            data = json.load(f)

        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        image_paths, captions = [], []

        for ann in data["annotations"]:
            img_id = ann["image_id"]
            filename = id_to_filename.get(img_id)
            if filename is None:
                continue
            image_paths.append(str(img_dir / filename))
            captions.append(ann["caption"])

        logger.info(f"Loaded {len(captions)} caption annotations for {split}")
        return image_paths, captions

    def build_vocabulary(self) -> COCOVocabulary:
        """Build vocabulary from training captions."""
        if Path(self.tokenizer_path).exists():
            logger.info(f"Loading existing vocabulary from {self.tokenizer_path}")
            self.vocab = COCOVocabulary.load(self.tokenizer_path)
            return self.vocab

        _, train_captions = self._load_annotations("train")
        self.vocab = COCOVocabulary(max_vocab_size=self.max_vocab_size)
        self.vocab.build(train_captions)
        self.vocab.save(self.tokenizer_path)
        return self.vocab

    def extract_features(self):
        """Extract and cache InceptionV3 features for all train/val images."""
        for split in ["train", "val"]:
            image_paths, _ = self._load_annotations(split)
            unique_paths = list(set(image_paths))
            logger.info(f"Extracting features for {split} ({len(unique_paths)} unique images)...")
            self.feature_extractor.extract_and_cache(unique_paths)

    def build_datasets(
        self, batch_size: int = 64
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Build training and validation tf.data.Datasets."""
        assert self.vocab is not None, "Call build_vocabulary() first"

        def make_generator(split: str):
            image_paths, captions = self._load_annotations(split)

            def generator():
                for img_path, caption in zip(image_paths, captions):
                    try:
                        feat = self.feature_extractor.load_feature(img_path)
                    except FileNotFoundError:
                        continue

                    tokens = self.vocab.encode(caption, add_special_tokens=True)
                    # Pad or truncate to max_caption_length
                    tokens = tokens[: self.max_caption_length]
                    tokens += [self.vocab.word2idx[COCOVocabulary.PAD_TOKEN]] * (
                        self.max_caption_length - len(tokens)
                    )

                    # Input: all tokens except last; Target: all tokens except first
                    inp = np.array(tokens[:-1], dtype=np.int32)
                    tgt = np.array(tokens[1:], dtype=np.int32)
                    yield (feat.astype(np.float32), inp), tgt

            return generator

        def build_tf_dataset(split: str, shuffle: bool) -> tf.data.Dataset:
            image_paths, _ = self._load_annotations(split)
            feat_dim = 2048

            ds = tf.data.Dataset.from_generator(
                make_generator(split),
                output_signature=(
                    (
                        tf.TensorSpec(shape=(feat_dim,), dtype=tf.float32),
                        tf.TensorSpec(shape=(self.max_caption_length - 1,), dtype=tf.int32),
                    ),
                    tf.TensorSpec(shape=(self.max_caption_length - 1,), dtype=tf.int32),
                ),
            )
            if shuffle:
                ds = ds.shuffle(buffer_size=8192, reshuffle_each_iteration=True)
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_ds = build_tf_dataset("train", shuffle=True)
        val_ds = build_tf_dataset("val", shuffle=False)
        return train_ds, val_ds
