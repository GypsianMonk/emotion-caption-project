"""
Caption Generator
------------------
Inference module for image captioning.
Supports greedy decoding and beam search (width 1-5).

Usage:
    generator = CaptionGenerator.from_checkpoints(
        encoder_ckpt="checkpoints/captioning/encoder",
        decoder_ckpt="checkpoints/captioning/decoder",
        vocab_path="data/processed/tokenizer.pkl",
    )
    caption = generator.caption_image("path/to/image.jpg", beam_width=3)
    print(caption)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from data.preprocessing.coco_preprocessor import COCOVocabulary
from models.captioning.encoder import PrecomputedFeatureProjector
from models.captioning.decoder import CaptionDecoder

logger = logging.getLogger(__name__)

IMG_SIZE = 299


class CaptionGenerator:
    """
    Wraps encoder + decoder for caption inference with greedy or beam search.

    Args:
        encoder:    Trained PrecomputedFeatureProjector
        decoder:    Trained CaptionDecoder
        vocab:      COCOVocabulary instance
        max_length: Maximum caption length
        image_size: Input image size for InceptionV3
    """

    def __init__(
        self,
        encoder: PrecomputedFeatureProjector,
        decoder: CaptionDecoder,
        vocab: COCOVocabulary,
        max_length: int = 50,
        image_size: int = IMG_SIZE,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.max_length = max_length
        self.image_size = image_size

        # Cache special token IDs
        self.start_id = vocab.word2idx[COCOVocabulary.START_TOKEN]
        self.end_id   = vocab.word2idx[COCOVocabulary.END_TOKEN]
        self.pad_id   = vocab.word2idx[COCOVocabulary.PAD_TOKEN]

        # InceptionV3 for processing raw images
        self._inception: Optional[tf.keras.Model] = None

    def _get_inception(self) -> tf.keras.Model:
        """Lazy-load InceptionV3 for raw image processing."""
        if self._inception is None:
            base = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet", pooling="avg"
            )
            base.trainable = False
            self._inception = base
        return self._inception

    def preprocess_image(self, image_path: str) -> tf.Tensor:
        """Load and preprocess an image file for InceptionV3."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return tf.expand_dims(img, 0)

    def extract_features(self, image_path: str) -> tf.Tensor:
        """Extract 2048-dim features from a raw image path."""
        inception = self._get_inception()
        img_tensor = self.preprocess_image(image_path)
        raw_features = inception(img_tensor, training=False)          # (1, 2048)
        projected = self.encoder(raw_features, training=False)        # (1, proj_dim)
        return projected

    def greedy_decode(self, features: tf.Tensor) -> str:
        """
        Greedy decoding: always pick argmax at each step.

        Args:
            features: (1, proj_dim) projected encoder output

        Returns:
            Generated caption string
        """
        tokens = [self.start_id]
        states = self.decoder.reset_state(batch_size=1)

        for _ in range(self.max_length):
            token_input = tf.constant([[tokens[-1]]], dtype=tf.int32)
            logits, hidden, cell = self.decoder(
                (features, token_input), states=states, training=False
            )
            states = (hidden, cell)
            next_id = int(tf.argmax(logits[0]).numpy())
            if next_id == self.end_id:
                break
            tokens.append(next_id)

        return self.vocab.decode(tokens[1:], skip_special=True)

    def beam_search(self, features: tf.Tensor, beam_width: int = 3) -> str:
        """
        Beam search decoding for higher quality captions.

        Args:
            features:   (1, proj_dim) projected encoder output
            beam_width: Number of beams to maintain

        Returns:
            Best caption string (highest log probability, length-normalized)
        """
        # Beam: list of (log_prob, token_sequence, (hidden, cell))
        init_states = self.decoder.reset_state(batch_size=1)
        beams = [(0.0, [self.start_id], init_states)]
        completed = []

        for step in range(self.max_length):
            candidates = []
            for log_prob, seq, states in beams:
                token_input = tf.constant([[seq[-1]]], dtype=tf.int32)
                logits, hidden, cell = self.decoder(
                    (features, token_input), states=states, training=False
                )
                log_probs = tf.nn.log_softmax(logits[0]).numpy()

                # Expand top-k candidates
                top_k_ids = np.argsort(log_probs)[-beam_width:]
                for token_id in top_k_ids:
                    new_log_prob = log_prob + log_probs[token_id]
                    new_seq = seq + [int(token_id)]
                    new_states = (hidden, cell)

                    if int(token_id) == self.end_id:
                        # Length-normalize completed hypothesis
                        length_penalty = len(new_seq) ** 0.7
                        completed.append((new_log_prob / length_penalty, new_seq))
                    else:
                        candidates.append((new_log_prob, new_seq, new_states))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        # If no complete sequences, use the best ongoing beam
        if not completed:
            beams.sort(key=lambda x: x[0], reverse=True)
            best_seq = beams[0][1]
        else:
            completed.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed[0][1]

        return self.vocab.decode(best_seq[1:], skip_special=True)

    def caption_image(
        self,
        image_path: str,
        beam_width: int = 3,
    ) -> str:
        """
        Full pipeline: image path → caption string.

        Args:
            image_path: Path to image file (JPEG/PNG)
            beam_width: 1 = greedy, >1 = beam search

        Returns:
            Generated caption string
        """
        features = self.extract_features(image_path)
        if beam_width <= 1:
            return self.greedy_decode(features)
        return self.beam_search(features, beam_width=beam_width)

    def caption_array(
        self,
        image_array: np.ndarray,
        beam_width: int = 3,
    ) -> str:
        """
        Caption from a numpy image array (e.g. from OpenCV frame).

        Args:
            image_array: BGR or RGB numpy array (H, W, 3)
            beam_width:  Beam search width

        Returns:
            Generated caption string
        """
        # Convert BGR to RGB if needed
        if image_array.shape[-1] == 3:
            rgb = image_array[..., ::-1] if image_array.dtype == np.uint8 else image_array
        else:
            rgb = image_array

        img = tf.image.resize(
            tf.cast(rgb, tf.float32),
            [self.image_size, self.image_size]
        )
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.expand_dims(img, 0)

        inception = self._get_inception()
        raw_features = inception(img, training=False)
        features = self.encoder(raw_features, training=False)

        if beam_width <= 1:
            return self.greedy_decode(features)
        return self.beam_search(features, beam_width=beam_width)

    @classmethod
    def from_checkpoints(
        cls,
        encoder_ckpt: str,
        decoder_ckpt: str,
        vocab_path: str,
        projection_dim: int = 512,
        lstm_units: int = 512,
        embedding_dim: int = 256,
        max_length: int = 50,
    ) -> "CaptionGenerator":
        """Factory: load encoder, decoder, vocab from checkpoint paths."""
        vocab = COCOVocabulary.load(vocab_path)

        encoder = PrecomputedFeatureProjector(projection_dim=projection_dim)
        encoder.load_weights(encoder_ckpt)

        decoder = CaptionDecoder(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            feature_dim=projection_dim,
        )
        decoder.load_weights(decoder_ckpt)

        logger.info(f"CaptionGenerator loaded from {encoder_ckpt}, {decoder_ckpt}")
        return cls(encoder=encoder, decoder=decoder, vocab=vocab, max_length=max_length)
