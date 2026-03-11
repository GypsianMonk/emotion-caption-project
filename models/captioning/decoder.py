"""
LSTM Caption Decoder with Bahdanau Attention
---------------------------------------------
Autoregressive LSTM decoder that generates captions word-by-word,
conditioned on image features via attention at each step.

Architecture per decoding step:
  1. Embed current token → (batch, embed_dim)
  2. Compute attention over image features → context vector
  3. Concatenate [embedding, context] → (batch, embed_dim + feature_dim)
  4. Pass through LSTM → hidden state, cell state
  5. Dense → logits over vocabulary
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, Optional
import logging

from models.captioning.attention import BahdanauAttention

logger = logging.getLogger(__name__)


class CaptionDecoder(tf.keras.Model):
    """
    LSTM-based caption decoder with Bahdanau attention.

    Args:
        vocab_size:      Total vocabulary size (including special tokens)
        embedding_dim:   Word embedding dimensionality
        lstm_units:      LSTM hidden state units
        feature_dim:     Encoder feature dimension (projected, e.g. 512)
        attention_units: Attention projection units
        dropout_rate:    Dropout on LSTM output
        recurrent_dropout: Recurrent dropout inside LSTM
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        lstm_units: int = 512,
        feature_dim: int = 512,
        attention_units: int = 512,
        dropout_rate: float = 0.5,
        recurrent_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.feature_dim = feature_dim

        self.embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True,
            name="word_embedding",
        )
        self.attention = BahdanauAttention(attention_units, name="bahdanau_attention")

        # LSTM input: [embedding (embed_dim) + context (feature_dim)]
        self.lstm = layers.LSTM(
            lstm_units,
            return_sequences=False,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name="decoder_lstm",
        )

        self.fc1 = layers.Dense(lstm_units, activation="relu", name="fc1")
        self.dropout = layers.Dropout(dropout_rate, name="output_dropout")
        self.fc_out = layers.Dense(vocab_size, name="output_logits")

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single decoder step.

        Args:
            inputs: Tuple of (features, token_ids)
                features:  (batch, feature_dim) — projected image features
                token_ids: (batch, seq_len) — tokenized input sequence
            states: Optional (hidden_state, cell_state) for inference step-by-step
            training: bool

        Returns:
            logits:           (batch, seq_len, vocab_size)
            hidden_state:     (batch, lstm_units)
            cell_state:       (batch, lstm_units)
        """
        features, token_ids = inputs

        # Word embeddings: (batch, seq_len, embed_dim)
        x = self.embedding(token_ids)

        # Attention context: (batch, feature_dim)
        hidden = states[0] if states is not None else tf.zeros((tf.shape(features)[0], self.lstm_units))
        context_vector, _ = self.attention(features, hidden, training=training)

        # Expand context to match sequence length and concatenate
        seq_len = tf.shape(x)[1]
        context_expanded = tf.tile(
            tf.expand_dims(context_vector, 1), [1, seq_len, 1]
        )  # (batch, seq_len, feature_dim)
        lstm_input = tf.concat([x, context_expanded], axis=-1)  # (batch, seq_len, embed+feat)

        # LSTM: output (batch, seq_len, lstm_units) with return_sequences or (batch, lstm_units)
        # We use return_sequences=False here (single step during inference),
        # for training we loop or use teacher-forcing via the trainer
        lstm_out, hidden_state, cell_state = self.lstm(
            lstm_input,
            initial_state=states,
            training=training,
        )

        # Dense head
        x = self.fc1(lstm_out)
        x = self.dropout(x, training=training)
        logits = self.fc_out(x)  # (batch, vocab_size)

        return logits, hidden_state, cell_state

    def reset_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Initialize zero hidden and cell states."""
        return (
            tf.zeros((batch_size, self.lstm_units)),
            tf.zeros((batch_size, self.lstm_units)),
        )

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "feature_dim": self.feature_dim,
        }


class SequenceDecoder(tf.keras.Model):
    """
    Teacher-forcing wrapper for batch training.
    Takes full sequence input and produces logits for all positions in parallel.

    Used during training with tf.GradientTape for efficiency.
    """

    def __init__(self, decoder: CaptionDecoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            inputs: Tuple of (features, input_tokens)
                features:     (batch, feature_dim)
                input_tokens: (batch, seq_len) — shifted right (START + tokens, no END)
        Returns:
            all_logits: (batch, seq_len, vocab_size)
        """
        features, input_tokens = inputs
        batch_size = tf.shape(features)[0]
        seq_len = tf.shape(input_tokens)[1]

        states = self.decoder.reset_state(batch_size)
        all_logits = tf.TensorArray(dtype=tf.float32, size=seq_len)

        for t in tf.range(seq_len):
            token_t = tf.expand_dims(input_tokens[:, t], 1)  # (batch, 1)
            logits_t, hidden, cell = self.decoder(
                (features, token_t), states=states, training=training
            )
            states = (hidden, cell)
            all_logits = all_logits.write(t, logits_t)

        # Stack: (seq_len, batch, vocab_size) → (batch, seq_len, vocab_size)
        return tf.transpose(all_logits.stack(), [1, 0, 2])
