"""
Bahdanau (Additive) Attention Mechanism
-----------------------------------------
Used in the image captioning decoder to allow the LSTM to attend
to different parts of the image feature representation at each decoding step.

Reference: Bahdanau et al. "Neural Machine Translation by Jointly Learning
           to Align and Translate" (2015), adapted for image captioning.
"""

import tensorflow as tf
from tensorflow.keras import layers


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau additive attention.

    At each decoder step t, computes a context vector as a weighted sum
    of the encoder features, where weights (attention scores) reflect
    relevance to the current decoder hidden state.

    Score function:
        e_t = V * tanh(W1 * features + W2 * hidden_state)
        alpha_t = softmax(e_t)
        context_t = sum(alpha_t * features)

    Args:
        units: Dimensionality of the attention space
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units, use_bias=False, name="attention_W1")
        self.W2 = layers.Dense(units, use_bias=False, name="attention_W2")
        self.V  = layers.Dense(1, use_bias=False, name="attention_V")

    def call(
        self,
        features: tf.Tensor,
        hidden: tf.Tensor,
        training: bool = False,
    ) -> tuple:
        """
        Compute context vector and attention weights.

        Args:
            features: Encoder output — shape (batch, feature_dim)
                      Will be expanded to (batch, 1, feature_dim) for broadcasting
            hidden:   Decoder hidden state — shape (batch, lstm_units)

        Returns:
            context_vector: (batch, feature_dim) — attended encoder output
            attention_weights: (batch, 1, 1) — for visualization
        """
        # Expand dims for broadcasting: (batch, 1, feature_dim)
        features_expanded = tf.expand_dims(features, 1)

        # Hidden state projection: (batch, 1, units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # Additive attention score: (batch, 1, units)
        score = self.V(
            tf.nn.tanh(
                self.W1(features_expanded) + self.W2(hidden_with_time_axis)
            )
        )

        # Attention weights: (batch, 1, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector: (batch, feature_dim)
        context_vector = attention_weights * features_expanded
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
