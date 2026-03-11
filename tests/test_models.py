"""
Unit Tests
-----------
Tests for model architectures, inference pipeline, and API endpoints.
Run: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Model Architecture Tests ─────────────────────────────────────────────────

class TestEmotionCNN:
    def test_model_builds(self):
        from models.emotion.cnn_model import build_emotion_cnn
        model = build_emotion_cnn()
        assert model is not None
        assert model.output_shape == (None, 7)

    def test_forward_pass(self):
        import tensorflow as tf
        from models.emotion.cnn_model import build_emotion_cnn
        model = build_emotion_cnn()
        dummy = tf.zeros((4, 48, 48, 1))
        out = model(dummy, training=False)
        assert out.shape == (4, 7)
        # Probabilities sum to 1
        np.testing.assert_allclose(
            out.numpy().sum(axis=1), np.ones(4), atol=1e-5
        )

    def test_parameter_count(self):
        from models.emotion.cnn_model import build_emotion_cnn
        model = build_emotion_cnn()
        params = model.count_params()
        # Should be between 500K and 5M for a reasonable CNN
        assert 500_000 < params < 5_000_000, f"Unexpected param count: {params:,}"

    def test_compile(self):
        from models.emotion.cnn_model import build_emotion_cnn, compile_emotion_model
        model = build_emotion_cnn()
        compile_emotion_model(model)
        assert model.optimizer is not None
        assert model.loss == "categorical_crossentropy"


class TestAttention:
    def test_attention_output_shapes(self):
        import tensorflow as tf
        from models.captioning.attention import BahdanauAttention
        attn = BahdanauAttention(units=256)
        batch, feat_dim, hidden_dim = 8, 512, 512
        features = tf.random.normal((batch, feat_dim))
        hidden   = tf.random.normal((batch, hidden_dim))
        context, weights = attn(features, hidden)
        assert context.shape == (batch, feat_dim)
        assert weights.shape == (batch, 1, 1)

    def test_attention_weights_are_valid_probabilities(self):
        import tensorflow as tf
        from models.captioning.attention import BahdanauAttention
        attn = BahdanauAttention(units=128)
        features = tf.random.normal((4, 512))
        hidden = tf.random.normal((4, 512))
        _, weights = attn(features, hidden)
        # Softmax output: sum ≈ 1 over attention dim
        # Weights shape (4, 1, 1) — just verify they're in [0, 1]
        w = weights.numpy()
        assert np.all(w >= 0) and np.all(w <= 1)


class TestDecoder:
    def test_decoder_single_step(self):
        import tensorflow as tf
        from models.captioning.decoder import CaptionDecoder
        vocab_size, batch = 1000, 4
        decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embedding_dim=64,
            lstm_units=128,
            feature_dim=256,
            attention_units=128,
        )
        features   = tf.random.normal((batch, 256))
        token_ids  = tf.zeros((batch, 1), dtype=tf.int32)
        states     = decoder.reset_state(batch_size=batch)

        logits, h, c = decoder((features, token_ids), states=states)
        assert logits.shape == (batch, vocab_size)
        assert h.shape == (batch, 128)
        assert c.shape == (batch, 128)


# ─── Vocabulary Tests ─────────────────────────────────────────────────────────

class TestCOCOVocabulary:
    def test_build_and_encode(self):
        from data.preprocessing.coco_preprocessor import COCOVocabulary
        vocab = COCOVocabulary(max_vocab_size=100)
        captions = ["a dog runs fast", "the cat sits on a mat", "a dog sits"]
        vocab.build(captions)

        encoded = vocab.encode("a dog runs", add_special_tokens=True)
        assert encoded[0] == vocab.word2idx["<START>"]
        assert encoded[-1] == vocab.word2idx["<END>"]

    def test_decode_round_trip(self):
        from data.preprocessing.coco_preprocessor import COCOVocabulary
        vocab = COCOVocabulary(max_vocab_size=200)
        captions = ["a brown dog runs in the park", "two cats sit on a mat"] * 10
        vocab.build(captions)

        original = "a dog runs in the park"
        encoded = vocab.encode(original, add_special_tokens=False)
        decoded = vocab.decode(encoded, skip_special=True)
        assert decoded == original

    def test_oov_handling(self):
        from data.preprocessing.coco_preprocessor import COCOVocabulary
        vocab = COCOVocabulary(max_vocab_size=50)
        vocab.build(["a simple sentence"] * 5)

        encoded = vocab.encode("unknown_word_xyz", add_special_tokens=False)
        assert vocab.word2idx["<OOV>"] in encoded


# ─── Metrics Tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_bleu_perfect_score(self):
        from utils.metrics import compute_corpus_bleu
        refs = [[["a", "dog", "runs", "fast"]]]
        hyps = [["a", "dog", "runs", "fast"]]
        scores = compute_corpus_bleu(refs, hyps)
        assert scores["bleu4"] == pytest.approx(1.0, abs=0.01)

    def test_bleu_zero_score(self):
        from utils.metrics import compute_corpus_bleu
        refs = [[["a", "dog", "runs", "fast"]]]
        hyps = [["cat", "sits", "on", "mat"]]
        scores = compute_corpus_bleu(refs, hyps)
        assert scores["bleu4"] == pytest.approx(0.0, abs=0.1)

    def test_emotion_metrics(self):
        from utils.metrics import compute_emotion_metrics
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])  # Perfect
        metrics = compute_emotion_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)


# ─── API Schema Tests ─────────────────────────────────────────────────────────

class TestAPISchemas:
    def test_emotion_result_validation(self):
        from api.schemas import EmotionResult
        result = EmotionResult(
            bbox=[10, 20, 50, 50],
            emotion="happy",
            confidence=0.92,
            all_scores={"happy": 0.92, "neutral": 0.05, "sad": 0.01,
                        "angry": 0.01, "fear": 0.005, "disgust": 0.005, "surprise": 0.0},
        )
        assert result.emotion == "happy"
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_bounds(self):
        from api.schemas import EmotionResult
        with pytest.raises(Exception):
            EmotionResult(
                bbox=[0, 0, 50, 50],
                emotion="happy",
                confidence=1.5,  # Invalid — > 1.0
                all_scores={},
            )
