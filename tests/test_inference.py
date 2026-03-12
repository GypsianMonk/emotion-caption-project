"""
Inference Module Tests
-----------------------
Tests for EmotionDetector, CaptionGenerator, and FaceDetector.
Uses dummy models and mock data to verify inference pipelines.

Run: pytest tests/test_inference.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── FaceDetector Tests ──────────────────────────────────────────────────────

class TestFaceDetector:
    def test_haar_detector_initializes(self):
        from inference.emotion_detector import FaceDetector
        detector = FaceDetector(method="haar")
        assert detector.method == "haar"
        assert detector.detector is not None

    def test_invalid_method_raises(self):
        from inference.emotion_detector import FaceDetector
        with pytest.raises(ValueError, match="Unknown method"):
            FaceDetector(method="invalid_method")

    def test_haar_detect_returns_list(self):
        from inference.emotion_detector import FaceDetector
        detector = FaceDetector(method="haar")
        # Create a blank image (no faces expected)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        assert isinstance(result, list)

    def test_haar_detect_on_face_like_pattern(self):
        """Verify detect returns bounding box tuples."""
        from inference.emotion_detector import FaceDetector
        detector = FaceDetector(method="haar")
        # Use a white image — unlikely to detect faces, but verifies return type
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = detector.detect(white)
        assert isinstance(result, list)
        # Each result should be a tuple of 4 ints if any face found
        for box in result:
            assert len(box) == 4


# ─── EmotionDetector Tests ───────────────────────────────────────────────────

class TestEmotionDetector:
    @pytest.fixture
    def mock_model(self):
        """Create a mock Keras model that returns dummy predictions."""
        model = MagicMock()
        # Return a 7-class softmax-like output
        model.predict.return_value = np.array([[0.1, 0.05, 0.05, 0.5, 0.15, 0.05, 0.1]])
        model.input_shape = (None, 48, 48, 1)
        return model

    def test_preprocess_face(self):
        """Verify face preprocessing outputs correct shape."""
        from inference.emotion_detector import EmotionDetector

        with patch("tensorflow.keras.models.load_model") as mock_load:
            mock_load.return_value = MagicMock()
            detector = EmotionDetector.__new__(EmotionDetector)
            detector.input_size = 48
            detector.use_tflite = False

        # Create a dummy BGR face crop
        face_roi = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        processed = detector._preprocess_face(face_roi)

        assert processed.shape == (1, 48, 48, 1)
        assert processed.dtype == np.float32
        assert 0.0 <= processed.min() and processed.max() <= 1.0

    def test_annotate_frame_returns_same_shape(self):
        """Verify annotation doesn't change frame dimensions."""
        from inference.emotion_detector import EmotionDetector

        detector = EmotionDetector.__new__(EmotionDetector)
        detector._fps_buffer = [0.03, 0.04]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{
            "bbox": (100, 100, 80, 80),
            "emotion": "happy",
            "confidence": 0.92,
            "all_scores": {"happy": 0.92, "neutral": 0.05, "sad": 0.03},
        }]

        annotated = detector.annotate_frame(frame, results)
        assert annotated.shape == frame.shape
        # Annotated frame should not be same object
        assert annotated is not frame

    def test_get_fps_empty_buffer(self):
        """FPS should be 0 with empty buffer."""
        from inference.emotion_detector import EmotionDetector
        detector = EmotionDetector.__new__(EmotionDetector)
        detector._fps_buffer = []
        assert detector.get_fps() == 0.0

    def test_get_fps_with_data(self):
        """FPS should be calculated from buffer."""
        from inference.emotion_detector import EmotionDetector
        detector = EmotionDetector.__new__(EmotionDetector)
        # 40ms per frame = 25 FPS
        detector._fps_buffer = [0.04] * 10
        fps = detector.get_fps()
        assert abs(fps - 25.0) < 0.1


# ─── CaptionGenerator Tests ────────────────────────────────────────────────

class TestCaptionGenerator:
    def test_greedy_decode_produces_string(self):
        """Verify greedy decoding returns a non-empty string."""
        import tensorflow as tf
        from inference.caption_generator import CaptionGenerator
        from data.preprocessing.coco_preprocessor import COCOVocabulary

        # Build a minimal vocabulary
        vocab = COCOVocabulary(max_vocab_size=50)
        vocab.build(["a dog runs", "a cat sits", "the bird flies"] * 5)

        # Create a mock encoder and decoder
        mock_encoder = MagicMock()
        mock_decoder = MagicMock()

        # Mock decoder reset_state
        mock_decoder.reset_state.return_value = (
            tf.zeros((1, 128)),
            tf.zeros((1, 128)),
        )

        # Mock decoder call — return END token logits after 3 steps
        end_id = vocab.word2idx[COCOVocabulary.END_TOKEN]
        call_count = [0]

        def mock_call(inputs, states=None, training=False):
            call_count[0] += 1
            logits = tf.zeros((1, len(vocab)))
            if call_count[0] >= 3:
                # Generate END token
                logits_np = np.zeros((1, len(vocab)), dtype=np.float32)
                logits_np[0, end_id] = 10.0
                logits = tf.constant(logits_np)
            else:
                # Generate a word token
                logits_np = np.zeros((1, len(vocab)), dtype=np.float32)
                logits_np[0, 4] = 10.0  # First real word
                logits = tf.constant(logits_np)
            return logits, tf.zeros((1, 128)), tf.zeros((1, 128))

        mock_decoder.side_effect = mock_call

        generator = CaptionGenerator(
            encoder=mock_encoder,
            decoder=mock_decoder,
            vocab=vocab,
            max_length=10,
        )

        features = tf.zeros((1, 512))
        caption = generator.greedy_decode(features)
        assert isinstance(caption, str)
        assert len(caption) > 0


# ─── Visualization Tests ────────────────────────────────────────────────────

class TestVisualization:
    def test_draw_emotion_overlay(self):
        from utils.visualization import draw_emotion_overlay
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{
            "bbox": (100, 100, 80, 80),
            "emotion": "happy",
            "confidence": 0.9,
            "all_scores": {"happy": 0.9, "sad": 0.05, "neutral": 0.05},
        }]
        annotated = draw_emotion_overlay(frame, results)
        assert annotated.shape == frame.shape

    def test_draw_caption_bar(self):
        from utils.visualization import draw_caption_bar
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_caption_bar(frame, "a cat sitting on a mat")
        assert result.shape == frame.shape

    def test_draw_emotion_overlay_empty(self):
        from utils.visualization import draw_emotion_overlay
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated = draw_emotion_overlay(frame, [])
        assert np.array_equal(annotated, frame)
