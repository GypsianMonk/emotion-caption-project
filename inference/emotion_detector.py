"""
Real-Time Emotion Detector
----------------------------
Performs facial detection + emotion classification on video frames.
Designed for 24+ FPS throughput on CPU (with TFLite) or GPU.

Pipeline per frame:
  1. Detect faces with OpenCV Haar Cascade or DNN detector
  2. Crop + preprocess each face ROI → (48, 48, 1)
  3. Batch predict emotions with CNN
  4. Return annotated frame + structured results

Usage:
    detector = EmotionDetector("checkpoints/emotion/best_model.h5")
    results = detector.predict_frame(frame)   # numpy BGR frame
    annotated = detector.annotate_frame(frame, results)
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS = {
    "angry":    (0,   0,   255),   # Red
    "disgust":  (0,   100, 0),     # Dark green
    "fear":     (128, 0,   128),   # Purple
    "happy":    (0,   255, 255),   # Yellow
    "neutral":  (200, 200, 200),   # Light gray
    "sad":      (255, 0,   0),     # Blue
    "surprise": (0,   165, 255),   # Orange
}


class FaceDetector:
    """
    Wraps OpenCV face detection.
    Uses DNN-based SSD detector (more accurate) with Haar fallback.
    """

    def __init__(self, method: str = "haar"):
        self.method = method
        if method == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
        elif method == "dnn":
            # OpenCV DNN face detector (download required)
            self.detector = self._load_dnn_detector()
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'haar' or 'dnn'")

    def _load_dnn_detector(self):
        model_path = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        proto_path  = "models/face_detector/deploy.prototxt"
        if not Path(model_path).exists():
            logger.warning(
                "DNN face detector weights not found. Falling back to Haar."
                " Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
            )
            self.method = "haar"
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            return cv2.CascadeClassifier(cascade_path)
        return cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if self.method == "haar":
            return self._detect_haar(frame_bgr)
        else:
            return self._detect_dnn(frame_bgr)

    def _detect_haar(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _detect_dnn(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False,
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


class EmotionDetector:
    """
    End-to-end facial emotion recognition for real-time video streams.

    Args:
        model_path:   Path to saved .h5 or SavedModel
        face_method:  'haar' or 'dnn' face detector
        use_tflite:   Load TFLite model for faster CPU inference
        input_size:   Model input size (default 48 for FER-2013 models)
    """

    def __init__(
        self,
        model_path: str,
        face_method: str = "haar",
        use_tflite: bool = False,
        input_size: int = 48,
    ):
        self.input_size = input_size
        self.use_tflite = use_tflite
        self.face_detector = FaceDetector(method=face_method)
        self._fps_buffer = []

        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
            logger.info(f"TFLite model loaded: {model_path}")
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.interpreter = None
            logger.info(f"Keras model loaded: {model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")

    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Resize and normalize a face crop for model input."""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.input_size, self.input_size))
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, self.input_size, self.input_size, 1)

    def _predict_batch(self, face_inputs: np.ndarray) -> np.ndarray:
        """Run batch inference. Returns (N, 7) softmax probabilities."""
        if self.use_tflite:
            results = []
            for i in range(len(face_inputs)):
                self.interpreter.set_tensor(
                    self.input_details[0]["index"],
                    face_inputs[i:i+1]
                )
                self.interpreter.invoke()
                results.append(
                    self.interpreter.get_tensor(self.output_details[0]["index"])[0]
                )
            return np.array(results)
        else:
            return self.model.predict(face_inputs, verbose=0)

    def predict_frame(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Detect all faces in a frame and classify emotions.

        Args:
            frame_bgr: OpenCV BGR frame as numpy array

        Returns:
            List of dicts per detected face:
            {
                "bbox": (x, y, w, h),
                "emotion": "happy",
                "confidence": 0.94,
                "all_scores": {"happy": 0.94, "neutral": 0.04, ...}
            }
        """
        t0 = time.perf_counter()
        faces = self.face_detector.detect(frame_bgr)
        results = []

        if not faces:
            return results

        # Batch process all detected faces
        face_crops = []
        valid_boxes = []
        for x, y, w, h in faces:
            # Clamp to frame bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_bgr.shape[1], x + w)
            y2 = min(frame_bgr.shape[0], y + h)

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue  # Skip tiny detections

            face_roi = frame_bgr[y1:y2, x1:x2]
            face_crops.append(self._preprocess_face(face_roi)[0])
            valid_boxes.append((x1, y1, x2 - x1, y2 - y1))

        if not face_crops:
            return results

        batch = np.stack(face_crops)
        probabilities = self._predict_batch(batch)

        for bbox, probs in zip(valid_boxes, probabilities):
            pred_idx = int(np.argmax(probs))
            results.append({
                "bbox": bbox,
                "emotion": EMOTION_LABELS[pred_idx],
                "confidence": float(probs[pred_idx]),
                "all_scores": {
                    label: float(score)
                    for label, score in zip(EMOTION_LABELS, probs)
                },
            })

        # FPS tracking
        elapsed = time.perf_counter() - t0
        self._fps_buffer.append(elapsed)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)

        return results

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        results: List[Dict],
        show_bar: bool = True,
        show_fps: bool = True,
    ) -> np.ndarray:
        """
        Draw emotion annotations onto the frame.

        Args:
            frame_bgr:  Original BGR frame
            results:    Output from predict_frame()
            show_bar:   Draw confidence bar chart per face
            show_fps:   Overlay FPS counter

        Returns:
            Annotated BGR frame
        """
        annotated = frame_bgr.copy()

        for result in results:
            x, y, w, h = result["bbox"]
            emotion = result["emotion"]
            confidence = result["confidence"]
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label background
            label = f"{emotion} {confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x, y - lh - 10), (x + lw + 6, y), color, -1)
            cv2.putText(
                annotated, label,
                (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

            # Confidence bars for top-3 emotions
            if show_bar:
                sorted_emotions = sorted(
                    result["all_scores"].items(), key=lambda e: e[1], reverse=True
                )[:3]
                bar_x = x + w + 10
                bar_y = y
                for i, (emo, score) in enumerate(sorted_emotions):
                    bar_len = int(score * 80)
                    bar_color = EMOTION_COLORS.get(emo, (200, 200, 200))
                    cv2.rectangle(
                        annotated,
                        (bar_x, bar_y + i * 20),
                        (bar_x + bar_len, bar_y + i * 20 + 14),
                        bar_color, -1
                    )
                    cv2.putText(
                        annotated, f"{emo[:4]} {score:.0%}",
                        (bar_x + bar_len + 4, bar_y + i * 20 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA,
                    )

        # FPS overlay
        if show_fps and self._fps_buffer:
            avg_fps = 1.0 / (sum(self._fps_buffer) / len(self._fps_buffer))
            cv2.putText(
                annotated, f"FPS: {avg_fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

        return annotated

    def get_fps(self) -> float:
        """Return current moving average FPS."""
        if not self._fps_buffer:
            return 0.0
        return 1.0 / (sum(self._fps_buffer) / len(self._fps_buffer))
