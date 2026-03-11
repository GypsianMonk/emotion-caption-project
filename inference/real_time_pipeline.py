"""
Real-Time Unified Pipeline
----------------------------
Combines EmotionDetector + CaptionGenerator into a single OpenCV webcam loop.
Runs emotion detection every frame (fast), caption generation periodically (slow).

Controls:
    Q / ESC   — quit
    C         — generate caption for current frame
    S         — save screenshot
    SPACE     — pause/resume
    R         — reset caption

Usage:
    python inference/real_time_pipeline.py \
        --emotion_ckpt checkpoints/emotion/best_model.h5 \
        --encoder_ckpt checkpoints/captioning/encoder_weights \
        --decoder_ckpt checkpoints/captioning/decoder_weights \
        --vocab_path   data/processed/tokenizer.pkl
"""

import cv2
import numpy as np
import argparse
import time
import threading
from pathlib import Path
from typing import Optional
import logging

from inference.emotion_detector import EmotionDetector
from inference.caption_generator import CaptionGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class RealTimePipeline:
    """
    Production-ready real-time inference pipeline.

    Emotion detection runs synchronously every frame for low latency.
    Caption generation runs in a background thread to avoid blocking.
    """

    WINDOW_NAME = "Emotion Detection & Image Captioning"
    CAPTION_INTERVAL = 5.0  # seconds between auto-caption updates

    def __init__(
        self,
        emotion_detector: EmotionDetector,
        caption_generator: Optional[CaptionGenerator] = None,
        camera_index: int = 0,
        target_fps: int = 30,
        auto_caption: bool = False,
    ):
        self.emotion_detector = emotion_detector
        self.caption_generator = caption_generator
        self.camera_index = camera_index
        self.frame_delay = int(1000 / target_fps)
        self.auto_caption = auto_caption

        # State
        self.current_caption: str = "Press 'C' to generate caption"
        self.caption_generating: bool = False
        self.paused: bool = False
        self.last_caption_time: float = 0.0
        self._caption_thread: Optional[threading.Thread] = None

    def _generate_caption_async(self, frame: np.ndarray):
        """Generate caption in background thread to avoid blocking video."""
        if self.caption_generator is None:
            return
        if self.caption_generating:
            return

        def _worker():
            self.caption_generating = True
            self.current_caption = "Generating..."
            try:
                caption = self.caption_generator.caption_array(frame, beam_width=3)
                self.current_caption = caption
                logger.info(f"Caption: {caption}")
            except Exception as e:
                self.current_caption = "Caption error"
                logger.error(f"Caption generation failed: {e}")
            finally:
                self.caption_generating = False
                self.last_caption_time = time.time()

        self._caption_thread = threading.Thread(target=_worker, daemon=True)
        self._caption_thread.start()

    def _draw_caption_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render caption text with wrapped display at bottom of frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent caption bar
        bar_height = 60
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Caption text (truncate if too long)
        caption = self.current_caption
        if len(caption) > 80:
            caption = caption[:77] + "..."

        status = " ⏳" if self.caption_generating else ""
        text = f"Caption: {caption}{status}"

        cv2.putText(
            frame, text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return frame

    def _draw_controls_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render keyboard controls hint."""
        controls = "[Q] Quit  [C] Caption  [SPACE] Pause  [S] Save"
        cv2.putText(
            frame, controls,
            (10, frame.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (180, 180, 180), 1, cv2.LINE_AA,
        )
        return frame

    def run(self):
        """Main video loop."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        logger.info(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
                    f"{int(cap.get(cv2.CAP_PROP_FPS))}fps")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)

        screenshot_counter = 0
        Path("outputs/screenshots").mkdir(parents=True, exist_ok=True)

        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Frame read failed")
                        break

                    # Emotion detection (every frame)
                    results = self.emotion_detector.predict_frame(frame)
                    frame = self.emotion_detector.annotate_frame(frame, results)

                    # Auto-caption trigger
                    if (
                        self.auto_caption
                        and self.caption_generator is not None
                        and not self.caption_generating
                        and time.time() - self.last_caption_time > self.CAPTION_INTERVAL
                    ):
                        self._generate_caption_async(frame.copy())

                    frame = self._draw_caption_overlay(frame)
                    frame = self._draw_controls_overlay(frame)

                else:
                    # Paused — display PAUSED overlay
                    cv2.putText(
                        frame, "PAUSED",
                        (frame.shape[1] // 2 - 60, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 255, 255), 3, cv2.LINE_AA,
                    )

                cv2.imshow(self.WINDOW_NAME, frame)

                key = cv2.waitKey(self.frame_delay) & 0xFF
                if key in (ord("q"), 27):  # Q or ESC
                    break
                elif key == ord("c"):
                    self._generate_caption_async(frame.copy())
                elif key == ord(" "):
                    self.paused = not self.paused
                    logger.info("Paused" if self.paused else "Resumed")
                elif key == ord("s"):
                    path = f"outputs/screenshots/frame_{screenshot_counter:04d}.jpg"
                    cv2.imwrite(path, frame)
                    logger.info(f"Screenshot saved: {path}")
                    screenshot_counter += 1
                elif key == ord("r"):
                    self.current_caption = "Caption cleared"

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Pipeline stopped")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Emotion + Caption Pipeline")
    parser.add_argument("--emotion_ckpt",  required=True, help="Path to emotion model (.h5)")
    parser.add_argument("--encoder_ckpt",  default=None,  help="Encoder checkpoint path")
    parser.add_argument("--decoder_ckpt",  default=None,  help="Decoder checkpoint path")
    parser.add_argument("--vocab_path",    default=None,  help="Vocabulary pickle path")
    parser.add_argument("--camera",        type=int, default=0, help="Camera device index")
    parser.add_argument("--face_method",   default="haar", choices=["haar", "dnn"])
    parser.add_argument("--use_tflite",    action="store_true", help="Use TFLite emotion model")
    parser.add_argument("--auto_caption",  action="store_true", help="Auto-generate captions")
    parser.add_argument("--beam_width",    type=int, default=3)
    args = parser.parse_args()

    # Initialize emotion detector
    emotion_detector = EmotionDetector(
        model_path=args.emotion_ckpt,
        face_method=args.face_method,
        use_tflite=args.use_tflite,
    )
    logger.info("Emotion detector initialized")

    # Initialize caption generator (optional)
    caption_generator = None
    if args.encoder_ckpt and args.decoder_ckpt and args.vocab_path:
        caption_generator = CaptionGenerator.from_checkpoints(
            encoder_ckpt=args.encoder_ckpt,
            decoder_ckpt=args.decoder_ckpt,
            vocab_path=args.vocab_path,
        )
        logger.info("Caption generator initialized")
    else:
        logger.warning("Caption generator not loaded (missing --encoder_ckpt/--decoder_ckpt/--vocab_path)")

    pipeline = RealTimePipeline(
        emotion_detector=emotion_detector,
        caption_generator=caption_generator,
        camera_index=args.camera,
        auto_caption=args.auto_caption,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
