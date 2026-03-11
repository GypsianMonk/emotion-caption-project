"""
FastAPI REST API
-----------------
Production REST endpoints for emotion detection and image captioning.
Handles image uploads, runs inference, and returns structured JSON responses.

Endpoints:
    POST /emotion          — Detect emotions in uploaded image
    POST /caption          — Generate caption for uploaded image
    POST /pipeline         — Run both emotion + caption
    GET  /health           — Health check + model status
    GET  /metrics          — Current performance metrics
"""

import io
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    EmotionResponse, CaptionResponse, PipelineResponse,
    HealthResponse, EmotionResult
)

logger = logging.getLogger(__name__)

# Global model instances (initialized on startup)
_emotion_detector = None
_caption_generator = None
_start_time = time.time()
_request_counts = {"emotion": 0, "caption": 0, "pipeline": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _emotion_detector, _caption_generator

    import os
    emotion_ckpt = os.getenv("EMOTION_CKPT", "checkpoints/emotion/best_model.h5")
    encoder_ckpt = os.getenv("ENCODER_CKPT", "")
    decoder_ckpt = os.getenv("DECODER_CKPT", "")
    vocab_path   = os.getenv("VOCAB_PATH", "")

    # Load emotion model
    if Path(emotion_ckpt).exists():
        from inference.emotion_detector import EmotionDetector
        _emotion_detector = EmotionDetector(
            model_path=emotion_ckpt,
            face_method=os.getenv("FACE_METHOD", "haar"),
        )
        logger.info("Emotion detector loaded")
    else:
        logger.warning(f"Emotion model not found at {emotion_ckpt}")

    # Load caption model
    if encoder_ckpt and decoder_ckpt and vocab_path:
        if Path(vocab_path).exists():
            from inference.caption_generator import CaptionGenerator
            _caption_generator = CaptionGenerator.from_checkpoints(
                encoder_ckpt=encoder_ckpt,
                decoder_ckpt=decoder_ckpt,
                vocab_path=vocab_path,
            )
            logger.info("Caption generator loaded")
        else:
            logger.warning(f"Vocab path not found: {vocab_path}")

    yield  # API is running

    # Cleanup
    _emotion_detector = None
    _caption_generator = None
    logger.info("Models unloaded")


app = FastAPI(
    title="Emotion Detection & Image Captioning API",
    description=(
        "Production API for real-time facial emotion recognition (82% acc, 7 classes) "
        "and image captioning (BLEU-4: 0.28) using InceptionV3+LSTM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR numpy array."""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(pil_img)
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model availability."""
    uptime = time.time() - _start_time
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 1),
        emotion_model_loaded=_emotion_detector is not None,
        caption_model_loaded=_caption_generator is not None,
        version="1.0.0",
    )


@app.get("/metrics")
async def get_metrics():
    """Return request counts and current FPS (emotion detector)."""
    fps = _emotion_detector.get_fps() if _emotion_detector else None
    return {
        "request_counts": _request_counts,
        "emotion_fps": round(fps, 2) if fps else None,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.post("/emotion", response_model=EmotionResponse)
async def detect_emotion(file: UploadFile = File(..., description="Image file (JPEG/PNG)")):
    """
    Detect facial emotions in an uploaded image.

    Returns bounding boxes, emotion labels, and confidence scores
    for each detected face.
    """
    if _emotion_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion model not loaded. Check EMOTION_CKPT environment variable."
        )

    contents = await file.read()
    frame = _read_image(contents)

    t0 = time.perf_counter()
    results = _emotion_detector.predict_frame(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    _request_counts["emotion"] += 1

    faces = [
        EmotionResult(
            bbox=list(r["bbox"]),
            emotion=r["emotion"],
            confidence=round(r["confidence"], 4),
            all_scores={k: round(v, 4) for k, v in r["all_scores"].items()},
        )
        for r in results
    ]

    return EmotionResponse(
        faces_detected=len(faces),
        faces=faces,
        latency_ms=round(latency_ms, 2),
        image_size=list(frame.shape[:2]),
    )


@app.post("/caption", response_model=CaptionResponse)
async def generate_caption(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    beam_width: int = 3,
):
    """
    Generate a natural language caption for an uploaded image.
    Uses beam search (beam_width=3 by default) for best quality.
    """
    if _caption_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Caption model not loaded. Check ENCODER_CKPT/DECODER_CKPT env vars."
        )

    if not 1 <= beam_width <= 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="beam_width must be between 1 and 5"
        )

    contents = await file.read()
    frame = _read_image(contents)
    rgb_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    caption = _caption_generator.caption_array(rgb_array, beam_width=beam_width)
    latency_ms = (time.perf_counter() - t0) * 1000

    _request_counts["caption"] += 1

    return CaptionResponse(
        caption=caption,
        beam_width=beam_width,
        latency_ms=round(latency_ms, 2),
        image_size=list(frame.shape[:2]),
    )


@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    beam_width: int = 3,
):
    """
    Run both emotion detection and image captioning in a single call.
    Most efficient for applications that need both outputs.
    """
    contents = await file.read()
    frame = _read_image(contents)

    _request_counts["pipeline"] += 1
    t0 = time.perf_counter()

    # Emotion detection
    emotion_results = []
    if _emotion_detector is not None:
        raw = _emotion_detector.predict_frame(frame)
        emotion_results = [
            EmotionResult(
                bbox=list(r["bbox"]),
                emotion=r["emotion"],
                confidence=round(r["confidence"], 4),
                all_scores={k: round(v, 4) for k, v in r["all_scores"].items()},
            )
            for r in raw
        ]

    # Caption generation
    caption = None
    if _caption_generator is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        caption = _caption_generator.caption_array(rgb, beam_width=beam_width)

    total_latency = (time.perf_counter() - t0) * 1000

    return PipelineResponse(
        caption=caption,
        faces_detected=len(emotion_results),
        faces=emotion_results,
        beam_width=beam_width,
        total_latency_ms=round(total_latency, 2),
        image_size=list(frame.shape[:2]),
    )
