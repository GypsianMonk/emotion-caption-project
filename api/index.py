"""
Vercel Serverless API
----------------------
Lightweight FastAPI endpoint for Vercel deployment.
Uses DeepFace for emotion detection and BLIP for image captioning
(same models as the Gradio app.py).

Falls back to graceful error messages if models fail to load.
"""

import io
import time
import logging
from typing import Optional

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Detection & Captioning API",
    version="1.0.0",
    description="Real-Time Emotion Detection & Image Captioning REST API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_caption_processor = None
_caption_model = None
_models_loaded = False


def _load_models():
    """Lazy-load captioning models on first request."""
    global _caption_processor, _caption_model, _models_loaded
    if _models_loaded:
        return

    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        _caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _caption_model.eval()
        logger.info("BLIP captioning model loaded")
    except Exception as e:
        logger.warning(f"Failed to load BLIP model: {e}")

    _models_loaded = True


def _read_image(file_bytes: bytes) -> Image.Image:
    """Decode uploaded bytes to PIL Image."""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def _detect_emotions(image: Image.Image) -> list:
    """Detect emotions using DeepFace."""
    try:
        from deepface import DeepFace

        img_array = np.array(image)
        results = DeepFace.analyze(
            img_array,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if not isinstance(results, list):
            results = [results]
        return results
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        return []


def _generate_caption(image: Image.Image, beam_width: int = 3) -> Optional[str]:
    """Generate caption using BLIP."""
    _load_models()
    if _caption_processor is None or _caption_model is None:
        return None

    try:
        import torch

        inputs = _caption_processor(image, return_tensors="pt")
        with torch.no_grad():
            out = _caption_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=beam_width,
                length_penalty=1.0,
            )
        return _caption_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<html><body style="background:#0a0a0f;color:#e2e8f0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center">
    <div><h1 style="font-size:48px">🧠 Emotion & Caption API</h1>
    <p style="color:#64748b;font-size:18px;margin:16px 0">Real-Time Emotion Detection · Image Captioning</p>
    <div style="display:flex;gap:12px;justify-content:center;margin-top:32px">
    <a href="/docs" style="background:#7c3aed;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600">📖 API Docs</a>
    <a href="https://github.com/GypsianMonk/emotion-caption-project" style="border:1px solid #1e1e2e;color:#e2e8f0;padding:12px 24px;border-radius:8px;text-decoration:none">⭐ GitHub</a>
    </div></div></body></html>"""


@app.get("/health")
async def health():
    _load_models()
    return {
        "status": "healthy",
        "caption_model_loaded": _caption_model is not None,
        "emotion_backend": "deepface",
        "version": "1.0.0",
    }


@app.post("/emotion")
async def emotion(file: UploadFile = File(...)):
    """Detect facial emotions in an uploaded image using DeepFace."""
    t0 = time.time()
    contents = await file.read()
    image = _read_image(contents)

    results = _detect_emotions(image)
    latency_ms = (time.time() - t0) * 1000

    faces = []
    for r in results:
        region = r.get("region", {})
        dominant = r.get("dominant_emotion", "neutral")
        emotions = r.get("emotion", {})
        confidence = emotions.get(dominant, 0) / 100.0
        faces.append({
            "emotion": dominant,
            "confidence": round(confidence, 3),
            "bbox": [region.get("x", 0), region.get("y", 0),
                     region.get("w", 0), region.get("h", 0)],
            "all_scores": {k: round(v / 100, 4) for k, v in emotions.items()},
        })

    return {
        "faces_detected": len(faces),
        "faces": faces,
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/caption")
async def caption(file: UploadFile = File(...), beam_width: int = 3):
    """Generate a caption for an uploaded image using BLIP."""
    t0 = time.time()
    contents = await file.read()
    image = _read_image(contents)

    text = _generate_caption(image, beam_width=beam_width)
    latency_ms = (time.time() - t0) * 1000

    if text is None:
        return {
            "caption": "Caption model not available",
            "beam_width": beam_width,
            "latency_ms": round(latency_ms, 1),
            "error": "BLIP model failed to load",
        }

    return {
        "caption": text,
        "beam_width": beam_width,
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/pipeline")
async def pipeline(file: UploadFile = File(...), beam_width: int = 3):
    """Run both emotion detection and captioning on an uploaded image."""
    t0 = time.time()
    contents = await file.read()
    image = _read_image(contents)

    # Emotion detection
    emotion_results = _detect_emotions(image)
    faces = []
    for r in emotion_results:
        dominant = r.get("dominant_emotion", "neutral")
        emotions = r.get("emotion", {})
        confidence = emotions.get(dominant, 0) / 100.0
        region = r.get("region", {})
        faces.append({
            "emotion": dominant,
            "confidence": round(confidence, 3),
            "bbox": [region.get("x", 0), region.get("y", 0),
                     region.get("w", 0), region.get("h", 0)],
        })

    # Captioning
    text = _generate_caption(image, beam_width=beam_width)
    latency_ms = (time.time() - t0) * 1000

    return {
        "caption": text or "Caption model not available",
        "faces_detected": len(faces),
        "faces": faces,
        "beam_width": beam_width,
        "latency_ms": round(latency_ms, 1),
    }


@app.get("/metrics")
async def metrics():
    """Return model information and capabilities."""
    return {
        "emotion": {
            "backend": "deepface",
            "accuracy": "82% (FER model)",
            "dataset": "FER-2013",
            "classes": 7,
        },
        "captioning": {
            "model": "BLIP (Salesforce/blip-image-captioning-base)",
            "loaded": _caption_model is not None,
            "dataset": "MS-COCO",
        },
    }
