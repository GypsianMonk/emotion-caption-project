"""
Vercel Serverless API
----------------------
Lightweight FastAPI endpoint for Vercel deployment.

Since Vercel's serverless environment has strict size limits (~250MB),
heavy ML models (PyTorch, TensorFlow, DeepFace) cannot be deployed here.
This endpoint provides:
  - API documentation and schema reference
  - A demo mode with realistic mock responses
  - Links to the full Hugging Face Spaces deployment for real inference

For real inference, use:
  - Gradio app: python app.py (or HF Spaces deployment)
  - Full API: uvicorn api.main:app (with models loaded locally)
"""

import io
import time
import logging
import random

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Detection & Captioning API",
    version="1.0.0",
    description=(
        "Real-Time Emotion Detection & Image Captioning REST API.\n\n"
        "**This Vercel deployment runs in demo mode.** "
        "For real inference, deploy with `uvicorn api.main:app` or use the Gradio app."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Demo data ─────────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEMO_CAPTIONS = [
    "a person smiling at the camera in a bright room",
    "a group of friends laughing together outdoors",
    "a woman working at her desk with a laptop",
    "a man standing outside a modern building",
    "a child playing with a dog in the park",
    "two people having a conversation over coffee",
    "a family gathering around a dinner table",
    "a person looking thoughtfully out of a window",
]

_real_inference = False

# Try to load real models (will succeed locally, fail on Vercel)
try:
    import numpy as np
    from PIL import Image

    _has_imaging = True
except ImportError:
    _has_imaging = False

try:
    from deepface import DeepFace
    _has_deepface = True
except ImportError:
    _has_deepface = False

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _has_blip = True
except ImportError:
    _has_blip = False

# Lazy model state
_caption_processor = None
_caption_model = None
_models_attempted = False


def _try_load_caption_model():
    """Attempt to load BLIP model (succeeds locally, fails on Vercel)."""
    global _caption_processor, _caption_model, _models_attempted, _real_inference
    if _models_attempted:
        return
    _models_attempted = True

    if not (_has_blip and _has_imaging):
        return

    try:
        _caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _caption_model.eval()
        _real_inference = True
        logger.info("BLIP captioning model loaded — real inference enabled")
    except Exception as e:
        logger.info(f"BLIP model not available, using demo mode: {e}")


def _generate_demo_emotion():
    """Generate a realistic demo emotion result."""
    dominant = random.choice(EMOTIONS)
    scores = {}
    remaining = 100.0
    for emo in EMOTIONS:
        if emo == dominant:
            score = random.uniform(65, 95)
            remaining -= score
            scores[emo] = round(score, 1)
        else:
            score = random.uniform(0, remaining / (len(EMOTIONS) - 1))
            remaining -= score
            scores[emo] = round(score, 1)
    return dominant, scores


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    mode = "🟢 Real Inference" if _real_inference else "🟡 Demo Mode"
    return f"""<html><body style="background:#0a0a0f;color:#e2e8f0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center">
    <div><h1 style="font-size:48px">🧠 Emotion & Caption API</h1>
    <p style="color:#64748b;font-size:18px;margin:16px 0">Real-Time Emotion Detection · Image Captioning</p>
    <p style="color:#94a3b8;font-size:14px;margin:8px 0">{mode}</p>
    <div style="display:flex;gap:12px;justify-content:center;margin-top:32px">
    <a href="/docs" style="background:#7c3aed;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600">📖 API Docs</a>
    <a href="https://github.com/GypsianMonk/emotion-caption-project" style="border:1px solid #1e1e2e;color:#e2e8f0;padding:12px 24px;border-radius:8px;text-decoration:none">⭐ GitHub</a>
    </div></div></body></html>"""


@app.get("/health")
async def health():
    _try_load_caption_model()
    return {
        "status": "healthy",
        "mode": "real" if _real_inference else "demo",
        "deepface_available": _has_deepface,
        "blip_available": _has_blip and _caption_model is not None,
        "version": "1.0.0",
    }


@app.post("/emotion")
async def emotion(file: UploadFile = File(...)):
    """Detect facial emotions in an uploaded image."""
    t0 = time.time()
    contents = await file.read()

    # Real inference path
    if _has_deepface and _has_imaging:
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img_array = np.array(image)
            results = DeepFace.analyze(
                img_array, actions=["emotion"],
                enforce_detection=False, silent=True,
            )
            if not isinstance(results, list):
                results = [results]

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
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "mode": "real",
            }
        except Exception as e:
            logger.error(f"Real emotion detection failed: {e}")

    # Demo fallback
    dominant, scores = _generate_demo_emotion()
    return {
        "faces_detected": 1,
        "faces": [{
            "emotion": dominant,
            "confidence": round(scores[dominant] / 100, 3),
            "bbox": [120, 80, 64, 64],
            "all_scores": {k: round(v / 100, 4) for k, v in scores.items()},
        }],
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "mode": "demo",
    }


@app.post("/caption")
async def caption(file: UploadFile = File(...), beam_width: int = 3):
    """Generate a caption for an uploaded image."""
    t0 = time.time()
    contents = await file.read()

    _try_load_caption_model()

    # Real inference path
    if _real_inference and _has_imaging:
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            inputs = _caption_processor(image, return_tensors="pt")
            with torch.no_grad():
                out = _caption_model.generate(
                    **inputs, max_new_tokens=50,
                    num_beams=beam_width, length_penalty=1.0,
                )
            text = _caption_processor.decode(out[0], skip_special_tokens=True)
            return {
                "caption": text,
                "beam_width": beam_width,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "mode": "real",
            }
        except Exception as e:
            logger.error(f"Real caption generation failed: {e}")

    # Demo fallback
    return {
        "caption": random.choice(DEMO_CAPTIONS),
        "beam_width": beam_width,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "mode": "demo",
    }


@app.post("/pipeline")
async def pipeline(file: UploadFile = File(...), beam_width: int = 3):
    """Run both emotion detection and captioning on an uploaded image."""
    t0 = time.time()
    contents = await file.read()

    _try_load_caption_model()

    # Try real inference
    faces = []
    caption_text = None

    if _has_deepface and _has_imaging:
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img_array = np.array(image)
            results = DeepFace.analyze(
                img_array, actions=["emotion"],
                enforce_detection=False, silent=True,
            )
            if not isinstance(results, list):
                results = [results]
            for r in results:
                dominant = r.get("dominant_emotion", "neutral")
                emotions = r.get("emotion", {})
                region = r.get("region", {})
                faces.append({
                    "emotion": dominant,
                    "confidence": round(emotions.get(dominant, 0) / 100, 3),
                    "bbox": [region.get("x", 0), region.get("y", 0),
                             region.get("w", 0), region.get("h", 0)],
                })
        except Exception:
            pass

    if _real_inference and _has_imaging:
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            inputs = _caption_processor(image, return_tensors="pt")
            with torch.no_grad():
                out = _caption_model.generate(
                    **inputs, max_new_tokens=50,
                    num_beams=beam_width, length_penalty=1.0,
                )
            caption_text = _caption_processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            pass

    # Fill demo fallbacks
    if not faces:
        dominant, scores = _generate_demo_emotion()
        faces = [{"emotion": dominant, "confidence": round(scores[dominant] / 100, 3)}]

    if caption_text is None:
        caption_text = random.choice(DEMO_CAPTIONS)

    return {
        "caption": caption_text,
        "faces_detected": len(faces),
        "faces": faces,
        "beam_width": beam_width,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "mode": "real" if _real_inference else "demo",
    }


@app.get("/metrics")
async def metrics():
    """Return model information and capabilities."""
    return {
        "emotion": {
            "backend": "deepface" if _has_deepface else "demo",
            "accuracy": "82% (FER model)",
            "dataset": "FER-2013",
            "classes": 7,
        },
        "captioning": {
            "model": "BLIP (Salesforce/blip-image-captioning-base)",
            "loaded": _caption_model is not None,
            "dataset": "MS-COCO",
        },
        "mode": "real" if _real_inference else "demo",
    }
