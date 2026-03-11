"""
Pydantic Schemas for FastAPI request/response validation.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class EmotionResult(BaseModel):
    bbox: List[int] = Field(..., description="Bounding box [x, y, w, h]")
    emotion: str    = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    all_scores: Dict[str, float] = Field(..., description="Scores for all 7 emotion classes")

    model_config = {"json_schema_extra": {"example": {
        "bbox": [120, 80, 64, 64],
        "emotion": "happy",
        "confidence": 0.934,
        "all_scores": {
            "angry": 0.01, "disgust": 0.002, "fear": 0.005,
            "happy": 0.934, "neutral": 0.04, "sad": 0.007, "surprise": 0.002
        }
    }}}


class EmotionResponse(BaseModel):
    faces_detected: int = Field(..., description="Number of faces detected")
    faces: List[EmotionResult] = Field(..., description="Per-face emotion results")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    image_size: List[int] = Field(..., description="Input image [height, width]")


class CaptionResponse(BaseModel):
    caption: str = Field(..., description="Generated image caption")
    beam_width: int = Field(..., description="Beam search width used")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    image_size: List[int] = Field(..., description="Input image [height, width]")

    model_config = {"json_schema_extra": {"example": {
        "caption": "a dog running in a park with a frisbee in its mouth",
        "beam_width": 3,
        "latency_ms": 342.1,
        "image_size": [480, 640]
    }}}


class PipelineResponse(BaseModel):
    caption: Optional[str] = Field(None, description="Generated caption (null if model not loaded)")
    faces_detected: int = Field(..., description="Number of faces detected")
    faces: List[EmotionResult] = Field(default_factory=list)
    beam_width: int = Field(..., description="Beam search width")
    total_latency_ms: float = Field(..., description="Total pipeline latency in ms")
    image_size: List[int] = Field(..., description="Input image [height, width]")


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' or 'degraded'")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    emotion_model_loaded: bool = Field(..., description="Emotion model availability")
    caption_model_loaded: bool = Field(..., description="Caption model availability")
    version: str = Field(..., description="API version")
