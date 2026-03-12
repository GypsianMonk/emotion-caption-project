"""
API Endpoint Tests
-------------------
Tests for the FastAPI REST API using TestClient.
Uses the lightweight index.py API (Vercel deployment version).

Run: pytest tests/test_api.py -v
"""

import pytest
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Create a FastAPI TestClient for the API."""
    from fastapi.testclient import TestClient
    from api.index import app
    return TestClient(app)


@pytest.fixture
def dummy_image_bytes():
    """Create a minimal valid JPEG image in memory."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


# ─── Health Check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"


# ─── Root Endpoint ───────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_root_contains_title(self, client):
        response = client.get("/")
        assert "Emotion" in response.text


# ─── Emotion Endpoint ────────────────────────────────────────────────────────

class TestEmotionEndpoint:
    def test_emotion_returns_200(self, client, dummy_image_bytes):
        response = client.post(
            "/emotion",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_emotion_response_structure(self, client, dummy_image_bytes):
        data = client.post(
            "/emotion",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        ).json()
        assert "faces_detected" in data
        assert "faces" in data
        assert isinstance(data["faces"], list)

    def test_emotion_face_has_required_fields(self, client, dummy_image_bytes):
        data = client.post(
            "/emotion",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        ).json()
        if data["faces"]:
            face = data["faces"][0]
            assert "emotion" in face
            assert "confidence" in face


# ─── Caption Endpoint ────────────────────────────────────────────────────────

class TestCaptionEndpoint:
    def test_caption_returns_200(self, client, dummy_image_bytes):
        response = client.post(
            "/caption",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_caption_response_has_caption(self, client, dummy_image_bytes):
        data = client.post(
            "/caption",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        ).json()
        assert "caption" in data
        assert isinstance(data["caption"], str)
        assert len(data["caption"]) > 0

    def test_caption_custom_beam_width(self, client, dummy_image_bytes):
        data = client.post(
            "/caption?beam_width=1",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        ).json()
        assert data["beam_width"] == 1


# ─── Pipeline Endpoint ───────────────────────────────────────────────────────

class TestPipelineEndpoint:
    def test_pipeline_returns_200(self, client, dummy_image_bytes):
        response = client.post(
            "/pipeline",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_pipeline_has_both_outputs(self, client, dummy_image_bytes):
        data = client.post(
            "/pipeline",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        ).json()
        assert "caption" in data
        assert "faces_detected" in data


# ─── Metrics Endpoint ────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_all_tasks(self, client):
        data = client.get("/metrics").json()
        assert "emotion" in data
        assert "captioning" in data
