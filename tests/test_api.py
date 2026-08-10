"""
Tests for the FastAPI inference service.

These tests validate the main API endpoints:
- health check;
- model metadata;
- image prediction.

They ensure that the deployed API can load the model and return structured detections.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "app" / "sample_images" / "originals"

client = TestClient(app)


def test_root_endpoint_returns_landing_page():
    """Test root endpoint returns API landing page."""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    assert "Image-Based Quality Control for Electronic Boards" in response.text
    assert "Business objective" in response.text
    assert "API output" in response.text
    assert "Project links" in response.text

    assert "Agnès REGAUD" in response.text
    assert "https://www.linkedin.com/in/agnes-regaud/" in response.text
    assert "GitHub repository" in response.text
    assert "Streamlit dashboard" in response.text

    assert "/docs" in response.text
    assert "/health" in response.text


def get_sample_image_path() -> Path:
    """Return one sample image available for API testing."""
    sample_images = sorted(SAMPLE_IMAGES_DIR.glob("*.jpg"))

    if not sample_images:
        raise FileNotFoundError(f"No sample image found in {SAMPLE_IMAGES_DIR}")

    return sample_images[0]


def test_health_endpoint_returns_ok():
    """Test API health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info_endpoint_returns_model_metadata():
    """Test model metadata endpoint."""
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "YOLOv8s tuned"
    assert data["task"] == "PCB defect object detection"
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) == 6
    assert data["image_size"] == 640
    assert data["confidence_threshold"] == 0.30
    assert data["iou_threshold"] == 0.50


def test_predict_endpoint_returns_structured_detections():
    """Test prediction endpoint on one sample image."""
    image_path = get_sample_image_path()

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == image_path.name
    assert isinstance(data["detections"], list)
    assert "annotated_image_base64" in data
    assert isinstance(data["annotated_image_base64"], str)
    assert len(data["annotated_image_base64"]) > 0

    for detection in data["detections"]:
        assert set(detection.keys()) == {
            "class_id",
            "class_name",
            "confidence",
            "bbox",
        }

        assert isinstance(detection["class_id"], int)
        assert isinstance(detection["class_name"], str)
        assert isinstance(detection["confidence"], float)
        assert 0 <= detection["confidence"] <= 1

        assert isinstance(detection["bbox"], list)
        assert len(detection["bbox"]) == 4

        for coordinate in detection["bbox"]:
            assert isinstance(coordinate, float)