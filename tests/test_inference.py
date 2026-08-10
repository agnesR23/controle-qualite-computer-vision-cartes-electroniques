"""
Tests for YOLO inference utilities.

These tests validate the reusable inference logic from src/inference.py:
- model loading;
- inference on a sample image;
- structured detection output.

They ensure that notebooks, API and tests rely on the same inference code.
"""

from pathlib import Path

import yaml

from src.inference import get_model_path, load_yolo_model, predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yml"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best" / "best_yolo_tuned.pt"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "app" / "sample_images" / "originals"


def load_config() -> dict:
    """Load project configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_sample_image_path() -> Path:
    """Return one available sample image used by the Streamlit demo."""
    sample_images = sorted(SAMPLE_IMAGES_DIR.glob("*.jpg"))

    if not sample_images:
        raise FileNotFoundError(f"No sample image found in {SAMPLE_IMAGES_DIR}")

    return sample_images[0]


def test_yolo_inference_returns_structured_detections():
    """Test YOLO inference on one sample image with configured parameters."""
    config = load_config()

    model = load_yolo_model(get_model_path(MODEL_PATH))
    image_path = get_sample_image_path()

    detections = predict_image(
        model=model,
        image_path=image_path,
        confidence_threshold=config["inference"]["thresholds"]["confidence"],
        iou_threshold=config["inference"]["thresholds"]["iou"],
        image_size=config["inference"]["image_size"],
        device="cpu",
    )

    assert isinstance(detections, list)

    for detection in detections:
        assert set(detection.keys()) == {
            "class_id",
            "class_name",
            "confidence",
            "bbox",
        }

        assert isinstance(detection["class_id"], int)
        assert isinstance(detection["class_name"], str)
        assert isinstance(detection["confidence"], float)
        assert isinstance(detection["bbox"], list)

        assert 0 <= detection["confidence"] <= 1
        assert len(detection["bbox"]) == 4

        for coordinate in detection["bbox"]:
            assert isinstance(coordinate, float)