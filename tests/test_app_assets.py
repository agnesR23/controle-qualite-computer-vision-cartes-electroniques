# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for Streamlit dashboard assets.

These tests validate the static files required by the Streamlit app:
- metrics CSV files;
- confusion matrix image;
- original sample images;
- fallback prediction images.

They ensure that the dashboard and fallback demo can run without missing assets.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = APP_DIR / "assets"
ORIGINALS_DIR = APP_DIR / "sample_images" / "originals"
PREDICTIONS_DIR = APP_DIR / "sample_images" / "predictions"

REQUIRED_ASSETS = [
    "benchmark_results.csv",
    "yolo_tuning_results.csv",
    "per_class_metrics.csv",
    "results.csv",
    "confusion_matrix_normalized.png",
]

EXPECTED_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]


def test_streamlit_assets_exist():
    """Check that all static dashboard assets exist."""
    for filename in REQUIRED_ASSETS:
        asset_path = ASSETS_DIR / filename
        assert asset_path.exists(), f"Missing Streamlit asset: {asset_path}"


def test_sample_image_directories_exist():
    """Check that demo image directories exist."""
    assert ORIGINALS_DIR.exists()
    assert PREDICTIONS_DIR.exists()


def test_each_original_image_has_fallback_prediction():
    """Check that every original sample image has a matching fallback prediction."""
    original_images = sorted(ORIGINALS_DIR.glob("*.jpg"))

    assert original_images, "No original sample images found."

    for original_path in original_images:
        prediction_path = PREDICTIONS_DIR / f"{original_path.stem}_pred.jpg"
        assert prediction_path.exists(), (
            f"Missing fallback prediction for {original_path.name}"
        )


def test_each_expected_class_has_at_least_one_demo_image():
    """Check that each PCB defect class has at least one demo image."""
    original_names = [path.stem for path in ORIGINALS_DIR.glob("*.jpg")]

    for class_name in EXPECTED_CLASSES:
        has_class_example = any(
            image_name.startswith(f"{class_name}_") for image_name in original_names
        )

        assert has_class_example, f"No demo image found for class: {class_name}"


def test_prediction_images_are_not_empty():
    """Check that fallback prediction images are not empty files."""
    prediction_images = sorted(PREDICTIONS_DIR.glob("*.jpg"))

    assert prediction_images, "No fallback prediction images found."

    for prediction_path in prediction_images:
        assert prediction_path.stat().st_size > 0, (
            f"Empty prediction image: {prediction_path.name}"
        )