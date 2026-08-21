# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for Streamlit dashboard assets.

These tests validate:
- metric and visualization files;
- original images used by the live demonstration;
- representation of every expected defect class.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_DIR = PROJECT_ROOT / "app"
ASSETS_DIR = APP_DIR / "assets"
ORIGINALS_DIR = APP_DIR / "sample_images" / "originals"

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


def test_each_expected_class_has_at_least_one_demo_image():
    """Check that each PCB defect class has at least one demo image."""
    original_names = [path.stem for path in ORIGINALS_DIR.glob("*.jpg")]

    for class_name in EXPECTED_CLASSES:
        has_class_example = any(
            image_name.startswith(f"{class_name}_") for image_name in original_names
        )

        assert has_class_example, f"No demo image found for class: {class_name}"