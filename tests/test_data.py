"""
Tests for YOLO dataset integrity and PyTorch dataset loading.

These tests validate:
- YOLO dataset configuration;
- train / val / test split availability;
- YOLO label format for labels linked to existing images;
- PCBDetectionDataset output format;
- collate function for object detection batches.

They ensure that training and evaluation rely on consistent image / annotation loading.
"""

from pathlib import Path

import torch
import yaml

from src.data import PCBDetectionDataset, collate_fn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "pcb-defect-dataset"
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

EXPECTED_SPLITS = ["train", "val", "test"]


def load_data_yaml() -> dict:
    """Load YOLO dataset configuration."""
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_data_yaml_exists():
    """Check that YOLO data.yaml exists."""
    assert DATA_YAML_PATH.exists()


def test_data_yaml_has_expected_keys():
    """Check that data.yaml contains the required keys."""
    data_config = load_data_yaml()

    for key in ["train", "val", "test", "names"]:
        assert key in data_config


def test_dataset_splits_have_images_and_labels():
    """Check that each split contains images and labels."""
    for split in EXPECTED_SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        assert images_dir.exists()
        assert labels_dir.exists()

        assert len(list(images_dir.glob("*.jpg"))) > 0
        assert len(list(labels_dir.glob("*.txt"))) > 0


def test_yolo_labels_linked_to_images_are_valid():
    """Check YOLO labels for images actually present in the dataset."""
    data_config = load_data_yaml()
    class_ids = set(range(len(data_config["names"])))

    for split in EXPECTED_SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        image_paths = sorted(images_dir.glob("*.jpg"))

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            lines = label_path.read_text(encoding="utf-8").strip().splitlines()

            for line in lines:
                values = line.split()

                assert len(values) == 5, (
                    f"Invalid YOLO format in {label_path.name}: {line}"
                )

                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])

                assert class_id in class_ids
                assert 0 <= x_center <= 1
                assert 0 <= y_center <= 1
                assert 0 < width <= 1
                assert 0 < height <= 1


def test_pcb_detection_dataset_returns_valid_sample():
    """Check that PCBDetectionDataset returns an image tensor and a valid target."""
    images_dir = DATASET_DIR / "train" / "images"
    labels_dir = DATASET_DIR / "train" / "labels"

    dataset = PCBDetectionDataset(images_dir=images_dir, labels_dir=labels_dir)

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3
    assert image.shape[0] == 3
    assert image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1

    assert set(target.keys()) == {"boxes", "labels", "image_id"}
    assert isinstance(target["boxes"], torch.Tensor)
    assert isinstance(target["labels"], torch.Tensor)
    assert isinstance(target["image_id"], torch.Tensor)

    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4
    assert target["labels"].ndim == 1

    assert len(target["boxes"]) == len(target["labels"])

    if len(target["labels"]) > 0:
        assert target["labels"].min() >= 1


def test_collate_fn_keeps_detection_batch_as_lists():
    """Check that collate_fn keeps variable-size detection targets as lists."""
    images_dir = DATASET_DIR / "train" / "images"
    labels_dir = DATASET_DIR / "train" / "labels"

    dataset = PCBDetectionDataset(images_dir=images_dir, labels_dir=labels_dir)

    batch = [dataset[0], dataset[1]]
    images, targets = collate_fn(batch)

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert len(images) == 2
    assert len(targets) == 2

    assert all(isinstance(image, torch.Tensor) for image in images)
    assert all(isinstance(target, dict) for target in targets)