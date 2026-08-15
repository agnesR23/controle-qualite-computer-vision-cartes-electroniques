# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Load a YAML file and return its content."""
    import yaml

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class PCBDetectionDataset(Dataset):
    """PyTorch dataset for object detection from YOLO-format labels."""

    def __init__(self, images_dir: Path, labels_dir: Path) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        # Keep only supported image files and sort them for reproducibility.
        self.image_paths = sorted(
            [
                path
                for path in self.images_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        # Load image as RGB and convert it to a float tensor in [0, 1].
        image = Image.open(image_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        width, height = image.size
        boxes: list[list[float]] = []
        labels: list[int] = []

        # Read YOLO labels if the annotation file exists.
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    # YOLO format: class_id x_center y_center width height
                    class_id, x_center, y_center, box_width, box_height = map(float, line.split())

                    # Convert normalized YOLO boxes to absolute XYXY coordinates.
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    x_min = x_center - box_width / 2
                    y_min = y_center - box_height / 2
                    x_max = x_center + box_width / 2
                    y_max = y_center + box_height / 2

                    boxes.append([x_min, y_min, x_max, y_max])

                    # Torchvision detection models usually expect class ids starting at 1.
                    labels.append(int(class_id) + 1)

        # Handle images without annotations.
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
        }

        return image_tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate function for object detection batches."""
    # Detection batches contain variable numbers of boxes per image,
    # so we keep images and targets as lists.
    images, targets = zip(*batch)
    return list(images), list(targets)