# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

import filecmp
import os
import shutil
from hashlib import sha256
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


def prepare_pcb_dataset(
    raw_root: Path,
    processed_root: Path,
    random_seed: int = 42,
) -> list[dict[str, int]]:
    """
    Prepare independent train, validation and test splits.

    Exact duplicate originals are grouped by content. Augmented versions
    are included only in the training split.
    """
    import yaml
    from sklearn.model_selection import train_test_split

    augmentation_prefixes = (
        "",
        "rotation_90_",
        "rotation_270_",
        "l_",
    )

    def get_sample_id(path: Path) -> str:
        for suffix in ("_600", "_256"):
            if path.stem.endswith(suffix):
                return path.stem.removesuffix(suffix)
        return path.stem

    def compute_file_hash(path: Path) -> str:
        digest = sha256()

        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    records_by_name: dict[str, dict[str, Any]] = {}

    for split in ("train", "val", "test"):
        images_dir = raw_root / split / "images"
        labels_dir = raw_root / split / "labels"

        label_paths = sorted(labels_dir.glob("*.txt"))
        labels_by_sample = {
            get_sample_id(path): path
            for path in label_paths
        }

        if len(labels_by_sample) != len(label_paths):
            raise ValueError(
                f"Duplicate label association detected in {split}."
            )

        image_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for image_path in image_paths:
            sample_id = get_sample_id(image_path)
            label_path = labels_by_sample.get(sample_id)

            if label_path is None:
                raise FileNotFoundError(
                    f"No label found for image: {image_path.name}"
                )

            if image_path.name in records_by_name:
                raise ValueError(
                    f"Duplicate image name detected: {image_path.name}"
                )

            records_by_name[image_path.name] = {
                "image_path": image_path,
                "label_path": label_path,
            }

    original_records = [
        record
        for name, record in records_by_name.items()
        if not name.startswith(augmentation_prefixes[1:])
    ]

    records_by_hash: dict[str, list[dict[str, Any]]] = {}

    for record in original_records:
        image_hash = compute_file_hash(record["image_path"])
        records_by_hash.setdefault(image_hash, []).append(record)

    unique_samples = []

    for duplicate_group in records_by_hash.values():
        class_ids = set()

        for record in duplicate_group:
            label_lines = [
                line
                for line in record["label_path"].read_text().splitlines()
                if line.strip()
            ]
            class_ids.update(
                int(line.split()[0])
                for line in label_lines
            )

        if len(class_ids) != 1:
            raise ValueError(
                "An original image contains inconsistent class annotations."
            )

        representative = min(
            duplicate_group,
            key=lambda record: record["image_path"].name,
        )

        unique_samples.append(
            {
                "source_name": representative["image_path"].name,
                "class_id": next(iter(class_ids)),
            }
        )

    unique_samples = sorted(
        unique_samples,
        key=lambda sample: sample["source_name"],
    )

    train_samples, temporary_samples = train_test_split(
        unique_samples,
        test_size=0.20,
        random_state=random_seed,
        stratify=[
            sample["class_id"]
            for sample in unique_samples
        ],
    )

    val_samples, test_samples = train_test_split(
        temporary_samples,
        test_size=0.50,
        random_state=random_seed,
        stratify=[
            sample["class_id"]
            for sample in temporary_samples
        ],
    )

    samples_by_split = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    preparation_summary = []

    for split, samples in samples_by_split.items():
        processed_images_dir = processed_root / split / "images"
        processed_labels_dir = processed_root / split / "labels"

        processed_images_dir.mkdir(parents=True, exist_ok=True)
        processed_labels_dir.mkdir(parents=True, exist_ok=True)

        prefixes = (
            augmentation_prefixes
            if split == "train"
            else ("",)
        )

        expected_image_names = set()

        for sample in samples:
            for prefix in prefixes:
                image_name = f"{prefix}{sample['source_name']}"
                record = records_by_name.get(image_name)

                if record is None:
                    raise FileNotFoundError(
                        f"Missing dataset version: {image_name}"
                    )

                image_path = record["image_path"]
                label_path = record["label_path"]
                prepared_image_path = (
                    processed_images_dir / image_name
                )
                prepared_label_path = (
                    processed_labels_dir
                    / f"{Path(image_name).stem}.txt"
                )

                expected_image_names.add(image_name)

                if prepared_image_path.exists():
                    if not os.path.samefile(
                        image_path,
                        prepared_image_path,
                    ):
                        raise FileExistsError(
                            "Unexpected prepared image: "
                            f"{prepared_image_path}"
                        )
                else:
                    os.link(image_path, prepared_image_path)

                if (
                    not prepared_label_path.exists()
                    or not filecmp.cmp(
                        label_path,
                        prepared_label_path,
                        shallow=False,
                    )
                ):
                    shutil.copy2(
                        label_path,
                        prepared_label_path,
                    )

        prepared_image_names = {
            path.name
            for path in processed_images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        prepared_label_names = {
            path.name
            for path in processed_labels_dir.glob("*.txt")
        }
        expected_label_names = {
            f"{Path(name).stem}.txt"
            for name in expected_image_names
        }

        if prepared_image_names != expected_image_names:
            missing_images = sorted(
                expected_image_names - prepared_image_names
            )
            extra_images = sorted(
                prepared_image_names - expected_image_names
            )

            raise ValueError(
                f"Unexpected prepared images in {split}. "
                f"Missing: {missing_images[:3]} "
                f"({len(missing_images)} total). "
                f"Extra: {extra_images[:3]} "
                f"({len(extra_images)} total)."
            )

        if prepared_label_names != expected_label_names:
            raise ValueError(
                f"Unexpected prepared labels in {split}."
            )

        preparation_summary.append(
            {
                "split": split,
                "source_images": len(samples),
                "images": len(prepared_image_names),
                "labels": len(prepared_label_names),
            }
        )

    raw_config = load_yaml_file(raw_root / "data.yaml")
    prepared_config = {
        "path": str(processed_root.resolve()),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": raw_config["names"],
    }

    config_content = yaml.safe_dump(
        prepared_config,
        sort_keys=False,
        allow_unicode=True,
    )
    config_path = processed_root / "data.yaml"

    if (
        not config_path.exists()
        or config_path.read_text(encoding="utf-8") != config_content
    ):
        config_path.write_text(
            config_content,
            encoding="utf-8",
        )

    return preparation_summary


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