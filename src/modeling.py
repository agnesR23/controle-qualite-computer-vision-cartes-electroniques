# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path
from typing import Any
import json

import mlflow


def train_yolov8(
    model_name: str,
    data_yaml_path: Path,
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    outputs_dir: Path,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train a YOLOv8 model and return raw benchmark metrics."""
    import time
    from ultralytics import YOLO

    outputs_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(
        f"{model_name}.pt" if model_config["pretrained"] else f"{model_name}.yaml"
    )

    start_time = time.perf_counter()

    train_results = model.train(
        data=str(data_yaml_path),
        imgsz=training_config["image_size"],
        epochs=training_config["epochs"],
        batch=model_config["batch_size"],
        device=training_config["device"],
        workers=training_config["num_workers"],
        project=str(outputs_dir.resolve()),
        name=run_name or model_name,
        exist_ok=True,
        optimizer=model_config["optimizer"],
        lr0=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
        fliplr=training_config["augmentation"]["horizontal_flip"],
        flipud=training_config["augmentation"]["vertical_flip"],
        degrees=training_config["augmentation"]["rotation_degrees"],
        cache="disk",
        plots=False,
        verbose=False,
    )

    training_time = time.perf_counter() - start_time

    val_results = model.val(
        data=str(data_yaml_path),
        imgsz=training_config["image_size"],
        batch=model_config["batch_size"],
        device=training_config["device"],
        split="val",
        verbose=False,
    )

    model_path = Path(train_results.save_dir) / "weights" / "best.pt"

    return {
        "map50_95": float(val_results.box.map),
        "map50": float(val_results.box.map50),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "training_time": float(training_time),
        "inference_time": float(val_results.speed["inference"] / 1000),
        "model_path": model_path,
        "model_extension": ".pt",
    }


def evaluate_torchvision_model(
    model: Any,
    dataloader: Any,
    device: Any,
) -> dict[str, float]:
    """Evaluate a torchvision detection model on a dataloader."""
    import time
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    device = torch.device(device)
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        backend="faster_coco_eval",
    )

    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]

            if device.type == "mps":
                torch.mps.synchronize()

            start_time = time.perf_counter()
            predictions = model(images)

            if device.type == "mps":
                torch.mps.synchronize()

            total_inference_time += time.perf_counter() - start_time
            total_images += len(images)

            predictions_cpu = [
                {
                    "boxes": pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu(),
                }
                for pred in predictions
            ]
            targets_cpu = [
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
                for target in targets
            ]

            metric.update(predictions_cpu, targets_cpu)

    results = metric.compute()

    return {
        "map50_95": float(results["map"]),
        "map50": float(results["map_50"]),
        "inference_time": total_inference_time / max(total_images, 1),
    }


def train_torchvision_model(
    model_name: str,
    data_yaml_path: Path,
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, Any]:
    """Train a torchvision detection model and return raw results."""
    import time
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset
    from torchvision.models.detection import fasterrcnn_resnet50_fpn   
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from src.data import PCBDetectionDataset, collate_fn

    device = torch.device(training_config["device"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    dataset_root = Path(data_config["path"]).resolve()
    train_dataset = PCBDetectionDataset(
        images_dir=dataset_root / data_config["train"] / "images",
        labels_dir=dataset_root / data_config["train"] / "labels",
    )
    val_dataset = PCBDetectionDataset(
        images_dir=dataset_root / data_config["val"] / "images",
        labels_dir=dataset_root / data_config["val"] / "labels",
    )

    train_subset_size = min(model_config["train_subset_size"], len(train_dataset))
    val_subset_size = min(model_config["val_subset_size"], len(val_dataset))
    
    train_dataset = Subset(train_dataset, range(train_subset_size))
    val_dataset = Subset(val_dataset, range(val_subset_size))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        collate_fn=collate_fn,
    )

    num_classes = len(data_config["names"]) + 1

    if model_name != "faster_rcnn_resnet50_fpn":
        raise ValueError(f"Unsupported torchvision model: {model_name}")
    
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if model_config["pretrained"] else None,
        weights_backbone="DEFAULT" if model_config["pretrained"] else None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    
    model.to(device)

    optimizer_name = model_config["optimizer"].lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model_config["learning_rate"],
            momentum=0.9,
            weight_decay=model_config["weight_decay"],
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {model_config['optimizer']}")

    start_time = time.perf_counter()

    for _ in range(training_config["epochs"]):
        model.train()

        for images, targets in train_dataloader:
            images = [image.to(device) for image in images]
            targets = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    training_time = time.perf_counter() - start_time

    eval_metrics = evaluate_torchvision_model(
        model=model,
        dataloader=val_dataloader,
        device=device,
    )

    model_dir = outputs_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.pth"
    torch.save(model.state_dict(), model_path)

    return {
        "map50_95": float(eval_metrics["map50_95"]),
        "map50": float(eval_metrics["map50"]),
        "precision": None,
        "recall": None,
        "training_time": float(training_time),
        "inference_time": float(eval_metrics["inference_time"]),
        "model_path": model_path,
        "model_extension": ".pth",
    }


def train_model(
    model_name: str,
    data_yaml_path: Path,
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    outputs_dir: Path,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Dispatch training to the appropriate backend according to the model name."""
    if model_name == "yolov8s":
        return train_yolov8(
            model_name=model_name,
            data_yaml_path=data_yaml_path,
            training_config=training_config,
            model_config=model_config,
            outputs_dir=outputs_dir,
            run_name=run_name,
        )

    if model_name == "faster_rcnn_resnet50_fpn":
        return train_torchvision_model(
            model_name=model_name,
            data_yaml_path=data_yaml_path,
            training_config=training_config,
            model_config=model_config,
            outputs_dir=outputs_dir,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def format_benchmark_result(
    model_name: str,
    raw_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Format raw metrics into a common structure for model comparison."""
    return {
        "model_name": model_name,
        "map50_95": float(raw_metrics["map50_95"]),
        "map50": float(raw_metrics["map50"]),
        "precision": (
            float(raw_metrics["precision"])
            if raw_metrics.get("precision") is not None
            else None
        ),
        "recall": (
            float(raw_metrics["recall"])
            if raw_metrics.get("recall") is not None
            else None
        ),
        "training_time": (
            float(raw_metrics["training_time"])
            if raw_metrics.get("training_time") is not None
            else None
        ),
        "inference_time": float(raw_metrics["inference_time"]),
    }


def is_better_result(
    candidate_result: dict[str, Any],
    best_result: dict[str, Any] | None,
) -> bool:
    """Return True if the candidate result is better than the current best result."""
    if best_result is None:
        return True

    if candidate_result["map50_95"] > best_result["map50_95"]:
        return True
    if candidate_result["map50_95"] < best_result["map50_95"]:
        return False

    if candidate_result["inference_time"] < best_result["inference_time"]:
        return True
    if candidate_result["inference_time"] > best_result["inference_time"]:
        return False

    return candidate_result["map50"] > best_result["map50"]


def log_run_to_mlflow(
    model_name: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    metrics: dict[str, Any],
    artifact_dir: Path,
) -> None:
    """Log one benchmark run to MLflow."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_metrics_path: Path = artifact_dir / "run_metrics.json"
    with open(run_metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("image_size", image_size)

        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_artifact(str(run_metrics_path))