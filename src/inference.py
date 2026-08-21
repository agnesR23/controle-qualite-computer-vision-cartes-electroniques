# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

import os
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def load_yolo_model(model_path: Path) -> YOLO:
    """
    Load a trained YOLO model.

    Parameters
    ----------
    model_path : Path
        Path to the YOLO model checkpoint.

    Returns
    -------
    YOLO
        Loaded YOLO model.
    """
    return YOLO(str(model_path))


def get_model_path(model_path: Path) -> Path:
    """
    Return local model path or download it from Hugging Face if missing.
    """
    if model_path.exists():
        return model_path

    repo_id = os.getenv("MODEL_REPO_ID", "agnesR23/pcb-defect-detection-api")
    filename = os.getenv(
        "MODEL_FILENAME",
        "outputs/models/best/best_yolo_tuned.pt",
    )

    token = os.getenv("HF_TOKEN")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="space",
        token=token,
    )

    return Path(downloaded_path)


def predict_image(
    model: YOLO,
    image_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    device: str,
) -> list[dict[str, Any]]:
    """
    Run object detection on one image and return structured detections only.

    This function is useful for tests or API responses when no annotated image
    is required.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    image_path : Path
        Path to the input image.
    confidence_threshold : float
        Minimum confidence score.
    iou_threshold : float
        IoU threshold used during prediction.
    image_size : int
        Image size used by YOLO during inference.
    device : str
        Inference device.

    Returns
    -------
    list[dict[str, Any]]
        Detected objects with class, confidence and bounding box.
    """
    result = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]

    if result.boxes is None:
        return []

    detections: list[dict[str, Any]] = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
                "bbox": [
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max),
                ],
            }
        )

    return detections



def predict_image_with_annotation(
    model: YOLO,
    image_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Run object detection once and return both detections and annotated image.

    This function is intended for live inference through the API, so that the
    Streamlit dashboard can display a prediction generated at request time.

    Unlike predict_image(), which returns structured detections only,
    this function also returns the annotated image required by the live API.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    image_path : Path
        Path to the input image.
    confidence_threshold : float
        Minimum confidence score.
    iou_threshold : float
        IoU threshold used during prediction.
    image_size : int
        Image size used by YOLO during inference.
    device : str
        Inference device.

    Returns
    -------
    tuple[list[dict[str, Any]], np.ndarray]
        Structured detections and RGB annotated image.
    """
    result = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]

    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": confidence,
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(x_max),
                        float(y_max),
                    ],
                }
            )

    annotated_image = result.plot()

    # Ultralytics returns BGR arrays. Convert to RGB for Streamlit / PIL.
    annotated_image_rgb = annotated_image[..., ::-1]

    return detections, annotated_image_rgb