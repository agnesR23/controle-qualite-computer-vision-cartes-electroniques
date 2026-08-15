# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """
    Single object detection returned by the model.
    """

    class_id: int = Field(..., description="Detected class identifier")
    class_name: str = Field(..., description="Detected defect class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="Bounding box [x_min, y_min, x_max, y_max]")


class PredictionResponse(BaseModel):
    """
    API response returned after live inference.

    annotated_image_base64 contains the annotated prediction image generated
    by the API at request time.
    """

    filename: str
    detections: list[Detection]
    annotated_image_base64: str


class HealthResponse(BaseModel):
    """
    API health check response.
    """

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """
    Basic information about the deployed model.
    """

    model_name: str
    task: str
    classes: list[str]
    image_size: int
    confidence_threshold: float
    iou_threshold: float