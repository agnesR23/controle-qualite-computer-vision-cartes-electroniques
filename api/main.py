# Copyright (C) 2026 Agnès REGAUD
# SPDX-License-Identifier: AGPL-3.0-only

import base64
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse
from src.inference import (
    get_model_path,
    load_yolo_model,
    predict_image_with_annotation,
)

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Paths and configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yml"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best" / "best_yolo_tuned.pt"


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load project configuration from a YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config(CONFIG_PATH)

model = load_yolo_model(get_model_path(MODEL_PATH))

INFERENCE_CONFIG = config["inference"]
CONFIDENCE_THRESHOLD = INFERENCE_CONFIG["thresholds"]["confidence"]
IOU_THRESHOLD = INFERENCE_CONFIG["thresholds"]["iou"]
IMAGE_SIZE = INFERENCE_CONFIG["image_size"]

# CPU is safer for deployment environments.
DEVICE = "cpu"

CLASS_NAMES = list(model.names.values())


# -----------------------------
# API
# -----------------------------

app = FastAPI(
    title="Image-Based Quality Control for Electronic Boards",
    description="FastAPI service for PCB defect detection with YOLOv8s.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # To restrict later with the Streamlit app URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Return a simple landing page for the API Space."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Image-Based Quality Control for Electronic Boards</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 820px;
                    margin: 56px auto;
                    padding: 0 24px;
                    line-height: 1.6;
                    color: #1f2937;
                }

                h1 {
                    color: #111827;
                    margin-bottom: 8px;
                }

                .subtitle {
                    color: #4b5563;
                    margin-bottom: 28px;
                }

                .card {
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 20px;
                    margin-top: 20px;
                    background: #f9fafb;
                }

                .links {
                    margin-top: 24px;
                }

                a {
                    display: inline-block;
                    margin-right: 14px;
                    margin-top: 10px;
                    color: #2563eb;
                    text-decoration: none;
                    font-weight: bold;
                }

                code {
                    background: #eef2ff;
                    padding: 2px 6px;
                    border-radius: 5px;
                }
            </style>
        </head>

        <body>
            <h1>Image-Based Quality Control for Electronic Boards</h1>

            <p class="subtitle">
                FastAPI service for live defect detection on electronic board images.
            </p>

            <div class="card">
                <h2>Business objective</h2>
                <p>
                    From manual visual inspection to automated defect detection
                    on electronic boards, with the objective of reducing customer returns
                    and improving customer retention.
                </p>
            </div>

            <div class="card">
                <h2>API output</h2>
                <p>
                    The <code>/predict</code> endpoint runs live inference with a tuned
                    YOLOv8s model and returns:
                </p>
                <ul>
                    <li>detected defect class;</li>
                    <li>confidence score;</li>
                    <li>bounding box coordinates;</li>
                    <li>annotated image encoded in base64.</li>
                </ul>
            </div>

            <div class="card">
                <h2>Project links</h2>
                <p>
                    Project by <strong>Agnès REGAUD</strong>.
                </p>
                <div class="links">
                    <a href="https://www.linkedin.com/in/agnes-regaud/" target="_blank">
                        LinkedIn profile
                    </a>
                    <a href="#" target="_blank">
                        GitHub repository — coming soon
                    </a>
                    <a href="#" target="_blank">
                        Streamlit dashboard — coming soon
                    </a>
                </div>
            </div>

            <div class="links">
                <a href="/docs">Open API documentation</a>
                <a href="/health">Check API health</a>
            </div>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Check API health and model loading status.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """
    Return deployed model information and inference parameters.
    """
    return ModelInfoResponse(
        model_name="YOLOv8s tuned",
        task="PCB defect object detection",
        classes=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Run YOLO inference on an uploaded image.
    """
    suffix = Path(file.filename or "uploaded_image.jpg").suffix or ".jpg"

    with NamedTemporaryFile(delete=True, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()

        detections, annotated_image = predict_image_with_annotation(
            model=model,
            image_path=Path(temp_file.name),
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            image_size=IMAGE_SIZE,
            device=DEVICE,
        )

    image_buffer = BytesIO()
    Image.fromarray(annotated_image).save(image_buffer, format="JPEG")
    annotated_image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        detections=detections,
        annotated_image_base64=annotated_image_base64,
    )