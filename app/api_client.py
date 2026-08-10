from pathlib import Path
from typing import Any

import requests


def check_api_health(api_url: str, timeout: int = 3) -> bool:
    """Return True if the inference API is available."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def predict_image_with_api(
    api_url: str,
    image_path: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Send one image to the FastAPI inference service.
    """
    endpoint = f"{api_url.rstrip('/')}/predict"

    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                endpoint,
                files={
                    "file": (
                        image_path.name,
                        image_file,
                        "image/jpeg",
                    )
                },
                timeout=timeout,
            )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        raise RuntimeError(f"Erreur API : {error}") from error