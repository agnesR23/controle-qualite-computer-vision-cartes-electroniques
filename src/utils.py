"""Shared utility functions."""

def resolve_device(device: str) -> str:
    """Convert 'auto' into a device supported by Ultralytics."""
    if device != "auto":
        return device

    import torch

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "0"

    return "cpu"