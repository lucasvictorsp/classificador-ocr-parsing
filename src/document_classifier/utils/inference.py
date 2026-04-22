"""Reusable inference helpers shared by simple and batch pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from document_classifier.data import build_transforms
from document_classifier.models import build_model


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained checkpoint and rebuild its architecture.

    Args:
        checkpoint_path: Path to a checkpoint created by training.
        device: Device used for inference.

    Returns:
        Tuple with loaded model and checkpoint metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    model_info = build_model(
        num_classes=len(metadata["class_names"]),
        model_name=metadata["model_name"],
        pretrained=False,
        freeze_backbone=False,
    )
    model = model_info.model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: list[str],
    input_size: int,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    """Predict one image and apply unknown rejection.

    Args:
        model: Loaded classifier.
        image_path: Image to classify.
        class_names: Known class names.
        input_size: Model input size.
        threshold: Confidence threshold for ``outros``.
        device: Device used for inference.

    Returns:
        Prediction dictionary with probabilities and final label.
    """
    transform = build_transforms(input_size)["eval"]
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    confidence, predicted_index_tensor = torch.max(probabilities, dim=0)
    predicted_index = int(predicted_index_tensor.item())
    confidence_value = float(confidence.item())
    predicted_label = class_names[predicted_index]
    final_label = predicted_label if confidence_value >= threshold else "outros"
    return {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "final_label": final_label,
        "confidence": confidence_value,
        "threshold": float(threshold),
        "probabilities": {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(class_names)
        },
    }
