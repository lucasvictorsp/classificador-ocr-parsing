"""Input and output helpers for image and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import document_type_to_slug


def ensure_directory(path: Path) -> Path:
    """Create a directory when needed and return it.

    Args:
        path: Directory path to create.

    Returns:
        The same directory path.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from disk using OpenCV with Unicode-safe path handling.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a BGR NumPy array.

    Raises:
        ValueError: If OpenCV cannot decode the image.
    """

    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")
    return image


def write_image(image_path: Path, image: np.ndarray) -> Path:
    """Write an image to disk using OpenCV with Unicode-safe path handling.

    Args:
        image_path: Destination path.
        image: Image array to encode.

    Returns:
        Destination path.

    Raises:
        ValueError: If the image extension cannot be encoded.
    """

    ensure_directory(image_path.parent)
    extension = image_path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Não foi possível codificar a imagem para {image_path}")
    encoded.tofile(str(image_path))
    return image_path


def write_json(json_path: Path, payload: dict[str, Any]) -> Path:
    """Serialize a dictionary as UTF-8 JSON.

    Args:
        json_path: Destination file path.
        payload: JSON-serializable payload.

    Returns:
        Destination path.
    """

    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def write_text(text_path: Path, content: str) -> Path:
    """Write OCR text as UTF-8.

    Args:
        text_path: Destination file path.
        content: Text to persist.

    Returns:
        Destination path.
    """

    ensure_directory(text_path.parent)
    text_path.write_text(content, encoding="utf-8")
    return text_path


def build_output_paths(input_image: Path, document_type: str, output_dir: Path) -> dict[str, Path]:
    """Build deterministic artifact paths for one pipeline execution.

    Args:
        input_image: Original input image path.
        document_type: Known document type.
        output_dir: Base output directory.

    Returns:
        Mapping with paths for rectified image, OCR image, OCR text, OCR data, and final JSON.
    """

    slug = document_type_to_slug(document_type)
    base_name = f"{input_image.stem}_{slug}"
    run_dir = ensure_directory(output_dir / base_name)
    return {
        "run_dir": run_dir,
        "rectified_image": run_dir / f"{base_name}_retificada.jpg",
        "ocr_image": run_dir / f"{base_name}_pre_ocr.png",
        "ocr_text": run_dir / f"{base_name}_ocr.txt",
        "ocr_data": run_dir / f"{base_name}_ocr_detalhado.json",
        "parsed_json": run_dir / f"{base_name}_resultado.json",
    }
