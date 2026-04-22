"""OpenCV I/O, annotation parsing, and augmentation artifact writers."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from data_augmentation.models import AugmentationResult, TextBox


ANNOTATION_HEADER = ["x", "y", "width", "height", "transcription"]


def read_image(path: Path) -> np.ndarray:
    """Read a color image with OpenCV.

    Args:
        path: Image file path.

    Returns:
        Image array in OpenCV BGR channel order.

    Raises:
        ValueError: If OpenCV cannot decode the image.
    """
    image = _imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nao foi possivel ler a imagem: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    """Read a single-channel mask with OpenCV.

    Args:
        path: Mask file path.

    Returns:
        Grayscale mask array.

    Raises:
        ValueError: If OpenCV cannot decode the mask.
    """
    mask = _imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Nao foi possivel ler a mascara: {path}")
    return mask


def save_image(image: np.ndarray, path: Path, quality: int) -> None:
    """Write a color image to disk.

    Args:
        image: Image array in OpenCV format.
        path: Destination path.
        quality: JPEG quality used when the destination is JPEG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(path, image, quality)


def save_mask(mask: np.ndarray, path: Path, quality: int) -> None:
    """Write a mask image to disk.

    Args:
        mask: Single-channel mask array.
        path: Destination path.
        quality: JPEG quality used when the destination is JPEG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(path, mask, quality)


def copy_file(source: Path, destination: Path) -> None:
    """Copy a file while creating the destination directory.

    Args:
        source: Existing source path.
        destination: Destination file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_annotations(path: Path) -> list[TextBox]:
    """Parse OCR annotation rows into text boxes.

    Supports both regular CSV rows and BID polygon rows where x/y coordinates
    are stored as lists.

    Args:
        path: Annotation text file path.

    Returns:
        Parsed text boxes.
    """
    rows: list[TextBox] = []
    for line in _read_text_with_fallback(path).splitlines():
        line = line.strip()
        if not line or line.lower().startswith("x, y, width, height"):
            continue
        rows.append(_parse_annotation_line(line))
    return rows


def _read_text_with_fallback(path: Path) -> str:
    """Read annotation text using common encodings found in BID files.

    Args:
        path: Text file path.

    Returns:
        Decoded text content.
    """
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def write_annotations(path: Path, boxes: Iterable[TextBox]) -> None:
    """Write text boxes using the canonical CSV annotation format.

    Args:
        path: Destination annotation file.
        boxes: Text boxes to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANNOTATION_HEADER)
        for box in boxes:
            writer.writerow(
                [
                    round(box.x),
                    round(box.y),
                    round(box.width),
                    round(box.height),
                    box.transcription,
                ]
            )


def write_jsonl(path: Path, records: Iterable[AugmentationResult]) -> None:
    """Write augmentation records as JSON Lines.

    Args:
        path: Destination JSONL path.
        records: Records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def write_sidecar_json(path: Path, record: AugmentationResult) -> None:
    """Write a per-triplet JSON log next to generated files.

    Args:
        path: Destination sidecar JSON path.
        record: Augmentation record for one output triplet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def write_manifest_csv(path: Path, records: Iterable[AugmentationResult]) -> None:
    """Write the MLflow-friendly manifest CSV.

    Args:
        path: Destination manifest path.
        records: Manifest records to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_name",
        "label",
        "document_id",
        "variant_index",
        "image_path",
        "mask_path",
        "annotation_path",
        "is_augmented",
        "scenario",
        "seed",
        "source_image",
        "source_mask",
        "source_annotation",
        "source_boxes",
        "output_boxes",
        "transforms_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "class_name": record.class_name,
                    "label": record.class_name,
                    "document_id": record.document_id,
                    "variant_index": record.variant_index,
                    "image_path": record.output_image,
                    "mask_path": record.output_mask,
                    "annotation_path": record.output_annotation,
                    "is_augmented": record.is_augmented,
                    "scenario": record.scenario,
                    "seed": record.seed,
                    "source_image": record.source_image,
                    "source_mask": record.source_mask,
                    "source_annotation": record.source_annotation,
                    "source_boxes": record.source_boxes,
                    "output_boxes": record.output_boxes,
                    "transforms_json": json.dumps(
                        [item.model_dump() for item in record.transforms],
                        ensure_ascii=False,
                    ),
                }
            )


def write_summary_csv(path: Path, records: Iterable[AugmentationResult]) -> None:
    """Write aggregate counts by class, scenario, and augmentation flag.

    Args:
        path: Destination summary path.
        records: Manifest records to aggregate.
    """
    counts: dict[tuple[str, str, bool], int] = {}
    for record in records:
        key = (record.class_name, record.scenario, record.is_augmented)
        counts[key] = counts.get(key, 0) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "scenario", "is_augmented", "count"])
        for (class_name, scenario, is_augmented), count in sorted(counts.items()):
            writer.writerow([class_name, scenario, is_augmented, count])


def write_params_json(path: Path, payload: dict[str, object]) -> None:
    """Write run parameters as a JSON artifact.

    Args:
        path: Destination JSON path.
        payload: Serializable parameter payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_annotation_line(line: str) -> TextBox:
    """Parse one annotation line into a text box.

    Args:
        line: Raw annotation row without the header.

    Returns:
        Parsed text box.

    Raises:
        ValueError: If the line cannot be parsed as CSV or polygon annotation.
    """
    polygon_match = re.match(
        r"^\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*(.*)$",
        line,
    )
    if polygon_match:
        x_values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", polygon_match.group(1))]
        y_values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", polygon_match.group(2))]
        if not x_values or not y_values:
            raise ValueError(f"Linha de anotacao poligonal invalida: {line}")
        x_min = min(x_values)
        y_min = min(y_values)
        return TextBox(
            x=x_min,
            y=y_min,
            width=max(x_values) - x_min,
            height=max(y_values) - y_min,
            transcription=polygon_match.group(3).strip(),
        )

    row = next(csv.reader([line]))
    if len(row) < 5:
        msg = f"Linha de anotacao invalida: {row}"
        raise ValueError(msg)
    transcription = ",".join(row[4:]).strip()
    return TextBox(
        x=float(row[0]),
        y=float(row[1]),
        width=float(row[2]),
        height=float(row[3]),
        transcription=transcription,
    )


def _imread(path: Path, flags: int) -> np.ndarray | None:
    """Read an image using OpenCV while supporting non-ASCII Windows paths.

    Args:
        path: Image path.
        flags: OpenCV imread flags.

    Returns:
        Decoded image array or ``None`` when decoding fails.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _imwrite(path: Path, image: np.ndarray, quality: int) -> None:
    """Write an image using OpenCV while supporting non-ASCII Windows paths.

    Args:
        path: Destination path.
        image: Image array to encode.
        quality: JPEG quality used for JPEG outputs.

    Raises:
        ValueError: If OpenCV cannot encode the image.
    """
    suffix = path.suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"Nao foi possivel codificar arquivo: {path}")
    encoded.tofile(str(path))
