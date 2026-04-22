"""Albumentations transform builders and replay metadata serialization."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np

from data_augmentation.models import TextBox, TransformRecord


GEOMETRIC_TRANSFORMS = {
    "Affine",
    "Perspective",
    "Rotate",
    "ShiftScaleRotate",
    "HorizontalFlip",
    "VerticalFlip",
    "Transpose",
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
}
CONTAINER_TRANSFORMS = {"OneOf", "SomeOf", "RandomOrder", "Sequential", "Compose", "ReplayCompose"}


@dataclass(frozen=True)
class AugmentedSample:
    """Augmented image triplet and replay records.

    Attributes:
        image: Augmented OpenCV BGR image.
        mask: Augmented grayscale mask.
        boxes: Augmented OCR bounding boxes.
        records: Serializable transform records extracted from replay data.
    """

    image: np.ndarray
    mask: np.ndarray
    boxes: list[TextBox]
    records: list[TransformRecord]


def colored_background(rng: random.Random) -> tuple[int, int, int]:
    """Sample a BGR background color for OpenCV/Albumentations.

    Args:
        rng: Random generator used for deterministic color sampling.

    Returns:
        Three-channel BGR fill color.
    """

    palettes = (
        ((120, 185), (93, 140), (32, 74)),
        ((86, 160), (132, 210), (120, 190)),
        ((148, 220), (76, 150), (108, 185)),
        ((180, 245), (175, 235), (170, 235)),
        ((95, 165), (95, 170), (80, 150)),
    )
    channel_ranges = rng.choice(palettes)
    return tuple(rng.randint(low, high) for low, high in channel_ranges)


def apply_replay_compose(
    image: np.ndarray,
    mask: np.ndarray,
    boxes: list[TextBox],
    transform: A.ReplayCompose,
    seed: int,
) -> AugmentedSample:
    """Apply an Albumentations replay pipeline to all synchronized targets.

    Args:
        image: Source OpenCV BGR image.
        mask: Source grayscale mask.
        boxes: Source OCR boxes.
        transform: Replayable Albumentations pipeline.
        seed: Seed applied to OpenCV and Albumentations RNGs.

    Returns:
        Augmented sample containing image, mask, transformed boxes, and records.
    """

    cv2.setRNGSeed(seed & 0x7FFFFFFF)
    transform.set_random_seed(seed)

    bboxes = [_to_pascal_voc(box) for box in boxes]
    labels = [box.transcription for box in boxes]
    result = transform(image=image, mask=mask, bboxes=bboxes, bbox_labels=labels)

    transformed_boxes = [
        _from_pascal_voc(bbox, label)
        for bbox, label in zip(result["bboxes"], result["bbox_labels"], strict=True)
    ]
    return AugmentedSample(
        image=result["image"],
        mask=result["mask"],
        boxes=transformed_boxes,
        records=_replay_to_records(result["replay"]),
    )


def make_replay_compose(transforms: list[A.BasicTransform], min_bbox_size: int) -> A.ReplayCompose:
    """Create a replayable Albumentations composition with bbox support.

    Args:
        transforms: Ordered Albumentations transforms.
        min_bbox_size: Minimum bbox width and height retained after transforms.

    Returns:
        Configured ``ReplayCompose`` instance.
    """
    return A.ReplayCompose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["bbox_labels"],
            min_width=float(min_bbox_size),
            min_height=float(min_bbox_size),
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


def affine(
    *,
    fill: tuple[int, int, int],
    rotate: tuple[float, float],
    scale: tuple[float, float],
    translate: tuple[float, float],
    shear: tuple[float, float] = (-1.5, 1.5),
) -> A.Affine:
    """Build a synchronized affine transform.

    Args:
        fill: BGR fill color for newly exposed image regions.
        rotate: Rotation range in degrees.
        scale: Scale range.
        translate: Translation range as image-size percentages.
        shear: Horizontal shear range in degrees.

    Returns:
        Configured Albumentations ``Affine`` transform.
    """
    return A.Affine(
        scale=scale,
        translate_percent={"x": translate, "y": translate},
        rotate=rotate,
        shear={"x": shear, "y": (-0.5, 0.5)},
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=fill,
        fill_mask=0,
        rotate_method="largest_box",
        p=1.0,
    )


def perspective(*, fill: tuple[int, int, int], scale: tuple[float, float]) -> A.Perspective:
    """Build a synchronized perspective transform.

    Args:
        fill: BGR fill color for newly exposed image regions.
        scale: Perspective distortion scale range.

    Returns:
        Configured Albumentations ``Perspective`` transform.
    """
    return A.Perspective(
        scale=scale,
        keep_size=True,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=fill,
        fill_mask=0,
        p=1.0,
    )


def replay_records_as_json(records: list[TransformRecord]) -> str:
    """Serialize transform records to JSON.

    Args:
        records: Transform records to serialize.

    Returns:
        JSON string with non-ASCII text preserved.
    """
    return json.dumps([record.model_dump() for record in records], ensure_ascii=False)


def _to_pascal_voc(box: TextBox) -> tuple[float, float, float, float]:
    """Convert a text box to Pascal VOC coordinates.

    Args:
        box: Text box in ``x, y, width, height`` format.

    Returns:
        Tuple ``(x_min, y_min, x_max, y_max)``.
    """
    return (box.x, box.y, box.x + box.width, box.y + box.height)


def _from_pascal_voc(bbox: tuple[float, float, float, float], label: str) -> TextBox:
    """Convert Pascal VOC coordinates back to a text box.

    Args:
        bbox: Tuple ``(x_min, y_min, x_max, y_max)``.
        label: OCR transcription associated with the box.

    Returns:
        Text box in canonical project format.
    """
    x_min, y_min, x_max, y_max = bbox
    return TextBox(
        x=float(x_min),
        y=float(y_min),
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        transcription=str(label),
    )


def _replay_to_records(replay: dict[str, Any]) -> list[TransformRecord]:
    """Extract applied transform records from Albumentations replay output.

    Args:
        replay: Replay dictionary returned by ``ReplayCompose``.

    Returns:
        Applied transform records with JSON-safe parameters.
    """
    records: list[TransformRecord] = []
    _collect_records(replay.get("transforms", []), records)
    return records


def _collect_records(items: list[dict[str, Any]], records: list[TransformRecord]) -> None:
    """Collect applied transforms recursively from replay items.

    Args:
        items: Replay transform entries.
        records: Mutable output list that receives applied transform records.
    """
    for item in items:
        class_name = str(item.get("__class_fullname__", "unknown")).split(".")[-1]
        nested = item.get("transforms", [])
        if nested and class_name in CONTAINER_TRANSFORMS:
            _collect_records(nested, records)
            continue

        if item.get("applied", False):
            params = {
                key: _jsonable(value)
                for key, value in item.items()
                if key not in {"__class_fullname__", "applied", "transforms"}
            }
            records.append(
                TransformRecord(
                    name=class_name,
                    params=params,
                    applied_to=_applied_targets(class_name),
                )
            )


def _applied_targets(class_name: str) -> tuple[str, ...]:
    """Infer which targets were affected by a transform class.

    Args:
        class_name: Albumentations transform class name.

    Returns:
        Tuple of affected target names for logging.
    """
    if class_name in GEOMETRIC_TRANSFORMS:
        return ("image", "mask", "annotations")
    return ("image",)


def _jsonable(value: Any) -> Any:
    """Convert nested replay values into JSON-serializable objects.

    Args:
        value: Arbitrary value from Albumentations replay metadata.

    Returns:
        JSON-safe representation of the value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
