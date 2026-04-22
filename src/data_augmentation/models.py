"""Pydantic data models used by the augmentation workflow and logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TextBox(BaseModel):
    """OCR text bounding box parsed from annotation files.

    Attributes:
        x: Left coordinate of the box.
        y: Top coordinate of the box.
        width: Box width in pixels.
        height: Box height in pixels.
        transcription: Text content inside the box.
    """

    x: float
    y: float
    width: float
    height: float
    transcription: str

    def corners(self) -> list[tuple[float, float]]:
        """Return the four box corners in clockwise order.

        Returns:
            A list of ``(x, y)`` corner coordinates.
        """
        x2 = self.x + self.width
        y2 = self.y + self.height
        return [(self.x, self.y), (x2, self.y), (x2, y2), (self.x, y2)]


class DocumentTriplet(BaseModel):
    """Paths that represent one source document sample.

    Attributes:
        class_name: Dataset class folder name.
        document_id: Stable document identifier derived from file names.
        image_path: Path to the original document image.
        mask_path: Path to the segmentation/text mask image.
        annotation_path: Path to the OCR annotation text file.
    """

    class_name: str
    document_id: str
    image_path: Path
    mask_path: Path
    annotation_path: Path


class TransformRecord(BaseModel):
    """Single applied transform captured from Albumentations replay data.

    Attributes:
        name: Transform class name.
        params: Serializable transform configuration and sampled parameters.
        applied_to: Targets affected by this transform.
    """

    name: str
    params: dict[str, Any]
    applied_to: tuple[str, ...] = Field(default=("image",))


class AugmentationResult(BaseModel):
    """Manifest row describing one original or augmented output triplet.

    Attributes:
        class_name: Dataset class name used as the training label.
        document_id: Source document identifier.
        variant_index: Zero for original copies, positive for augmented variants.
        source_image: Path to the input image.
        source_mask: Path to the input mask.
        source_annotation: Path to the input annotation file.
        output_image: Path to the generated image.
        output_mask: Path to the generated mask.
        output_annotation: Path to the generated annotation file.
        is_augmented: Whether this row was transformed or copied as original.
        scenario: High-level augmentation scenario name.
        seed: Seed used for the row.
        original_width: Input image width.
        original_height: Input image height.
        output_width: Output image width.
        output_height: Output image height.
        source_boxes: Number of parsed source OCR boxes.
        output_boxes: Number of boxes remaining after augmentation.
        transforms: Applied transform records.
    """

    class_name: str
    document_id: str
    variant_index: int
    source_image: str
    source_mask: str
    source_annotation: str
    output_image: str
    output_mask: str
    output_annotation: str
    is_augmented: bool
    scenario: str
    seed: int
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    source_boxes: int
    output_boxes: int
    transforms: list[TransformRecord]
