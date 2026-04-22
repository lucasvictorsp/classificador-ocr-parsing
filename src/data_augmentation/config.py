"""Configuration model and defaults for the augmentation pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt, field_validator


DEFAULT_CLASSES = (
    "CNH_Frente",
    "CNH_Verso",
    "RG_Frente",
    "RG_Verso",
    "CPF_Frente",
    "CPF_Verso",
)


class AugmentationConfig(BaseModel):
    """Configuration for a full document augmentation run.

    Attributes:
        dataset_dir: Root directory containing one subdirectory per document class.
        output_dir: Directory where augmented files and logs are written.
        class_names: Class subdirectories that should be processed.
        seed: Base random seed used to derive deterministic per-document seeds.
        total_factor: Number of synthetic variants generated for each source document.
        copy_originals: Whether original triplets are copied to the output dataset.
        overwrite: Whether existing output files can be replaced.
        clean_output: Whether the output directory is removed before generation.
        jpeg_quality: JPEG quality used when writing images and masks.
        max_documents_per_class: Optional cap for fast tests or sampled runs.
        image_extensions: Accepted input image extensions.
        log_dir_name: Name of the output subdirectory that stores run logs.
        min_bbox_size: Minimum width and height for surviving bounding boxes.
    """

    dataset_dir: Path = Field(default=Path("dataset"))
    output_dir: Path = Field(default=Path("dataset_augmented"))
    class_names: tuple[str, ...] = Field(default=DEFAULT_CLASSES)
    seed: int = Field(default=42)
    total_factor: PositiveInt = Field(
        default=1,
        description="Number of augmented variants generated for each source document.",
    )
    copy_originals: bool = Field(default=True)
    overwrite: bool = Field(default=False)
    clean_output: bool = Field(default=False)
    jpeg_quality: int = Field(default=92, ge=40, le=100)
    max_documents_per_class: int | None = Field(default=None, ge=1)
    image_extensions: tuple[str, ...] = Field(default=(".jpg", ".jpeg", ".png"))
    log_dir_name: str = Field(default="logs")
    min_bbox_size: int = Field(default=2, ge=1)

    @field_validator("dataset_dir", "output_dir", mode="before")
    @classmethod
    def normalize_path(cls, value: str | Path) -> Path:
        """Convert path-like values into ``Path`` instances.

        Args:
            value: Raw path value provided by CLI, code, or defaults.

        Returns:
            Normalized ``Path`` instance.
        """
        return Path(value)

    @field_validator("image_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Normalize image extensions to lowercase dotted suffixes.

        Args:
            value: Sequence of extensions with or without a leading dot.

        Returns:
            Tuple of lowercase extensions with a leading dot.
        """
        return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value)

    @property
    def variants_per_document(self) -> int:
        """Compute how many augmented variants each source document should produce.

        Returns:
            Number of augmented samples to generate per source triplet.
        """
        return self.total_factor

    @property
    def log_dir(self) -> Path:
        """Build the directory path used for run logs.

        Returns:
            Output log directory path.
        """
        return self.output_dir / self.log_dir_name
