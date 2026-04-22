"""Configuration objects and constants for the OCR parsing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_DOCUMENT_TYPES = (
    "CNH_Frente",
    "CNH_Verso",
    "RG_Frente",
    "RG_Verso",
    "CPF_Frente",
    "CPF_Verso",
)

DEFAULT_OCR_LANGUAGE = "por+eng"
DEFAULT_TESSERACT_CONFIG = "--oem 1 --psm 6"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one OCR parsing execution.

    Attributes:
        input_image: Path to the document image received at inference time.
        document_type: Previously known document type.
        output_dir: Directory where images, OCR artifacts, and JSON are saved.
        ocr_language: Tesseract language pack expression.
        tesseract_config: Extra Tesseract flags used for OCR.
        min_field_confidence: Minimum confidence accepted before warning a field.
    """

    input_image: Path
    document_type: str
    output_dir: Path
    ocr_language: str = DEFAULT_OCR_LANGUAGE
    tesseract_config: str = DEFAULT_TESSERACT_CONFIG
    min_field_confidence: float = 55.0

    def validate(self) -> None:
        """Validate user supplied execution settings.

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If the document type or confidence threshold is invalid.
        """

        if not self.input_image.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {self.input_image}")
        if self.document_type not in VALID_DOCUMENT_TYPES:
            allowed = ", ".join(VALID_DOCUMENT_TYPES)
            raise ValueError(f"Tipo de documento inválido: {self.document_type}. Use: {allowed}.")
        if not 0 <= self.min_field_confidence <= 100:
            raise ValueError("min_field_confidence deve estar entre 0 e 100.")


def document_type_to_slug(document_type: str) -> str:
    """Convert a known document type to a filesystem-friendly slug.

    Args:
        document_type: Document type, such as ``CNH_Frente``.

    Returns:
        Lowercase slug safe to use in output filenames.
    """

    return document_type.lower().replace(" ", "_")
