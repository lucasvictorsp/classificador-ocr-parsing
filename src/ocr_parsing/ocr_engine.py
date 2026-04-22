"""Tesseract OCR wrapper with structured word and line output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OCRWord:
    """Single OCR word with position and confidence.

    Attributes:
        text: Recognized token.
        confidence: Tesseract confidence from 0 to 100.
        left: Left coordinate in the OCR image.
        top: Top coordinate in the OCR image.
        width: Token width.
        height: Token height.
        line_key: Tuple-like key encoded as text for grouping.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    line_key: str


@dataclass(frozen=True)
class OCRLine:
    """OCR text line grouped from Tesseract word data.

    Attributes:
        text: Full line text.
        confidence: Mean confidence of words in the line.
        left: Left coordinate.
        top: Top coordinate.
        width: Line width.
        height: Line height.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR result used by the parser and JSON serializer.

    Attributes:
        text: Full raw text from OCR.
        mean_confidence: Mean confidence among non-empty OCR words.
        words: Word-level OCR records.
        lines: Line-level OCR records.
    """

    text: str
    mean_confidence: float
    words: list[OCRWord]
    lines: list[OCRLine]

    def to_dict(self) -> dict[str, Any]:
        """Convert the OCR result to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of text, confidence, words, and lines.
        """

        return {
            "texto_completo": self.text,
            "confianca_media": round(self.mean_confidence, 2),
            "linhas": [
                {
                    "texto": line.text,
                    "confianca": round(line.confidence, 2),
                    "bbox": {
                        "x": line.left,
                        "y": line.top,
                        "largura": line.width,
                        "altura": line.height,
                    },
                }
                for line in self.lines
            ],
            "palavras": [
                {
                    "texto": word.text,
                    "confianca": round(word.confidence, 2),
                    "bbox": {
                        "x": word.left,
                        "y": word.top,
                        "largura": word.width,
                        "altura": word.height,
                    },
                    "linha": word.line_key,
                }
                for word in self.words
            ],
        }


def _load_pytesseract() -> Any:
    """Import pytesseract lazily so non-OCR imports remain lightweight.

    Returns:
        Imported pytesseract module.

    Raises:
        RuntimeError: If pytesseract is not installed.
    """

    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract não está instalado. Instale as dependências do projeto "
            "e garanta que o binário tesseract esteja disponível no sistema."
        ) from exc
    return pytesseract, Output


def parse_confidence(raw_confidence: object) -> float:
    """Parse Tesseract confidence values safely.

    Args:
        raw_confidence: Confidence value returned by Tesseract.

    Returns:
        Confidence clipped to the 0-100 interval, or 0 for invalid values.
    """

    try:
        value = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.0
    if value < 0:
        return 0.0
    return min(value, 100.0)


def build_words(data: dict[str, list[Any]]) -> list[OCRWord]:
    """Build clean word objects from ``pytesseract.image_to_data`` output.

    Args:
        data: Dictionary returned by pytesseract with ``Output.DICT``.

    Returns:
        List of non-empty OCR words.
    """

    words: list[OCRWord] = []
    total = len(data.get("text", []))
    for index in range(total):
        text = str(data["text"][index]).strip()
        confidence = parse_confidence(data["conf"][index])
        if not text or confidence <= 0:
            continue
        line_key = "-".join(
            str(data.get(key, [0] * total)[index])
            for key in ("page_num", "block_num", "par_num", "line_num")
        )
        words.append(
            OCRWord(
                text=text,
                confidence=confidence,
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                line_key=line_key,
            )
        )
    return words


def group_words_into_lines(words: list[OCRWord]) -> list[OCRLine]:
    """Group OCR words into text lines using Tesseract line identifiers.

    Args:
        words: OCR words with their line keys.

    Returns:
        Ordered line-level OCR records.
    """

    grouped: dict[str, list[OCRWord]] = {}
    for word in words:
        grouped.setdefault(word.line_key, []).append(word)

    lines: list[OCRLine] = []
    for line_words in grouped.values():
        ordered = sorted(line_words, key=lambda item: item.left)
        text = " ".join(word.text for word in ordered).strip()
        left = min(word.left for word in ordered)
        top = min(word.top for word in ordered)
        right = max(word.left + word.width for word in ordered)
        bottom = max(word.top + word.height for word in ordered)
        confidence = sum(word.confidence for word in ordered) / len(ordered)
        lines.append(
            OCRLine(
                text=text,
                confidence=confidence,
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )
        )
    return sorted(lines, key=lambda line: (line.top, line.left))


def run_tesseract_ocr(image: np.ndarray, language: str, config: str) -> OCRResult:
    """Run Tesseract OCR and return text, words, lines, and confidence.

    Args:
        image: Preprocessed OCR image.
        language: Tesseract language expression, for example ``por+eng``.
        config: Extra Tesseract command-line flags.

    Returns:
        Structured OCR result.

    Raises:
        RuntimeError: If Tesseract is missing or OCR execution fails.
    """

    pytesseract, output = _load_pytesseract()
    try:
        raw_text = pytesseract.image_to_string(image, lang=language, config=config)
        data = pytesseract.image_to_data(image, lang=language, config=config, output_type=output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Binário tesseract não encontrado. No Dockerfile do projeto ele é instalado via apt; "
            "fora do Docker, instale Tesseract OCR e o pacote de idioma português."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"Falha ao executar OCR com Tesseract: {exc}") from exc

    words = build_words(data)
    lines = group_words_into_lines(words)
    mean_confidence = (
        sum(word.confidence for word in words) / len(words)
        if words
        else 0.0
    )
    return OCRResult(
        text=raw_text.strip(),
        mean_confidence=mean_confidence,
        words=words,
        lines=lines,
    )
