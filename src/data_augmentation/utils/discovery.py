"""Dataset discovery helpers for grouping document images, masks, and annotations."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data_augmentation.config import AugmentationConfig
from data_augmentation.models import DocumentTriplet


MASK_TOKENS = ("mask", "mascara", "bbox", "boxes", "label", "labels", "gt", "segmentation")
IMAGE_TOKENS = ("image", "imagem", "img", "doc", "document", "documento", "original", "in")
SUFFIX_PATTERN = re.compile(
    (
        r"([_\-\s]?(gt[_\-\s]?ocr|gt[_\-\s]?segmentation|segmentation|mask|mascara|"
        r"bbox|boxes|labels?|gt|image|imagem|img|doc|documento?|original|in))+$"
    ),
    re.IGNORECASE,
)


def discover_triplets(config: AugmentationConfig) -> list[DocumentTriplet]:
    """Find document image, mask, and annotation triplets.

    Args:
        config: Runtime configuration with dataset root, classes, and extensions.

    Returns:
        List of triplets discovered across all configured classes.

    Raises:
        FileNotFoundError: If a configured class directory does not exist.
        ValueError: If files cannot be grouped into valid triplets.
    """

    triplets: list[DocumentTriplet] = []
    for class_name in config.class_names:
        class_dir = config.dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {class_dir}")

        class_triplets = _discover_class_triplets(class_dir, class_name, config.image_extensions)
        class_triplets = sorted(class_triplets, key=lambda item: item.document_id)
        if config.max_documents_per_class is not None:
            class_triplets = class_triplets[: config.max_documents_per_class]
        triplets.extend(class_triplets)

    return triplets


def _discover_class_triplets(
    class_dir: Path,
    class_name: str,
    image_extensions: tuple[str, ...],
) -> list[DocumentTriplet]:
    """Discover valid triplets inside one class directory.

    Args:
        class_dir: Directory containing files for a single class.
        class_name: Class label associated with the directory.
        image_extensions: Allowed image suffixes.

    Returns:
        Valid triplets for the class.

    Raises:
        ValueError: If any grouped document does not contain one txt and two images.
    """
    grouped: dict[str, list[Path]] = defaultdict(list)
    allowed_extensions = set(image_extensions) | {".txt"}

    for path in class_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        grouped[_document_key(path.stem)].append(path)

    triplets: list[DocumentTriplet] = []
    errors: list[str] = []
    for document_id, paths in grouped.items():
        txt_files = [path for path in paths if path.suffix.lower() == ".txt"]
        image_files = [path for path in paths if path.suffix.lower() in image_extensions]
        if len(txt_files) != 1 or len(image_files) != 2:
            errors.append(
                f"{class_dir.name}/{document_id}: esperado 1 txt e 2 imagens, "
                f"encontrado {len(txt_files)} txt e {len(image_files)} imagens"
            )
            continue

        mask_path = _select_mask(image_files)
        image_path = next(path for path in image_files if path != mask_path)
        triplets.append(
            DocumentTriplet(
                class_name=class_name,
                document_id=document_id,
                image_path=image_path,
                mask_path=mask_path,
                annotation_path=txt_files[0],
            )
        )

    if errors:
        sample = "\n".join(errors[:10])
        raise ValueError(
            f"Falha ao descobrir trincas em {class_dir}. Primeiros problemas:\n{sample}"
        )

    return triplets


def _document_key(stem: str) -> str:
    """Derive the shared document identifier from a file stem.

    Args:
        stem: File name without extension.

    Returns:
        Stem with known image, mask, and OCR suffixes removed.
    """
    key = SUFFIX_PATTERN.sub("", stem).strip("_- ")
    return key or stem


def _select_mask(paths: list[Path]) -> Path:
    """Choose the mask image from two image candidates.

    Args:
        paths: Two image paths belonging to the same document.

    Returns:
        Path that most likely represents the segmentation mask.
    """
    named_masks = [path for path in paths if _contains_any(path.stem, MASK_TOKENS)]
    if len(named_masks) == 1:
        return named_masks[0]

    named_images = [path for path in paths if _contains_any(path.stem, IMAGE_TOKENS)]
    if len(named_images) == 1:
        return next(path for path in paths if path != named_images[0])

    scores = [(path, _mask_likeness_score(path)) for path in paths]
    return max(scores, key=lambda item: item[1])[0]


def _contains_any(value: str, tokens: tuple[str, ...]) -> bool:
    """Check whether text contains any marker token.

    Args:
        value: Text to inspect.
        tokens: Candidate marker tokens.

    Returns:
        ``True`` when any token is present, otherwise ``False``.
    """
    value_lower = value.lower()
    return any(token in value_lower for token in tokens)


def _mask_likeness_score(path: Path) -> float:
    """Score how likely an image is to be a binary-ish mask.

    Args:
        path: Image path to inspect with OpenCV.

    Returns:
        Heuristic score based on near-binary pixels and black-pixel ratio.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0
    array = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    near_binary = np.mean((array < 20) | (array > 235))
    black_ratio = np.mean(array < 20)
    return float(near_binary + black_ratio)
