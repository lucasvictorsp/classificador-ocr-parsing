"""Lightweight OCR-oriented preprocessing routines."""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_ocr(image: np.ndarray, min_width: int = 1200, max_width: int = 2200) -> np.ndarray:
    """Resize a document to an OCR-friendly width.

    Args:
        image: BGR or grayscale image.
        min_width: Width below which the image is upscaled.
        max_width: Width above which the image is downscaled.

    Returns:
        Resized image with aspect ratio preserved.
    """

    height, width = image.shape[:2]
    target_width = width
    interpolation = cv2.INTER_LINEAR
    if width < min_width:
        target_width = min_width
        interpolation = cv2.INTER_CUBIC
    elif width > max_width:
        target_width = max_width
        interpolation = cv2.INTER_AREA
    if target_width == width:
        return image.copy()
    scale = target_width / float(width)
    target_size = (target_width, int(height * scale))
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Normalize local contrast with CLAHE.

    Args:
        gray: Grayscale image.

    Returns:
        Contrast-normalized grayscale image.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen_lightly(gray: np.ndarray) -> np.ndarray:
    """Apply a mild unsharp mask to improve text edges.

    Args:
        gray: Grayscale image.

    Returns:
        Sharpened grayscale image.
    """

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray, 1.35, blurred, -0.35, 0)


def binarize_when_useful(gray: np.ndarray) -> np.ndarray:
    """Binarize the image with Otsu while avoiding aggressive local artifacts.

    Args:
        gray: Grayscale image.

    Returns:
        Binary image optimized for Tesseract.
    """

    denoised = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Prepare a rectified document for OCR with lightweight OpenCV steps.

    Args:
        image: Rectified BGR image.

    Returns:
        Final single-channel image used by the OCR engine.
    """

    resized = resize_for_ocr(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    contrast = normalize_contrast(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, h=7, templateWindowSize=7, searchWindowSize=21)
    sharpened = sharpen_lightly(denoised)
    return binarize_when_useful(sharpened)
