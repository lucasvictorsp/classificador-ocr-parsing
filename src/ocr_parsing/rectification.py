"""Image rectification utilities based on lightweight OpenCV operations."""

from __future__ import annotations

import cv2
import numpy as np


def resize_for_detection(image: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    """Resize an image before contour detection while preserving coordinates.

    Args:
        image: BGR input image.
        max_side: Maximum side size used during detection.

    Returns:
        Tuple with resized image and scale factor from original to resized image.
    """

    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(largest_side)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Args:
        points: Array with four two-dimensional points.

    Returns:
        Ordered points as ``float32``.
    """

    rect = np.zeros((4, 2), dtype="float32")
    pts = points.reshape(4, 2).astype("float32")
    sums = pts.sum(axis=1)
    differences = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(differences)]
    rect[3] = pts[np.argmax(differences)]
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a perspective transform from four document corners.

    Args:
        image: Original BGR image.
        points: Four corner points in the original image coordinate space.

    Returns:
        Warped image that approximates a scan.
    """

    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_a), int(width_b), 1)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_a), int(height_b), 1)
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def find_document_quad(image: np.ndarray) -> np.ndarray | None:
    """Detect a quadrilateral document contour in a noisy capture.

    Args:
        image: BGR input image.

    Returns:
        Four corner points in resized-image coordinates, or ``None`` when no reliable contour is found.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")

    largest = contours[0]
    if cv2.contourArea(largest) < image_area * 0.05:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return box.astype("float32")


def rotate_to_preferred_orientation(image: np.ndarray, document_type: str) -> np.ndarray:
    """Rotate rectangular documents to the expected broad orientation.

    Args:
        image: Rectified BGR image.
        document_type: Known document type.

    Returns:
        Image rotated by 90 degrees only when the type has a clear expected orientation.
    """

    height, width = image.shape[:2]
    is_landscape_document = document_type.startswith(("CNH", "RG"))
    if is_landscape_document and height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def deskew_small_angle(image: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    """Correct a small residual text skew after perspective rectification.

    Args:
        image: BGR or grayscale image.
        max_angle: Maximum absolute angle accepted for correction.

    Returns:
        Deskewed image when a reliable small angle is found; otherwise the original image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    inverted = cv2.bitwise_not(gray)
    _, threshold = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(threshold > 0))
    if len(coords) < 100:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def rectify_document(image: np.ndarray, document_type: str) -> tuple[np.ndarray, dict[str, object]]:
    """Detect, warp, and lightly rotate a document image.

    Args:
        image: BGR image captured by a phone or scanner.
        document_type: Known document type.

    Returns:
        Tuple with rectified image and metadata about the fallback path used.
    """

    resized, scale = resize_for_detection(image)
    quad = find_document_quad(resized)
    metadata: dict[str, object] = {
        "document_contour_found": quad is not None,
        "rectification_scale": scale,
        "fallback": None,
    }
    if quad is None:
        metadata["fallback"] = "contorno_nao_encontrado; imagem_original_usada"
        rectified = image.copy()
    else:
        original_quad = quad / scale
        metadata["quad_points"] = original_quad.round(2).tolist()
        rectified = four_point_transform(image, original_quad)

    rectified = rotate_to_preferred_orientation(rectified, document_type)
    rectified = deskew_small_angle(rectified)
    return rectified, metadata
