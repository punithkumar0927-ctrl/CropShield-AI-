from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .config import MIN_IMAGE_DIMENSION


@dataclass(frozen=True)
class ImageQualityAssessment:
    width: int
    height: int
    brightness: float
    blur_score: float
    glare_fraction: float
    green_pixel_fraction: float
    leaf_coverage_fraction: float
    leaf_contour_count: int
    leaf_mask: np.ndarray
    acceptable: bool
    feedback: tuple[str, ...]


def segment_leaf_like_regions(image: Image.Image) -> tuple[np.ndarray, float, int, tuple[int, int, int, int] | None]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, 512 / max(width, height))
    resized = rgb.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    hsv = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([20, 35, 25]), np.array([100, 255, 255]))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(16, int(mask.shape[0] * mask.shape[1] * 0.002))
    significant = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    output_mask = np.zeros(mask.shape, dtype=np.uint8)
    bounding_box = None
    if significant:
        cv2.drawContours(output_mask, significant, -1, 255, thickness=cv2.FILLED)
        largest = max(significant, key=cv2.contourArea)
        x, y, box_width, box_height = cv2.boundingRect(largest)
        inv_scale = 1 / scale
        x1 = max(0, int(x * inv_scale))
        y1 = max(0, int(y * inv_scale))
        x2 = min(width, int((x + box_width) * inv_scale))
        y2 = min(height, int((y + box_height) * inv_scale))
        bounding_box = (x1, y1, x2, y2)
    output_mask = cv2.resize(output_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    coverage = float(np.count_nonzero(output_mask) / output_mask.size)
    return output_mask > 0, coverage, len(significant), bounding_box


def crop_largest_leaf_like_region(image: Image.Image) -> Image.Image | None:
    _mask, coverage, _contour_count, bounding_box = segment_leaf_like_regions(image)
    if bounding_box is None or coverage <= 0:
        return None
    x1, y1, x2, y2 = bounding_box
    padding_x = max(1, int((x2 - x1) * 0.08))
    padding_y = max(1, int((y2 - y1) * 0.08))
    left = max(0, x1 - padding_x)
    top = max(0, y1 - padding_y)
    right = min(image.width, x2 + padding_x)
    bottom = min(image.height, y2 + padding_y)
    return image.convert("RGB").crop((left, top, right, bottom))


def assess_image_quality(image: Image.Image) -> ImageQualityAssessment:
    """Assess capture quality and estimate green leaf-like regions, not leaf identity."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    rgb.thumbnail((512, 512), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32)
    red, green, blue = (pixels[:, :, channel] for channel in range(3))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    brightness = float(np.mean(luminance))

    gray = luminance
    if min(gray.shape) >= 3:
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )
        blur_score = float(np.var(laplacian))
    else:
        blur_score = 0.0

    maximum = np.max(pixels, axis=2)
    minimum = np.min(pixels, axis=2)
    glare_fraction = float(np.mean((luminance >= 245) & ((maximum - minimum) <= 18)))
    green_pixel_fraction = float(np.mean((green > red * 1.05) & (green > blue * 0.85)))
    leaf_mask, leaf_coverage_fraction, leaf_contour_count, _bounding_box = segment_leaf_like_regions(rgb)

    feedback: list[str] = []
    if min(width, height) < MIN_IMAGE_DIMENSION:
        feedback.append(
            f"Image is too small. Please upload a higher-resolution photo of at least "
            f"{MIN_IMAGE_DIMENSION} x {MIN_IMAGE_DIMENSION} pixels."
        )
    if brightness < 35:
        feedback.append("The photo is very dark. Move to brighter, indirect light.")
    elif brightness > 235:
        feedback.append("The photo is overexposed. Avoid direct sunlight and retake it in even light.")
    if blur_score < 20:
        feedback.append("The subject looks out of focus. Hold the camera steady and tap to focus on the leaf.")
    if glare_fraction > 0.15:
        feedback.append("Bright glare is obscuring detail. Change the angle or move out of direct sunlight.")
    if max(width / height, height / width) > 3:
        feedback.append("The framing is unusually narrow. Center one affected leaf and include its edges.")
    if green_pixel_fraction < 0.03:
        feedback.append(
            "There is little green-colored area. Check that the photo shows the plant; "
            "this color check cannot identify a leaf and may miss yellowing or other leaf colors."
        )
    if leaf_coverage_fraction <= 0.01:
        feedback.append(
            "No clear green leaf-like region was detected. This color segmentation is experimental "
            "and can miss yellow, brown, or differently lit leaves."
        )
    elif leaf_coverage_fraction < 0.2:
        feedback.append(
            f"Leaf-like color occupies about {leaf_coverage_fraction:.0%} of the image. "
            "Move closer so the affected leaf fills more of the frame."
        )
    if leaf_contour_count > 1:
        feedback.append(
            f"Multiple leaf-like regions detected ({leaf_contour_count}). For a clearer result, "
            "photograph one affected leaf."
        )

    flat_or_blank = float(np.std(luminance)) < 2.0
    if flat_or_blank:
        feedback.append("Image appears blank or corrupted. Please upload a clear plant photo.")
    blocking = (
        min(width, height) < MIN_IMAGE_DIMENSION
        or brightness < 20
        or brightness > 245
        or blur_score < 5
        or flat_or_blank
    )
    return ImageQualityAssessment(
        width=width,
        height=height,
        brightness=round(brightness, 2),
        blur_score=round(blur_score, 2),
        glare_fraction=round(glare_fraction, 4),
        green_pixel_fraction=round(green_pixel_fraction, 4),
        leaf_coverage_fraction=round(leaf_coverage_fraction, 4),
        leaf_contour_count=leaf_contour_count,
        leaf_mask=leaf_mask,
        acceptable=not blocking,
        feedback=tuple(feedback),
    )
