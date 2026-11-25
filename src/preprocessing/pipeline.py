"""Reproducible preprocessing pipeline for anterior-segment eye images.

Order of operations:
1. Pupillary / ocular ROI crop
2. Specular glare / reflection reduction
3. Local contrast enhancement via CLAHE (LAB L-channel)
4. Resize to 224x224x3 (RGB) and ImageNet normalization
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def crop_pupillary_roi(image_bgr: np.ndarray) -> np.ndarray:
    """Locate and crop the pupillary/ocular region of interest.

    Uses Hough Circle detection on a blurred grayscale image to find the iris/pupil
    circle. If a plausible central circle is detected, crops a padded square around it;
    otherwise falls back to a safe centered square crop preserving the eye region.
    """
    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        return image_bgr

    min_dim = min(h, w)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 2),
        param1=80,
        param2=30,
        minRadius=max(8, int(min_dim * 0.12)),
        maxRadius=max(12, int(min_dim * 0.48)),
    )

    if circles is not None and len(circles) > 0:
        cx, cy, r = circles[0][0]
        # Accept circle only if its center is reasonably near the middle of the frame
        if 0.2 * w <= cx <= 0.8 * w and 0.2 * h <= cy <= 0.8 * h:
            pad_r = int(r * 1.45)
            x1 = max(0, int(cx - pad_r))
            y1 = max(0, int(cy - pad_r))
            x2 = min(w, int(cx + pad_r))
            y2 = min(h, int(cy + pad_r))
            if (x2 - x1) >= max(16, int(min_dim * 0.35)) and (y2 - y1) >= max(16, int(min_dim * 0.35)):
                return image_bgr[y1:y2, x1:x2]

    # Safe fallback: central square crop (90% of min_dim)
    crop_size = max(16, int(min_dim * 0.90))
    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2
    return image_bgr[y1 : y1 + crop_size, x1 : x1 + crop_size]


def remove_glare(image_bgr: np.ndarray) -> np.ndarray:
    """Suppress small specular flash reflections without erasing lens opacity.

    Targets bright, low-saturation specular spots (V > 235, S < 45) whose connected
    area is < 3% of the image area, and inpaints them with Telea's method.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    raw_mask = ((v > 235) & (s < 45)).astype(np.uint8) * 255

    if not np.any(raw_mask):
        return image_bgr

    h, w = image_bgr.shape[:2]
    max_spot_area = 0.03 * (h * w)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    filtered_mask = np.zeros_like(raw_mask)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if 0 < area <= max_spot_area:
            filtered_mask[labels == label_idx] = 255

    if not np.any(filtered_mask):
        return image_bgr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
    return cv2.inpaint(image_bgr, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance local contrast on the L channel of LAB color space."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_chan)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def resize_and_normalize(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Convert BGR -> RGB, resize to (size, size, 3), and apply ImageNet normalization."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interp)
    normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    return normalized


def preprocess_image(image_bgr: np.ndarray, size: int = IMAGE_SIZE, max_work_dim: int = 512) -> np.ndarray:
    """Full deterministic preprocessing pipeline returning float32 RGB (size, size, 3)."""
    h, w = image_bgr.shape[:2]
    max_side = max(h, w)
    if max_side > max_work_dim:
        scale = max_work_dim / float(max_side)
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    roi = crop_pupillary_roi(image_bgr)
    deglared = remove_glare(roi)
    enhanced = apply_clahe(deglared)
    return resize_and_normalize(enhanced, size=size)


def augment_normalized_rgb(image_rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Lightweight train-only augmentation on normalized (224, 224, 3) float32 images."""
    gen = rng if rng is not None else np.random.default_rng()
    out = image_rgb.copy()

    # Random horizontal flip
    if gen.random() < 0.5:
        out = out[:, ::-1, :].copy()

    # Small random rotation (-15 to +15 degrees)
    if gen.random() < 0.6:
        angle = float(gen.uniform(-15.0, 15.0))
        h, w = out.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Slight brightness/contrast jitter in normalized space
    if gen.random() < 0.5:
        scale = float(gen.uniform(0.9, 1.1))
        shift = float(gen.uniform(-0.1, 0.1))
        out = out * scale + shift

    return out.astype(np.float32)
