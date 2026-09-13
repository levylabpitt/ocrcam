"""Frame preprocessing to improve OCR accuracy/consistency:
- blur detection, to skip OCR on garbage frames (motion blur, compression stutter)
- contrast enhancement + capped upscaling, since small/low-contrast text is
  the main thing that trips up any OCR engine on a compressed RTSP feed.
"""
import cv2
import numpy as np


def blur_score(frame) -> float:
    """Higher = sharper. Uses variance of the Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_too_blurry(frame, threshold: float = 60.0) -> bool:
    return blur_score(frame) < threshold


def enhance(frame, target_width: int = 1280, max_upscale: float = 2.0):
    """Boost local contrast (CLAHE) + mild denoise, and upscale only if the
    frame/crop is smaller than `target_width` (capped at `max_upscale`x).

    Upscaling an already-large frame (e.g. a full 1920px-wide feed) just
    makes the OCR model do far more work for no accuracy gain and tanks
    speed - it's only useful for small crops (e.g. a tight ROI on a sign).
    """
    h, w = frame.shape[:2]
    scale = min(max_upscale, max(1.0, target_width / w))
    if scale > 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
    return frame
