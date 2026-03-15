from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CardPatch:
    """Represents a detected and perspective-corrected card image patch.

    Attributes:
        image: The cropped, perspective-corrected card image as a NumPy array.
        source_image_path: Path to the original source image.
        bbox: Bounding box (x, y, w, h) in the original image coordinate space.
        detection_confidence: Confidence score from the detector (0.0 - 1.0).
        patch_index: Zero-based index of this patch within the source image.
    """

    image: np.ndarray
    source_image_path: str
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    patch_index: int
