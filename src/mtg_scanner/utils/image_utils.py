"""Image utility helpers used across the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from *path* using OpenCV, with Pillow fallback for AVIF/HEIC.

    Args:
        path: Filesystem path to the image file.

    Returns:
        A BGR NumPy array, or ``None`` if loading fails.
    """
    try:
        img = cv2.imread(path)
        if img is not None:
            return img
        # OpenCV returned None — try Pillow (handles AVIF, HEIC, etc.)
        logger.debug("cv2.imread returned None for %s, trying Pillow fallback", path)
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as pil_exc:
            logger.error("Failed to load image %s (Pillow fallback): %s", path, pil_exc)
            return None
    except Exception as exc:
        logger.error("Failed to load image %s: %s", path, exc)
        return None


def resize_with_aspect(
    image: np.ndarray, max_dim: int = 1024
) -> np.ndarray:
    """Resize *image* so that its longest dimension is at most *max_dim*.

    The aspect ratio is preserved.  Images smaller than *max_dim* are returned
    unchanged.

    Args:
        image: Input image array (H x W x C or H x W).
        max_dim: Maximum side length in pixels.

    Returns:
        Resized image array.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    Args:
        image: BGR input array.

    Returns:
        Single-channel grayscale array.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalisation to a grayscale image.

    Args:
        gray: Single-channel uint8 image.
        clip_limit: CLAHE clip limit.
        tile_grid: Tile grid size (square).

    Returns:
        Equalised grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def crop_region(
    image: np.ndarray,
    y_start_frac: float,
    y_end_frac: float,
    x_margin_px: int = 0,
) -> np.ndarray:
    """Crop a horizontal band from *image* using fractional row coordinates.

    Args:
        image: Source image (H x W x C or H x W).
        y_start_frac: Top boundary as a fraction of image height (0.0 - 1.0).
        y_end_frac: Bottom boundary as a fraction of image height (0.0 - 1.0).
        x_margin_px: Horizontal margin in pixels to remove from each side.

    Returns:
        Cropped sub-image.
    """
    h, w = image.shape[:2]
    y0 = int(h * y_start_frac)
    y1 = int(h * y_end_frac)
    x0 = x_margin_px
    x1 = w - x_margin_px if x_margin_px > 0 else w
    x1 = max(x0 + 1, x1)
    return image[y0:y1, x0:x1]


def image_to_pil(image: np.ndarray):
    """Convert a BGR NumPy array to a Pillow Image.

    Args:
        image: BGR uint8 array.

    Returns:
        ``PIL.Image.Image`` in RGB mode.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise ImportError("Pillow is required: pip install pillow") from exc

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(pil_image) -> np.ndarray:
    """Convert a Pillow Image to a BGR NumPy array.

    Args:
        pil_image: ``PIL.Image.Image`` instance.

    Returns:
        BGR uint8 NumPy array.
    """
    import numpy as np  # noqa: F811

    rgb = np.array(pil_image)
    if rgb.ndim == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_image_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """Encode *image* as a base64 string.

    Args:
        image: BGR uint8 array.
        fmt: OpenCV-compatible file extension (e.g. ``'.png'``, ``'.jpg'``).

    Returns:
        Base64-encoded string of the encoded image bytes.
    """
    import base64

    success, buf = cv2.imencode(fmt, image)
    if not success:
        raise ValueError(f"cv2.imencode failed for format {fmt!r}")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def save_image(image: np.ndarray, path: str) -> bool:
    """Write *image* to *path*.

    Creates parent directories as needed.

    Args:
        image: BGR uint8 array.
        path: Destination file path.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(path, image)
    except Exception as exc:
        logger.error("Failed to save image to %s: %s", path, exc)
        return False
