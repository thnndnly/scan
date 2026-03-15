"""OpenCV-based MTG card detector using contour analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from mtg_scanner.config import get_config
from mtg_scanner.detection.base import BaseDetector
from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.utils.image_utils import load_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort four corner points into a consistent (TL, TR, BR, BL) order.

    Args:
        pts: Array of shape (4, 2) containing corner (x, y) coordinates.

    Returns:
        Array of shape (4, 2) ordered as top-left, top-right,
        bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coords: smallest → TL, largest → BR
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coords: smallest → TR, largest → BL
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective transformation to extract a rectangular region.

    Given four corner points that define a quadrilateral in *image*, this
    function computes the destination rectangle dimensions (preserving the
    natural aspect ratio of the detected shape) and performs the warp.

    Args:
        image: Source image.
        pts: Array of shape (4, 2) with the four corner points.

    Returns:
        Warped (perspective-corrected) image patch.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Width: max of top and bottom edge lengths
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Height: max of left and right edge lengths
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


# ---------------------------------------------------------------------------
# Detector implementation
# ---------------------------------------------------------------------------


class OpenCVDetector(BaseDetector):
    """Card detector that uses classical computer vision with OpenCV.

    Pipeline
    --------
    1. Convert to grayscale.
    2. Gaussian blur (5 × 5).
    3. CLAHE for contrast normalisation.
    4. Canny edge detection.
    5. Morphological closing to bridge edge gaps.
    6. Contour detection (RETR_EXTERNAL).
    7. Approximate polygon fitting; keep quadrilaterals / pentagons / hexagons
       that satisfy minimum area and aspect-ratio constraints.
    8. Perspective correction via four-point transform.

    Args:
        config_override: Optional dict of config values that override
            ``config.yaml`` settings for the ``detection`` section.
    """

    def __init__(self, config_override: Optional[dict] = None) -> None:
        cfg = get_config().detection
        self._confidence_threshold: float = cfg.confidence_threshold
        self._aspect_ratio_min: float = cfg.aspect_ratio_min
        self._aspect_ratio_max: float = cfg.aspect_ratio_max
        self._min_card_area_px: int = cfg.min_card_area_px
        self._save_debug: bool = cfg.save_debug

        if config_override:
            for k, v in config_override.items():
                setattr(self, f"_{k}", v)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Return a binary edge map suitable for contour extraction.

        Args:
            image: BGR source image.

        Returns:
            Binary edge image (single channel, uint8).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to suppress high-frequency noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # CLAHE: improve local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Canny edge detection — lower thresholds catch more card borders
        edges = cv2.Canny(enhanced, 30, 100)

        # Morphological closing: small kernel so adjacent cards don't merge
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return closed

    # ------------------------------------------------------------------
    # Contour filtering
    # ------------------------------------------------------------------

    def _is_valid_card_contour(
        self, contour: np.ndarray, image_area: int
    ) -> tuple[bool, float]:
        """Check whether *contour* looks like an MTG card.

        Args:
            contour: Contour array from ``cv2.findContours``.
            image_area: Total image area in pixels (used to normalise size).

        Returns:
            ``(valid, confidence)`` where *confidence* is a rough estimate
            based on aspect ratio deviation from the ideal MTG ratio (~0.714).
        """
        area = cv2.contourArea(contour)
        if area < self._min_card_area_px:
            return False, 0.0

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        n_corners = len(approx)

        # Only accept shapes with 4-6 vertices
        if not (4 <= n_corners <= 6):
            return False, 0.0

        # Fit a rotated bounding rectangle and check aspect ratio
        _, (w, h), _ = cv2.minAreaRect(contour)
        if w == 0 or h == 0:
            return False, 0.0

        aspect = min(w, h) / max(w, h)
        if not (self._aspect_ratio_min <= aspect <= self._aspect_ratio_max):
            return False, 0.0

        # Confidence based on how close the aspect ratio is to the ideal
        ideal_ratio = 0.714
        aspect_diff = abs(aspect - ideal_ratio) / ideal_ratio
        confidence = max(0.0, 1.0 - aspect_diff)

        return True, float(confidence)

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> list[CardPatch]:
        """Detect cards in the image at *image_path*.

        Args:
            image_path: Path to the source image file.

        Returns:
            List of :class:`~mtg_scanner.models.card_patch.CardPatch` objects
            sorted by (y, x) position in the original image.
        """
        image = load_image(image_path)
        if image is None:
            logger.error("Could not load image: %s", image_path)
            return []

        try:
            edges = self._preprocess(image)
        except Exception as exc:
            logger.error("Preprocessing failed for %s: %s", image_path, exc)
            return []

        # RETR_LIST finds all contours including those inside merged blobs,
        # which is essential when cards are touching or slightly overlapping.
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.info("No contours found in %s", image_path)
            return []

        image_area = image.shape[0] * image.shape[1]

        # Collect all valid candidates as (confidence, bbox, warped) tuples
        candidates: list[tuple[float, tuple, np.ndarray]] = []

        for contour in contours:
            valid, confidence = self._is_valid_card_contour(contour, image_area)
            if not valid or confidence < self._confidence_threshold:
                continue

            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Use the convex hull for the four-point transform if polygon has > 4 points
            if len(approx) > 4:
                hull = cv2.convexHull(approx)
                approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

            if len(approx) != 4:
                # Fall back to bounding rectangle if we can't get exactly 4 corners
                x, y, w, h = cv2.boundingRect(contour)
                approx = np.array(
                    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
                ).reshape(-1, 1, 2)

            pts = approx.reshape(4, 2).astype(np.float32)

            try:
                warped = four_point_transform(image, pts)
            except Exception as exc:
                logger.warning("Perspective transform failed: %s", exc)
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox = (int(x), int(y), int(w), int(h))
            candidates.append((confidence, bbox, warped))

        # NMS: sort by area descending (largest first), then suppress overlapping boxes
        candidates.sort(key=lambda c: c[1][2] * c[1][3], reverse=True)
        kept: list[tuple[float, tuple, np.ndarray]] = []
        for conf, bbox, warped in candidates:
            if any(self._iou(bbox, k[1]) > 0.3 for k in kept):
                continue
            kept.append((conf, bbox, warped))

        patches: list[CardPatch] = []
        for conf, bbox, warped in kept:
            patch = CardPatch(
                image=warped,
                source_image_path=image_path,
                bbox=bbox,
                detection_confidence=conf,
                patch_index=len(patches),
            )
            patches.append(patch)

        # Sort by top-to-bottom, left-to-right order
        patches.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
        # Re-index after sort
        for idx, patch in enumerate(patches):
            patch.patch_index = idx

        # Save debug images if enabled
        if self._save_debug:
            self._save_debug_image(image_path, image, edges, patches)

        logger.info("Detected %d card(s) in %s", len(patches), image_path)
        return patches

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
        """Compute Intersection-over-Union for two (x, y, w, h) bounding boxes."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = ix * iy
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def _save_debug_image(
        self, image_path: str, image: np.ndarray, edges: np.ndarray, patches: list
    ) -> None:
        """Save edge map and annotated detection image to output/debug/."""
        try:
            stem = Path(image_path).stem
            debug_dir = Path("output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Save edge map
            cv2.imwrite(str(debug_dir / f"{stem}_edges.png"), edges)

            # Save annotated image
            annotated = image.copy()
            for patch in patches:
                x, y, w, h = patch.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    annotated,
                    f"#{patch.patch_index}",
                    (x + 5, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(str(debug_dir / f"{stem}_detections.png"), annotated)
            logger.info("Debug images saved to %s", debug_dir)
        except Exception as exc:
            logger.warning("Could not save debug images: %s", exc)
