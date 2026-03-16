"""OpenCV-based MTG card detector using contour analysis with grid fallback."""

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
# Card geometry constants
# ---------------------------------------------------------------------------

# Standard MTG card aspect ratio (63 mm × 88 mm = 0.7159…)
_CARD_IDEAL_ASPECT_RATIO: float = 0.714

# Grid fallback: refuse to produce more than this many cells (rows × cols).
# Prevents false positives on dense textures that Hough mistakes for a grid.
_GRID_MAX_CELLS: int = 24

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort four corner points into a consistent (TL, TR, BR, BL) order."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective transformation to extract a rectangular region."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


# ---------------------------------------------------------------------------
# Detector implementation
# ---------------------------------------------------------------------------


class OpenCVDetector(BaseDetector):
    """Card detector using classical computer vision with a grid-layout fallback.

    Primary pipeline: Canny edges → contour analysis → NMS.
    Fallback (when primary finds 0 cards): Hough lines → grid subdivision.
    """

    def __init__(self, config_override: Optional[dict] = None) -> None:
        cfg = get_config().detection
        self._confidence_threshold: float = cfg.confidence_threshold
        self._aspect_ratio_min: float = cfg.aspect_ratio_min
        self._aspect_ratio_max: float = cfg.aspect_ratio_max
        self._min_card_area_px: int = cfg.min_card_area_px
        self._max_card_area_frac: float = cfg.max_card_area_frac
        self._save_debug: bool = cfg.save_debug
        self._iou_nms_threshold: float = cfg.iou_nms_threshold

        if config_override:
            for k, v in config_override.items():
                setattr(self, f"_{k}", v)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        edges = cv2.Canny(enhanced, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ------------------------------------------------------------------
    # Contour filtering
    # ------------------------------------------------------------------

    def _is_valid_card_contour(
        self, contour: np.ndarray, image_area: int
    ) -> tuple[bool, float]:
        area = cv2.contourArea(contour)
        if area < self._min_card_area_px:
            return False, 0.0
        if area > self._max_card_area_frac * image_area:
            return False, 0.0

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if not (4 <= len(approx) <= 6):
            return False, 0.0

        _, (w, h), _ = cv2.minAreaRect(contour)
        if w == 0 or h == 0:
            return False, 0.0

        aspect = min(w, h) / max(w, h)
        if not (self._aspect_ratio_min <= aspect <= self._aspect_ratio_max):
            return False, 0.0

        confidence = max(0.0, 1.0 - abs(aspect - _CARD_IDEAL_ASPECT_RATIO) / _CARD_IDEAL_ASPECT_RATIO)
        return True, float(confidence)

    # ------------------------------------------------------------------
    # Grid fallback detection
    # ------------------------------------------------------------------

    def _detect_grid(self, image: np.ndarray) -> list[tuple[float, tuple, np.ndarray]]:
        """Detect cards in grid layouts using Hough line intersection.

        When cards are laid out in a touching grid, contour detection finds the
        outer border of the entire group instead of individual cards.  This
        fallback:
        1. Detects horizontal and vertical lines via HoughLinesP.
        2. Clusters nearby lines to find row/column separators.
        3. Extracts each grid cell as a card candidate.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=100,
            minLineLength=min(w, h) // 6, maxLineGap=20,
        )
        if lines is None:
            return []

        h_lines: list[int] = []  # y positions of horizontal lines
        v_lines: list[int] = []  # x positions of vertical lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170:           # nearly horizontal
                h_lines.append((y1 + y2) // 2)
            elif 80 < angle < 100:                  # nearly vertical
                v_lines.append((x1 + x2) // 2)

        def _cluster(positions: list[int], gap: int = 20) -> list[int]:
            if not positions:
                return []
            positions = sorted(set(positions))
            clusters = [[positions[0]]]
            for p in positions[1:]:
                if p - clusters[-1][-1] < gap:
                    clusters[-1].append(p)
                else:
                    clusters.append([p])
            return [int(np.mean(c)) for c in clusters]

        ys = _cluster(h_lines)
        xs = _cluster(v_lines)

        # Add image borders so we cover the full image
        if not ys or ys[0] > h * 0.1:
            ys = [0] + ys
        if not ys or ys[-1] < h * 0.9:
            ys = ys + [h]
        if not xs or xs[0] > w * 0.1:
            xs = [0] + xs
        if not xs or xs[-1] < w * 0.9:
            xs = xs + [w]

        rows = len(ys) - 1
        cols = len(xs) - 1

        if rows < 1 or cols < 1 or rows * cols > _GRID_MAX_CELLS:
            return []

        # Each cell's aspect ratio must look like a card
        cell_w = (xs[-1] - xs[0]) / cols
        cell_h = (ys[-1] - ys[0]) / rows
        aspect = min(cell_w, cell_h) / max(cell_w, cell_h)
        if not (0.50 <= aspect <= 0.95):
            return []

        logger.info("Grid fallback: detected %d×%d grid", rows, cols)

        candidates = []
        for r in range(rows):
            for c in range(cols):
                y0, y1_ = ys[r], ys[r + 1]
                x0, x1_ = xs[c], xs[c + 1]
                cell = image[y0:y1_, x0:x1_]
                if cell.size == 0:
                    continue
                bbox = (int(x0), int(y0), int(x1_ - x0), int(y1_ - y0))
                ideal = 0.714
                conf = max(0.0, 1.0 - abs(aspect - ideal) / ideal)
                candidates.append((conf, bbox, cell.copy()))

        return candidates

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> list[CardPatch]:
        image = load_image(image_path)
        if image is None:
            logger.error("Could not load image: %s", image_path)
            return []

        try:
            edges = self._preprocess(image)
        except Exception as exc:
            logger.error("Preprocessing failed for %s: %s", image_path, exc)
            return []

        image_area = image.shape[0] * image.shape[1]
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, tuple, np.ndarray]] = []

        for contour in contours:
            valid, confidence = self._is_valid_card_contour(contour, image_area)
            if not valid or confidence < self._confidence_threshold:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) > 4:
                hull = cv2.convexHull(approx)
                approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx) != 4:
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
            candidates.append((confidence, (int(x), int(y), int(w), int(h)), warped))

        # NMS
        candidates.sort(key=lambda c: c[1][2] * c[1][3], reverse=True)
        kept: list[tuple[float, tuple, np.ndarray]] = []
        for conf, bbox, warped in candidates:
            if any(self._iou(bbox, k[1]) > self._iou_nms_threshold for k in kept):
                continue
            kept.append((conf, bbox, warped))

        # Grid fallback when primary detection finds nothing
        if not kept:
            logger.info("Primary detection found 0 cards — trying grid fallback")
            kept = self._detect_grid(image)

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

        patches.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
        for idx, patch in enumerate(patches):
            patch.patch_index = idx

        if self._save_debug:
            self._save_debug_image(image_path, image, edges, patches)

        logger.info("Detected %d card(s) in %s", len(patches), image_path)
        return patches

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
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
        try:
            stem = Path(image_path).stem
            debug_dir = Path("output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{stem}_edges.png"), edges)
            annotated = image.copy()
            for patch in patches:
                x, y, w, h = patch.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    annotated, f"#{patch.patch_index}",
                    (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
            cv2.imwrite(str(debug_dir / f"{stem}_detections.png"), annotated)
            logger.info("Debug images saved to %s", debug_dir)
        except Exception as exc:
            logger.warning("Could not save debug images: %s", exc)
