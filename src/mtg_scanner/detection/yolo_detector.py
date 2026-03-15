"""YOLOv8-based MTG card detector using Ultralytics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from mtg_scanner.config import get_config
from mtg_scanner.detection.base import BaseDetector
from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.utils.image_utils import load_image

logger = logging.getLogger(__name__)

_FALLBACK_MODEL = "yolov8n.pt"


class YOLODetector(BaseDetector):
    """Card detector that leverages a YOLOv8 model via the Ultralytics library.

    The Ultralytics package is imported lazily so that the rest of the project
    can function without it being installed.

    Args:
        model_path: Path to the ``.pt`` weights file.  If not supplied the
            value is read from ``config.yaml``.  Falls back to ``yolov8n.pt``
            (auto-downloaded by Ultralytics) when the configured path does not
            exist.
        confidence_threshold: Minimum detection confidence (0.0 – 1.0).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        cfg = get_config().detection
        self._model_path = model_path or cfg.yolo_model_path
        self._confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else cfg.confidence_threshold
        )
        self._model = None  # lazy-loaded on first use

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Import Ultralytics and load the YOLO model (once per instance).

        Falls back to ``yolov8n.pt`` when the configured weights file is not
        found on disk.
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for YOLO detection.\n"
                "Install it with:  pip install 'mtg-card-scanner[yolo]'\n"
                "or:               pip install ultralytics"
            ) from exc

        weights = self._model_path
        if not Path(weights).exists():
            logger.warning(
                "YOLO model not found at %r; falling back to %s",
                weights,
                _FALLBACK_MODEL,
            )
            weights = _FALLBACK_MODEL

        logger.info("Loading YOLO model from %s", weights)
        self._model = YOLO(weights)

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> list[CardPatch]:
        """Detect cards in *image_path* using YOLOv8 inference.

        Args:
            image_path: Path to the source image file.

        Returns:
            List of :class:`~mtg_scanner.models.card_patch.CardPatch` objects,
            one per detection that meets the confidence threshold.
        """
        if self._model is None:
            self._load_model()

        image = load_image(image_path)
        if image is None:
            logger.error("Could not load image: %s", image_path)
            return []

        try:
            results = self._model(image, conf=self._confidence_threshold, verbose=False)
        except Exception as exc:
            logger.error("YOLO inference failed for %s: %s", image_path, exc)
            return []

        patches: list[CardPatch] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                try:
                    conf = float(boxes.conf[i].item())
                    if conf < self._confidence_threshold:
                        continue

                    # xyxy bounding box in pixel coordinates
                    x1, y1, x2, y2 = (int(v.item()) for v in boxes.xyxy[i])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    patch = CardPatch(
                        image=crop,
                        source_image_path=image_path,
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        detection_confidence=conf,
                        patch_index=len(patches),
                    )
                    patches.append(patch)

                except Exception as exc:
                    logger.warning("Failed to process YOLO box %d: %s", i, exc)
                    continue

        # Sort top-to-bottom, left-to-right
        patches.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
        for idx, patch in enumerate(patches):
            patch.patch_index = idx

        logger.info("YOLO detected %d card(s) in %s", len(patches), image_path)
        return patches
