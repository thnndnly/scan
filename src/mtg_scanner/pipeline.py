"""Main processing pipeline: detection → recognition → Scryfall lookup."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from mtg_scanner.config import get_config
from mtg_scanner.detection.base import BaseDetector
from mtg_scanner.detection.opencv_detector import OpenCVDetector
from mtg_scanner.detection.yolo_detector import YOLODetector
from mtg_scanner.lookup.cache import ScryfallCache
from mtg_scanner.lookup.scryfall_client import ScryfallClient
from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.models.recognized_card import RecognizedCard
from mtg_scanner.models.scan_result import ScanResult
from mtg_scanner.recognition.base import BaseRecognizer
from mtg_scanner.recognition.hash_recognizer import HashRecognizer
from mtg_scanner.recognition.ocr_recognizer import OCRRecognizer
from mtg_scanner.utils.image_utils import load_image, save_image

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency
_DatasetLogger = None


def _get_dataset_logger_class():
    global _DatasetLogger
    if _DatasetLogger is None:
        from mtg_scanner.dataset import DatasetLogger  # noqa: PLC0415
        _DatasetLogger = DatasetLogger
    return _DatasetLogger

_ImageArchive = None


def _get_image_archive_class():
    global _ImageArchive
    if _ImageArchive is None:
        from mtg_scanner.image_archive import ImageArchive  # noqa: PLC0415
        _ImageArchive = ImageArchive
    return _ImageArchive

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".avif", ".heic", ".heif"}


class Pipeline:
    """Orchestrates the full MTG card scanning pipeline.

    The pipeline performs three stages for each input image:

    1. **Detection** – locate card regions and produce
       :class:`~mtg_scanner.models.card_patch.CardPatch` objects.
    2. **Recognition** – identify each patch using the configured primary and
       fallback methods.
    3. **Lookup** – enrich recognised cards with Scryfall data.

    Args:
        detector: Detector instance.  Defaults to :class:`OpenCVDetector`.
        primary_recognizer: Primary recogniser.  Defaults to
            :class:`OCRRecognizer`.
        fallback_recognizer: Fallback recogniser used when the primary fails.
            Defaults to :class:`HashRecognizer`.
        scryfall_client: Scryfall client for card data enrichment.  Created
            automatically if not provided.
        save_patches: When ``True`` each detected patch is saved to disk under
            the output directory.
        output_dir: Directory for output files.
    """

    def __init__(
        self,
        detector: Optional[BaseDetector] = None,
        primary_recognizer: Optional[BaseRecognizer] = None,
        fallback_recognizer: Optional[BaseRecognizer] = None,
        scryfall_client: Optional[ScryfallClient] = None,
        save_patches: Optional[bool] = None,
        output_dir: Optional[str] = None,
        dataset_logger=None,
        archive=None,
    ) -> None:
        cfg = get_config()

        self._detector: BaseDetector = detector or self._build_detector(cfg)
        self._primary: BaseRecognizer = primary_recognizer or self._build_recognizer(
            cfg.recognition.primary_method
        )
        self._fallback: Optional[BaseRecognizer] = fallback_recognizer or self._build_recognizer(
            cfg.recognition.fallback_method
        )
        self._llm: Optional[BaseRecognizer] = None
        if cfg.recognition.llm_fallback_enabled:
            from mtg_scanner.recognition.llm_recognizer import LLMRecognizer

            self._llm = LLMRecognizer()

        cache = ScryfallCache(
            db_path=cfg.scryfall.cache_db_path,
            ttl_hours=cfg.scryfall.cache_ttl_hours,
        )
        self._scryfall: ScryfallClient = scryfall_client or ScryfallClient(
            cache=cache,
            rate_limit_ms=cfg.scryfall.rate_limit_ms,
        )

        self._save_patches = (
            save_patches if save_patches is not None else cfg.output.save_card_patches
        )
        self._output_dir = output_dir or cfg.output.output_dir

        # Dataset logger — initialise from config if not explicitly passed
        if dataset_logger is not None:
            self._dataset_logger = dataset_logger
        elif cfg.dataset.enabled:
            try:
                DatasetLogger = _get_dataset_logger_class()
                self._dataset_logger = DatasetLogger(
                    db_path=cfg.dataset.db_path,
                    save_patches=cfg.dataset.save_patches,
                )
            except Exception as exc:
                logger.warning("Could not initialise DatasetLogger: %s", exc)
                self._dataset_logger = None
        else:
            self._dataset_logger = None

        # Image archive
        if archive is not None:
            self._archive = archive
        elif cfg.archive.enabled:
            try:
                ImageArchive = _get_image_archive_class()
                self._archive = ImageArchive(
                    db_path=cfg.archive.db_path,
                    index_path=cfg.archive.index_path,
                    max_dimension=cfg.archive.max_dimension,
                    jpeg_quality=cfg.archive.jpeg_quality,
                    thumbnail_size=cfg.archive.thumbnail_size,
                    thumbnail_quality=cfg.archive.thumbnail_quality,
                )
            except Exception as exc:
                logger.warning("Could not initialise ImageArchive: %s", exc)
                self._archive = None
        else:
            self._archive = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_detector(cfg) -> BaseDetector:
        method = cfg.detection.method
        if method == "yolo":
            return YOLODetector()
        return OpenCVDetector()

    @staticmethod
    def _build_recognizer(method: str) -> BaseRecognizer:
        if method == "hash":
            return HashRecognizer()
        if method == "llm":
            from mtg_scanner.recognition.llm_recognizer import LLMRecognizer

            return LLMRecognizer()
        return OCRRecognizer()

    # ------------------------------------------------------------------
    # Recognition with fallback chain
    # ------------------------------------------------------------------

    def _recognize_with_fallback(
        self, patch: CardPatch
    ) -> tuple[Optional[str], float, str]:
        """Try primary → fallback → LLM → unknown recognition chain.

        Args:
            patch: Detected card patch to recognise.

        Returns:
            ``(card_name, confidence, method)`` where *method* is one of
            ``'ocr'``, ``'hash'``, ``'llm'``, or ``'unknown'``.
        """
        cfg = get_config().recognition

        # Primary attempt
        try:
            name, conf = self._primary.recognize(patch)
            method = cfg.primary_method
            if name is not None and conf >= cfg.ocr_confidence_threshold:
                return name, conf, method
        except Exception as exc:
            logger.warning("Primary recogniser raised an exception: %s", exc)

        # Fallback attempt
        if self._fallback is not None:
            try:
                name, conf = self._fallback.recognize(patch)
                method = cfg.fallback_method
                if name is not None and conf > 0.0:
                    return name, conf, method
            except Exception as exc:
                logger.warning("Fallback recogniser raised an exception: %s", exc)

        # LLM attempt (optional, expensive)
        if self._llm is not None:
            try:
                name, conf = self._llm.recognize(patch)
                if name is not None and conf > 0.0:
                    return name, conf, "llm"
            except Exception as exc:
                logger.warning("LLM recogniser raised an exception: %s", exc)

        return None, 0.0, "unknown"

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _save_patch(self, patch: CardPatch, output_dir: str) -> None:
        """Persist a card patch image to *output_dir* if configured to do so."""
        stem = Path(patch.source_image_path).stem
        filename = f"{stem}_patch{patch.patch_index:02d}.png"
        dest = str(Path(output_dir) / "patches" / filename)
        if not save_image(patch.image, dest):
            logger.warning("Could not save patch to %s", dest)

    def process_image(self, image_path: str) -> ScanResult:
        """Run the full pipeline on a single image file.

        Args:
            image_path: Path to the input image.

        Returns:
            :class:`~mtg_scanner.models.scan_result.ScanResult` with all
            detected and recognised cards.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("Processing image: %s", image_path)

        # --- Detection ---
        try:
            patches = self._detector.detect(image_path)
        except Exception as exc:
            logger.error("Detection failed for %s: %s", image_path, exc)
            patches = []

        # Load image for dataset logging and archiving
        img = None
        if self._dataset_logger is not None or self._archive is not None:
            try:
                img = load_image(image_path)
            except Exception as exc:
                logger.warning("Could not load image for logging: %s", exc)

        # --- Dataset: log scan ---
        scan_id: Optional[int] = None
        if self._dataset_logger is not None and img is not None:
            try:
                image_shape = img.shape if img is not None else (0, 0)
                scan_id = self._dataset_logger.log_scan(image_path, image_shape)
            except Exception as exc:
                logger.warning("DatasetLogger.log_scan failed: %s", exc)

        # --- Archive: store original image ---
        archive_id: Optional[int] = None
        if self._archive is not None and img is not None:
            try:
                archive_id, was_new = self._archive.store(
                    img, original_filename=image_path, scan_id=scan_id
                )
                if not was_new:
                    logger.debug("Archive: image already stored for %s", image_path)
            except Exception as exc:
                logger.warning("ImageArchive.store failed: %s", exc)

        recognized_cards: list[RecognizedCard] = []

        for patch in patches:
            if self._save_patches:
                self._save_patch(patch, self._output_dir)

            # --- Dataset: log detection ---
            detection_id: Optional[int] = None
            if self._dataset_logger is not None and scan_id is not None:
                try:
                    detection_id = self._dataset_logger.log_detection(scan_id, patch)
                except Exception as exc:
                    logger.warning("DatasetLogger.log_detection failed: %s", exc)

            # --- Recognition ---
            card_name, confidence, method = self._recognize_with_fallback(patch)

            # --- Dataset: log recognition attempt ---
            if self._dataset_logger is not None and detection_id is not None:
                try:
                    self._dataset_logger.log_recognition_attempt(
                        detection_id=detection_id,
                        method=method,
                        raw_text=card_name or "",
                        cleaned_text=card_name or "",
                        matched_name=card_name,
                        confidence=confidence,
                    )
                except Exception as exc:
                    logger.warning("DatasetLogger.log_recognition_attempt failed: %s", exc)

            # --- Scryfall lookup ---
            card_data = None
            if card_name:
                try:
                    card_data = self._scryfall.lookup(card_name)
                except Exception as exc:
                    logger.warning("Scryfall lookup failed for %r: %s", card_name, exc)

            recognized_card = RecognizedCard(
                patch=patch,
                card_name=card_name,
                recognition_confidence=confidence,
                recognition_method=method,
                card_data=card_data,
            )
            recognized_cards.append(recognized_card)

            # --- Dataset: log result ---
            if self._dataset_logger is not None and detection_id is not None:
                try:
                    self._dataset_logger.log_result(detection_id, recognized_card)
                except Exception as exc:
                    logger.warning("DatasetLogger.log_result failed: %s", exc)

        total_recognized = sum(1 for c in recognized_cards if c.card_name is not None)
        total_unknown = len(recognized_cards) - total_recognized

        result = ScanResult(
            image_path=image_path,
            scan_timestamp=timestamp,
            cards=recognized_cards,
            total_detected=len(patches),
            total_recognized=total_recognized,
            total_unknown=total_unknown,
        )

        # --- Dataset: finish scan ---
        if self._dataset_logger is not None and scan_id is not None:
            try:
                self._dataset_logger.finish_scan(scan_id, result)
            except Exception as exc:
                logger.warning("DatasetLogger.finish_scan failed: %s", exc)

        logger.info(
            "Finished %s: detected=%d recognised=%d unknown=%d",
            image_path,
            result.total_detected,
            result.total_recognized,
            result.total_unknown,
        )
        return result

    def process_directory(
        self,
        directory: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[ScanResult]:
        """Process all images found in *directory*.

        Args:
            directory: Path to a directory containing image files.
            progress_callback: Optional callable invoked as
                ``callback(current_index, total, image_path)`` after each
                image is processed.

        Returns:
            List of :class:`~mtg_scanner.models.scan_result.ScanResult`
            objects, one per image.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.error("Not a directory: %s", directory)
            return []

        image_paths = [
            str(p)
            for p in sorted(dir_path.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ]

        if not image_paths:
            logger.warning("No images found in %s", directory)
            return []

        results: list[ScanResult] = []
        total = len(image_paths)
        logger.info("Processing %d image(s) in %s", total, directory)

        for idx, image_path in enumerate(image_paths):
            result = self.process_image(image_path)
            results.append(result)
            if progress_callback:
                try:
                    progress_callback(idx + 1, total, image_path)
                except Exception as exc:
                    logger.warning("Progress callback raised: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: list[ScanResult],
        output_dir: Optional[str] = None,
        fmt: str = "both",
    ) -> list[str]:
        """Write scan results to disk in the requested format.

        Args:
            results: List of scan results to serialise.
            output_dir: Destination directory.  Defaults to the configured
                output directory.
            fmt: One of ``'csv'``, ``'json'``, or ``'both'``.

        Returns:
            List of paths to the files that were written.
        """
        out = Path(output_dir or self._output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        for result in results:
            stem = Path(result.image_path).stem

            if fmt in ("json", "both"):
                json_path = out / f"{stem}_scan.json"
                try:
                    with open(json_path, "w", encoding="utf-8") as fh:
                        json.dump(result.to_json(), fh, indent=2, ensure_ascii=False)
                    written.append(str(json_path))
                except Exception as exc:
                    logger.error("Failed to write JSON for %s: %s", stem, exc)

            if fmt in ("csv", "both"):
                csv_path = out / f"{stem}_scan.csv"
                rows = result.to_csv_rows()
                if rows:
                    try:
                        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                            writer.writeheader()
                            writer.writerows(rows)
                        written.append(str(csv_path))
                    except Exception as exc:
                        logger.error("Failed to write CSV for %s: %s", stem, exc)

        return written
