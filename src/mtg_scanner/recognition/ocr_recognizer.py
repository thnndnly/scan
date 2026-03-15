"""OCR-based card recogniser using EasyOCR."""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from mtg_scanner.config import get_config
from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.recognition.base import BaseRecognizer
from mtg_scanner.utils.fuzzy_search import best_match, best_match_multilingual, load_card_names
from mtg_scanner.utils.image_utils import crop_region

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Character cleaning helpers
# ---------------------------------------------------------------------------

# Common OCR substitution errors on MTG card title regions
_OCR_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b0\b"), "O"),    # isolated zero → capital O
    (re.compile(r"(?<=[A-Z])0"), "O"),  # zero after uppercase → O
    (re.compile(r"\bl\b"), "I"),    # isolated lowercase L → I
    (re.compile(r"\b1\b"), "I"),    # isolated 1 → I (in title context)
]


def _clean_ocr_text(text: str) -> str:
    """Apply heuristic fixes for common OCR errors in card title regions.

    Args:
        text: Raw concatenated OCR output.

    Returns:
        Cleaned string with common substitution errors corrected.
    """
    # Strip leading/trailing whitespace and normalise interior spaces
    text = " ".join(text.split())

    for pattern, replacement in _OCR_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)

    return text


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------


class OCRRecognizer(BaseRecognizer):
    """Card name recogniser that reads the title strip with EasyOCR.

    The EasyOCR ``Reader`` is initialised once when the first call to
    :meth:`recognize` is made (lazy initialisation) and then reused for all
    subsequent patches.

    Args:
        languages: List of language codes for EasyOCR (default: ``['en']``).
        confidence_threshold: Minimum OCR + fuzzy confidence to accept a
            result.  Below this value ``(None, score)`` is returned.
        names_file: Path to the card-names JSON file used for fuzzy matching.
    """

    # Mapping files searched in order after the English list fails
    _MAPPING_PATHS = [
        "data/card_names_de.json",
        "data/card_names_ja.json",
    ]

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        confidence_threshold: Optional[float] = None,
        names_file: str = "data/card_names.json",
    ) -> None:
        cfg = get_config().recognition
        self._languages = languages or cfg.ocr_languages          # e.g. ['en', 'de']
        self._languages_cjk = cfg.ocr_languages_cjk              # e.g. ['ja']
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.ocr_confidence_threshold
        )
        self._names_file = names_file
        self._reader = None      # primary (Latin) reader — lazy
        self._reader_cjk = None  # CJK reader — lazy, only when needed
        self._card_names: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reader(self):
        """Lazily initialise the primary (Latin) EasyOCR reader."""
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'easyocr' package is required for OCR recognition.\n"
                "Install it with:  pip install easyocr"
            ) from exc
        logger.info("Initialising EasyOCR reader for languages: %s", self._languages)
        self._reader = easyocr.Reader(self._languages, verbose=False)
        return self._reader

    def _get_reader_cjk(self):
        """Lazily initialise the CJK (e.g. Japanese) EasyOCR reader."""
        if self._reader_cjk is not None:
            return self._reader_cjk
        if not self._languages_cjk:
            return None
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'easyocr' package is required for OCR recognition.\n"
                "Install it with:  pip install easyocr"
            ) from exc
        logger.info("Initialising EasyOCR CJK reader for languages: %s", self._languages_cjk)
        self._reader_cjk = easyocr.Reader(self._languages_cjk, verbose=False)
        return self._reader_cjk

    def _get_card_names(self) -> list[str]:
        """Return the list of known card names, loading it once from disk."""
        if not self._card_names:
            self._card_names = load_card_names(self._names_file)
        return self._card_names

    # Minimum height of the title region in pixels before upscaling kicks in
    _MIN_TITLE_HEIGHT_PX = 80

    def _extract_title_region(self, patch_image: np.ndarray) -> np.ndarray:
        """Crop the title bar from the top of the card (top 13 % of height).

        If the resulting strip is very small (common when many cards are
        photographed at once), upscale it so EasyOCR has enough pixels to work
        with.

        Args:
            patch_image: BGR card image.

        Returns:
            Cropped (and possibly upscaled) title region.
        """
        region = crop_region(patch_image, y_start_frac=0.0, y_end_frac=0.13)
        h = region.shape[0]
        if h < self._MIN_TITLE_HEIGHT_PX and h > 0:
            scale = self._MIN_TITLE_HEIGHT_PX / h
            new_w = max(1, int(region.shape[1] * scale))
            region = cv2.resize(region, (new_w, self._MIN_TITLE_HEIGHT_PX),
                                interpolation=cv2.INTER_CUBIC)
        return region

    def _run_ocr(self, region: np.ndarray, reader=None) -> str:
        """Run EasyOCR on *region* and return aggregated text.

        Args:
            region: BGR image of the title area.
            reader: EasyOCR reader instance to use; defaults to primary reader.

        Returns:
            Concatenated text string from all detected text boxes.
        """
        if reader is None:
            reader = self._get_reader()
        try:
            results = reader.readtext(region, detail=1)
        except Exception as exc:
            logger.warning("EasyOCR readtext failed: %s", exc)
            return ""
        texts = [text for (_, text, _conf) in results if text]
        return " ".join(texts)

    # ------------------------------------------------------------------
    # BaseRecognizer interface
    # ------------------------------------------------------------------

    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Recognise the card in *patch* via OCR on its title region.

        Args:
            patch: Detected card patch.

        Returns:
            ``(card_name, confidence)`` or ``(None, 0.0)`` on failure.
        """
        try:
            title_region = self._extract_title_region(patch.image)
        except Exception as exc:
            logger.warning("Failed to extract title region: %s", exc)
            return None, 0.0

        if title_region.size == 0:
            return None, 0.0

        try:
            raw_text = self._run_ocr(title_region)
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return None, 0.0

        if not raw_text.strip():
            # Primary OCR produced nothing — try CJK reader on title strip
            cjk_reader = self._get_reader_cjk()
            if cjk_reader:
                try:
                    raw_text = self._run_ocr(title_region, reader=cjk_reader)
                except Exception as exc:
                    logger.warning("CJK OCR on title failed: %s", exc)
            # Still nothing — try CJK on full patch upscaled to at least 400px height
            if not raw_text.strip() and cjk_reader:
                try:
                    ph = patch.image.shape[0]
                    if ph < 400:
                        scale = 400 / ph
                        pw = max(1, int(patch.image.shape[1] * scale))
                        full_up = cv2.resize(patch.image, (pw, 400), interpolation=cv2.INTER_CUBIC)
                    else:
                        full_up = patch.image
                    raw_text = self._run_ocr(full_up, reader=cjk_reader)
                except Exception as exc:
                    logger.warning("CJK OCR on full patch failed: %s", exc)
            if not raw_text.strip():
                logger.debug("OCR produced no text for patch %d", patch.patch_index)
                return None, 0.0

        cleaned = _clean_ocr_text(raw_text)
        logger.debug("OCR raw=%r  cleaned=%r", raw_text, cleaned)

        names = self._get_card_names()
        cutoff = self._confidence_threshold * 100.0

        # Multilingual match: EN/DE names first, then localized mappings
        card_name, confidence = best_match_multilingual(
            cleaned, names, self._MAPPING_PATHS, score_cutoff=cutoff
        )

        if card_name is None or confidence < self._confidence_threshold:
            # Last resort: try CJK OCR (title strip, then full patch upscaled)
            cjk_reader = self._get_reader_cjk()
            if cjk_reader:
                cjk_text = ""
                # Try title strip first
                try:
                    cjk_text = self._run_ocr(title_region, reader=cjk_reader)
                except Exception as exc:
                    logger.warning("CJK OCR on title failed: %s", exc)
                # If title strip yields nothing, try full patch upscaled
                if not cjk_text.strip():
                    try:
                        ph = patch.image.shape[0]
                        if ph < 400:
                            scale = 400 / ph
                            pw = max(1, int(patch.image.shape[1] * scale))
                            full_up = cv2.resize(patch.image, (pw, 400),
                                                 interpolation=cv2.INTER_CUBIC)
                        else:
                            full_up = patch.image
                        cjk_text = self._run_ocr(full_up, reader=cjk_reader)
                    except Exception as exc:
                        logger.warning("CJK OCR on full patch failed: %s", exc)
                if cjk_text.strip():
                    cjk_cleaned = _clean_ocr_text(cjk_text)
                    logger.debug("CJK OCR raw=%r  cleaned=%r", cjk_text, cjk_cleaned)
                    card_name, confidence = best_match_multilingual(
                        cjk_cleaned, names, self._MAPPING_PATHS, score_cutoff=cutoff
                    )

        if card_name is None or confidence < self._confidence_threshold:
            logger.debug(
                "OCR: no match above threshold for %r (best=%.2f)", cleaned, confidence
            )
            return None, confidence

        logger.info(
            "OCR: matched %r → %r (confidence=%.2f)", cleaned, card_name, confidence
        )
        return card_name, confidence
