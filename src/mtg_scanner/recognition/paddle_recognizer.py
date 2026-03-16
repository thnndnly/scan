"""PaddleOCR-based MTG card recogniser (optional alternative to EasyOCR)."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from mtg_scanner.recognition.base import BaseRecognizer
from mtg_scanner.models.card_patch import CardPatch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_paddle_ocr = None  # module-level singleton


def _get_paddle_ocr():
    """Lazily initialise PaddleOCR; raises ImportError with helpful message if not installed."""
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    try:
        from paddleocr import PaddleOCR  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR is not installed. Install it with:\n"
            "  pip install 'mtg-card-scanner[paddle]'\n"
            "or:\n"
            "  pip install paddleocr paddlepaddle"
        ) from exc
    _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddle_ocr


class PaddleRecognizer(BaseRecognizer):
    """Card name recogniser using PaddleOCR on the title strip.

    PaddleOCR achieves approximately 5-10% better accuracy than EasyOCR on
    MTG card title strips, particularly for non-standard fonts and foil cards.
    Uses the same card name fuzzy-matching approach as OCRRecognizer.
    """

    def __init__(self, confidence_threshold: Optional[float] = None) -> None:
        from mtg_scanner.config import get_config
        cfg = get_config()
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.recognition.paddle_confidence_threshold
        )
        # Load card names for fuzzy matching
        self._card_names: list[str] = []
        self._load_card_names(cfg.claude.names_file)

    def _load_card_names(self, names_file: str) -> None:
        """Load card names from JSON for fuzzy matching."""
        import json
        from pathlib import Path
        path = Path(names_file)
        if not path.exists():
            logger.warning("Card names file not found: %s", names_file)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._card_names = data
            elif isinstance(data, dict):
                self._card_names = list(data.keys())
            logger.info("PaddleRecognizer loaded %d card names", len(self._card_names))
        except Exception as exc:
            logger.error("Failed to load card names: %s", exc)

    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Recognise the card name from the title strip using PaddleOCR.

        Args:
            patch: Detected card patch with image data.

        Returns:
            ``(card_name, confidence)`` tuple, or ``(None, 0.0)`` on failure.
        """
        import numpy as np
        try:
            ocr = _get_paddle_ocr()
        except ImportError as exc:
            logger.error("PaddleOCR unavailable: %s", exc)
            return None, 0.0

        image = patch.image
        if image is None or image.size == 0:
            return None, 0.0

        # Extract top 13% title strip (same as OCRRecognizer)
        h, w = image.shape[:2]
        title_h = max(1, int(h * 0.13))
        title_strip = image[:title_h, :]

        try:
            result = ocr.ocr(title_strip, cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR failed: %s", exc)
            return None, 0.0

        if not result or not result[0]:
            return None, 0.0

        # Collect all text fragments sorted by x-position
        fragments = []
        for line in result[0]:
            box, (text, score) = line
            x_pos = box[0][0]  # top-left x
            fragments.append((x_pos, text, score))

        fragments.sort(key=lambda t: t[0])
        raw_text = " ".join(t[1] for t in fragments).strip()
        avg_score = sum(t[2] for t in fragments) / len(fragments) if fragments else 0.0

        if not raw_text:
            return None, 0.0

        # Fuzzy match against card names
        if not self._card_names:
            return raw_text, float(avg_score)

        try:
            from rapidfuzz import process as rfprocess, fuzz
            best = rfprocess.extractOne(
                raw_text,
                self._card_names,
                scorer=fuzz.WRatio,
                score_cutoff=50,
            )
            if best is not None:
                matched_name, fuzzy_score, _ = best
                # Combine OCR confidence with fuzzy match score
                combined = (float(avg_score) * 0.5 + fuzzy_score / 100.0 * 0.5)
                if combined >= self._confidence_threshold:
                    logger.debug(
                        "PaddleOCR: %r → %r (ocr=%.2f, fuzzy=%.1f, combined=%.2f)",
                        raw_text, matched_name, avg_score, fuzzy_score, combined,
                    )
                    return matched_name, combined
        except Exception as exc:
            logger.warning("Fuzzy match failed: %s", exc)

        return None, 0.0
