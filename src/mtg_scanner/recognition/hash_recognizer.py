"""Perceptual-hash-based card recogniser backed by a SQLite database."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from mtg_scanner.config import get_config
from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.recognition.base import BaseRecognizer
from mtg_scanner.utils.image_utils import crop_region, image_to_pil

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Hash DB DDL (kept here for reference; built by scripts/build_hash_db.py)
# ------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS card_hashes (
    id          INTEGER PRIMARY KEY,
    card_name   TEXT NOT NULL,
    set_code    TEXT,
    hash_value  TEXT NOT NULL,
    image_uri   TEXT
);
CREATE INDEX IF NOT EXISTS idx_card_name ON card_hashes(card_name);
"""


def _compute_phash(image: np.ndarray, hash_size: int = 16) -> Optional[object]:
    """Compute a perceptual hash for *image*.

    Args:
        image: BGR card image (or artwork crop).
        hash_size: Hash size parameter for ``imagehash.phash``.

    Returns:
        ``imagehash.ImageHash`` object, or ``None`` on failure.
    """
    try:
        import imagehash  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'imagehash' package is required for hash recognition.\n"
            "Install it with:  pip install imagehash"
        ) from exc

    try:
        pil_img = image_to_pil(image)
        return imagehash.phash(pil_img, hash_size=hash_size)
    except Exception as exc:
        logger.warning("Failed to compute phash: %s", exc)
        return None


class HashRecognizer(BaseRecognizer):
    """Card recogniser that compares artwork perceptual hashes against a database.

    Args:
        db_path: Path to the SQLite database built by ``scripts/build_hash_db.py``.
        max_hamming_distance: Maximum Hamming distance to consider a match.
        top_k: Number of top candidates to retrieve from the database.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_hamming_distance: Optional[int] = None,
        top_k: int = 5,
    ) -> None:
        cfg = get_config()
        self._db_path = db_path or cfg.scryfall.cache_db_path.replace(
            "scryfall_cache.db", "card_hashes.db"
        )
        # Prefer explicit recognition config
        self._max_hamming = (
            max_hamming_distance
            if max_hamming_distance is not None
            else cfg.recognition.hash_max_hamming_distance
        )
        self._top_k = top_k
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> Optional[sqlite3.Connection]:
        """Return a sqlite3 connection, opening it lazily.

        Returns:
            Open :class:`sqlite3.Connection`, or ``None`` when the database
            file does not exist.
        """
        if self._conn is not None:
            return self._conn

        if not Path(self._db_path).exists():
            logger.warning(
                "Hash DB not found at %r.  Run scripts/build_hash_db.py first.", self._db_path
            )
            return None

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        return self._conn

    def _extract_artwork(self, image: np.ndarray) -> np.ndarray:
        """Crop the artwork region from a card (rows 20 %–65 %, 5 px side margins).

        Args:
            image: BGR card image.

        Returns:
            Cropped artwork region.
        """
        return crop_region(image, y_start_frac=0.20, y_end_frac=0.65, x_margin_px=5)

    def _query_candidates(self, conn: sqlite3.Connection) -> list[tuple[str, str, Optional[str]]]:
        """Fetch all hash rows from the database.

        Returns:
            List of ``(card_name, hash_value, set_code)`` tuples.
        """
        try:
            rows = conn.execute(
                "SELECT card_name, hash_value, set_code FROM card_hashes;"
            ).fetchall()
            return [(r["card_name"], r["hash_value"], r["set_code"]) for r in rows]
        except Exception as exc:
            logger.warning("Failed to query hash DB: %s", exc)
            return []

    # ------------------------------------------------------------------
    # BaseRecognizer interface
    # ------------------------------------------------------------------

    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Attempt to identify the card in *patch* by perceptual hash comparison.

        Extracts the artwork region, computes a pHash, and compares it against
        every entry in the hash database.  The top-K candidates by Hamming
        distance are considered; the best match is returned if it is within
        *max_hamming_distance*.

        Confidence is computed as::

            confidence = 1.0 - (distance / max_hamming_distance)

        Args:
            patch: Detected card patch.

        Returns:
            ``(card_name, confidence)`` or ``(None, 0.0)`` on failure.
        """
        try:
            import imagehash  # noqa: F401 – ensure the package is available
        except ImportError as exc:
            raise ImportError(
                "The 'imagehash' package is required for hash recognition.\n"
                "Install it with:  pip install imagehash"
            ) from exc

        conn = self._get_conn()
        if conn is None:
            return None, 0.0

        try:
            artwork = self._extract_artwork(patch.image)
        except Exception as exc:
            logger.warning("Artwork extraction failed: %s", exc)
            return None, 0.0

        if artwork.size == 0:
            return None, 0.0

        query_hash = _compute_phash(artwork)
        if query_hash is None:
            return None, 0.0

        candidates = self._query_candidates(conn)
        if not candidates:
            return None, 0.0

        best_name: Optional[str] = None
        best_dist = self._max_hamming + 1

        results: list[tuple[int, str]] = []

        for card_name, hash_str, _set_code in candidates:
            try:
                import imagehash  # type: ignore

                db_hash = imagehash.hex_to_hash(hash_str)
                dist = query_hash - db_hash
                results.append((dist, card_name))
            except Exception:
                continue

        if not results:
            return None, 0.0

        results.sort(key=lambda t: t[0])
        top_k = results[: self._top_k]

        best_dist, best_name = top_k[0]

        if best_dist > self._max_hamming:
            logger.debug(
                "Hash: best candidate %r has distance %d > max %d",
                best_name,
                best_dist,
                self._max_hamming,
            )
            return None, 0.0

        confidence = 1.0 - (best_dist / self._max_hamming)
        logger.info(
            "Hash: matched %r (hamming=%d, confidence=%.2f)", best_name, best_dist, confidence
        )
        return best_name, float(confidence)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
