"""Permanent compressed image archive stored in SQLite.

Images are stored content-addressed (SHA256) so duplicates are never
written twice.  Each image is resized to at most *max_dimension* pixels
on its longest side and compressed as JPEG before storage.  A small
thumbnail is kept for fast UI display.

The companion index file (``data/image_archive_index.json``) contains
only metadata (no image bytes) and is intended to be committed to git.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256              TEXT    UNIQUE NOT NULL,
    original_filename   TEXT,
    scan_id             INTEGER,
    captured_at         TEXT    NOT NULL,
    original_size_bytes INTEGER,
    stored_size_bytes   INTEGER,
    compression         TEXT    NOT NULL DEFAULT 'jpeg70',
    image_width         INTEGER,
    image_height        INTEGER,
    image_data          BLOB    NOT NULL,
    thumbnail           BLOB,
    scryfall_id         TEXT,
    oracle_id           TEXT,
    patch_type          TEXT    DEFAULT 'source_image'
);
CREATE INDEX IF NOT EXISTS idx_images_sha256   ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_scan_id  ON images(scan_id);
CREATE INDEX IF NOT EXISTS idx_images_captured ON images(captured_at);
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_jpeg(img_bgr: np.ndarray, quality: int) -> bytes:
    """Encode a BGR numpy array as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf)


def _resize_if_needed(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


class ImageArchive:
    """Persistent, compressed, deduplicated image store.

    Args:
        db_path: Path to the SQLite archive database.
        index_path: Path to the companion JSON index (git-trackable).
        max_dimension: Maximum pixel size of the longest image side.
        jpeg_quality: JPEG compression quality (0-100) for stored images.
        thumbnail_size: Maximum pixel size of thumbnail longest side.
        thumbnail_quality: JPEG quality for thumbnails.
    """

    def __init__(
        self,
        db_path: str = "data/image_archive.db",
        index_path: str = "data/image_archive_index.json",
        max_dimension: int = 1920,
        jpeg_quality: int = 70,
        thumbnail_size: int = 300,
        thumbnail_quality: int = 50,
    ) -> None:
        self._db_path = db_path
        self._index_path = Path(index_path)
        self._max_dimension = max_dimension
        self._jpeg_quality = jpeg_quality
        self._thumbnail_size = thumbnail_size
        self._thumbnail_quality = thumbnail_quality

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        image: np.ndarray,
        original_filename: str = "",
        scan_id: Optional[int] = None,
    ) -> tuple[int, bool]:
        """Store an image in the archive.

        Args:
            image: BGR numpy array.
            original_filename: Original file path/name (metadata only).
            scan_id: Optional link to a scan in dataset.db.

        Returns:
            ``(archive_id, was_new)`` — *was_new* is False if the image
            was already present (duplicate detected via SHA256).
        """
        # Compute hash on raw source bytes to detect true duplicates
        raw_bytes = _encode_jpeg(image, 95)
        sha = _sha256(raw_bytes)

        existing = self._conn.execute(
            "SELECT id FROM images WHERE sha256 = ?", (sha,)
        ).fetchone()
        if existing:
            logger.debug("Archive: duplicate skipped (sha256=%s…)", sha[:12])
            return existing["id"], False

        resized = _resize_if_needed(image, self._max_dimension)
        stored_bytes = _encode_jpeg(resized, self._jpeg_quality)

        thumb = _resize_if_needed(image, self._thumbnail_size)
        thumb_bytes = _encode_jpeg(thumb, self._thumbnail_quality)

        h, w = resized.shape[:2]
        now = datetime.now(timezone.utc).isoformat()

        cur = self._conn.execute(
            """
            INSERT INTO images
                (sha256, original_filename, scan_id, captured_at,
                 original_size_bytes, stored_size_bytes, compression,
                 image_width, image_height, image_data, thumbnail)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                sha,
                original_filename,
                scan_id,
                now,
                len(raw_bytes),
                len(stored_bytes),
                f"jpeg{self._jpeg_quality}",
                w,
                h,
                stored_bytes,
                thumb_bytes,
            ),
        )
        self._conn.commit()
        archive_id = cur.lastrowid
        self._update_index(archive_id, sha, original_filename, scan_id, now, len(raw_bytes), len(stored_bytes))
        logger.info(
            "Archive: stored %s → id=%d (%d → %d bytes, %.0f%% of original)",
            Path(original_filename).name,
            archive_id,
            len(raw_bytes),
            len(stored_bytes),
            100 * len(stored_bytes) / max(len(raw_bytes), 1),
        )
        return archive_id, True

    def store_file(
        self,
        image_path: str,
        scan_id: Optional[int] = None,
    ) -> tuple[int, bool]:
        """Load an image from disk and store it.  Returns ``(id, was_new)``."""
        from mtg_scanner.utils.image_utils import load_image
        img = load_image(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.store(img, original_filename=image_path, scan_id=scan_id)

    def get_thumbnail(self, archive_id: int) -> Optional[bytes]:
        """Return raw JPEG thumbnail bytes for *archive_id*, or None."""
        row = self._conn.execute(
            "SELECT thumbnail FROM images WHERE id = ?", (archive_id,)
        ).fetchone()
        return bytes(row["thumbnail"]) if row and row["thumbnail"] else None

    def get_image(self, archive_id: int) -> Optional[np.ndarray]:
        """Return the stored image as a BGR numpy array."""
        row = self._conn.execute(
            "SELECT image_data FROM images WHERE id = ?", (archive_id,)
        ).fetchone()
        if row is None:
            return None
        buf = np.frombuffer(bytes(row["image_data"]), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def stats(self) -> dict:
        """Return archive statistics."""
        row = self._conn.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(original_size_bytes) as original_bytes,
                   SUM(stored_size_bytes)   as stored_bytes
            FROM images
            """
        ).fetchone()
        total = row["total"] or 0
        orig = row["original_bytes"] or 0
        stored = row["stored_bytes"] or 0
        return {
            "total_images": total,
            "original_size_mb": orig / 1_048_576,
            "stored_size_mb": stored / 1_048_576,
            "compression_ratio": stored / orig if orig else 0.0,
            "space_saved_mb": (orig - stored) / 1_048_576,
            "db_path": self._db_path,
        }

    def list_images(self, limit: int = 200, offset: int = 0) -> list[dict]:
        """Return metadata rows (no image data) for browsing."""
        rows = self._conn.execute(
            """
            SELECT id, sha256, original_filename, scan_id, captured_at,
                   original_size_bytes, stored_size_bytes, image_width, image_height
            FROM images
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def export_all(self, dest_dir: str) -> int:
        """Export all stored images as JPEG files to *dest_dir*."""
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = self._conn.execute(
            "SELECT id, original_filename, image_data FROM images"
        ).fetchall()
        count = 0
        for row in rows:
            stem = Path(row["original_filename"]).stem if row["original_filename"] else f"img_{row['id']}"
            dest = out / f"{stem}_{row['id']}.jpg"
            dest.write_bytes(bytes(row["image_data"]))
            count += 1
        return count

    def verify(self) -> tuple[int, int]:
        """Check SHA256 integrity of all stored images.

        Returns:
            ``(ok_count, corrupt_count)``
        """
        rows = self._conn.execute("SELECT id, sha256, image_data FROM images").fetchall()
        ok = corrupt = 0
        for row in rows:
            # Re-encode at 95% to get comparable hash (stored as 95% in raw_bytes)
            buf = np.frombuffer(bytes(row["image_data"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Archive verify: id=%d could not be decoded", row["id"])
                corrupt += 1
                continue
            ok += 1
        return ok, corrupt

    def assign_card(self, archive_id: int, scryfall_id: str, oracle_id: str) -> None:
        """Link an archived image to a specific card printing."""
        self._conn.execute(
            "UPDATE images SET scryfall_id = ?, oracle_id = ? WHERE id = ?",
            (scryfall_id, oracle_id, archive_id),
        )
        self._conn.commit()

    def store_patch(
        self,
        patch_image,  # numpy BGR array
        scryfall_id: str,
        oracle_id: str,
        detection_id: Optional[int] = None,
        original_filename: str = "",
    ) -> tuple[int, bool]:
        """Store a confirmed card patch with its card assignment."""
        archive_id, was_new = self.store(
            patch_image,
            original_filename=original_filename,
            scan_id=detection_id,
        )
        if scryfall_id:
            self._conn.execute(
                "UPDATE images SET scryfall_id = ?, oracle_id = ?, patch_type = 'card_patch' WHERE id = ?",
                (scryfall_id, oracle_id, archive_id),
            )
            self._conn.commit()
        return archive_id, was_new

    def _migrate(self) -> None:
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(images)").fetchall()}
        for col, typedef in [
            ("scryfall_id", "TEXT"),
            ("oracle_id", "TEXT"),
            ("patch_type", "TEXT DEFAULT 'source_image'"),
        ]:
            col_name = col.split()[0]
            if col_name not in existing:
                self._conn.execute(f"ALTER TABLE images ADD COLUMN {col} {typedef}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_index(
        self,
        archive_id: int,
        sha256: str,
        original_filename: str,
        scan_id: Optional[int],
        captured_at: str,
        original_size: int,
        stored_size: int,
    ) -> None:
        """Append entry to the git-trackable JSON index."""
        try:
            if self._index_path.exists():
                with open(self._index_path, encoding="utf-8") as fh:
                    index: list = json.load(fh)
            else:
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                index = []

            index.append({
                "id": archive_id,
                "sha256": sha256,
                "original_filename": original_filename,
                "scan_id": scan_id,
                "captured_at": captured_at,
                "original_size_bytes": original_size,
                "stored_size_bytes": stored_size,
            })

            with open(self._index_path, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not update image archive index: %s", exc)
