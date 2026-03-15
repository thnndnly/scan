"""SQLite-based dataset logger for MTG card scan history.

Records every scan, detection, recognition attempt and result to a local
SQLite database so that the data can be reviewed, corrected, and exported
as training data later.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mtg_scanner.models.card_patch import CardPatch
    from mtg_scanner.models.recognized_card import RecognizedCard
    from mtg_scanner.models.scan_result import ScanResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY,
    image_path TEXT NOT NULL,
    scan_timestamp TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    total_detected INTEGER DEFAULT 0,
    total_recognised INTEGER DEFAULT 0,
    total_unknown INTEGER DEFAULT 0,
    total_value_eur REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    scan_id INTEGER NOT NULL REFERENCES scans(id),
    patch_index INTEGER NOT NULL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    detection_confidence REAL,
    patch_image_path TEXT
);

CREATE TABLE IF NOT EXISTS recognition_attempts (
    id INTEGER PRIMARY KEY,
    detection_id INTEGER NOT NULL REFERENCES detections(id),
    method TEXT NOT NULL,
    raw_text TEXT,
    cleaned_text TEXT,
    matched_name TEXT,
    confidence REAL,
    attempted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    detection_id INTEGER NOT NULL REFERENCES detections(id),
    card_name TEXT,
    en_name TEXT,
    recognition_method TEXT,
    confidence REAL,
    price_eur REAL,
    price_usd REAL,
    set_code TEXT,
    scryfall_uri TEXT,
    manually_corrected INTEGER DEFAULT 0,
    corrected_name TEXT,
    scryfall_id TEXT,
    oracle_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_scan_id ON detections(scan_id);
CREATE INDEX IF NOT EXISTS idx_attempts_detection_id ON recognition_attempts(detection_id);
CREATE INDEX IF NOT EXISTS idx_results_detection_id ON results(detection_id);
"""


class DatasetLogger:
    """Records scan events to a SQLite database for later analysis and training.

    Args:
        db_path: Path to the SQLite database file.  Created if it does not exist.
        save_patches: When ``True``, patch images are saved as PNG files under
            ``data/patches/YYYY-MM-DD/``.
    """

    def __init__(self, db_path: str, save_patches: bool = True) -> None:
        self._db_path = Path(db_path)
        self._save_patches = save_patches
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._migrate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Add new columns to existing databases if they don't exist yet."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(results)").fetchall()}
        for col, typedef in [("scryfall_id", "TEXT"), ("oracle_id", "TEXT")]:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE results ADD COLUMN {col} {typedef}")
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _patch_dir(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return Path("data") / "patches" / date_str

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_scan(self, image_path: str, image_shape: tuple) -> int:
        """Insert a scan record and return its ``scan_id``.

        Args:
            image_path: Path to the source image.
            image_shape: ``(height, width[, channels])`` tuple from NumPy / OpenCV.

        Returns:
            The row-id of the newly created scan record.
        """
        height = image_shape[0] if len(image_shape) >= 1 else None
        width = image_shape[1] if len(image_shape) >= 2 else None
        cur = self._conn.execute(
            "INSERT INTO scans (image_path, scan_timestamp, image_width, image_height) "
            "VALUES (?, ?, ?, ?)",
            (image_path, self._now(), width, height),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def log_detection(self, scan_id: int, patch: "CardPatch") -> int:
        """Insert a detection record, optionally save patch PNG, return ``detection_id``.

        Args:
            scan_id: Parent scan id returned by :meth:`log_scan`.
            patch: The detected card patch.

        Returns:
            The row-id of the newly created detection record.
        """
        patch_image_path: Optional[str] = None
        if self._save_patches:
            patch_image_path = self._save_patch_image(scan_id, patch)

        x, y, w, h = patch.bbox
        cur = self._conn.execute(
            "INSERT INTO detections "
            "(scan_id, patch_index, bbox_x, bbox_y, bbox_w, bbox_h, "
            "detection_confidence, patch_image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (scan_id, patch.patch_index, x, y, w, h,
             patch.detection_confidence, patch_image_path),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def _save_patch_image(self, scan_id: int, patch: "CardPatch") -> Optional[str]:
        """Save a patch image to disk and return its relative path."""
        try:
            import cv2  # type: ignore
            patch_dir = self._patch_dir()
            patch_dir.mkdir(parents=True, exist_ok=True)
            filename = f"scan_{scan_id}_patch_{patch.patch_index}.png"
            full_path = patch_dir / filename
            cv2.imwrite(str(full_path), patch.image)
            # Return relative path
            return str(full_path)
        except Exception as exc:
            logger.warning("Could not save patch image: %s", exc)
            return None

    def log_recognition_attempt(
        self,
        detection_id: int,
        method: str,
        raw_text: str,
        cleaned_text: str,
        matched_name: Optional[str],
        confidence: float,
    ) -> None:
        """Record one recognition attempt (OCR, hash, LLM, etc.).

        Args:
            detection_id: Parent detection id.
            method: One of ``'ocr_latin'``, ``'ocr_cjk'``, ``'hash'``, ``'llm'``.
            raw_text: Raw text / hash string from the recogniser.
            cleaned_text: Text after normalisation / cleaning.
            matched_name: The card name that was matched, or ``None``.
            confidence: Confidence score 0–1.
        """
        self._conn.execute(
            "INSERT INTO recognition_attempts "
            "(detection_id, method, raw_text, cleaned_text, matched_name, confidence, attempted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (detection_id, method, raw_text, cleaned_text,
             matched_name, confidence, self._now()),
        )
        self._conn.commit()

    def log_result(self, detection_id: int, card: "RecognizedCard") -> None:
        """Upsert the final recognition result for a detection.

        Args:
            detection_id: Parent detection id.
            card: The :class:`~mtg_scanner.models.recognized_card.RecognizedCard` result.
        """
        cd = card.card_data
        self._conn.execute(
            "INSERT OR REPLACE INTO results "
            "(detection_id, card_name, en_name, recognition_method, confidence, "
            "price_eur, price_usd, set_code, scryfall_uri) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                detection_id,
                card.card_name,
                cd.name if cd else card.card_name,
                card.recognition_method,
                card.recognition_confidence,
                cd.price_eur if cd else None,
                cd.price_usd if cd else None,
                cd.set_code if cd else None,
                cd.scryfall_uri if cd else None,
            ),
        )
        self._conn.commit()

    def finish_scan(self, scan_id: int, result: "ScanResult") -> None:
        """Update scan-level aggregate counts after all detections are logged.

        Args:
            scan_id: The scan id returned by :meth:`log_scan`.
            result: Completed :class:`~mtg_scanner.models.scan_result.ScanResult`.
        """
        total_value = sum(
            c.card_data.price_eur
            for c in result.cards
            if c.card_data and c.card_data.price_eur is not None
        )
        self._conn.execute(
            "UPDATE scans SET total_detected=?, total_recognised=?, "
            "total_unknown=?, total_value_eur=? WHERE id=?",
            (result.total_detected, result.total_recognized,
             result.total_unknown, total_value, scan_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_scan_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent scans as a list of dicts.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of dicts with scan record fields.
        """
        rows = self._conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_scan_detail(self, scan_id: int) -> dict:
        """Return full detail for one scan including detections and results.

        Args:
            scan_id: The scan id to retrieve.

        Returns:
            Dict with ``scan`` key (scan record) and ``detections`` key (list
            of detection dicts each containing nested ``result``).
        """
        scan_row = self._conn.execute(
            "SELECT * FROM scans WHERE id=?", (scan_id,)
        ).fetchone()
        if scan_row is None:
            return {}
        scan = dict(scan_row)

        det_rows = self._conn.execute(
            "SELECT * FROM detections WHERE scan_id=? ORDER BY patch_index",
            (scan_id,),
        ).fetchall()

        detections = []
        for det in det_rows:
            d = dict(det)
            result_row = self._conn.execute(
                "SELECT * FROM results WHERE detection_id=?", (det["id"],)
            ).fetchone()
            d["result"] = dict(result_row) if result_row else None
            detections.append(d)

        scan["detections"] = detections
        return scan

    def get_low_confidence_patches(self, threshold: float = 0.70) -> list[dict]:
        """Return detections whose recognition confidence is below *threshold*.

        Args:
            threshold: Confidence cutoff.  Rows with ``confidence < threshold``
                or with no result are returned.

        Returns:
            List of dicts with joined detection + result fields.
        """
        rows = self._conn.execute(
            """
            SELECT d.id AS detection_id, d.scan_id, d.patch_index,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                   d.detection_confidence, d.patch_image_path,
                   r.card_name, r.recognition_method, r.confidence,
                   r.manually_corrected, r.corrected_name,
                   s.image_path, s.scan_timestamp
            FROM detections d
            LEFT JOIN results r ON r.detection_id = d.id
            LEFT JOIN scans s ON s.id = d.scan_id
            WHERE r.confidence IS NULL OR r.confidence < ?
            ORDER BY r.confidence ASC NULLS FIRST
            """,
            (threshold,),
        ).fetchall()
        return [dict(r) for r in rows]

    def correct_card(self, detection_id: int, correct_name: str) -> None:
        """Store a manual correction for a detection result.

        Args:
            detection_id: The detection whose result should be corrected.
            correct_name: The verified correct card name.
        """
        self._conn.execute(
            "UPDATE results SET manually_corrected=1, corrected_name=? "
            "WHERE detection_id=?",
            (correct_name, detection_id),
        )
        # Insert a row if there is no result yet
        rows_changed = self._conn.execute(
            "SELECT changes()"
        ).fetchone()[0]
        if rows_changed == 0:
            self._conn.execute(
                "INSERT INTO results (detection_id, card_name, manually_corrected, corrected_name) "
                "VALUES (?, ?, 1, ?)",
                (detection_id, correct_name, correct_name),
            )
        self._conn.commit()

    def export_csv(self, output_path: str) -> int:
        """Export all results to a CSV file.

        Args:
            output_path: Destination file path.

        Returns:
            Number of rows written.
        """
        rows = self._conn.execute(
            """
            SELECT s.image_path, s.scan_timestamp,
                   d.patch_index, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                   d.detection_confidence, d.patch_image_path,
                   r.card_name, r.en_name, r.recognition_method, r.confidence,
                   r.price_eur, r.price_usd, r.set_code, r.scryfall_uri,
                   r.manually_corrected, r.corrected_name
            FROM detections d
            JOIN scans s ON s.id = d.scan_id
            LEFT JOIN results r ON r.detection_id = d.id
            ORDER BY s.id, d.patch_index
            """
        ).fetchall()

        if not rows:
            return 0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image_path", "scan_timestamp", "patch_index",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "detection_confidence", "patch_image_path",
            "card_name", "en_name", "recognition_method", "confidence",
            "price_eur", "price_usd", "set_code", "scryfall_uri",
            "manually_corrected", "corrected_name",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)

    def stats(self) -> dict:
        """Return aggregate statistics about the dataset.

        Returns:
            Dict with keys: ``total_scans``, ``total_detections``,
            ``total_recognised``, ``total_unknown``, ``total_corrections``,
            ``total_value_eur``, ``db_path``.
        """
        total_scans = self._conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        total_detections = self._conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        total_recognised = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE card_name IS NOT NULL"
        ).fetchone()[0]
        total_corrections = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE manually_corrected=1"
        ).fetchone()[0]
        total_value = self._conn.execute(
            "SELECT COALESCE(SUM(total_value_eur), 0.0) FROM scans"
        ).fetchone()[0]

        return {
            "total_scans": total_scans,
            "total_detections": total_detections,
            "total_recognised": total_recognised,
            "total_unknown": total_detections - total_recognised,
            "total_corrections": total_corrections,
            "total_value_eur": total_value,
            "db_path": str(self._db_path),
        }

    def assign_card(self, detection_id: int, scryfall_id: str, oracle_id: str) -> None:
        """Assign a specific card printing (scryfall_id) to a detection result."""
        # First try UPDATE on existing row
        self._conn.execute(
            """UPDATE results SET scryfall_id = ?, oracle_id = ?, manually_corrected = 1
               WHERE detection_id = ?""",
            (scryfall_id, oracle_id, detection_id),
        )
        # If no row existed, insert a minimal one
        if self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE detection_id = ?", (detection_id,)
        ).fetchone()[0] == 0:
            self._conn.execute(
                """INSERT INTO results (detection_id, scryfall_id, oracle_id, manually_corrected)
                   VALUES (?, ?, ?, 1)""",
                (detection_id, scryfall_id, oracle_id),
            )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
