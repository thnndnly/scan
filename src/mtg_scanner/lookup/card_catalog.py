"""Card catalog: query interface over the local Scryfall bulk-data SQLite DB.

The catalog is built once by running ``scripts/build_card_catalog.py`` and
stored in ``data/card_catalog.db``.  It contains every distinct card printing
(~130,000 rows from ``default_cards``) with full metadata but no image bytes —
only image URLs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CardCatalog:
    """Read-only query interface over the local card catalog SQLite database.

    Args:
        db_path: Path to ``card_catalog.db``.
    """

    def __init__(self, db_path: str = "data/card_catalog.db") -> None:
        self._db_path = db_path
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"Card catalog not found at {db_path}. "
                "Run:  python scripts/build_card_catalog.py"
            )
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def search_by_name(self, name: str, lang: str = "en", limit: int = 50) -> list[dict]:
        """Return all printings whose name fuzzy-matches *name*.

        Searches both ``name`` (English) and ``printed_name`` (localized).
        Results are sorted by release date descending (newest first).
        """
        q = f"%{name}%"
        rows = self._conn.execute(
            """
            SELECT * FROM cards
            WHERE (name LIKE ? OR printed_name LIKE ?)
              AND (lang = ? OR lang = 'en')
            ORDER BY released_at DESC
            LIMIT ?
            """,
            (q, q, lang, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_printings(self, oracle_id: str) -> list[dict]:
        """Return all printings of a card identified by *oracle_id*.

        Sorted by release date ascending (oldest first) so the picker can
        show chronological order.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM cards
            WHERE oracle_id = ?
            ORDER BY released_at ASC, collector_number ASC
            """,
            (oracle_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_scryfall_id(self, scryfall_id: str) -> Optional[dict]:
        """Return the card with this exact Scryfall UUID, or None."""
        row = self._conn.execute(
            "SELECT * FROM cards WHERE id = ?", (scryfall_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_set_number(
        self, set_code: str, collector_number: str, lang: str = "en"
    ) -> Optional[dict]:
        """Look up a card by its set + collector number + language."""
        row = self._conn.execute(
            """
            SELECT * FROM cards
            WHERE set_code = ? AND collector_number = ? AND lang = ?
            """,
            (set_code.lower(), collector_number, lang),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_oracle_id(self, name: str) -> Optional[str]:
        """Return the oracle_id for the first exact name match, or None."""
        row = self._conn.execute(
            "SELECT oracle_id FROM cards WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        return row["oracle_id"] if row else None

    def get_sets(self, set_type: Optional[str] = None) -> list[dict]:
        """Return all sets, optionally filtered by set_type."""
        if set_type:
            rows = self._conn.execute(
                "SELECT * FROM sets WHERE set_type = ? ORDER BY released_at DESC",
                (set_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sets ORDER BY released_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Return catalog statistics."""
        card_count = self._conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        set_count = self._conn.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
        meta = {
            row["key"]: row["value"]
            for row in self._conn.execute("SELECT key, value FROM catalog_meta").fetchall()
        }
        return {
            "total_cards": card_count,
            "total_sets": set_count,
            "bulk_type": meta.get("bulk_type", "unknown"),
            "updated_at": meta.get("updated_at", "unknown"),
            "imported_at": meta.get("imported_at", "unknown"),
            "db_path": self._db_path,
        }

    def is_built(self) -> bool:
        """Return True if the catalog database exists and has data."""
        try:
            return self._conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] > 0
        except Exception:
            return False

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row) -> dict:
        if row is None:
            return {}
        d = dict(row)
        # Deserialize JSON blob columns
        for col in ("colors", "color_identity", "finishes", "frame_effects",
                    "promo_types", "prices", "image_uris", "keywords"):
            if d.get(col) and isinstance(d[col], str):
                try:
                    d[col] = json.loads(d[col])
                except Exception:
                    pass
        return d
