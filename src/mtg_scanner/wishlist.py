"""Wish-list and trade-list manager for MTG cards.

Want-list  — cards the user is looking for.
Have-list  — cards the user has available for trade (duplicates).

Both live in ``data/wishlist.db``.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS want_list (
    id INTEGER PRIMARY KEY,
    scryfall_id TEXT,
    oracle_id TEXT,
    name TEXT NOT NULL,
    set_code TEXT,
    foil INTEGER DEFAULT 0,
    condition TEXT DEFAULT 'NM',
    quantity_wanted INTEGER DEFAULT 1,
    max_price_eur REAL,
    priority INTEGER DEFAULT 2,
    notes TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS have_list (
    id INTEGER PRIMARY KEY,
    scryfall_id TEXT,
    oracle_id TEXT,
    name TEXT NOT NULL,
    set_code TEXT,
    set_name TEXT,
    collector_number TEXT,
    foil INTEGER DEFAULT 0,
    condition TEXT DEFAULT 'NM',
    quantity INTEGER DEFAULT 1,
    ask_price_eur REAL,
    notes TEXT,
    added_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_want_name ON want_list(name);
CREATE INDEX IF NOT EXISTS idx_have_name ON have_list(name);
"""

# Priority labels
PRIORITIES = {1: "Hoch", 2: "Mittel", 3: "Niedrig"}


class WishlistManager:
    """Manages want-list and have-list (trade-list) in a SQLite database.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "data/wishlist.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Want-list
    # ------------------------------------------------------------------

    def add_want(
        self,
        name: str,
        scryfall_id: str = "",
        oracle_id: str = "",
        set_code: str = "",
        foil: bool = False,
        condition: str = "NM",
        quantity: int = 1,
        max_price_eur: Optional[float] = None,
        priority: int = 2,
        notes: str = "",
    ) -> int:
        """Add a card to the want-list.

        If an entry with the same name + foil + condition already exists the
        quantity is increased instead of adding a duplicate row.

        Returns:
            Entry id.
        """
        existing = self._conn.execute(
            "SELECT id, quantity_wanted FROM want_list "
            "WHERE name = ? AND foil = ? AND condition = ?",
            (name, int(foil), condition.upper()),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE want_list SET quantity_wanted = ? WHERE id = ?",
                (existing["quantity_wanted"] + quantity, existing["id"]),
            )
            self._conn.commit()
            return existing["id"]

        cur = self._conn.execute(
            """INSERT INTO want_list
               (scryfall_id, oracle_id, name, set_code, foil, condition,
                quantity_wanted, max_price_eur, priority, notes, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (scryfall_id, oracle_id, name, set_code, int(foil),
             condition.upper(), quantity, max_price_eur, priority, notes, self._now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def remove_want(self, entry_id: int) -> bool:
        """Remove a want-list entry."""
        self._conn.execute("DELETE FROM want_list WHERE id = ?", (entry_id,))
        self._conn.commit()
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    def get_want_list(self, name_filter: str = "") -> list[dict]:
        """Return all want-list entries, sorted by priority then name."""
        if name_filter:
            rows = self._conn.execute(
                "SELECT * FROM want_list WHERE name LIKE ? ORDER BY priority, name",
                (f"%{name_filter}%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM want_list ORDER BY priority, name"
            ).fetchall()
        return [dict(r) for r in rows]

    def export_want_list(self, output_path: str) -> int:
        """Export the want-list as CSV."""
        rows = self.get_want_list()
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Priority", "Quantity", "Name", "Set", "Condition",
                             "Foil", "Max EUR", "Notes"])
            for r in rows:
                writer.writerow([
                    PRIORITIES.get(r["priority"], str(r["priority"])),
                    r["quantity_wanted"],
                    r["name"],
                    (r["set_code"] or "").upper(),
                    r["condition"],
                    "Yes" if r["foil"] else "No",
                    r["max_price_eur"] or "",
                    r["notes"] or "",
                ])
        return len(rows)

    # ------------------------------------------------------------------
    # Have-list (trade list)
    # ------------------------------------------------------------------

    def add_have(
        self,
        name: str,
        scryfall_id: str = "",
        oracle_id: str = "",
        set_code: str = "",
        set_name: str = "",
        collector_number: str = "",
        foil: bool = False,
        condition: str = "NM",
        quantity: int = 1,
        ask_price_eur: Optional[float] = None,
        notes: str = "",
    ) -> int:
        """Add a card to the have (trade) list. Merges duplicates."""
        existing = self._conn.execute(
            "SELECT id, quantity FROM have_list "
            "WHERE scryfall_id = ? AND foil = ? AND condition = ?",
            (scryfall_id or "", int(foil), condition.upper()),
        ).fetchone()
        if existing and scryfall_id:
            self._conn.execute(
                "UPDATE have_list SET quantity = ? WHERE id = ?",
                (existing["quantity"] + quantity, existing["id"]),
            )
            self._conn.commit()
            return existing["id"]

        cur = self._conn.execute(
            """INSERT INTO have_list
               (scryfall_id, oracle_id, name, set_code, set_name, collector_number,
                foil, condition, quantity, ask_price_eur, notes, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (scryfall_id, oracle_id, name, set_code, set_name, collector_number,
             int(foil), condition.upper(), quantity, ask_price_eur, notes, self._now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def remove_have(self, entry_id: int) -> bool:
        """Remove a have-list entry."""
        self._conn.execute("DELETE FROM have_list WHERE id = ?", (entry_id,))
        self._conn.commit()
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    def get_have_list(self, name_filter: str = "") -> list[dict]:
        """Return all have-list (trade) entries."""
        if name_filter:
            rows = self._conn.execute(
                "SELECT * FROM have_list WHERE name LIKE ? ORDER BY name",
                (f"%{name_filter}%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM have_list ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def export_have_list(self, output_path: str) -> int:
        """Export the have-list as CSV."""
        rows = self.get_have_list()
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Quantity", "Name", "Set", "Collector#", "Condition",
                             "Foil", "Ask EUR", "Notes"])
            for r in rows:
                writer.writerow([
                    r["quantity"],
                    r["name"],
                    (r["set_code"] or "").upper(),
                    r["collector_number"] or "",
                    r["condition"],
                    "Yes" if r["foil"] else "No",
                    r["ask_price_eur"] or "",
                    r["notes"] or "",
                ])
        return len(rows)

    # ------------------------------------------------------------------
    # Cross-list analysis
    # ------------------------------------------------------------------

    def compare_want_vs_collection(self, collection_manager) -> list[dict]:
        """Find want-list cards that are already in the collection.

        Args:
            collection_manager: An open :class:`~mtg_scanner.collection.CollectionManager`.

        Returns:
            List of dicts: {want_id, name, quantity_wanted, quantity_owned,
            still_missing}.
        """
        wants = self.get_want_list()
        results = []
        for want in wants:
            owned = collection_manager.get_collection(
                name_filter=want["name"], limit=10
            )
            qty_owned = sum(e["quantity"] for e in owned
                           if e["name"].lower() == want["name"].lower())
            still_missing = max(0, want["quantity_wanted"] - qty_owned)
            results.append({
                "want_id": want["id"],
                "name": want["name"],
                "quantity_wanted": want["quantity_wanted"],
                "quantity_owned": qty_owned,
                "still_missing": still_missing,
                "priority": PRIORITIES.get(want["priority"], "?"),
                "foil": bool(want["foil"]),
                "condition": want["condition"],
            })
        return results

    def stats(self) -> dict:
        """Return statistics for both lists."""
        want_count = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(quantity_wanted), 0) FROM want_list"
        ).fetchone()
        have_count = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(quantity), 0) FROM have_list"
        ).fetchone()
        return {
            "want_entries": want_count[0],
            "want_cards": want_count[1],
            "have_entries": have_count[0],
            "have_cards": have_count[1],
            "db_path": str(self._db_path),
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
