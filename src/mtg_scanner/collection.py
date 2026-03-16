"""SQLite-based collection manager for MTG cards.

Tracks cards the user owns: quantity, condition, foil status, buy price, etc.
Supports export to Moxfield, TCGplayer, Cardmarket, MTG Arena, and generic CSV.
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
CREATE TABLE IF NOT EXISTS collection_entries (
    id INTEGER PRIMARY KEY,
    scryfall_id TEXT NOT NULL,
    oracle_id TEXT,
    name TEXT NOT NULL,
    set_code TEXT,
    set_name TEXT,
    collector_number TEXT,
    lang TEXT DEFAULT 'en',
    foil INTEGER DEFAULT 0,
    condition TEXT DEFAULT 'NM',
    quantity INTEGER DEFAULT 1,
    buy_price REAL,
    buy_date TEXT,
    notes TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY,
    scryfall_id TEXT NOT NULL,
    date TEXT NOT NULL,
    price_eur REAL,
    price_usd REAL,
    price_eur_foil REAL,
    source TEXT DEFAULT 'catalog',
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_collection_scryfall_id ON collection_entries(scryfall_id);
CREATE INDEX IF NOT EXISTS idx_collection_oracle_id ON collection_entries(oracle_id);
CREATE INDEX IF NOT EXISTS idx_collection_name ON collection_entries(name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_price_history_card_date
    ON price_history(scryfall_id, date);
"""

# Valid condition codes (NM = Near Mint, LP = Lightly Played, MP = Moderately Played,
# HP = Heavily Played, DMG = Damaged)
CONDITIONS = ("NM", "LP", "MP", "HP", "DMG")


class CollectionManager:
    """Manages the user's MTG card collection in a local SQLite database.

    Args:
        db_path: Path to the SQLite database file. Created if it does not exist.
    """

    def __init__(self, db_path: str = "data/collection.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_card(
        self,
        scryfall_id: str,
        name: str,
        oracle_id: str = "",
        set_code: str = "",
        set_name: str = "",
        collector_number: str = "",
        lang: str = "en",
        foil: bool = False,
        condition: str = "NM",
        quantity: int = 1,
        buy_price: Optional[float] = None,
        buy_date: Optional[str] = None,
        notes: str = "",
    ) -> int:
        """Add a card to the collection. Returns the new entry id.

        If an entry with the same scryfall_id + foil + condition already exists,
        the quantity is incremented instead of creating a new row.
        """
        condition = condition.upper()
        if condition not in CONDITIONS:
            condition = "NM"

        existing = self._conn.execute(
            "SELECT id, quantity FROM collection_entries "
            "WHERE scryfall_id = ? AND foil = ? AND condition = ?",
            (scryfall_id, int(foil), condition),
        ).fetchone()

        if existing:
            new_qty = existing["quantity"] + quantity
            self._conn.execute(
                "UPDATE collection_entries SET quantity = ? WHERE id = ?",
                (new_qty, existing["id"]),
            )
            self._conn.commit()
            return existing["id"]

        cur = self._conn.execute(
            """INSERT INTO collection_entries
               (scryfall_id, oracle_id, name, set_code, set_name, collector_number,
                lang, foil, condition, quantity, buy_price, buy_date, notes, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scryfall_id, oracle_id, name, set_code, set_name, collector_number,
                lang, int(foil), condition, quantity, buy_price,
                buy_date or self._now()[:10], notes, self._now(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_card(
        self,
        entry_id: int,
        quantity: Optional[int] = None,
        condition: Optional[str] = None,
        foil: Optional[bool] = None,
        buy_price: Optional[float] = None,
        buy_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update fields of an existing collection entry. Returns True on success."""
        updates = []
        params: list = []
        if quantity is not None:
            updates.append("quantity = ?")
            params.append(max(0, quantity))
        if condition is not None:
            cond = condition.upper()
            if cond in CONDITIONS:
                updates.append("condition = ?")
                params.append(cond)
        if foil is not None:
            updates.append("foil = ?")
            params.append(int(foil))
        if buy_price is not None:
            updates.append("buy_price = ?")
            params.append(buy_price)
        if buy_date is not None:
            updates.append("buy_date = ?")
            params.append(buy_date)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)

        if not updates:
            return False

        params.append(entry_id)
        self._conn.execute(
            f"UPDATE collection_entries SET {', '.join(updates)} WHERE id = ?", params
        )
        self._conn.commit()
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    def remove_card(self, entry_id: int) -> bool:
        """Delete a collection entry. Returns True if a row was deleted."""
        self._conn.execute("DELETE FROM collection_entries WHERE id = ?", (entry_id,))
        self._conn.commit()
        return self._conn.execute("SELECT changes()").fetchone()[0] > 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_collection(
        self,
        name_filter: str = "",
        set_filter: str = "",
        limit: int = 1000,
    ) -> list[dict]:
        """Return collection entries, optionally filtered by name or set."""
        clauses = []
        params: list = []
        if name_filter:
            clauses.append("name LIKE ?")
            params.append(f"%{name_filter}%")
        if set_filter:
            clauses.append("set_code = ?")
            params.append(set_filter.lower())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM collection_entries {where} "
            "ORDER BY name ASC, added_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_entry(self, entry_id: int) -> Optional[dict]:
        """Return a single collection entry by id."""
        row = self._conn.execute(
            "SELECT * FROM collection_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_duplicates(self) -> list[dict]:
        """Return oracle_ids that appear more than once (possible duplicates)."""
        rows = self._conn.execute(
            """SELECT oracle_id, name, SUM(quantity) AS total_qty,
                      COUNT(*) AS entry_count
               FROM collection_entries
               WHERE oracle_id != ''
               GROUP BY oracle_id
               HAVING SUM(quantity) > 1
               ORDER BY total_qty DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def record_price(
        self,
        scryfall_id: str,
        price_eur: Optional[float] = None,
        price_usd: Optional[float] = None,
        price_eur_foil: Optional[float] = None,
        source: str = "catalog",
        date: Optional[str] = None,
    ) -> bool:
        """Record a price data point for today (or a given date).

        Silently skips if an entry for this card + date already exists.

        Returns:
            True if a new row was inserted, False if it already existed.
        """
        today = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO price_history
                   (scryfall_id, date, price_eur, price_usd, price_eur_foil, source, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (scryfall_id, today, price_eur, price_usd, price_eur_foil,
                 source, self._now()),
            )
            self._conn.commit()
            return self._conn.execute("SELECT changes()").fetchone()[0] > 0
        except Exception as exc:
            logger.warning("record_price failed for %s: %s", scryfall_id, exc)
            return False

    def get_price_history(
        self, scryfall_id: str, days: int = 90
    ) -> list[dict]:
        """Return price history for one card, newest first.

        Args:
            scryfall_id: Scryfall UUID of the card.
            days: How many days of history to return (0 = all).

        Returns:
            List of dicts with keys: date, price_eur, price_usd, price_eur_foil, source.
        """
        if days > 0:
            rows = self._conn.execute(
                """SELECT date, price_eur, price_usd, price_eur_foil, source
                   FROM price_history
                   WHERE scryfall_id = ?
                     AND date >= date('now', ?)
                   ORDER BY date DESC""",
                (scryfall_id, f"-{days} days"),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT date, price_eur, price_usd, price_eur_foil, source
                   FROM price_history WHERE scryfall_id = ?
                   ORDER BY date DESC""",
                (scryfall_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def collection_value_over_time(self, days: int = 90) -> list[dict]:
        """Return the estimated total collection value per day.

        Aggregates ``SUM(quantity * price_eur)`` across all collection entries
        for each date where price data exists.

        Args:
            days: How many days of history (0 = all).

        Returns:
            List of ``{"date": str, "total_eur": float}`` dicts, oldest first.
        """
        date_filter = f"AND ph.date >= date('now', '-{days} days')" if days > 0 else ""
        rows = self._conn.execute(
            f"""
            SELECT ph.date,
                   COALESCE(SUM(
                       CASE WHEN ce.foil = 1 AND ph.price_eur_foil IS NOT NULL
                            THEN ce.quantity * ph.price_eur_foil
                            ELSE ce.quantity * COALESCE(ph.price_eur, 0)
                       END
                   ), 0.0) AS total_eur
            FROM price_history ph
            JOIN collection_entries ce ON ce.scryfall_id = ph.scryfall_id
            WHERE 1=1 {date_filter}
            GROUP BY ph.date
            ORDER BY ph.date ASC
            """
        ).fetchall()
        return [{"date": r["date"], "total_eur": r["total_eur"]} for r in rows]

    def update_prices_from_catalog(self, catalog) -> int:
        """Fetch current prices from the local catalog for all collection cards.

        Args:
            catalog: An open :class:`~mtg_scanner.lookup.card_catalog.CardCatalog`
                instance.

        Returns:
            Number of price records inserted.
        """
        entries = self.get_collection(limit=100_000)
        seen: set[str] = set()
        inserted = 0
        for entry in entries:
            sid = entry["scryfall_id"]
            if sid in seen:
                continue
            seen.add(sid)
            try:
                card = catalog.get_by_scryfall_id(sid)
                if not card:
                    continue
                prices = card.get("prices") or {}
                ok = self.record_price(
                    scryfall_id=sid,
                    price_eur=float(prices["eur"]) if prices.get("eur") else None,
                    price_usd=float(prices["usd"]) if prices.get("usd") else None,
                    price_eur_foil=float(prices["eur_foil"]) if prices.get("eur_foil") else None,
                    source="catalog",
                )
                if ok:
                    inserted += 1
            except Exception as exc:
                logger.warning("Price update failed for %s: %s", sid, exc)
        return inserted

    def stats(self) -> dict:
        """Return aggregate statistics about the collection."""
        total_entries = self._conn.execute(
            "SELECT COUNT(*) FROM collection_entries"
        ).fetchone()[0]
        total_cards = self._conn.execute(
            "SELECT COALESCE(SUM(quantity), 0) FROM collection_entries"
        ).fetchone()[0]
        unique_cards = self._conn.execute(
            "SELECT COUNT(DISTINCT oracle_id) FROM collection_entries WHERE oracle_id != ''"
        ).fetchone()[0]

        # Rough total value: quantity * buy_price where known
        total_value = self._conn.execute(
            "SELECT COALESCE(SUM(quantity * buy_price), 0.0) "
            "FROM collection_entries WHERE buy_price IS NOT NULL"
        ).fetchone()[0]

        foil_count = self._conn.execute(
            "SELECT COALESCE(SUM(quantity), 0) FROM collection_entries WHERE foil = 1"
        ).fetchone()[0]

        cond_rows = self._conn.execute(
            "SELECT condition, COALESCE(SUM(quantity), 0) AS cnt "
            "FROM collection_entries GROUP BY condition"
        ).fetchall()
        by_condition = {r["condition"]: r["cnt"] for r in cond_rows}

        # Current market value from latest price history
        market_value = self._conn.execute(
            """
            SELECT COALESCE(SUM(
                CASE WHEN ce.foil = 1 AND ph.price_eur_foil IS NOT NULL
                     THEN ce.quantity * ph.price_eur_foil
                     ELSE ce.quantity * COALESCE(ph.price_eur, 0)
                END
            ), 0.0)
            FROM collection_entries ce
            LEFT JOIN (
                SELECT scryfall_id,
                       price_eur, price_eur_foil,
                       MAX(date) AS latest_date
                FROM price_history
                GROUP BY scryfall_id
            ) ph ON ph.scryfall_id = ce.scryfall_id
            """
        ).fetchone()[0]

        price_days = self._conn.execute(
            "SELECT COUNT(DISTINCT date) FROM price_history"
        ).fetchone()[0]

        return {
            "total_entries": total_entries,
            "total_cards": total_cards,
            "unique_cards": unique_cards,
            "total_value_eur": total_value,
            "market_value_eur": market_value,
            "foil_count": foil_count,
            "by_condition": by_condition,
            "price_history_days": price_days,
            "db_path": str(self._db_path),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str) -> int:
        """Export the full collection as generic CSV. Returns row count."""
        rows = self.get_collection(limit=100_000)
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id", "name", "set_code", "set_name", "collector_number", "lang",
            "foil", "condition", "quantity", "buy_price", "buy_date",
            "scryfall_id", "oracle_id", "notes", "added_at",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def export_moxfield(self, output_path: str) -> int:
        """Export in Moxfield/Archidekt CSV format. Returns row count."""
        rows = self.get_collection(limit=100_000)
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Count", "Name", "Edition", "Condition", "Foil",
                             "Collector Number", "Language"])
            for r in rows:
                writer.writerow([
                    r["quantity"],
                    r["name"],
                    r["set_code"].upper() if r["set_code"] else "",
                    r["condition"],
                    "foil" if r["foil"] else "",
                    r["collector_number"] or "",
                    r["lang"] or "en",
                ])
        return len(rows)

    def export_tcgplayer(self, output_path: str) -> int:
        """Export in TCGplayer CSV format. Returns row count."""
        rows = self.get_collection(limit=100_000)
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # TCGplayer condition mapping
        cond_map = {"NM": "Near Mint", "LP": "Lightly Played",
                    "MP": "Moderately Played", "HP": "Heavily Played",
                    "DMG": "Damaged"}
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Quantity", "Name", "Set", "Condition", "Printing"])
            for r in rows:
                writer.writerow([
                    r["quantity"],
                    r["name"],
                    r["set_name"] or r["set_code"] or "",
                    cond_map.get(r["condition"], r["condition"]),
                    "Foil" if r["foil"] else "Normal",
                ])
        return len(rows)

    def export_cardmarket(self, output_path: str) -> int:
        """Export in Cardmarket CSV format. Returns row count."""
        rows = self.get_collection(limit=100_000)
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Cardmarket condition codes
        cond_map = {"NM": "MT", "LP": "EX", "MP": "GD", "HP": "LP", "DMG": "PO"}
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Amount", "Article", "Expansion", "Condition",
                             "Language", "Foil", "Signed", "Collector Number"])
            for r in rows:
                writer.writerow([
                    r["quantity"],
                    r["name"],
                    r["set_name"] or r["set_code"] or "",
                    cond_map.get(r["condition"], "EX"),
                    _lang_to_cardmarket(r["lang"] or "en"),
                    "Yes" if r["foil"] else "No",
                    "No",
                    r["collector_number"] or "",
                ])
        return len(rows)

    def export_arena(self, output_path: str) -> int:
        """Export in MTG Arena deck format (.dek). Returns card count."""
        rows = self.get_collection(limit=100_000)
        if not rows:
            return 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("// MTG Arena Export\n")
            for r in rows:
                set_code = (r["set_code"] or "").upper()
                num = r["collector_number"] or ""
                if set_code and num:
                    fh.write(f"{r['quantity']} {r['name']} ({set_code}) {num}\n")
                else:
                    fh.write(f"{r['quantity']} {r['name']}\n")
        return len(rows)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lang_to_cardmarket(lang: str) -> str:
    """Map Scryfall language codes to Cardmarket language names."""
    mapping = {
        "en": "English", "de": "German", "fr": "French", "it": "Italian",
        "es": "Spanish", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean",
        "ru": "Russian", "zhs": "Chinese Simplified", "zht": "Chinese Traditional",
        "he": "Hebrew", "ar": "Arabic", "ph": "Phyrexian",
    }
    return mapping.get(lang.lower(), "English")
