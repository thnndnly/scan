#!/usr/bin/env python3
"""Build the local card catalog from Scryfall bulk data.

Downloads the ``default_cards`` bulk file (~155 MB) and imports it into
``data/card_catalog.db``.  Run once; subsequent runs update only if Scryfall
has published a newer bulk file.

Usage:
    python scripts/build_card_catalog.py
    python scripts/build_card_catalog.py --force      # Re-download even if fresh
    python scripts/build_card_catalog.py --bulk-type oracle_cards
    python scripts/build_card_catalog.py --check      # Only check freshness, no download
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

BULK_API = "https://api.scryfall.com/bulk-data"
SETS_API = "https://api.scryfall.com/sets"
HEADERS = {"User-Agent": "mtg-card-scanner/0.1 (github.com/your-repo)"}
DEFAULT_DB = "data/card_catalog.db"
DEFAULT_BULK_TYPE = "default_cards"
BATCH_SIZE = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id                TEXT PRIMARY KEY,
    oracle_id         TEXT,
    name              TEXT NOT NULL,
    printed_name      TEXT,
    lang              TEXT NOT NULL DEFAULT 'en',
    set_code          TEXT NOT NULL,
    set_name          TEXT NOT NULL,
    set_type          TEXT,
    collector_number  TEXT NOT NULL,
    released_at       TEXT,
    rarity            TEXT,
    artist            TEXT,
    mana_cost         TEXT,
    cmc               REAL,
    type_line         TEXT,
    oracle_text       TEXT,
    colors            TEXT,
    color_identity    TEXT,
    finishes          TEXT,
    frame_effects     TEXT,
    border_color      TEXT,
    promo             INTEGER DEFAULT 0,
    promo_types       TEXT,
    variation         INTEGER DEFAULT 0,
    digital           INTEGER DEFAULT 0,
    full_art          INTEGER DEFAULT 0,
    reprint           INTEGER DEFAULT 0,
    layout            TEXT,
    illustration_id   TEXT,
    prices            TEXT,
    image_uris        TEXT,
    keywords          TEXT,
    scryfall_uri      TEXT
);

CREATE TABLE IF NOT EXISTS sets (
    code          TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    set_type      TEXT,
    released_at   TEXT,
    card_count    INTEGER,
    digital       INTEGER DEFAULT 0,
    foil_only     INTEGER DEFAULT 0,
    nonfoil_only  INTEGER DEFAULT 0,
    icon_svg_uri  TEXT,
    scryfall_uri  TEXT
);

CREATE TABLE IF NOT EXISTS catalog_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_cards_oracle_id  ON cards(oracle_id);
CREATE INDEX IF NOT EXISTS idx_cards_name       ON cards(name);
CREATE INDEX IF NOT EXISTS idx_cards_set        ON cards(set_code, collector_number);
CREATE INDEX IF NOT EXISTS idx_cards_lang       ON cards(lang);
CREATE INDEX IF NOT EXISTS idx_cards_released   ON cards(released_at);
"""


def _j(obj) -> str | None:
    """Serialize a Python object to JSON string, or None if falsy."""
    return json.dumps(obj, ensure_ascii=False) if obj else None


def fetch_bulk_meta(bulk_type: str) -> dict:
    print(f"Prüfe Scryfall Bulk-Daten ({bulk_type})…")
    resp = requests.get(BULK_API, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    for item in resp.json().get("data", []):
        if item.get("type") == bulk_type:
            return item
    raise ValueError(f"Bulk-Typ '{bulk_type}' nicht gefunden.")


def fetch_sets(conn: sqlite3.Connection) -> None:
    print("Lade Sets von Scryfall…", end=" ", flush=True)
    resp = requests.get(SETS_API, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    sets_data = resp.json().get("data", [])
    rows = []
    for s in sets_data:
        rows.append((
            s.get("code", ""),
            s.get("name", ""),
            s.get("set_type"),
            s.get("released_at"),
            s.get("card_count"),
            int(s.get("digital", False)),
            int(s.get("foil_only", False)),
            int(s.get("nonfoil_only", False)),
            s.get("icon_svg_uri"),
            s.get("scryfall_uri"),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO sets VALUES (?,?,?,?,?,?,?,?,?,?)""", rows
    )
    conn.commit()
    print(f"{len(rows)} Sets importiert.")


def download_bulk(url: str, dest: Path) -> None:
    print(f"Lade Bulk-Datei herunter: {url}")
    resp = requests.get(url, headers=HEADERS, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1_048_576
                total_mb = total / 1_048_576
                elapsed = time.time() - start
                speed = mb / elapsed if elapsed > 0 else 0
                print(
                    f"\r  {pct:.1f}%  {mb:.1f} / {total_mb:.1f} MB"
                    f"  ({speed:.1f} MB/s)    ",
                    end="",
                    flush=True,
                )
    print(f"\nDownload abgeschlossen: {downloaded / 1_048_576:.1f} MB")


def import_cards(conn: sqlite3.Connection, json_path: Path) -> int:
    print("Importiere Karten in SQLite…")
    print("  (Lädt JSON in Speicher — kann 1-2 GB RAM benötigen)")

    with open(json_path, encoding="utf-8") as fh:
        cards = json.load(fh)

    total = len(cards)
    print(f"  {total:,} Karten gefunden.")

    inserted = 0
    batch = []

    def flush(b):
        conn.executemany(
            """INSERT OR REPLACE INTO cards VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            b,
        )
        conn.commit()

    for i, c in enumerate(cards):
        img = c.get("image_uris") or {}
        # For double-faced cards, use front face image_uris
        if not img and c.get("card_faces"):
            img = c["card_faces"][0].get("image_uris") or {}
        # Keep only small/normal/art_crop URLs to save space
        img_slim = {k: img[k] for k in ("small", "normal", "art_crop") if k in img}

        batch.append((
            c.get("id", ""),
            c.get("oracle_id"),
            c.get("name", ""),
            c.get("printed_name"),
            c.get("lang", "en"),
            c.get("set", ""),
            c.get("set_name", ""),
            c.get("set_type"),
            c.get("collector_number", ""),
            c.get("released_at"),
            c.get("rarity"),
            c.get("artist"),
            c.get("mana_cost"),
            c.get("cmc"),
            c.get("type_line"),
            c.get("oracle_text"),
            _j(c.get("colors")),
            _j(c.get("color_identity")),
            _j(c.get("finishes")),
            _j(c.get("frame_effects")),
            c.get("border_color"),
            int(c.get("promo", False)),
            _j(c.get("promo_types")),
            int(c.get("variation", False)),
            int(c.get("digital", False)),
            int(c.get("full_art", False)),
            int(c.get("reprint", False)),
            c.get("layout"),
            c.get("illustration_id"),
            _j(c.get("prices")),
            _j(img_slim) if img_slim else None,
            _j(c.get("keywords")),
            c.get("scryfall_uri"),
        ))

        if len(batch) >= BATCH_SIZE:
            flush(batch)
            inserted += len(batch)
            batch = []
            if inserted % 10000 == 0:
                pct = inserted / total * 100
                print(f"  {inserted:,} / {total:,} ({pct:.0f}%)")

    if batch:
        flush(batch)
        inserted += len(batch)

    print(f"  {inserted:,} Karten importiert.")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Scryfall Card Catalog aufbauen")
    parser.add_argument("--bulk-type", default=DEFAULT_BULK_TYPE,
                        choices=["oracle_cards", "default_cards", "all_cards"],
                        help="Bulk-Datentyp (Standard: default_cards)")
    parser.add_argument("--db", default=DEFAULT_DB, help="Ziel-SQLite-Datei")
    parser.add_argument("--force", action="store_true", help="Neu laden auch wenn aktuell")
    parser.add_argument("--check", action="store_true", help="Nur Aktualität prüfen")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check freshness
    meta = fetch_bulk_meta(args.bulk_type)
    remote_updated_at = meta.get("updated_at", "")
    download_uri = meta["download_uri"]
    remote_size = meta.get("size", 0)

    if args.check:
        print(f"Remote updated_at : {remote_updated_at}")
        print(f"Remote size       : {remote_size / 1_048_576:.1f} MB (unkomprimiert)")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            local_updated = conn.execute(
                "SELECT value FROM catalog_meta WHERE key='updated_at'"
            ).fetchone()
            conn.close()
            local_val = local_updated[0] if local_updated else "nicht vorhanden"
            print(f"Lokale updated_at : {local_val}")
            if local_val == remote_updated_at:
                print("Status: AKTUELL")
            else:
                print("Status: VERALTET — neu bauen mit: python scripts/build_card_catalog.py")
        else:
            print("Lokale DB nicht vorhanden.")
        return

    # Check if update needed
    if not args.force and db_path.exists():
        try:
            conn_check = sqlite3.connect(str(db_path))
            row = conn_check.execute(
                "SELECT value FROM catalog_meta WHERE key='updated_at'"
            ).fetchone()
            conn_check.close()
            if row and row[0] == remote_updated_at:
                print(f"Katalog ist aktuell (updated_at={remote_updated_at}). Kein Download nötig.")
                print("Erzwinge mit --force.")
                return
        except Exception:
            pass

    # Download
    tmp_path = db_path.parent / "card_catalog_tmp.json"
    try:
        download_bulk(download_uri, tmp_path)

        # Build DB
        if db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)

        fetch_sets(conn)
        count = import_cards(conn, tmp_path)

        # Store metadata
        now = datetime.now(timezone.utc).isoformat()
        for k, v in [
            ("bulk_type", args.bulk_type),
            ("updated_at", remote_updated_at),
            ("imported_at", now),
            ("card_count", str(count)),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO catalog_meta VALUES (?,?)", (k, v)
            )
        conn.commit()
        conn.close()

        db_size_mb = db_path.stat().st_size / 1_048_576
        print(f"\nFertig! {count:,} Karten in {db_path} ({db_size_mb:.0f} MB)")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()
