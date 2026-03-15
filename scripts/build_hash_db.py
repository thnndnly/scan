"""Build the perceptual hash database from Scryfall card images.

Usage
-----
    python scripts/build_hash_db.py [--sets m21,lea] [--limit 500]

The script:
1. Downloads Scryfall bulk-data metadata to find the "default_cards" file URL.
2. Streams the JSON and optionally filters by set code.
3. Downloads each card's artwork, computes a pHash, and stores the result in
   ``data/card_hashes.db``.

The process is resumable: cards already present in the database are skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm

try:
    import imagehash  # type: ignore
except ImportError:
    print("imagehash is required: pip install imagehash", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
_DB_PATH = Path("data/card_hashes.db")
_RATE_LIMIT_S = 0.10  # 100 ms

_DDL = """
CREATE TABLE IF NOT EXISTS card_hashes (
    id          INTEGER PRIMARY KEY,
    card_name   TEXT NOT NULL,
    set_code    TEXT,
    hash_value  TEXT NOT NULL,
    image_uri   TEXT
);
CREATE INDEX IF NOT EXISTS idx_card_name ON card_hashes(card_name);
"""

_INSERT = """
INSERT OR IGNORE INTO card_hashes (card_name, set_code, hash_value, image_uri)
VALUES (?, ?, ?, ?);
"""

_CHECK_EXISTS = "SELECT 1 FROM card_hashes WHERE card_name = ? AND set_code = ? LIMIT 1;"


def _get_bulk_url(session: requests.Session) -> str:
    """Fetch the bulk-data index and return the URL for 'default_cards'."""
    resp = session.get(_BULK_DATA_URL, timeout=30)
    resp.raise_for_status()
    for item in resp.json().get("data", []):
        if item.get("type") == "default_cards":
            return item["download_uri"]
    raise RuntimeError("Could not find 'default_cards' in Scryfall bulk-data index.")


def _artwork_region(img: Image.Image) -> Image.Image:
    """Crop the artwork region (20 %–65 % height, 5 px horizontal margin)."""
    w, h = img.size
    top = int(h * 0.20)
    bottom = int(h * 0.65)
    return img.crop((5, top, w - 5, bottom))


def _compute_hash(image_uri: str, session: requests.Session) -> Optional[str]:
    """Download the card image and compute its pHash string."""
    try:
        resp = session.get(image_uri, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        artwork = _artwork_region(img)
        h = imagehash.phash(artwork, hash_size=16)
        return str(h)
    except Exception as exc:
        logger.warning("Could not hash %s: %s", image_uri, exc)
        return None


def build_hash_db(
    sets: Optional[list[str]] = None,
    limit: Optional[int] = None,
    db_path: Path = _DB_PATH,
    langs: Optional[list[str]] = None,
) -> None:
    """Download card images and populate the hash database.

    Args:
        sets: Optional list of set codes to include (e.g. ``['m21', 'lea']``).
              All sets are processed when ``None``.
        limit: Stop after processing this many cards (useful for testing).
        db_path: Path to the SQLite output database.
        langs: Optional list of language codes to include (e.g. ``['en', 'ja']``).
               All languages are processed when ``None``.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_DDL)
    conn.commit()

    session = requests.Session()
    session.headers["User-Agent"] = "mtg-card-scanner/0.1"

    # --- Get bulk-data URL ---
    logger.info("Fetching Scryfall bulk-data index…")
    bulk_url = _get_bulk_url(session)
    logger.info("Downloading bulk card list from %s …", bulk_url)

    resp = session.get(bulk_url, stream=True, timeout=60)
    resp.raise_for_status()
    total_bytes = int(resp.headers.get("Content-Length", 0))

    raw_bytes = bytearray()
    with tqdm(total=total_bytes or None, unit="B", unit_scale=True, desc="Bulk JSON") as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            raw_bytes.extend(chunk)
            bar.update(len(chunk))

    all_cards: list[dict] = json.loads(raw_bytes)
    logger.info("Loaded %d cards from bulk data.", len(all_cards))

    # Filter by set
    if sets:
        set_lower = {s.lower() for s in sets}
        all_cards = [c for c in all_cards if c.get("set", "").lower() in set_lower]
        logger.info("After set filter: %d cards.", len(all_cards))

    # Filter by language
    if langs:
        lang_lower = {l.lower() for l in langs}
        all_cards = [c for c in all_cards if c.get("lang", "en").lower() in lang_lower]
        logger.info("After language filter: %d cards.", len(all_cards))

    if limit is not None:
        all_cards = all_cards[:limit]

    processed = 0
    skipped = 0

    with tqdm(total=len(all_cards), desc="Building hash DB") as bar:
        for card in all_cards:
            name = card.get("name", "")
            set_code = card.get("set", "")
            image_uris = card.get("image_uris") or {}
            if not image_uris:
                faces = card.get("card_faces") or []
                if faces:
                    image_uris = faces[0].get("image_uris") or {}

            image_uri = image_uris.get("art_crop") or image_uris.get("normal") or ""

            if not image_uri:
                bar.update(1)
                skipped += 1
                continue

            # Skip if already in DB
            row = conn.execute(_CHECK_EXISTS, (name, set_code)).fetchone()
            if row:
                bar.update(1)
                skipped += 1
                continue

            hash_val = _compute_hash(image_uri, session)
            if hash_val:
                conn.execute(_INSERT, (name, set_code, hash_val, image_uri))
                conn.commit()
                processed += 1

            time.sleep(_RATE_LIMIT_S)
            bar.update(1)

    conn.close()
    logger.info(
        "Done. Processed=%d  Skipped=%d  DB=%s", processed, skipped, db_path
    )


def dry_run(
    sets: Optional[list[str]] = None,
    limit: Optional[int] = None,
    langs: Optional[list[str]] = None,
    db_path: Path = _DB_PATH,
) -> None:
    """Print a summary of what would be downloaded without actually downloading.

    Args:
        sets: Optional list of set codes to filter by.
        limit: Maximum number of cards to consider.
        langs: Optional list of language codes to filter by (e.g. ``['en', 'ja']``).
        db_path: Path to the hash database (used to count already-present cards).
    """
    print("Fetching Scryfall bulk-data index…")
    session = requests.Session()
    session.headers["User-Agent"] = "mtg-card-scanner/0.1"

    bulk_url = _get_bulk_url(session)
    print(f"Downloading bulk card list from {bulk_url} …")
    resp = session.get(bulk_url, timeout=60)
    resp.raise_for_status()
    all_cards: list[dict] = resp.json()
    print(f"Total cards in bulk data: {len(all_cards)}")

    if sets:
        set_lower = {s.lower() for s in sets}
        all_cards = [c for c in all_cards if c.get("set", "").lower() in set_lower]
        print(f"After set filter ({', '.join(sets)}): {len(all_cards)} cards")

    if langs:
        lang_lower = {l.lower() for l in langs}
        all_cards = [c for c in all_cards if c.get("lang", "en").lower() in lang_lower]
        print(f"After language filter ({', '.join(langs)}): {len(all_cards)} cards")

    if limit is not None:
        all_cards = all_cards[:limit]
        print(f"After limit ({limit}): {len(all_cards)} cards")

    # Check how many are already in the DB
    already_present = 0
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        for card in all_cards:
            row = conn.execute(
                _CHECK_EXISTS, (card.get("name", ""), card.get("set", ""))
            ).fetchone()
            if row:
                already_present += 1
        conn.close()

    to_download = len(all_cards) - already_present
    est_time_min = to_download * _RATE_LIMIT_S / 60
    est_size_mb = to_download * 50 / 1024  # ~50 KB per card image (hash only stored, not image)

    print()
    print("=== Dry-Run Summary ===")
    print(f"  Cards to process    : {len(all_cards)}")
    print(f"  Already in DB       : {already_present}")
    print(f"  Would download      : {to_download}")
    print(f"  Estimated time      : {est_time_min:.1f} minutes (@ 100ms/card)")
    print(f"  Estimated DB growth : {est_size_mb:.0f} MB (hashes only, ~50 KB/card avg)")
    print()
    print("Run without --dry-run to start the actual download.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MTG card pHash database.")
    parser.add_argument(
        "--sets",
        default=None,
        help="Comma-separated set codes to include (e.g. m21,lea).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N cards.",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Comma-separated language codes to include (e.g. en,ja,de).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Fetch bulk data, count how many cards would be downloaded, "
            "estimate time and storage, then exit without downloading."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    args = _parse_args()
    sets = [s.strip() for s in args.sets.split(",")] if args.sets else None
    langs = [l.strip() for l in args.lang.split(",")] if args.lang else None

    if args.dry_run:
        dry_run(sets=sets, limit=args.limit, langs=langs)
    else:
        build_hash_db(sets=sets, limit=args.limit, db_path=_DB_PATH, langs=langs)


if __name__ == "__main__":
    main()
