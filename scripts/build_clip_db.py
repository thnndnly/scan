"""Build CLIP embedding database for artwork-based card recognition.

Downloads card artwork images from Scryfall and computes CLIP embeddings for
the artwork region of each card.  Results are stored in data/clip_embeddings.db.

The database is resumable — cards already in the DB are skipped automatically.

Usage:
    python scripts/build_clip_db.py                   # All English cards
    python scripts/build_clip_db.py --sets m21,lea    # Specific sets
    python scripts/build_clip_db.py --limit 500       # First 500 cards
    python scripts/build_clip_db.py --lang en ja      # Include Japanese
    python scripts/build_clip_db.py --dry-run         # Preview only
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
_DATA_DIR = Path("data")
_RATE_LIMIT_S = 0.10
_CHUNK_SIZE = 65536

_DDL = """
CREATE TABLE IF NOT EXISTS clip_embeddings (
    id           INTEGER PRIMARY KEY,
    scryfall_id  TEXT    UNIQUE NOT NULL,
    card_name    TEXT    NOT NULL,
    set_code     TEXT,
    lang         TEXT    DEFAULT 'en',
    embedding    BLOB    NOT NULL,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clip_name   ON clip_embeddings(card_name);
CREATE INDEX IF NOT EXISTS idx_clip_set    ON clip_embeddings(set_code);
CREATE INDEX IF NOT EXISTS idx_clip_lang   ON clip_embeddings(lang);
"""


def _get_bulk_url(bulk_type: str = "default_cards") -> str:
    resp = requests.get(
        _BULK_DATA_URL, timeout=30, headers={"User-Agent": "mtg-card-scanner/1.0"}
    )
    resp.raise_for_status()
    for item in resp.json().get("data", []):
        if item.get("type") == bulk_type:
            return item["download_uri"]
    raise RuntimeError(f"Bulk data type '{bulk_type}' not found on Scryfall.")


def _load_clip():
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print(
            "ERROR: CLIP dependencies not installed.\n"
            "Install with:  pip install 'mtg-card-scanner[clip]'\n"
            "or:            pip install transformers torch"
        )
        sys.exit(1)
    print("Loading CLIP model (openai/clip-vit-base-patch32) …")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, torch


def _embed_artwork(image_bytes: bytes, model, processor, torch) -> np.ndarray:
    """Compute a normalised CLIP embedding for the artwork region of a card image."""
    import cv2
    from PIL import Image as PILImage

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is not None:
        h, w = img.shape[:2]
        y0, y1 = int(h * 0.20), int(h * 0.65)
        margin = 5
        crop = img[y0:y1, margin : w - margin]
        if crop.size == 0:
            crop = img
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(crop_rgb)
    else:
        # Fallback: let Pillow handle the format
        pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    emb = features[0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / max(norm, 1e-8)


def _fetch_card_list(
    sets: list[str] | None,
    langs: list[str],
    existing_ids: set[str],
    limit: int | None,
) -> list[dict]:
    """Download the Scryfall bulk data and return filtered card list."""
    bulk_type = "all_cards" if (langs and langs != ["en"]) else "default_cards"
    print(f"Fetching card list from Scryfall ({bulk_type}) …")
    url = _get_bulk_url(bulk_type)

    resp = requests.get(
        url, stream=True, timeout=120, headers={"User-Agent": "mtg-card-scanner/1.0"}
    )
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    raw = bytearray()
    with tqdm(
        total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading"
    ) as bar:
        for chunk in resp.iter_content(_CHUNK_SIZE):
            raw.extend(chunk)
            bar.update(len(chunk))

    all_cards = json.loads(raw)

    lang_set = set(langs)
    cards = [
        c
        for c in all_cards
        if c.get("lang", "en") in lang_set
        and (not sets or c.get("set", "") in sets)
        and c.get("id") not in existing_ids
        and c.get("image_uris", {}).get("art_crop")
    ]

    if limit:
        cards = cards[:limit]

    return cards


def build_clip_db(
    sets: list[str] | None = None,
    limit: int | None = None,
    langs: list[str] | None = None,
    db_path: str = "data/clip_embeddings.db",
) -> int:
    """Build or extend the CLIP embedding database.

    Args:
        sets: Optional list of Scryfall set codes to include (e.g. ``['m21']``).
        limit: Maximum number of new cards to embed.
        langs: Language codes to include (default: ``['en']``).
        db_path: Output SQLite database path.

    Returns:
        Total number of embeddings now in the database.
    """
    langs = langs or ["en"]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_file = Path(db_path)

    conn = sqlite3.connect(db_file)
    conn.executescript(_DDL)
    conn.commit()

    existing = {
        r[0] for r in conn.execute("SELECT scryfall_id FROM clip_embeddings").fetchall()
    }
    logger.info("Already in DB: %d embeddings", len(existing))

    cards = _fetch_card_list(sets, langs, existing, limit)
    print(f"Cards to embed: {len(cards)}  (skipping {len(existing)} already done)")

    if not cards:
        total = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
        conn.close()
        return total

    model, processor, torch = _load_clip()
    now = datetime.now(timezone.utc).isoformat()
    embedded = 0

    for card in tqdm(cards, desc="Embedding artworks"):
        art_url = card["image_uris"]["art_crop"]
        try:
            img_resp = requests.get(
                art_url, timeout=15, headers={"User-Agent": "mtg-card-scanner/1.0"}
            )
            img_resp.raise_for_status()
            emb = _embed_artwork(img_resp.content, model, processor, torch)
            conn.execute(
                "INSERT OR REPLACE INTO clip_embeddings "
                "(scryfall_id, card_name, set_code, lang, embedding, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    card["id"],
                    card["name"],
                    card.get("set", ""),
                    card.get("lang", "en"),
                    emb.tobytes(),
                    now,
                ),
            )
            conn.commit()
            embedded += 1
        except Exception as exc:
            logger.warning("Failed %s (%s): %s", card.get("name", "?"), card.get("id", "?"), exc)

        time.sleep(_RATE_LIMIT_S)

    total = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
    conn.close()
    print(f"\nEmbedded {embedded} new cards. Total in DB: {total}")
    return total


def dry_run(
    sets: list[str] | None = None,
    limit: int | None = None,
    langs: list[str] | None = None,
    db_path: str = "data/clip_embeddings.db",
) -> str:
    """Preview what would be downloaded without making any changes."""
    langs = langs or ["en"]
    db_file = Path(db_path)

    existing_count = 0
    if db_file.exists():
        conn = sqlite3.connect(db_path)
        existing_count = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
        existing_ids = {r[0] for r in conn.execute("SELECT scryfall_id FROM clip_embeddings")}
        conn.close()
    else:
        existing_ids = set()

    cards = _fetch_card_list(sets, langs, existing_ids, limit)
    est_time_min = len(cards) * (_RATE_LIMIT_S + 0.3) / 60  # ~300ms per card (dl + embed)
    est_size_mb = (len(cards) + existing_count) * 2048 * 4 / 1_000_000  # 512-dim float32

    lines = [
        f"Dry Run — CLIP Embedding Build",
        f"  Sets:              {', '.join(sets) if sets else 'alle'}",
        f"  Sprachen:          {', '.join(langs)}",
        f"  Limit:             {limit or 'kein'}",
        f"  Bereits in DB:     {existing_count}",
        f"  Neu zu embedden:   {len(cards)}",
        f"  Geschätzte Zeit:   ~{est_time_min:.0f} min",
        f"  Geschätzte Größe:  ~{est_size_mb:.0f} MB (DB gesamt)",
        f"  Ausgabedatei:      {db_path}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLIP embedding database for card artwork recognition."
    )
    parser.add_argument("--sets", help="Comma-separated set codes (e.g. m21,lea)")
    parser.add_argument("--limit", type=int, help="Maximum number of cards to embed")
    parser.add_argument("--lang", nargs="*", default=["en"], metavar="CODE")
    parser.add_argument("--db-path", default="data/clip_embeddings.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )

    sets = [s.strip() for s in args.sets.split(",")] if args.sets else None

    if args.dry_run:
        print(dry_run(sets=sets, limit=args.limit, langs=args.lang, db_path=args.db_path))
        return

    build_clip_db(sets=sets, limit=args.limit, langs=args.lang, db_path=args.db_path)


if __name__ == "__main__":
    main()
