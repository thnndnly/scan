"""Download multilingual card name mappings from Scryfall bulk data (all_cards).

Uses streaming JSON parsing (ijson) to handle the ~2.4 GB all_cards file
without loading it fully into RAM. Extracts DE, JA, and other language
name mappings in a single pass.

Usage:
    python scripts/download_card_names_fast.py --lang de ja
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import ijson
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
_DATA_DIR = Path("data")
_CHUNK_SIZE = 65536


def _get_bulk_download_url(bulk_type: str = "all_cards") -> tuple[str, int]:
    resp = requests.get(_BULK_DATA_URL, timeout=30, headers={"User-Agent": "mtg-card-scanner/1.0"})
    resp.raise_for_status()
    for item in resp.json().get("data", []):
        if item.get("type") == bulk_type:
            return item["download_uri"], item.get("size", 0)
    raise RuntimeError(f"Bulk data type '{bulk_type}' not found")


def download_and_extract(languages: list[str]) -> None:
    print(f"Fetching bulk data URL (all_cards)...")
    url, expected_size = _get_bulk_download_url("all_cards")
    print(f"Streaming ~{expected_size // 1_000_000} MB (no full RAM load)...")

    resp = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "mtg-card-scanner/1.0"})
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", expected_size or 0))

    lang_set = set(languages)
    mappings: dict[str, dict[str, str]] = {lang: {} for lang in languages}
    counts = {lang: 0 for lang in languages}

    # Stream through the HTTP response into ijson
    import io

    class _ChunkedStream(io.RawIOBase):
        """File-like wrapper around a requests chunked response."""
        def __init__(self, response, bar):
            self._chunks = response.iter_content(chunk_size=_CHUNK_SIZE)
            self._buf = b""
            self._bar = bar

        def readinto(self, b):
            n = len(b)
            while len(self._buf) < n:
                try:
                    chunk = next(self._chunks)
                    self._bar.update(len(chunk))
                    self._buf += chunk
                except StopIteration:
                    break
            out = self._buf[:n]
            self._buf = self._buf[n:]
            b[:len(out)] = out
            return len(out)

        def readable(self):
            return True

    print("Extracting name mappings (streaming, no full RAM load)...")
    with tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc="Streaming") as bar:
        stream = io.BufferedReader(_ChunkedStream(resp, bar), buffer_size=_CHUNK_SIZE * 4)
        parser = ijson.items(stream, "item")
        for card in parser:
            lang = card.get("lang", "en")
            if lang not in lang_set:
                continue
            printed = card.get("printed_name") or card.get("name", "")
            en_name = card.get("name", "")
            if printed and en_name and printed != en_name:
                mappings[lang][printed] = en_name
                counts[lang] += 1

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    for lang in languages:
        out = _DATA_DIR / f"card_names_{lang}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(mappings[lang], fh, ensure_ascii=False, indent=2)
        print(f"{lang.upper()}: {counts[lang]} mappings -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast multilingual card name download using Scryfall all_cards bulk data."
    )
    parser.add_argument("--lang", nargs="+", default=["de", "ja"], metavar="CODE")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    download_and_extract(args.lang)


if __name__ == "__main__":
    main()
