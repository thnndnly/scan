#!/usr/bin/env python3
"""Download MTGJSON foreignData and produce card_names_{lang}.json files.

Downloads AllPrintings.json.bz2 from MTGJSON and extracts localized card name
mappings compatible with mtg_scanner/utils/fuzzy_search.py.

Usage:
    python scripts/download_mtgjson.py
    python scripts/download_mtgjson.py --lang de ja fr
    python scripts/download_mtgjson.py --output-dir data/
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MTGJSON_URL = "https://mtgjson.com/api/v5/AllPrintings.json.bz2"
LANG_MAP = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese (Brazil)",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "zhs": "Chinese Simplified",
    "zht": "Chinese Traditional",
}


def _download_file(url: str, dest: Path) -> None:
    logger.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = block_num * block_size * 100 / total_size
            print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _load_all_printings(bz2_path: Path) -> dict:
    logger.info("Decompressing %s …", bz2_path)
    with bz2.open(bz2_path, "rb") as fh:
        data = json.load(fh)
    return data.get("data", {})


def _extract_language(all_printings: dict, lang_code: str) -> dict[str, str]:
    """Return {localized_name: english_name} mapping for the given language."""
    target_lang = LANG_MAP.get(lang_code, lang_code)
    mapping: dict[str, str] = {}
    seen_en: set[str] = set()

    for set_data in all_printings.values():
        for card in set_data.get("cards", []):
            en_name: str = card.get("name", "")
            if not en_name:
                continue
            for foreign in card.get("foreignData", []):
                if foreign.get("language") == target_lang:
                    loc_name: str = foreign.get("name", "")
                    if loc_name and loc_name not in mapping:
                        mapping[loc_name] = en_name
                    break
            # Also collect English names for completeness
            if en_name not in seen_en:
                seen_en.add(en_name)

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MTGJSON localized card names")
    parser.add_argument(
        "--lang", nargs="+", default=["de", "ja"],
        help=f"Language codes to extract. Available: {', '.join(LANG_MAP.keys())}"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--cache-file", default="data/AllPrintings.json.bz2",
        help="Path to cache the downloaded file (avoids re-downloading)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_file)

    if not cache_path.exists():
        _download_file(MTGJSON_URL, cache_path)
    else:
        logger.info("Using cached file: %s", cache_path)

    logger.info("Parsing MTGJSON data …")
    all_printings = _load_all_printings(cache_path)
    logger.info("Loaded %d sets", len(all_printings))

    for lang in args.lang:
        if lang not in LANG_MAP:
            logger.warning("Unknown language code %r, skipping. Known: %s", lang, list(LANG_MAP))
            continue
        logger.info("Extracting %s names …", LANG_MAP[lang])
        mapping = _extract_language(all_printings, lang)
        out_path = output_dir / f"card_names_{lang}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote %d entries to %s", len(mapping), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
