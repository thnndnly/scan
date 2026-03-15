"""Download the canonical MTG card name list and multilingual name mappings.

Usage
-----
    python scripts/download_card_names.py              # English only
    python scripts/download_card_names.py --lang de    # + German mapping
    python scripts/download_card_names.py --lang de ja # + German & Japanese

The English list is saved to ``data/card_names.json``.
Multilingual mappings (localized_name → english_name) are saved to
``data/card_names_de.json``, ``data/card_names_ja.json``, etc.

The multilingual mappings are built from the Scryfall ``default_cards``
bulk-data file (~100 MB download, streamed and parsed incrementally).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CATALOG_URL = "https://api.scryfall.com/catalog/card-names"
_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
_DATA_DIR = Path("data")
_CHUNK_SIZE = 65536  # bytes

# Scryfall language codes we support for mapping downloads
_SUPPORTED_LANGS = {"de", "ja", "fr", "es", "it", "pt", "ru", "ko", "zhs", "zht"}


# ---------------------------------------------------------------------------
# English name list
# ---------------------------------------------------------------------------


def download_card_names(output_path: Path = _DATA_DIR / "card_names.json") -> int:
    """Download the English card name catalog from Scryfall."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Requesting %s …", _CATALOG_URL)
    response = requests.get(_CATALOG_URL, stream=True, timeout=60,
                            headers={"User-Agent": "mtg-card-scanner/1.0"})
    response.raise_for_status()

    total_bytes = int(response.headers.get("Content-Length", 0))
    raw_bytes = bytearray()
    with tqdm(total=total_bytes or None, unit="B", unit_scale=True,
              unit_divisor=1024, desc="Downloading English card names") as bar:
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            raw_bytes.extend(chunk)
            bar.update(len(chunk))

    data = json.loads(raw_bytes)
    names: list[str] = data["data"] if isinstance(data, dict) else data

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(names, fh, indent=2, ensure_ascii=False)

    logger.info("Saved %d English card names to %s", len(names), output_path)
    return len(names)


# ---------------------------------------------------------------------------
# Multilingual mappings from Scryfall bulk data
# ---------------------------------------------------------------------------


_SEARCH_URL = "https://api.scryfall.com/cards/search"
_RATE_LIMIT_S = 0.12  # 120 ms between requests as per Scryfall guidelines


def _fetch_lang_mapping(lang: str) -> dict[str, str]:
    """Fetch all printed_name → english_name pairs for *lang* via Scryfall search.

    Uses paginated GET /cards/search?q=lang:{lang}&unique=prints.

    Args:
        lang: Scryfall language code, e.g. ``'de'`` or ``'ja'``.

    Returns:
        Dict mapping localized printed name to English canonical name.
    """
    import time

    mapping: dict[str, str] = {}
    url: str | None = f"{_SEARCH_URL}?q=lang%3A{lang}&unique=prints&order=name"
    page = 0

    with tqdm(desc=f"Fetching {lang.upper()} cards", unit="cards") as bar:
        while url:
            resp = requests.get(
                url, timeout=30, headers={"User-Agent": "mtg-card-scanner/1.0"}
            )
            if resp.status_code == 404:
                break  # no results for this language
            resp.raise_for_status()
            data = resp.json()

            for card in data.get("data", []):
                printed = card.get("printed_name") or card.get("name", "")
                en_name = card.get("name", "")
                if printed and en_name and printed != en_name:
                    mapping[printed] = en_name

            bar.update(len(data.get("data", [])))
            page += 1

            if data.get("has_more"):
                url = data.get("next_page")
                time.sleep(_RATE_LIMIT_S)
            else:
                url = None

    return mapping


def download_multilingual_mappings(languages: list[str]) -> dict[str, int]:
    """Download localized name -> English name mappings for *languages*.

    Uses the Scryfall paginated search API (``lang:{code}``).

    Args:
        languages: List of Scryfall language codes, e.g. ``['de', 'ja']``.

    Returns:
        Dict mapping language code to number of entries saved.
    """
    languages = [lang for lang in languages if lang in _SUPPORTED_LANGS]
    if not languages:
        logger.warning("No supported languages requested.")
        return {}

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for lang in languages:
        logger.info("Downloading %s name mappings …", lang.upper())
        mapping = _fetch_lang_mapping(lang)
        out_path = _DATA_DIR / f"card_names_{lang}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
        counts[lang] = len(mapping)
        logger.info("Saved %d %s name mappings to %s", len(mapping), lang, out_path)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MTG card name lists for OCR recognition."
    )
    parser.add_argument(
        "--lang",
        nargs="*",
        metavar="CODE",
        help=(
            "Additional language codes to download mappings for "
            f"(supported: {', '.join(sorted(_SUPPORTED_LANGS))}). "
            "Example: --lang de ja"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    try:
        count = download_card_names()
        print(f"English: {count} card names -> data/card_names.json")
    except Exception as exc:
        logger.error("Failed to download English names: %s", exc)
        sys.exit(1)

    if args.lang:
        try:
            counts = download_multilingual_mappings(args.lang)
            for lang, n in counts.items():
                print(f"{lang.upper()}: {n} name mappings -> data/card_names_{lang}.json")
        except Exception as exc:
            logger.error("Failed to download multilingual mappings: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
