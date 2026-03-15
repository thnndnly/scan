"""Fuzzy card-name search backed by rapidfuzz."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache: {names_file_path -> list[str]}
_names_cache: dict[str, list[str]] = {}

# Module-level cache for localized name→English mappings: {path -> dict}
_mapping_cache: dict[str, dict[str, str]] = {}


def load_name_mapping(path: str) -> dict[str, str]:
    """Load a localized-name → English-name mapping from a JSON file.

    The file is expected to be a JSON object ``{"LocalName": "EnglishName", ...}``.

    Args:
        path: Path to the mapping JSON file.

    Returns:
        Dict mapping localized names to English canonical names.
        Empty dict if the file is missing or cannot be parsed.
    """
    global _mapping_cache
    if path in _mapping_cache:
        return _mapping_cache[path]

    p = Path(path)
    if not p.exists():
        logger.warning("Name mapping file not found: %s", path)
        return {}

    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("Unexpected format in %s; expected a JSON object.", path)
            return {}
        _mapping_cache[path] = data
        logger.info("Loaded %d name mappings from %s", len(data), path)
        return data
    except Exception as exc:
        logger.error("Failed to load name mapping from %s: %s", path, exc)
        return {}


def load_card_names(path: str = "data/card_names.json") -> list[str]:
    """Load the list of known card names from *path*.

    Results are cached per-path so the file is only read once per session.

    Args:
        path: Path to a JSON file containing an array of card name strings.

    Returns:
        List of card name strings.  Empty list if the file is missing or
        cannot be parsed.
    """
    global _names_cache
    if path in _names_cache:
        return _names_cache[path]

    p = Path(path)
    if not p.exists():
        logger.warning("Card names file not found: %s", path)
        return []

    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            names: list[str] = [str(n) for n in data]
        elif isinstance(data, dict) and "data" in data:
            names = [str(n) for n in data["data"]]
        else:
            logger.warning("Unexpected format in %s; expected a JSON array.", path)
            names = []
        _names_cache[path] = names
        logger.info("Loaded %d card names from %s", len(names), path)
        return names
    except Exception as exc:
        logger.error("Failed to load card names from %s: %s", path, exc)
        return []


def fuzzy_match(
    query: str,
    names: list[str],
    score_cutoff: float = 60.0,
    limit: int = 1,
) -> list[tuple[str, float]]:
    """Find the closest card name(s) in *names* for the given *query* string.

    Uses ``rapidfuzz.process.extractBests`` with the token-sort-ratio scorer,
    which handles word-order variations common in OCR output.

    Args:
        query: Raw string to match (e.g. from OCR).
        names: List of candidate card names.
        score_cutoff: Minimum similarity score (0-100).  Matches below this
            threshold are excluded.
        limit: Maximum number of results to return.

    Returns:
        List of ``(card_name, normalised_confidence)`` tuples ordered by
        descending confidence.  *normalised_confidence* is in the range
        ``[0.0, 1.0]``.
    """
    try:
        from rapidfuzz import process, fuzz  # type: ignore
    except ImportError as exc:
        raise ImportError("rapidfuzz is required: pip install rapidfuzz") from exc

    if not query or not names:
        return []

    results = process.extract(
        query,
        names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff,
        limit=limit,
    )
    # process.extract returns (match, score, index)
    return [(match, score / 100.0) for match, score, _ in results]


def best_match(
    query: str,
    names: list[str],
    score_cutoff: float = 60.0,
    names_file: str = "data/card_names.json",
) -> tuple[Optional[str], float]:
    """Return the single best matching card name for *query*.

    If *names* is empty the function will attempt to load names from
    *names_file* automatically.

    Args:
        query: Raw string to match.
        names: Candidate list; loaded from *names_file* if empty.
        score_cutoff: Minimum similarity (0-100).
        names_file: Fallback names file path.

    Returns:
        ``(matched_name, confidence)`` where *confidence* is in ``[0.0, 1.0]``,
        or ``(None, 0.0)`` when nothing meets the cutoff.
    """
    if not names:
        names = load_card_names(names_file)

    matches = fuzzy_match(query, names, score_cutoff=score_cutoff, limit=1)
    if matches:
        return matches[0]
    return None, 0.0


def best_match_multilingual(
    query: str,
    en_names: list[str],
    mapping_paths: list[str],
    score_cutoff: float = 60.0,
) -> tuple[Optional[str], float]:
    """Find the best English card name for *query*, searching across languages.

    First tries a direct fuzzy match against the English name list.  If that
    fails, looks up *query* in each localized-name → English-name mapping file
    (exact lookup, then fuzzy against localized keys) and returns the English
    canonical name.

    Args:
        query: OCR text to match (may be in any language).
        en_names: List of English canonical card names.
        mapping_paths: Ordered list of paths to localized name mapping files
            (e.g. ``["data/card_names_de.json", "data/card_names_ja.json"]``).
        score_cutoff: Minimum similarity score (0-100) for fuzzy matching.

    Returns:
        ``(english_name, confidence)`` or ``(None, 0.0)`` if no match found.
    """
    # 1. Try English names directly
    name, conf = best_match(query, en_names, score_cutoff=score_cutoff)
    if name is not None:
        return name, conf

    # 2. Try each localized mapping
    for path in mapping_paths:
        mapping = load_name_mapping(path)
        if not mapping:
            continue

        # Exact lookup (case-insensitive)
        query_lower = query.strip().lower()
        for local_name, en_name in mapping.items():
            if local_name.lower() == query_lower:
                logger.info("Multilingual exact match: %r → %r (%s)", query, en_name, path)
                return en_name, 1.0

        # Fuzzy match against localized keys
        local_names = list(mapping.keys())
        local_match, local_conf = best_match(query, local_names, score_cutoff=score_cutoff)
        if local_match is not None:
            en_name = mapping[local_match]
            logger.info(
                "Multilingual fuzzy match: %r → %r → %r (conf=%.2f, %s)",
                query, local_match, en_name, local_conf, path
            )
            return en_name, local_conf

    return None, 0.0
