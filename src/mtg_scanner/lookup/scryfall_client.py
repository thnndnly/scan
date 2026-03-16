"""Scryfall API client with caching and rate limiting.

Lookup priority (when ``prefer_local=True`` and catalog is available):
  1. Local SQLite cache (``scryfall_cache.db``)
  2. Local card catalog (``card_catalog.db``)  — no network, instant
  3. Scryfall REST API                          — rate-limited, requires internet
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import requests

from mtg_scanner.models.recognized_card import CardData
from mtg_scanner.lookup.cache import ScryfallCache

if TYPE_CHECKING:
    from mtg_scanner.lookup.card_catalog import CardCatalog

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.scryfall.com"


def _parse_card_data(data: dict) -> CardData:
    """Parse a raw Scryfall card JSON object into a :class:`CardData` instance.

    Args:
        data: Raw dict from the Scryfall API response.

    Returns:
        Populated :class:`CardData` object.
    """
    prices = data.get("prices", {}) or {}
    eur_raw = prices.get("eur")
    usd_raw = prices.get("usd")

    image_uris = data.get("image_uris") or {}
    # Some cards (DFCs) store image URIs under card_faces
    if not image_uris:
        faces = data.get("card_faces") or []
        if faces:
            image_uris = faces[0].get("image_uris") or {}

    return CardData(
        name=data.get("name", ""),
        set_code=data.get("set", ""),
        collector_number=data.get("collector_number", ""),
        rarity=data.get("rarity", ""),
        type_line=data.get("type_line", ""),
        price_eur=float(eur_raw) if eur_raw else None,
        price_usd=float(usd_raw) if usd_raw else None,
        scryfall_uri=data.get("scryfall_uri", ""),
        image_uri=image_uris.get("normal", ""),
    )


def _parse_card_data_from_catalog(card: dict) -> CardData:
    """Convert a catalog row dict (from CardCatalog) into a :class:`CardData`.

    Args:
        card: Dict as returned by :meth:`CardCatalog._row_to_dict`.

    Returns:
        Populated :class:`CardData`.
    """
    prices = card.get("prices") or {}
    eur_raw = prices.get("eur")
    usd_raw = prices.get("usd")

    image_uris = card.get("image_uris") or {}
    if isinstance(image_uris, str):
        import json as _json
        try:
            image_uris = _json.loads(image_uris)
        except Exception:
            image_uris = {}

    return CardData(
        name=card.get("name", ""),
        set_code=card.get("set_code", ""),
        collector_number=card.get("collector_number", ""),
        rarity=card.get("rarity", ""),
        type_line=card.get("type_line", ""),
        price_eur=float(eur_raw) if eur_raw else None,
        price_usd=float(usd_raw) if usd_raw else None,
        scryfall_uri=card.get("scryfall_uri", ""),
        image_uri=image_uris.get("normal", "") if isinstance(image_uris, dict) else "",
    )


class ScryfallClient:
    """HTTP client for the Scryfall REST API.

    Handles caching, rate limiting, and basic error recovery (404 → None,
    429 → retry with back-off).

    The rate limit is enforced globally across all instances via a class-level
    lock and timestamp, so multiple Pipeline instances cannot exceed Scryfall's
    110 ms inter-request requirement.

    Args:
        cache: :class:`ScryfallCache` instance to use for response caching.
        rate_limit_ms: Minimum delay in milliseconds between consecutive
            outbound requests.
        max_retries: How many times to retry on HTTP 429 responses.
    """

    # Shared across all instances so parallel pipelines stay within rate limit
    _global_last_request_time: float = 0.0
    _global_rate_lock = threading.Lock()

    def __init__(
        self,
        cache: Optional[ScryfallCache] = None,
        rate_limit_ms: int = 110,
        max_retries: int = 3,
        catalog: Optional["CardCatalog"] = None,
        prefer_local: bool = True,
    ) -> None:
        self._cache = cache
        self._rate_limit_s: float = rate_limit_ms / 1000.0
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "mtg-card-scanner/0.1"})
        self._catalog = catalog
        self._prefer_local = prefer_local

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        """Block until the minimum inter-request delay has elapsed (global across all instances)."""
        with ScryfallClient._global_rate_lock:
            elapsed = time.monotonic() - ScryfallClient._global_last_request_time
            remaining = self._rate_limit_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
            ScryfallClient._global_last_request_time = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Perform a rate-limited GET request with retry on HTTP 429.

        Args:
            url: Full URL to request.
            params: Optional query parameters.

        Returns:
            Parsed JSON dict, or ``None`` on 404.

        Raises:
            requests.HTTPError: For non-404, non-429 HTTP errors.
            requests.RequestException: For network-level failures.
        """
        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self._session.get(url, params=params, timeout=15)

                if response.status_code == 404:
                    return None

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "1"))
                    logger.warning(
                        "Rate limited by Scryfall; waiting %ds (attempt %d/%d)",
                        retry_after,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.ConnectionError as exc:
                logger.error("Network error contacting Scryfall: %s", exc)
                raise

        logger.error("Exceeded max retries for %s", url)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _lookup_local(self, card_name: str) -> Optional[CardData]:
        """Try to resolve *card_name* from the local card catalog.

        Performs an exact name match first, then falls back to a LIKE search.

        Args:
            card_name: Card name to look up.

        Returns:
            :class:`CardData` if found in the local catalog, otherwise ``None``.
        """
        if self._catalog is None or not self._prefer_local:
            return None
        try:
            # Exact oracle match first (fastest)
            oracle_id = self._catalog.get_oracle_id(card_name.strip())
            if oracle_id:
                printings = self._catalog.get_printings(oracle_id)
                if printings:
                    # Use the newest printing (last in ascending-date list)
                    return _parse_card_data_from_catalog(printings[-1])
            # Fuzzy LIKE search
            results = self._catalog.search_by_name(card_name.strip(), limit=1)
            if results:
                return _parse_card_data_from_catalog(results[0])
        except Exception as exc:
            logger.debug("Local catalog lookup failed for %r: %s", card_name, exc)
        return None

    def lookup(self, card_name: str) -> Optional[CardData]:
        """Look up a single card by name.

        Lookup priority:
        1. Local SQLite response cache
        2. Local card catalog (if ``prefer_local=True`` and catalog attached)
        3. Scryfall REST API (rate-limited)

        A 404 API response is cached as ``"NOT_FOUND"`` to prevent repeat requests.

        Args:
            card_name: Card name to search for.

        Returns:
            :class:`CardData` on success, or ``None`` when no card is found.
        """
        if not card_name or len(card_name.strip()) < 2:
            return None

        cache_key = card_name.strip().lower()

        # --- 1. Response cache ---
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, dict) and cached.get("_not_found"):
                    return None
                return _parse_card_data(cached)

        # --- 2. Local catalog ---
        if self._prefer_local and self._catalog is not None:
            local = self._lookup_local(card_name)
            if local is not None:
                logger.debug("Local catalog hit for %r", card_name)
                return local

        # --- 3. Scryfall API ---
        url = f"{_BASE_URL}/cards/named"
        try:
            data = self._get(url, params={"fuzzy": card_name})
        except Exception as exc:
            logger.error("Scryfall lookup failed for %r: %s", card_name, exc)
            return None

        if data is None:
            logger.info("Scryfall: no card found for %r", card_name)
            if self._cache is not None:
                self._cache.set(cache_key, {"_not_found": True})
            return None

        if self._cache is not None:
            self._cache.set(cache_key, data)

        try:
            return _parse_card_data(data)
        except Exception as exc:
            logger.error("Failed to parse Scryfall response for %r: %s", card_name, exc)
            return None

    def search(self, query: str, page: int = 1) -> list[dict]:
        """Perform a full-text Scryfall search and return raw card dicts.

        Args:
            query: Scryfall search query string.
            page: Page number (1-based).

        Returns:
            List of raw card JSON dicts (may be empty).
        """
        url = f"{_BASE_URL}/cards/search"
        try:
            data = self._get(url, params={"q": query, "page": page})
        except Exception as exc:
            logger.error("Scryfall search failed for %r: %s", query, exc)
            return []
        if data is None:
            return []
        return data.get("data", [])

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()
