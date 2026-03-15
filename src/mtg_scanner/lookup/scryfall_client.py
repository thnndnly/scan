"""Scryfall API client with caching and rate limiting."""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from mtg_scanner.models.recognized_card import CardData
from mtg_scanner.lookup.cache import ScryfallCache

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


class ScryfallClient:
    """HTTP client for the Scryfall REST API.

    Handles caching, rate limiting, and basic error recovery (404 → None,
    429 → retry with back-off).

    Args:
        cache: :class:`ScryfallCache` instance to use for response caching.
        rate_limit_ms: Minimum delay in milliseconds between consecutive
            outbound requests.
        max_retries: How many times to retry on HTTP 429 responses.
    """

    def __init__(
        self,
        cache: Optional[ScryfallCache] = None,
        rate_limit_ms: int = 110,
        max_retries: int = 3,
    ) -> None:
        self._cache = cache
        self._rate_limit_s: float = rate_limit_ms / 1000.0
        self._max_retries = max_retries
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "mtg-card-scanner/0.1"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_rate_limit(self) -> None:
        """Block until the minimum inter-request delay has elapsed."""
        elapsed = time.monotonic() - self._last_request_time
        remaining = self._rate_limit_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

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
                self._last_request_time = time.monotonic()

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

    def lookup(self, card_name: str) -> Optional[CardData]:
        """Look up a single card by name using the Scryfall fuzzy-name endpoint.

        Results are stored in and retrieved from the local cache.  A 404
        response (card not found) is cached as the literal string ``"NOT_FOUND"``
        so that repeated lookups for non-existent names are not re-requested.

        Args:
            card_name: Card name to search for (fuzzy matching is performed
                server-side).

        Returns:
            :class:`CardData` on success, or ``None`` when Scryfall has no
            matching card or a network error occurs.
        """
        if not card_name:
            return None

        cache_key = card_name.strip().lower()

        # --- cache hit ---
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                if cached == "NOT_FOUND":
                    return None
                return _parse_card_data(cached)

        # --- outbound request ---
        url = f"{_BASE_URL}/cards/named"
        try:
            data = self._get(url, params={"fuzzy": card_name})
        except Exception as exc:
            logger.error("Scryfall lookup failed for %r: %s", card_name, exc)
            return None

        if data is None:
            logger.info("Scryfall: no card found for %r", card_name)
            if self._cache is not None:
                self._cache.set(cache_key, "NOT_FOUND")  # type: ignore[arg-type]
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
