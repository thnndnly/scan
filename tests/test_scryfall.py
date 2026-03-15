"""Tests for the Scryfall client and cache."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from mtg_scanner.lookup.cache import ScryfallCache
from mtg_scanner.lookup.scryfall_client import ScryfallClient, _parse_card_data
from mtg_scanner.models.recognized_card import CardData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, json_data: dict | None = None):
    """Create a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests

        mock.raise_for_status.side_effect = requests.HTTPError(
            response=mock
        )
    return mock


# ---------------------------------------------------------------------------
# _parse_card_data
# ---------------------------------------------------------------------------


class TestParseCardData:
    def test_all_fields(self, mock_scryfall_response):
        card = _parse_card_data(mock_scryfall_response)
        assert card.name == "Lightning Bolt"
        assert card.set_code == "m21"
        assert card.collector_number == "167"
        assert card.rarity == "common"
        assert card.type_line == "Instant"
        assert card.price_usd == pytest.approx(0.30)
        assert card.price_eur == pytest.approx(0.25)
        assert "scryfall.com" in card.scryfall_uri
        assert "cards.scryfall.io" in card.image_uri

    def test_missing_prices(self, mock_scryfall_response):
        mock_scryfall_response["prices"] = {}
        card = _parse_card_data(mock_scryfall_response)
        assert card.price_usd is None
        assert card.price_eur is None

    def test_dfc_image_uri(self):
        """Double-faced cards store image URIs under card_faces."""
        data = {
            "name": "Delver of Secrets // Insectile Aberration",
            "set": "isd",
            "collector_number": "51",
            "rarity": "common",
            "type_line": "Creature",
            "prices": {},
            "scryfall_uri": "https://scryfall.com/card/isd/51",
            "card_faces": [
                {
                    "image_uris": {
                        "normal": "https://cards.scryfall.io/normal/isd/51a.jpg"
                    }
                }
            ],
        }
        card = _parse_card_data(data)
        assert "isd/51a" in card.image_uri


# ---------------------------------------------------------------------------
# ScryfallClient
# ---------------------------------------------------------------------------


class TestScryfallClient:
    def _client_with_mock(self, response, cache=None):
        client = ScryfallClient(cache=cache, rate_limit_ms=0)
        client._session = MagicMock()
        client._session.get.return_value = response
        return client

    def test_successful_lookup(self, mock_scryfall_response):
        mock_resp = _make_response(200, mock_scryfall_response)
        client = self._client_with_mock(mock_resp)
        card = client.lookup("Lightning Bolt")
        assert card is not None
        assert isinstance(card, CardData)
        assert card.name == "Lightning Bolt"

    def test_404_returns_none(self):
        mock_resp = _make_response(404)
        mock_resp.raise_for_status = MagicMock()  # 404 handled manually
        client = self._client_with_mock(mock_resp)
        result = client.lookup("Definitely Not A Card")
        assert result is None

    def test_cache_prevents_second_request(self, tmp_path, mock_scryfall_response):
        """A second lookup for the same card must not issue a new HTTP request."""
        cache = ScryfallCache(db_path=str(tmp_path / "cache.db"), ttl_hours=24)
        mock_resp = _make_response(200, mock_scryfall_response)
        client = self._client_with_mock(mock_resp, cache=cache)

        card1 = client.lookup("Lightning Bolt")
        card2 = client.lookup("Lightning Bolt")

        assert card1 is not None
        assert card2 is not None
        assert card1.name == card2.name
        # HTTP session.get should have been called only once
        assert client._session.get.call_count == 1

    def test_rate_limiting_sleep_called(self, mock_scryfall_response):
        """ScryfallClient must sleep between consecutive requests."""
        mock_resp = _make_response(200, mock_scryfall_response)
        client = ScryfallClient(cache=None, rate_limit_ms=50)
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        with patch("time.sleep") as mock_sleep:
            # Force first request to set _last_request_time in the past
            client._last_request_time = time.monotonic()
            client.lookup("Lightning Bolt")
            client._last_request_time = time.monotonic()
            client.lookup("Dark Ritual")

        # sleep must have been called at least once
        assert mock_sleep.called


# ---------------------------------------------------------------------------
# ScryfallCache
# ---------------------------------------------------------------------------


class TestScryfallCache:
    def test_set_and_get(self, tmp_path):
        cache = ScryfallCache(db_path=str(tmp_path / "test.db"), ttl_hours=1)
        cache.set("lightning bolt", {"name": "Lightning Bolt"})
        result = cache.get("lightning bolt")
        assert result == {"name": "Lightning Bolt"}

    def test_missing_key_returns_none(self, tmp_path):
        cache = ScryfallCache(db_path=str(tmp_path / "test.db"), ttl_hours=1)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_path):
        """An entry older than the TTL must be treated as a cache miss."""
        cache = ScryfallCache(db_path=str(tmp_path / "test.db"), ttl_hours=0)
        cache.set("bolt", {"name": "Lightning Bolt"})
        # TTL=0 means everything is immediately expired
        result = cache.get("bolt")
        assert result is None

    def test_purge_expired(self, tmp_path):
        cache = ScryfallCache(db_path=str(tmp_path / "test.db"), ttl_hours=0)
        cache.set("bolt", {"name": "Lightning Bolt"})
        removed = cache.purge_expired()
        assert removed >= 1

    def test_stats(self, tmp_path):
        cache = ScryfallCache(db_path=str(tmp_path / "test.db"), ttl_hours=24)
        cache.set("a", {"name": "A"})
        cache.set("b", {"name": "B"})
        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 0
