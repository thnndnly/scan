"""SQLite-backed cache for Scryfall API responses."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# DDL for the cache table
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scryfall_cache (
    cache_key   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO scryfall_cache (cache_key, payload, created_at)
VALUES (?, ?, ?);
"""

_SELECT_SQL = "SELECT payload, created_at FROM scryfall_cache WHERE cache_key = ?;"

_DELETE_EXPIRED_SQL = "DELETE FROM scryfall_cache WHERE created_at < ?;"


class ScryfallCache:
    """Thread-safe SQLite cache with TTL-based expiry for Scryfall responses.

    Args:
        db_path: Path to the SQLite database file.  Created automatically if
            it does not exist.
        ttl_hours: Number of hours after which a cached entry is considered
            stale.
    """

    def __init__(self, db_path: str = "data/scryfall_cache.db", ttl_hours: int = 24) -> None:
        self._db_path = db_path
        self._ttl_seconds: float = ttl_hours * 3600
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create the database file and table if they do not yet exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[dict]:
        """Retrieve a cached value by *key*.

        Returns ``None`` when the key is absent or the entry has expired.

        Args:
            key: Cache key (typically a lower-cased card name).

        Returns:
            Deserialised dict payload or ``None``.
        """
        with self._lock:
            try:
                conn = self._get_conn()
                row = conn.execute(_SELECT_SQL, (key,)).fetchone()
                if row is None:
                    return None
                age = time.time() - row["created_at"]
                if age > self._ttl_seconds:
                    conn.execute("DELETE FROM scryfall_cache WHERE cache_key = ?;", (key,))
                    conn.commit()
                    return None
                return json.loads(row["payload"])
            except Exception as exc:
                logger.warning("Cache get failed for key %r: %s", key, exc)
                return None

    def set(self, key: str, value: dict) -> None:
        """Store *value* in the cache under *key*.

        Overwrites any existing entry for the same key.

        Args:
            key: Cache key.
            value: JSON-serialisable dict to store.
        """
        with self._lock:
            try:
                conn = self._get_conn()
                payload = json.dumps(value)
                conn.execute(_INSERT_SQL, (key, payload, time.time()))
                conn.commit()
            except Exception as exc:
                logger.warning("Cache set failed for key %r: %s", key, exc)

    def delete(self, key: str) -> None:
        """Remove a single entry from the cache.

        Args:
            key: Cache key to remove.
        """
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute("DELETE FROM scryfall_cache WHERE cache_key = ?;", (key,))
                conn.commit()
            except Exception as exc:
                logger.warning("Cache delete failed for key %r: %s", key, exc)

    def purge_expired(self) -> int:
        """Delete all entries that have exceeded the TTL.

        Returns:
            Number of entries deleted.
        """
        with self._lock:
            try:
                conn = self._get_conn()
                cutoff = time.time() - self._ttl_seconds
                cur = conn.execute(_DELETE_EXPIRED_SQL, (cutoff,))
                conn.commit()
                count = cur.rowcount
                logger.info("Purged %d expired cache entries.", count)
                return count
            except Exception as exc:
                logger.warning("Cache purge_expired failed: %s", exc)
                return 0

    def stats(self) -> dict:
        """Return basic cache statistics.

        Returns:
            Dict with keys ``total_entries`` and ``expired_entries``.
        """
        with self._lock:
            try:
                conn = self._get_conn()
                total = conn.execute("SELECT COUNT(*) FROM scryfall_cache;").fetchone()[0]
                cutoff = time.time() - self._ttl_seconds
                expired = conn.execute(
                    "SELECT COUNT(*) FROM scryfall_cache WHERE created_at < ?;", (cutoff,)
                ).fetchone()[0]
                return {"total_entries": total, "expired_entries": expired}
            except Exception as exc:
                logger.warning("Cache stats failed: %s", exc)
                return {"total_entries": 0, "expired_entries": 0}

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
