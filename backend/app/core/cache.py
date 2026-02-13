# app/core/cache.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple
import time
import hashlib
import json

@dataclass
class CacheEntry:
    value: Any
    expires_at: float

class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 2048):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        ent = self._store.get(key)
        if not ent:
            return None
        if ent.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return ent.value

    def set(self, key: str, value: Any) -> None:
        # simple eviction: drop oldest-ish by popping arbitrary keys if too big
        if len(self._store) >= self.max_items:
            for _ in range(max(1, self.max_items // 10)):
                try:
                    self._store.pop(next(iter(self._store)))
                except StopIteration:
                    break

        self._store[key] = CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)

def stable_key(*parts: Any) -> str:
    """
    Creates a stable cache key from arbitrary JSON-serializable inputs.
    """
    raw = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
