"""A tiny in-memory LRU cache with an optional time-to-live. Thread-safe (retrieval runs in worker threads).

Used for (1) query embeddings, (2) retrieval results and (3) finished answers, so a repeated question skips the
slow parts. In-process on purpose: with 1-2 gateway replicas that is enough, and it needs no extra service.
"""
import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class TTLCache:
    def __init__(self, maxsize: int = 256, ttl: float = 0.0, clock=time.monotonic):
        """maxsize <= 0 disables the cache. ttl <= 0 means entries never expire (only LRU eviction)."""
        self.maxsize, self.ttl, self._clock = maxsize, ttl, clock
        self._data: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.maxsize > 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        if not self.enabled:
            return default
        with self._lock:
            item = self._data.get(key)
            if item is not None:
                stored_at, value = item
                if self.ttl > 0 and self._clock() - stored_at > self.ttl:
                    del self._data[key]
                else:
                    self._data.move_to_end(key)
                    self.hits += 1
                    return value
            self.misses += 1
            return default

    def set(self, key: Hashable, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._data), "hits": self.hits, "misses": self.misses}

    def __len__(self) -> int:
        return len(self._data)


def normalize_query(text: str) -> str:
    """Cache key for a question: case, spacing and trailing punctuation do not change the meaning."""
    return " ".join(text.lower().split()).rstrip("?.! ")
