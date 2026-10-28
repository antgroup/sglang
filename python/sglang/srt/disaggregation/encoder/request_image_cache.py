"""Bounded, request-scoped image snapshots shared by /encode and /preprocess_meta.

This is not a URL cache: the scope identifies one PD bootstrap attempt. Keep
compressed immutable bytes only, never decoded images or GPU tensors. The cache
belongs to one EncoderPreprocessor and is accessed by its I/O executor threads.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Callable

import msgspec

from sglang.srt.multimodal.cache.identity import MediaSnapshot


class _Entry(msgspec.Struct):
    # concurrent.futures.Future of the snapshot; waiters of the same key join it.
    future: Any
    expires_at: float = float("inf")
    size: int = 0


class RequestImageCache:
    def __init__(self, max_bytes: int, *, ttl: float = 60.0, capacity: int = 4096):
        self.max_bytes = max(0, max_bytes)
        self.ttl = ttl
        self.capacity = capacity
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def eligible(scope, url) -> bool:
        return (
            isinstance(scope, str)
            and len(scope) == 64
            and all(c in "0123456789abcdef" for c in scope)
            and isinstance(url, str)
            and url.startswith(("http://", "https://"))
        )

    def _remove(self, key):
        entry = self._entries.pop(key)
        self._bytes -= entry.size

    def _prune(self, now):
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                self._remove(key)
        # Do not evict in-flight owners: their other PD consumer must join
        # the same download, not start another one while this one is active.
        for key, entry in list(self._entries.items()):
            if self._bytes <= self.max_bytes:
                break
            if entry.expires_at != float("inf"):
                self._remove(key)

    def get(self, scope, url, loader: Callable[[str], MediaSnapshot]) -> MediaSnapshot:
        if not self.max_bytes or self.ttl <= 0 or not self.eligible(scope, url):
            return loader(url)
        key = (scope, hashlib.sha256(url.encode("utf-8", "surrogatepass")).digest())
        owner = False
        with self._lock:
            self._prune(time.monotonic())
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            else:
                # Bound key/Future storage as well as payload bytes. At capacity
                # prefer evicting completed entries; otherwise bypass retention.
                if len(self._entries) >= self.capacity:
                    for old_key, old in list(self._entries.items()):
                        if old.expires_at != float("inf"):
                            self._remove(old_key)
                            break
                if len(self._entries) < self.capacity:
                    entry = _Entry(Future())
                    self._entries[key] = entry
                    owner = True
        if entry is None:
            return loader(url)
        if not owner:
            return entry.future.result()
        try:
            snapshot = loader(url)
        except BaseException as exc:
            with self._lock:
                self._remove(key)
            entry.future.set_exception(exc)
            raise
        with self._lock:
            # Only immutable raw bytes are retained. An oversized image still
            # serves existing waiters, but cannot exceed the retained-byte cap.
            size = len(snapshot.data) if isinstance(snapshot.data, bytes) else 0
            if size and size <= self.max_bytes:
                entry.size = size
                entry.expires_at = time.monotonic() + self.ttl
                self._bytes += size
                self._prune(time.monotonic())
            else:
                self._remove(key)
        entry.future.set_result(snapshot)
        return snapshot
