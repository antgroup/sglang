# SPDX-License-Identifier: Apache-2.0
"""Shared-memory (L2) embedding store backend.

Entries are files under a tmpfs directory, so all encoder processes of one
host (or container) share one cache without a transfer engine or master service. The
controller hands in pinned host pointers, so plain memcpy is enough.

The directory is LRU-bounded across processes. Recency (an OrderedDict) and
used bytes live in each process and only see its own reads and writes; a
directory scan reconciles them with the other processes. It runs when the index
is unknown (first put, failed write), when the in-process LRU cannot free
enough, and every `_resync_interval()` bytes this process wrote, so the
directory overshoots the cap by at most N x interval for N writers.
"""

import ctypes
import logging
import os
import struct
import tempfile
from collections import OrderedDict
from typing import List, Optional

import psutil

from sglang.srt.mem_cache.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

_CACHE_DIR = "/dev/shm/sglang_emb_cache"
# Low watermark after an eviction, so the next puts stay on the O(1) path. The
# surplus is byte-capped: 15% of a 32GB pool is one unlink wave of ~4.8GB that
# blocks the shared IO thread; 256MB is ~16 full-res images.
_EVICT_LOW_WATER_RATIO = 0.85
_EVICT_MAX_SURPLUS_BYTES = 256 * 1024 * 1024
# Resync interval floor; it scales as cap/64, so a 384GB pool rescans every
# ~6GB written (tens of ms) and 4 writers overshoot the cap by at most ~6%.
_RESYNC_MIN_BYTES = 256 * 1024 * 1024
# Entry files are named by get_key. The directory may hold other shm users'
# files (NCCL, torch), so nothing without these prefixes is counted or evicted.
_ENTRY_PREFIX = "emb_"
# In-flight writes are "tmp_emb_<writer pid>_<random>"; the pid lets a rescan
# reclaim the temporaries of a writer that died before its rename.
_TMP_PREFIX = "tmp_emb_"

# Entry file = header + payload. The header pins the writing model and the
# exact payload length, so a stale, foreign or truncated file reads as a miss.
_HEADER = struct.Struct("<8s16sQ")
_MAGIC = b"SGLEMB01"
_UNBOUND_FINGERPRINT = bytes(16)


def _tmp_writer_alive(name: str) -> bool:
    try:
        pid = int(name[len(_TMP_PREFIX) :].split("_", 1)[0])
    except ValueError:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _cgroup_dir(mount: str, path: str) -> str:
    # A namespaced or bind-mounted hierarchy shows the own cgroup at the mount
    # root, while /proc/self/cgroup may still name the host-side path.
    directory = os.path.normpath(os.path.join(mount, path.lstrip("/")))
    inside = os.path.commonpath([mount, directory]) == mount
    return directory if inside and os.path.isdir(directory) else mount


def _cgroup_memory_limits(proc_root: str, cgroup_root: str) -> List[int]:
    """Memory limits of this process's cgroup and its visible ancestors."""
    root = os.path.normpath(cgroup_root)
    limits = []
    membership = _read_text(os.path.join(proc_root, "self/cgroup")) or ""
    for line in membership.splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            # cgroup v2 limits are per level, so walk up to the mount root.
            directory = _cgroup_dir(root, path)
            while True:
                for name in ("memory.max", "memory.high"):
                    value = _read_text(os.path.join(directory, name))
                    if value is not None and value.strip() != "max":
                        limits.append(int(value))
                if directory == root:
                    break
                directory = os.path.dirname(directory)
        elif "memory" in controllers.split(","):
            # cgroup v1 hierarchical_memory_limit already covers the ancestors.
            directory = _cgroup_dir(os.path.join(root, "memory"), path)
            stat = _read_text(os.path.join(directory, "memory.stat")) or ""
            for stat_line in stat.splitlines():
                key, value = stat_line.split()
                if key == "hierarchical_memory_limit":
                    limits.append(int(value))
    return limits


def _default_cap_bytes(
    cache_dir: str, *, proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup"
) -> int:
    # tmpfs pages are charged to the writer's memory cgroup and cannot be reclaimed
    # without swap. Half of the fs leaves room for NCCL and torch segments; a quarter
    # of the memory limit (arbitrary) leaves the rest to the processes sharing it.
    vfs = os.statvfs(cache_dir)
    fs_bytes = vfs.f_frsize * vfs.f_blocks
    memory_limit = min(
        [psutil.virtual_memory().total, *_cgroup_memory_limits(proc_root, cgroup_root)]
    )
    return min(fs_bytes // 2, memory_limit // 4)


class ShmEmbeddingStore(EmbeddingStore):
    def __init__(self, cache_dir: str = _CACHE_DIR, max_bytes: Optional[int] = None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_bytes = (
            _default_cap_bytes(self.cache_dir) if max_bytes is None else max_bytes
        )
        # None means unknown (first put, or a failed write invalidated it).
        self._used_bytes: Optional[int] = None
        # Entry file name -> size, oldest first. Rebuilt on every directory scan.
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._written_since_resync = 0
        self._fingerprint = _UNBOUND_FINGERPRINT
        self._key_prefix = f"{_ENTRY_PREFIX}{_UNBOUND_FINGERPRINT[:8].hex()}_"
        logger.info(
            "Shm embedding store at %s, cap %d MB",
            self.cache_dir,
            self.max_bytes // (1024 * 1024),
        )

    def bind_model(self, fingerprint: bytes) -> None:
        self._fingerprint = fingerprint
        self._key_prefix = f"{_ENTRY_PREFIX}{fingerprint[:8].hex()}_"

    def get_key(self, mm_hash: str) -> str:
        return f"{self._key_prefix}{mm_hash}"

    def _path(self, mm_hash: str) -> str:
        return os.path.join(self.cache_dir, self.get_key(mm_hash))

    def _header_matches(self, f, nbytes: Optional[int]) -> bool:
        """Whether an open entry file was written by this model; with nbytes,
        also whether it holds exactly that many payload bytes."""
        size = os.fstat(f.fileno()).st_size
        if size < _HEADER.size:
            return False
        magic, fingerprint, payload = _HEADER.unpack(f.read(_HEADER.size))
        if magic != _MAGIC or fingerprint != self._fingerprint:
            return False
        return nbytes is None or (payload == nbytes and size == _HEADER.size + nbytes)

    def _entry_matches(self, path: str, nbytes: Optional[int]) -> bool:
        try:
            with open(path, "rb") as f:
                return self._header_matches(f, nbytes)
        except OSError:
            return False

    # ===== Cross-process LRU budget =====

    def _scan_dir(self):
        """Return (used_bytes, evictable entries as (mtime, size, path))."""
        used = 0
        entries = []
        with os.scandir(self.cache_dir) as it:
            for entry in it:
                is_entry = entry.name.startswith(_ENTRY_PREFIX)
                is_tmp = entry.name.startswith(_TMP_PREFIX)
                if not is_entry and not is_tmp:
                    continue
                if is_tmp and not _tmp_writer_alive(entry.name):
                    _unlink_quietly(entry.path)
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                used += st.st_size
                # A live writer's temporary is counted but never evicted.
                if is_entry:
                    entries.append((st.st_mtime, st.st_size, entry.path))
        return used, entries

    def _target_used(self, need: int) -> int:
        extra_free = min(
            int(self.max_bytes * (1.0 - _EVICT_LOW_WATER_RATIO)),
            _EVICT_MAX_SURPLUS_BYTES,
        )
        return self.max_bytes - max(need, extra_free)

    def _resync_interval(self) -> int:
        return max(_RESYNC_MIN_BYTES, self.max_bytes // 64)

    def _touch(self, name: str) -> None:
        if name in self._lru:
            self._lru.move_to_end(name)

    def _track_write(self, name: str, nbytes: int) -> None:
        prev = self._lru.pop(name, None)
        if prev is not None and self._used_bytes is not None:
            self._used_bytes = max(0, self._used_bytes - prev)
        self._lru[name] = nbytes
        if self._used_bytes is not None:
            self._used_bytes += nbytes
        self._written_since_resync += nbytes

    def _evict_from_lru(self, target_used: int) -> None:
        while self._lru and self._used_bytes > target_used:
            name, size = self._lru.popitem(last=False)
            _unlink_quietly(os.path.join(self.cache_dir, name))
            self._used_bytes = max(0, self._used_bytes - size)

    def _resync_from_disk(self) -> None:
        used, entries = self._scan_dir()
        self._used_bytes = used
        self._lru.clear()
        for _, size, path in sorted(entries):
            self._lru[os.path.basename(path)] = size
        self._written_since_resync = 0

    def _fits(self, need: int) -> bool:
        return self._used_bytes + need <= self.max_bytes

    def _ensure_budget(self, need: int) -> bool:
        """Evict LRU entries until `need` bytes fit; scan the directory only when
        the index is unknown or stale, or the in-process LRU cannot free enough."""
        if need > self.max_bytes:
            return False
        try:
            if (
                self._used_bytes is None
                or self._written_since_resync >= self._resync_interval()
            ):
                self._resync_from_disk()
            if self._fits(need):
                return True
            target_used = self._target_used(need)
            self._evict_from_lru(target_used)
            if self._fits(need):
                return True
            # The directory holds files this index never saw (other processes).
            self._resync_from_disk()
            self._evict_from_lru(target_used)
            return self._fits(need)
        except OSError:
            return False

    # ===== Entry files =====

    def _write_atomic(self, path: str, chunks: List[bytes]) -> bool:
        # Rename into place so concurrent readers never see a partial file.
        try:
            fd, tmp = tempfile.mkstemp(
                dir=self.cache_dir, prefix=f"{_TMP_PREFIX}{os.getpid()}_"
            )
        except OSError as e:
            logger.warning("shm emb put failed for %s: %s", path, e)
            return False
        renamed = False
        try:
            with os.fdopen(fd, "wb") as f:
                for c in chunks:
                    f.write(c)
            os.rename(tmp, path)
            renamed = True
            return True
        except OSError as e:
            logger.warning("shm emb put failed for %s: %s", path, e)
            return False
        finally:
            if not renamed:
                _unlink_quietly(tmp)

    def _put_new(self, path: str, chunks: List[bytes], nbytes: int) -> bool:
        file_bytes = _HEADER.size + nbytes
        if not self._ensure_budget(file_bytes):
            return False
        header = _HEADER.pack(_MAGIC, self._fingerprint, nbytes)
        if not self._write_atomic(path, [header, *chunks]):
            # Another process may have filled the directory; resync next put.
            self._used_bytes = None
            return False
        self._track_write(os.path.basename(path), file_bytes)
        return True

    def _read_into(self, path: str, ptrs: List[int], sizes: List[int]) -> bool:
        try:
            with open(path, "rb") as f:
                if not self._header_matches(f, sum(sizes)):
                    return False
                for ptr, size in zip(ptrs, sizes):
                    data = f.read(size)
                    if len(data) != size:
                        return False
                    ctypes.memmove(ptr, data, size)
        except OSError:
            return False
        self._touch(os.path.basename(path))
        try:
            # Disk mtime is the recency the next directory scan sees.
            os.utime(path)
        except OSError:
            pass
        return True

    # ===== EmbeddingStore API =====

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        return self.batch_get_into_multi_buffers(
            hashes, [[p] for p in ptrs], [[s] for s in sizes]
        )

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        return self.batch_put_from_multi_buffers(
            hashes, [[p] for p in ptrs], [[s] for s in sizes]
        )

    def batch_get_into_multi_buffers(
        self, hashes: List[str], ptrs: List[List[int]], sizes: List[List[int]]
    ) -> List[bool]:
        return [
            self._read_into(self._path(h), pl, sl)
            for h, pl, sl in zip(hashes, ptrs, sizes)
        ]

    def batch_put_from_multi_buffers(
        self, hashes: List[str], ptrs: List[List[int]], sizes: List[List[int]]
    ) -> List[bool]:
        pending = []
        out = [True] * len(hashes)
        for i, (h, pl, sl) in enumerate(zip(hashes, ptrs, sizes)):
            path = self._path(h)
            nbytes = sum(sl)
            if self._entry_matches(path, nbytes):
                self._touch(os.path.basename(path))
                continue
            chunks = [ctypes.string_at(p, s) for p, s in zip(pl, sl)]
            pending.append((i, path, chunks, nbytes))
            out[i] = False
        if pending:
            # One eviction pass for the largest item; the watermark covers the rest.
            self._ensure_budget(max(nbytes for _, _, _, nbytes in pending))
        for i, path, chunks, nbytes in pending:
            out[i] = self._put_new(path, chunks, nbytes)
        return out

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        return [self._entry_matches(self._path(h), None) for h in hashes]
