"""Unit tests for srt/mem_cache/storage/shm/shm_embedding_store.py: the LRU cap
across processes, ownership of directory files, and model-scoped entries."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import ctypes
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.embedding_store import (
    EmbeddingStoreFactory,
    model_fingerprint,
)
from sglang.srt.mem_cache.storage.shm import shm_embedding_store as shm_store
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import CustomTestCase

CAP = 1024 * 1024
D = 400_000


def _fingerprint(
    model_path="/models/a", revision=None, dtype=torch.bfloat16, dims=None
):
    return model_fingerprint(
        model_path=model_path,
        revision=revision,
        dtype=dtype,
        dims_by_modality=dims or {"IMAGE": 4096, "AUDIO": 2048},
    )


def _buffer(nbytes, byte=0x78):
    buf = ctypes.create_string_buffer(bytes([byte]) * nbytes, nbytes)
    return buf, ctypes.addressof(buf)


def _put(store, key, nbytes, byte=0x78):
    _buf, ptr = _buffer(nbytes, byte)
    return store.batch_put_from_multi_buffers([key], [[ptr]], [[nbytes]]) == [True]


def _get(store, key, nbytes):
    out = ctypes.create_string_buffer(nbytes)
    hit = store.batch_get_into_multi_buffers(
        [key], [[ctypes.addressof(out)]], [[nbytes]]
    )
    return hit == [True], out.raw


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _count_scans(store):
    scans = {"n": 0}
    scan_dir = store._scan_dir

    def counted():
        scans["n"] += 1
        return scan_dir()

    store._scan_dir = counted
    return scans


class TestShmEmbeddingStore(CustomTestCase):
    def test_factory_rejects_shm_without_epd_selection(self):
        """Python/config-file backend selection must not bypass the CLI gate."""
        with get_flags().dsv41_sm90.override(subsystems=frozenset()):
            with self.assertRaisesRegex(
                ValueError, "Unknown embedding store backend 'shm'.*\\['mooncake'\\]"
            ):
                EmbeddingStoreFactory.create_backend("shm")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.store = shm_store.ShmEmbeddingStore(cache_dir=self.dir, max_bytes=CAP)

    def tearDown(self):
        self._tmp.cleanup()

    def _ls(self):
        return sorted(os.listdir(self.dir))

    def _used(self):
        return sum(os.path.getsize(os.path.join(self.dir, n)) for n in self._ls())

    def test_hit_refreshes_recency(self):
        """A multi-buffer read returns the chunks in order and makes the entry
        the most recent, so the next overflow evicts the other one."""
        half = D // 2
        _a1, p1 = _buffer(half, 0x11)
        _a2, p2 = _buffer(half, 0x22)
        put = self.store.batch_put_from_multi_buffers(["a"], [[p1, p2]], [[half, half]])
        self.assertEqual(put, [True])
        time.sleep(0.02)
        self.assertTrue(_put(self.store, "b", D))
        time.sleep(0.02)
        o1 = ctypes.create_string_buffer(half)
        o2 = ctypes.create_string_buffer(half)
        hit = self.store.batch_get_into_multi_buffers(
            ["a"], [[ctypes.addressof(o1), ctypes.addressof(o2)]], [[half, half]]
        )
        self.assertEqual(hit, [True])
        self.assertEqual((o1.raw, o2.raw), (b"\x11" * half, b"\x22" * half))
        time.sleep(0.02)
        self.assertTrue(_put(self.store, "c", D))
        self.assertIn(self.store.get_key("a"), self._ls())
        self.assertNotIn(self.store.get_key("b"), self._ls())

    def test_eviction_spares_files_the_store_does_not_own(self):
        """The cache directory may be shared with other shm users (NCCL, MPI,
        torch); eviction must neither count nor delete their files."""
        foreign = ["nccl-shm-abc", "psm_1234", "torch_5678_0"]
        old = time.time() - 3600
        for name in foreign:
            path = os.path.join(self.dir, name)
            with open(path, "wb") as f:
                f.write(b"f" * D)
            os.utime(path, (old, old))
        for key in ["a", "b", "c"]:
            self.assertTrue(_put(self.store, key, D))
        names = self._ls()
        self.assertTrue(all(name in names for name in foreign))
        self.assertIn(self.store.get_key("c"), names)

    def test_write_failure_leaves_no_temporary(self):
        """A put that fails after creating its temporary, for any exception type,
        must not leave the temporary to count against the cap forever."""
        with self.assertRaises(ValueError):
            _put(self.store, "bad\x00key", D)
        self.assertEqual(self._ls(), [])

    def test_crashed_writer_temporaries_are_reclaimed(self):
        """Temporaries of a writer that died before its rename must not fill the
        cap; a live writer's temporary must not be taken from under it."""
        dead = _dead_pid()
        for i in range(2):
            with open(os.path.join(self.dir, f"tmp_emb_{dead}_{i}"), "wb") as f:
                f.write(b"t" * D)
        live = f"tmp_emb_{os.getpid()}_live"
        with open(os.path.join(self.dir, live), "wb") as f:
            f.write(b"t" * 1000)
        self.assertTrue(_put(self.store, "a", D))
        self.assertTrue(_put(self.store, "b", D))
        expected = sorted([self.store.get_key("a"), self.store.get_key("b"), live])
        self.assertEqual(self._ls(), expected)

    def test_entry_of_a_different_length_is_a_miss(self):
        """An entry is a hit only when it holds exactly the bytes the reader
        expects; a longer stale entry must not be returned as a prefix."""
        self.assertTrue(_put(self.store, "42", 2048, 0xAA))
        self.assertFalse(_get(self.store, "42", 1024)[0])

    def test_entries_are_scoped_to_the_model(self):
        """The content hash covers the preprocessed input, not the encoder that
        embedded it: another model's entry for the same hash must be a miss."""
        model_a = shm_store.ShmEmbeddingStore(cache_dir=self.dir, max_bytes=CAP)
        model_b = shm_store.ShmEmbeddingStore(cache_dir=self.dir, max_bytes=CAP)
        model_a.bind_model(_fingerprint(dtype=torch.bfloat16))
        model_b.bind_model(_fingerprint(dtype=torch.float16))
        self.assertTrue(_put(model_a, "42", 1024, 0xAA))
        self.assertEqual(model_b.batch_is_exist(["42"]), [False])
        self.assertFalse(_get(model_b, "42", 1024)[0])
        self.assertEqual(_get(model_a, "42", 1024), (True, b"\xaa" * 1024))

    def test_put_replaces_an_entry_this_model_cannot_read(self):
        """An existing file under the key that fails validation (an older format,
        a torn copy) must be overwritten, not kept as a permanent miss."""
        with open(os.path.join(self.dir, self.store.get_key("43")), "wb") as f:
            f.write(b"\xaa" * 1024)
        self.assertTrue(_put(self.store, "43", 1024, 0xCC))
        self.assertEqual(_get(self.store, "43", 1024), (True, b"\xcc" * 1024))

    def test_model_fingerprint_covers_weights_dtype_and_widths(self):
        base = _fingerprint()
        self.assertEqual(base, _fingerprint())
        self.assertNotEqual(base, _fingerprint(model_path="/models/other"))
        self.assertNotEqual(base, _fingerprint(revision="v2"))
        self.assertNotEqual(base, _fingerprint(dtype=torch.float16))
        self.assertNotEqual(base, _fingerprint(dims={"IMAGE": 4096, "AUDIO": 1024}))

    def test_oversized_item_refused_without_wipeout(self):
        self.assertTrue(_put(self.store, "a", D))
        self.assertFalse(_put(self.store, "huge", 2 * CAP))
        self.assertIn(self.store.get_key("a"), self._ls())

    def test_overflow_evicts_without_rescanning(self):
        """Puts under the cap and evictions of this process's own entries stay
        on the in-process index; the directory is not rescanned per put."""
        item = 50_000
        self.assertTrue(_put(self.store, "f0", item))
        scans = _count_scans(self.store)
        for i in range(1, CAP // item + 3):
            self.assertTrue(_put(self.store, f"f{i}", item))
        self.assertEqual(scans["n"], 0)
        self.assertLessEqual(self._used(), CAP)

    def test_eviction_surplus_is_byte_capped(self):
        # 15% of 1MB is ~157KB (3 files); a 100KB surplus cap frees only 2.
        item = 50_000
        with mock.patch.object(shm_store, "_EVICT_MAX_SURPLUS_BYTES", 100_000):
            n = CAP // item
            for i in range(n):
                self.assertTrue(_put(self.store, f"f{i}", item))
            self.assertTrue(_put(self.store, "new", item))
        self.assertEqual(len(self._ls()), n - 2 + 1)

    def test_resync_when_memory_lru_cannot_cover_disk(self):
        """When the directory is full of entries this process's index does not
        hold (other writers), a put rescans and evicts instead of refusing."""
        item = 50_000
        for i in range(CAP // item):
            self.assertTrue(_put(self.store, f"f{i}", item))
        self.store._lru.clear()
        self.store._used_bytes = CAP
        self.assertTrue(_put(self.store, "foreign", item))
        self.assertIn(self.store.get_key("foreign"), self._ls())

    def test_shared_dir_cap_holds_across_processes(self):
        """N stores on one directory must not let it grow to ~N x cap: each
        store's used-bytes counter only sees its own writes."""
        item = 50_000
        interval = 100_000
        stores = [
            shm_store.ShmEmbeddingStore(cache_dir=self.dir, max_bytes=CAP)
            for _ in range(4)
        ]
        # Each store can miss at most one interval of every other store's writes.
        slack = len(stores) * interval + item
        peak = 0
        with mock.patch.object(shm_store, "_RESYNC_MIN_BYTES", interval):
            for i in range(160):
                self.assertTrue(_put(stores[i % len(stores)], f"k{i}", item))
                used = self._used()
                peak = max(peak, used)
                self.assertLessEqual(used, CAP + slack, f"dir grew at put {i}")
        self.assertGreater(peak, CAP * 0.9)

    def test_default_cap_is_bounded_by_the_memory_limit(self):
        """tmpfs pages are charged to the memory cgroup, so the cap is a quarter
        of the tightest visible limit when that is below half of the filesystem."""
        vfs = os.statvfs(self.dir)
        fs_half = vfs.f_frsize * vfs.f_blocks // 2
        limit = 64 * 1024 * 1024
        cases = {
            "v2 limit on a parent": (
                "0::/pod/ctr\n",
                {"pod/memory.max": str(limit), "pod/ctr/memory.max": "max"},
                limit // 4,
            ),
            "v1 mount of the own cgroup": (
                "11:memory:/kubepods/pod/ctr\n",
                {"memory/memory.stat": f"cache 0\nhierarchical_memory_limit {limit}\n"},
                limit // 4,
            ),
            "unlimited": ("0::/\n", {"memory.max": "max"}, fs_half),
        }
        host = SimpleNamespace(total=1 << 62)
        with mock.patch.object(shm_store.psutil, "virtual_memory", return_value=host):
            for name, (membership, files, expected) in cases.items():
                with self.subTest(name), tempfile.TemporaryDirectory() as root:
                    proc = os.path.join(root, "proc")
                    cgroup = os.path.join(root, "cgroup")
                    os.makedirs(os.path.join(proc, "self"))
                    with open(os.path.join(proc, "self/cgroup"), "w") as f:
                        f.write(membership)
                    for rel, text in files.items():
                        path = os.path.join(cgroup, rel)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w") as f:
                            f.write(text)
                    cap = shm_store._default_cap_bytes(
                        self.dir, proc_root=proc, cgroup_root=cgroup
                    )
                    self.assertEqual(cap, expected)


if __name__ == "__main__":
    unittest.main()
