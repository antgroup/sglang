"""Engram host table startup (page cache, THP collapse, layouts) and module conversions. CPU only."""

import mmap
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from sglang.srt.layers import engram
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@contextmanager
def _startup_sequence(*, huge_kb: int):
    """_HostTable's mmap / prefault / cudaHostRegister / smaps steps, faked so a CPU test
    walks the real startup sequence and records the page-cache drops and collapses."""
    raw = torch.zeros(8192, dtype=torch.uint8)
    fake_mmap = SimpleNamespace(
        mmap=Mock(return_value=Mock()),
        MAP_SHARED=1,
        MAP_PRIVATE=2,
        MAP_ANONYMOUS=4,
        PROT_READ=1,
        PROT_WRITE=2,
        MADV_HUGEPAGE=14,
        PAGESIZE=4096,
    )
    events = []
    runtime = SimpleNamespace(
        cudaHostRegister=Mock(side_effect=lambda *a: events.append("register") or 0)
    )
    with (
        patch.object(engram, "mmap", fake_mmap),
        patch.object(engram.torch, "frombuffer", return_value=raw),
        patch.object(engram.np, "frombuffer", return_value=np.zeros(8192, np.uint8)),
        patch.object(engram.torch.cuda, "cudart", return_value=runtime),
        patch.object(engram, "_page_cache_dropped", False),
        patch.object(
            engram,
            "drop_checkpoint_page_cache",
            side_effect=lambda: events.append("drop") or (48, 475 << 30),
        ),
        patch.object(
            engram._HostTable,
            "_collapse",
            side_effect=lambda: events.append("collapse"),
        ),
        patch.object(engram, "_huge_pages_backing", return_value=(8, huge_kb)),
    ):
        yield events


def _host_embedding(rows: int):
    embedding = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    nn.Module.__init__(embedding)
    embedding.rows = rows
    return embedding


class TestHostTableStartup(CustomTestCase):
    def test_fast_startup_keeps_the_checkpoint_page_cache(self):
        """With the startup subsystem a per-rank table must neither drop the checkpoint
        page cache nor force a THP collapse: on TP8 H20 the eight ranks spent minutes
        in posix_fadvise(DONTNEED) over the 475 GiB checkpoint and the DSpark draft
        then re-read it from disk. Off, the upstream drop + collapse stay."""
        for keep, expected in (
            (True, ["register"]),
            (False, ["drop", "register", "drop", "collapse"]),
        ):
            with self.subTest(keep=keep), _startup_sequence(huge_kb=0) as events:
                table = engram._HostTable(
                    "per_rank", 8192, "t", Mock(), keep_checkpoint_cache=keep
                )
                table.dirty = True
                table.finish_load("layer 0")
                self.assertEqual(events, expected)


class TestHostTableModuleConversion(CustomTestCase):
    def _embedding(self, host: bool):
        embedding = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
        nn.Module.__init__(embedding)
        backing = mmap.mmap(-1, 264)
        raw = torch.frombuffer(backing, dtype=torch.uint8)
        embedding.host_table = SimpleNamespace(backing=backing) if host else None
        embedding.weight = nn.Parameter(
            raw[:256].view(torch.float8_e4m3fn), requires_grad=False
        )
        embedding.scale = nn.Parameter(
            raw[256:].view(torch.float8_e8m0fnu), requires_grad=False
        )
        return embedding

    def test_parent_conversions_keep_host_table_parameters(self):
        """The gather kernels read the registered host mapping; a parent .half() / .to()
        must not convert the FP8 rows or move them off the mapping."""
        embedding = self._embedding(host=True)
        parent = nn.Module()
        parent.embedding = embedding
        parent.other = nn.Parameter(torch.ones(1))
        saved = [(p, p.data_ptr(), p.dtype) for p in embedding.parameters()]
        parent.half()
        parent.to(device="meta")
        self.assertEqual(parent.other.device.type, "meta")
        for current, (param, pointer, dtype) in zip(embedding.parameters(), saved):
            self.assertIs(current, param)
            self.assertEqual(current.data_ptr(), pointer)
            self.assertEqual(current.dtype, dtype)

    def test_device_table_still_converts(self):
        embedding = self._embedding(host=False)
        embedding.to(device="meta")
        self.assertTrue(all(p.device.type == "meta" for p in embedding.parameters()))


if __name__ == "__main__":
    unittest.main()
