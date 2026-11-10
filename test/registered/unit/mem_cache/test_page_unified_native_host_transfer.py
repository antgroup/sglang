"""CUDA byte-preserving transfers for native page_unified side pools."""

import unittest

import torch

from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo, IndexKeyLayout
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=50, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 16
DEVICE_PAGES = 6
HOST_PAGES = 8


def _device_buffers(shape, dtype=torch.uint8):
    return tuple(torch.empty(shape, dtype=dtype, device="cuda") for _ in range(2))


def _minimax_pool(pool_cls, page_size, dtype):
    return pool_cls(
        size=(DEVICE_PAGES - 1) * page_size,
        page_size=page_size,
        dtype=dtype,
        head_num=2,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=[0, 1],
        disable_value_sparse_layer_ids=[0, 1],
        device="cuda",
        start_layer=0,
        end_layer=2,
    )


def _minimax_main_host(host_cls, pool):
    main = pool.main_pool
    return host_cls(
        main.get_device_pool_infos()[0].buffer_info,
        2,
        0,
        main.page_size,
        "page_unified",
        device_capacity_tokens=main.size,
        start_layer=main.start_layer,
        end_layer=main.end_layer,
    )


def _indices(pages, page_size=PAGE_SIZE):
    return (
        torch.tensor(pages, dtype=torch.int64)[:, None] * page_size
        + torch.arange(page_size, dtype=torch.int64)
    ).flatten()


def _expected_serialized_page(host, native_layers):
    """Reference byte addresses from the native format, independent of its codec."""
    pieces = []
    if isinstance(host, DSAIndexerPoolHost):
        info = host._index_key_info
        slots = host.page_size // info.compress_ratio
        native_slots = info.kernel_page_size or slots
        tile = info.key_shuffle_tile
        for row in native_layers:
            for token in range(slots):
                page, offset = divmod(token, native_slots)
                base = page * native_slots * 132
                if tile:
                    key = row[
                        torch.tensor(
                            [
                                base
                                + (
                                    (offset // tile * (128 // tile) + channel // tile)
                                    * tile
                                    + offset % tile
                                )
                                * tile
                                + channel % tile
                                for channel in range(128)
                            ]
                        )
                    ]
                else:
                    key = row[base + offset * 128 : base + (offset + 1) * 128]
                scale = base + native_slots * 128 + offset * 4
                pieces.extend((key, row[scale : scale + 4]))
    else:
        pieces = native_layers
    return torch.cat(pieces)


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA host-transfer tests require a GPU"
)
class TestPageUnifiedIndexerTransfer(CustomTestCase):
    def _round_trip(self, host, buffers, backend, *, layer_ids=None):
        # HiCache copies run on dedicated streams in the controller. CUDA's
        # batched async host copies reject the legacy default stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._round_trip_on_stream(host, buffers, backend, layer_ids=layer_ids)
        stream.synchronize()

    def _round_trip_on_stream(self, host, buffers, backend, *, layer_ids=None):
        """The native payload, scales and padding survive page relocation."""
        try:
            page_size = host.page_size
            source_pages = [3, 1, 4]
            host_pages = [0, 4, 2]
            restored_pages = [5, 1, 3]
            host_indices_cpu = _indices(host_pages, page_size)
            restored_indices_cpu = _indices(restored_pages, page_size)
            host_indices = (
                host_indices_cpu.cuda() if backend == "kernel" else host_indices_cpu
            )
            restored_indices = (
                restored_indices_cpu.cuda()
                if backend == "kernel"
                else restored_indices_cpu
            )
            device_indices = _indices(source_pages, page_size).cuda()
            expected = []
            for layer, buffer in enumerate(buffers):
                byte_rows = buffer.view(torch.uint8).reshape(DEVICE_PAGES, -1)
                byte_rows.copy_(
                    (
                        torch.arange(
                            byte_rows.numel(), device="cuda", dtype=torch.int32
                        )
                        + layer * 43
                    )
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape_as(byte_rows)
                )
                expected.append(byte_rows[source_pages].cpu())
            backing = host.get_hybrid_pool_buffer()[0]
            self.assertTrue(backing.is_pinned())
            backing.fill_(239)
            host.backup_from_device_all_layer(
                None, host_indices, device_indices, backend
            )
            torch.cuda.synchronize()

            serialized = [
                host.get_data_page(page * page_size).clone() for page in host_pages
            ]
            for index, data in enumerate(serialized):
                self.assertTrue(
                    torch.equal(
                        data.view(torch.uint8),
                        _expected_serialized_page(
                            host, [layer[index] for layer in expected]
                        ),
                    )
                )
            self.assertTrue(torch.all(backing[6] == 239))
            self.assertEqual(
                sum(
                    component.byte_length
                    for component in host.storage_component_metadata
                ),
                serialized[0].nbytes,
            )
            backing.zero_()
            for page, data in zip(restored_pages, serialized):
                host.set_from_flat_data_page(page * page_size, data)
                self.assertTrue(
                    torch.equal(
                        host.get_data_page(page * page_size).view(torch.uint8),
                        data.view(torch.uint8),
                    )
                )
            self.assertTrue(torch.all(backing[0] == 0))
            self.assertEqual(
                host.get_dummy_flat_data_page().nbytes, serialized[0].nbytes
            )
            for buffer in buffers:
                buffer.zero_()
            for layer in range(len(buffers)) if layer_ids is None else layer_ids:
                host.load_to_device_per_layer(
                    None, restored_indices, device_indices, layer, backend
                )
            torch.cuda.synchronize()
            for buffer, data in zip(buffers, expected):
                byte_rows = buffer.view(torch.uint8).reshape(DEVICE_PAGES, -1)
                self.assertTrue(torch.equal(byte_rows[source_pages].cpu(), data))
                self.assertEqual(torch.count_nonzero(byte_rows[[0, 2, 5]]).item(), 0)
        finally:
            torch.cuda.synchronize()
            host.destroy()

    def test_dsa_compact_and_layer_sharded_pages(self):
        for backend in ("kernel", "direct"):
            for sharded in (False, True):
                with (
                    self.subTest(backend=backend, sharded=sharded),
                    host_memory_budget_scope(1 << 20),
                ):
                    buffers = _device_buffers((DEVICE_PAGES, PAGE_SIZE * 132))
                    info = IndexKeyBufferInfo(
                        page_size=PAGE_SIZE,
                        buffers=buffers,
                        compress_ratio=1,
                        key_layout=IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES,
                    )
                    host = DSAIndexerPoolHost(
                        info,
                        layer_mapping={1: 0, 3: 1},
                        target_device_layer_num=4,
                        num_host_pages=HOST_PAGES,
                        transfer_page_size=PAGE_SIZE,
                        layout="page_unified",
                        buffer_device="cuda",
                        dtype=torch.bfloat16,
                        layer_sharded=sharded,
                    )
                    self._round_trip(host, buffers, backend, layer_ids=[1, 3])

    def test_minimax_bfloat16_k_only_pages(self):
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
        from sglang.srt.mem_cache.pool_host.mha import (
            MHATokenToKOnlyPoolHost,
            MHATokenToKVPoolHost,
        )

        for backend in ("kernel", "direct"):
            with self.subTest(backend=backend), host_memory_budget_scope(1 << 20):
                pool = _minimax_pool(MiniMaxSparseKVPool, PAGE_SIZE, torch.bfloat16)
                anchor = _minimax_main_host(MHATokenToKVPoolHost, pool)
                try:
                    host = MHATokenToKOnlyPoolHost(
                        pool.index_k_pool.get_device_pool_infos()[0].buffer_info,
                        anchor,
                        "page_unified",
                    )
                    self._round_trip(host, pool.index_k_pool.k_buffer, backend)
                finally:
                    anchor.destroy()


if __name__ == "__main__":
    unittest.main()
