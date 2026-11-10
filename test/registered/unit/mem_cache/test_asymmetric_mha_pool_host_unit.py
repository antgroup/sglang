"""Unit tests for asymmetric MHA host KV pool transfer dispatch."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.device_pool_info import (
    KVBufferLayout,
    KVBuffers,
    KVScaleBuffers,
    KVScaleLayout,
    MHABufferInfo,
)
from sglang.srt.mem_cache.hicache_key_scheme import plan_unified_kv
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
    NativePageUnifiedMHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.platforms.interface import PlatformCapabilities
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cpu_host_allocation(test):
    return mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {"cpu": lambda dims, **kw: torch.empty(dims, dtype=kw["dtype"])},
    )(test)


def _mha_info(k, v, *, page_size, head_num, head_dim, layout=KVBufferLayout.NHD, **kw):
    return MHABufferInfo(
        buffers=KVBuffers(
            k=k,
            v=v,
            layout=layout,
            k_data_ptrs=torch.tensor([b.data_ptr() for b in k], dtype=torch.uint64),
            v_data_ptrs=torch.tensor([b.data_ptr() for b in v], dtype=torch.uint64),
        ),
        page_size=page_size,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=kw.pop("v_head_dim", head_dim),
        dtype=kw.pop("dtype", torch.uint8),
        device=torch.device("cpu"),
        **kw,
    )


def _unified_host(info, *, end_layer, start_layer=0, capacity=8, budget=1 << 20, **kw):
    with host_memory_budget_scope(budget):
        return get_mha_host_pool_cls(info, layout="page_unified")(
            info,
            2,
            0,
            info.page_size,
            "page_unified",
            pin_memory=False,
            device_capacity_tokens=capacity,
            start_layer=start_layer,
            end_layer=end_layer,
            **kw,
        )


def _make_host(layout: str) -> AsymmetricMHATokenToKVPoolHost:
    host = AsymmetricMHATokenToKVPoolHost.__new__(AsymmetricMHATokenToKVPoolHost)
    host._transfer_device = torch.device("cpu")
    host.layout = layout
    host.page_size = 2
    host.layer_num = 3
    host.target_layer_num = 3
    host.head_num = 2
    host.head_dim = 4
    host.v_head_dim = 6
    host.dtype = torch.float16

    if layout == "page_first":
        k_dims = (8, host.layer_num, host.head_num, host.head_dim)
        v_dims = (8, host.layer_num, host.head_num, host.v_head_dim)
    elif layout == "page_first_direct":
        host.page_num = 4
        k_dims = (
            host.page_num,
            host.layer_num,
            host.page_size,
            host.head_num,
            host.head_dim,
        )
        v_dims = (
            host.page_num,
            host.layer_num,
            host.page_size,
            host.head_num,
            host.v_head_dim,
        )
    else:
        raise ValueError(f"Unsupported test layout: {layout}")

    host.kv_buffer = (torch.empty(k_dims), torch.empty(v_dims))
    host.can_use_write_back_jit = False
    return host


def _make_device_pool(host: AsymmetricMHATokenToKVPoolHost) -> SimpleNamespace:
    size = 8
    k_buffer = [
        torch.empty(size, host.head_num, host.head_dim) for _ in range(host.layer_num)
    ]
    v_buffer = [
        torch.empty(size, host.head_num, host.v_head_dim) for _ in range(host.layer_num)
    ]
    host.device_k_buffers = tuple(k_buffer)
    host.device_v_buffers = tuple(v_buffer)
    host.device_k_data_ptrs = torch.tensor(
        [x.data_ptr() for x in k_buffer], dtype=torch.uint64
    )
    host.device_v_data_ptrs = torch.tensor(
        [x.data_ptr() for x in v_buffer], dtype=torch.uint64
    )
    host.device_pool = SimpleNamespace(
        layer_num=host.layer_num,
        layer_shard_enabled=False,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=host.device_k_data_ptrs,
        v_data_ptrs=host.device_v_data_ptrs,
    )
    return host.device_pool


class TestAsymmetricMHATokenToKVPoolHost(CustomTestCase):
    @_cpu_host_allocation
    def test_native_unified_pages_preserve_all_components_and_strides(self):
        """A stored native page must restore payloads and scales byte-for-byte.

        K/V widths, scale dtype, physical page axes and row gaps cannot be
        inferred from a logical head dimension. Dropping any of them corrupts
        FP4/MXFP8 or restores a neighbouring layer in a strided device pool.
        """
        formats = (
            (KVBufferLayout.NHD, None, False),
            (KVBufferLayout.NHD, KVScaleLayout.TOKEN, True),
            (KVBufferLayout.NHD, KVScaleLayout.MXFP8_INTERLEAVED, False),
            (KVBufferLayout.HND, None, False),
            (KVBufferLayout.PAGED_NHD, None, False),
            (KVBufferLayout.VECTORIZED_5D, None, False),
        )
        for layout, scale_layout, packed_fp4 in formats:
            with self.subTest(layout=layout, scale_layout=scale_layout):
                layers, page_size, pages = 3, 2, 6

                def buffers(width, *, scales=False):
                    dtype = torch.float32 if scales else torch.uint8
                    if scales and scale_layout == KVScaleLayout.MXFP8_INTERLEAVED:
                        shape = (pages, 2, 1, page_size, width)
                    elif layout == KVBufferLayout.NHD or scales:
                        shape = (pages * page_size, 2, width)
                    elif layout == KVBufferLayout.HND:
                        shape = (pages, 2, page_size, width)
                    elif layout == KVBufferLayout.PAGED_NHD:
                        shape = (pages, page_size, 2, width)
                    else:
                        shape = (pages, 2, width // 2, page_size, 2)
                    # Different component widths yield different row strides.
                    backing = torch.empty((shape[0], layers, *shape[1:]), dtype=dtype)
                    backing.view(torch.uint8).random_(0, 256)
                    return tuple(backing[:, layer] for layer in range(layers))

                k = buffers(2 if packed_fp4 else 4)
                v = buffers(4 if packed_fp4 else 6)
                scales = (
                    None
                    if scale_layout is None
                    else KVScaleBuffers(
                        k=buffers(1, scales=True),
                        v=buffers(2, scales=True),
                        layout=scale_layout,
                    )
                )
                info = _mha_info(
                    k,
                    v,
                    layout=layout,
                    page_size=page_size,
                    head_num=2,
                    head_dim=4,
                    v_head_dim=6,
                    scales=scales,
                )
                host_cls = get_mha_host_pool_cls(info, layout="page_unified")
                self.assertIs(host_cls, NativePageUnifiedMHATokenToKVPoolHost)
                writer, reader = [
                    _unified_host(info, start_layer=5, end_layer=8) for _ in range(2)
                ]
                try:
                    host_indices = torch.tensor([6, 7, 2, 3])
                    source_indices = torch.tensor([4, 5, 8, 9])
                    restore_indices = torch.tensor([0, 1, 6, 7])
                    writer.backup_from_device_all_layer(
                        None, host_indices, source_indices, "kernel"
                    )
                    for source, target in ((6, 4), (2, 0)):
                        reader.set_from_flat_data_page(
                            target, writer.get_data_page(source).clone()
                        )
                    reader_indices = torch.tensor([4, 5, 0, 1])
                    originals = []
                    for component, paged, _ in writer._native_components:
                        source_rows = torch.tensor([2, 4]) if paged else source_indices
                        target_rows = torch.tensor([0, 3]) if paged else restore_indices
                        originals.append(
                            (
                                component,
                                target_rows,
                                [
                                    b.view(torch.uint8)
                                    .index_select(0, source_rows)
                                    .clone()
                                    for b in component
                                ],
                            )
                        )
                        for buffer in component:
                            buffer.view(torch.uint8).fill_(173)
                    # Restoring one layer must leave the others untouched.
                    reader.load_to_device_per_layer(
                        None, reader_indices, restore_indices, 1, "direct"
                    )
                    for component, _, expected in originals:
                        self.assertTrue(
                            torch.all(component[0].view(torch.uint8) == 173)
                        )
                    for layer in (0, 2):
                        reader.load_to_device_per_layer(
                            None, reader_indices, restore_indices, layer, "kernel"
                        )
                    for component, target_rows, expected in originals:
                        for actual, saved in zip(component, expected):
                            self.assertTrue(
                                torch.equal(
                                    actual.view(torch.uint8).index_select(
                                        0, target_rows
                                    ),
                                    saved,
                                )
                            )
                            unselected = torch.ones(actual.shape[0], dtype=torch.bool)
                            unselected[target_rows] = False
                            self.assertTrue(
                                torch.all(actual.view(torch.uint8)[unselected] == 173)
                            )
                    self.assertEqual(
                        reader.page_unified_layout.bytes_per_page,
                        reader.get_dummy_flat_data_page().numel(),
                    )
                finally:
                    writer.destroy()
                    reader.destroy()

    @_cpu_host_allocation
    def test_native_file_objects_restore_across_tp_and_uneven_pp(self):
        """Payload and scales share global layer/head objects across deployments."""
        layers, heads, tokens, page_size = 5, 4, 8, 2
        originals = [
            ((torch.arange(layers * tokens * heads * width) + offset) % 251)
            .reshape(layers, tokens, heads, width)
            .to(dtype)
            for width, dtype, offset in (
                (2, torch.uint8, 7),
                (4, torch.uint8, 31),
                (1, torch.float32, 57),
                (2, torch.float32, 83),
            )
        ]

        def make_host(start, end, first, last, *, source=False):
            components = [
                tuple(
                    tensor[layer, :, first:last].clone()
                    if source
                    else torch.full_like(tensor[layer, :, first:last], 173)
                    for layer in range(start, end)
                )
                for tensor in originals
            ]
            info = _mha_info(
                components[0],
                components[1],
                page_size=page_size,
                head_num=last - first,
                head_dim=4,
                v_head_dim=8,
                scales=KVScaleBuffers(
                    k=components[2], v=components[3], layout=KVScaleLayout.TOKEN
                ),
            )
            host = _unified_host(
                info, start_layer=start, end_layer=end, head_group_num=last - first
            )
            return host, components

        def backend(host, directory, *, tp_rank=0, tp_size=1, pp_rank=0):
            plan = plan_unified_kv(
                model_id=storage_model_name("org/model", host),
                dtype="uint8",
                page_size=page_size,
                rank_replicated=False,
                local_kv_heads=host.head_num,
                attn_tp_rank=tp_rank,
                attn_tp_size=tp_size,
                attn_cp_size=1,
                start_layer=host.start_layer,
                end_layer=host.end_layer,
                is_final_stage=host.end_layer == layers,
                head_group=1,
                layer_partition=1,
            )
            spans = [
                host.page_unified_layout.chunk_span(window, group)
                for window in plan.layer_ranges
                for group in range(plan.head_group_num)
            ]
            config = HiCacheStorageConfig(
                tp_rank=tp_rank,
                tp_size=tp_size,
                pp_rank=pp_rank,
                pp_size=2,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=False,
                enable_storage_metrics=False,
                is_page_first_layout=False,
                model_name="org/model",
                unified_suffixes=plan.suffixes,
                unified_chunk_spans=spans,
                extra_config={"max_size": 0, "min_free_space": 0},
            )
            return HiCacheFile(config, file_path=directory)

        hosts = []
        try:
            with TemporaryDirectory() as directory:
                writer, _ = make_host(0, layers, 0, heads, source=True)
                hosts.append(writer)
                writer.backup_from_device_all_layer(
                    None, torch.arange(2), torch.arange(2, 4), "kernel"
                )
                storage = backend(writer, directory)
                self.assertTrue(storage.set("prefix", writer.get_data_page(0)))
                self.assertEqual(
                    len(list(Path(directory).glob("*.bin"))), layers * heads
                )
                for rank in range(2):
                    for pp_rank, (start, end) in enumerate(((0, 3), (3, layers))):
                        reader, components = make_host(
                            start, end, rank * 2, rank * 2 + 2
                        )
                        hosts.append(reader)
                        self.assertEqual(
                            writer.storage_format_tag, reader.storage_format_tag
                        )
                        store = backend(
                            reader, directory, tp_rank=rank, tp_size=2, pp_rank=pp_rank
                        )
                        data = reader.get_dummy_flat_data_page()
                        self.assertIsNotNone(store.get("prefix", data))
                        reader.set_from_flat_data_page(2, data)
                        for layer in range(end - start):
                            reader.load_to_device_per_layer(
                                None,
                                torch.arange(2, 4),
                                torch.arange(4, 6),
                                layer,
                                "direct",
                            )
                        for actual, saved in zip(components, originals):
                            for local, layer in enumerate(range(start, end)):
                                self.assertTrue(
                                    torch.equal(
                                        actual[local][4:6],
                                        saved[layer, 2:4, rank * 2 : rank * 2 + 2],
                                    )
                                )
                                self.assertTrue(torch.all(actual[local][:4] == 173))
                        # A missing scale-containing object may never partially publish a page.
                        path = Path(directory) / (
                            store._get_component_keys("prefix")[0] + ".bin"
                        )
                        original = path.read_bytes()
                        path.write_bytes(original[:-1])
                        data.fill_(199)
                        self.assertIsNone(store.get("prefix", data))
                        self.assertTrue(torch.all(data == 199))
                        path.write_bytes(original)
        finally:
            for host in hosts:
                host.destroy()

    def test_native_fp4_scale_buffers_are_explicitly_rejected(self):
        buffer = (torch.zeros(8, 2, 4, dtype=torch.uint8),)
        info = _mha_info(
            buffer,
            buffer,
            page_size=2,
            head_num=2,
            head_dim=8,
            has_native_scale_buffers=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "native NVFP4 scale"):
            get_mha_host_pool_cls(info, layout="page_unified")

    def test_factory_selects_asymmetric_pool_for_mismatched_kv_dims(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        def info(v_head_dim):
            return (
                MHATokenToKVPool(
                    size=8,
                    page_size=2,
                    dtype=torch.float16,
                    head_num=1,
                    head_dim=4,
                    v_head_dim=v_head_dim,
                    layer_num=1,
                    device="cpu",
                    enable_memory_saver=False,
                )
                .get_device_pool_infos()[0]
                .buffer_info
            )

        symmetric_pool = info(4)
        asymmetric_pool = info(6)

        self.assertIs(get_mha_host_pool_cls(symmetric_pool), MHATokenToKVPoolHost)
        self.assertIs(
            get_mha_host_pool_cls(asymmetric_pool), AsymmetricMHATokenToKVPoolHost
        )

    def test_builds_over_a_real_device_pool(self):
        # The asymmetric pool has per-side row widths and no `token_stride_size`,
        # so building it must not need one for its packed device rows.
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        device = MHATokenToKVPool(
            size=15,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            v_head_dim=6,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
        )
        host = AsymmetricMHATokenToKVPoolHost(
            buffer_info=device.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=2,
            layout="page_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
            device_capacity_tokens=device.size
            if device.host_capacity_tokens is None
            else device.host_capacity_tokens,
            start_layer=device.start_layer,
            end_layer=device.end_layer,
            write_back_staging=device.hicache_write_back_staging,
        )
        self.assertTrue(host.device_rows_packed)

    def test_staged_write_back_jit_uses_separate_kv_buffers(self):
        host = _make_host("page_first")
        host.page_num = 4
        host.v_head_dim = 8
        buffer_info = SimpleNamespace(
            device=torch.device("cuda"),
            head_num=host.head_num,
            head_dim=host.head_dim,
            v_head_dim=host.v_head_dim,
            dtype=host.dtype,
        )
        cpu_empty = torch.empty

        def _cpu_empty(shape, *, dtype, device):
            return cpu_empty(shape, dtype=dtype)

        with (
            mock.patch("sglang.srt.mem_cache.pool_host.mha._is_cuda", True),
            mock.patch("sglang.srt.mem_cache.pool_host.mha._is_hip", False),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.current_platform",
                SimpleNamespace(
                    capabilities=PlatformCapabilities(hicache_device_kernels=True)
                ),
            ),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.can_use_write_back_jit_kernel",
                return_value=True,
            ) as can_use,
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.torch.empty",
                side_effect=_cpu_empty,
            ),
        ):
            host._init_write_back_staging_buffers(buffer_info, None)

        self.assertTrue(host.can_use_write_back_jit)
        self.assertEqual(host.staging_page_capacity, 4)
        self.assertEqual(host.staging_token_capacity, 8)
        self.assertEqual(host.staging_k_buffer.shape, (8, 3, 2, 4))
        self.assertEqual(host.staging_v_buffer.shape, (8, 3, 2, 8))
        self.assertEqual(
            [call.kwargs["element_size"] for call in can_use.call_args_list], [16, 32]
        )

    def test_kernel_load_splits_k_and_v_with_separate_strides(self):
        # Dispatch-only test: the CUDA kernel is mocked; this verifies that K and
        # V are sent as separate single-buffer calls with their own byte strides.
        host = _make_host("page_first")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_mla_pf_lf",
            create=True,
        ) as transfer:
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=1,
                io_backend="kernel",
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertIs(k_call.kwargs["src"], host.k_buffer)
        self.assertIs(k_call.kwargs["dst"], device_pool.k_buffer[1])
        self.assertEqual(k_call.kwargs["item_size"], 16)
        self.assertEqual(k_call.kwargs["src_layout_dim"], 48)
        self.assertIs(v_call.kwargs["src"], host.v_buffer)
        self.assertIs(v_call.kwargs["dst"], device_pool.v_buffer[1])
        self.assertEqual(v_call.kwargs["item_size"], 24)
        self.assertEqual(v_call.kwargs["src_layout_dim"], 72)

    def test_kernel_backup_splits_k_and_v_with_separate_strides(self):
        # Dispatch-only test: D2H backup must pass separate K/V layer pointer
        # tables so the single-buffer MLA kernel gets the correct stride per side.
        host = _make_host("page_first")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_mla_lf_pf",
            create=True,
        ) as transfer:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertIs(k_call.kwargs["src_layers"], device_pool.k_data_ptrs)
        self.assertIs(k_call.kwargs["dst"], host.k_buffer)
        self.assertEqual(k_call.kwargs["item_size"], 16)
        self.assertEqual(k_call.kwargs["dst_layout_dim"], 48)
        self.assertIs(v_call.kwargs["src_layers"], device_pool.v_data_ptrs)
        self.assertIs(v_call.kwargs["dst"], host.v_buffer)
        self.assertEqual(v_call.kwargs["item_size"], 24)
        self.assertEqual(v_call.kwargs["dst_layout_dim"], 72)

    def test_kernel_backup_uses_staged_kernel_for_each_kv_buffer(self):
        host = _make_host("page_first")
        host.can_use_write_back_jit = True
        host.staging_k_buffer = torch.empty(4, 3, 2, 4)
        host.staging_v_buffer = torch.empty(4, 3, 2, 6)
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with (
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.jit_transfer_hicache_all_layer_mla_staged_lf_pf"
            ) as staged,
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_mla_lf_pf",
                create=True,
            ) as fallback,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        self.assertEqual(staged.call_count, 2)
        self.assertEqual(fallback.call_count, 0)
        k_call, v_call = staged.call_args_list
        self.assertIs(k_call.kwargs["ptr_src"], device_pool.k_data_ptrs)
        self.assertIs(k_call.kwargs["staging"], host.staging_k_buffer)
        self.assertIs(k_call.kwargs["dst"], host.k_buffer)
        self.assertIs(v_call.kwargs["ptr_src"], device_pool.v_data_ptrs)
        self.assertIs(v_call.kwargs["staging"], host.staging_v_buffer)
        self.assertIs(v_call.kwargs["dst"], host.v_buffer)

    def test_staged_kernel_backup_load_roundtrip_preserves_asymmetric_kv(self):
        """Staged write-back must preserve both K and V values across a round trip."""
        host = _make_host("page_first")
        host.can_use_write_back_jit = True
        host.staging_k_buffer = torch.empty(4, 3, 2, 4)
        host.staging_v_buffer = torch.empty(4, 3, 2, 6)
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        for layer_id in range(host.layer_num):
            device_pool.k_buffer[layer_id].copy_(
                torch.arange(device_pool.k_buffer[layer_id].numel()).reshape_as(
                    device_pool.k_buffer[layer_id]
                )
                + layer_id * 1000
            )
            device_pool.v_buffer[layer_id].copy_(
                torch.arange(device_pool.v_buffer[layer_id].numel()).reshape_as(
                    device_pool.v_buffer[layer_id]
                )
                + layer_id * 10000
            )

        expected_k = [buffer[device_indices].clone() for buffer in device_pool.k_buffer]
        expected_v = [buffer[device_indices].clone() for buffer in device_pool.v_buffer]
        buffers_by_ptrs = {
            tuple(device_pool.k_data_ptrs.tolist()): device_pool.k_buffer,
            tuple(device_pool.v_data_ptrs.tolist()): device_pool.v_buffer,
        }

        def staged_copy(*, ptr_src, src_indices, dst_indices, dst, **_):
            src_buffers = buffers_by_ptrs[tuple(ptr_src.tolist())]
            for layer_id, src in enumerate(src_buffers):
                dst[dst_indices, layer_id] = src[src_indices]

        def per_layer_copy(*, src, dst, src_indices, dst_indices, layer_id, **_):
            dst[dst_indices] = src[src_indices, layer_id]

        with (
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                side_effect=staged_copy,
            ),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_mla_pf_lf",
                side_effect=per_layer_copy,
                create=True,
            ),
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )
            for buffer in device_pool.k_buffer + device_pool.v_buffer:
                buffer.zero_()
            for layer_id in range(host.layer_num):
                host.load_to_device_per_layer(
                    device_pool,
                    host_indices,
                    device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        for layer_id in range(host.layer_num):
            torch.testing.assert_close(
                host.k_buffer[host_indices, layer_id], expected_k[layer_id]
            )
            torch.testing.assert_close(
                host.v_buffer[host_indices, layer_id], expected_v[layer_id]
            )
            torch.testing.assert_close(
                device_pool.k_buffer[layer_id][device_indices], expected_k[layer_id]
            )
            torch.testing.assert_close(
                device_pool.v_buffer[layer_id][device_indices], expected_v[layer_id]
            )

    def test_direct_load_splits_k_and_v_for_page_first_direct(self):
        # Direct kernels derive copy size from each call's first tensor, so K/V
        # must be dispatched separately when their head dims differ.
        host = _make_host("page_first_direct")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_direct_pf_lf",
            create=True,
        ) as transfer:
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=2,
                io_backend="direct",
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertEqual(len(k_call.kwargs["src_ptrs"]), 1)
        self.assertEqual(len(k_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(k_call.kwargs["src_ptrs"][0], host.k_buffer)
        self.assertIs(k_call.kwargs["dst_ptrs"][0], device_pool.k_buffer[2])
        self.assertEqual(len(v_call.kwargs["src_ptrs"]), 1)
        self.assertEqual(len(v_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(v_call.kwargs["src_ptrs"][0], host.v_buffer)
        self.assertIs(v_call.kwargs["dst_ptrs"][0], device_pool.v_buffer[2])

    def test_direct_backup_splits_k_and_v_for_page_first_direct(self):
        host = _make_host("page_first_direct")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_direct_lf_pf",
            create=True,
        ) as transfer:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="direct"
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertEqual(len(k_call.kwargs["src_ptrs"]), host.layer_num)
        self.assertEqual(len(k_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(k_call.kwargs["src_ptrs"][0], device_pool.k_buffer[0])
        self.assertIs(k_call.kwargs["dst_ptrs"][0], host.k_buffer)
        self.assertEqual(len(v_call.kwargs["src_ptrs"]), host.layer_num)
        self.assertEqual(len(v_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(v_call.kwargs["src_ptrs"][0], device_pool.v_buffer[0])
        self.assertIs(v_call.kwargs["dst_ptrs"][0], host.v_buffer)

    def test_direct_requires_page_first_direct_layout(self):
        host = _make_host("page_first")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "expected 'page_first_direct'"):
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=2,
                io_backend="direct",
            )


if __name__ == "__main__":
    unittest.main()
