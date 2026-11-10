"""Unit tests for asymmetric MHA host KV pool transfer dispatch."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.device_pool_info import (
    IndexKeyBufferInfo,
    IndexKeyLayout,
    KVBufferLayout,
    KVBuffers,
    MHABufferInfo,
)
from sglang.srt.mem_cache.hicache_key_scheme import plan_pool_components
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
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
    def test_page_unified_rejects_unadapted_mha_formats(self):
        from sglang.srt.mem_cache.device_pool_info import KVScaleBuffers, KVScaleLayout

        rows = (torch.zeros(12, 1, 32, dtype=torch.uint8),)
        scales = KVScaleBuffers(k=rows, v=rows, layout=KVScaleLayout.TOKEN)
        for overrides in (
            {"scales": scales},
            {"v_head_dim": 16},
            {"layout": KVBufferLayout.HND},
        ):
            with self.subTest(overrides=overrides):
                info = _mha_info(
                    rows, rows, page_size=4, head_num=1, head_dim=32, **overrides
                )
                with self.assertRaisesRegex(NotImplementedError, "page_unified"):
                    get_mha_host_pool_cls(info, layout="page_unified")
                with self.assertRaisesRegex(NotImplementedError, "page_unified"):
                    MHATokenToKVPoolHost(
                        info,
                        2,
                        0,
                        4,
                        "page_unified",
                        pin_memory=False,
                        device_capacity_tokens=8,
                        start_layer=0,
                        end_layer=1,
                    )

    @_cpu_host_allocation
    def test_packed_draft_keeps_its_own_global_heads_and_cp_writer(self):
        """Equal local shapes can represent replicated target and sharded draft heads."""
        published = set()
        for cp_rank in range(2):
            for rank in range(4):
                with self.subTest(fp4=False, rank=rank, cp_rank=cp_rank):

                    def info(layers, *, draft):
                        k = tuple(
                            (
                                torch.empty(12, 1, 32, dtype=torch.uint8)
                                for _ in range(layers)
                            )
                        )
                        v = tuple((torch.empty_like(row) for row in k))
                        scales = None
                        return _mha_info(
                            k,
                            v,
                            page_size=4,
                            head_num=1,
                            head_dim=32,
                            scales=scales,
                            global_head_start=rank if draft else rank // 2,
                            global_head_num=4 if draft else 2,
                            is_storage_writer=cp_rank == 0
                            if draft
                            else cp_rank == 1 and rank % 2 == 0,
                        )

                    target, draft = (info(2, draft=False), info(1, draft=True))
                    host = _unified_host(
                        target,
                        start_layer=10,
                        end_layer=12,
                        draft_buffer_infos=(draft,),
                    )
                    target_only = _unified_host(target, start_layer=10, end_layer=12)
                    try:
                        metadata = host.storage_component_metadata
                        self.assertEqual(
                            [component.shard_coordinate for component in metadata],
                            [f"H{rank // 2}-{rank // 2 + 1}"] * 2
                            + [f"H{rank}-{rank + 1}"],
                        )
                        self.assertEqual(
                            [component.is_storage_writer for component in metadata],
                            [target.is_storage_writer] * 2 + [draft.is_storage_writer],
                        )

                        def plan(pool, layers):
                            return plan_pool_components(
                                model_id=f"target-with-different-draft-heads-{False}",
                                pool_name="kv",
                                object_layout=getattr(pool, "storage_format_tag", None)
                                or "page-unified-v1",
                                page_size=4,
                                components=pool.storage_component_metadata,
                                layer_ids=layers,
                                tp_rank=rank,
                                tp_size=4,
                                attn_cp_rank=cp_rank,
                                attn_cp_size=2,
                            )

                        combined = plan(host, ("L10", "L11", "Dtest0"))
                        standalone = plan(target_only, ("L10", "L11"))
                        self.assertEqual(combined.suffixes[:2], standalone.suffixes)
                        for key, writer in zip(combined.suffixes, combined.writer_mask):
                            if writer:
                                self.assertNotIn(key, published)
                                published.add(key)
                    finally:
                        host.destroy()
                        target_only.destroy()
        self.assertEqual(len(published), 8)

    @_cpu_host_allocation
    def test_unified_objects_restore_global_heads_across_tp_and_uneven_pp(self):
        """A different rank/stage grid consumes the same independently named bytes."""
        for writer_tp, reader_tp, total_heads in (
            (2, 4, 8),
            (4, 2, 8),
            (8, 4, 4),
            (4, 8, 4),
        ):
            with self.subTest(mode="dense", writer_tp=writer_tp):
                size = 4
                tokens = size * 4
                payload_width = 32
                v_width = payload_width
                layout = KVBufferLayout.NHD

                def logical(width, layer, component):
                    return (
                        torch.arange(tokens * total_heads * width, dtype=torch.int64)
                        .reshape(tokens, total_heads, width)
                        .add(layer * 53 + component * 79)
                        .remainder(251)
                        .to(torch.uint8)
                    )

                def physical(data, name):
                    return data.contiguous()

                def make(rank, tp, layers, *, empty):
                    heads = max(1, total_heads // tp)
                    replicas = max(1, tp // total_heads)
                    start = rank // replicas * heads
                    sources = {}
                    for name, width, component in (
                        ("k", payload_width, 0),
                        ("v", v_width, 1),
                        *(),
                    ):
                        sources[name] = tuple(
                            (
                                physical(
                                    logical(width, layer, component)[
                                        :, start : start + heads
                                    ],
                                    name,
                                )
                                for layer in layers
                            )
                        )
                    expected = {
                        name: tuple((x.clone() for x in rows))
                        for name, rows in sources.items()
                    }
                    if empty:
                        for rows in sources.values():
                            for row in rows:
                                row.fill_(173)
                    scales = None
                    info = _mha_info(
                        sources["k"],
                        sources["v"],
                        layout=layout,
                        page_size=size,
                        head_num=heads,
                        head_dim=32,
                        v_head_dim=32,
                        scales=scales,
                        global_head_start=start,
                        global_head_num=total_heads,
                        is_storage_writer=rank % replicas == 0,
                    )
                    host = _unified_host(
                        info,
                        capacity=tokens - size,
                        start_layer=layers[0],
                        end_layer=layers[-1] + 1,
                        head_group_num=max(1, heads // 2),
                        budget=1 << 24,
                    )
                    plan = plan_pool_components(
                        model_id=f"global-head-pattern-{'dense'}",
                        pool_name="kv",
                        object_layout=getattr(host, "storage_format_tag", None)
                        or "page-unified-v1",
                        page_size=size,
                        components=host.storage_component_metadata,
                        layer_ids=tuple((f"L{layer}" for layer in layers)),
                        tp_rank=rank,
                        tp_size=tp,
                    )
                    return (host, plan, sources, expected)

                objects = {}
                source = torch.arange(size, 2 * size)
                host_indices = torch.arange(size)
                for rank in range(writer_tp):
                    writer, plan, _, _ = make(rank, writer_tp, range(3), empty=False)
                    try:
                        writer.backup_from_device_all_layer(
                            None, host_indices, source, "kernel"
                        )
                        page = writer.get_data_page(0).view(torch.uint8)
                        for component in writer.storage_component_metadata:
                            start, end = map(
                                int, component.shard_coordinate[1:].split("-")
                            )
                            width = getattr(writer, "record_bytes", 32)
                            reference = torch.zeros(
                                (component.shape[0], 2, size, end - start, width),
                                dtype=torch.uint8,
                            )
                            for local_layer in range(component.shape[0]):
                                layer = component.layer_offset + local_layer
                                reference[local_layer, 0, ..., :payload_width] = (
                                    logical(payload_width, layer, 0)[source, start:end]
                                )
                                reference[local_layer, 1, ..., :v_width] = logical(
                                    v_width, layer, 1
                                )[source, start:end]
                            self.assertTrue(
                                torch.equal(
                                    page[
                                        component.byte_offset : component.byte_offset
                                        + component.byte_length
                                    ],
                                    reference.flatten(),
                                )
                            )
                        for key, (offset, length), owns_object in zip(
                            plan.suffixes, plan.chunk_spans, plan.writer_mask
                        ):
                            if not owns_object:
                                continue
                            self.assertNotIn(key, objects)
                            objects[key] = page[offset : offset + length].clone()
                    finally:
                        writer.destroy()
                consumed = set()
                for layers in (range(1), range(1, 3)):
                    for rank in range(reader_tp):
                        reader, plan, sources, expected = make(
                            rank, reader_tp, layers, empty=True
                        )
                        try:
                            page = reader.get_dummy_flat_data_page().view(torch.uint8)
                            for key, (offset, length) in zip(
                                plan.suffixes, plan.chunk_spans
                            ):
                                page[offset : offset + length].copy_(objects[key])
                                consumed.add(key)
                            reader.set_from_flat_data_page(0, page)
                            target = torch.arange(2 * size, 3 * size)
                            for layer in range(len(layers)):
                                reader.load_to_device_per_layer(
                                    None, host_indices, target, layer, "kernel"
                                )
                            for name, rows in sources.items():
                                paged = layout != KVBufferLayout.NHD or (
                                    name.endswith("scale") and "dense" == "mxfp8"
                                )
                                if name.endswith("scale") and "dense" != "mxfp8":
                                    paged = False
                                for actual, original in zip(rows, expected[name]):
                                    self.assertTrue(
                                        torch.equal(
                                            actual[2] if paged else actual[target],
                                            original[1] if paged else original[source],
                                        ),
                                        name,
                                    )
                                    guard = torch.ones(
                                        actual.shape[0], dtype=torch.bool
                                    )
                                    guard[2 if paged else target] = False
                                    self.assertTrue(
                                        torch.all(actual[guard] == 173), name
                                    )
                        finally:
                            reader.destroy()
                self.assertEqual(consumed, objects.keys())

    @_cpu_host_allocation
    def test_canonical_unified_strided_rows_keep_the_existing_byte_order(self):
        """Strided K/V rows must serialize exactly like packed canonical MHA."""
        layers, page_size, heads, width, token_count = 3, 2, 4, 8, 12
        k_backing = torch.empty(token_count, layers, heads, width, dtype=torch.float16)
        v_backing = torch.empty(
            token_count, layers * 2, heads, width, dtype=torch.float16
        )
        k_backing.view(torch.uint8).random_(0, 256)
        v_backing.view(torch.uint8).random_(0, 256)
        k = tuple(k_backing[:, layer] for layer in range(layers))
        v = tuple(v_backing[:, layer * 2] for layer in range(layers))
        info = _mha_info(
            k,
            v,
            page_size=page_size,
            head_num=heads,
            head_dim=width,
            dtype=torch.float16,
        )
        self.assertIs(
            get_mha_host_pool_cls(info, layout="page_unified"), MHATokenToKVPoolHost
        )
        host = _unified_host(info, end_layer=layers, head_group_num=2)
        try:
            host_indices = torch.tensor([6, 7, 0, 1])
            device_indices = torch.tensor([4, 5, 8, 9])
            host.backup_from_device_all_layer(
                None, host_indices, device_indices, "kernel"
            )
            for host_page, device_page in ((3, 2), (0, 4)):
                # The canonical byte protocol is head-group, layer, K/V, token,
                # head-in-group, width, independent of the device row strides.
                expected = torch.stack(
                    [
                        torch.stack(
                            [
                                torch.stack(
                                    [
                                        buffers[layer][
                                            device_page * page_size : (device_page + 1)
                                            * page_size,
                                            group * 2 : (group + 1) * 2,
                                        ]
                                        for buffers in (k, v)
                                    ]
                                )
                                for layer in range(layers)
                            ]
                        )
                        for group in range(2)
                    ]
                )
                self.assertTrue(
                    torch.equal(
                        host.get_data_page(host_page * page_size).view(torch.uint8),
                        expected.flatten().view(torch.uint8),
                    )
                )
            originals = [
                [
                    b.view(torch.uint8).index_select(0, device_indices).clone()
                    for b in component
                ]
                for component in (k, v)
            ]
            for component in (k, v):
                for buffer in component:
                    buffer.view(torch.uint8).fill_(173)
            target_indices = torch.tensor([0, 1, 6, 7])
            for layer in range(layers):
                host.load_to_device_per_layer(
                    None, host_indices, target_indices, layer, "direct"
                )
            for component, expected_layers in zip((k, v), originals):
                for actual, expected in zip(component, expected_layers):
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.uint8).index_select(0, target_indices),
                            expected,
                        )
                    )
                    self.assertTrue(torch.all(actual.view(torch.uint8)[2:6] == 173))
        finally:
            host.destroy()

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
            device_capacity_tokens=(
                device.size
                if device.host_capacity_tokens is None
                else device.host_capacity_tokens
            ),
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


class TestPageUnifiedKOnly(CustomTestCase):
    @mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {"cpu": lambda dims, **kw: torch.zeros(dims, dtype=kw["dtype"])},
    )
    def test_replicated_index_writer_survives_device_descriptor_and_host(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKOnlyPool

        device = MHATokenToKOnlyPool.__new__(MHATokenToKOnlyPool)
        device.page_size = 2
        device.head_num = 1
        device.head_dim = 4
        device.model_layer_ids = (3, 7)
        device.k_buffer = [torch.zeros(10, 1, 4) for _ in range(2)]
        for writer in (False, True):
            with self.subTest(writer=writer):
                device.is_storage_writer = writer
                info = device.get_device_pool_infos()[0].buffer_info
                with host_memory_budget_scope(1 << 20):
                    host = MHATokenToKOnlyPoolHost(
                        info,
                        SimpleNamespace(size=8, page_num=4, page_size=2),
                        "page_unified",
                        pin_memory=False,
                    )
                try:
                    plan = plan_pool_components(
                        model_id="replicated-index",
                        pool_name="indexer",
                        object_layout=host.storage_format_tag,
                        page_size=host.page_size,
                        components=host.storage_component_metadata,
                        layer_ids=("L3", "L7"),
                        tp_rank=0,
                        tp_size=1,
                    )
                    self.assertEqual(plan.chunk_writers, (writer, writer))
                    self.assertEqual(plan.is_storage_writer, writer)
                    self.assertEqual(
                        host.storage_component_metadata[0].shard_coordinate, ""
                    )
                finally:
                    host.destroy()

    @mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {"cpu": lambda dims, **kw: torch.zeros(dims, dtype=kw["dtype"])},
    )
    def test_sparse_index_pages_restore_only_selected_rows_and_layers(self):
        backing = torch.arange(10 * 2 * 2 * 3, dtype=torch.float32).reshape(10, 2, 2, 3)
        buffers = tuple(backing[:, layer].contiguous() for layer in range(2))
        info = IndexKeyBufferInfo(
            page_size=2,
            buffers=buffers,
            compress_ratio=1,
            key_layout=IndexKeyLayout.KEY_ROWS,
        )
        with host_memory_budget_scope(1 << 20):
            host = MHATokenToKOnlyPoolHost(
                info,
                SimpleNamespace(size=8, page_num=4, page_size=2),
                "page_unified",
                pin_memory=False,
            )
        self.addCleanup(host.destroy)
        host_ids = torch.tensor([0, 1, 4, 5])
        device_ids = torch.tensor([2, 3, 6, 7])
        expected = backing[device_ids].clone()
        host.backup_from_device_all_layer(None, host_ids, device_ids, "kernel")
        for start in (0, 4):
            page = host.get_data_page(start).clone()
            self.assertEqual(
                page.nbytes, host.storage_component_metadata[0].byte_length
            )
            host.k_buffer[start // 2].zero_()
            host.set_from_flat_data_page(start, page)
        for buffer in buffers:
            buffer.zero_()
        for layer in range(2):
            host.load_to_device_per_layer(None, host_ids, device_ids, layer, "kernel")
            torch.testing.assert_close(buffers[layer][device_ids], expected[:, layer])
            self.assertEqual(
                buffers[layer][[0, 1, 4, 5, 8, 9]].count_nonzero().item(), 0
            )


if __name__ == "__main__":
    unittest.main()
