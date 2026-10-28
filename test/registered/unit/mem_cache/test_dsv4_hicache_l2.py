"""Focused CPU contracts for DeepSeek V4 HiCache L2 (NPU, and V4.1 --hicache-size)."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSV4DecodeCapacity(CustomTestCase):
    def test_decode_capacity_uses_exact_c128_page_demand(self):
        allocator = SimpleNamespace(
            page_size=128,
            c128_attn_allocator=SimpleNamespace(page_size=16),
            evict_to_free_tokens=MagicMock(),
            ensure_c128_capacity=MagicMock(return_value=True),
            full_swa_available_size=MagicMock(return_value=384),
        )
        allocator.c128_num_pages_needed = MethodType(
            DSV4NPUTokenToKVPoolAllocator.c128_num_pages_needed, allocator
        )
        requests = [
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=length))
            for length in (127, 2175, 2047)
        ]

        result = DSV4NPUTokenToKVPoolAllocator.check_decode_capacity(
            allocator,
            num_tokens=384,
            tree_cache=MagicMock(),
            requests=requests,
            spec_algorithm=SimpleNamespace(is_some=lambda: False),
        )

        self.assertTrue(result)
        # 127 -> 128 and 2175 -> 2176 open new physical C128 pages;
        # 2047 -> 2048 only fills the current page.
        allocator.ensure_c128_capacity.assert_called_once()
        self.assertEqual(allocator.ensure_c128_capacity.call_args.args[1], 2)
        allocator.evict_to_free_tokens.assert_called_once()


class TestDSV4HostViews(CustomTestCase):
    @staticmethod
    def _layout_buffer(raw, layout):
        if layout == "layer_first":
            return [raw[:, layer, :].clone() for layer in range(raw.shape[1])]
        if layout == "page_first":
            return raw.clone()
        return raw.unsqueeze(2).clone()

    @staticmethod
    def _selected_layer(buffer, layout, layer):
        if layout == "layer_first":
            return buffer[layer]
        if layout == "page_first":
            return buffer[:, layer, :]
        return buffer[:, layer, 0, :]

    def test_paged_and_state_host_views_support_every_layout(self):
        pages, layers, page_slots, last_dim = 2, 2, 2, 3
        item_bytes = page_slots * last_dim
        raw = torch.arange(pages * layers * item_bytes, dtype=torch.uint8).reshape(
            pages, layers, item_bytes
        )

        state_dtype = torch.float16
        ring_size = 2
        state_last_dim = 3
        state_page_bytes = ring_size * state_last_dim * state_dtype.itemsize
        state_raw = torch.arange(
            pages * layers * state_page_bytes, dtype=torch.uint8
        ).reshape(pages, layers, state_page_bytes)

        for layout in ("layer_first", "page_first", "page_first_direct"):
            with self.subTest(pool="paged", layout=layout):
                host = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
                host.pool_name = "c4"
                host.layout = layout
                host.num_host_pages = pages
                host.slot_page_size = page_slots
                host.device_buffers = [
                    torch.empty((4, page_slots, 1, last_dim), dtype=torch.int8)
                    for _ in range(layers)
                ]
                host.kv_buffer = self._layout_buffer(raw, layout)

                view = host._host_page_view(1)
                expected = self._selected_layer(host.kv_buffer, layout, 1).view(
                    torch.int8
                )
                self.assertEqual(view.shape, (pages, 1, page_slots, 1, last_dim))
                self.assertTrue(
                    torch.equal(view.reshape(pages, -1), expected.reshape(pages, -1))
                )

            with self.subTest(pool="state", layout=layout):
                host = DeepSeekV4StateHostPool.__new__(DeepSeekV4StateHostPool)
                host.pool_name = "c4_state"
                host.layout = layout
                host.num_host_pages = pages
                host.ring_size = ring_size
                host.state_page_bytes = state_page_bytes
                host.state_pools = [
                    SimpleNamespace(
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.empty(1, dtype=state_dtype)
                        )
                    )
                    for _ in range(layers)
                ]
                host.kv_buffer = self._layout_buffer(state_raw, layout)

                view = host._state_host_page_view(1)
                expected = self._selected_layer(host.kv_buffer, layout, 1).view(
                    state_dtype
                )
                self.assertEqual(view.shape, (pages, 1, ring_size, 1, state_last_dim))
                self.assertTrue(
                    torch.equal(view.reshape(pages, -1), expected.reshape(pages, -1))
                )


class TestC128L2Ownership(CustomTestCase):
    def test_eviction_hooks_accept_cache_actions(self):
        pages = torch.tensor([7, 8])
        data = SimpleNamespace(value=pages, host_value=None)
        node = SimpleNamespace(component_data={ComponentType.C128: data})
        component = C128SidecarComponent.__new__(C128SidecarComponent)
        component.tree_core = SimpleNamespace(
            component_evictable_size_={ComponentType.C128: 2}
        )
        device_frees = {ComponentType.C128: []}
        host_frees = {ComponentType.C128: []}
        actions = []

        self.assertEqual(
            component.evict_component(
                node, device_frees, host_frees, cache_actions=actions
            ),
            (0, 0),
        )
        self.assertIsNone(data.value)
        self.assertIs(device_frees[ComponentType.C128][0], pages)
        self.assertEqual(
            component.tree_core.component_evictable_size_[ComponentType.C128], 0
        )

        component.is_evict_device_ongoing = False
        component.evict_device_start(1)
        try:
            self.assertIsNone(
                component.evict_device_next_node(
                    {}, device_frees, host_frees, cache_actions=actions
                )
            )
        finally:
            component.evict_device_end()
        self.assertEqual(actions, [])

    def test_load_back_retains_and_attaches_each_c128_page(self):
        allocator = SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=4),
            retain_c128_pages=MagicMock(),
        )
        nodes = {
            node_id: SimpleNamespace(
                component_data={
                    ComponentType.C128: SimpleNamespace(
                        host_value=torch.arange(4), value=None
                    )
                }
            )
            for node_id in (1, 2)
        }
        tree_core = SimpleNamespace(
            node_by_id=lambda node_id: nodes[node_id],
            set_component_device_value=MagicMock(),
        )
        component = C128SidecarComponent.__new__(C128SidecarComponent)
        component.cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        component.tree_core = tree_core
        transfer = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C128,
            device_indices=torch.arange(28, 36),
            nodes_to_load=[1, 2],
        )

        component.commit_hicache_transfer(
            node=None,
            phase=CacheTransferPhase.LOAD_BACK,
            transfers=[transfer],
            cache_actions=[],
        )

        retained = [
            call.args[0].tolist() for call in allocator.retain_c128_pages.call_args_list
        ]
        attached = [
            (call.args[0], call.args[2].tolist())
            for call in tree_core.set_component_device_value.call_args_list
        ]
        self.assertEqual(retained, [[7], [8]])
        self.assertEqual(attached, [(1, [7]), (2, [8])])


class TestDSV4PoolAssembly(CustomTestCase):
    def test_npu_indexer_registers_k_and_scale_as_separate_host_pools(self):
        k_buffer = [torch.empty((2, 32, 1, 4), dtype=torch.int8)]
        scale_buffer = [torch.empty((2, 32, 1, 1), dtype=torch.float16)]
        c4_buffer = [torch.empty((2, 32, 1, 8), dtype=torch.int8)]
        indexer_pool = SimpleNamespace(
            has_npu_storage=True,
            index_k_buffer=k_buffer,
            index_scale_buffer=scale_buffer,
        )
        kvcache = SimpleNamespace(
            _unified_kv=True,
            size=256,
            swa_size=0,
            swa_page_size=256,
            c4_kv_pool=SimpleNamespace(kernel_page_size=32),
            c4_indexer_kv_pool=indexer_pool,
            unified_region_buffers=lambda ratio: (c4_buffer, 256),
            unified_rope_region_buffers=lambda ratio: None,
        )
        params = SimpleNamespace(
            page_size=128,
            token_to_kv_pool_allocator=SimpleNamespace(size_full=256),
            mtp_draft_device_pools=[],
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        mappings = assembler._DeepSeekV4LayerMappings(
            transfer_layer_id_max=1,
            full={0: 0},
            swa={},
            c4={0: 0},
            c128={},
            c4_state={},
            c4_state_global_layers=[],
        )
        memory = SimpleNamespace(
            hicache_mem_layout="page_first_direct",
            hicache_write_policy="write_through",
            hicache_io_backend="kernel_ascend",
            hicache_host_memory_mode="cache",
        )
        logical_pool = SimpleNamespace(
            layout=memory.hicache_mem_layout,
            page_size=128,
            device="cpu",
            size=256,
            logical_size=256,
            can_use_write_back_jit=False,
        )

        def make_host_pool(**kwargs):
            return SimpleNamespace(
                **kwargs,
                page_size=kwargs["slot_page_size"],
                device="cpu",
                size=kwargs["num_host_pages"] * kwargs["slot_page_size"],
                logical_size=kwargs["num_host_pages"] * kwargs["slot_page_size"],
                can_use_write_back_jit=False,
            )

        controller = object()
        with (
            patch.object(assembler, "get_memory", return_value=memory),
            patch.object(
                assembler, "_deepseek_v4_num_host_pages", return_value=(2, 0, 0)
            ),
            patch.object(assembler, "_get_allocator_type", return_value="default"),
            patch.object(assembler, "LogicalHostPool", return_value=logical_pool),
            patch.object(
                assembler,
                "DeepSeekV4PagedHostPool",
                side_effect=make_host_pool,
            ),
            patch.object(assembler, "HybridCacheController", return_value=controller),
        ):
            group, built_controller = assembler.build_deepseek_v4_hicache_stack(
                params=params,
                kvcache=kvcache,
                load_cache_event=None,
                storage_backend=None,
                layer_mappings=mappings,
            )

        k_host = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER].host_pool
        scale_host = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE].host_pool
        self.assertIs(k_host.device_buffers, k_buffer)
        self.assertIs(scale_host.device_buffers, scale_buffer)
        self.assertEqual((k_host.item_bytes, scale_host.item_bytes), (128, 64))
        self.assertEqual((k_host.slot_page_size, scale_host.slot_page_size), (128, 128))
        self.assertIs(built_controller, controller)


def _v41_gpu_kvcache():
    """Non-unified GPU V4.1 geometry: C4 (+ fused indexer and compressor states),
    C128, shared C1/C2 sources, paged SWA, and a DSpark draft's SWA layers."""

    def rows(num_rows, row_bytes):
        return torch.empty((num_rows, row_bytes), dtype=torch.uint8)

    def state(ring_size, slot_bytes):
        kv_score = torch.empty((8, slot_bytes), dtype=torch.uint8)
        return SimpleNamespace(
            ring_size=ring_size, kv_score_buffer=SimpleNamespace(kv_score=kv_score)
        )

    def low_ratio_index_pool():
        # 256-token pages over 64-slot index pages: 4 index rows per FULL page.
        return SimpleNamespace(page_size=64, index_k_with_scale_buffer=[rows(21, 32)])

    kvcache = SimpleNamespace(
        start_layer=0,
        end_layer=4,
        size=10 * 256,
        swa_size=4 * 256,
        swa_page_size=256,
        layer_mapping=[
            SimpleNamespace(compress_ratio=ratio, compress_layer_id=index)
            for ratio, index in ((4, 0), (128, 0), (4, 1), (128, 1))
        ],
        swa_kv_pool=SimpleNamespace(
            page_size=256, bytes_per_page_padded=1000, kv_buffer=[rows(5, 1000)] * 4
        ),
        c4_kv_pool=SimpleNamespace(
            bytes_per_page_padded=300, kv_buffer=[rows(5, 300)] * 2
        ),
        c4_indexer_kv_pool=SimpleNamespace(index_k_with_scale_buffer=[rows(5, 70)] * 2),
        c128_kv_pool=SimpleNamespace(
            bytes_per_page_padded=20, kv_buffer=[rows(5, 20)] * 2
        ),
        compress_state_pools={0: state(2, 40), 2: state(2, 40)},
        indexer_compress_state_pools={0: state(3, 8), 2: state(3, 8)},
        sources_by_ratio={1: [3], 2: [1]},
        kv_pools={
            1: SimpleNamespace(kv_buffer=[rows(5, 500)]),
            2: SimpleNamespace(kv_buffer=[rows(5, 250)]),
        },
        index_pools={1: low_ratio_index_pool(), 2: low_ratio_index_pool()},
    )
    params = SimpleNamespace(
        page_size=256,
        token_to_kv_pool_allocator=SimpleNamespace(
            size_full=kvcache.size,
            swa_attn_allocator=SimpleNamespace(alloc=None, free=None),
        ),
        mtp_draft_device_pools=[
            SimpleNamespace(swa_kv_pool=SimpleNamespace(kv_buffer=[rows(5, 1000)] * 3))
        ],
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    return kvcache, params


def _build_v41_stack(*, hicache_size_gb, runtime_on):
    """Run the V4 stack builder with recording host pools; returns
    {pool name: (bytes per host page, host pages)}."""
    kvcache, params = _v41_gpu_kvcache()
    memory = SimpleNamespace(
        hicache_size=hicache_size_gb,
        hicache_ratio=2.0,
        hicache_mem_layout="page_first",
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_host_memory_mode="cache",
    )
    pools = {}

    def paged(**kwargs):
        page_bytes = len(kwargs["device_buffers"]) * kwargs["item_bytes"]
        pools[kwargs["pool_name"]] = (page_bytes, kwargs["num_host_pages"])

    def state(**kwargs):
        page_bytes = sum(
            pool.ring_size * pool.kv_score_buffer.kv_score[0].nbytes
            for pool in kwargs["state_pools"]
        )
        pools[kwargs["pool_name"]] = (page_bytes, kwargs["num_host_pages"])

    flags = frozenset({"runtime"}) if runtime_on else frozenset()
    with (
        get_flags().dsv41_sm90.override(subsystems=flags),
        patch.object(assembler, "get_memory", return_value=memory),
        patch.object(assembler, "_get_allocator_type", return_value="default"),
        patch.object(assembler, "LogicalHostPool"),
        patch.object(assembler, "DeepSeekV4PagedHostPool", side_effect=paged),
        patch.object(assembler, "DeepSeekV4StateHostPool", side_effect=state),
        patch.object(assembler, "build_pool_entry"),
        patch.object(assembler, "HostPoolGroup"),
        patch.object(assembler, "HybridCacheController"),
    ):
        assembler.build_deepseek_v4_hicache_stack(
            params=params, kvcache=kvcache, load_cache_event=None, storage_backend=None
        )
    return pools


class TestDSV41HiCacheSize(CustomTestCase):
    """--hicache-size for DeepSeek-V4.1 (dsv41_sm90_on("runtime")): GB per rank over
    every host pool, keeping the device FULL:SWA page ratio."""

    _SWA_SPACE = (
        str(PoolName.SWA),
        str(PoolName.DEEPSEEK_V4_C4_STATE),
        str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
    )

    def test_budget_covers_every_host_pool_the_stack_builds(self):
        budget_gb = 1e-4
        pools = _build_v41_stack(hicache_size_gb=budget_gb, runtime_on=True)
        full_space = {k: v for k, v in pools.items() if k not in self._SWA_SPACE}
        swa_space = {k: v for k, v in pools.items() if k in self._SWA_SPACE}
        self.assertEqual(len(swa_space), 3)
        # C4, its indexer, C128 (a FULL-page sidecar on GPU), C1/C2 KV and indexers.
        self.assertEqual(len(full_space), 7)
        (num_full,) = {pages for _, pages in full_space.values()}
        (num_swa,) = {pages for _, pages in swa_space.values()}
        self.assertEqual(num_swa, max(1, num_full * 4 // 10))
        full_bytes = sum(page_bytes for page_bytes, _ in full_space.values())
        swa_bytes = sum(page_bytes for page_bytes, _ in swa_space.values())
        budget = int(budget_gb * 1e9)
        self.assertLessEqual(num_full * full_bytes + num_swa * swa_bytes, budget)
        # Maximal: one more FULL page (and its share of SWA) would not fit.
        next_swa = max(1, (num_full + 1) * 4 // 10)
        self.assertGreater((num_full + 1) * full_bytes + next_swa * swa_bytes, budget)

    def test_switch_off_keeps_the_upstream_rejection(self):
        with self.assertRaisesRegex(ValueError, "does not support --hicache-size"):
            _build_v41_stack(hicache_size_gb=1e-4, runtime_on=False)

    def test_fit_is_the_largest_feasible_page_count(self):
        for page_bytes, device_pages in (
            ((7, 3, 0), (10, 4, None)),
            ((0, 5, 2), (3, 8, None)),
            ((4, 0, 6), (6, 2, 3)),
        ):
            full_pages, swa_pages, c128_pages = device_pages

            def cost(n):
                swa = max(1, n * swa_pages // full_pages)
                c128 = n if c128_pages is None else max(1, n * c128_pages // full_pages)
                return n * page_bytes[0] + swa * page_bytes[1] + c128 * page_bytes[2]

            for budget in (40, 101, 997):
                with self.subTest(page_bytes=page_bytes, budget=budget):
                    best = max(n for n in range(1, 2000) if cost(n) <= budget)
                    got = assembler._deepseek_v4_fixed_host_pages(
                        budget_bytes=budget,
                        page_bytes=assembler._DeepSeekV4HostPageBytes(*page_bytes),
                        device_full_pages=full_pages,
                        device_swa_pages=swa_pages,
                        device_c128_pages=c128_pages,
                    )
                    self.assertEqual(got[0], best)
        with self.assertRaisesRegex(ValueError, "cannot hold one"):
            assembler._deepseek_v4_fixed_host_pages(
                budget_bytes=9,
                page_bytes=assembler._DeepSeekV4HostPageBytes(7, 3, 0),
                device_full_pages=10,
                device_swa_pages=4,
                device_c128_pages=None,
            )


if __name__ == "__main__":
    unittest.main()
