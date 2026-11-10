"""Typed page components keep sparse layers and side pools distinct in L3."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_key_scheme import (
    host_page_components,
    plan_pool_components,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host.page_unified import (
    PageStorageComponent,
    PageUnifiedLayout,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _component(layers=4, **overrides):
    values = dict(
        name="state",
        shape=(layers, 4),
        dtype="float16",
        byte_offset=0,
        byte_length=layers * 8,
    )
    values.update(overrides)
    return PageStorageComponent(**values)


def _plan(records, coordinates, **overrides):
    values = dict(
        model_id="test/heterogeneous",
        pool_name=str(PoolName.MAMBA),
        object_layout="native-state-v1",
        page_size=1,
        components=tuple(records),
        layer_ids=tuple(coordinates),
        tp_rank=0,
        tp_size=1,
    )
    values.update(overrides)
    return plan_pool_components(**values)


def _controller_entry(name, *, mapping=None, drafts=(), **host_values):
    from sglang.srt.mem_cache.pool_host import PoolEntry

    host = dict(
        layout="page_unified",
        page_size=1,
        dtype=torch.float16,
        device="cpu",
        size=8,
        logical_size=8,
        can_use_write_back_jit=False,
        storage_format_tag="cp-pool-scope-v1",
        storage_component_metadata=(
            _component(1, name=str(name), shard_coordinate=""),
        ),
    )
    host.update(host_values)
    return PoolEntry(
        name=name,
        host_pool=SimpleNamespace(**host),
        device_pool=SimpleNamespace(dtype=host["dtype"]),
        layer_mapper=lambda layer: layer,
        transfer_layer_mapping={0: 0} if mapping is None else mapping,
        is_primary_index_anchor=name == PoolName.KV,
        packed_draft_device_pools=drafts,
    )


def _controller_for_entries(entries, *, start_layer=10, end_layer=11, hybrid=False):
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.mem_cache.pool_host import HostPoolGroup

    if hybrid:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.extra_host_mem_release_queues = {}
        controller.enable_storage = False
        controller.attn_cp_group = None
    else:
        controller = HiCacheController.__new__(HiCacheController)
    controller.mem_pool_host = HostPoolGroup(entries)
    controller.storage_host_pool = controller.mem_pool_host.anchor_entry.host_pool
    controller.mem_pool_device = SimpleNamespace(start_layer=start_layer)
    controller.storage_start_layer = start_layer
    controller.storage_end_layer = end_layer
    return controller


def _cp_topology(cp=1, **overrides):
    """Four-rank attention domain on the second of two PP stages."""
    values = dict(
        tp_size=4,
        tp_rank=0,
        attn_tp_size=4 // cp,
        attn_tp_rank=0,
        attn_cp_size=cp,
        attn_cp_rank=0,
        attn_dcp_size=1,
        pp_size=2,
        pp_rank=1,
        moe_tp_size=4,
    )
    return get_parallel().override(**(values | overrides))


def _file_backend(pools, plans, *, model="test/heterogeneous", **topology):
    """Run the real file backend with the same no-eviction fixture defaults."""
    anchor = pools[PoolName.KV]
    anchor_plan = plans[PoolName.KV]
    config = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=model,
        unified_pool_plans=plans,
        unified_suffixes=list(anchor_plan.suffixes),
        unified_chunk_spans=list(anchor_plan.chunk_spans),
        extra_config={
            "enable_metadata_cache": True,
            "metadata_ttl": -1,
            "max_size": 0,
            "min_free_space": 0,
        },
    )
    backend = HiCacheFile(HiCacheStorageConfig(**(config | topology)))
    backend.register_mem_pool_host(anchor)
    for name, pool in pools.items():
        backend.register_mem_host_pool_v2(pool, name)
    return backend


class TestPoolComponentPlans(CustomTestCase):
    def test_minimax_gate_uses_the_resolved_native_stack(self):
        """BF16 MiniMax must stay available when only NVFP4 needs a later PR."""
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import override_platform

        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg.mambaish_config = None
        cfg.is_hybrid_swa = cfg.is_hybrid_swa_compress = False
        cfg.sliding_window_size = None
        cfg.kv_cache_dtype = torch.bfloat16
        for blackwell, quantization in (
            (False, None),
            (True, None),
            (True, "fp8"),
            (False, "minimax_m3_nvfp4"),
            (True, "minimax_m3_nvfp4"),
        ):
            with (
                self.subTest(blackwell=blackwell, quantization=quantization),
                get_context().override_server_args(
                    enable_hierarchical_cache=True,
                    hicache_mem_layout="page_unified",
                    enable_unified_memory=False,
                ),
                override_platform(is_blackwell=blackwell),
                _cp_topology(1),
            ):
                cfg.model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["MiniMaxM3SparseForCausalLM"]
                    ),
                    quantization=quantization,
                )
                if blackwell and quantization == "minimax_m3_nvfp4":
                    with self.assertRaisesRegex(NotImplementedError, "minimax_nvfp4"):
                        cfg._validate_hicache_cp_pools()
                else:
                    cfg._validate_hicache_cp_pools()

    def test_full_arena_views_keep_global_hicache_ownership(self):
        """Allocation and descriptor capture cannot discard full-view coordinates."""
        from unittest.mock import patch

        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_kv_host_pool,
        )
        from sglang.srt.mem_cache.kv.decl import (
            MHAGeometry,
            MLAGeometry,
            kv_layer_decls,
        )
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        for mla in (False, True):
            for cp_rank, tp_rank in ((0, 0), (0, 2), (1, 0), (1, 2)):
                with (
                    self.subTest(mla=mla, cp=cp_rank, tp=tp_rank),
                    get_context().override_server_args(
                        enable_hierarchical_cache=True,
                        enable_unified_memory=True,
                        hicache_mem_layout="page_unified",
                        hicache_storage_key_scheme="unified",
                        hicache_storage_head_partition=1,
                        hicache_ratio=2,
                        hicache_size=0,
                    ),
                    _cp_topology(
                        2,
                        tp_size=8,
                        moe_tp_size=8,
                        tp_rank=cp_rank * 4 + tp_rank,
                        attn_tp_size=4,
                        attn_tp_rank=tp_rank,
                        attn_cp_rank=cp_rank,
                    ),
                    patch.dict(
                        ALLOC_MEMORY_FUNCS,
                        {
                            "cpu": lambda dims, **kw: torch.empty(
                                dims, dtype=kw["dtype"]
                            )
                        },
                    ),
                    patch(
                        "sglang.srt.mem_cache.pool_host.base._cuda_host_unregister",
                        lambda _: None,
                    ),
                    host_memory_budget_scope(1 << 20),
                ):
                    geometry = (
                        MLAGeometry(
                            kv_lora_rank=8,
                            qk_rope_head_dim=8,
                            kv_cache_dtype=torch.bfloat16,
                        )
                        if mla
                        else MHAGeometry(
                            head_num=1,
                            head_dim=16,
                            v_head_dim=16,
                            kv_cache_dtype=torch.bfloat16,
                        )
                    )
                    cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
                    cfg.page_size, cfg.device = 4, "cpu"
                    cfg.is_draft_worker, cfg.post_capture_kv_active = False, False
                    cfg.forward_stream = None
                    cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=3)
                    cfg.model_config = SimpleNamespace(
                        context_len=32,
                        sliding_window_size=8,
                        head_dim=16,
                        get_num_kv_heads=lambda tp, dcp: 1,
                        get_total_num_kv_heads=lambda: 2,
                        get_swa_num_kv_heads=lambda tp: 2,
                    )
                    # A valid arena needs two lifetimes. Only the full view is
                    # exercised here; hybrid/CP admission is tested separately.
                    sizes = SimpleNamespace(
                        unified_memory_pool_bytes=8192,
                        unified_total_bytes=None,
                        full_max_total_num_tokens=64,
                        swa_max_total_num_tokens=16,
                        max_running_requests=2,
                    )
                    with (
                        patch.object(
                            KVCacheConfigurator,
                            "_kv_layer_decls",
                            return_value=kv_layer_decls(
                                full_layer_ids=(0, 1),
                                full_geometry=geometry,
                                window_layer_ids=(2,),
                                window_geometry=MHAGeometry(
                                    head_num=1,
                                    head_dim=16,
                                    v_head_dim=16,
                                    kv_cache_dtype=torch.bfloat16,
                                ),
                                window=8,
                            ),
                        ),
                        patch.object(
                            KVCacheConfigurator,
                            "_unified_window_and_state",
                            return_value=(True, False),
                        ),
                        patch.object(
                            KVCacheConfigurator,
                            "_fused_draft_for_swa_factory",
                            return_value=None,
                        ),
                        patch.object(
                            KVCacheConfigurator,
                            "_unified_request_rows",
                            return_value=(0, 0),
                        ),
                    ):
                        storage = cfg._init_unified_pools(sizes=sizes)
                    view = storage.view("full")
                    info = view.get_device_pool_infos()[0].buffer_info
                    owner = cp_rank == 0 and (tp_rank == 0 if mla else tp_rank % 2 == 0)
                    self.assertEqual(info.is_storage_writer, owner)
                    if not mla:
                        self.assertEqual(
                            (info.global_head_start, info.global_head_num),
                            (tp_rank // 2, 2),
                        )
                    host = build_kv_host_pool(kv_pool=view, page_size=4, use_mla=mla)
                    try:
                        records = host.storage_component_metadata
                        self.assertTrue(records)
                        self.assertTrue(
                            all(record.is_storage_writer == owner for record in records)
                        )
                        coordinate = (
                            "" if mla else f"H{tp_rank // 2}-{tp_rank // 2 + 1}"
                        )
                        self.assertEqual(
                            {record.shard_coordinate for record in records},
                            {coordinate},
                        )
                    finally:
                        host.destroy()

    def test_layer_partition_preserves_complete_global_grid_across_tp_and_pp(self):
        component = _component(6, shard_coordinate="H4-6")
        coordinates = tuple(f"L{i}" for i in range(6))
        per_layer = _plan((component,), coordinates)
        explicit_one = _plan((component,), coordinates, layer_partition=1)
        self.assertEqual(per_layer, explicit_one)
        grouped = _plan((component,), coordinates, layer_partition=2, stage_end_layer=6)
        left = _plan(
            (_component(4, shard_coordinate="H4-6"),),
            coordinates[:4],
            layer_partition=2,
            stage_end_layer=4,
            is_final_stage=False,
            tp_size=4,
        )
        right = _plan(
            (_component(2, shard_coordinate="H4-6"),),
            coordinates[4:],
            layer_partition=2,
            stage_end_layer=6,
            is_final_stage=True,
            tp_size=8,
        )
        self.assertEqual(grouped.suffixes, left.suffixes + right.suffixes)
        self.assertEqual(grouped.chunk_spans, ((0, 16), (16, 16), (32, 16)))
        self.assertTrue(
            all(
                key.endswith(("_L0-2_H4-6", "_L2-4_H4-6", "_L4-6_H4-6"))
                for key in grouped.suffixes
            )
        )
        whole_bytes = bytes(range(48))
        records = {
            key: whole_bytes[offset : offset + size]
            for key, (offset, size) in zip(grouped.suffixes, grouped.chunk_spans)
        }
        for plan, data in ((left, whole_bytes[:32]), (right, whole_bytes[32:])):
            for key, (offset, size) in zip(plan.suffixes, plan.chunk_spans):
                self.assertEqual(records[key], data[offset : offset + size])
        triples = _plan((component,), coordinates, layer_partition=3, stage_end_layer=6)
        self.assertEqual(len(triples.suffixes), 2)
        self.assertTrue(set(grouped.suffixes).isdisjoint(triples.suffixes))
        self.assertTrue(set(per_layer.suffixes).isdisjoint(grouped.suffixes))

    def test_layer_partition_rejects_sparse_schema_owner_and_byte_gaps(self):
        for coordinates in (("L1", "L2"), ("L0", "L2")):
            with (
                self.subTest(coordinates=coordinates),
                self.assertRaisesRegex(ValueError, "complete global layer grid"),
            ):
                _plan(
                    (_component(2, shard_coordinate=""),),
                    coordinates,
                    layer_partition=2,
                    stage_end_layer=4,
                )
        layers = tuple(
            _component(1, byte_offset=i * 8, layer_offset=i, shard_coordinate="")
            for i in range(2)
        )
        for changed in (
            replace(layers[1], name="different-schema"),
            replace(layers[1], is_storage_writer=False),
        ):
            with (
                self.subTest(component=changed),
                self.assertRaisesRegex(ValueError, "complete global layer grid"),
            ):
                _plan(
                    (layers[0], changed),
                    ("L0", "L1"),
                    layer_partition=2,
                    stage_end_layer=2,
                )
        # Complete global layers can still be physically interleaved with
        # another component; no one contiguous L3 object can span that gap.
        interleaved = tuple(
            _component(
                1,
                name=name,
                byte_offset=i * 8,
                layer_offset=i // 2,
                shard_coordinate="",
            )
            for i, name in enumerate(("a", "b", "a", "b"))
        )
        with self.assertRaisesRegex(ValueError, "byte-adjacent"):
            _plan(interleaved, ("L0", "L1"), layer_partition=2, stage_end_layer=2)
        valid = _plan(layers, ("L0", "L1"), layer_partition=2, stage_end_layer=2)
        self.assertEqual(valid.chunk_spans, ((0, 16),))

    def test_layer_partition_only_final_model_remainder_can_end_short(self):
        component = _component(1, shard_coordinate="")
        final = _plan(
            (component,),
            ("L4",),
            layer_partition=4,
            is_final_stage=True,
            stage_end_layer=5,
        )
        self.assertTrue(final.suffixes[0].endswith("_L4-5"))
        for options in (
            dict(is_final_stage=False, stage_end_layer=5),
            dict(is_final_stage=True, stage_end_layer=None),
            dict(is_final_stage=True, stage_end_layer=7),
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "complete global layer grid"),
            ):
                _plan((component,), ("L4",), layer_partition=4, **options)

    def test_layer_partition_keeps_drafts_and_slot_side_state_independent(self):
        components = (
            _component(6, shard_coordinate=""),
            _component(
                1,
                name="request",
                shape=(4,),
                layer_axis=None,
                layer_coordinate="L0",
                byte_offset=48,
                byte_length=8,
                shard_coordinate="",
                is_storage_writer=False,
            ),
        )
        plan = _plan(
            components,
            ("L0", "L1", "L2", "L3", "D0", "D1"),
            layer_partition=2,
            stage_end_layer=4,
        )
        self.assertEqual(
            plan.chunk_spans, ((0, 16), (16, 16), (32, 8), (40, 8), (48, 8))
        )
        self.assertEqual(plan.chunk_writers, (True, True, True, True, False))
        self.assertTrue(plan.suffixes[2].endswith("_D0"))
        self.assertTrue(plan.suffixes[3].endswith("_D1"))
        self.assertTrue(plan.suffixes[4].endswith("_L0"))

    def test_controller_forwards_layer_partition_and_true_model_stage_end(self):
        entry = _controller_entry(
            PoolName.KV,
            mapping={i: i for i in range(6)},
            storage_format_tag="state-v1",
            storage_component_metadata=(_component(6, shard_coordinate=""),),
        )
        controller = _controller_for_entries([entry], start_layer=0, end_layer=6)
        with get_context().override_server_args(
            hicache_storage_key_scheme="unified",
            hicache_storage_layer_partition=4,
            pp_size=2,
        ):
            with get_parallel().override(
                tp_rank=0, attn_tp_rank=0, pp_rank=1, pp_size=2
            ):
                plan = controller._build_unified_pool_plans("model", 1)[PoolName.KV]
                self.assertEqual(plan.chunk_spans, ((0, 32), (32, 16)))
                self.assertTrue(plan.suffixes[-1].endswith("_L4-6"))
            with (
                get_parallel().override(
                    tp_rank=0, attn_tp_rank=0, pp_rank=0, pp_size=2
                ),
                self.assertRaisesRegex(ValueError, "complete global layer grid"),
            ):
                controller._build_unified_pool_plans("model", 1)

    def test_pure_state_model_empty_attention_anchor_has_no_head_coordinates(self):
        from sglang.srt.mem_cache.kv.memory.views import MHAKVView
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        class EmptyAttention(MHAKVView):
            layer_num = 0

        config = SimpleNamespace(
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 0),
            is_draft_worker=False,
        )
        anchor = EmptyAttention()
        with get_context().override_server_args(hicache_mem_layout="page_unified"):
            KVCacheConfigurator._bind_hicache_head_coordinates(config, anchor)
        self.assertFalse(hasattr(anchor, "global_head_start"))

    def test_model_head_ownership_handles_gqa_replicas_and_cp(self):
        """Local-head count times TP overcounts replicated GQA KV heads."""
        from sglang.srt.mem_cache.kv.memory.views import MHAKVView
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        class Pool(MHAKVView):
            pass

        config = SimpleNamespace(
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 2),
            is_draft_worker=False,
        )
        config._bind_hicache_head_coordinates = lambda pool, **kwargs: (
            KVCacheConfigurator._bind_hicache_head_coordinates(config, pool, **kwargs)
        )
        with get_context().override_server_args(hicache_mem_layout="page_unified"):
            for cp_rank in (0, 1):
                for tp_rank in range(4):
                    with get_parallel().override(
                        tp_size=8,
                        tp_rank=cp_rank * 4 + tp_rank,
                        moe_tp_size=8,
                        attn_tp_rank=tp_rank,
                        attn_tp_size=4,
                        attn_cp_rank=cp_rank,
                        attn_cp_size=2,
                    ):
                        pool = Pool()
                        config._bind_hicache_head_coordinates(pool)
                        self.assertEqual(pool.global_head_num, 2)
                        self.assertEqual(pool.global_head_start, tp_rank // 2)
                        self.assertEqual(
                            pool.is_storage_writer, cp_rank == 0 and tp_rank % 2 == 0
                        )

    def test_explicit_global_components_reuse_keys_across_tp_pp_and_cp(self):
        """A fixed global head/layer cell must not name its producing rank."""
        components = (
            _component(2, name="kv", shard_coordinate="H4-6"),
            _component(
                2,
                name="replicated",
                byte_offset=16,
                shard_coordinate="",
                is_storage_writer=False,
            ),
        )
        writer = _plan(components, ("L3", "L9"), tp_size=2, tp_rank=1)
        reader = _plan(
            tuple(replace(c, is_storage_writer=False) for c in components),
            ("L3", "L9"),
            tp_size=8,
            tp_rank=6,
            attn_tp_size=4,
            attn_tp_rank=2,
            attn_cp_size=2,
            attn_cp_rank=1,
        )
        self.assertEqual(writer.suffixes, reader.suffixes)
        self.assertEqual(writer.chunk_writers, (True, True, False, False))
        self.assertEqual(writer.owned_suffixes, writer.suffixes[:2])
        self.assertFalse(reader.is_storage_writer)
        split = _plan(
            (
                replace(components[0], shape=(1, 4), byte_length=8),
                replace(components[1], shape=(1, 4), byte_offset=8, byte_length=8),
            ),
            ("L9",),
            tp_size=4,
            tp_rank=2,
        )
        self.assertEqual(split.suffixes, writer.suffixes[1::2])

    def test_request_scoped_replicated_side_state_survives_pp_repartition(self):
        """Unchanged request state cannot depend on unrelated local layers."""
        side = _component(
            1,
            name="ngram",
            shape=(4,),
            layer_axis=None,
            layer_coordinate="request",
            shard_coordinate="",
        )
        whole = _plan((side,), ("L1", "L5", "L8"), tp_size=2)
        stage = _plan((side,), ("L8",), tp_size=4)
        self.assertEqual(whole.suffixes, stage.suffixes)

    def test_hybrid_pp_origin_survives_compact_device_pool_unwrapping(self):
        """A compact full-attention child starts at zero on every PP stage.

        The controller's original model-layer origin must still distinguish
        later stages' index rows from equal-shaped index rows on the first.
        """
        from sglang.srt.managers import cache_controller as module

        entry = _controller_entry(
            PoolName.INDEXER,
            mapping={1: 0},
            storage_format_tag="index-v1",
            storage_component_metadata=(_component(1),),
        )
        compact_child = SimpleNamespace(
            start_layer=0,
            device="cpu",
            layer_num=1,
            register_layer_transfer_counter=lambda counter: None,
        )
        original = SimpleNamespace(
            start_layer=10,
            layer_num=2,
            end_layer=11,  # Legacy device pools may expose an inclusive end.
            layer_router=SimpleNamespace(wrapped_view=lambda: compact_child),
        )
        allocator = SimpleNamespace(get_kvcache=lambda: original)
        # Event/stream creation is the external accelerator boundary; run the
        # real constructor so removing its stage-origin capture breaks this test.
        with (
            patch.object(module.device_module, "Event", return_value=None),
            patch.object(module.device_module, "Stream", return_value=None),
        ):
            controller = module.HiCacheController(
                allocator,
                SimpleNamespace(entries=[entry]),
                page_size=1,
                tp_group=None,
                load_cache_event=None,
                io_backend="kernel",
            )
        controller.storage_host_pool = entry.host_pool
        self.assertEqual(controller.storage_end_layer, 12)
        with (
            get_context().override_server_args(hicache_storage_key_scheme="unified"),
            get_parallel().override(tp_rank=0, attn_tp_rank=0),
        ):
            plan = controller._build_unified_pool_plans("test/model", 1)[
                PoolName.INDEXER
            ]
        self.assertTrue(plan.suffixes[0].endswith("_L11"))
        first_stage = _plan(
            (_component(1),),
            ("L1",),
            model_id="test/model",
            object_layout="index-v1",
            logical_dtype="float16",
        )
        self.assertTrue(set(plan.suffixes).isdisjoint(first_stage.suffixes))

    def test_sparse_anchor_mapping_cannot_use_dense_canonical_coordinates(self):
        """A dense physical buffer does not imply contiguous model layer IDs."""
        geometry = PageUnifiedLayout(
            page_size=4,
            layer_num=2,
            head_num=1,
            head_group_num=1,
            head_dim=8,
            itemsize=2,
        )
        entry = _controller_entry(
            PoolName.KV,
            mapping={3: 0, 7: 1},
            page_size=4,
            storage_format_tag=None,
            storage_component_metadata=None,
            page_unified_layout=geometry,
            layer_num=2,
            start_layer=0,
            end_layer=2,
            head_num=1,
            head_group_num=1,
        )
        controller = _controller_for_entries([entry], start_layer=0)
        with (
            get_context().override_server_args(hicache_storage_key_scheme="unified"),
            get_parallel().override(tp_rank=0, attn_tp_rank=0, pp_rank=0),
        ):
            plan = controller._build_unified_pool_plans("test/model", 1)[PoolName.KV]
        self.assertEqual(
            [suffix.rsplit("_", 1)[-1] for suffix in plan.suffixes], ["L3", "L7"]
        )
        self.assertEqual(
            plan.chunk_spans,
            ((0, geometry.layer_bytes), (geometry.layer_bytes, geometry.layer_bytes)),
        )

    def test_live_cp_plans_accept_full_kv_and_its_indexer(self):
        names = (PoolName.KV, PoolName.INDEXER)
        with (
            get_context().override_server_args(hicache_storage_key_scheme="unified"),
            _cp_topology(2),
        ):
            controller = _controller_for_entries([_controller_entry(n) for n in names])
            baseline = controller._build_unified_pool_plans("test/full-cp", 1)
            for cp_rank in (0, 1):
                with self.subTest(cp_rank=cp_rank):
                    plans = controller._build_unified_pool_plans(
                        "test/full-cp", 2, attn_cp_rank=cp_rank
                    )
                    self.assertEqual(set(plans), set(names))
                    self.assertEqual(
                        {name: plan.suffixes for name, plan in plans.items()},
                        {name: plan.suffixes for name, plan in baseline.items()},
                    )

    def test_live_non_full_pools_are_rejected_before_planning(self):
        full_names = {
            PoolName.KV,
            PoolName.INDEXER,
        }
        with (
            get_context().override_server_args(hicache_storage_key_scheme="unified"),
            _cp_topology(),
        ):
            for name in PoolName:
                if name in full_names:
                    continue
                with self.subTest(pool=name):
                    entries = [_controller_entry(PoolName.KV), _controller_entry(name)]
                    controller = _controller_for_entries(entries)
                    with self.assertRaisesRegex(
                        NotImplementedError, "unsupported pools"
                    ):
                        controller._build_unified_pool_plans("test/pool-scope", 1)
                    # A logical anchor (e.g. DSV4) must not hide a side pool.
                    entries[0].host_pool.storage_component_metadata = ()
                    for group_cp, parallel_cp in ((1, 1), (2, 1), (1, 2), (2, 2)):
                        with (
                            self.subTest(group_cp=group_cp, parallel_cp=parallel_cp),
                            get_parallel().override(
                                attn_tp_size=4 // parallel_cp,
                                attn_cp_size=parallel_cp,
                            ),
                            self.assertRaisesRegex(
                                NotImplementedError, "unsupported pools"
                            ),
                        ):
                            controller._build_unified_pool_plans(
                                "test/pool-scope", group_cp
                            )

    def test_model_semantics_reject_cp_before_a_pure_swa_can_look_like_full_kv(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        defaults = dict(
            mambaish_config=None,
            is_hybrid_swa=False,
            is_hybrid_swa_compress=False,
            sliding_window_size=None,
        )
        cases = (
            ({}, False),
            ({"sliding_window_size": -1}, False),
            ({"sliding_window_size": 0}, False),
            ({"sliding_window_size": 128}, True),
            ({"is_hybrid_swa": True}, True),
            ({"is_hybrid_swa_compress": True}, True),
            ({"mambaish_config": object()}, True),
        )
        for layout, enabled, cp in (
            ("page_unified", True, 2),
            ("page_unified", True, 1),
            ("page_unified", False, 2),
            ("page_first", True, 2),
        ):
            with (
                get_context().override_server_args(
                    enable_hierarchical_cache=enabled,
                    hicache_mem_layout=layout,
                ),
                _cp_topology(cp),
            ):
                for change, unsupported in cases:
                    with self.subTest(layout=layout, enabled=enabled, cp=cp, **change):
                        for key, value in (defaults | change).items():
                            setattr(cfg, key, value)
                        if layout == "page_unified" and enabled and unsupported:
                            with self.assertRaisesRegex(
                                NotImplementedError, "unsupported pools"
                            ):
                                cfg._init_pools(
                                    sizes=None,
                                    req_to_token_pool=None,
                                    token_to_kv_pool_allocator=None,
                                )
                        else:
                            cfg._validate_hicache_cp_pools()

    def test_dynamic_non_full_cp_registration_is_rejected_without_partial_state(self):
        with get_context().override_server_args(
            hicache_mem_layout="page_unified",
            hicache_storage_key_scheme="unified",
        ):
            for group_cp, parallel_cp in ((1, 1), (2, 1), (1, 2), (2, 2)):
                with _cp_topology(parallel_cp):
                    for attached in (False, True):
                        for name in (
                            PoolName.MAMBA,
                            PoolName.DRAFT,
                            PoolName.DRAFT_INDEXER,
                            PoolName.SWA,
                            PoolName.DRAFT_SWA,
                            PoolName.DEEPSEEK_V4_C4,
                        ):
                            with self.subTest(
                                group_cp=group_cp,
                                parallel_cp=parallel_cp,
                                attached=attached,
                                pool=name,
                            ):
                                anchor = _controller_entry(PoolName.KV)
                                controller = _controller_for_entries(
                                    [anchor], hybrid=True
                                )
                                queue = Queue()
                                controller.extra_host_mem_release_queues = {
                                    PoolName.INDEXER: queue
                                }
                                controller.enable_storage = attached
                                backend_changes = []
                                controller.storage_backend = SimpleNamespace(
                                    register_unified_pool_plan=lambda *args: (
                                        backend_changes.append(args)
                                    ),
                                    register_mem_host_pool_v2=lambda *args: (
                                        backend_changes.append(args)
                                    ),
                                )
                                original_plan = object()
                                controller.storage_config = SimpleNamespace(
                                    model_name="test/dynamic-cp",
                                    unified_pool_plans={PoolName.KV: original_plan},
                                )
                                controller.get_attn_cp_rank_and_size = lambda: (
                                    0,
                                    group_cp,
                                )
                                with self.assertRaisesRegex(
                                    NotImplementedError, "unsupported pools"
                                ):
                                    controller.register_host_pool_entry(
                                        _controller_entry(name)
                                    )
                                self.assertEqual(
                                    controller.mem_pool_host.entries, [anchor]
                                )
                                self.assertEqual(
                                    controller.mem_pool_host.entry_map,
                                    {PoolName.KV: anchor},
                                )
                                self.assertEqual(
                                    controller.extra_host_mem_release_queues,
                                    {PoolName.INDEXER: queue},
                                )
                                self.assertEqual(
                                    controller.storage_config.unified_pool_plans,
                                    {PoolName.KV: original_plan},
                                )
                                self.assertEqual(backend_changes, [])

    def test_dynamic_full_registration_accepts_tp_pp_and_cp(self):
        with get_context().override_server_args(hicache_mem_layout="page_unified"):
            for cp, names in (
                (2, (PoolName.INDEXER,)),
                (1, (PoolName.INDEXER,)),
            ):
                with (
                    self.subTest(cp=cp),
                    _cp_topology(cp),
                ):
                    controller = _controller_for_entries(
                        [_controller_entry(PoolName.KV)], hybrid=True
                    )
                    for name in names:
                        entry = _controller_entry(name)
                        controller.register_host_pool_entry(entry)
                        self.assertIs(controller.mem_pool_host.entry_map[name], entry)
                        self.assertIsInstance(
                            controller.extra_host_mem_release_queues[name], Queue
                        )

    def test_sidecar_writer_selection_is_independent_of_anchor_writer(self):
        """Replicated MLA's nonwriter rank still owns opaque sidecar objects."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        controller = object.__new__(HybridCacheController)
        sidecars = (PoolName.INDEXER, PoolName.DRAFT, PoolName.DRAFT_INDEXER)
        for anchor_writer in (False, True):
            plans = {
                PoolName.KV: _plan(
                    (_component(1),), ("L0",), is_storage_writer=anchor_writer
                ),
                **{
                    name: _plan(
                        (_component(1),),
                        ("L0",),
                        pool_name=str(name),
                        is_storage_writer=not anchor_writer,
                    )
                    for name in sidecars
                },
            }
            controller.storage_config = SimpleNamespace(unified_pool_plans=plans)
            controller.storage_backend_type = "file"
            for name in sidecars:
                with self.subTest(anchor_writer=anchor_writer, sidecar=name):
                    self.assertEqual(
                        controller.should_backup(PoolTransfer(name=name)),
                        not anchor_writer,
                    )

    def test_sparse_layers_reuse_the_same_records_after_pp_split(self):
        coordinates = ("L1", "L5", "L8", "L10")
        whole = _plan((_component(),), coordinates)
        left = _plan((_component(2),), coordinates[:2])
        right = _plan((_component(2),), coordinates[2:])
        self.assertEqual(whole.suffixes, left.suffixes + right.suffixes)
        full_page = bytes(range(32))
        whole_records = {
            suffix: full_page[offset : offset + size]
            for suffix, (offset, size) in zip(whole.suffixes, whole.chunk_spans)
        }
        for plan, page in ((left, full_page[:16]), (right, full_page[16:])):
            for suffix, (offset, size) in zip(plan.suffixes, plan.chunk_spans):
                self.assertEqual(page[offset : offset + size], whole_records[suffix])

    def test_head_groups_do_not_alias_after_attention_cp_resharding(self):
        """Global TP4 rank2 owns different heads when CP changes from 1 to 2.

        Head partition one keeps each local group's shape unchanged, so local
        group numbers and global TP identity alone would reuse the wrong heads.
        PP splits must still reuse the same global layer under one topology.
        """
        plans = []
        for local_heads, attention in (
            (2, dict(attn_tp_rank=2, attn_tp_size=4, attn_cp_rank=0, attn_cp_size=1)),
            (4, dict(attn_tp_rank=0, attn_tp_size=2, attn_cp_rank=1, attn_cp_size=2)),
        ):
            geometry = PageUnifiedLayout(
                page_size=4,
                layer_num=2,
                head_num=local_heads,
                head_group_num=local_heads,
                head_dim=8,
                itemsize=2,
            )
            host = SimpleNamespace(dtype=torch.float16, page_unified_layout=geometry)
            components = host_page_components(host)
            whole = _plan(
                components,
                ("L7", "L11"),
                pool_name=str(PoolName.KV),
                page_size=4,
                tp_rank=2,
                tp_size=4,
                **attention,
            )
            plans.append(whole)
            split_components = tuple(
                replace(
                    component,
                    shape=(1, *component.shape[1:]),
                    byte_offset=index * geometry.layer_bytes,
                    byte_length=geometry.layer_bytes,
                )
                for index, component in enumerate(components)
            )
            right_stage = _plan(
                split_components,
                ("L11",),
                pool_name=str(PoolName.KV),
                page_size=4,
                tp_rank=2,
                tp_size=4,
                **attention,
            )
            self.assertEqual(whole.suffixes[1::2], right_stage.suffixes)
        self.assertTrue(set(plans[0].suffixes).isdisjoint(plans[1].suffixes))

    def test_equal_width_types_and_opaque_tp_shards_never_alias(self):
        reference = _plan((_component(1),), ("L5",))
        for overrides in (
            {"components": (_component(1, dtype="bfloat16"),)},
            {"pool_name": str(PoolName.INDEXER)},
            {"object_layout": "different-state-v1"},
            {"tp_rank": 0, "tp_size": 2},
            {"tp_rank": 1, "tp_size": 2},
        ):
            with self.subTest(overrides=overrides):
                changed = _plan((_component(1),), ("L5",), **overrides)
                self.assertTrue(set(reference.suffixes).isdisjoint(changed.suffixes))
        first = _plan((_component(1),), ("L5",), tp_size=2)
        second = _plan((_component(1),), ("L5",), tp_size=2, tp_rank=1)
        self.assertTrue(set(first.suffixes).isdisjoint(second.suffixes))

    def test_heterogeneous_layer_offsets_and_invalid_metadata(self):
        components = (
            _component(1),
            _component(1, dtype="float32", shape=(1, 2), byte_offset=8, layer_offset=1),
        )
        plan = _plan(components, ("L3", "L9"))
        self.assertTrue(plan.suffixes[0].endswith("_L3"))
        self.assertTrue(plan.suffixes[1].endswith("_L9"))
        self.assertEqual(plan.chunk_spans, ((0, 8), (8, 8)))
        for bad in (
            (_component(1, byte_offset=1),),  # A missing first byte.
            (_component(1), _component(1, byte_offset=9, layer_offset=1)),
            (_component(2, byte_length=15),),
            (_component(1, layer_offset=2),),
            (_component(1, layer_axis=1),),
        ):
            with self.subTest(components=bad), self.assertRaises(ValueError):
                _plan(bad, ("L3", "L9"))

    def test_controller_maps_compact_indexer_and_draft_layers(self):
        entry = _controller_entry(
            PoolName.INDEXER,
            mapping={1: 1, 3: 3, 4: 4, 5: 5},
            drafts=(object(), object()),
            page_size=4,
            dtype=torch.uint8,
            storage_format_tag="index-pages-v1",
            storage_component_metadata=(_component(4),),
            _device_to_host_layer={1: 0, 3: 1, 4: 2, 5: 3},
        )
        controller = _controller_for_entries([entry])
        controller._draft_storage_coordinate = lambda depth: f"Dtest-{depth}"
        with (
            get_context().override_server_args(hicache_storage_key_scheme="unified"),
            get_parallel().override(tp_rank=0, attn_tp_rank=0),
        ):
            plan = controller._build_unified_pool_plans("test/model", 1)[
                PoolName.INDEXER
            ]
            self.assertEqual(
                [suffix.rsplit("_", 1)[-1] for suffix in plan.suffixes],
                ["L11", "L13", "Dtest-0", "Dtest-1"],
            )
            del entry.transfer_layer_mapping[3]
            with self.assertRaisesRegex(ValueError, "incomplete.*mapping"):
                controller._build_unified_pool_plans("test/model", 1)


class _PagePool:
    def __init__(self, page_size, rows=3, data=None):
        self.page_size = page_size
        self.page = (
            torch.arange(rows * 16, dtype=torch.uint8).reshape(rows, 16)
            if data is None
            else data.contiguous().view(torch.uint8).reshape(1, -1)
        )
        self.storage_component_metadata = (_component(2),)

    def get_data_page(self, offset, flat=True):
        return self.page[int(offset) // self.page_size]

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.page[0])

    def set_from_flat_data_page(self, offset, page):
        self.page[int(offset) // self.page_size].copy_(page)


class TestUnifiedSidePoolFiles(CustomTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        context = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory.name)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_file_restore_reassembles_different_tp_pp_and_cp_partitions(self):
        """Independent readers must reconstruct global values, not safe-miss."""
        for writer_tp, reader_tp in ((2, 4), (4, 2)):
            with self.subTest(writer_tp=writer_tp, reader_tp=reader_tp):
                self._cross_topology_file_round_trip(writer_tp, reader_tp)

    def test_mixed_state_writer_does_not_overwrite_another_replica(self):
        """A rank can own sharded state while sharing a replicated component."""
        for use_v2 in (False, True):
            with self.subTest(use_v2=use_v2):
                self._mixed_state_writer_does_not_overwrite_another_replica(use_v2)

    def _mixed_state_writer_does_not_overwrite_another_replica(self, use_v2):
        kv, state = _PagePool(1), _PagePool(1)
        components = (
            PageStorageComponent(
                name="replicated",
                shape=(1, 8),
                dtype="uint8",
                byte_offset=0,
                byte_length=8,
                shard_coordinate="",
            ),
            PageStorageComponent(
                name="heads",
                shape=(1, 8),
                dtype="uint8",
                byte_offset=8,
                byte_length=8,
                shard_coordinate="H0-2",
                is_storage_writer=False,
            ),
        )
        kv.storage_component_metadata = components
        original = kv.page[0].clone()
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=[f"mixed-writers-{use_v2}"],
            host_indices=torch.tensor([0]),
        )

        def publish(backend):
            if use_v2:
                return backend.batch_set_v2([transfer])
            return {PoolName.KV: [backend.set(transfer.keys[0], kv.get_data_page(0))]}

        owner = self._backend(kv, state)
        self.assertEqual(publish(owner), {PoolName.KV: [True]})
        kv.storage_component_metadata = (
            replace(components[0], is_storage_writer=False),
            replace(components[1], is_storage_writer=True),
        )
        peer = self._backend(kv, state)
        kv.page[0, :8].fill_(237)
        kv.page[0, 8:].add_(50)
        expected = torch.cat((original[:8], kv.page[0, 8:].clone()))
        self.assertEqual(publish(peer), {PoolName.KV: [True]})
        kv.page[0].zero_()
        self.assertEqual(owner.batch_get_v2([transfer]), {PoolName.KV: [True]})
        self.assertTrue(torch.equal(kv.page[0], expected))

    def _cross_topology_file_round_trip(self, writer_tp, reader_tp):
        def make(tp, rank, pp, stage, cp, cp_rank):
            layers = range(stage * 4 // pp, (stage + 1) * 4 // pp)
            heads = range(rank * 8 // tp, (rank + 1) * 8 // tp)
            values = torch.tensor(
                [
                    [[layer * 100 + head, layer * 100 + head + 1] for layer in layers]
                    for head in range(heads.start, heads.stop, 2)
                ],
                dtype=torch.int32,
            )
            spans = tuple(
                PageStorageComponent(
                    name="kv",
                    shape=(len(layers), 2),
                    dtype="int32",
                    byte_offset=index * len(layers) * 8,
                    byte_length=len(layers) * 8,
                    shard_coordinate=f"H{head}-{head + 2}",
                    is_storage_writer=cp_rank == 0,
                )
                for index, head in enumerate(range(heads.start, heads.stop, 2))
            )
            plan = _plan(
                spans,
                tuple(f"L{layer}" for layer in layers),
                pool_name=str(PoolName.KV),
                tp_size=tp * cp,
                tp_rank=cp_rank * tp + rank,
                attn_tp_size=tp,
                attn_tp_rank=rank,
                attn_cp_size=cp,
                attn_cp_rank=cp_rank,
            )
            page = _PagePool(1, data=values)
            backend = _file_backend(
                {PoolName.KV: page},
                {PoolName.KV: plan},
                model="test/global-grid",
                tp_rank=rank,
                tp_size=tp,
                pp_rank=stage,
                pp_size=pp,
                attn_cp_rank=cp_rank,
                attn_cp_size=cp,
            )
            return page, plan, backend

        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=[f"cross-{writer_tp}-{reader_tp}"],
            host_indices=torch.tensor([0]),
        )
        written = set()
        for rank in range(writer_tp):
            page, plan, backend = make(writer_tp, rank, 1, 0, 1, 0)
            self.assertEqual(backend.batch_set_v2([transfer]), {PoolName.KV: [True]})
            written.update(plan.suffixes)
        required = set()
        for cp_rank in range(2):
            for stage in range(2):
                for rank in range(reader_tp):
                    page, plan, backend = make(reader_tp, rank, 2, stage, 2, cp_rank)
                    expected = page.page.clone()
                    page.page.fill_(237)
                    self.assertEqual(
                        backend.batch_get_v2([transfer]), {PoolName.KV: [True]}
                    )
                    self.assertTrue(torch.equal(page.page, expected))
                    required.update(plan.suffixes)
        self.assertEqual(written, required)

    def _backend(self, anchor, state):
        pools = {PoolName.KV: anchor, PoolName.INDEXER: state}
        plans = {
            name: _plan(
                host_page_components(pool),
                ("L0", "L2"),
                pool_name=str(name),
                page_size=pool.page_size,
            )
            for name, pool in pools.items()
        }
        return _file_backend(pools, plans)

    def test_multi_pool_round_trip_and_corrupt_sidecar_read_is_atomic(self):
        kv, state = _PagePool(4), _PagePool(1)
        backend = self._backend(kv, state)
        keys = [f"page-{i}" for i in range(3)]
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=keys, host_indices=torch.arange(12)),
            PoolTransfer(
                name=PoolName.INDEXER, keys=keys, host_indices=torch.arange(3)
            ),
        ]
        expected = {PoolName.KV: kv.page.clone(), PoolName.INDEXER: state.page.clone()}
        self.assertEqual(
            backend.batch_set_v2(transfers), {name: [True] * 3 for name in expected}
        )
        kv.page.zero_()
        state.page.zero_()
        reader = self._backend(kv, state)
        self.assertEqual(
            reader.batch_get_v2(transfers), {name: [True] * 3 for name in expected}
        )
        self.assertTrue(torch.equal(kv.page, expected[PoolName.KV]))
        self.assertTrue(torch.equal(state.page, expected[PoolName.INDEXER]))
        stem = reader._get_component_keys(keys[1], PoolName.INDEXER)[-1]
        self.assertTrue(reader.metadata_cache.contains(stem))
        Path(reader.file_path, stem + ".bin").write_bytes(b"short")
        state.page.fill_(237)
        self.assertEqual(
            reader.batch_get_v2([transfers[1]]), {PoolName.INDEXER: [True, False, True]}
        )
        self.assertTrue(torch.all(state.page[1] == 237))
        self.assertTrue(torch.equal(state.page[0], expected[PoolName.INDEXER][0]))
        self.assertEqual(reader.batch_exists_v2(keys, [transfers[1]]).kv_hit_pages, 1)


if __name__ == "__main__":
    unittest.main()
