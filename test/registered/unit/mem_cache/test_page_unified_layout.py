"""The ``page_unified`` page layout and the unified L3 key scheme that names it.

Two halves of one thing: the layout fixes where a chunk's bytes sit inside a
page, and the key scheme names that same chunk as a layer-range x head-range
coordinate. They share a file because they have to agree on the grid.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.device_pool_info import MLABufferInfo
from sglang.srt.mem_cache.hicache_key_scheme import (
    derive_namespace,
    namespace_digest,
    normalize_dtype,
    plan_pool_components,
    plan_unified_kv,
)
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mla import (
    MLATokenToKVPoolHost,
    get_mla_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mha(**overrides):
    kwargs = dict(
        page_size=4,
        layer_num=6,
        head_num=4,
        head_group_num=2,
        head_dim=8,
        itemsize=2,
    )
    kwargs.update(overrides)
    return PageUnifiedLayout(**kwargs)


class TestPageUnifiedLayout(CustomTestCase):
    def test_chunks_tile_the_page_exactly_once(self):
        """The grid's chunks must partition a page: no gap, no overlap.

        A gap is KV that is never stored; an overlap is two keys naming the
        same bytes, which diverge as soon as one is evicted.
        """
        layout = _mha(layer_num=6, head_num=3, head_group_num=3)
        covered = bytearray(layout.bytes_per_page)
        for head_group in range(layout.head_group_num):
            for start in range(0, layout.layer_num, 2):
                offset, length = layout.chunk_span((start, start + 2), head_group)
                for byte in range(offset, offset + length):
                    covered[byte] += 1
        self.assertEqual(set(covered), {1})

    def test_ragged_layer_window_is_still_one_run(self):
        """A short final window is one run, which is the point of the axis order.

        DeepSeek-V3's 61 layers at layer_partition=30 leave a width-1 tail. If
        that tail were several runs the transport would have to switch
        descriptor forms mid-plan.
        """
        layout = _mha(layer_num=61, head_num=4, head_group_num=1)
        spans = [layout.chunk_span(r) for r in [(0, 30), (30, 60), (60, 61)]]
        self.assertEqual(
            spans,
            [
                (0, 30 * layout.layer_bytes),
                (30 * layout.layer_bytes, 30 * layout.layer_bytes),
                (60 * layout.layer_bytes, layout.layer_bytes),
            ],
        )
        self.assertEqual(sum(length for _, length in spans), layout.bytes_per_page)

    def test_unaligned_group_row_is_rejected(self):
        """The transfer kernels move 16-byte vectors and cannot express a tail."""
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            _mha(head_num=2, head_group_num=2, head_dim=7, itemsize=1)

    def test_head_group_must_divide_the_pools_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            PageUnifiedLayout(
                page_size=4,
                layer_num=2,
                head_num=6,
                head_group_num=4,
                head_dim=16,
                itemsize=2,
            )

    def test_chunk_span_rejects_out_of_range_coordinates(self):
        layout = _mha()
        with self.assertRaisesRegex(ValueError, "layer range"):
            layout.chunk_span((0, layout.layer_num + 1))
        with self.assertRaisesRegex(ValueError, "layer range"):
            layout.chunk_span((3, 3))
        with self.assertRaisesRegex(ValueError, "head group"):
            layout.chunk_span((0, 1), layout.head_group_num)


class TestPageUnifiedMLATransfer(CustomTestCase):
    def test_page_unified_rejects_split_or_scaled_mla(self):
        from sglang.srt.mem_cache.device_pool_info import MLASplitBuffers

        latent = torch.zeros(12, 1, 16)
        split = MLASplitBuffers(
            k=latent.reshape(3, 4, 1, 16),
            v=None,
            index_k=None,
            index_scales=None,
            index_layer_indices=(),
            k_width=16,
            v_width=0,
            index_width=0,
        )
        for buffers, scales in (((latent,), (latent,)), (split, None)):
            with self.subTest(split=isinstance(buffers, MLASplitBuffers)):
                info = MLABufferInfo(
                    buffers=buffers,
                    data_ptrs=None,
                    layer_indices=(0,),
                    device_layer_num=1,
                    page_size=4,
                    kv_lora_rank=8,
                    qk_rope_head_dim=8,
                    kv_cache_dim=16,
                    dtype=latent.dtype,
                    device=latent.device,
                    scales=scales,
                )
                with self.assertRaisesRegex(NotImplementedError, "page_unified"):
                    get_mla_host_pool_cls(info, layout="page_unified")
                with self.assertRaisesRegex(NotImplementedError, "page_unified"):
                    MLATokenToKVPoolHost(
                        info,
                        2,
                        0,
                        4,
                        "page_unified",
                        pin_memory=False,
                        device_capacity_tokens=8,
                        start_layer=0,
                        end_layer=1,
                        target_layer_num=1,
                    )

    @mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {"cpu": lambda dims, **kw: torch.empty(dims, dtype=kw["dtype"])},
    )
    def test_replicated_mla_objects_restore_across_tp_and_sparse_pp(self):
        """A stage without index rows reuses the same latent objects as PP1."""
        for writer_tp, reader_tp in ((2, 4), (4, 2)):
            with self.subTest(mode="canonical", writer_tp=writer_tp):
                size, tokens = (4, 16)

                def logical(layer, width, component):
                    return (
                        (
                            torch.arange(tokens * width).reshape(tokens, 1, width)
                            + layer * 53
                            + component * 79
                        )
                        .remainder(251)
                        .to(torch.uint8)
                    )

                def make(layers, rank, tp, *, empty):

                    def rows(width, component, *, paged=False, indices=None):
                        indices = range(len(layers)) if indices is None else indices
                        data = tuple(
                            (logical(layers[i], width, component) for i in indices)
                        )
                        if paged:
                            data = tuple((x.reshape(4, size, 1, width) for x in data))
                        return data

                    buffers = rows(32, 0)
                    scales = None
                    observed_rows = [*buffers, *(scales or ())]
                    observable = [
                        (row, row.clone(), "canonical" == "split")
                        for row in observed_rows
                    ]
                    if empty:
                        for row in observed_rows:
                            row.fill_(173)
                    info = MLABufferInfo(
                        buffers=buffers,
                        data_ptrs=torch.tensor(
                            [x.data_ptr() for x in buffers], dtype=torch.uint64
                        ),
                        layer_indices=tuple(range(len(layers))),
                        device_layer_num=len(layers),
                        page_size=size,
                        kv_lora_rank=24,
                        qk_rope_head_dim=8,
                        kv_cache_dim=32,
                        dtype=torch.uint8,
                        device=torch.device("cpu"),
                        scales=scales,
                    )
                    cls = get_mla_host_pool_cls(info, layout="page_unified")
                    with host_memory_budget_scope(1 << 20):
                        host = cls(
                            info,
                            2,
                            0,
                            size,
                            "page_unified",
                            pin_memory=False,
                            device_capacity_tokens=tokens - size,
                            start_layer=layers[0],
                            end_layer=layers[-1] + 1,
                            target_layer_num=len(layers),
                        )
                    plan = plan_pool_components(
                        model_id=f"replicated-mla-{'canonical'}",
                        pool_name="kv",
                        object_layout=getattr(host, "storage_format_tag", None)
                        or "page-unified-v1",
                        page_size=size,
                        components=host.storage_component_metadata,
                        layer_ids=tuple((f"L{layer}" for layer in layers)),
                        tp_rank=rank,
                        tp_size=tp,
                    )
                    return (host, plan, observable)

                source, host_indices, target = (
                    torch.arange(4, 8),
                    torch.arange(4),
                    torch.arange(8, 12),
                )
                writer, plan, _ = make((0, 1, 2), 0, writer_tp, empty=False)
                try:
                    writer.backup_from_device_all_layer(
                        None, host_indices, source, "kernel"
                    )
                    page = writer.get_data_page(0).view(torch.uint8)
                    for component in writer.storage_component_metadata:
                        layer = component.layer_offset
                        fields = ((32, 0),)
                        reference = torch.cat(
                            [
                                logical(layer, width, field)[source, 0]
                                for width, field in fields
                            ],
                            dim=-1,
                        ).flatten()
                        self.assertTrue(
                            torch.equal(
                                page[
                                    component.byte_offset : component.byte_offset
                                    + component.byte_length
                                ],
                                reference,
                            )
                        )
                    objects = {
                        key: page[offset : offset + length].clone()
                        for key, (offset, length) in zip(
                            plan.suffixes, plan.chunk_spans
                        )
                    }
                finally:
                    writer.destroy()
                consumed = set()
                for layers in ((0,), (1, 2)):
                    for rank in range(reader_tp):
                        reader, plan, observable = make(
                            layers, rank, reader_tp, empty=True
                        )
                        try:
                            page = reader.get_dummy_flat_data_page().view(torch.uint8)
                            for key, (offset, length) in zip(
                                plan.suffixes, plan.chunk_spans
                            ):
                                page[offset : offset + length].copy_(objects[key])
                                consumed.add(key)
                            reader.set_from_flat_data_page(0, page)
                            for layer in range(len(layers)):
                                reader.load_to_device_per_layer(
                                    None, host_indices, target, layer, "kernel"
                                )
                            for actual, saved, paged in observable:
                                self.assertTrue(
                                    torch.equal(
                                        actual[2] if paged else actual[target],
                                        saved[1] if paged else saved[source],
                                    )
                                )
                                guard = torch.ones(actual.shape[0], dtype=torch.bool)
                                guard[2 if paged else target] = False
                                self.assertTrue(torch.all(actual[guard] == 173))
                        finally:
                            reader.destroy()
                self.assertEqual(consumed, objects.keys())

    @mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {
            "cpu": lambda dims, **kw: torch.empty(
                dims, dtype=kw["dtype"], device=kw["device"]
            )
        },
    )
    def test_mla_pages_preserve_owned_layers_and_drafts(self):
        """A page restores only its owned MLA layers, with all packed scales.

        CP layer holes and reserved host rows cannot shift draft layers, and
        sparse split index components cannot be inferred from latent width.
        """
        with self.subTest(native_format="canonical"):
            infos, observable = ([], [])
            for layer_ids in ((2, 4), (0,)):
                count = len(layer_ids)
                dtype = torch.float16
                width = 8
                backing = torch.empty(12, count + 1, 1, width, dtype=dtype)
                backing.view(torch.uint8).random_(0, 256)
                buffers = tuple((backing[:, layer] for layer in range(count)))
                observable.extend(((row, False) for row in buffers))
                scales = None
                infos.append(
                    MLABufferInfo(
                        buffers=buffers,
                        data_ptrs=torch.tensor(
                            [b.data_ptr() for b in buffers], dtype=torch.uint64
                        ),
                        layer_indices=layer_ids,
                        device_layer_num=5 if count == 2 else 1,
                        page_size=2,
                        kv_lora_rank=4,
                        qk_rope_head_dim=4,
                        kv_cache_dim=8,
                        dtype=dtype,
                        device=torch.device("cpu"),
                        scales=scales,
                    )
                )
            host_cls = get_mla_host_pool_cls(infos[0], layout="page_unified")
            self.assertIs(host_cls, MLATokenToKVPoolHost)
            with host_memory_budget_scope(1 << 20):
                writer, reader = [
                    host_cls(
                        infos[0],
                        2,
                        0,
                        2,
                        "page_unified",
                        pin_memory=False,
                        device_capacity_tokens=8,
                        start_layer=0,
                        end_layer=5,
                        target_layer_num=3,
                        layer_shard_enabled=True,
                        draft_buffer_infos=(infos[1],),
                    )
                    for _ in range(2)
                ]
            try:
                host_indices, source_indices = (
                    torch.tensor([6, 7, 0, 1]),
                    torch.tensor([4, 5, 8, 9]),
                )
                writer.backup_from_device_all_layer(
                    None, host_indices, source_indices, "kernel"
                )
                for source, target in ((6, 4), (0, 2)):
                    reader.set_from_flat_data_page(
                        target, writer.get_data_page(source).clone()
                    )
                saved = []
                for buffer, paged in observable:
                    source_rows = torch.tensor([2, 4]) if paged else source_indices
                    saved.append(
                        buffer.view(torch.uint8).index_select(0, source_rows).clone()
                    )
                    buffer.view(torch.uint8).fill_(173)
                reader_indices, restore_indices = (
                    torch.tensor([4, 5, 2, 3]),
                    torch.tensor([0, 1, 6, 7]),
                )
                reader.load_to_device_per_layer(
                    None, reader_indices, restore_indices, 3, "kernel"
                )
                self.assertTrue(
                    all(
                        (
                            torch.all(buffer.view(torch.uint8) == 173)
                            for buffer, _ in observable
                        )
                    )
                )
                for layer in (2, 4, 5):
                    reader.load_to_device_per_layer(
                        None,
                        reader_indices,
                        restore_indices,
                        layer,
                        "kernel",
                        is_draft=layer == 5,
                    )
                for (buffer, paged), expected in zip(observable, saved):
                    rows = torch.tensor([0, 3]) if paged else restore_indices
                    self.assertTrue(
                        torch.equal(
                            buffer.view(torch.uint8).index_select(0, rows), expected
                        )
                    )
                    unselected = torch.ones(buffer.shape[0], dtype=torch.bool)
                    unselected[rows] = False
                    self.assertTrue(
                        torch.all(buffer.view(torch.uint8)[unselected] == 173)
                    )
            finally:
                writer.destroy()
                reader.destroy()


# GQA-70B shape: 8 kv heads, 80 layers.
BASE = dict(
    model_id="meta-llama/Llama-3.1-70B",
    dtype="bfloat16",
    page_size=64,
    rank_replicated=False,
    attn_cp_size=1,
    start_layer=0,
    end_layer=80,
    is_final_stage=True,
)


def _plan(**overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    return plan_unified_kv(**kwargs)


class TestUnifiedKeyGrid(CustomTestCase):
    def test_tp_shards_of_one_grid_cover_each_other(self):
        """TP2's objects are exactly the union of TP4's, in order.

        This is the whole point of the scheme: a cache written by one topology
        must be readable by another. Nothing else in the stack checks it, and
        the failure mode is a silent 0% hit rate.
        """
        tp2 = [
            _plan(local_kv_heads=4, attn_tp_rank=r, attn_tp_size=2, head_group=2)
            for r in range(2)
        ]
        tp4 = [
            _plan(local_kv_heads=2, attn_tp_rank=r, attn_tp_size=4, head_group=2)
            for r in range(4)
        ]
        self.assertEqual(
            [s for p in tp2 for s in p.suffixes],
            [s for p in tp4 for s in p.suffixes],
        )

    def test_pp_stages_cover_the_unsplit_range(self):
        """Same property along the layer axis, including a ragged tail."""
        whole = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            end_layer=61,
        )
        stage0 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            start_layer=0,
            end_layer=30,
            is_final_stage=False,
        )
        stage1 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            start_layer=30,
            end_layer=61,
        )
        self.assertEqual(whole.suffixes, stage0.suffixes + stage1.suffixes)
        # The 61st layer forms a short trailing chunk rather than being dropped.
        self.assertTrue(whole.suffixes[-1].endswith("_L60-61_H0"))

    def test_suffix_order_matches_the_range_cross_product(self):
        """Layer-major / head-minor, and the ranges pair with it positionally.

        The storage layer zips suffixes against layer_ranges x head_ranges to
        find each object's bytes; a reordering here writes every chunk under
        its neighbour's key.
        """
        plan = _plan(
            local_kv_heads=4,
            attn_tp_rank=0,
            attn_tp_size=2,
            head_group=2,
            layer_partition=40,
        )
        self.assertEqual(len(plan.layer_ranges), 2)
        self.assertEqual(plan.head_ranges, [(0, 2), (2, 4)])
        digest = namespace_digest(plan.namespace)
        self.assertEqual(
            plan.suffixes,
            [
                f"{digest}_L0-40_H0",
                f"{digest}_L0-40_H1",
                f"{digest}_L40-80_H0",
                f"{digest}_L40-80_H1",
            ],
        )

    def test_head_index_is_global_not_rank_local(self):
        """Rank 1's groups must not collide with rank 0's."""
        plans = [
            _plan(local_kv_heads=2, attn_tp_rank=r, attn_tp_size=4, head_group=2)
            for r in range(4)
        ]
        self.assertEqual(
            [p.suffixes[0].rsplit("_", 1)[1] for p in plans],
            ["H0", "H1", "H2", "H3"],
        )

    def test_local_ranges_are_pool_relative(self):
        """A PP stage indexes its own pool, so its ranges start at 0."""
        stage1 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=20,
            start_layer=40,
            end_layer=80,
        )
        self.assertEqual(stage1.layer_ranges, [(0, 20), (20, 40)])
        self.assertTrue(stage1.suffixes[0].endswith("_L40-60_H0"))

    def test_attention_cp_preserves_head_shards_and_namespace(self):
        for rank_replicated in (False, True):
            for tp_rank in range(2):
                with self.subTest(rank_replicated=rank_replicated, tp_rank=tp_rank):
                    shard = dict(
                        rank_replicated=rank_replicated,
                        local_kv_heads=4,
                        attn_tp_rank=tp_rank,
                        attn_tp_size=2,
                        head_group=4,
                    )
                    self.assertEqual(
                        _plan(**shard, attn_cp_size=4),
                        _plan(**shard, attn_cp_size=1),
                    )


class TestNamespaceIdentity(CustomTestCase):
    def test_identity_fields_partition_the_keyspace(self):
        """Every identity field must change the digest.

        A field that does not reach the digest lets two deployments with
        different bytes collide on one key, which is corruption rather than a
        miss.
        """
        base = dict(
            model_id="m",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
            head_group=2,
        )
        digests = {namespace_digest(derive_namespace(**base))}
        for field, value in [
            ("model_id", "other"),
            ("dtype", "float16"),
            ("page_size", 32),
            ("total_kv_heads", 16),
            ("head_group", 4),
            ("layer_partition", 30),
            ("object_layout", "something-else"),
        ]:
            digests.add(namespace_digest(derive_namespace(**{**base, field: value})))
        self.assertEqual(len(digests), 8)

    def test_fp8_variants_do_not_share_a_keyspace(self):
        """Both store as uint8, so only the logical dtype separates them."""
        self.assertNotEqual(
            namespace_digest(
                _plan(
                    local_kv_heads=8,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    head_group=8,
                    dtype="float8_e4m3fn",
                ).namespace
            ),
            namespace_digest(
                _plan(
                    local_kv_heads=8,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    head_group=8,
                    dtype="float8_e5m2",
                ).namespace
            ),
        )

    def test_rank_replicated_drops_the_head_axis(self):
        plan = _plan(
            rank_replicated=True,
            local_kv_heads=0,
            attn_tp_rank=3,
            attn_tp_size=8,
        )
        self.assertEqual(plan.head_ranges, [(0, 1)])
        self.assertNotIn("_H", plan.suffixes[0])
        # Every rank derives the same keys, which is what makes one writer enough.
        other = _plan(
            rank_replicated=True, local_kv_heads=0, attn_tp_rank=0, attn_tp_size=8
        )
        self.assertEqual(plan.suffixes, other.suffixes)

    def test_normalize_dtype_strips_the_torch_prefix(self):
        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")


class TestUnifiedKeySchemeRejections(CustomTestCase):
    def test_head_group_must_tile_this_ranks_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide this rank's"):
            _plan(local_kv_heads=3, attn_tp_rank=0, attn_tp_size=2, head_group=2)

    def test_misaligned_stage_start_is_rejected(self):
        """A stage starting off-grid would name chunks no reader can assemble."""
        with self.assertRaisesRegex(ValueError, "does not start on a multiple"):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                layer_partition=30,
                start_layer=20,
                end_layer=50,
            )

    def test_non_final_stage_may_not_end_short(self):
        with self.assertRaisesRegex(ValueError, "only the FINAL pipeline stage"):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                layer_partition=30,
                start_layer=0,
                end_layer=50,
                is_final_stage=False,
            )

    def test_one_kv_head_per_rank_needs_an_explicit_head_group(self):
        """1 head per rank is ambiguous: sharded, or replicated across ranks?

        Replication would have several ranks racing to write one key, so the
        operator has to attest which it is.
        """
        with self.assertRaises(NotImplementedError):
            _plan(local_kv_heads=1, attn_tp_rank=0, attn_tp_size=8)

    def test_empty_model_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-empty model_id"):
            _plan(
                model_id="",
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
            )


if __name__ == "__main__":
    unittest.main()
