"""The ``page_unified`` page layout and the unified L3 key scheme that names it.

Two halves of one thing: the layout fixes where a chunk's bytes sit inside a
page, and the key scheme names that same chunk as a layer-range x head-range
coordinate. They share a file because they have to agree on the grid.
"""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.device_pool_info import MLABufferInfo
from sglang.srt.mem_cache.hicache_key_scheme import (
    derive_namespace,
    namespace_digest,
    normalize_dtype,
    plan_unified_kv,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mla import (
    NativePageUnifiedMLATokenToKVPoolHost,
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


class TestNativePageUnifiedMLATransfer(CustomTestCase):
    @mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {"cpu": lambda dims, **kw: torch.empty(dims, dtype=kw["dtype"])},
    )
    def test_npu_fia_nz_and_token_layout_share_canonical_records(self):
        """Dim-tiled NPU pages normalize to the same token records and restore."""
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        hosts, infos, originals = [], [], []
        try:
            for nz in (False, True):
                with (
                    mock.patch.dict("os.environ", {"SGLANG_USE_FIA_NZ": str(nz)}),
                    mock.patch(
                        "sglang.srt.hardware_backend.npu.memory_pool_npu.get_parallel",
                        return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
                    ),
                ):
                    pool = NPUMLATokenToKVPool(
                        size=12,
                        page_size=4,
                        dtype=torch.float16,
                        kv_lora_rank=32,
                        qk_rope_head_dim=16,
                        layer_num=2,
                        device="cpu",
                        enable_memory_saver=False,
                    )
                info = pool.get_device_pool_infos()[0].buffer_info
                self.assertEqual(info.is_fia_nz, nz)
                infos.append(info)
                saved = []
                for component, buffer in enumerate((info.buffers.k, info.buffers.v)):
                    logical = (
                        torch.arange(buffer.numel()).reshape_as(buffer) + component * 59
                    ).to(buffer.dtype)
                    if nz:
                        physical = (
                            logical.reshape(2, 4, 4, -1, 16)
                            .transpose(2, 3)
                            .contiguous()
                            .reshape_as(buffer)
                        )
                    else:
                        physical = logical
                    buffer.copy_(physical)
                    saved.append(physical.clone())
                originals.append(saved)
                with host_memory_budget_scope(1 << 20):
                    host = NativePageUnifiedMLATokenToKVPoolHost(
                        info,
                        2,
                        0,
                        4,
                        "page_unified",
                        pin_memory=False,
                        device_capacity_tokens=12,
                        start_layer=0,
                        end_layer=2,
                        target_layer_num=2,
                    )
                hosts.append(host)
                host.backup_from_device_all_layer(
                    None, torch.arange(4), torch.arange(4, 8), "kernel"
                )
            self.assertEqual(hosts[0].storage_format_tag, hosts[1].storage_format_tag)
            pages = [host.get_data_page(0).clone() for host in hosts]
            self.assertTrue(torch.equal(*pages))
            for index, host in enumerate(hosts):
                host.set_from_flat_data_page(0, pages[1 - index])
                for buffer in (infos[index].buffers.k, infos[index].buffers.v):
                    buffer.view(torch.uint8).fill_(173)
                for layer in range(2):
                    host.load_to_device_per_layer(
                        None, torch.arange(4), torch.arange(8, 12), layer, "kernel"
                    )
                for buffer, saved in zip(
                    (infos[index].buffers.k, infos[index].buffers.v), originals[index]
                ):
                    self.assertTrue(
                        torch.equal(
                            buffer[:, 2].view(torch.uint8),
                            saved[:, 1].view(torch.uint8),
                        )
                    )
                    self.assertTrue(
                        torch.all(buffer[:, [0, 1, 3]].view(torch.uint8) == 173)
                    )
        finally:
            for host in hosts:
                host.destroy()

    @mock.patch.dict(
        ALLOC_MEMORY_FUNCS,
        {"cpu": lambda dims, **kw: torch.empty(dims, dtype=kw["dtype"])},
    )
    def test_fp4_mla_file_objects_restore_after_pp_and_tp_changes(self):
        originals = [
            ((torch.arange(3 * 12 * width) + offset) % 251)
            .reshape(3, 12, 1, width)
            .to(torch.uint8)
            for width, offset in ((16, 13), (2, 47))
        ]
        hosts = []

        def host(start, end):
            payload = tuple(originals[0][start:end].clone())
            scales = tuple(originals[1][start:end, :, 0].clone())
            info = MLABufferInfo(
                buffers=payload,
                scales=scales,
                data_ptrs=None,
                layer_indices=tuple(range(end - start)),
                device_layer_num=end - start,
                page_size=2,
                kv_lora_rank=16,
                qk_rope_head_dim=16,
                kv_cache_dim=32,
                dtype=torch.uint8,
                device=torch.device("cpu"),
            )
            self.assertIs(
                get_mla_host_pool_cls(info, layout="page_unified"),
                NativePageUnifiedMLATokenToKVPoolHost,
            )
            with host_memory_budget_scope(1 << 20):
                result = NativePageUnifiedMLATokenToKVPoolHost(
                    info,
                    2,
                    0,
                    2,
                    "page_unified",
                    pin_memory=False,
                    device_capacity_tokens=10,
                    start_layer=start,
                    end_layer=end,
                    target_layer_num=end - start,
                )
            hosts.append(result)
            return result, (payload, scales)

        def backend(pool, directory, tp_size):
            plan = plan_unified_kv(
                model_id=storage_model_name("org/model", pool),
                dtype="fp4",
                page_size=2,
                rank_replicated=True,
                local_kv_heads=0,
                attn_tp_rank=0,
                attn_tp_size=tp_size,
                attn_cp_size=1,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                is_final_stage=pool.end_layer == 3,
                layer_partition=1,
            )
            return HiCacheFile(
                HiCacheStorageConfig(
                    tp_rank=0,
                    tp_size=tp_size,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=True,
                    enable_storage_metrics=False,
                    is_page_first_layout=False,
                    model_name="org/model",
                    unified_suffixes=plan.suffixes,
                    unified_chunk_spans=[
                        pool.page_unified_layout.chunk_span(window)
                        for window in plan.layer_ranges
                    ],
                    extra_config={"max_size": 0, "min_free_space": 0},
                ),
                file_path=directory,
            )

        try:
            with TemporaryDirectory() as directory:
                writer, _ = host(0, 3)
                writer.backup_from_device_all_layer(
                    None, torch.arange(2), torch.arange(2, 4), "direct"
                )
                self.assertTrue(
                    backend(writer, directory, 1).set("prefix", writer.get_data_page(0))
                )
                for start, end in ((0, 1), (1, 3)):
                    reader, components = host(start, end)
                    self.assertEqual(
                        writer.storage_format_tag, reader.storage_format_tag
                    )
                    data = reader.get_dummy_flat_data_page()
                    self.assertIsNotNone(
                        backend(reader, directory, 4).get("prefix", data)
                    )
                    reader.set_from_flat_data_page(2, data)
                    for buffers in components:
                        for buffer in buffers:
                            buffer.fill_(173)
                    for layer in range(end - start):
                        reader.load_to_device_per_layer(
                            None,
                            torch.arange(2, 4),
                            torch.arange(6, 8),
                            layer,
                            "kernel",
                        )
                    for buffers, saved in zip(components, originals):
                        for local, layer in enumerate(range(start, end)):
                            self.assertTrue(
                                torch.equal(
                                    buffers[local][6:8].reshape(2, -1),
                                    saved[layer, 2:4].reshape(2, -1),
                                )
                            )
                            self.assertTrue(torch.all(buffers[local][:6] == 173))
        finally:
            for pool in hosts:
                pool.destroy()


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
