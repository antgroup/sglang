"""The SM90 head<->row all-to-all (deepseek_v4_head_a2a), backend side: attending
the row slices of a decode batch (`row_slice`, including the all-to-all's pad
rows) through `DeepseekV4AttnBackend.forward` matches attending the whole batch.

The ranks are simulated one after another on one GPU; the exchange itself is
covered on CPU by test/registered/unit/models/test_deepseek_v4_head_a2a.py.
"""

import importlib.util
import unittest

import torch

from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
    DSV4_PAGE_SIZE,
    DSV4AttentionCase,
    _populate_extra_kv_cache,
    _populate_swa_kv_cache,
    build_dsv4_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

_FLASH_MLA_AVAILABLE = (
    importlib.util.find_spec("sgl_kernel") is not None
    and importlib.util.find_spec("sgl_kernel.flash_mla") is not None
)
_EXTRA_ENTRIES = 32
# 13 requests: no multiple of the group sizes below, so the last slices hold
# pad rows (all of them for 8 ranks).
_PREFIX_LENS = tuple(40 + 9 * i for i in range(13))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(not _FLASH_MLA_AVAILABLE, "flash_mla is required for DSV4")
class TestDSV4DecodeHeadA2aRowSlice(CustomTestCase):
    def _check(self, compress_ratio: int, world: int):
        case = DSV4AttentionCase(
            name=f"dsv4_head_a2a_c{compress_ratio}_w{world}",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=_PREFIX_LENS,
            compress_ratio=compress_ratio,
            attn_sink_value=0.25,
        )
        # The full pool spans the SWA locs; 13 requests need more than the default.
        fixture = build_dsv4_attention_fixture(
            self, case, compression_ratios=[compress_ratio], swa_size=4096
        )
        backend, module = fixture.backend, fixture.actual_module
        _populate_swa_kv_cache(
            fixture,
            max_context_len=fixture.runner.req_to_token_pool.req_to_token.shape[1],
            device="cuda",
        )
        if compress_ratio:
            _populate_extra_kv_cache(fixture, layer_id=0, num_entries=_EXTRA_ENTRIES)
        q, _ = module.project(fixture.input_hidden)
        rows = q.shape[0]
        chunk = -(-rows // world)
        q_padded = q.new_zeros(chunk * world, *q.shape[1:])
        q_padded[:rows] = q

        def attend(q, row_slice=None):
            # One schedule per call: in production a rank attends one slice.
            setattr(
                backend.forward_metadata.core_attn_metadata,
                f"c{compress_ratio}_flashmla_metadata",
                deepseek_v4_backend._create_flashmla_metadata(),
            )
            return backend.forward(
                q=q,
                k=q,
                v=q,
                layer=module.attn,
                forward_batch=fixture.forward_batch,
                compress_ratio=compress_ratio,
                save_kv_cache=False,
                attn_sink=module.attn_sink,
                **({} if row_slice is None else {"row_slice": row_slice}),
            )

        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            backend.init_forward_metadata(fixture.forward_batch)
            if compress_ratio:
                self._seed_extra_indices(backend, compress_ratio)
            expected = attend(q)
            actual = torch.cat(
                [
                    attend(
                        q_padded[r * chunk : (r + 1) * chunk],
                        (r * chunk, (r + 1) * chunk, rows),
                    )
                    for r in range(world)
                ]
            )

        # Pad rows attend dummy metadata: finite, then dropped by the model.
        self.assertTrue(torch.isfinite(actual).all())
        # Not bitwise: FlashMLA's split-KV schedule depends on the row count.
        torch.testing.assert_close(
            actual[:rows].float(), expected.float(), atol=1e-2, rtol=1e-2
        )

    @staticmethod
    def _seed_extra_indices(backend, ratio: int):
        """The fixture runs no indexer: give every row the populated entries,
        with per-row lengths so that slicing the lengths matters too."""
        md = backend.forward_metadata.core_metadata
        indices = md.sparse_page_indices(ratio)
        seeded = torch.full_like(indices, -1)
        seeded[:, :_EXTRA_ENTRIES] = torch.arange(
            _EXTRA_ENTRIES, dtype=indices.dtype, device=indices.device
        )
        lengths = _EXTRA_ENTRIES - torch.arange(
            indices.shape[0], device=indices.device
        ).remainder(8)
        md.set_sparse_topk(
            ratio,
            page_indices=seeded,
            topk_lengths=lengths.to(md.sparse_topk_lengths(ratio).dtype),
        )

    def test_swa_only(self):
        for world in (4, 8):
            with self.subTest(world=world):
                self._check(0, world)

    def test_c4(self):
        for world in (4, 8):
            with self.subTest(world=world):
                self._check(4, world)


if __name__ == "__main__":
    unittest.main()
