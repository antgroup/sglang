"""Decode CUDA graph batch sizes under CP decode attention TP (V4.1, TP8 + prefill CP8)."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import padding
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import get_context, get_flags
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_CAPTURE_BS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64]
# DSpark block size 5: one verify forward is 6 rows per request.
_VERIFY_ROWS_PER_REQ = 6


def _tp8_cp8(**overrides):
    # TP8 with prefill CP8 and CP decode attention TP, no DP attention.
    fields = dict(
        enable_cp_decode_attn_tp=True,
        attn_cp_size=8,
        attn_tp_size=1,
        attn_dp_size=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def capture(monkeypatch):
    override = get_context().override_server_args(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="default", bs=list(_CAPTURE_BS))
        ),
        enable_torch_compile=False,
        enable_two_batch_overlap=False,
    )
    override.install()
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(model_impl="auto", dllm_algorithm=None),
        req_to_token_pool=SimpleNamespace(size=64),
    )

    def run(*, parallel, subsystems=frozenset({"mix_decode"}), role="mix", kind=None):
        monkeypatch.setattr(common, "get_parallel", lambda: parallel)
        monkeypatch.setattr(padding, "get_parallel", lambda: parallel)
        monkeypatch.setattr(
            cp_base,
            "get_cp_strategy_kind",
            lambda: kind or cp_base.ContextParallelStrategyKind.INTERLEAVE,
        )
        with get_flags().dsv41_sm90.override(subsystems=subsystems, role=role):
            capture_bs, _ = get_batch_sizes_to_capture(
                model_runner, _VERIFY_ROWS_PER_REQ, gathered_buffer_required=False
            )
        return capture_bs

    try:
        yield run
    finally:
        override.restore()


def test_mix_decode_keeps_logical_batch_sizes(capture):
    """A single request must replay a bs=1 graph, not the bs=4 one (24 rows)."""
    assert capture(parallel=_tp8_cp8()) == _CAPTURE_BS


def test_other_selections_keep_the_cp_alignment(capture):
    # 6 rows per request: bs * 6 % 8 == 0 only for multiples of 4.
    aligned = [bs for bs in _CAPTURE_BS if bs % 4 == 0]
    for kwargs in (
        dict(subsystems=frozenset(), role=""),
        dict(subsystems=frozenset({"cp", "decode"})),
        dict(role="decode"),
        dict(role="prefill"),
        dict(parallel=_tp8_cp8(enable_cp_decode_attn_tp=False)),
        dict(parallel=_tp8_cp8(attn_dp_size=2)),
        dict(parallel=_tp8_cp8(attn_tp_size=2)),
        dict(kind=cp_base.ContextParallelStrategyKind.ZIGZAG),
    ):
        kwargs.setdefault("parallel", _tp8_cp8())
        assert capture(**kwargs) == aligned, kwargs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
