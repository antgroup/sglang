"""Unit tests for cuLA SM90 KDA prefill kernel.

Compares cuLA kernel output against Triton chunk_kda reference.
"""

import pytest
import torch

# Skip all tests if not on SM90 GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="cuLA KDA requires SM90 (Hopper) GPU",
)


def _try_import_cula():
    try:
        from sgl_kernel import kda_fwd_prefill

        return kda_fwd_prefill
    except ImportError:
        pytest.skip("sgl_kernel.kda_fwd_prefill not available (cula_kda_ops not built)")


def _try_import_triton_ref():
    try:
        from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
        from sglang.srt.layers.attention.fla.kda import chunk_kda
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        return chunk_kda, l2norm_fwd, chunk_local_cumsum
    except ImportError:
        pytest.skip("Triton reference (sglang) not available")


def _make_cu_seqlens(batch_size, seq_len, device):
    """Create uniform cu_seqlens for batch_size sequences of seq_len."""
    seqlens = [seq_len] * batch_size
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seqlens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l
    return cu_seqlens


def _run_cula_vs_triton(B, T, H, D, device="cuda"):
    """Run cuLA and Triton KDA, compare outputs."""
    kda_fwd_prefill = _try_import_cula()
    chunk_kda, l2norm_fwd, chunk_local_cumsum = _try_import_triton_ref()

    torch.manual_seed(42)
    packed_seq = B * T

    # Generate inputs in [1, packed_seq, H, D] format (SGLang convention)
    q = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, packed_seq, H, D, dtype=torch.bfloat16, device=device)
    # Gate values in safe range [-5, 0)
    g = torch.rand(1, packed_seq, H, D, dtype=torch.bfloat16, device=device) * (-4.9)
    beta = torch.randn(1, packed_seq, H, dtype=torch.bfloat16, device=device)

    # Initial state in KV layout [N, H, K, V] for cuLA
    initial_state_kv = (
        torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01
    )
    # Same state in VK layout [N, H, V, K] for Triton
    initial_state_vk = initial_state_kv.transpose(-1, -2).contiguous()

    cu_seqlens = _make_cu_seqlens(B, T, device)

    # --- Triton reference ---
    triton_out = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone().contiguous(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=initial_state_vk.clone(),
        initial_state_indices=torch.arange(B, device=device),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens.long(),
    )

    # --- cuLA kernel ---
    # Preprocess: l2norm Q, K
    q_norm = l2norm_fwd(q.clone().contiguous())
    k_norm = l2norm_fwd(k.clone().contiguous())

    # Gate cumsum
    g_cum = chunk_local_cumsum(g.clone(), chunk_size=64, cu_seqlens=cu_seqlens.long())

    # Reshape for C++ kernel: [packed_seq, H, D]
    q_packed = q_norm.reshape(packed_seq, H, D).contiguous()
    k_packed = k_norm.reshape(packed_seq, H, D).contiguous()
    v_packed = v.reshape(packed_seq, H, D).contiguous()
    g_packed = g_cum.reshape(packed_seq, H, D).contiguous()
    beta_packed = beta.reshape(packed_seq, H).contiguous()

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sm_count * 128, dtype=torch.uint8, device=device)

    scale = D**-0.5

    cula_output, cula_state = kda_fwd_prefill(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        cu_seqlens=cu_seqlens,
        workspace_buffer=workspace,
        scale=scale,
        safe_gate=True,
        input_state=initial_state_kv.clone(),
        alpha=g_packed,
        beta=beta_packed,
    )

    # Reshape cuLA output back to [1, packed_seq, H, D]
    cula_output = cula_output.reshape(1, packed_seq, H, D)

    # Compare outputs
    # Use relaxed tolerance for bf16 + fused kernel differences
    atol = 5e-2
    rtol = 5e-2
    torch.testing.assert_close(cula_output, triton_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B,T,H,D",
    [
        (1, 63, 1, 128),
        (2, 500, 3, 128),
        (4, 1024, 4, 128),
        (4, 2048, 8, 128),
    ],
)
def test_cula_vs_triton(B, T, H, D):
    _run_cula_vs_triton(B, T, H, D)


def test_cula_varlen():
    """Test with variable-length sequences."""
    kda_fwd_prefill = _try_import_cula()

    torch.manual_seed(42)
    device = "cuda"
    H, D = 4, 128
    seqlens = [63, 128, 256]
    B = len(seqlens)
    packed_seq = sum(seqlens)

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seqlens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l

    q = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    g = torch.rand(packed_seq, H, D, dtype=torch.float32, device=device) * (-4.9)
    beta = torch.randn(packed_seq, H, dtype=torch.float32, device=device)
    initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sm_count * 128, dtype=torch.uint8, device=device)

    scale = D**-0.5

    output, output_state = kda_fwd_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        workspace_buffer=workspace,
        scale=scale,
        safe_gate=True,
        input_state=initial_state,
        alpha=g,
        beta=beta,
    )

    # Basic shape checks
    assert output.shape == (packed_seq, H, D)
    assert output_state.shape == (B, H, D, D)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


def test_cula_no_initial_state():
    """Test without initial state (should allocate zeros internally)."""
    kda_fwd_prefill = _try_import_cula()

    torch.manual_seed(42)
    device = "cuda"
    B, T, H, D = 2, 256, 4, 128
    packed_seq = B * T

    cu_seqlens = _make_cu_seqlens(B, T, device)
    q = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(packed_seq, H, D, dtype=torch.bfloat16, device=device)
    g = torch.rand(packed_seq, H, D, dtype=torch.float32, device=device) * (-4.9)
    beta = torch.randn(packed_seq, H, dtype=torch.float32, device=device)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sm_count * 128, dtype=torch.uint8, device=device)

    scale = D**-0.5

    output, output_state = kda_fwd_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        workspace_buffer=workspace,
        scale=scale,
        safe_gate=True,
        alpha=g,
        beta=beta,
    )

    assert output.shape == (packed_seq, H, D)
    assert output_state.shape == (B, H, D, D)
    assert not torch.isnan(output).any(), "Output contains NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
