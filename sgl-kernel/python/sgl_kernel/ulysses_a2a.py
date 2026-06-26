from typing import List

import torch


def init_ulysses_a2a(
    out_ipc_ptrs: List[int],
    signal_ipc_ptrs: List[int],
    rank: int,
    world_size: int,
    full_nvlink: bool,
) -> int:
    return torch.ops.sgl_kernel.init_ulysses_a2a.default(
        out_ipc_ptrs, signal_ipc_ptrs, rank, world_size, full_nvlink
    )


def dispose_ulysses_a2a(fa: int) -> None:
    torch.ops.sgl_kernel.dispose_ulysses_a2a.default(fa)


def ulysses_a2a(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    B: int,
    S_local: int,
    H: int,
    D: int,
    mode: int,
) -> None:
    """Fused-transpose Ulysses all-to-all over NVLink P2P.

    mode == 0: inp [B, S_local, H, D]        -> out [B, S_global, H_local, D]
    mode == 1: inp [B, S_global, H_local, D] -> out [B, S_local, H, D]
    """
    torch.ops.sgl_kernel.ulysses_a2a.default(fa, inp, out, B, S_local, H, D, mode)


def ulysses_a2a_tk(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    B: int,
    S_local: int,
    H: int,
    D: int,
    mode: int,
) -> None:
    """Coalesced task-index decomposition variant."""
    torch.ops.sgl_kernel.ulysses_a2a_tk.default(fa, inp, out, B, S_local, H, D, mode)
