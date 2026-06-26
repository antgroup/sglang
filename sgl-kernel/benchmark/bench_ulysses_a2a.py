"""Latency benchmark for fused NVLink-P2P Ulysses all-to-all.

Run with:
    python sgl-kernel/benchmark/bench_ulysses_a2a.py
"""

import ctypes
import multiprocessing as mp
import os
import socket
from typing import List, Optional

import sgl_kernel  # noqa: F401
import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

_SHAPES = [
    (1, 1024, 24, 128),
    (1, 4096, 24, 128),
    (1, 16384, 40, 128),
    (2, 8192, 40, 128),
]
_DTYPE = torch.bfloat16
_WARMUP = 10
_ITERS = 50


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup]
) -> List[int]:
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    input_tensor = torch.ByteTensor(list(handle_bytes)).to(f"cuda:{rank}")
    gathered = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, input_tensor, group=group)
    pointers: List[int] = []
    for i, tensor in enumerate(gathered):
        if i == rank:
            pointers.append(pointer.value)
        else:
            handle_obj = type(handle)()
            ctypes.memmove(
                ctypes.addressof(handle_obj),
                bytes(tensor.cpu().tolist()),
                ctypes.sizeof(handle_obj),
            )
            pointers.append(lib.cudaIpcOpenMemHandle(handle_obj).value)
    dist.barrier(group=group)
    return pointers


def ref_input_a2a(x: torch.Tensor, world_size: int, group) -> torch.Tensor:
    b, s_local, h_global, d = x.shape
    h_local, s_global = h_global // world_size, s_local * world_size
    xx = x.permute(2, 0, 1, 3).contiguous().flatten().contiguous()
    out = torch.empty_like(xx)
    dist.all_to_all_single(out, xx, group=group)
    out = out.reshape(world_size, h_local, b, s_local, d)
    return out.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_global, h_local, d)


def _bench_fn(fn) -> float:
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / _ITERS * 1000.0


def _worker(world_size: int, rank: int, port: int):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    max_bytes = max(b * s * h * d for (b, s, h, d) in _SHAPES) * 2 + (1 << 20)
    out_ptrs = create_shared_buffer(max_bytes, group)
    sig_ptrs = create_shared_buffer(custom_ops.meta_size(), group)
    CudaRTLibrary().cudaMemset(
        ctypes.c_void_p(sig_ptrs[rank]), 0, custom_ops.meta_size()
    )
    dist.barrier(group=group)
    fa = sgl_kernel.init_ulysses_a2a(out_ptrs, sig_ptrs, rank, world_size, True)

    if rank == 0:
        print(f"\nworld_size={world_size} dtype={_DTYPE}")
        print(
            f"{'shape (B,Sloc,H,D)':>28} | {'NCCL us':>10} | {'fused us':>10} | "
            f"{'tk us':>10} | {'fused x':>8} | {'tk x':>8}"
        )
        print("-" * 96)

    for B, S_local, H, D in _SHAPES:
        if H % world_size != 0:
            continue
        H_local = H // world_size
        S_global = S_local * world_size
        x = torch.randn(B, S_local, H, D, dtype=_DTYPE, device=device)
        out = torch.empty(B, S_global, H_local, D, dtype=_DTYPE, device=device)
        out_tk = torch.empty(B, S_global, H_local, D, dtype=_DTYPE, device=device)

        nccl_us = _bench_fn(lambda: ref_input_a2a(x, world_size, group))
        fused_us = _bench_fn(
            lambda: sgl_kernel.ulysses_a2a(fa, x, out, B, S_local, H, D, 0)
        )
        tk_us = _bench_fn(
            lambda: sgl_kernel.ulysses_a2a_tk(fa, x, out_tk, B, S_local, H, D, 0)
        )
        if rank == 0:
            print(
                f"{str((B, S_local, H, D)):>28} | {nccl_us:>10.2f} | "
                f"{fused_us:>10.2f} | {tk_us:>10.2f} | "
                f"{nccl_us / fused_us:>7.2f}x | {nccl_us / tk_us:>7.2f}x"
            )

    dist.barrier(group=group)
    sgl_kernel.dispose_ulysses_a2a(fa)
    dist.destroy_process_group(group=group)


def main():
    available = torch.cuda.device_count()
    if available < 2:
        print("Need at least 2 GPUs.")
        return
    world_size = int(os.environ.get("WORLD_SIZE", available))
    world_size = min(world_size, available, 8)
    for supported in (8, 6, 4, 2):
        if world_size >= supported:
            world_size = supported
            break
    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = [
        mp.Process(target=_worker, args=(world_size, i, port))
        for i in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
