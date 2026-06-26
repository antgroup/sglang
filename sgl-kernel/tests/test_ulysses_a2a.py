"""Correctness tests for fused NVLink-P2P Ulysses all-to-all."""

import ctypes
import multiprocessing as mp
import socket
import unittest
from typing import Any, List, Optional

import sgl_kernel  # noqa: F401
import sgl_kernel.allreduce as custom_ops
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    input_tensor = torch.ByteTensor(list(handle_bytes)).to(f"cuda:{rank}")
    gathered_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, input_tensor, group=group)

    handles = []
    handle_type = type(handle)
    for tensor in gathered_tensors:
        bytes_data = bytes(tensor.cpu().tolist())
        handle_obj = handle_type()
        ctypes.memmove(ctypes.addressof(handle_obj), bytes_data, len(bytes_data))
        handles.append(handle_obj)

    pointers: List[int] = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)
        else:
            pointers.append(lib.cudaIpcOpenMemHandle(h).value)

    dist.barrier(group=group)
    return pointers


def free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group=group)
    lib = CudaRTLibrary()
    if pointers and len(pointers) > rank and pointers[rank] is not None:
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))
    dist.barrier(group=group)


def ref_input_a2a(x: torch.Tensor, world_size: int, group) -> torch.Tensor:
    b, s_local, h_global, d = x.shape
    h_local, s_global = h_global // world_size, s_local * world_size
    xx = x.permute(2, 0, 1, 3).contiguous().flatten().contiguous()
    out = torch.empty_like(xx)
    dist.all_to_all_single(out, xx, group=group)
    out = out.reshape(world_size, h_local, b, s_local, d)
    return out.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_global, h_local, d)


def ref_output_a2a(x: torch.Tensor, world_size: int, group) -> torch.Tensor:
    b, s_global, h_local, d = x.shape
    s_local, h_global = s_global // world_size, h_local * world_size
    xx = x.permute(1, 0, 2, 3).contiguous().flatten().contiguous()
    out = torch.empty_like(xx)
    dist.all_to_all_single(out, xx, group=group)
    out = out.reshape(world_size, s_local, b, h_local, d)
    return out.permute(2, 1, 0, 3, 4).contiguous().reshape(b, s_local, h_global, d)


_SHAPES = [
    (1, 8, 8, 64),
    (2, 16, 8, 128),
    (1, 32, 16, 128),
]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _run_correctness_worker(world_size, rank, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    fa = None
    out_ptrs = None
    sig_ptrs = None
    try:
        max_bytes = max(b * s * h * d for (b, s, h, d) in _SHAPES) * 4 + (1 << 20)
        out_ptrs = create_shared_buffer(max_bytes, group=group)
        sig_ptrs = create_shared_buffer(custom_ops.meta_size(), group=group)
        CudaRTLibrary().cudaMemset(
            ctypes.c_void_p(sig_ptrs[rank]), 0, custom_ops.meta_size()
        )
        dist.barrier(group=group)

        fa = sgl_kernel.init_ulysses_a2a(out_ptrs, sig_ptrs, rank, world_size, True)

        torch.manual_seed(1234 + rank)
        for dtype in _DTYPES:
            for B, S_local, H, D in _SHAPES:
                if H % world_size != 0:
                    continue
                H_local = H // world_size
                S_global = S_local * world_size

                x = torch.randn(B, S_local, H, D, dtype=dtype, device=device)
                ref = ref_input_a2a(x, world_size, group)
                out = torch.empty(B, S_global, H_local, D, dtype=dtype, device=device)
                sgl_kernel.ulysses_a2a(fa, x, out, B, S_local, H, D, 0)
                torch.cuda.synchronize()
                assert torch.equal(out, ref)

                out_tk = torch.empty(
                    B, S_global, H_local, D, dtype=dtype, device=device
                )
                sgl_kernel.ulysses_a2a_tk(fa, x, out_tk, B, S_local, H, D, 0)
                torch.cuda.synchronize()
                assert torch.equal(out_tk, ref)

                back = torch.empty(B, S_local, H, D, dtype=dtype, device=device)
                sgl_kernel.ulysses_a2a(fa, out, back, B, S_local, H, D, 1)
                torch.cuda.synchronize()
                assert torch.equal(back, x)

                back_tk = torch.empty(B, S_local, H, D, dtype=dtype, device=device)
                sgl_kernel.ulysses_a2a_tk(fa, out_tk, back_tk, B, S_local, H, D, 1)
                torch.cuda.synchronize()
                assert torch.equal(back_tk, x)

                ref_out = ref_output_a2a(out, world_size, group)
                assert torch.equal(back, ref_out)
    finally:
        dist.barrier(group=group)
        if fa is not None:
            sgl_kernel.dispose_ulysses_a2a(fa)
        if out_ptrs:
            free_shared_buffer(out_ptrs, group)
        if sig_ptrs:
            free_shared_buffer(sig_ptrs, group)
        dist.destroy_process_group(group=group)


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = [
        mp.Process(
            target=test_target,
            args=(world_size, i, port) + target_args,
            name=f"Worker-{i}",
        )
        for i in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {i} failed (exit {proc.exitcode})"


class TestUlyssesA2A(unittest.TestCase):
    world_sizes = [2, 4, 8]

    def test_correctness(self):
        available = torch.cuda.device_count()
        ran_any = False
        for world_size in self.world_sizes:
            if world_size > available:
                continue
            ran_any = True
            multi_process_parallel(world_size, _run_correctness_worker)
        if not ran_any:
            self.skipTest("requires at least 2 GPUs")


if __name__ == "__main__":
    unittest.main()
