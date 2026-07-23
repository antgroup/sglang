"""NVLink P2P fast path for Ulysses all-to-all.

This manager owns the IPC-shared output staging buffer and signal buffer for
one Ulysses process group. Unsupported cases fall back to the existing
all_to_all implementation.
"""

import ctypes
import logging
import os
import threading
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.base import (
    UlyssesA2ACommitError,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_full_nvlink,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_BUFFER_GROW_PADDING = 1 << 20
_SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _ops_available() -> bool:
    if not _is_cuda:
        return False
    try:
        import sgl_kernel  # noqa: F401  # pyright: ignore[reportMissingImports]

        return hasattr(torch.ops.sgl_kernel, "ulysses_a2a")
    except Exception:
        return False


class UlyssesP2PAllToAll:
    """One coordinator-owned IPC/NVLink Ulysses communicator.

    The kernel has one signal/output staging area, so calls are stream-ordered
    with ``max_inflight=1``. Capability misses happen before shared allocation;
    failures after allocation starts are fatal and must not fall back to NCCL.
    """

    max_inflight = 1

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        tk_style: bool = True,
    ) -> None:
        self.enabled = False
        self.disabled_reason = "not_initialized"
        self.closed = False
        self.group = group
        self.device = device
        self.tk_style = tk_style
        self.fa: Optional[int] = None
        self.out_ptrs: Optional[List[int]] = None
        self.signal_ptrs: Optional[List[int]] = None
        self.max_bytes = 0
        self._launch_lock = threading.Lock()
        self._done_events: list[torch.cuda.Event] = []
        self._last_done_event: torch.cuda.Event | None = None
        self._next_event_index = 0

        if not _ops_available():
            self.disabled_reason = "sgl_kernel_ulysses_a2a_op_unavailable"
            return

        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size == 1:
            self.disabled_reason = "world_size_one"
            return
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            self.disabled_reason = f"unsupported_world_size:{self.world_size}"
            logger.debug(
                "UlyssesP2PAllToAll disabled: unsupported world size %d.",
                self.world_size,
            )
            return

        full_nvlink = self._check_intra_node_p2p()
        if not full_nvlink:
            self.disabled_reason = "topology_not_single_node_full_nvlink"
            logger.debug("UlyssesP2PAllToAll disabled: not intra-node NVLink/P2P.")
            return

        self.full_nvlink = full_nvlink
        try:
            import sgl_kernel  # pyright: ignore[reportMissingImports]

            from sglang.srt.distributed.device_communicators.cuda_wrapper import (
                CudaRTLibrary,
            )
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            signal_bytes = sgl_kernel.meta_size()
            self.signal_ptrs = CustomAllreduce.create_shared_buffer(
                signal_bytes, group=group
            )
            CudaRTLibrary().cudaMemset(
                ctypes.c_void_p(self.signal_ptrs[self.rank]), 0, signal_bytes
            )
            dist.barrier(group=group)
        except Exception as e:
            self._teardown_local()
            raise UlyssesA2ACommitError(
                f"sgl_p2p signal allocation failed after backend commit: {e!r}"
            ) from e

        self._done_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.enabled = True
        self.disabled_reason = ""

    @property
    def disabled(self) -> bool:
        return not self.enabled

    @staticmethod
    def supports_semantics(
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mode: int,
        world_size: int,
    ) -> tuple[bool, str]:
        if len(shape) != 4:
            return False, "rank_not_four"
        if device.type != "cuda":
            return False, "device_not_cuda"
        if dtype not in _SUPPORTED_DTYPES:
            return False, f"unsupported_dtype:{dtype}"
        if world_size not in _SUPPORTED_WORLD_SIZES:
            return False, f"unsupported_world_size:{world_size}"
        if mode == 0:
            if shape[2] % world_size != 0:
                return False, "heads_not_divisible"
        elif mode == 1:
            if shape[1] % world_size != 0:
                return False, "sequence_not_divisible"
        else:
            return False, f"unsupported_mode:{mode}"
        return True, ""

    @staticmethod
    def runtime_available() -> tuple[bool, str]:
        if not _ops_available():
            return False, "sgl_kernel_ulysses_a2a_op_unavailable"
        return True, ""

    @classmethod
    def supports_tensor(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mode: int,
        world_size: int,
    ) -> tuple[bool, str]:
        supported, reason = cls.supports_semantics(
            shape,
            dtype,
            device,
            mode,
            world_size,
        )
        if not supported:
            return supported, reason
        return cls.runtime_available()

    @staticmethod
    def _node_identity() -> str:
        try:
            with open("/proc/sys/kernel/random/boot_id", "r") as f:
                return f.read().strip()
        except Exception:
            import socket

            return socket.gethostname()

    @staticmethod
    def _physical_device_id(local_device: int) -> int:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            try:
                return list(map(int, cuda_visible_devices.split(",")))[local_device]
            except Exception:
                return local_device
        return local_device

    def _check_intra_node_p2p(self) -> bool:
        try:
            world_size = dist.get_world_size(group=self.group)
            local_rank = dist.get_rank(group=self.group)
            local_device = self.device.index
            if local_device is None:
                local_device = torch.cuda.current_device()

            node_id = self._node_identity()
            node_ids: List[object] = [None] * world_size
            dist.all_gather_object(node_ids, node_id, group=self.group)

            phys_t = torch.tensor(
                [self._physical_device_id(local_device)],
                dtype=torch.int,
                device=self.device,
            )
            gathered_phys = [torch.empty_like(phys_t) for _ in range(world_size)]
            dist.all_gather(gathered_phys, phys_t, group=self.group)

            loc_t = torch.tensor([local_device], dtype=torch.int, device=self.device)
            gathered_loc = [torch.empty_like(loc_t) for _ in range(world_size)]
            dist.all_gather(gathered_loc, loc_t, group=self.group)
        except Exception:
            return False

        ok = True
        try:
            if any(nid != node_id for nid in node_ids):
                ok = False
            if ok:
                physical_device_ids = [int(t.item()) for t in gathered_phys]
                if not is_full_nvlink(physical_device_ids, world_size):
                    ok = False
            if ok:
                local_devices = [int(t.item()) for t in gathered_loc]
                for i, peer_device in enumerate(local_devices):
                    if i == local_rank:
                        continue
                    if not torch.cuda.can_device_access_peer(local_device, peer_device):
                        ok = False
                        break
        except Exception:
            ok = False

        try:
            ok_t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=self.device)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN, group=self.group)
            return bool(ok_t.item())
        except Exception:
            return False

    def _ensure_capacity(self, nbytes: int) -> bool:
        if not self.enabled or self.closed:
            return False
        if self.fa is not None and nbytes <= self.max_bytes:
            return True

        self._drain_inflight()
        requested = torch.tensor([nbytes], dtype=torch.int64, device=self.device)
        maximum = requested.clone()
        minimum = requested.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=self.group)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=self.group)
        if int(maximum.item()) != int(minimum.item()):
            raise UlyssesA2ACommitError(
                "sgl_p2p capacity request diverged across ranks: "
                f"min={int(minimum.item())}, max={int(maximum.item())}"
            )

        new_bytes = max(nbytes, self.max_bytes) + _BUFFER_GROW_PADDING
        try:
            import sgl_kernel  # pyright: ignore[reportMissingImports]

            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            if self.fa is not None:
                sgl_kernel.dispose_ulysses_a2a(self.fa)
                self.fa = None
            if self.out_ptrs is not None:
                CustomAllreduce.free_shared_buffer(self.out_ptrs, group=self.group)
                self.out_ptrs = None

            self.out_ptrs = CustomAllreduce.create_shared_buffer(
                new_bytes, group=self.group
            )
            self.fa = sgl_kernel.init_ulysses_a2a(
                self.out_ptrs,
                self.signal_ptrs,
                self.rank,
                self.world_size,
                self.full_nvlink,
            )
            self.max_bytes = new_bytes
            return True
        except Exception as e:
            self._teardown_local()
            raise UlyssesA2ACommitError(
                f"sgl_p2p output allocation failed after backend commit: {e!r}"
            ) from e

    def _drain_inflight(self) -> None:
        if self._last_done_event is not None:
            self._last_done_event.synchronize()
            self._last_done_event = None

    def _teardown_local(self) -> None:
        """Best-effort cleanup for partially initialized/fatal paths."""
        self.enabled = False
        try:
            import sgl_kernel  # pyright: ignore[reportMissingImports]

            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )

            if self.fa is not None:
                sgl_kernel.dispose_ulysses_a2a(self.fa)
            if self.out_ptrs is not None:
                CustomAllreduce.free_shared_buffer(self.out_ptrs, group=self.group)
            if self.signal_ptrs is not None:
                CustomAllreduce.free_shared_buffer(self.signal_ptrs, group=self.group)
        except Exception:
            pass
        self.fa = None
        self.out_ptrs = None
        self.signal_ptrs = None
        self.max_bytes = 0

    def _prepare_call(
        self, x: torch.Tensor, mode: int
    ) -> Optional[tuple[torch.Tensor, tuple[int, int, int, int], int, int, int, int]]:
        if not self.enabled or self.closed:
            return None
        if x.dim() != 4 or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None

        x = x.contiguous()
        B, dim1, dim2, D = (int(v) for v in x.shape)
        W = self.world_size

        if mode == 0:
            S_local, H = dim1, dim2
            if H % W != 0:
                return None
            H_local = H // W
            out_shape = (B, S_local * W, H_local, D)
        elif mode == 1:
            S_global, H_local = dim1, dim2
            if S_global % W != 0:
                return None
            S_local = S_global // W
            H = H_local * W
            out_shape = (B, S_local, H, D)
        else:
            return None

        nbytes = x.numel() * x.element_size()
        if not self._ensure_capacity(nbytes):
            return None
        return x, out_shape, B, S_local, H, D

    def all_to_all(self, x: torch.Tensor, mode: int) -> Optional[torch.Tensor]:
        with self._launch_lock:
            prepared = self._prepare_call(x, mode)
            if prepared is None:
                return None
            x, out_shape, B, S_local, H, D = prepared

            import sgl_kernel  # pyright: ignore[reportMissingImports]

            current_stream = torch.cuda.current_stream(device=x.device)
            if self._last_done_event is not None:
                current_stream.wait_event(self._last_done_event)

            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            if self.tk_style:
                sgl_kernel.ulysses_a2a_tk(self.fa, x, out, B, S_local, H, D, mode)
            else:
                sgl_kernel.ulysses_a2a(self.fa, x, out, B, S_local, H, D, mode)
            done_event = self._done_events[self._next_event_index]
            self._next_event_index = (self._next_event_index + 1) % len(
                self._done_events
            )
            done_event.record(current_stream)
            self._last_done_event = done_event
            return out

    def close(self) -> None:
        """Collectively drain and release backend-owned resources."""
        if self.closed:
            return
        self.closed = True
        if not self.enabled:
            self._teardown_local()
            return

        self._drain_inflight()
        try:
            dist.barrier(group=self.group)
            self._teardown_local()
            dist.barrier(group=self.group)
        finally:
            self.enabled = False
            self._done_events.clear()
