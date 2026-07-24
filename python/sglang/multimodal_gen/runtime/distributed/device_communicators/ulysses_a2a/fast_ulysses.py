"""Adapter for triple-mu/fast-ulysses.

The upstream symmetric output buffers are reused by tag. A faster rank must not
start writing the next generation while a slower rank is still consuming the
previous one. SGLang therefore requires the small ``pre_write_barrier`` binding
shipped by ``scripts/fast_ulysses`` and executes it on the caller's CUDA stream
before every reuse.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import A2AMode, A2ASlot, UlyssesA2ACommitError, UlyssesA2AUnsupportedError

logger = logging.getLogger(__name__)


class FastUlyssesA2ABackend:
    name = "fast_ulysses"

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        transfer: str,
    ) -> None:
        if importlib.util.find_spec("fast_ulysses") is None:
            raise UlyssesA2AUnsupportedError(
                "fast_ulysses is not installed; forced backend cannot start"
            )
        from fast_ulysses import UlyssesGroup

        if not hasattr(UlyssesGroup, "pre_write_barrier"):
            raise UlyssesA2AUnsupportedError(
                "installed fast_ulysses lacks the required stream-ordered "
                "pre_write_barrier binding; run scripts/fast_ulysses/install.sh"
            )

        self.process_group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.transfer = transfer
        self._closed = False
        self._logged_signatures: set[tuple[object, ...]] = set()
        self._slot_calls: dict[A2ASlot, int] = defaultdict(int)
        try:
            self.tag_pool_size = int(
                os.environ.get("SGLANG_FAST_ULYSSES_TAG_POOL_SIZE", "1")
            )
        except ValueError as exc:
            raise UlyssesA2AUnsupportedError(
                "SGLANG_FAST_ULYSSES_TAG_POOL_SIZE must be an integer"
            ) from exc
        if not 1 <= self.tag_pool_size <= 64:
            raise UlyssesA2AUnsupportedError(
                "SGLANG_FAST_ULYSSES_TAG_POOL_SIZE must be in [1, 64]"
            )

        if self.world_size != dist.get_world_size():
            raise UlyssesA2AUnsupportedError(
                "fast_ulysses requires the Ulysses process group to span WORLD"
            )
        if not 2 <= self.world_size <= 8:
            raise UlyssesA2AUnsupportedError(
                f"fast_ulysses world size must be in [2, 8], got {self.world_size}"
            )
        if transfer == "tma":
            major, _ = torch.cuda.get_device_capability(self.device)
            if major < 9:
                raise UlyssesA2AUnsupportedError(
                    "fast_ulysses transfer=tma requires sm90 or newer"
                )

        gathered_pool_sizes: list[object] = [None] * self.world_size
        dist.all_gather_object(
            gathered_pool_sizes,
            self.tag_pool_size,
            group=group,
        )
        if any(size != self.tag_pool_size for size in gathered_pool_sizes):
            raise UlyssesA2AUnsupportedError(
                "fast_ulysses tag pool size differs across ranks: "
                f"{gathered_pool_sizes}"
            )

        try:
            self._group = UlyssesGroup(
                process_group=group,
                device=self.device,
            )
        except Exception as exc:
            raise UlyssesA2ACommitError(
                f"fast_ulysses collective initialization failed: {exc}"
            ) from exc

        import fast_ulysses

        module_path = Path(fast_ulysses.__file__).resolve()
        logger.info(
            "Initialized genuine fast_ulysses backend version=%s module=%s "
            "world_size=%d transfer=%s tag_pool_size=%d "
            "safety=stream_ordered_pre_write_barrier",
            getattr(fast_ulysses, "__version__", "unknown"),
            module_path,
            self.world_size,
            self.transfer,
            self.tag_pool_size,
        )

    @staticmethod
    def supports_semantics(
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        mode: int,
        world_size: int,
        *,
        head_dim: int,
    ) -> tuple[bool, str]:
        if importlib.util.find_spec("fast_ulysses") is None:
            return False, "fast_ulysses_not_installed"
        if len(shape) != 4:
            return False, "rank_not_four"
        if device.type != "cuda":
            return False, "device"
        if dtype not in (torch.bfloat16, torch.float16):
            return False, "dtype"
        if head_dim != 2:
            return False, "head_dim"
        if not 2 <= world_size <= 8:
            return False, "world_size"
        if shape[3] * dtype.itemsize % 16 != 0:
            return False, "row_alignment"
        split_dim = 2 if mode == int(A2AMode.INPUT) else 1
        if shape[split_dim] % world_size != 0:
            return False, "non_uniform_split"
        return True, ""

    def all_to_all(
        self,
        x: torch.Tensor,
        mode: int,
        *,
        slot: A2ASlot,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("fast_ulysses backend is closed")

        call_index = self._slot_calls[slot]
        tag_index = call_index % self.tag_pool_size
        # A tag is not reused during the first pool cycle. At each wrap, one
        # stream-ordered barrier retires every tag from the previous cycle:
        # every rank has reached this point after consuming all earlier outputs.
        if call_index >= self.tag_pool_size and tag_index == 0:
            self._group.pre_write_barrier(x)
        self._slot_calls[slot] = call_index + 1
        tag = (
            f"sglang:{slot.value}"
            if self.tag_pool_size == 1
            else f"sglang:{slot.value}:{tag_index}"
        )
        try:
            if self.transfer == "ce":
                out = self._group.all_to_all_single_4d_ce(x, mode=mode, tag=tag)
            else:
                use_tma = {
                    "auto": None,
                    "sm": False,
                    "tma": True,
                }[self.transfer]
                out = self._group.all_to_all_single_4d(
                    x,
                    mode=mode,
                    tag=tag,
                    use_tma=use_tma,
                )
        except Exception as exc:
            raise UlyssesA2ACommitError(
                f"fast_ulysses failed after collective commit: {exc}"
            ) from exc

        signature = (
            mode,
            tuple(x.shape),
            str(x.dtype),
            slot.value,
            self.transfer,
        )
        if signature not in self._logged_signatures:
            logger.info(
                "genuine fast_ulysses collective active mode=%d tag=%s "
                "shape=%s dtype=%s transfer=%s",
                mode,
                tag,
                tuple(x.shape),
                x.dtype,
                self.transfer,
            )
            self._logged_signatures.add(signature)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._group.destroy()
        self._closed = True
