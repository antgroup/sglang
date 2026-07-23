"""NCCL implementation of semantic Ulysses all-to-all."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch.distributed import ProcessGroup


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


class NCCLUlyssesA2ABackend:
    name = "nccl"

    def __init__(self, group: ProcessGroup) -> None:
        self.group = group
        self.world_size = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)

    def _all_to_all_single(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = ft_c.all_to_all_single(
            x.flatten(),
            output_split_sizes=None,
            input_split_sizes=None,
            group=self.group,
        )
        return _maybe_wait(x).reshape(x_shape)

    def _all_to_all_single_with_sizes(
        self,
        x: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
    ) -> torch.Tensor:
        x = x.flatten().contiguous()
        output = torch.empty(sum(output_split_sizes), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            output,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.group,
        )
        return output

    def input_all_to_all(
        self,
        x: torch.Tensor,
        *,
        head_dim: int,
        seq_lens: list[int] | None,
    ) -> torch.Tensor:
        if self.world_size <= 1:
            return x
        if seq_lens is not None:
            return self._input_variable(x, seq_lens=seq_lens, head_dim=head_dim)

        assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
        assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
        if head_dim == 1:
            b, h_global, s_local, d = x.shape
            permute_order = (1, 0, 2, 3)
        else:
            b, s_local, h_global, d = x.shape
            permute_order = (2, 0, 1, 3)

        assert (
            h_global % self.world_size == 0
        ), f"h_global ({h_global}) must be divisible by world_size ({self.world_size})"
        h_local = h_global // self.world_size
        s_global = s_local * self.world_size
        x = self._all_to_all_single(x.permute(permute_order).contiguous())
        x = x.reshape(self.world_size, h_local, b, s_local, d)
        if head_dim == 1:
            return (
                x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, h_local, s_global, d)
            )
        return x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_global, h_local, d)

    def output_all_to_all(
        self,
        x: torch.Tensor,
        *,
        head_dim: int,
        seq_lens: list[int] | None,
    ) -> torch.Tensor:
        if self.world_size <= 1:
            return x
        if seq_lens is not None:
            return self._output_variable(x, seq_lens=seq_lens, head_dim=head_dim)

        assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
        assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
        if head_dim == 1:
            b, h_local, s_global, d = x.shape
            permute_order = (2, 0, 1, 3)
        else:
            b, s_global, h_local, d = x.shape
            permute_order = (1, 0, 2, 3)

        assert (
            s_global % self.world_size == 0
        ), f"s_global ({s_global}) must be divisible by world_size ({self.world_size})"
        s_local = s_global // self.world_size
        h_global = h_local * self.world_size
        x = self._all_to_all_single(x.permute(permute_order).contiguous())
        x = x.reshape(self.world_size, s_local, b, h_local, d)
        if head_dim == 1:
            return (
                x.permute(2, 0, 3, 1, 4).contiguous().reshape(b, h_global, s_local, d)
            )
        return x.permute(2, 1, 0, 3, 4).contiguous().reshape(b, s_local, h_global, d)

    def _input_variable(
        self, x: torch.Tensor, *, seq_lens: list[int], head_dim: int
    ) -> torch.Tensor:
        assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
        assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
        assert (
            len(seq_lens) == self.world_size
        ), f"seq_lens must have length {self.world_size}, got {len(seq_lens)}"
        if head_dim == 1:
            b, h_global, s_local, d = x.shape
            permute_order = (1, 0, 2, 3)
        else:
            b, s_local, h_global, d = x.shape
            permute_order = (2, 0, 1, 3)
        assert s_local == seq_lens[self.rank], (
            f"s_local ({s_local}) must equal "
            f"seq_lens[{self.rank}] ({seq_lens[self.rank]})"
        )
        assert h_global % self.world_size == 0
        h_local = h_global // self.world_size
        x = x.permute(permute_order).contiguous()
        x = x.reshape(self.world_size, h_local, b, s_local, d)
        input_split_sizes = [h_local * b * s_local * d] * self.world_size
        output_split_sizes = [h_local * b * seq_len * d for seq_len in seq_lens]
        x = self._all_to_all_single_with_sizes(x, output_split_sizes, input_split_sizes)
        chunks = []
        offset = 0
        for seq_len, split_size in zip(seq_lens, output_split_sizes):
            chunks.append(
                x[offset : offset + split_size].reshape(h_local, b, seq_len, d)
            )
            offset += split_size
        x = torch.cat(chunks, dim=2)
        if head_dim == 1:
            return x.permute(1, 0, 2, 3).contiguous()
        return x.permute(1, 2, 0, 3).contiguous()

    def _output_variable(
        self, x: torch.Tensor, *, seq_lens: list[int], head_dim: int
    ) -> torch.Tensor:
        assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
        assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
        assert (
            len(seq_lens) == self.world_size
        ), f"seq_lens must have length {self.world_size}, got {len(seq_lens)}"
        if head_dim == 1:
            b, h_local, s_global, d = x.shape
            permute_order = (1, 0, 2, 3)
        else:
            b, s_global, h_local, d = x.shape
            permute_order = (2, 0, 1, 3)
        assert s_global == sum(
            seq_lens
        ), f"s_global ({s_global}) must equal sum(seq_lens) ({sum(seq_lens)})"
        s_local = seq_lens[self.rank]
        x = x.permute(permute_order).contiguous()
        input_chunks = []
        start = 0
        for seq_len in seq_lens:
            end = start + seq_len
            input_chunks.append(x[:, :, start:end, :].contiguous().reshape(-1))
            start = end
        x = torch.cat(input_chunks, dim=0)
        input_split_sizes = [h_local * b * seq_len * d for seq_len in seq_lens]
        output_split_sizes = [h_local * b * s_local * d] * self.world_size
        x = self._all_to_all_single_with_sizes(x, output_split_sizes, input_split_sizes)
        chunks = []
        offset = 0
        for split_size in output_split_sizes:
            chunks.append(
                x[offset : offset + split_size].reshape(h_local, b, s_local, d)
            )
            offset += split_size
        x = torch.cat(chunks, dim=0)
        if head_dim == 1:
            return x.permute(1, 0, 2, 3).contiguous()
        return x.permute(1, 2, 0, 3).contiguous()

    def close(self) -> None:
        return
