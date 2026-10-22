# SPDX-License-Identifier: Apache-2.0
"""Bounded, single-stream pipelines for same-timestep noisy history.

The schedule is independent of the model and transport. A rank owns a contiguous
piece of the expanded (denoising step, transformer layer) chain. It processes
chunks in order, satisfying both activation and same-stage history dependencies.
Timestep partitioning keeps entire forwards together; block partitioning can
split a forward across ranks. Neither changes a clean-history model into an
equivalent noisy-history model.
"""

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Callable, Literal, Protocol


@dataclass(frozen=True)
class CausalPipelinePlacement:
    """Physical ranks for denoising and the stateful, ordered decoder."""

    denoising_ranks: int
    vae_placement: Literal["shared", "dedicated"] = "shared"
    vae_rank: int | None = None

    def __post_init__(self):
        if self.denoising_ranks < 1:
            raise ValueError("Causal pipeline parallel size must be positive")
        if self.vae_placement not in ("shared", "dedicated"):
            raise ValueError("Causal VAE placement must be shared or dedicated")
        if self.vae_placement == "shared" and not (
            0 <= self.decoder_rank < self.denoising_ranks
        ):
            raise ValueError("Shared VAE rank must be a denoising rank")
        if (
            self.vae_placement == "dedicated"
            and self.decoder_rank != self.denoising_ranks
        ):
            raise ValueError("Dedicated VAE rank must follow all denoising ranks")

    @property
    def decoder_rank(self) -> int:
        if self.vae_rank is not None:
            return self.vae_rank
        if self.vae_placement == "dedicated":
            return self.denoising_ranks
        return self.denoising_ranks - 1

    @property
    def world_size(self) -> int:
        extra_rank = 1 if self.vae_placement == "dedicated" else 0
        return self.denoising_ranks + extra_rank


def causal_pipeline_seed(seed: int, chunk: int, timestep: int) -> int:
    """Noise is independent of rank count, layer partitioning and window size."""
    key = f"{seed}:{chunk}:{timestep}".encode("ascii")
    return int.from_bytes(blake2b(key, digest_size=8).digest(), "little") % (2**63)


@dataclass(frozen=True)
class CausalPipelineSlice:
    timestep: int
    block_start: int
    block_end: int


@dataclass(frozen=True)
class CausalPipelinePlan:
    num_steps: int
    num_blocks: int
    degree: int
    mode: Literal["timestep", "block"]

    def __post_init__(self) -> None:
        if self.mode not in ("timestep", "block"):
            raise ValueError("Causal pipeline mode must be timestep or block")
        if min(self.num_steps, self.num_blocks, self.degree) < 1:
            raise ValueError("Causal pipeline dimensions must be positive")
        units = self.num_steps if self.mode == "timestep" else self.num_stages
        if self.degree > units:
            raise ValueError(
                f"{self.mode} partitioning has {units} units for {self.degree} ranks"
            )

    @property
    def num_stages(self) -> int:
        return self.num_steps * self.num_blocks

    def slices(self, rank: int) -> tuple[CausalPipelineSlice, ...]:
        if not 0 <= rank < self.degree:
            raise ValueError(f"Invalid causal pipeline rank {rank}")
        units = self.num_steps if self.mode == "timestep" else self.num_stages
        start = units * rank // self.degree
        end = units * (rank + 1) // self.degree
        if self.mode == "timestep":
            start *= self.num_blocks
            end *= self.num_blocks
        result = []
        while start < end:
            step, block_start = divmod(start, self.num_blocks)
            stop = min(end, (step + 1) * self.num_blocks)
            result.append(
                CausalPipelineSlice(step, block_start, stop - step * self.num_blocks)
            )
            start = stop
        return tuple(result)


class CausalPipelineTransport(Protocol):
    rank: int

    def receive(self) -> dict[str, Any]: ...

    def send(self, payload: dict[str, Any]) -> None: ...

    def finish(self, error: str | None) -> None: ...


def run_causal_pipeline(
    plan: CausalPipelinePlan,
    *,
    chunk_indices: list[int],
    transport: CausalPipelineTransport,
    initialize: Callable[[int], dict[str, Any]],
    execute: Callable[[int, CausalPipelineSlice, dict[str, Any]], dict[str, Any]],
    consume: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one bounded window; only the final rank returns payloads.

    Model errors travel downstream while every rank drains the agreed window.
    ``finish`` propagates failure back to all ranks before state may be reused.
    A transport failure itself is fatal and relies on the process-group timeout.
    """
    if not chunk_indices or any(
        b != a + 1 for a, b in zip(chunk_indices, chunk_indices[1:])
    ):
        raise ValueError("A causal pipeline window needs consecutive chunk indices")
    if chunk_indices[0] < 0:
        raise ValueError("Chunk indices must be nonnegative")
    slices = plan.slices(transport.rank)
    error = None
    outputs = []
    for index in chunk_indices:
        payload = {} if transport.rank == 0 else transport.receive()
        try:
            if transport.rank and payload.get("chunk_index") != index:
                raise ValueError(f"Expected chunk {index}, received a different chunk")
            error = error or payload.get("error")
            if error is None:
                if transport.rank == 0:
                    payload = initialize(index)
                for partition in slices:
                    payload = execute(index, partition, payload)
        except Exception as exc:
            error = f"Causal pipeline rank {transport.rank}, chunk {index}: {exc}"
        if error is not None:
            payload = {"error": error}
        payload["chunk_index"] = index
        if transport.rank + 1 < plan.degree:
            transport.send(payload)
        elif consume is not None:
            # The decoder consumes both success and error envelopes, so a failed
            # chunk cannot strand it waiting for the rest of the window.
            consume(payload)
        else:
            outputs.append(payload)
    transport.finish(error)
    return outputs


class TorchCausalPipelineTransport:
    """Neighbor transfers on SGLang's existing pipeline process group."""

    def __init__(self, group):
        self.group = group
        self.rank = group.rank_in_group

    def receive(self):
        return self.group.recv_tensor_dict(src=self.rank - 1)

    def send(self, payload):
        self.group.send_tensor_dict(payload, dst=self.rank + 1)

    def finish(self, error):
        import torch.distributed as dist

        errors = [None] * self.group.world_size
        dist.all_gather_object(errors, error, group=self.group.cpu_group)
        first_error = next((item for item in errors if item is not None), None)
        if first_error is not None:
            raise RuntimeError(first_error)
