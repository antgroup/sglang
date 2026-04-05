"""CPU host radix cache for HiSparse prefix sharing.

Implements BasePrefixCache so it can serve as the scheduler's tree_cache.
All KV data lives on the CPU host pool; match_prefix returns host_hit_length
so the scheduler can trigger init_load_back to copy prefix KV from CPU to GPU
before prefill, reducing redundant GPU computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class HiSparseRadixCache(BasePrefixCache):
    """Radix tree managing CPU host KV indices for HiSparse prefix sharing.

    The internal host-side RadixCache stores host pool indices in TreeNode.value.
    match_prefix returns empty device_indices (nothing cached on GPU) but sets
    host_hit_length so the scheduler's add_one_req path can call init_load_back
    to copy the prefix KV from CPU to GPU before the forward pass.
    """

    def __init__(self, params: CacheInitParams):
        # Device-side pools from CacheInitParams (used by cache_finished_req,
        # cache_unfinished_req, init_load_back, and available_and_evictable_str)
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # Host pool and internal radix tree are set up lazily via set_host_pool()
        # because the hisparse coordinator (which owns the host pool) is created
        # after the scheduler's tree_cache.
        self.host_pool: Optional[HostKVCache] = None
        self._host_cache: Optional[RadixCache] = None

    # -- Lazy initialisation of the host-side radix tree --

    def set_host_pool(self, host_pool: HostKVCache) -> None:
        """Attach the host memory pool and create the internal radix tree.

        Must be called once after the HiSparseCoordinator is available.
        """
        self.host_pool = host_pool
        host_params = CacheInitParams(
            disable=False,
            # Only tree primitives (match/insert/evict/lock) are used on the
            # host cache; the high-level cache_finished_req / cache_unfinished_req
            # are handled by HiSparseRadixCache itself.
            req_to_token_pool=None,
            token_to_kv_pool_allocator=host_pool,
            page_size=host_pool.page_size,
        )
        self._host_cache = RadixCache(host_params)

    @property
    def _cache(self) -> RadixCache:
        assert self._host_cache is not None, (
            "HiSparseRadixCache: host pool not set yet — call set_host_pool() first"
        )
        return self._host_cache

    # -- BasePrefixCache required properties --

    @property
    def disable(self) -> bool:
        return False

    # -- BasePrefixCache abstract method implementations --

    def reset(self):
        pass

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        empty = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self._host_cache is None or len(params.key) == 0:
            return MatchResult(
                device_indices=empty,
                last_device_node=None,
                last_host_node=None,
                host_hit_length=0,
            )

        key = RadixKey(
            token_ids=params.key.token_ids, extra_key=params.key.extra_key
        )
        result = self._cache.match_prefix(MatchPrefixParams(key=key))
        host_hit_len = len(result.device_indices)

        # device_indices is empty because the KV is only on the CPU host pool.
        # host_hit_length tells the scheduler how many tokens can be loaded back.
        return MatchResult(
            device_indices=empty,
            last_device_node=None,
            last_host_node=result.last_device_node,
            host_hit_length=host_hit_len,
        )

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Any]:
        """Load prefix KV from CPU host pool to GPU device pool.

        Allocates device KV pool slots and performs a synchronous per-layer
        CPU->GPU DMA transfer.  Returns (device_indices, last_node).
        """
        last_node = params.last_host_node
        host_hit_length = params.host_hit_length

        if last_node is None or host_hit_length <= 0 or self.host_pool is None:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        # Collect host indices by walking from root to last_node
        host_indices = self._collect_host_indices(last_node, host_hit_length)
        if host_indices is None or host_indices.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        num_tokens = host_indices.numel()

        # Allocate device pool slots
        device_indices = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if device_indices is None:
            logger.warning(
                "HiSparseRadixCache: init_load_back failed to allocate %d device slots",
                num_tokens,
            )
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )

        # Synchronous per-layer CPU->GPU transfer
        device_pool = self.token_to_kv_pool_allocator.get_kvcache()
        host_indices_dev = host_indices.to(device=self.device)
        for layer_id in range(self.host_pool.start_layer, self.host_pool.end_layer):
            self.host_pool.load_to_device_per_layer(
                device_pool,
                host_indices_dev,
                device_indices,
                layer_id,
                io_backend="kernel",
            )

        # Locking is managed by the coordinator (via _req_radix_node tracking),
        # not here, to avoid interference with _lock_node in schedule_policy.
        return device_indices, last_node

    def _collect_host_indices(
        self, node: TreeNode, expected_len: int
    ) -> Optional[torch.Tensor]:
        """Walk from root to *node* and concatenate the host pool indices."""
        segments = []
        cur = node
        while cur is not None and cur.key is not None:
            if cur.value is not None:
                segments.append(cur.value)
            cur = cur.parent
        if not segments:
            return None
        segments.reverse()
        indices = torch.cat(segments)
        return indices[:expected_len]

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Release device KV pool slots.  Host indices are managed by the
        coordinator's request_finished which inserts into the host tree."""
        kv_committed_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Set prefix_indices for chunked prefill scheduling."""
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, params: EvictParams) -> EvictResult:
        if self._host_cache is None:
            return EvictResult()
        result = self._cache.evict(params)
        return result

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if self._host_cache is None or node is None:
            return IncLockRefResult(delta=0)
        return self._cache.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self._host_cache is None or node is None:
            return DecLockRefResult(delta=0)
        return self._cache.dec_lock_ref(node, params)

    def evictable_size(self) -> int:
        if self._host_cache is None:
            return 0
        return self._cache.evictable_size()

    def protected_size(self):
        if self._host_cache is None:
            return 0
        return self._cache.protected_size()

    def pretty_print(self):
        if self._host_cache is None:
            return "<HiSparseRadixCache: host pool not initialised>"
        return self._cache.pretty_print()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return (
            f"Available tokens: {available_size + evictable_size} "
            f"({available_size=} + {evictable_size=})\n"
        )

    # -- Convenience methods for the coordinator --

    def host_match_prefix(
        self,
        token_ids: list,
        extra_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Any, int]:
        """Direct host-tree match returning (host_indices, last_node, matched_len)."""
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        result = self._cache.match_prefix(MatchPrefixParams(key=key))
        return result.device_indices, result.last_device_node, len(result.device_indices)

    def host_insert(
        self,
        token_ids: list,
        host_indices: torch.Tensor,
        extra_key: Optional[str] = None,
    ) -> int:
        """Insert host pool indices into the host tree.

        Returns the prefix_len (tokens already present).
        """
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        result = self._cache.insert(InsertParams(key=key, value=host_indices))
        return result.prefix_len

    def host_evict(self, num_tokens: int):
        return self._cache.evict(EvictParams(num_tokens=num_tokens))

    def host_evictable_size(self) -> int:
        return self._cache.evictable_size()

    def host_inc_lock_ref(self, node: TreeNode):
        return self._cache.inc_lock_ref(node)

    def host_dec_lock_ref(self, node: TreeNode):
        return self._cache.dec_lock_ref(node)

    @property
    def root_node(self) -> TreeNode:
        return self._cache.root_node
