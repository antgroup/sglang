"""CPU host radix cache for HiSparse prefix sharing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class HiSparseRadixCache:
    """Radix tree managing CPU host KV indices for HiSparse prefix sharing.

    Wraps a RadixCache where TreeNode.value stores host pool indices.
    HostKVCache is used directly as the allocator since RadixCache only
    needs .device and .free() from it, both of which HostKVCache provides.
    """

    def __init__(self, host_pool: HostKVCache):
        params = CacheInitParams(
            disable=False,
            # We only use tree primitives (match/insert/evict/lock), never the
            # high-level cache_finished_req / cache_unfinished_req which read
            # from req_to_token_pool.  Index mapping is managed externally by
            # HiSparseCoordinator.req_to_host_pool.
            req_to_token_pool=None,
            token_to_kv_pool_allocator=host_pool,
            page_size=host_pool.page_size,
        )
        self._cache = RadixCache(params)
        self.host_pool = host_pool

    def match_prefix(
        self,
        token_ids: list,
        extra_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, TreeNode, int]:
        """Match prefix against the host radix tree.

        Returns:
            (host_indices, last_node, matched_len) where host_indices is a 1-D
            int64 tensor of matched host pool indices.
        """
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        result = self._cache.match_prefix(MatchPrefixParams(key=key))
        return result.device_indices, result.last_device_node, len(result.device_indices)

    def insert(
        self,
        token_ids: list,
        host_indices: torch.Tensor,
        extra_key: Optional[str] = None,
    ) -> int:
        """Insert a sequence's host indices into the tree.

        Returns:
            prefix_len: number of tokens already present in the tree.
        """
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        result = self._cache.insert(InsertParams(key=key, value=host_indices))
        return result.prefix_len

    def inc_lock_ref(self, node: TreeNode):
        return self._cache.inc_lock_ref(node)

    def dec_lock_ref(self, node: TreeNode):
        return self._cache.dec_lock_ref(node)

    def evict(self, num_tokens: int):
        return self._cache.evict(EvictParams(num_tokens=num_tokens))

    def evictable_size(self) -> int:
        return self._cache.evictable_size()

    @property
    def root_node(self) -> TreeNode:
        return self._cache.root_node
