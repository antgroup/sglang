from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MLATokenToKVPoolHost,
    NSAIndexerPoolHost,
    PoolEntry,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def build_nsa_hybrid_stack(
    radix_cache: "HiRadixCache",
    params: "CacheInitParams",
    server_args: "ServerArgs",
    *,
    extra_config: dict,
    prefetch_threshold: int,
    enable_storage_metrics: bool,
    load_cache_event,
) -> None:
    """HostPoolGroup (KV + indexer) + HybridCacheController for NSA (DSA)."""
    kv = radix_cache.kv_cache
    mla_host = MLATokenToKVPoolHost(
        kv,
        server_args.hicache_ratio,
        server_args.hicache_size,
        radix_cache.page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        override_kv_cache_dim=kv.kv_cache_dim,
    )
    indexer_host = NSAIndexerPoolHost(
        kv,
        mla_host,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    layer_num = kv.layer_num

    def layer_mapper(layer_id: int):
        if 0 <= layer_id < layer_num:
            return layer_id
        return None

    host_pool_group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=mla_host,
                device_pool=kv,
                layer_mapper=layer_mapper,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.INDEXER,
                host_pool=indexer_host,
                device_pool=kv,
                layer_mapper=layer_mapper,
                share_indices_with_anchor=True,
            ),
        ]
    )
    radix_cache.full_kv_pool_host = mla_host
    radix_cache.token_to_kv_pool_host = host_pool_group
    radix_cache.cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        radix_cache.page_size,
        radix_cache.tp_group,
        load_cache_event=load_cache_event,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=server_args.hicache_storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=server_args.served_model_name,
        storage_backend_extra_config=extra_config,
        pp_rank=radix_cache.pp_rank,
        pp_size=radix_cache.pp_size,
        transfer_layer_num=layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    logger.info(
        "Hybrid hierarchical cache: HostPoolGroup(KV + INDEXER), HybridCacheController, "
        "transfer_layer_num=%s",
        layer_num,
    )
