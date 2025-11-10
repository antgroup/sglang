import math
import torch

from sglang.srt.mem_cache.memory_pool import DCPAwareMLATokenToKVPool


def apply_dcp_filter_and_remap(kv_indices: torch.Tensor, kv_indptr: torch.Tensor, d: int, r: int):
    """
    Replicate the runtime DCP logic:
      - filter: own = (v % d) == r
      - remap: phys = floor(v / d)
      - per-seq lens: count of own tokens in each [kv_indptr[i]:kv_indptr[i+1])
    """
    mask = (kv_indices % d) == r
    kv_indices_remap = torch.div(kv_indices[mask], d, rounding_mode="floor").to(torch.int32)

    bs = kv_indptr.numel() - 1
    lens = torch.empty((bs,), dtype=torch.int32)
    for i in range(bs):
        st = int(kv_indptr[i].item())
        ed = int(kv_indptr[i + 1].item())
        lens[i] = int(mask[st:ed].sum().item())

    kv_indptr_remap = torch.empty_like(kv_indptr)
    kv_indptr_remap[0] = 0
    kv_indptr_remap[1:] = torch.cumsum(lens, dim=0)
    return kv_indices_remap, kv_indptr_remap, lens


def test_dcp_filter_remap_decode_simple():
    # kv_indices for 2 sequences: [0,1,2] and [3,4,5]
    kv_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 3, 6], dtype=torch.int32)
    d, r = 2, 1  # keep odd indices

    remap, indptr_remap, lens = apply_dcp_filter_and_remap(kv_indices, kv_indptr, d, r)
    # expected: keep [1] from first seq -> [0] after floor(1/2)
    #           keep [3,5] from second seq -> [1,2] after floor(/2)
    assert torch.equal(remap, torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(lens, torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(indptr_remap, torch.tensor([0, 1, 3], dtype=torch.int32))


def test_dcp_order_stable():
    kv_indices = torch.tensor([2, 3, 5, 8, 11], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 5], dtype=torch.int32)  # single sequence
    d, r = 3, 2  # keep values %3 == 2

    remap, _, _ = apply_dcp_filter_and_remap(kv_indices, kv_indptr, d, r)
    # kept: [2,5,11] -> floor(/3) = [0,1,3] (order preserved)
    assert torch.equal(remap, torch.tensor([0, 1, 3], dtype=torch.int32))


def test_dcp_local_capacity_cpu():
    # Ensure the per-rank pool reduces physical token capacity to ceil(size/d)
    device = "cpu"
    global_size = 100
    page_size = 1
    d = 4
    local_size = math.ceil(global_size / d)

    pool = DCPAwareMLATokenToKVPool(
        size=global_size,
        page_size=page_size,
        dtype=torch.float32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        layer_num=2,
        device=device,
        enable_memory_saver=False,
        dcp_rank=1,
        dcp_world_size=d,
        start_layer=0,
        end_layer=1,
    )

    # Check allocated kv_buffer token dimension equals local_size + page_size (padded slot 0)
    for buf in pool.kv_buffer:
        assert buf.shape[0] == local_size + page_size


