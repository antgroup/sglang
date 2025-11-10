import torch

from sglang.srt.mem_cache.memory_pool import DCPAwareMLATokenToKVPool


def test_dcp_mla_get_physical_loc_basic():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small pool just to exercise mapping; we won't allocate KV data
    pool = DCPAwareMLATokenToKVPool(
        size=64,
        page_size=1,
        dtype=torch.float16,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        dcp_rank=1,
        dcp_world_size=4,
        start_layer=0,
        end_layer=0,
    )

    locs = torch.arange(0, 32, dtype=torch.int32, device=pool.device)
    phys, mask = pool.get_physical_loc(locs)

    assert mask.dtype == torch.bool
    assert phys.dtype == torch.int32

    # Only tokens with loc % 4 == 1 belong to rank 1
    expected_mask = (locs % 4) == 1
    assert torch.equal(mask, expected_mask)

    # Physical locations are compacted by floor(loc / 4)
    expected_phys = torch.div(locs, 4, rounding_mode="floor").to(torch.int32)
    # For tokens not owned, phys is still defined but we only care about owned subset
    assert torch.equal(phys[mask], expected_phys[mask])


