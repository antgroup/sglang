"""The page_unified host pool round trip: device -> host -> device.

Write-back (#39606's staged relayout) and load-back (the page_unified JIT
kernel) are two halves of one contract, and the contract is a byte order.
Each kernel has its own unit tests against its own reference; what neither can
catch is the pair disagreeing, because a permutation applied twice in opposite
directions cancels only if both sides read the same geometry off the same pool.

So this drives the real ``MHATokenToKVPoolHost`` / ``MLATokenToKVPoolHost``
transfer arms rather than the kernels directly, and compares the device pool
against itself.
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="page_unified host pool tests require CUDA/ROCm.",
)

DEVICE = "cuda"
PAGE_SIZE = 16
LAYERS = 4
HEAD_DIM = 128


def _page_indices(pages, page_size):
    offsets = torch.arange(page_size, dtype=torch.int64)
    return (
        torch.tensor(pages, dtype=torch.int64)[:, None] * page_size + offsets
    ).flatten()


def _pinned_host_pool(host_pool_cls, device_pool, **kwargs):
    """Allocate the host pool with pin_memory rather than cudaHostRegister.

    Registering is what the runtime does, but it is process-global: a second
    pool landing on a freed pool's address fails with
    cudaErrorHostMemoryAlreadyRegistered. Parametrized cases would then pass or
    fail on allocator reuse, which says nothing about the layout.
    """
    from sglang.srt.mem_cache.device_pool_info import MLABufferInfo
    from sglang.srt.mem_cache.pool_host.common import (
        ALLOC_MEMORY_FUNCS,
        alloc_with_pin_memory,
    )

    buffer_info = device_pool.get_device_pool_infos()[0].buffer_info
    if isinstance(buffer_info, MLABufferInfo):
        kwargs.update(
            target_layer_num=device_pool.layer_num,
            override_kv_cache_dim=device_pool.kv_cache_dim,
        )
    original = ALLOC_MEMORY_FUNCS[DEVICE]
    ALLOC_MEMORY_FUNCS[DEVICE] = alloc_with_pin_memory
    try:
        return host_pool_cls(
            buffer_info,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="page_unified",
            pin_memory=True,
            device="cpu",
            **kwargs,
        )
    finally:
        ALLOC_MEMORY_FUNCS[DEVICE] = original


def _restore_storage_pages(host_pool, host_indices):
    """Exercise the flat page API used by generic file storage at new offsets."""
    stored_pages = [
        host_pool.get_data_page(int(index)).clone()
        for index in host_indices[::PAGE_SIZE]
    ]
    dummy_page = host_pool.get_dummy_flat_data_page()
    assert all(page.shape == dummy_page.shape for page in stored_pages)
    assert dummy_page.nbytes == host_pool.page_unified_layout.bytes_per_page

    host_pool.kv_buffer.zero_()
    restored_indices = _page_indices([7, 3, 5, 1], PAGE_SIZE)
    for index, page in zip(restored_indices[::PAGE_SIZE], stored_pages):
        host_pool.set_from_flat_data_page(int(index), page)
        restored_page = host_pool.get_data_page(int(index), flat=False)
        assert restored_page.is_contiguous()
        assert torch.equal(restored_page.flatten(), page)
    return restored_indices


def _prepare_staging(device_pool, *, is_mla=False, head_group_num=1):
    from sglang.srt.mem_cache.pool_host.base import (
        prepare_page_unified_write_back_staging,
    )
    from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout

    geometry = PageUnifiedLayout(
        page_size=PAGE_SIZE,
        layer_num=LAYERS,
        head_num=1 if is_mla else device_pool.head_num,
        head_group_num=head_group_num,
        head_dim=device_pool.kv_cache_dim if is_mla else HEAD_DIM,
        itemsize=device_pool.store_dtype.itemsize,
        is_mla=is_mla,
    )
    return prepare_page_unified_write_back_staging(
        geometry, dtype=device_pool.store_dtype, device=torch.device(DEVICE)
    )


@pytest.mark.parametrize("head_num,head_group_num", [(8, 1), (8, 2), (8, 8), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("preallocate", [False, True])
def test_mha_page_unified(head_num, head_group_num, dtype, preallocate):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    device_pool = MHATokenToKVPool(
        size=PAGE_SIZE * 8,
        page_size=PAGE_SIZE,
        dtype=dtype,
        head_num=head_num,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYERS,
    )
    staging = (
        _prepare_staging(device_pool, head_group_num=head_group_num)
        if preallocate
        else None
    )
    host_pool = _pinned_host_pool(
        MHATokenToKVPoolHost,
        device_pool,
        head_group_num=head_group_num,
        write_back_staging=staging,
    )
    if preallocate:
        assert host_pool.staging_buffer.data_ptr() == staging.data_ptr()
        assert (
            host_pool.staging_buffer.shape[0] == host_pool.page_num < staging.shape[0]
        )

    generator = torch.Generator(device=DEVICE).manual_seed(20260915)
    for layer in range(LAYERS):
        for buffer in (device_pool.k_buffer[layer], device_pool.v_buffer[layer]):
            buffer.copy_(
                torch.randint(
                    0,
                    128,
                    buffer.shape,
                    generator=generator,
                    dtype=torch.int32,
                    device=DEVICE,
                ).to(dtype)
            )
    original = [
        (device_pool.k_buffer[layer].clone(), device_pool.v_buffer[layer].clone())
        for layer in range(LAYERS)
    ]

    # Page order is shuffled and the host pages are not the device pages: an
    # arm that ignored one of the two index vectors would still pass on an
    # identity mapping.
    device_indices = _page_indices([5, 1, 3, 0], PAGE_SIZE).to(DEVICE)
    host_indices_cpu = _page_indices([2, 6, 0, 4], PAGE_SIZE)

    # Write-back takes CPU host page ids, which is what the cache controller
    # hands the staged paths.
    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_cpu, device_indices, "kernel"
    )
    torch.cuda.synchronize()
    host_indices_cpu = _restore_storage_pages(host_pool, host_indices_cpu)

    for layer in range(LAYERS):
        device_pool.k_buffer[layer].zero_()
        device_pool.v_buffer[layer].zero_()

    # Load-back takes device-resident host indices, as move_indices produces.
    host_indices = host_indices_cpu.to(DEVICE)
    for layer in range(LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer, "kernel"
        )
    torch.cuda.synchronize()

    for layer in range(LAYERS):
        expected_k, expected_v = original[layer]
        assert torch.equal(
            device_pool.k_buffer[layer][device_indices], expected_k[device_indices]
        ), f"K mismatch at layer {layer}"
        assert torch.equal(
            device_pool.v_buffer[layer][device_indices], expected_v[device_indices]
        ), f"V mismatch at layer {layer}"


@pytest.mark.parametrize("kv_lora_rank,qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("preallocate", [False, True])
def test_mla_page_unified(kv_lora_rank, qk_rope_head_dim, dtype, preallocate):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    device_pool = MLATokenToKVPool(
        size=PAGE_SIZE * 8,
        page_size=PAGE_SIZE,
        dtype=dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=LAYERS,
    )
    staging = _prepare_staging(device_pool, is_mla=True) if preallocate else None
    host_pool = _pinned_host_pool(
        MLATokenToKVPoolHost, device_pool, write_back_staging=staging
    )
    if preallocate:
        assert host_pool.staging_buffer.data_ptr() == staging.data_ptr()
        assert (
            host_pool.staging_buffer.shape[0] == host_pool.page_num < staging.shape[0]
        )

    generator = torch.Generator(device=DEVICE).manual_seed(20260916)
    for layer in range(LAYERS):
        buffer = device_pool.kv_buffer[layer]
        buffer.copy_(
            torch.randint(
                0,
                128,
                buffer.shape,
                generator=generator,
                dtype=torch.int32,
                device=DEVICE,
            ).to(dtype)
        )
    original = [device_pool.kv_buffer[layer].clone() for layer in range(LAYERS)]

    device_indices = _page_indices([5, 1, 3, 0], PAGE_SIZE).to(DEVICE)
    host_indices_cpu = _page_indices([2, 6, 0, 4], PAGE_SIZE)

    host_pool.backup_from_device_all_layer(
        device_pool, host_indices_cpu, device_indices, "kernel"
    )
    torch.cuda.synchronize()
    host_indices_cpu = _restore_storage_pages(host_pool, host_indices_cpu)
    for layer in range(LAYERS):
        device_pool.kv_buffer[layer].zero_()

    host_indices = host_indices_cpu.to(DEVICE)
    for layer in range(LAYERS):
        host_pool.load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer, "kernel"
        )
    torch.cuda.synchronize()

    for layer in range(LAYERS):
        assert torch.equal(
            device_pool.kv_buffer[layer][device_indices],
            original[layer][device_indices],
        ), f"latent KV mismatch at layer {layer}"


def _expected_serialized_page(host, native_layers):
    """Reference byte addresses from the native format, independent of its codec."""
    from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
    from sglang.srt.mem_cache.pool_host.minimax_m3_nvfp4 import (
        MiniMaxM3Nvfp4IndexKPoolHost,
        MiniMaxM3Nvfp4MainKVPoolHost,
    )

    pieces = []
    if isinstance(host, DSAIndexerPoolHost):
        info = host._index_key_info
        slots = host.page_size // info.compress_ratio
        native_slots = info.kernel_page_size or slots
        tile = info.key_shuffle_tile
        for row in native_layers:
            for token in range(slots):
                page, offset = divmod(token, native_slots)
                base = page * native_slots * 132
                if tile:
                    # HIP's native cache tiles token/channel coordinates.
                    key = row[
                        torch.tensor(
                            [
                                base
                                + (
                                    ((offset // tile) * (128 // tile) + channel // tile)
                                    * tile
                                    + offset % tile
                                )
                                * tile
                                + channel % tile
                                for channel in range(128)
                            ]
                        )
                    ]
                else:
                    key = row[base + offset * 128 : base + (offset + 1) * 128]
                scale = base + native_slots * 128 + offset * 4
                pieces.extend((key, row[scale : scale + 4]))
    elif isinstance(host, MiniMaxM3Nvfp4MainKVPoolHost):
        slots, heads = host.page_size, host.head_num
        group_heads = heads // host.head_group_num
        payload_bytes = heads * slots * 64
        scale_bytes = heads * slots * 8
        for group in range(host.head_group_num):
            for row in native_layers:
                for kv in range(2):
                    base = kv * (payload_bytes + scale_bytes)
                    for token in range(slots):
                        for head in range(
                            group * group_heads, (group + 1) * group_heads
                        ):
                            payload = base + (head * slots + token) * 64
                            scale = base + payload_bytes + (head * slots + token) * 8
                            pieces.extend(
                                (row[payload : payload + 64], row[scale : scale + 8])
                            )
    elif isinstance(host, MiniMaxM3Nvfp4IndexKPoolHost):
        slots = host.page_size
        for row in native_layers:
            for token in range(slots):
                scale = slots * 64 + token * 8
                pieces.extend(
                    (row[token * 64 : (token + 1) * 64], row[scale : scale + 8])
                )
    else:
        pieces = native_layers
    return torch.cat(pieces)


def _specialized_round_trip(host, buffers, backend, layer_ids=None):
    """Move real CUDA bytes through relocated host pages and serialized pages."""
    page_size = host.page_size
    source_pages, host_pages, restored_pages = [3, 1, 4], [0, 4, 2], [5, 1, 3]
    host_indices = _page_indices(host_pages, page_size)
    restored_indices = _page_indices(restored_pages, page_size)
    if backend == "kernel":
        host_indices, restored_indices = host_indices.cuda(), restored_indices.cuda()
    device_indices = _page_indices(source_pages, page_size).cuda()
    expected = []
    for layer, buffer in enumerate(buffers):
        rows = buffer.view(torch.uint8).reshape(6, -1)
        rows.copy_(
            (torch.arange(rows.numel(), device=DEVICE, dtype=torch.int32) + layer * 43)
            .remainder(251)
            .to(torch.uint8)
            .reshape_as(rows)
        )
        expected.append(rows[source_pages].cpu())
    backing = host.get_hybrid_pool_buffer()[0]
    assert backing.is_pinned()
    backing.fill_(239)
    host.backup_from_device_all_layer(None, host_indices, device_indices, backend)
    torch.cuda.synchronize()
    serialized = [host.get_data_page(page * page_size).clone() for page in host_pages]
    for index, page in enumerate(serialized):
        assert torch.equal(
            page.view(torch.uint8).flatten(),
            _expected_serialized_page(host, [layer[index] for layer in expected]),
        )
    assert torch.all(backing[6] == 239)
    backing.zero_()
    for page, data in zip(restored_pages, serialized):
        host.set_from_flat_data_page(page * page_size, data)
    assert torch.all(backing[0] == 0)
    for buffer in buffers:
        buffer.zero_()
    for layer in range(len(buffers)) if layer_ids is None else layer_ids:
        host.load_to_device_per_layer(
            None, restored_indices, device_indices, layer, backend
        )
    torch.cuda.synchronize()
    for buffer, original in zip(buffers, expected):
        rows = buffer.view(torch.uint8).reshape(6, -1)
        assert torch.equal(rows[source_pages].cpu(), original)
        assert torch.count_nonzero(rows[[0, 2, 5]]) == 0


@pytest.mark.parametrize("backend", ["kernel", "direct"])
@pytest.mark.parametrize(
    "kind,dtype",
    [
        ("dsa", torch.uint8),
        ("dsa_shuffled", torch.uint8),
        ("k_only", torch.bfloat16),
        ("dsv4", torch.uint8),
        ("dsv4", torch.float16),
        ("dsv4", torch.float32),
        ("dsv4_state", torch.uint8),
        ("minimax_nvfp4", torch.uint8),
    ],
)
def test_specialized_page_unified_cuda(kind, dtype, backend):
    from types import SimpleNamespace

    from sglang.srt.mem_cache.deepseek_v4_buffer_info import DSV4StateBufferInfo
    from sglang.srt.mem_cache.device_pool_info import (
        IndexKeyBufferInfo,
        IndexKeyLayout,
        PageRowBufferInfo,
    )
    from sglang.srt.mem_cache.memory_pool_host import (
        DeepSeekV4PagedHostPool,
        DeepSeekV4StateHostPool,
    )
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
    from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKOnlyPoolHost
    from sglang.srt.mem_cache.pool_host.minimax_m3_nvfp4 import (
        MiniMaxM3Nvfp4IndexKPoolHost,
        MiniMaxM3Nvfp4MainKVPoolHost,
    )

    def buffers(shape, dtype=dtype):
        return tuple(torch.empty(shape, dtype=dtype, device=DEVICE) for _ in range(2))

    hosts = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with host_memory_budget_scope(4 << 20), torch.cuda.stream(stream):
            if kind.startswith("dsa"):
                device = buffers((6, PAGE_SIZE * 132))
                info = IndexKeyBufferInfo(
                    page_size=PAGE_SIZE,
                    buffers=device,
                    compress_ratio=1,
                    key_layout=IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES,
                    key_shuffle_tile=16 if kind == "dsa_shuffled" else 0,
                )
                host = DSAIndexerPoolHost(
                    info,
                    layer_mapping={1: 0, 3: 1},
                    target_device_layer_num=4,
                    num_host_pages=8,
                    transfer_page_size=PAGE_SIZE,
                    layout="page_unified",
                    buffer_device=DEVICE,
                    dtype=torch.bfloat16,
                )
                hosts.append(host)
                _specialized_round_trip(host, device, backend, layer_ids=[1, 3])
            elif kind == "k_only":
                device = buffers((6 * PAGE_SIZE, 1, HEAD_DIM))
                info = IndexKeyBufferInfo(
                    page_size=PAGE_SIZE,
                    buffers=device,
                    compress_ratio=1,
                    key_layout=IndexKeyLayout.KEY_ROWS,
                )
                host = MHATokenToKOnlyPoolHost(
                    info,
                    SimpleNamespace(
                        page_size=PAGE_SIZE, page_num=8, size=8 * PAGE_SIZE
                    ),
                    "page_unified",
                )
                hosts.append(host)
                _specialized_round_trip(host, device, backend)
            elif kind == "dsv4":
                device = buffers((6, 4, 1, 32 // dtype.itemsize))
                host = DeepSeekV4PagedHostPool(
                    "c4",
                    device,
                    item_bytes=128,
                    num_host_pages=8,
                    slot_page_size=PAGE_SIZE,
                    layout="page_unified",
                    page_aligned_only=True,
                )
                hosts.append(host)
                _specialized_round_trip(host, device, backend)
            elif kind == "dsv4_state":
                infos = [
                    DSV4StateBufferInfo(
                        pages=torch.empty((6, 128), dtype=torch.uint8, device=DEVICE),
                        ring_size=4,
                        dtype=state_dtype,
                    )
                    for state_dtype in (torch.bfloat16, torch.float32)
                ]
                host = DeepSeekV4StateHostPool(
                    "c4_state",
                    infos,
                    num_host_pages=8,
                    swa_page_size=PAGE_SIZE,
                    layout="page_unified",
                )
                hosts.append(host)
                _specialized_round_trip(host, [info.pages for info in infos], backend)
            else:
                # Transfer packed payload/scales directly: no FP4 quantizer or
                # SM100 arithmetic kernel participates in this byte codec test.
                def info(heads, components):
                    slab = torch.empty(
                        (2, 6, components * heads * 128 * 72),
                        dtype=torch.uint8,
                        device=DEVICE,
                    )
                    return PageRowBufferInfo(
                        slab=slab,
                        rows=tuple(slab),
                        row_bytes=slab.shape[-1],
                        page_size=128,
                        num_pages=6,
                        head_num=heads,
                        global_head_start=0,
                        global_head_num=heads,
                    )

                main_info, index_info = info(2, 2), info(1, 1)
                main = MiniMaxM3Nvfp4MainKVPoolHost(
                    main_info,
                    2,
                    0,
                    128,
                    "page_unified",
                    device_capacity_tokens=5 * 128,
                    start_layer=0,
                    end_layer=2,
                    head_group_num=2,
                )
                hosts.append(main)
                index = MiniMaxM3Nvfp4IndexKPoolHost(
                    index_info, main, "page_unified", start_layer=0, end_layer=2
                )
                hosts.append(index)
                _specialized_round_trip(main, main_info.rows, backend)
                _specialized_round_trip(index, index_info.rows, backend)
        stream.synchronize()
    finally:
        stream.synchronize()
        for host in reversed(hosts):
            host.destroy()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
