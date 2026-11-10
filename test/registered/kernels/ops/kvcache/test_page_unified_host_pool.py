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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))


@pytest.mark.parametrize(
    "native_format",
    [
        "asymmetric",
        "fp4",
        "mxfp8",
        "hnd",
        "paged",
        "vectorized",
        "mla_fp4",
        "mla_split",
    ],
)
@pytest.mark.parametrize("backend", ["kernel", "direct"])
def test_native_page_unified_payload_and_scales(native_format, backend):
    """The real device copies preserve each payload/scale field at new slots."""
    from unittest.mock import patch

    from sglang.srt.mem_cache.device_pool_info import (
        KVBufferLayout,
        KVBuffers,
        KVScaleBuffers,
        KVScaleLayout,
        MHABufferInfo,
        MLABufferInfo,
        MLASplitBuffers,
    )
    from sglang.srt.mem_cache.pool_host.common import (
        ALLOC_MEMORY_FUNCS,
        alloc_with_pin_memory,
    )
    from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
    from sglang.srt.mem_cache.pool_host.mla import get_mla_host_pool_cls

    pages, layers, heads = 6, 2, 2
    originals = []

    def buffers(width, *, paged=False, scale=False):
        if native_format == "mxfp8" and scale:
            shape = (pages, heads, 2, PAGE_SIZE // 2, width)
        elif native_format == "hnd":
            shape = (pages, heads, PAGE_SIZE, width)
        elif native_format == "vectorized":
            shape = (pages, heads, width // 2, PAGE_SIZE, 2)
        elif paged:
            shape = (pages, PAGE_SIZE, heads, width)
        else:
            shape = (pages * PAGE_SIZE, heads, width)
        result = tuple(
            torch.empty(shape, dtype=torch.uint8, device=DEVICE) for _ in range(layers)
        )
        originals.append(
            (
                result,
                paged
                or native_format in ("hnd", "vectorized")
                or (native_format == "mxfp8" and scale),
            )
        )
        return result

    if native_format.startswith("mla"):
        payload = tuple(
            torch.empty(pages * PAGE_SIZE, 1, 32, dtype=torch.uint8, device=DEVICE)
            for _ in range(layers)
        )
        scales = tuple(
            torch.empty(pages * PAGE_SIZE, 2, dtype=torch.uint8, device=DEVICE)
            for _ in range(layers)
        )
        if native_format == "mla_split":
            k = torch.empty(
                layers, pages, PAGE_SIZE, 1, 32, dtype=torch.uint8, device=DEVICE
            )
            v = torch.empty(
                layers, pages, PAGE_SIZE, 1, 16, dtype=torch.uint8, device=DEVICE
            )
            native = MLASplitBuffers(
                k=k,
                v=v,
                index_k=None,
                index_scales=None,
                index_layer_indices=(),
                k_width=32,
                v_width=16,
                index_width=0,
            )
            originals = [(tuple(k), True), (tuple(v), True)]
            scales = None
        else:
            native = payload
            originals = [(payload, False), (scales, False)]
        info = MLABufferInfo(
            buffers=native,
            scales=scales,
            data_ptrs=None,
            layer_indices=tuple(range(layers)),
            device_layer_num=layers,
            page_size=PAGE_SIZE,
            kv_lora_rank=32,
            qk_rope_head_dim=32,
            kv_cache_dim=64,
            dtype=torch.uint8,
            device=torch.device(DEVICE, torch.cuda.current_device()),
        )
        host_cls = get_mla_host_pool_cls(info, layout="page_unified")
        kwargs = {"target_layer_num": layers}
    else:
        layout = {
            "hnd": KVBufferLayout.HND,
            "paged": KVBufferLayout.PAGED_NHD,
            "vectorized": KVBufferLayout.VECTORIZED_5D,
        }.get(native_format, KVBufferLayout.NHD)
        k = buffers(32, paged=native_format == "paged")
        v = buffers(48, paged=native_format == "paged")
        scales = None
        if native_format in ("fp4", "mxfp8"):
            scales = KVScaleBuffers(
                k=buffers(2, scale=True),
                v=buffers(3, scale=True),
                layout=KVScaleLayout.MXFP8_INTERLEAVED
                if native_format == "mxfp8"
                else KVScaleLayout.TOKEN,
            )
        info = MHABufferInfo(
            buffers=KVBuffers(
                k=k, v=v, k_data_ptrs=None, v_data_ptrs=None, layout=layout
            ),
            scales=scales,
            page_size=PAGE_SIZE,
            head_num=heads,
            head_dim=64,
            v_head_dim=96,
            dtype=torch.uint8,
            device=torch.device(DEVICE, torch.cuda.current_device()),
        )
        host_cls = get_mha_host_pool_cls(info, layout="page_unified")
        kwargs = {"head_group_num": heads}

    def allocate_pinned(*args, **kwargs):
        buffer = alloc_with_pin_memory(*args, **kwargs)
        # Torch owns the pin; this buffer did not use cudaHostRegister.
        buffer._sglang_cuda_host_registered_ranges = []
        return buffer

    with patch.dict(ALLOC_MEMORY_FUNCS, {DEVICE: allocate_pinned}):
        host = host_cls(
            info,
            2,
            0,
            PAGE_SIZE,
            "page_unified",
            pin_memory=True,
            device_capacity_tokens=pages * PAGE_SIZE,
            start_layer=0,
            end_layer=layers,
            **kwargs,
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        # The controller runs HiCache copies on dedicated transfer streams.
        with torch.cuda.stream(stream):
            for ordinal, (component, _) in enumerate(originals):
                for layer, buffer in enumerate(component):
                    buffer.copy_(
                        (
                            (
                                torch.arange(buffer.numel(), device=DEVICE)
                                + ordinal * 37
                                + layer * 11
                            )
                            % 251
                        )
                        .reshape_as(buffer)
                        .to(buffer.dtype)
                    )
            saved = [
                [buffer.clone() for buffer in component] for component, _ in originals
            ]
            source, target = [3, 1], [5, 2]
            host_indices = _page_indices([0, 4], PAGE_SIZE)
            host.backup_from_device_all_layer(
                None, host_indices, _page_indices(source, PAGE_SIZE).to(DEVICE), backend
            )
            torch.cuda.synchronize()
            for component, _ in originals:
                for buffer in component:
                    buffer.fill_(173)
            for layer in range(layers):
                host.load_to_device_per_layer(
                    None,
                    host_indices.to(DEVICE) if backend == "kernel" else host_indices,
                    _page_indices(target, PAGE_SIZE).to(DEVICE),
                    layer,
                    backend,
                )
            torch.cuda.synchronize()
            for (component, paged), previous in zip(originals, saved):
                src = (
                    torch.tensor(source, device=DEVICE)
                    if paged
                    else _page_indices(source, PAGE_SIZE).to(DEVICE)
                )
                dst = (
                    torch.tensor(target, device=DEVICE)
                    if paged
                    else _page_indices(target, PAGE_SIZE).to(DEVICE)
                )
                for actual, expected in zip(component, previous):
                    assert torch.equal(
                        actual.index_select(0, dst), expected.index_select(0, src)
                    )
                    untouched = torch.ones(
                        actual.shape[0], dtype=torch.bool, device=DEVICE
                    )
                    untouched[dst] = False
                    assert torch.all(actual[untouched] == 173)
            assert host.kv_buffer.is_pinned()
            assert (
                host.get_data_page(0).nbytes == host.page_unified_layout.bytes_per_page
            )
    finally:
        stream.synchronize()
        host.destroy()
