import unittest

import torch

from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    ALLOC_MEMORY_FUNCS,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    alloc_with_pin_memory,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")


def _token_indices_for_pages(pages, page_size, device):
    parts = [
        torch.arange(
            int(p) * page_size,
            (int(p) + 1) * page_size,
            device=device,
            dtype=torch.int64,
        )
        for p in pages.tolist()
    ]
    return torch.cat(parts, dim=0)


class TestMHAJITTransfer(unittest.TestCase):
    """Test MHA (non-MLA) JIT kernel transfers for layer_first and page_first."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required.")
        if is_npu() or is_xpu():
            self.skipTest("Only CUDA/ROCm supported.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def _create_pools(self, layout):
        # head_num=8, head_dim=128 → element_size = 8*128*2 = 2048 = 128*16 ✓
        page_size = 1 if is_hip() else 16
        layer_num = 2
        head_num = 8
        head_dim = 128
        size = page_size * 8

        device_pool = MHATokenToKVPool(
            size=size,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )

        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            host_pool = MHATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=True,
                device="cpu",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        self.assertTrue(host_pool.can_use_jit, "JIT kernel should be available for MHA")
        return device_pool, host_pool, page_size, layer_num

    def _fill_device_pool(self, device_pool, layer_num):
        for layer_id in range(layer_num):
            k_buf = device_pool.k_buffer[layer_id]
            k_data = torch.arange(
                k_buf.numel(), device=k_buf.device, dtype=k_buf.dtype
            ).view_as(k_buf)
            k_buf.copy_(k_data + layer_id)
            v_buf = device_pool.v_buffer[layer_id]
            v_data = torch.arange(
                v_buf.numel(), device=v_buf.device, dtype=v_buf.dtype
            ).view_as(v_buf)
            v_buf.copy_(v_data + layer_id + 100)

    def _run_backup_and_load(self, layout):
        device_pool, host_pool, page_size, layer_num = self._create_pools(layout)
        self._fill_device_pool(device_pool, layer_num)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
        device_indices = _token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = _token_indices_for_pages(host_pages, page_size, device="cuda")

        # backup: device -> host (all layer)
        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, "kernel"
        )
        torch.cuda.synchronize()

        # verify backup
        for layer_id in range(layer_num):
            for hp, dp in zip(host_pages.tolist(), device_pages.tolist()):
                hs, ds = hp * page_size, dp * page_size
                got_k = host_pool.k_data_refs[layer_id][hs : hs + page_size].cpu()
                exp_k = device_pool.k_buffer[layer_id][ds : ds + page_size].cpu()
                self.assertTrue(
                    torch.equal(got_k, exp_k),
                    f"k backup mismatch layout={layout} layer={layer_id} hp={hp} dp={dp}",
                )
                got_v = host_pool.v_data_refs[layer_id][hs : hs + page_size].cpu()
                exp_v = device_pool.v_buffer[layer_id][ds : ds + page_size].cpu()
                self.assertTrue(
                    torch.equal(got_v, exp_v),
                    f"v backup mismatch layout={layout} layer={layer_id} hp={hp} dp={dp}",
                )

        # load: host -> device (per layer)
        for layer_id in range(layer_num):
            device_pool.k_buffer[layer_id].zero_()
            device_pool.v_buffer[layer_id].zero_()

        load_device_pages = torch.tensor([4, 5, 6], device="cuda", dtype=torch.int64)
        load_device_indices = _token_indices_for_pages(
            load_device_pages, page_size, device="cuda"
        )

        for layer_id in range(layer_num):
            host_pool.load_to_device_per_layer(
                device_pool, host_indices, load_device_indices, layer_id, "kernel"
            )
        torch.cuda.synchronize()

        # verify load
        for layer_id in range(layer_num):
            for hp, dp in zip(host_pages.tolist(), load_device_pages.tolist()):
                hs, ds = hp * page_size, dp * page_size
                got_k = device_pool.k_buffer[layer_id][ds : ds + page_size].cpu()
                exp_k = host_pool.k_data_refs[layer_id][hs : hs + page_size].cpu()
                self.assertTrue(
                    torch.equal(got_k, exp_k),
                    f"k load mismatch layout={layout} layer={layer_id} hp={hp} dp={dp}",
                )
                got_v = device_pool.v_buffer[layer_id][ds : ds + page_size].cpu()
                exp_v = host_pool.v_data_refs[layer_id][hs : hs + page_size].cpu()
                self.assertTrue(
                    torch.equal(got_v, exp_v),
                    f"v load mismatch layout={layout} layer={layer_id} hp={hp} dp={dp}",
                )

    def test_mha_layer_first(self):
        self._run_backup_and_load("layer_first")

    def test_mha_page_first(self):
        self._run_backup_and_load("page_first")


class TestMLAJITTransfer(unittest.TestCase):
    """Test MLA JIT kernel transfers for layer_first and page_first."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required.")
        if is_npu() or is_xpu():
            self.skipTest("Only CUDA/ROCm supported.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def _create_pools(self, layout):
        # kv_cache_dim=256 → element_size = 256*2 = 512 = 128*4 ✓
        page_size = 1 if is_hip() else 16
        layer_num = 2
        kv_lora_rank = 192
        qk_rope_head_dim = 64
        size = page_size * 8

        device_pool = MLATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dtype=torch.bfloat16,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )

        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            host_pool = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=True,
                device="cpu",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        self.assertTrue(host_pool.can_use_jit, "JIT kernel should be available for MLA")
        return device_pool, host_pool, page_size, layer_num

    def _fill_device_pool(self, device_pool, layer_num):
        for layer_id in range(layer_num):
            buf = device_pool.kv_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=buf.dtype
            ).view_as(buf)
            buf.copy_(data + layer_id)

    def _run_backup_and_load(self, layout):
        device_pool, host_pool, page_size, layer_num = self._create_pools(layout)
        self._fill_device_pool(device_pool, layer_num)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
        device_indices = _token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = _token_indices_for_pages(host_pages, page_size, device="cuda")

        # backup: device -> host (all layer)
        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, "kernel"
        )
        torch.cuda.synchronize()

        # verify backup
        for layer_id in range(layer_num):
            for hp, dp in zip(host_pages.tolist(), device_pages.tolist()):
                hs, ds = hp * page_size, dp * page_size
                got = host_pool.data_refs[layer_id][hs : hs + page_size].cpu()
                expected = device_pool.kv_buffer[layer_id][ds : ds + page_size].cpu()
                self.assertTrue(
                    torch.equal(got, expected),
                    f"backup mismatch layout={layout} layer={layer_id} hp={hp} dp={dp}",
                )

        # load: host -> device (per layer)
        for layer_id in range(layer_num):
            device_pool.kv_buffer[layer_id].zero_()

        load_device_pages = torch.tensor([4, 5, 6], device="cuda", dtype=torch.int64)
        load_device_indices = _token_indices_for_pages(
            load_device_pages, page_size, device="cuda"
        )

        for layer_id in range(layer_num):
            host_pool.load_to_device_per_layer(
                device_pool, host_indices, load_device_indices, layer_id, "kernel"
            )
        torch.cuda.synchronize()

        # verify load
        for layer_id in range(layer_num):
            for hp, dp in zip(host_pages.tolist(), load_device_pages.tolist()):
                hs, ds = hp * page_size, dp * page_size
                got = device_pool.kv_buffer[layer_id][ds : ds + page_size].cpu()
                expected = host_pool.data_refs[layer_id][hs : hs + page_size].cpu()
                self.assertTrue(
                    torch.equal(got, expected),
                    f"load mismatch layout={layout} layer={layer_id} hp={hp} dp={dp}",
                )

    def test_mla_layer_first(self):
        self._run_backup_and_load("layer_first")

    def test_mla_page_first(self):
        self._run_backup_and_load("page_first")


if __name__ == "__main__":
    unittest.main()
