// Modified copy of humming-kernels 0.1.12 include/humming/datatype/dequant_fused.cuh (Apache-2.0).
// SGLang header overlay v0_1_12_ntail (DeepSeek-V4.1 SM90 MoE path); see
// sglang/srt/layers/moe/moe_runner/humming_overlay.py. Changes: leaner MXFP4 -> FP8 fused-e8m0 dequant
// instruction sequence (identical bytes).

#pragma once

#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class TargetType>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4(const uint32_t qb, const uint32_t exp_offset) {
  static_assert(std::is_same<TargetType, Float8E4M3>::value || std::is_same<TargetType, Int8>::value);
  return {0, 0};
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Float8E4M3>(const uint32_t qb, const uint32_t exp_offset) {
  // [sglang] Same bytes as upstream, fewer ALU-pipe instructions (as in humming-kernels 0.1.15):
  // one 0x77777777 mask feeds both permutes (__byte_perm re-masks each half with 0x7777), and raw
  // prmt.b32 skips that masking. Selector bit 3 is zero, so prmt default mode == __byte_perm.
  const uint32_t exp_offset_buffer1 = (exp_offset * 0x08080800) + ((0x03020100 << 2) - 0x00000400);
  const uint32_t exp_offset_buffer2 = (exp_offset * 0x08080808) + (0x07060504 << 2);
  const uint32_t sel = qb & 0x77777777;
  const uint32_t sel_hi = sel >> 16;
  uint32_t lo, hi;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(lo) : "r"(exp_offset_buffer1), "r"(exp_offset_buffer2), "r"(sel));
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(hi) : "r"(exp_offset_buffer1), "r"(exp_offset_buffer2), "r"(sel_hi));

  uint32_t res[2] = {
      lop3_and_or(qb << 4, 0x80808080, lo),
      lop3_and_or(qb, 0x80808080, hi)};

  return *reinterpret_cast<uint2 *>(res);
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Int8>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t buffer1 = 0x03020100 << exp_offset;
  uint32_t buffer2 = 0x0C080604 << exp_offset;

  uint32_t res[2];
  uint32_t signs[2] = {qb >> 3, qb >> 7};
  uint32_t int8s[2] = {
      __byte_perm(buffer1, buffer2, qb),
      __byte_perm(buffer1, buffer2, qb >> 16)};

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t val = __byte_perm(int8s[0], int8s[1], 0x6420 + 0x1111 * i);
    uint32_t flag = signs[i] & 0x01010101;
    uint32_t mask = flag * 0xFF;
    res[i] = (val - flag) ^ mask;
  }

  return *reinterpret_cast<uint2 *>(res);
}

template <class TargetType, uint32_t kCount, bool kUseWgmma>
CUDA_INLINE void fused_dequant_for_mxfp4(const uint32_t *qb_ptrs, uint32_t *res_ptrs, uint32_t *scales_ptr) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    // [sglang] scale byte extraction via IDP4A (FMA pipe) instead of SHF + LOP3 (as in 0.1.15)
    uint32_t exp_offset = __dp4a(scales_ptr[i / 4], 1u << (8 * (i % 4)), 0u);
    uint2 res = fused_dequant_single_for_mxfp4<TargetType>(qb_ptrs[i], exp_offset);
    res_ptrs[i * 2] = res.x;
    res_ptrs[i * 2 + 1] = res.y;
  }

  if constexpr (kUseWgmma) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kCount; i++) {
      uint32_t tmp = res_ptrs[i * 4 + 1];
      res_ptrs[i * 4 + 1] = res_ptrs[i * 4 + 2];
      res_ptrs[i * 4 + 2] = tmp;
    }
  }
}
