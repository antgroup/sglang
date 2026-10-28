#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct MhcCollapseNormParams {
  const bf16_t* __restrict__ x;         // [m, 5120] sublayer output (post-mix only)
  const bf16_t* __restrict__ residual;  // [m, 4, 5120] HC streams
  const float* __restrict__ post;       // [m, 4] (post-mix only)
  const float* __restrict__ comb;       // [m, 4, 4], comb[i][o]: stream i feeds output o (post-mix only)
  const float* __restrict__ pre;        // [m, 4] the next sublayer's collapse, strided
  const bf16_t* __restrict__ weight;    // [5120] the next sublayer's RMSNorm weight
  bf16_t* __restrict__ residual_out;    // [m, 4, 5120] post-mixed streams (post-mix only)
  bf16_t* __restrict__ output;          // [m, 5120] RMSNorm(collapse(streams))
  int64_t pre_stride_m;
  int64_t pre_stride_k;
  int64_t num_rows;
  float eps;
};

/// \brief Collapse the HC=4 streams with `pre` and RMSNorm the result; with
///        kWithPost, post-mix `x` into the streams first and store them too.
///
/// The post-mix and collapse repeat `_mhc_post_combine_kernel` (or, without the
/// post-mix, `_mhc_combine_kernel`) element for element: FMA order, BF16 round of
/// each stream before the collapse, BF16 collapse. The RMSNorm repeats FlashInfer's
/// CuTe-DSL `RMSNormKernel` for H = 5120 BF16 (flashinfer-python 0.6.18): 64
/// threads per row, thread `t` owning columns `t * 8 + v * 512 + e`, a per-thread
/// fold in (v, e) order, a butterfly over offsets 1..16, the sum of the row's two
/// warp partials, IEEE division by H, `rsqrt.approx`, then `(y * rstd) * w`.
/// Squares of BF16 values are exact in FP32, so the fold's contraction is moot.
///
/// \tparam kWithPost   Whether to post-mix `x` into the streams first.
/// \tparam kRowsPerCta Rows per CTA; does not change any row's arithmetic.
/// \tparam kLoadGroup  Vector blocks (of 10 per thread) whose loads are issued together.
template <bool kWithPost, int kRowsPerCta, int kLoadGroup>
__global__
__launch_bounds__(64 * kRowsPerCta) void mhc_collapse_norm_kernel(const MhcCollapseNormParams __grid_constant__ p) {
  using namespace device;
  using V = AlignedVector<bf16x2_t, 4>;
  constexpr int kHidden = 5120;
  constexpr int kThreadsPerRow = 64;
  constexpr int kVecBlocks = kHidden / (8 * kThreadsPerRow);  // 10
  __shared__ float warp_sums[kRowsPerCta][2];

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int row_in_cta = tid / kThreadsPerRow;
  const int t = tid % kThreadsPerRow;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerCta + row_in_cta;
  const bool live = row < p.num_rows;

  float post[4] = {}, comb[4][4] = {}, pre[4] = {};
  if (live) {
#pragma unroll
    for (int o = 0; o < 4; ++o) {
      pre[o] = p.pre[row * p.pre_stride_m + o * p.pre_stride_k];
      if constexpr (kWithPost) {
        post[o] = p.post[row * 4 + o];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          comb[i][o] = p.comb[row * 16 + i * 4 + o];
      }
    }
  }

  // Loads of a group of vector blocks are issued together, before any store.
  const bf16_t* __restrict__ x_row = kWithPost ? p.x + row * kHidden : nullptr;
  const bf16_t* __restrict__ r_row = p.residual + row * 4 * kHidden;
  bf16_t* __restrict__ out_row = kWithPost ? p.residual_out + row * 4 * kHidden : nullptr;
  V collapsed[kVecBlocks];
  float sum = 0.f;
#pragma unroll
  for (int g = 0; g < kVecBlocks; g += kLoadGroup) {
    V x[kLoadGroup], r[kLoadGroup][4];
    if (live) {
#pragma unroll
      for (int j = 0; j < kLoadGroup; ++j) {
        const int64_t vid = t + (g + j) * kThreadsPerRow;  // 8-element vector index within the row
        if constexpr (kWithPost) x[j].load(x_row, vid);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          r[j][i].load(r_row + i * kHidden, vid);
      }
    }
#pragma unroll
    for (int j = 0; j < kLoadGroup; ++j) {
      const int v = g + j;
      const int64_t vid = t + v * kThreadsPerRow;
      float acc_y[8] = {};
      if (live) {
#pragma unroll
        for (int o = 0; o < 4; ++o) {
          V stream;
          if constexpr (kWithPost) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const auto xx = cast<fp32x2_t>(x[j][k]);
              const auto r0 = cast<fp32x2_t>(r[j][0][k]);
              const auto r1 = cast<fp32x2_t>(r[j][1][k]);
              const auto r2 = cast<fp32x2_t>(r[j][2][k]);
              const auto r3 = cast<fp32x2_t>(r[j][3][k]);
              float a = __fmaf_rn(post[o], xx.x, __fmul_rn(comb[0][o], r0.x));
              float b = __fmaf_rn(post[o], xx.y, __fmul_rn(comb[0][o], r0.y));
              a = __fmaf_rn(comb[1][o], r1.x, a);
              b = __fmaf_rn(comb[1][o], r1.y, b);
              a = __fmaf_rn(comb[2][o], r2.x, a);
              b = __fmaf_rn(comb[2][o], r2.y, b);
              a = __fmaf_rn(comb[3][o], r3.x, a);
              b = __fmaf_rn(comb[3][o], r3.y, b);
              stream[k] = cast<bf16x2_t>(fp32x2_t{a, b});
            }
            stream.store(out_row + o * kHidden, vid);
          } else {
            stream = r[j][o];
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const auto value = cast<fp32x2_t>(stream[k]);
            acc_y[2 * k] = __fmaf_rn(pre[o], value.x, acc_y[2 * k]);
            acc_y[2 * k + 1] = __fmaf_rn(pre[o], value.y, acc_y[2 * k + 1]);
          }
        }
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        collapsed[v][k] = cast<bf16x2_t>(fp32x2_t{acc_y[2 * k], acc_y[2 * k + 1]});
        const auto value = cast<fp32x2_t>(collapsed[v][k]);
        sum = __fmaf_rn(value.x, value.x, sum);
        sum = __fmaf_rn(value.y, value.y, sum);
      }
    }
  }

#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
  if (lane == 0) warp_sums[row_in_cta][t / 32] = sum;
  __syncthreads();
  if (!live) return;
  const float total = __fadd_rn(warp_sums[row_in_cta][0], warp_sums[row_in_cta][1]);
  float inv_rms;
  asm("rsqrt.approx.ftz.f32 %0, %1;"
      : "=f"(inv_rms)
      : "f"(__fadd_rn(__fdiv_rn(total, static_cast<float>(kHidden)), p.eps)));

#pragma unroll
  for (int v = 0; v < kVecBlocks; ++v) {
    const int64_t vid = t + v * kThreadsPerRow;
    V w, out;
    w.load(p.weight, vid);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto value = cast<fp32x2_t>(collapsed[v][k]);
      const auto weight = cast<fp32x2_t>(w[k]);
      // FlashInfer scales by (weight + weight_bias) with weight_bias = 0.
      out[k] = cast<bf16x2_t>(fp32x2_t{
          __fmul_rn(__fmul_rn(value.x, inv_rms), __fadd_rn(weight.x, 0.f)),
          __fmul_rn(__fmul_rn(value.y, inv_rms), __fadd_rn(weight.y, 0.f))});
    }
    out.store(p.output + row * kHidden, vid);
  }
}

template <int kRowsPerCta, int kLoadGroup>
struct MhcCollapseNorm {
  static void check_bf16_aligned(std::initializer_list<const void*> ptrs) {
    for (const auto* ptr : ptrs) {
      host::RuntimeCheck(reinterpret_cast<uintptr_t>(ptr) % 16 == 0, "BF16 operands must be 16-byte aligned");
    }
  }

  static void launch(const MhcCollapseNormParams& params, bool with_post, DLDevice device) {
    using namespace host;
    if (params.num_rows == 0) return;
    const auto grid = static_cast<uint32_t>((params.num_rows + kRowsPerCta - 1) / kRowsPerCta);
    const auto kernel = with_post ? mhc_collapse_norm_kernel<true, kRowsPerCta, kLoadGroup>
                                  : mhc_collapse_norm_kernel<false, kRowsPerCta, kLoadGroup>;
    LaunchKernel(dim3(grid), dim3(64 * kRowsPerCta), device)(kernel, params);
  }

  static void post_combine_norm(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView post,
      tvm::ffi::TensorView comb,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto pre_stride_m = SymbolicSize{"pre_stride_m"};
    auto pre_stride_k = SymbolicSize{"pre_stride_k"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(output);
    TensorMatcher({m, 4, 5120}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(post);
    TensorMatcher({m, 4}).with_strides({pre_stride_m, pre_stride_k}).with_dtype<float>().with_device(dev).verify(pre);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    TensorMatcher({5120}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    check_bf16_aligned(
        {x.data_ptr(), residual.data_ptr(), weight.data_ptr(), residual_out.data_ptr(), output.data_ptr()});
    const MhcCollapseNormParams params{
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()),
        static_cast<const float*>(pre.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<bf16_t*>(output.data_ptr()),
        static_cast<int64_t>(pre_stride_m.unwrap()),
        static_cast<int64_t>(pre_stride_k.unwrap()),
        static_cast<int64_t>(m.unwrap()),
        eps};
    launch(params, /*with_post=*/true, dev.unwrap());
  }

  static void combine_norm(
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto pre_stride_m = SymbolicSize{"pre_stride_m"};
    auto pre_stride_k = SymbolicSize{"pre_stride_k"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(dev).verify(output);
    TensorMatcher({m, 4, 5120}).with_dtype<bf16_t>().with_device(dev).verify(residual);
    TensorMatcher({m, 4}).with_strides({pre_stride_m, pre_stride_k}).with_dtype<float>().with_device(dev).verify(pre);
    TensorMatcher({5120}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    check_bf16_aligned({residual.data_ptr(), weight.data_ptr(), output.data_ptr()});
    const MhcCollapseNormParams params{
        nullptr,
        static_cast<const bf16_t*>(residual.data_ptr()),
        nullptr,
        nullptr,
        static_cast<const float*>(pre.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        nullptr,
        static_cast<bf16_t*>(output.data_ptr()),
        static_cast<int64_t>(pre_stride_m.unwrap()),
        static_cast<int64_t>(pre_stride_k.unwrap()),
        static_cast<int64_t>(m.unwrap()),
        eps};
    launch(params, /*with_post=*/false, dev.unwrap());
  }
};

}  // namespace sglang
