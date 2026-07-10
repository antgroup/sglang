#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic tensor metadata
#include <sgl_kernel/utils.h>   // For RuntimeCheck and div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct GatherRingKVParams {
  const void* __restrict__ k_cache;
  const void* __restrict__ v_cache;
  void* __restrict__ k_out;
  void* __restrict__ v_out;
  int64_t cache_batch_stride_bytes;
  int64_t out_batch_stride_bytes;
  int32_t batch_size;
  int32_t output_tokens;
  int32_t sink_tokens;
  int32_t tail_capacity;
  int32_t tail_start;
  int32_t first_start;
  int32_t first_length;
  int32_t second_start;
};

constexpr int32_t kWarpThreads = 32;
constexpr int32_t kWarpsPerBlock = 8;
constexpr int32_t kThreadsPerBlock = kWarpThreads * kWarpsPerBlock;

template <int64_t kRowBytes>
__global__ void gather_ring_kv_kernel(const __grid_constant__ GatherRingKVParams params) {
  using CopyVector = device::AlignedVector<uint32_t, 4>;
  static_assert(sizeof(CopyVector) == 16);
  static_assert(kRowBytes % sizeof(CopyVector) == 0);

  constexpr int32_t kVectorsPerRow = kRowBytes / sizeof(CopyVector);
  const int32_t flat_warp_id =
      static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + static_cast<int32_t>(threadIdx.x) / kWarpThreads;
  const int32_t batch_idx = flat_warp_id / params.output_tokens;
  const int32_t output_idx = flat_warp_id % params.output_tokens;
  if (batch_idx >= params.batch_size) return;

  const int32_t logical_idx = output_idx < params.first_length ? params.first_start + output_idx
                                                               : params.second_start + output_idx - params.first_length;
  const int32_t physical_idx =
      logical_idx < params.sink_tokens
          ? logical_idx
          : params.sink_tokens + (params.tail_start + logical_idx - params.sink_tokens) % params.tail_capacity;

  const auto* k_src = reinterpret_cast<const CopyVector*>(
      static_cast<const uint8_t*>(params.k_cache) + batch_idx * params.cache_batch_stride_bytes +
      static_cast<int64_t>(physical_idx) * kRowBytes);
  const auto* v_src = reinterpret_cast<const CopyVector*>(
      static_cast<const uint8_t*>(params.v_cache) + batch_idx * params.cache_batch_stride_bytes +
      static_cast<int64_t>(physical_idx) * kRowBytes);
  auto* k_dst = reinterpret_cast<CopyVector*>(
      static_cast<uint8_t*>(params.k_out) + batch_idx * params.out_batch_stride_bytes +
      static_cast<int64_t>(output_idx) * kRowBytes);
  auto* v_dst = reinterpret_cast<CopyVector*>(
      static_cast<uint8_t*>(params.v_out) + batch_idx * params.out_batch_stride_bytes +
      static_cast<int64_t>(output_idx) * kRowBytes);

  const int32_t lane_id = static_cast<int32_t>(threadIdx.x) % kWarpThreads;
#pragma unroll
  for (int32_t vector_idx = lane_id; vector_idx < kVectorsPerRow; vector_idx += kWarpThreads) {
    k_dst[vector_idx] = k_src[vector_idx];
    v_dst[vector_idx] = v_src[vector_idx];
  }
}

template <int64_t kRowBytes>
struct GatherRingKVKernel {
  static_assert(kRowBytes > 0 && kRowBytes % 16 == 0);

  static void
  run(const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView k_out,
      const tvm::ffi::TensorView v_out,
      const int64_t sink_tokens,
      const int64_t tail_start,
      const int64_t first_start,
      const int64_t first_length,
      const int64_t second_start,
      const int64_t second_length) {
    using namespace host;

    auto batch_size = SymbolicSize{"batch_size"};
    auto cache_tokens = SymbolicSize{"cache_tokens"};
    auto output_tokens = SymbolicSize{"output_tokens"};
    auto row_elements = SymbolicSize{"row_elements"};
    auto cache_batch_stride = SymbolicSize{"cache_batch_stride"};
    auto out_batch_stride = SymbolicSize{"out_batch_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({batch_size, cache_tokens, row_elements})  //
        .with_strides({cache_batch_stride, row_elements, 1})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({batch_size, output_tokens, row_elements})  //
        .with_strides({out_batch_stride, row_elements, 1})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(k_out)
        .verify(v_out);

    const int64_t batch = batch_size.unwrap();
    const int64_t cache_size = cache_tokens.unwrap();
    const int64_t output_size = output_tokens.unwrap();
    const int64_t tail_capacity = cache_size - sink_tokens;
    RuntimeCheck(kRowBytes == dtype_bytes(dtype.unwrap()) * row_elements.unwrap(), "Unexpected ring KV row size");
    RuntimeCheck(batch > 0 && output_size > 0, "Ring KV gather requires non-empty tensors");
    RuntimeCheck(sink_tokens >= 0 && sink_tokens <= cache_size, "Invalid sink token count");
    RuntimeCheck(tail_capacity > 0, "Ring KV gather requires positive tail capacity");
    RuntimeCheck(tail_start >= 0 && tail_start < tail_capacity, "Invalid ring tail start");
    RuntimeCheck(first_length >= 0 && second_length >= 0, "Ring KV gather lengths must be non-negative");
    RuntimeCheck(first_length + second_length == output_size, "Ring KV gather lengths must match output size");
    RuntimeCheck(first_start >= 0 && first_start + first_length <= cache_size, "Invalid first logical range");
    RuntimeCheck(second_start >= 0 && second_start + second_length <= cache_size, "Invalid second logical range");
    RuntimeCheck(
        batch <= INT32_MAX && output_size <= INT32_MAX && cache_size <= INT32_MAX, "Ring KV tensor is too large");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(v_cache.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(k_out.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(v_out.data_ptr()) % 16 == 0,
        "Ring KV tensors must be 16-byte aligned");
    RuntimeCheck(
        cache_batch_stride.unwrap() * dtype_bytes(dtype.unwrap()) % 16 == 0 &&
            out_batch_stride.unwrap() * dtype_bytes(dtype.unwrap()) % 16 == 0,
        "Ring KV batch strides must be 16-byte aligned");

    const auto params = GatherRingKVParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .k_out = k_out.data_ptr(),
        .v_out = v_out.data_ptr(),
        .cache_batch_stride_bytes = cache_batch_stride.unwrap() * dtype_bytes(dtype.unwrap()),
        .out_batch_stride_bytes = out_batch_stride.unwrap() * dtype_bytes(dtype.unwrap()),
        .batch_size = static_cast<int32_t>(batch),
        .output_tokens = static_cast<int32_t>(output_size),
        .sink_tokens = static_cast<int32_t>(sink_tokens),
        .tail_capacity = static_cast<int32_t>(tail_capacity),
        .tail_start = static_cast<int32_t>(tail_start),
        .first_start = static_cast<int32_t>(first_start),
        .first_length = static_cast<int32_t>(first_length),
        .second_start = static_cast<int32_t>(second_start),
    };
    const int64_t num_warps = batch * output_size;
    const int64_t num_blocks = div_ceil(num_warps, static_cast<int64_t>(kWarpsPerBlock));
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())(gather_ring_kv_kernel<kRowBytes>, params);
  }
};

}  // namespace
