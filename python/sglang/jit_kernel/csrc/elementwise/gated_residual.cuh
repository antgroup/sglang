#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For dtype_trait and fp16/bf16/fp32 conversions
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA dtype aliases
#include <sgl_kernel/vec.cuh>    // For 128-bit AlignedVector loads and stores

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int64_t kOptimizedHiddenSize = 5120;
constexpr int64_t kVectorBytes = 16;
constexpr uint32_t kBlockSize = 256;

// ---------------------------------------------------------------------------
// Vectorized kernel
//
// Each block owns one [D] token row. Inputs and output use eight 16-bit values
// per 128-bit transaction. Gate uses two groups of four FP32 values, also via
// 128-bit transactions. kHiddenSize=5120 makes the LingBot production shape a
// compile-time constant; kHiddenSize=0 is the generic aligned-width path.
// ---------------------------------------------------------------------------
template <typename T, int64_t kHiddenSize>
__global__ void gated_residual_vector_kernel(
    T* __restrict__ dst,
    const T* __restrict__ residual,
    const T* __restrict__ x,
    const fp32_t* __restrict__ gate,
    int64_t sequence_length,
    int64_t num_frames,
    int64_t tokens_per_frame,
    int64_t runtime_hidden_size) {
  static_assert(sizeof(T) == 2, "gated_residual only supports 16-bit input types");
  constexpr int64_t kInputVectorElements = kVectorBytes / sizeof(T);
  constexpr int64_t kGateVectorElements = kVectorBytes / sizeof(fp32_t);
  static_assert(kInputVectorElements == 2 * kGateVectorElements);

  using input_vec_t = device::AlignedVector<T, kInputVectorElements>;
  using gate_vec_t = device::AlignedVector<fp32_t, kGateVectorElements>;

  const int64_t hidden_size = kHiddenSize == 0 ? runtime_hidden_size : kHiddenSize;
  const int64_t input_vectors_per_row = hidden_size / kInputVectorElements;
  const int64_t gate_vectors_per_row = hidden_size / kGateVectorElements;
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = token / sequence_length;
  const int64_t sequence_index = token - batch * sequence_length;
  const int64_t frame = sequence_index / tokens_per_frame;
  const int64_t gate_row = batch * num_frames + frame;

  for (int64_t vector_index = threadIdx.x; vector_index < input_vectors_per_row; vector_index += blockDim.x) {
    const int64_t input_offset = token * input_vectors_per_row + vector_index;
    const int64_t gate_offset = gate_row * gate_vectors_per_row + 2 * vector_index;

    input_vec_t residual_values;
    input_vec_t x_values;
    input_vec_t output_values;
    gate_vec_t gate_low;
    gate_vec_t gate_high;
    residual_values.load(residual, input_offset);
    x_values.load(x, input_offset);
    gate_low.load(gate, gate_offset);
    gate_high.load(gate, gate_offset + 1);

#pragma unroll
    for (int i = 0; i < kInputVectorElements; ++i) {
      const fp32_t residual_value = device::cast<fp32_t, T>(residual_values[i]);
      const fp32_t x_value = device::cast<fp32_t, T>(x_values[i]);
      const fp32_t gate_value = i < kGateVectorElements ? gate_low[i] : gate_high[i - kGateVectorElements];
      output_values[i] = device::cast<T, fp32_t>(residual_value + x_value * gate_value);
    }
    output_values.store(dst, input_offset);
  }
}

// ---------------------------------------------------------------------------
// Scalar fallback
//
// Rows whose hidden size is not 128-bit aligned use this path. It preserves the
// same FP32 arithmetic semantics while keeping the LingBot D=5120 fast path
// entirely vectorized.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gated_residual_scalar_kernel(
    T* __restrict__ dst,
    const T* __restrict__ residual,
    const T* __restrict__ x,
    const fp32_t* __restrict__ gate,
    int64_t sequence_length,
    int64_t num_frames,
    int64_t tokens_per_frame,
    int64_t hidden_size) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = token / sequence_length;
  const int64_t sequence_index = token - batch * sequence_length;
  const int64_t frame = sequence_index / tokens_per_frame;
  const int64_t gate_row = batch * num_frames + frame;
  const int64_t row_offset = token * hidden_size;
  const int64_t gate_offset = gate_row * hidden_size;

  for (int64_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    const fp32_t residual_value = device::cast<fp32_t, T>(residual[row_offset + d]);
    const fp32_t x_value = device::cast<fp32_t, T>(x[row_offset + d]);
    dst[row_offset + d] = device::cast<T, fp32_t>(residual_value + x_value * gate[gate_offset + d]);
  }
}

// ---------------------------------------------------------------------------
// TVM-FFI launcher
//
// TensorMatcher validates every tensor's shape, contiguous layout, dtype, and
// common CUDA device. LaunchKernel resolves the active stream and checks launch
// failures. RuntimeCheck handles the semantic constraint S % F == 0.
// ---------------------------------------------------------------------------
template <typename T>
struct GatedResidualKernel {
  static void
  run(tvm::ffi::TensorView dst, tvm::ffi::TensorView residual, tvm::ffi::TensorView x, tvm::ffi::TensorView gate) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"sequence_length"};
    auto D = SymbolicSize{"hidden_size"};
    auto F = SymbolicSize{"num_frames"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, S, D})  // dst, residual, x: contiguous [B, S, D]
        .with_dtype<T>()
        .with_device(device)
        .verify(dst)
        .verify(residual)
        .verify(x);
    TensorMatcher({B, F, 1, D})  // gate: contiguous FP32 [B, F, 1, D]
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(gate);

    const int64_t batch_size = B.unwrap();
    const int64_t sequence_length = S.unwrap();
    const int64_t hidden_size = D.unwrap();
    const int64_t num_frames = F.unwrap();
    RuntimeCheck(batch_size > 0, "gated_residual: batch size must be positive, got ", batch_size);
    RuntimeCheck(sequence_length > 0, "gated_residual: sequence length must be positive, got ", sequence_length);
    RuntimeCheck(hidden_size > 0, "gated_residual: hidden size must be positive, got ", hidden_size);
    RuntimeCheck(num_frames > 0, "gated_residual: num_frames must be positive, got ", num_frames);
    RuntimeCheck(
        sequence_length % num_frames == 0,
        "gated_residual: sequence length ",
        sequence_length,
        " must be divisible by num_frames ",
        num_frames);
    RuntimeCheck(
        batch_size <= std::numeric_limits<int32_t>::max() / sequence_length,
        "gated_residual: B*S exceeds the CUDA 1D grid limit");

    const int64_t num_tokens = batch_size * sequence_length;
    const int64_t tokens_per_frame = sequence_length / num_frames;
    const auto grid = static_cast<uint32_t>(num_tokens);
    const DLDevice launch_device = device.unwrap();
    auto* dst_ptr = static_cast<T*>(dst.data_ptr());
    const auto* residual_ptr = static_cast<const T*>(residual.data_ptr());
    const auto* x_ptr = static_cast<const T*>(x.data_ptr());
    const auto* gate_ptr = static_cast<const fp32_t*>(gate.data_ptr());
    constexpr std::uintptr_t kAlignmentMask = kVectorBytes - 1;
    const bool vector_aligned =
        ((reinterpret_cast<std::uintptr_t>(dst_ptr) | reinterpret_cast<std::uintptr_t>(residual_ptr) |
          reinterpret_cast<std::uintptr_t>(x_ptr) | reinterpret_cast<std::uintptr_t>(gate_ptr)) &
         kAlignmentMask) == 0;

    if (vector_aligned && hidden_size == kOptimizedHiddenSize) {
      LaunchKernel(grid, kBlockSize, launch_device)(
          gated_residual_vector_kernel<T, kOptimizedHiddenSize>,
          dst_ptr,
          residual_ptr,
          x_ptr,
          gate_ptr,
          sequence_length,
          num_frames,
          tokens_per_frame,
          hidden_size);
    } else if (vector_aligned && hidden_size % (kVectorBytes / sizeof(T)) == 0) {
      LaunchKernel(grid, kBlockSize, launch_device)(
          gated_residual_vector_kernel<T, 0>,
          dst_ptr,
          residual_ptr,
          x_ptr,
          gate_ptr,
          sequence_length,
          num_frames,
          tokens_per_frame,
          hidden_size);
    } else {
      LaunchKernel(grid, kBlockSize, launch_device)(
          gated_residual_scalar_kernel<T>,
          dst_ptr,
          residual_ptr,
          x_ptr,
          gate_ptr,
          sequence_length,
          num_frames,
          tokens_per_frame,
          hidden_size);
    }
  }
};

}  // namespace
