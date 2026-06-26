// Fused-transpose Ulysses all-to-all over NVLink P2P (CUDA IPC).
//
// This implements the head-scatter / sequence-gather collective used by
// Ulysses sequence parallelism with the layout permutation folded directly into
// cross-GPU write addresses. It reuses the custom all-reduce Signal barrier for
// inter-GPU synchronization.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "allreduce/custom_all_reduce.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace sglang {

class UlyssesA2A {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  RankSignals sg_;
  Signal* self_sg_;
  RankData out_ptrs_;
  void* local_out_buf_;

  UlyssesA2A(Signal** signals, void** out_bufs, int rank, int world_size, bool full_nvlink)
      : rank_(rank), world_size_(world_size), full_nvlink_(full_nvlink), self_sg_(signals[rank]) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
      out_ptrs_.ptrs[i] = out_bufs[i];
    }
    local_out_buf_ = out_bufs[rank];
  }
};

template <typename T, int NGPUS, int MODE>
__device__ __forceinline__ void
ulysses_a2a_move(const T* __restrict__ local_in, RankData out_ptrs, int rank, int B, int S_local, int H_local, int D) {
  static_assert(MODE == 0 || MODE == 1, "MODE must be 0 or 1");
  const int W = NGPUS;
  const int64_t H = static_cast<int64_t>(H_local) * W;
  const int64_t S_global = static_cast<int64_t>(S_local) * W;
  const int64_t block_len = static_cast<int64_t>(H_local) * D;
  const int64_t num_rows = static_cast<int64_t>(B) * W * S_local;

  using Vec = int4;
  constexpr int kVecBytes = sizeof(Vec);
  const int64_t row_bytes = block_len * static_cast<int64_t>(sizeof(T));
  const bool vec_ok = (row_bytes % kVecBytes) == 0 && (reinterpret_cast<uintptr_t>(local_in) % kVecBytes) == 0;

  const int64_t rows_per_block = (num_rows + gridDim.x - 1) / gridDim.x;
  const int64_t row_lo = static_cast<int64_t>(blockIdx.x) * rows_per_block;
  int64_t row_hi = row_lo + rows_per_block;
  if (row_hi > num_rows) row_hi = num_rows;
  if (row_lo >= row_hi) return;

  const int tid = threadIdx.x;
  const int nthr = blockDim.x;

  auto offsets = [&](int64_t row, int64_t& src_off, int64_t& dst_off) {
    const int64_t s = row % S_local;
    const int64_t tmp = row / S_local;
    const int64_t peer = tmp % W;
    const int64_t b = tmp / W;
    if constexpr (MODE == 0) {
      src_off = ((b * S_local + s) * H + peer * H_local) * D;
      dst_off = (b * S_global + static_cast<int64_t>(rank) * S_local + s) * block_len;
    } else {
      src_off = (b * S_global + peer * S_local + s) * block_len;
      dst_off = ((b * S_local + s) * H + static_cast<int64_t>(rank) * H_local) * D;
    }
    return peer;
  };

  if (vec_ok) {
    const int64_t units_per_row = row_bytes / kVecBytes;
    const int64_t total_units = (row_hi - row_lo) * units_per_row;
    for (int64_t u = tid; u < total_units; u += nthr) {
      const int64_t local_row = u / units_per_row;
      const int64_t unit = u - local_row * units_per_row;
      const int64_t row = row_lo + local_row;
      int64_t src_off, dst_off;
      const int64_t peer = offsets(row, src_off, dst_off);
      const Vec* s4 = reinterpret_cast<const Vec*>(local_in + src_off);
      Vec* d4 = reinterpret_cast<Vec*>((T*)out_ptrs.ptrs[peer] + dst_off);
      d4[unit] = s4[unit];
    }
  } else {
    for (int64_t row = row_lo; row < row_hi; ++row) {
      int64_t src_off, dst_off;
      const int64_t peer = offsets(row, src_off, dst_off);
      const T* s_ptr = local_in + src_off;
      T* d_ptr = (T*)out_ptrs.ptrs[peer] + dst_off;
      for (int64_t i = tid; i < block_len; i += nthr) {
        d_ptr[i] = s_ptr[i];
      }
    }
  }
}

template <typename T, int NGPUS>
__global__ void __launch_bounds__(kDefaultThreads, 1) ulysses_a2a_push_kernel(
    const T* __restrict__ local_in,
    RankData out_ptrs,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    int B,
    int S_local,
    int H_local,
    int D,
    int mode) {
  multi_gpu_barrier<NGPUS, true>(sg, self_sg, rank);
  if (mode == 0) {
    ulysses_a2a_move<T, NGPUS, 0>(local_in, out_ptrs, rank, B, S_local, H_local, D);
  } else {
    ulysses_a2a_move<T, NGPUS, 1>(local_in, out_ptrs, rank, B, S_local, H_local, D);
  }
  multi_gpu_barrier<NGPUS, false, true>(sg, self_sg, rank);
}

template <typename T, int NGPUS, int MODE>
__global__ void __launch_bounds__(kDefaultThreads, 1) ulysses_a2a_tk_style_kernel(
    const T* __restrict__ local_in,
    RankData out_ptrs,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    int B,
    int S_local,
    int H_local,
    int D) {
  multi_gpu_barrier<NGPUS, true>(sg, self_sg, rank);
  ulysses_a2a_move<T, NGPUS, MODE>(local_in, out_ptrs, rank, B, S_local, H_local, D);
  multi_gpu_barrier<NGPUS, false, true>(sg, self_sg, rank);
}

}  // namespace sglang

fptr_t init_ulysses_a2a(
    const std::vector<fptr_t>& out_ipc_ptrs,
    const std::vector<fptr_t>& signal_ipc_ptrs,
    int64_t rank,
    int64_t world_size,
    bool full_nvlink) {
  if (world_size > 8) throw std::invalid_argument("ulysses a2a world size > 8 is not supported");
  if (world_size != 2 && world_size != 4 && world_size != 6 && world_size != 8)
    throw std::invalid_argument("ulysses a2a only supports world size in (2, 4, 6, 8)");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");
  if (static_cast<int64_t>(out_ipc_ptrs.size()) != world_size)
    throw std::invalid_argument("out_ipc_ptrs size must equal world_size");
  if (static_cast<int64_t>(signal_ipc_ptrs.size()) != world_size)
    throw std::invalid_argument("signal_ipc_ptrs size must equal world_size");

  sglang::Signal* signals[8];
  void* out_bufs[8];
  for (int i = 0; i < world_size; i++) {
    signals[i] = reinterpret_cast<sglang::Signal*>(signal_ipc_ptrs[i]);
    out_bufs[i] = reinterpret_cast<void*>(out_ipc_ptrs[i]);
  }
  return (fptr_t) new sglang::UlyssesA2A(signals, out_bufs, rank, world_size, full_nvlink);
}

void dispose_ulysses_a2a(fptr_t _fa) {
  delete reinterpret_cast<sglang::UlyssesA2A*>(_fa);
}

void ulysses_a2a(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& out,
    int64_t B,
    int64_t S_local,
    int64_t H,
    int64_t D,
    int64_t mode) {
  auto fa = reinterpret_cast<sglang::UlyssesA2A*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "ulysses_a2a inputs must be CUDA tensors");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "ulysses_a2a inputs must be contiguous");
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(mode == 0 || mode == 1, "ulysses_a2a mode must be 0 or 1");

  const int W = fa->world_size_;
  TORCH_CHECK(H % W == 0, "global head count must be divisible by world size");
  const int H_local = static_cast<int>(H / W);
  const int64_t num_copy_blocks = B * S_local * W;
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(sglang::kMaxBlocks, num_copy_blocks)));
  const int threads = sglang::kDefaultThreads;
  const size_t out_bytes = out.numel() * out.element_size();

#define LAUNCH_ULYSSES_A2A(T, NG)                                         \
  sglang::ulysses_a2a_push_kernel<T, NG><<<blocks, threads, 0, stream>>>( \
      reinterpret_cast<const T*>(inp.data_ptr()),                         \
      fa->out_ptrs_,                                                      \
      fa->sg_,                                                            \
      fa->self_sg_,                                                       \
      fa->rank_,                                                          \
      static_cast<int>(B),                                                \
      static_cast<int>(S_local),                                          \
      H_local,                                                            \
      static_cast<int>(D),                                                \
      static_cast<int>(mode))

#define DISPATCH_NGPUS(T)                                                            \
  switch (W) {                                                                       \
    case 2:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 2);                                                      \
      break;                                                                         \
    case 4:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 4);                                                      \
      break;                                                                         \
    case 6:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 6);                                                      \
      break;                                                                         \
    case 8:                                                                          \
      LAUNCH_ULYSSES_A2A(T, 8);                                                      \
      break;                                                                         \
    default:                                                                         \
      throw std::runtime_error("ulysses_a2a only supports world size in (2,4,6,8)"); \
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      DISPATCH_NGPUS(float);
      break;
    }
    case at::ScalarType::Half: {
      DISPATCH_NGPUS(half);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      DISPATCH_NGPUS(nv_bfloat16);
      break;
    }
#endif
    default:
      throw std::runtime_error("ulysses_a2a only supports float32, float16 and bfloat16");
  }

#undef DISPATCH_NGPUS
#undef LAUNCH_ULYSSES_A2A

  AT_CUDA_CHECK(cudaGetLastError());
  AT_CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), fa->local_out_buf_, out_bytes, cudaMemcpyDeviceToDevice, stream));
}

void ulysses_a2a_tk(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& out,
    int64_t B,
    int64_t S_local,
    int64_t H,
    int64_t D,
    int64_t mode) {
  auto fa = reinterpret_cast<sglang::UlyssesA2A*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "ulysses_a2a_tk inputs must be CUDA tensors");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "ulysses_a2a_tk inputs must be contiguous");
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(mode == 0 || mode == 1, "ulysses_a2a_tk mode must be 0 or 1");

  const int W = fa->world_size_;
  TORCH_CHECK(H % W == 0, "global head count must be divisible by world size");
  const int H_local = static_cast<int>(H / W);
  const int64_t num_rows = B * static_cast<int64_t>(W) * S_local;
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(sglang::kMaxBlocks, num_rows)));
  const int threads = sglang::kDefaultThreads;
  const size_t out_bytes = out.numel() * out.element_size();

#define LAUNCH_ULYSSES_A2A_TK(T, NG, MD)                                          \
  sglang::ulysses_a2a_tk_style_kernel<T, NG, MD><<<blocks, threads, 0, stream>>>( \
      reinterpret_cast<const T*>(inp.data_ptr()),                                 \
      fa->out_ptrs_,                                                              \
      fa->sg_,                                                                    \
      fa->self_sg_,                                                               \
      fa->rank_,                                                                  \
      static_cast<int>(B),                                                        \
      static_cast<int>(S_local),                                                  \
      H_local,                                                                    \
      static_cast<int>(D))

#define DISPATCH_NGPUS_TK(T, MD)                                                        \
  switch (W) {                                                                          \
    case 2:                                                                             \
      LAUNCH_ULYSSES_A2A_TK(T, 2, MD);                                                  \
      break;                                                                            \
    case 4:                                                                             \
      LAUNCH_ULYSSES_A2A_TK(T, 4, MD);                                                  \
      break;                                                                            \
    case 6:                                                                             \
      LAUNCH_ULYSSES_A2A_TK(T, 6, MD);                                                  \
      break;                                                                            \
    case 8:                                                                             \
      LAUNCH_ULYSSES_A2A_TK(T, 8, MD);                                                  \
      break;                                                                            \
    default:                                                                            \
      throw std::runtime_error("ulysses_a2a_tk only supports world size in (2,4,6,8)"); \
  }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#define DISPATCH_BF16_TK(MD)            \
  case at::ScalarType::BFloat16: {      \
    DISPATCH_NGPUS_TK(nv_bfloat16, MD); \
    break;                              \
  }
#else
#define DISPATCH_BF16_TK(MD)
#endif

#define DISPATCH_DTYPE_TK(MD)                                                                 \
  switch (out.scalar_type()) {                                                                \
    case at::ScalarType::Float: {                                                             \
      DISPATCH_NGPUS_TK(float, MD);                                                           \
      break;                                                                                  \
    }                                                                                         \
    case at::ScalarType::Half: {                                                              \
      DISPATCH_NGPUS_TK(half, MD);                                                            \
      break;                                                                                  \
    }                                                                                         \
      DISPATCH_BF16_TK(MD)                                                                    \
    default:                                                                                  \
      throw std::runtime_error("ulysses_a2a_tk only supports float32, float16 and bfloat16"); \
  }

  if (mode == 0) {
    DISPATCH_DTYPE_TK(0);
  } else {
    DISPATCH_DTYPE_TK(1);
  }

#undef DISPATCH_DTYPE_TK
#undef DISPATCH_BF16_TK
#undef DISPATCH_NGPUS_TK
#undef LAUNCH_ULYSSES_A2A_TK

  AT_CUDA_CHECK(cudaGetLastError());
  AT_CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), fa->local_out_buf_, out_bytes, cudaMemcpyDeviceToDevice, stream));
}
