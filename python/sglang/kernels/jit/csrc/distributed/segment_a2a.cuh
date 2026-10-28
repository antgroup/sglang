// Intra-node all-to-all of fixed-size byte segments over NVLink, pushed by SM
// stores straight into every receiver's symmetric buffer at its final layout.
//
// Sender `r` owns `kWorldSize x rows` segments: segment (p, t) is read from
// `src + p * src_peer_stride + t * src_row_stride` and lands in receiver p's
// buffer at `dst[p] + r * dst_rank_stride + t * dst_row_stride`. The strides
// let one kernel express both directions of a head <-> row exchange, so the
// layout permute around a plain all-to-all is folded into the copy.
//
// Synchronization is the pull plane's per-block two-phase barrier: phase 0
// proves every receiver has started this kernel (its stream finished reading
// the buffer from the previous call), phase 1 publishes the pushed bytes. The
// result therefore stays valid in the receive buffer until the next call on the
// same plane, and the bytes are moved unmodified (no in-band markers).
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/object.h>

#include <algorithm>
#include <array>
#include <cstdint>

namespace sglang {

using device::distributed::Semaphore;
using host::distributed::CommunicatorRef;

inline constexpr uint32_t kSegmentA2AMaxThreads = 1024;

template <uint32_t kWorldSize>
struct SegmentA2AParams {
  const uint8_t* __restrict__ src;
  std::array<uint8_t*, kWorldSize> dst;
  std::array<Semaphore*, kWorldSize> semaphores;
  int64_t src_peer_stride;
  int64_t src_row_stride;
  int64_t dst_rank_stride;
  int64_t dst_row_stride;
  uint32_t seg_vecs;       // 16-byte vectors per segment
  uint32_t tiles_per_seg;  // warp tiles per segment
  uint32_t rows;           // segments per peer
  uint32_t valid_rows;     // segment (p, t) is sent as zeros once p * rows + t >= valid_rows
  uint32_t rank;
};

template <uint32_t kWorldSize, uint32_t kUnroll, bool kUsePDL>
__global__ __launch_bounds__(kSegmentA2AMaxThreads, 1) void segment_a2a_push_kernel(
    const __grid_constant__ SegmentA2AParams<kWorldSize> params) {
  using namespace device;
  using vec_t = AlignedVector<uint32_t, 4>;
  constexpr uint32_t kTileVecs = kWarpThreads * kUnroll;

  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>{params.semaphores.data(), params.rank, /*num_arrives=*/2};
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto lane = threadIdx.x % kWarpThreads;
  const auto warps_per_block = blockDim.x / kWarpThreads;
  // Block-fastest warp numbering spreads neighbouring tiles over the SMs.
  const auto warp = blockIdx.x + gridDim.x * (threadIdx.x / kWarpThreads);
  const auto num_warps = gridDim.x * warps_per_block;
  const auto num_tiles = kWorldSize * params.rows * params.tiles_per_seg;
  for (uint32_t tile = warp; tile < num_tiles; tile += num_warps) {
    // Peer-fastest: consecutive warps feed different links, and rank r starts
    // at peer r + 1, so at any moment the ranks target a permutation of peers.
    const auto peer = (params.rank + 1 + tile % kWorldSize) % kWorldSize;
    const auto rest = tile / kWorldSize;
    const auto t = rest / params.tiles_per_seg;
    const auto s = rest % params.tiles_per_seg;
    const bool valid = peer * params.rows + t < params.valid_rows;
    const auto src = params.src + peer * params.src_peer_stride + t * params.src_row_stride;
    const auto dst = params.dst[peer] + params.rank * params.dst_rank_stride + t * params.dst_row_stride;
    const auto base = s * kTileVecs + lane;
    vec_t vecs[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
      const auto vid = base + i * kWarpThreads;
      if (valid && vid < params.seg_vecs) {
        vecs[i].load(src, vid);
      } else {
        vecs[i].fill(0u);
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
      const auto vid = base + i * kWarpThreads;
      if (vid < params.seg_vecs) vecs[i].store(dst, vid);
    }
  }

  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
}

template <uint32_t kWorldSize, uint32_t kUnroll, bool kUsePDL>
struct SegmentA2A {
  /// \brief Push this rank's segments into every rank's pull workspace.
  /// \param comm Communicator whose pull plane holds the receive buffers and
  ///             the per-block semaphores (one per launched block).
  /// \param src Contiguous source bytes; any dtype, 16-byte aligned.
  /// \param dst_offset Byte offset of the result inside every workspace.
  /// \param num_blocks Grid size; must be the same on every rank for a call.
  static void
  run(const CommunicatorRef comm,
      const tvm::ffi::TensorView src,
      int64_t dst_offset,
      int64_t src_peer_stride,
      int64_t src_row_stride,
      int64_t dst_rank_stride,
      int64_t dst_row_stride,
      int64_t seg_bytes,
      int64_t rows,
      int64_t valid_rows,
      int64_t num_blocks,
      int64_t num_threads) {
    using namespace host;
    const auto& pull = comm->get_pull_obj();
    CHECK_HOST(pull.world_size == kWorldSize) << pull.world_size << " != " << kWorldSize;
    CHECK_HOST(src.device().device_type == kDLCUDA && src.IsContiguous());
    const auto src_bytes = src.numel() * static_cast<int64_t>((src.dtype().bits * src.dtype().lanes + 7) / 8);
    const auto src_ptr = static_cast<const uint8_t*>(src.data_ptr());
    for (const int64_t v : {seg_bytes, src_peer_stride, src_row_stride, dst_rank_stride, dst_row_stride, dst_offset}) {
      CHECK_HOST(v >= 0 && v % 16 == 0) << "segment a2a needs 16-byte aligned sizes and strides, got " << v;
    }
    CHECK_HOST(reinterpret_cast<uintptr_t>(src_ptr) % 16 == 0);
    CHECK_HOST(seg_bytes > 0 && rows > 0 && 0 <= valid_rows && valid_rows <= rows * kWorldSize);
    CHECK_HOST(1 <= num_blocks && num_blocks <= pull.num_blocks)
        << num_blocks << " blocks, plane has " << pull.num_blocks;
    CHECK_HOST(
        kWorldSize <= num_threads && num_threads <= kSegmentA2AMaxThreads && num_threads % device::kWarpThreads == 0);
    const auto dst_end = dst_offset + (kWorldSize - 1) * dst_rank_stride + (rows - 1) * dst_row_stride + seg_bytes;
    CHECK_HOST(dst_end <= pull.num_bytes)
        << "segments end at byte " << dst_end << " of a " << pull.num_bytes << "-byte workspace";
    if (valid_rows > 0) {
      // Strides are non-negative, so the furthest source byte is in the last
      // valid segment or the one closing the previous peer's run.
      const auto last = valid_rows - 1;
      auto src_end = (last / rows) * src_peer_stride + (last % rows) * src_row_stride;
      if (last >= rows) src_end = std::max(src_end, (last / rows - 1) * src_peer_stride + (rows - 1) * src_row_stride);
      CHECK_HOST(src_end + seg_bytes <= src_bytes) << "segments read past the " << src_bytes << "-byte source";
    }
    const auto seg_vecs = static_cast<uint32_t>(seg_bytes / 16);
    const auto tiles_per_seg = div_ceil(seg_vecs, device::kWarpThreads * kUnroll);
    CHECK_HOST(static_cast<int64_t>(kWorldSize) * rows * tiles_per_seg < (int64_t{1} << 31));

    auto params = SegmentA2AParams<kWorldSize>{
        .src = src_ptr,
        .dst = {},
        .semaphores = {},
        .src_peer_stride = src_peer_stride,
        .src_row_stride = src_row_stride,
        .dst_rank_stride = dst_rank_stride,
        .dst_row_stride = dst_row_stride,
        .seg_vecs = seg_vecs,
        .tiles_per_seg = tiles_per_seg,
        .rows = static_cast<uint32_t>(rows),
        .valid_rows = static_cast<uint32_t>(valid_rows),
        .rank = pull.rank,
    };
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.dst[i] = pull.workspaces[i] + dst_offset;
      params.semaphores[i] = pull.semaphores[i];
    }
    const auto kernel = segment_a2a_push_kernel<kWorldSize, kUnroll, kUsePDL>;
    LaunchKernel(static_cast<uint32_t>(num_blocks), static_cast<uint32_t>(num_threads), src.device())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
