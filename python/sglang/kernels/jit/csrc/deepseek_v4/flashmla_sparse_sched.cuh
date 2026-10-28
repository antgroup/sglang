// Adapted from
// https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
// Adds parallel overhead candidates, cost scoring, and a partition-parallel scheduler for SM90.
// Copyright (c) 2025 DeepSeek
// SPDX-License-Identifier: MIT
// See ../../THIRDPARTYNOTICES.txt for the full license.

// Tile-scheduler metadata for the FlashMLA SM90 sparse decode kernel
// (smxx::decode::get_mla_metadata_kernel replacement, s_q == 1).
//
// FlashMLA splits the requests' 64-token blocks greedily over the SM
// partitions, charging fixed_overhead_num_blocks = 5 per split piece. Here
// one warp leader per candidate overhead runs the same greedy, scores the
// schedule with a linear cost model (blocks, pieces, split pieces, combine)
// and the best schedule is written. A single candidate of 5 reproduces the
// stock schedule exactly.
//
// Output layout is FlashMLA's DecodingSchedMeta (csrc/params.h):
// begin_req_idx, end_req_idx, begin_block_idx, end_block_idx,
// begin_split_idx, is_first_req_splitted, is_last_req_splitted, pad.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

inline constexpr int32_t kSchedBlock = 64;    // FlashMLA TOPK_BLOCK_SIZE
inline constexpr int32_t kSchedMaxCand = 4;   // one warp per candidate
inline constexpr int32_t kSchedMetaInts = 8;  // sizeof(DecodingSchedMeta) / 4

struct FlashMLASchedParams {
  const int32_t* __restrict__ topk_length;        // [b]
  const int32_t* __restrict__ extra_topk_length;  // [b] or nullptr
  int32_t* __restrict__ meta;                     // [num_parts, 8]
  int32_t* __restrict__ num_splits;               // [b + 1]
  int32_t b;
  int32_t topk;
  int32_t extra_topk;  // 0: no extra KV cache
  int32_t num_parts;
  int32_t num_cand;
  int32_t overheads[kSchedMaxCand];
  // Cost model in arbitrary units: per block, per piece, per split piece,
  // once if any request is split (the combine kernel then does work).
  int32_t cost_block;
  int32_t cost_piece;
  int32_t cost_split;
  int32_t cost_combine;
  // flashmla_sparse_sched_par_kernel: shared ints for the cost-space tables.
  int32_t table_ints;
};

SGL_DEVICE int32_t sched_num_blocks(const FlashMLASchedParams& p, int32_t i) {
  // Mirrors get_mla_metadata_kernel's cur_s_k and the decode kernel's
  // total_topk_padded / TOPK_BLOCK_SIZE.
  int32_t s = p.topk_length[i];
  if (s == 0) s = 1;
  if (p.extra_topk) {
    s = (s + kSchedBlock - 1) / kSchedBlock * kSchedBlock;
    s += p.extra_topk_length != nullptr ? p.extra_topk_length[i] : p.extra_topk;
  }
  const int32_t last = s - 1 > 0 ? s - 1 : 0;
  return last / kSchedBlock + 1;
}

SGL_DEVICE int32_t warp_sum(int32_t v) {
#pragma unroll
  for (int32_t o = 16; o >= 1; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

/// \brief FlashMLA's greedy split for overhead `ov` on one thread (a
///        lane-parallel scan is slower: every partition becomes a chain of
///        dependent shuffles). The schedule goes to shared-memory staging.
/// \return modeled makespan, or INT32_MAX if the greedy does not place
///         every block (never for the stock overhead).
SGL_DEVICE int32_t sched_greedy(
    const FlashMLASchedParams& p, const int32_t* nb, int32_t total_nb, int32_t ov, int4* meta, int32_t* splits) {
  const int32_t b = p.b;
  const int32_t payload = (total_nb + b * ov + p.num_parts - 1) / p.num_parts + ov;
  int32_t req = 0, blk = 0, nsplit = 0, cum = 0, worst = 0;
  bool any_split = false;
  splits[0] = 0;
  for (int32_t part = 0; part < p.num_parts; ++part) {
    const int32_t begin_req = req, begin_blk = blk, begin_split = nsplit;
    int32_t rem = payload, cost = 0, cur_nb = req < b ? nb[req] : 0;
    while (req < b) {
      const int32_t left = cur_nb - blk;
      if (rem >= left + ov) {
        cost += p.cost_block * left + p.cost_piece + (blk != 0 ? p.cost_split : 0);
        cum += nsplit + 1;
        splits[req + 1] = cum;
        rem -= left + ov;
        ++req;
        blk = 0;
        nsplit = 0;
        cur_nb = req < b ? nb[req] : 0;
      } else {
        if (rem - ov > 0) {
          blk += rem - ov;
          ++nsplit;
          any_split = true;
          cost += p.cost_block * (rem - ov) + p.cost_piece + p.cost_split;
        }
        break;
      }
    }
    worst = cost > worst ? cost : worst;
    int32_t end_req, end_blk, last_split;
    if (blk > 0) {
      end_req = req;
      end_blk = blk;
      last_split = blk != cur_nb;
    } else {
      end_req = req - 1;
      end_blk = req > 0 ? nb[req - 1] : 0;
      last_split = 0;
    }
    int32_t first_split = begin_blk != 0;
    if (begin_req == end_req) first_split = last_split = first_split || last_split;
    meta[2 * part] = make_int4(begin_req, end_req, begin_blk, end_blk);
    meta[2 * part + 1] = make_int4(begin_split, first_split, last_split, 0);
  }
  if (req != b || blk != 0 || nsplit != 0) return INT32_MAX;
  return worst + (any_split ? p.cost_combine : 0);
}

/// Dynamic shared memory: nb [b], then per candidate a meta staging
/// [num_parts * 2] int4 and a num_splits staging [b + 1] int32.
SGL_DEVICE size_t sched_cand_stride_ints(int32_t b, int32_t num_parts) {
  return (static_cast<size_t>(num_parts) * kSchedMetaInts + b + 1 + 3) / 4 * 4;
}

__global__ void flashmla_sparse_sched_kernel(const FlashMLASchedParams __grid_constant__ p) {
  extern __shared__ int4 smem[];
  __shared__ int32_t s_total;
  __shared__ int32_t s_cost[kSchedMaxCand];
  const int32_t b = p.b;
  int32_t* nb = reinterpret_cast<int32_t*>(smem);
  int32_t* stage = nb + (b + 3) / 4 * 4;
  const size_t stride = sched_cand_stride_ints(b, p.num_parts);
  const int32_t warp = threadIdx.x / 32;
  const bool leader = threadIdx.x % 32 == 0;
  if (threadIdx.x == 0) s_total = 0;
  __syncthreads();
  int32_t part_sum = 0;
  for (int32_t i = threadIdx.x; i < b; i += blockDim.x) {
    nb[i] = sched_num_blocks(p, i);
    part_sum += nb[i];
  }
  part_sum = warp_sum(part_sum);
  if (leader) atomicAdd(&s_total, part_sum);
  __syncthreads();
  // One candidate per warp leader: independent warps run concurrently.
  if (leader && warp < p.num_cand) {
    int32_t* mine = stage + warp * stride;
    s_cost[warp] = sched_greedy(
        p, nb, s_total, p.overheads[warp], reinterpret_cast<int4*>(mine), mine + p.num_parts * kSchedMetaInts);
  }
  __syncthreads();
  int32_t best = 0;
  for (int32_t c = 1; c < p.num_cand; ++c)
    if (s_cost[c] < s_cost[best]) best = c;
  const int32_t* src = stage + best * stride;
  const int4* src_meta = reinterpret_cast<const int4*>(src);
  int4* dst_meta = reinterpret_cast<int4*>(p.meta);
  for (int32_t i = threadIdx.x; i < p.num_parts * 2; i += blockDim.x)
    dst_meta[i] = src_meta[i];
  const int32_t* src_splits = src + p.num_parts * kSchedMetaInts;
  for (int32_t i = threadIdx.x; i <= b; i += blockDim.x)
    p.num_splits[i] = src_splits[i];
}

// ---------------------------------------------------------------------------
// Same schedules as flashmla_sparse_sched_kernel, bit for bit, with the greedy
// walked per partition instead of per request: a part starting at block
// position pos of request r takes requests r..k-1 whole, k the largest with
// G[k] <= pos + ov * r + payload where G[k] = S[k] + ov * k (S = request block
// starts), then rem - ov blocks of request k if positive. When the tables fit
// in shared memory a step is one lookup of the precomputed next part start;
// else a warp per candidate finds k with a ballot over 32 requests. Costs,
// metadata and num_splits are then computed per part / per request in parallel.
// ---------------------------------------------------------------------------

inline constexpr int32_t kSchedParThreads = 512;

SGL_DEVICE int32_t sched_par_last_le(const int32_t* P, int32_t n, int32_t x) {
  // Largest q in [0, n) with P[q] <= x; P[0] == 0 <= x and P is non-decreasing.
  int32_t lo = 0, hi = n - 1;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) >> 1;
    if (P[mid] <= x)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

/// Dynamic shared memory (int32): S [b + 1], then per candidate the part start
/// positions P [num_parts + 1] and their requests R [num_parts + 1], then the
/// mid-request boundary prefix count M [num_parts + 1], then table_ints ints for
/// the cost-space tables, the request of every block position and the next
/// partition start of every position per candidate.
__global__ void __launch_bounds__(kSchedParThreads)
    flashmla_sparse_sched_par_kernel(const FlashMLASchedParams __grid_constant__ p) {
  extern __shared__ int32_t smem_par[];
  __shared__ int32_t s_warp_sum[kSchedParThreads / 32];
  __shared__ int32_t s_cost[kSchedMaxCand];
  __shared__ int32_t s_any[kSchedMaxCand];
  const int32_t b = p.b, parts = p.num_parts;
  const int32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int32_t* S = smem_par;
  int32_t* PR = S + b + 1;
  const int32_t pr_stride = 2 * (parts + 1);
  int32_t* M = PR + kSchedMaxCand * pr_stride;

  if (threadIdx.x < kSchedMaxCand) {
    s_cost[threadIdx.x] = 0;
    s_any[threadIdx.x] = 0;
  }
  // Exclusive scan of the per-request block counts into S.
  if (threadIdx.x == 0) S[0] = 0;
  int32_t carry = 0;
  for (int32_t base = 0; base < b; base += kSchedParThreads) {
    const int32_t i = base + threadIdx.x;
    const int32_t v = i < b ? sched_num_blocks(p, i) : 0;
    int32_t inc = v;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1) {
      const int32_t t = __shfl_up_sync(0xffffffffu, inc, o);
      if (lane >= o) inc += t;
    }
    if (lane == 31) s_warp_sum[warp] = inc;
    __syncthreads();
    int32_t warp_off = 0, chunk = 0;
#pragma unroll
    for (int32_t w = 0; w < kSchedParThreads / 32; ++w) {
      warp_off += w < warp ? s_warp_sum[w] : 0;
      chunk += s_warp_sum[w];
    }
    if (i < b) S[i + 1] = carry + warp_off + inc;
    carry += chunk;
    __syncthreads();
  }
  const int32_t total = S[b];

  // T[x] = (k << 20) | S[k] for the largest request k with G[k] <= x.
  int32_t payloads[kSchedMaxCand];
  int32_t table_off[kSchedMaxCand + 1];
  table_off[0] = 0;
  for (int32_t c = 0; c < p.num_cand; ++c) {
    const int32_t ov = p.overheads[c];
    payloads[c] = (total + b * ov + parts - 1) / parts + ov;
    table_off[c + 1] = table_off[c] + total + b * ov + payloads[c] + 1;
  }
  const int32_t npos = total + 1;
  int32_t* T = M + parts + 1;
  int32_t* row_of = T + table_off[p.num_cand];
  int32_t* next = row_of + npos;
  const bool use_table = total < (1 << 20) && table_off[p.num_cand] + (1 + p.num_cand) * npos <= p.table_ints;
  if (use_table) {
    for (int32_t t = threadIdx.x; t < p.num_cand * (b + 1); t += kSchedParThreads) {
      const int32_t c = t / (b + 1), k = t % (b + 1);
      const int32_t ov = p.overheads[c];
      const int32_t lo = S[k] + ov * k;
      const int32_t hi = k < b ? S[k + 1] + ov * (k + 1) : table_off[c + 1] - table_off[c];
      const int32_t v = (k << 20) | S[k];
      for (int32_t x = lo; x < hi; ++x)
        T[table_off[c] + x] = v;
    }
    for (int32_t k = threadIdx.x; k <= b; k += kSchedParThreads) {
      const int32_t hi = k < b ? S[k + 1] : npos;
      for (int32_t x = S[k]; x < hi; ++x)
        row_of[x] = k;
    }
    __syncthreads();
    for (int32_t c = 0; c < p.num_cand; ++c) {
      const int32_t ov = p.overheads[c], payload = payloads[c];
      for (int32_t pos = threadIdx.x; pos < npos; pos += kSchedParThreads) {
        const int32_t r = row_of[pos];
        int32_t nxt = pos;
        if (r < b) {
          const int32_t lim = pos + ov * r + payload;
          const int32_t v = T[table_off[c] + lim];
          const int32_t k = v >> 20, sk = v & ((1 << 20) - 1);
          const int32_t rem = lim - (sk + ov * k);
          nxt = (k < b && rem - ov > 0) ? sk + rem - ov : sk;
        }
        next[c * npos + pos] = nxt;
      }
    }
    __syncthreads();
  }
  if (warp < p.num_cand) {
    int32_t* P = PR + warp * pr_stride;
    int32_t* R = P + parts + 1;
    if (use_table) {
      if (lane == 0) {
        const int32_t* nx = next + warp * npos;
        int32_t pos = 0;
        for (int32_t part = 0; part < parts; ++part) {
          P[part] = pos;
          R[part] = row_of[pos];
          pos = nx[pos];
        }
        P[parts] = pos;
        R[parts] = row_of[pos];
      }
    } else {
      const int32_t ov = p.overheads[warp], payload = payloads[warp];
      int32_t pos = 0, r = 0;
      for (int32_t part = 0; part < parts; ++part) {
        if (lane == 0) {
          P[part] = pos;
          R[part] = r;
        }
        if (r < b) {
          const int32_t lim = pos + ov * r + payload;
          int32_t k = r, sk = S[r];
          while (true) {
            const int32_t idx = k + 1 + lane;
            const int32_t sv = idx <= b ? S[idx] : 0;
            const bool fits = idx <= b && sv + ov * idx <= lim;
            const int32_t cnt = __popc(__ballot_sync(0xffffffffu, fits));
            if (cnt > 0) sk = __shfl_sync(0xffffffffu, sv, cnt - 1);
            k += cnt;
            if (cnt < 32) break;
          }
          const int32_t rem = lim - (sk + ov * k);
          pos = (k < b && rem - ov > 0) ? sk + rem - ov : sk;
          r = k;
        }
      }
      if (lane == 0) {
        P[parts] = pos;
        R[parts] = r;
      }
    }
  }
  __syncthreads();

  // Cost of every (candidate, part), as sched_greedy charges it.
  for (int32_t t = threadIdx.x; t < p.num_cand * parts; t += kSchedParThreads) {
    const int32_t c = t / parts, q = t % parts;
    const int32_t* P = PR + c * pr_stride;
    const int32_t* R = P + parts + 1;
    const int32_t s = P[q], e = P[q + 1];
    if (e <= s) continue;
    const int32_t br = R[q], er = e == S[R[q + 1]] ? R[q + 1] - 1 : R[q + 1];
    const int32_t first = s != S[br], last = e != S[er + 1];
    const int32_t split_pieces = br == er ? (first | last) : first + last;
    const int32_t cost = p.cost_block * (e - s) + p.cost_piece * (er - br + 1) + p.cost_split * split_pieces;
    atomicMax(&s_cost[c], cost);
    if (last) s_any[c] = 1;
  }
  __syncthreads();
  int32_t best = 0, best_cost = 0;
  for (int32_t c = 0; c < p.num_cand; ++c) {
    const int32_t* P = PR + c * pr_stride;
    const int32_t cost = P[parts] != total ? INT32_MAX : s_cost[c] + (s_any[c] ? p.cost_combine : 0);
    if (c == 0 || cost < best_cost) {
      best = c;
      best_cost = cost;
    }
  }
  const int32_t* P = PR + best * pr_stride;
  const int32_t* R = P + parts + 1;

  // M[q]: partition boundaries 1..q that fall inside a request.
  if (warp == 0) {
    int32_t run = 0;
    for (int32_t base = 0; base <= parts; base += 32) {
      const int32_t q = base + lane;
      const int32_t mid = (q >= 1 && q <= parts && P[q] < total && P[q] != S[R[q]]) ? 1 : 0;
      int32_t inc = mid;
#pragma unroll
      for (int32_t o = 1; o < 32; o <<= 1) {
        const int32_t t = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += t;
      }
      if (q <= parts) M[q] = run + inc;
      run += __shfl_sync(0xffffffffu, inc, 31);
    }
  }
  __syncthreads();

  int4* meta = reinterpret_cast<int4*>(p.meta);
  for (int32_t q = threadIdx.x; q < parts; q += kSchedParThreads) {
    const int32_t s = P[q], e = P[q + 1];
    if (s >= total) {
      meta[2 * q] = make_int4(b, b - 1, 0, S[b] - S[b - 1]);
      meta[2 * q + 1] = make_int4(0, 0, 0, 0);
      continue;
    }
    const int32_t br = R[q], er = e == S[R[q + 1]] ? R[q + 1] - 1 : R[q + 1];
    const int32_t begin_blk = s - S[br];
    int32_t first = begin_blk != 0, last = e != S[er + 1];
    const int32_t begin_split = first ? q - sched_par_last_le(P, parts, S[br]) : 0;
    if (br == er) first = last = first | last;
    meta[2 * q] = make_int4(br, er, begin_blk, e - S[er]);
    meta[2 * q + 1] = make_int4(begin_split, first, last, 0);
  }
  for (int32_t i = threadIdx.x; i <= b; i += kSchedParThreads) {
    p.num_splits[i] = i == 0 ? 0 : i + M[sched_par_last_le(P, parts, S[i] - 1)];
  }
}

struct FlashMLASparseSchedKernel {
  static void
  run(const tvm::ffi::TensorView topk_length,
      const tvm::ffi::TensorView extra_topk_length,
      const tvm::ffi::TensorView meta,
      const tvm::ffi::TensorView num_splits,
      int64_t topk,
      int64_t extra_topk,
      int64_t has_extra_length,
      int64_t num_cand,
      int64_t ov0,
      int64_t ov1,
      int64_t ov2,
      int64_t ov3,
      int64_t cost_block,
      int64_t cost_piece,
      int64_t cost_split,
      int64_t cost_combine,
      int64_t per_part) {
    using namespace host;
    auto B = SymbolicSize{"b"};
    auto B1 = SymbolicSize{"b_plus_1"};
    auto P = SymbolicSize{"num_parts"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(topk_length);
    TensorMatcher({P, kSchedMetaInts}).with_dtype<int32_t>().with_device(device).verify(meta);
    TensorMatcher({B1}).with_dtype<int32_t>().with_device(device).verify(num_splits);
    const int64_t b = B.unwrap();
    CHECK_HOST(B1.unwrap() == b + 1) << "num_splits must have b + 1 entries";
    CHECK_HOST(b >= 1) << "b out of range: " << b;
    CHECK_HOST(num_cand >= 1 && num_cand <= kSchedMaxCand) << "num_cand " << num_cand;
    if (has_extra_length) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(extra_topk_length);
    }
    const auto params = FlashMLASchedParams{
        .topk_length = static_cast<const int32_t*>(topk_length.data_ptr()),
        .extra_topk_length = has_extra_length ? static_cast<const int32_t*>(extra_topk_length.data_ptr()) : nullptr,
        .meta = static_cast<int32_t*>(meta.data_ptr()),
        .num_splits = static_cast<int32_t*>(num_splits.data_ptr()),
        .b = static_cast<int32_t>(b),
        .topk = static_cast<int32_t>(topk),
        .extra_topk = static_cast<int32_t>(extra_topk),
        .num_parts = static_cast<int32_t>(P.unwrap()),
        .num_cand = static_cast<int32_t>(num_cand),
        .overheads =
            {static_cast<int32_t>(ov0),
             static_cast<int32_t>(ov1),
             static_cast<int32_t>(ov2),
             static_cast<int32_t>(ov3)},
        .cost_block = static_cast<int32_t>(cost_block),
        .cost_piece = static_cast<int32_t>(cost_piece),
        .cost_split = static_cast<int32_t>(cost_split),
        .cost_combine = static_cast<int32_t>(cost_combine),
        .table_ints = 0,
    };
    if (per_part) {
      const size_t parts1 = static_cast<size_t>(params.num_parts) + 1;
      const size_t base = (b + 1 + (2 * kSchedMaxCand + 1) * parts1) * sizeof(int32_t);
      CHECK_HOST(base <= 48 * 1024) << "flashmla_sparse_sched_par: b=" << b << " needs " << base << " B smem";
      // Tables for the largest total the lengths allow (topk_length <= topk).
      const int64_t blocks_per_req =
          extra_topk ? (topk + kSchedBlock - 1) / kSchedBlock + (extra_topk + kSchedBlock - 1) / kSchedBlock
                     : ((topk > 1 ? topk : 1) + kSchedBlock - 1) / kSchedBlock;
      const int64_t max_total = b * blocks_per_req;
      int64_t table_ints = (1 + num_cand) * (max_total + 1);
      for (int64_t c = 0; c < num_cand; ++c) {
        const int64_t ov = params.overheads[c];
        const int64_t payload = (max_total + b * ov + params.num_parts - 1) / params.num_parts + ov;
        table_ints += max_total + b * ov + payload + 1;
      }
      // Arbitrary cap; above it the kernel takes the ballot search instead.
      constexpr size_t kMaxSmem = 160 * 1024;
      if (base + table_ints * sizeof(int32_t) > kMaxSmem) table_ints = 0;
      auto launch_params = params;
      launch_params.table_ints = static_cast<int32_t>(table_ints);
      const size_t smem = base + table_ints * sizeof(int32_t);
      RuntimeDeviceCheck(cudaFuncSetAttribute(
          flashmla_sparse_sched_par_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem));
      LaunchKernel(dim3(1), kSchedParThreads, device.unwrap(), smem)(flashmla_sparse_sched_par_kernel, launch_params);
      return;
    }
    const size_t stride = (static_cast<size_t>(params.num_parts) * kSchedMetaInts + b + 1 + 3) / 4 * 4;
    const size_t smem = ((b + 3) / 4 * 4 + num_cand * stride) * sizeof(int32_t);
    CHECK_HOST(smem <= 48 * 1024) << "flashmla_sparse_sched: b=" << b << " needs " << smem << " B smem";
    LaunchKernel(dim3(1), 32 * kSchedMaxCand, device.unwrap(), smem)(flashmla_sparse_sched_kernel, params);
  }
};

}  // namespace sglang
