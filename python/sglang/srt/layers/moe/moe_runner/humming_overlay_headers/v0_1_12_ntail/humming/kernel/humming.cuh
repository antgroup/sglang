// Modified copy of humming-kernels 0.1.12 include/humming/kernel/humming.cuh (Apache-2.0).
// SGLang header overlay v0_1_12_ntail (DeepSeek-V4.1 SM90 MoE path); see
// sglang/srt/layers/moe/moe_runner/humming_overlay.py. Same as the v0_1_12_bm32 overlay (block_m <= 32: 0.1.15
// style main loop, per-block wgmma width n8 / n16 / n24 / full) plus, for block_m 48 / 64, upstream's main loop
// instantiated per wgmma width (n8 .. n{block_m}) with the narrowest one covering the block's valid rows chosen
// once per block, so an expert's last, partly filled m-block does not issue tensor work for its padding rows;
// and, for a K padded past the last k32 slab (TP8 w2: 288 -> 320), the all-padding slabs' wgmma is skipped.
// block_m <= 32 kernels are the bm32 overlay's.

#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>

#include <humming/datatype/dequant.cuh>


template <uint32_t kValue>
struct OvCompileTimeConstant {
  static constexpr uint32_t value = kValue;
};

template <uint32_t kIndex = 0, uint32_t kEnd, class Func>
CUDA_INLINE void ov_static_for(Func &&func) {
  if constexpr (kIndex < kEnd) {
    func(OvCompileTimeConstant<kIndex>{});
    ov_static_for<kIndex + 1, kEnd>(static_cast<Func &&>(func));
  }
}

// Runs func(OvCompileTimeConstant<g>) for the block's wgmma width: g 8-row groups, 0 = full block_m.
// block_m 16 / 32: the bm32 overlay's dispatch as written (decode kernels unchanged); 48 / 64: any g.
template <class MMA, class Func>
CUDA_INLINE void ov_dispatch_n_groups(const MMA &mma, Func &&func) {
  if constexpr (MMA::kNSelCapable && MMA::kNumNGroups <= 4) {
    if (mma.num_n_groups == 1) return func(OvCompileTimeConstant<1>{});
    if constexpr (MMA::kNumNGroups == 4) {
      if (mma.num_n_groups == 2) return func(OvCompileTimeConstant<2>{});
      if (mma.num_n_groups == 3) return func(OvCompileTimeConstant<3>{});
    }
  } else if constexpr (MMA::kNSelCapable) {
    bool done = false;
    ov_static_for<1, MMA::kNumNGroups>([&](auto groups) {
      if (!done && mma.num_n_groups == decltype(groups)::value) {
        func(groups);
        done = true;
      }
    });
    if (done) return;
  }
  func(OvCompileTimeConstant<0>{});
}

template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};

CUDA_INLINE const void *param_to_ptr(const CUtensorMap &x) { return &x; }
CUDA_INLINE const void *param_to_ptr(void *const &x) { return x; }


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaAS>::Type AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS2>::Type BS2,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  uint64_t debug_start_clock = debug_kernel_timer_start();
  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Ctx = KernelContext<
      MmaOpClass, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Scheduler = Scheduler<Ctx>;
  using ProducerPipeline = ProducerPipeline<Ctx>;
  using ConsumerPipeline = ConsumerPipeline<Ctx>;
  using MainloopArithmetic = MainloopArithmetic<Ctx>;
  using EpilogueArithmetic = EpilogueArithmetic<Ctx>;
  using MMA = Mma<Ctx, MainloopArithmetic>;
  using Epilogue = EpiloguePipeline<Ctx, MMA, EpilogueArithmetic>;
  using S2RMemoryPipeline = S2RMemoryPipeline<Ctx, MMA, Epilogue>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  const KernelParams params{
      shape_m, top_k, use_int64_expert_layout,
      param_to_ptr(A), param_to_ptr(B), param_to_ptr(AS), param_to_ptr(BS),
      param_to_ptr(BZP), param_to_ptr(Bias), param_to_ptr(C), param_to_ptr(BS2),
      sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr,
      tensor_map_buffer, locks};
  auto ctx = Ctx(smem, params);

  auto scheduler = Scheduler(ctx);
  auto mainloop_arith = MainloopArithmetic();
  auto epilogue_arith = EpilogueArithmetic();
  auto mma = MMA(ctx, mainloop_arith);
  auto epilogue = Epilogue(ctx, epilogue_arith);
  auto producer = ProducerPipeline(ctx);
  auto consumer = ConsumerPipeline(ctx);
  auto s2r_pipe = S2RMemoryPipeline(ctx, mma, epilogue);

  // [sglang] K-tail skip (block_m > 32 tiles): with K padded (PadShape::K, e.g. TP8 w2 K 288 -> 320),
  // the k32 slabs of the last k-block that lie wholly in the padding hold zero activations (zero-filled
  // loads) and zero weights; their wgmma is not issued. Data-parallel tiles only (no stream-K).
  constexpr uint32_t kValidK = ProblemShape::K - PadShape::K;
  constexpr uint32_t kLastBlockValidK = kValidK - (ProblemShape::K / BlockShape::K - 1) * BlockShape::K;
  constexpr bool kSkipKTail = Ctx::kUseWgmma && !Ctx::kUseStreamK && !Ctx::kUsePackedKLayout &&
                              PadShape::K > 0 && ProblemShape::K % BlockShape::K == 0 &&
                              kLastBlockValidK < BlockShape::K && kLastBlockValidK % Ctx::kPartMmaShapeK == 0;
  auto skip_k_slab = [&](bool last_block, uint32_t warp_iter_id) -> bool {
    if constexpr (kSkipKTail) {
      return last_block && warp_iter_id * Ctx::kPartMmaShapeK + ctx.k_warp_offset() >= kLastBlockValidK;
    } else {
      return false;
    }
  };

  producer.init_mbarrier();
  mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();

  while (scheduler.get_next_block()) {
    debug_kernel_timeout_check(debug_start_clock);
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    if constexpr (TuningConfig::kUseTmaC) {
      tma_wait_store_group<0, true>();
      __syncthreads();
    }
    // [sglang] Decode tiles (block_m <= 32) take the restructured main loop; larger tiles keep
    // upstream's loop, which measured faster for some prefill configs.
    if constexpr (BlockShape::M <= 32) {
      constexpr uint32_t kStaticSliceIters = ProblemShape::K / BlockShape::K;
      const uint32_t num_slice_iters = Ctx::kUseStreamK ? slice_iters : kStaticSliceIters;
      producer.template load_stage<true, true>(0);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
        producer.load_stage(stage_id, stage_id < num_slice_iters);
      };

      consumer.template wait_stage<true>(kNumStages);
      s2r_pipe.template load_stage_iter<true>(0, 0);
      mma.transform_b(0, 0);

      auto mainloop = [&](auto groups_tag) {
        constexpr uint32_t kGroups = decltype(groups_tag)::value;
        auto run_mma = [&](uint32_t stage_id, uint32_t warp_iter_id) {
          if constexpr (Ctx::kUseWgmma) {
            mma.template run<kGroups>(stage_id, warp_iter_id);
          } else {
            mma.run(stage_id, warp_iter_id);
          }
        };
        auto consume_stage = [&](auto stage, uint32_t slice_iter) {
          constexpr uint32_t stage_id = decltype(stage)::value;
          debug_kernel_timeout_check(debug_start_clock);
          const uint32_t remaining_iters = num_slice_iters - slice_iter;
          if (remaining_iters == 1) producer.load_channel();
          PRAGMA_UNROLL
          for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
            if (warp_iter_id == Ctx::kWarpIters - 1 && remaining_iters > 1) {
              consumer.wait_stage((stage_id + 1) % kNumStages);
            }
            s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
            run_mma(stage_id, warp_iter_id);
            if (warp_iter_id == Ctx::kWarpIters - 2) {
              __syncthreads();
              if constexpr (kNumStages > 2) {
                producer.load_stage(stage_id + kNumStages - 1, remaining_iters >= kNumStages);
              }
            }
            mma.transform_b(
                (warp_iter_id + 1) % 2,
                (warp_iter_id + 1) % Ctx::kWarpIters);
          }
          if constexpr (kNumStages == 2) {
            producer.load_stage(stage_id, remaining_iters > kNumStages);
          }
        };

        const uint32_t num_full_stage_cycles = num_slice_iters / kNumStages;
        for (uint32_t cycle_id = 0; cycle_id < num_full_stage_cycles; cycle_id++) {
          ov_static_for<0, kNumStages>([&](auto stage) {
            consume_stage(stage, cycle_id * kNumStages + decltype(stage)::value);
          });
        }
        const uint32_t tail_stage_iters = num_slice_iters % kNumStages;
        ov_static_for<0, kNumStages>([&](auto stage) {
          constexpr uint32_t stage_id = decltype(stage)::value;
          if (stage_id < tail_stage_iters) {
            consume_stage(stage, num_full_stage_cycles * kNumStages + stage_id);
          }
        });
      };
      if constexpr (Ctx::kUseWgmma) {
        ov_dispatch_n_groups(mma, mainloop);
      } else {
        mainloop(OvCompileTimeConstant<0>{});
      }
    } else {
      producer.template load_stage<true, true>(0);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
        producer.load_stage(stage_id, stage_id < slice_iters);
      };

      consumer.template wait_stage<true>(kNumStages);
      s2r_pipe.template load_stage_iter<true>(0, 0);
      mma.transform_b(0, 0);

      // [sglang] ntail overlay: an m-block whose valid rows fit in fewer 8-row groups runs a copy of the
      // loop that issues the narrow wgmma; full blocks run upstream's loop as written (kept outside the
      // narrow copies' dispatch so its code generation is unchanged).
      auto mainloop_narrow = [&](auto groups_tag) {
        constexpr uint32_t kGroups = decltype(groups_tag)::value;
        while (slice_iters) {
          debug_kernel_timeout_check(debug_start_clock);
          PRAGMA_UNROLL
          for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
            debug_kernel_timeout_check(debug_start_clock);
            if (slice_iters == 1) producer.load_channel();
            PRAGMA_UNROLL
            for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
              s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
              if constexpr (Ctx::kUseWgmma) {
                if (!skip_k_slab(slice_iters == 1, warp_iter_id)) mma.template run<kGroups>(stage_id, warp_iter_id);
              } else {
                mma.run(stage_id, warp_iter_id);
              }
              if (warp_iter_id == Ctx::kWarpIters - 2) {
                if constexpr (kNumStages == 2) {
                  __syncthreads();
                  if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
                } else {
                  __syncthreads();
                  producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
                  if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
                }
              }

              mma.transform_b(
                  (warp_iter_id + 1) % 2,
                  (warp_iter_id + 1) % Ctx::kWarpIters);
            }

            if constexpr (kNumStages == 2) producer.load_stage(stage_id, slice_iters > kNumStages);
            slice_iters--;
            if (!slice_iters) break;
          };
        };
      };
      bool ran_narrow = false;
      if constexpr (Ctx::kUseWgmma && MMA::kNSelCapable && BlockShape::M > 32) {
        if (mma.num_n_groups < MMA::kNumNGroups) {
          ov_static_for<1, MMA::kNumNGroups>([&](auto groups) {
            if (!ran_narrow && mma.num_n_groups == decltype(groups)::value) {
              mainloop_narrow(groups);
              ran_narrow = true;
            }
          });
        }
      }
      if (!ran_narrow) {
      while (slice_iters) {
        debug_kernel_timeout_check(debug_start_clock);
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          debug_kernel_timeout_check(debug_start_clock);
          if (slice_iters == 1) producer.load_channel();
          PRAGMA_UNROLL
          for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
            s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
            if constexpr (Ctx::kUseWgmma) {
              if (!skip_k_slab(slice_iters == 1, warp_iter_id)) mma.run(stage_id, warp_iter_id);
            } else {
              mma.run(stage_id, warp_iter_id);
            }
            if (warp_iter_id == Ctx::kWarpIters - 2) {
              if constexpr (kNumStages == 2) {
                __syncthreads();
                if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
              } else {
                __syncthreads();
                producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
                if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
              }
            }

            mma.transform_b(
                (warp_iter_id + 1) % 2,
                (warp_iter_id + 1) % Ctx::kWarpIters);
          }

          if constexpr (kNumStages == 2) producer.load_stage(stage_id, slice_iters > kNumStages);
          slice_iters--;
          if (!slice_iters) break;
        };
      };
      }
    }

    consumer.wait_channel();
    s2r_pipe.load_channel(scheduler.slice_id);
    __syncthreads();
    epilogue.call(mma.final_regs_c_as_ptr());
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB > 1) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};
