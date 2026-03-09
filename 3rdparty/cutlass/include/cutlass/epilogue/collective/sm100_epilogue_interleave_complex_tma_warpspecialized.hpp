
/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/



#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/trace.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"
// {$nv-internal-release file}
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {


template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class CopyOpT2R_,
  class AlignmentC_,
  class AlignmentD_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_
>
class CollectiveEpilogue<
    Sm100InterleavedComplexTmaWarpSpecialized,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ThreadEpilogueOp_,
    CopyOpT2R_,
    AlignmentC_,
    AlignmentD_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    cute::enable_if_t<IsDefaultFusionOp<ThreadEpilogueOp_>::value>
> {
public:
  //
  // Type Aliases
  //
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  using DispatchPolicy = Sm100InterleavedComplexTmaWarpSpecialized;
  constexpr static int StagesC = DispatchPolicy::StagesC;
  using EpilogueTile = EpilogueTile_;
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementBias = typename cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpT2R = CopyOpT2R_;
  using AlignmentC = AlignmentC_;
  using AlignmentD = AlignmentD_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpG2S = CopyOpG2S_;
  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>,ElementD,ElementC>; // prevents void ref breakages

  constexpr static int ThreadCount = 128;
  constexpr static int kOutputAlignment = ThreadEpilogueOp::kCount;
  constexpr static bool isEpilogueBiasSupported = cutlass::epilogue::collective::detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::value;
  constexpr static bool isSourceNeeded = not cute::is_void_v<ElementC>;
  static constexpr int NumAccumulatorMtxs = 1;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>; // 0 stage to disable smem alloc
  using LoadPipelineState = cutlass::PipelineState<StagesC>;

  using StorePipeline = cutlass::PipelineTmaStore<1>; // tma store pipe has no smem alloc
  using StorePipelineState = cutlass::PipelineState<1>;

  using PipelineStorage = typename LoadPipeline::SharedStorage;

  // Epilog assumes a max scheduler pipe count to calculate the number of asynchronous tma update buffer they need.
  // In these epilogues, we don't need to update tensormaps at all. Setting this to INT_MAX.
  constexpr static uint32_t NumMaxSchedulerPipelineStageCount = INT_MAX;
  constexpr static bool is_m_major_C = cutlass::epilogue::collective::detail::is_m_major<StrideC>();
    using SmemLayoutStageC = decltype(tile_to_shape(SmemLayoutAtomC{}, product_each(shape(EpilogueTile{})),
      cute::conditional_t<is_m_major_C, Step<_2,_1>, Step<_1,_2>>{} ));
  constexpr static int StrideStageC = cute::cosize_v<SmemLayoutStageC>;
  constexpr static uint32_t TmaTransactionBytes = StrideStageC * sizeof_bits<ElementC>::value / 8;
  using SmemLayoutC = decltype(cute::append<3>(SmemLayoutStageC{}, Layout<Int<StagesC>, Int<StrideStageC>>{}));
  constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  struct CollectiveStorage {
    alignas(SmemAlignmentC) ArrayEngine<ElementC, cosize_v<SmemLayoutC>> smem_C;
  };

  struct SharedStorage { 
    struct TensorStorage {
      CollectiveStorage collective;
    } tensors;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  using TensorStorage = SharedStorage;
  using TensorMapStorage = SharedStorage;
  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

private:
  template <class ProblemShapeMNL>
  static constexpr auto
  get_tma_load_c(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    Tensor tensor_c = make_tensor(make_gmem_ptr(args.ptr_C),
                                  make_layout(problem_shape_mnl, append<3>(args.dC, _0{})));
    return make_tma_copy<typename GmemElementC::value_type>(CopyOpG2S{}, tensor_c, SmemLayoutStageC{}, EpilogueTile{}, _1{});
  }

public:
  // Device side epilogue params
  struct Params {
    using TMA_C = decltype(get_tma_load_c(repeat_like(append<3>(StrideC{},_1{}), int32_t(0)), Arguments{}));

    typename ThreadEpilogueOp::Params thread{};
    TMA_C tma_load_c;
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    auto problem_shape_mnl = select<0,1,3>(append<4>(problem_shape, 1));
    typename Params::TMA_C tma_load_c = get_tma_load_c(problem_shape_mnl, args);

    return { args.thread, tma_load_c, args.ptr_D, args.dD};
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }


  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    auto shape = cute::make_shape(M,N,L);

    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<AlignmentD{}>(shape, StrideD{});
    if constexpr (isSourceNeeded) {
      implementable = implementable && cutlass::detail::check_alignment<AlignmentC{}>(shape, StrideC{});
    }
    return implementable;  
  }

  //
  // Constructor and Data Members
  //
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params, SharedStorage&) : params(params) { };


  //
  // Non-static Device Methods
  //
public:

  // API with Global Accumulator in registers for FastFP32 (emulated MMA) kernels.
  // The accumulator in TMEM periodically loaded into the registers so that the MMA can clear out the TMEM accumulator
  // values for better accuracy. This epilogue accepts the accumulator in registers and take TiledCopy for the
  // TMEM->Reg as a parameter to be used in partitioning GMEM tensors C and D.
  template<
    class LoadPipeline,
    class LoadPipelineState,
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledCopy
  >
  CUTLASS_DEVICE auto
  operator()(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK cta_tile_shape_mnk,
      TileCoordMNKL cta_coord_mnkl,
      cute::Tensor<AccEngine, AccLayout>& tTR_rGlobAcc,                                      // (MMA,MMA_M,MMA_N)
      [[maybe_unused]] SharedStorage& shared_tensors,
      TiledCopy tiled_t2r) {

    using namespace cute;
    using X = Underscore;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be Register resident.");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(AccLayout{}) == 5, "Accumulators must be copy-partitioned:  (T2R,T2R_M,T2R_N,EPI_M,EPI_N)");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    auto problem_shape_mnl = select<0,1,3>(problem_shape_mnkl);
    auto cta_coord_mnl = select<0,1,3>(cta_coord_mnkl);
    auto cta_tiler = take<0,2>(cta_tile_shape_mnk);

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), problem_shape_mnl, append<3>(params.dD,_0{}));      // (M,N,L)
    Tensor gD = local_tile(mD, cta_tiler, cta_coord_mnl);                                              // (CTA_M,CTA_N)
    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.tensors.collective.smem_C.begin();
    Tensor sC =  cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Partition source and destination tiles according to tmem copy T2R partitioning (tTR_)
    auto thread_t2r = tiled_t2r.get_slice(threadIdx.x % size(tiled_t2r));
    Tensor tTR_sC   = thread_t2r.partition_D(sC);                                                  // (T2R,T2R_M,T2R_N)
    Tensor tTR_gD   = thread_t2r.partition_D(gD);                                                  // (T2R,T2R_M,T2R_N)


    Tensor coordCD = make_identity_tensor(problem_shape_mnl);                                     // (M,N,L) -> (m,n,l)
    Tensor cCD = local_tile(coordCD, cta_tiler, cta_coord_mnl);                             // (CTA_M,CTA_N) -> (m,n,l)
    Tensor tTR_cCD = thread_t2r.partition_D(cCD);                                       // (T2R,T2R_M,T2R_N) -> (m,n,l)
    // 2. Apply element-wise operation and store to gmem
    ThreadEpilogueOp epilogue_op{params.thread};
    CUTLASS_PRAGMA_UNROLL
    for (int iter_n = 0; iter_n < size<4>(tTR_rGlobAcc); ++iter_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter_m = 0; iter_m < size<3>(tTR_rGlobAcc); ++iter_m) { 
        int epi_m = iter_m, epi_n = iter_n;
        Tensor acc = tTR_rGlobAcc(_,_,_,epi_m,epi_n);
        Tensor tTR_gD_cur = tTR_gD(_,epi_m,epi_n);
        Tensor tTR_sC_cur = tTR_sC(_,_,_,load_pipe_consumer_state.index());
        Tensor tTR_cCD_cur = tTR_cCD(_,epi_m,epi_n);
        // source is needed
        if (epilogue_op.is_source_needed()) {
          load_pipeline.consumer_wait(load_pipe_consumer_state);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(acc); ++i) {
            if (elem_less(tTR_cCD_cur(i), problem_shape_mnl)) {
              tTR_gD_cur(i) = epilogue_op(acc(i), tTR_sC_cur(i));
            }
          }
          cutlass::arch::fence_view_async_shared();
          load_pipeline.consumer_release(load_pipe_consumer_state);
        }
        // source is not needed, avoid load
        else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(acc); ++i) {
            if (elem_less(tTR_cCD_cur(i), problem_shape_mnl)) {
              tTR_gD_cur(i) = epilogue_op(acc(i));
            }
          }
        }
        ++load_pipe_consumer_state;
      }
    }
    return load_pipe_consumer_state;
  }
  template<class CtaTileMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_load_pipe_increment(CtaTileMNK) {
    return 1;
  }

  template<class CtaTileMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_store_pipe_increment(CtaTileMNK) {
    return 1;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors([[maybe_unused]] Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_c.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    ThreadEpilogueOp epilogue_op{params.thread};
    return epilogue_op.is_source_needed();
  }


  template <bool... Args>
  CUTLASS_DEVICE auto
  tensormaps_init(
      [[maybe_unused]] Params const& params,
      [[maybe_unused]] TensorMapStorage& shared_tensormaps,
      [[maybe_unused]] int32_t sm_count,
      [[maybe_unused]] int32_t sm_idx,
      [[maybe_unused]] int32_t warp_group_idx = 0) const {
    // In the async tensormap update kernels, we will use operator[] to index the return value to locate the correct tensormap.
    // In other kernels, we will use return value as tensormap pointer directly.
    struct {
      CUTLASS_DEVICE operator cute::TmaDescriptor *() const {
        return reinterpret_cast<cute::TmaDescriptor*>(0);
      }
      CUTLASS_DEVICE auto operator [] (int) const {
        return reinterpret_cast<cute::TmaDescriptor*>(0);
      }
    } ret;
    return ret;
  }

  template <bool... Args>
  CUTLASS_DEVICE auto
  load_init(
      Params const& params,
      TensorMapStorage& shared_tensormap,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    return cute::make_tuple(
      tensormaps_init<true>(params, shared_tensormap, sm_count, sm_idx, 0)
    );
  }

  template<
    bool ReuseTmem = false,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  load(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      TensorStorage& shared_tensors,
      bool reverse_epi_n = false)
  {
    using namespace cute;

    int lane_idx = canonical_lane_idx();
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

    // The tma tensor C under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape = make_coord(m_coord, n_coord, l_coord);

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_c.get_tma_tensor(make_shape(M,N,L));                                //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(cta_tile_mnk));
    Tensor gC = local_tile(mC, take<0,2>(cta_tile_mnk), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sC = shared_tensors.tensors.collective.smem_C.begin();
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (TMA,TMA_M,TMA_N,PIPE_C)
    bool issue_tma_load = cute::elect_one_sync();
    ThreadEpilogueOp epilogue_op{params.thread};
    bool is_C_load_needed  = epilogue_op.is_source_needed();
    // Acquire the lock for this stage
    constexpr uint16_t mcast_mask = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int iter_n = 0; iter_n < size<3>(gC_epi); ++iter_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter_m = 0; iter_m < size<2>(gC_epi); ++iter_m) {
        uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
        int epi_m = iter_m, epi_n = iter_n;
        load_pipeline.producer_acquire(load_pipe_producer_state);

        // Execute the TMA load for C if needed
        if (issue_tma_load && is_C_load_needed) {

          load_pipeline.producer_expect_transaction(load_pipe_producer_state);
          copy(params.tma_load_c.with(*tma_barrier, mcast_mask),
              bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
        }
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }
    return load_pipe_producer_state;
  }


  CUTLASS_DEVICE void
  load_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      [[maybe_unused]] LoadPipelineState load_pipe_producer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      [[maybe_unused]] StorePipelineState store_pipe_producer_state)
  {
    load_pipeline.producer_tail(load_pipe_producer_state);
  }

  template <bool... Args>
  CUTLASS_DEVICE auto
  store_init(
      Params const& params,
      TensorMapStorage& shared_tensormap,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    return cute::make_tuple(
      tensormaps_init<false>(params, shared_tensormap, sm_count, sm_idx, 0)
    );
  }

  // FastF32 API
  template<
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TiledCopyT2R
  >
  CUTLASS_DEVICE auto
  store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine, AccLayout>& tTR_rAcc,
      TensorStorage& shared_tensors,
      TiledCopyT2R tiled_t2r)
  {
    auto load_pipe_consumer_state_new = (*this)(
      load_pipeline,
      load_pipe_consumer_state,
      problem_shape_mnkl,
      cta_tile_mnk,
      cta_coord_mnkl,
      tTR_rAcc,
      shared_tensors,
      tiled_t2r);

    return cute::make_tuple(load_pipe_consumer_state_new, store_pipe_producer_state);
  }

  template <class CtaTileMNK>
  CUTLASS_DEVICE void
  store_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      [[maybe_unused]] LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      [[maybe_unused]] StorePipelineState store_pipe_producer_state,
      [[maybe_unused]] CtaTileMNK cta_tile_mnk)
  {
  }

  // Dummy methods to perform different parts of TMA/Tensormap modifications
  template <bool IsLoad, bool WaitForInflightTmaRequests = true, class ProblemShape>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      [[maybe_unused]] TensorMapStorage& shared_tensormap,
      [[maybe_unused]] Params const& params,
      [[maybe_unused]] cute::TmaDescriptor const* tensormap,
      [[maybe_unused]] ProblemShape problem_shape,
      [[maybe_unused]] int32_t next_batch
      ,[[maybe_unused]] bool fence_release = false // {$nv-internal-release}
  ) { }

  template <bool IsLoad, bool WaitForInflightTmaRequests = true>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release(
      [[maybe_unused]] TensorMapStorage& shared_tensormap,
      [[maybe_unused]] cute::TmaDescriptor const* tensormap
      ,[[maybe_unused]] bool fence_release = false // {$nv-internal-release}
  ) { }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire([[maybe_unused]] cute::TmaDescriptor const* tensormap) { }

protected:
  Params const& params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
