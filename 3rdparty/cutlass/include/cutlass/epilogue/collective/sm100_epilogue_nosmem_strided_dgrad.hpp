/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// POC for functional verification, do not release {$nv-release-never file}

#pragma once

#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {

struct NoSmemWsStridedDgrad {};

namespace collective {

// Strided Dgrad NoSmem Builder
template <
  class CtaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueScheduleType,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    CtaTileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleType,
    epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute, RoundStyle>,
    cute::enable_if_t<cute::is_same_v<EpilogueScheduleType, NoSmemWsStridedDgrad>>> {

  static_assert(cute::is_same_v<EpilogueTileType, EpilogueTileAuto>, "Epilogue subtiling requires smem");
  static_assert(cute::sizeof_bits_v<ElementD> != 4 and cute::sizeof_bits_v<ElementD> != 6, "Output element requires smem");

  // Passing void C disables source load
  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = DisableSource ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  using EpilogueTile = decltype(take<0,2>(CtaTileShape_MNK{}));
  using AccLoadOp = cute::conditional_t<cute::sizeof_bits_v<ElementAccumulator> == 16,
                      SM100_TMEM_LOAD_32dp32b32x_16b, SM100_TMEM_LOAD_32dp32b32x>;

  using ThreadOp = thread::LinearCombination<
    ElementD, 1, ElementAccumulator, ElementCompute,
    ScaleType, RoundStyle, ElementC>;

  using CollectiveOp = cutlass::epilogue::collective::CollectiveEpilogue<
      cutlass::epilogue::Sm100NoSmemWsStridedDgrad,
      EpilogueTile,
      ElementC,
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      ThreadOp,
      AccLoadOp
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class CopyOpT2R_
>
class CollectiveEpilogue<
    Sm100NoSmemWsStridedDgrad,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ThreadEpilogueOp_,
    CopyOpT2R_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Sm100NoSmem;
  using EpilogueTile = EpilogueTile_;
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementCompute = float;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpT2R = CopyOpT2R_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  using LoadPipeline = cutlass::PipelineTransactionAsync<0>; // 0 stage to disable smem alloc
  using LoadPipelineState = cutlass::PipelineState<0>;

  using StorePipeline = cutlass::PipelineTmaStore<1>; // tma store pipe has no smem alloc
  using StorePipelineState = cutlass::PipelineState<1>;

  using PipelineStorage = typename LoadPipeline::SharedStorage;

  constexpr static int ThreadCount = 128;
  constexpr static uint32_t TmaTransactionBytes = 0;

private:
  static constexpr int NumSpatialDims = rank<0>(StrideD{}) - 1;
  static_assert(0 < NumSpatialDims && NumSpatialDims <= 3);
  static_assert(is_congruent<StrideC, StrideD>::value);

  using ProblemShape = cutlass::conv::ConvProblemShape<conv::Operator::kDgrad, NumSpatialDims>;
  static constexpr int MaxTraversalStride = 8; // TMA limitation, kept here for parity

public:

  struct SharedStorage { };
  using TensorStorage = SharedStorage;

  constexpr static int NumAccumulatorMtxs = 1;

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  struct Params : Arguments {
    using Arguments::Arguments;

    Params(Arguments args) : Arguments(args) {}

    using ShapeC = decltype(declval<ProblemShape>().get_shape_C());
    ShapeC shape_WHDNC;
  };

  //
  // Conv Host Methods
  //

  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    Params params(args);
    params.shape_WHDNC = problem_shape.get_shape_C();
    return params;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace,
                       cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    bool implementable = true;

    const auto & traversal_stride  = problem_shape.traversal_stride;
    for (auto stride : traversal_stride) {
      implementable &= (stride > 0 && stride <= MaxTraversalStride);
    }

    return implementable;
  }

  //
  // Static Device Methods
  //
  template<class CtaTileMNK>
  CUTLASS_DEVICE
  static constexpr int
  get_load_pipe_increment(CtaTileMNK) {
    return 1;
  }

  template<class CtaTileMNK>
  CUTLASS_DEVICE
  static constexpr int
  get_store_pipe_increment(CtaTileMNK) {
    return 1;
  }

  CUTLASS_DEVICE
  static void
  prefetch_tma_descriptors(Params const&) {
  }

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return false;
  }

  //
  // Constructor and Data Members
  //
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params, SharedStorage&) : params(params) { };

protected:
  Params const& params;

  //
  // Non-static Device Methods
  //
public:

  template<
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
      TensorStorage& shared_tensors) {
    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE void
  load_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state) {
  }

  template<
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TileScheduler
  >
  CUTLASS_DEVICE auto
  store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TensorStorage& shared_tensors,
      TileScheduler const& scheduler,
      typename TileScheduler::WorkTileInfo const& work_tile_info,
      uint32_t num_barriers_per_tile,
      uint32_t first_barrier_in_tile) {
    using namespace cute;
    using X = Underscore;

    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(CtaCoordMNKL{}) == 4, "CtaCoordMNKL must be rank 4");

    // Represent the full output tensor
    auto shape_WHDNC = params.shape_WHDNC;
    Tensor mC_whdnc = make_tensor(make_gmem_ptr(params.ptr_C), shape_WHDNC, params.dC);               // ((W,H,D,N),C)
    Tensor mD_whdnc = make_tensor(make_gmem_ptr(params.ptr_D), shape_WHDNC, params.dD);               // ((W,H,D,N),C)

    // Get decomposition tile+offsets and slice current decomposition
    auto traversal_stride = make_shape(get<3>(problem_shape_mnkl));                                       // ((V,U,O))
    auto act_offset = make_coord(get<3>(cta_coord_mnkl));                                                 // ((w,h,d))
    Tensor mC = outer_partition(mC_whdnc, traversal_stride, act_offset);                                  // ((W/V,H/U,D/O,N),C)
    Tensor mD = outer_partition(mD_whdnc, traversal_stride, act_offset);                                  // ((W/V,H/U,D/O,N),C)

    // Apply MMA tiler and slice current MMA tile
    static constexpr int atom_ctas = size(typename TiledMma::AtomThrID{});
    auto mma_coord_mn = make_coord(get<0>(cta_coord_mnkl) / atom_ctas, get<1>(cta_coord_mnkl));
    Tensor gC = local_tile(mC, take<0,2>(mma_tile_mnk), mma_coord_mn);                                    // (TILE_M,TILE_N)
    Tensor gD = local_tile(mD, take<0,2>(mma_tile_mnk), mma_coord_mn);                                    // (TILE_M,TILE_N)

    // Partition to get data this CTA is responsible for
    auto atom_cta_idx = blockIdx.x % atom_ctas;
    ThrMMA cta_mma = tiled_mma.get_slice(atom_cta_idx);
    Tensor tCgC = cta_mma.partition_C(gC);                                                         // (MMA,MMA_M,MMA_N)
    Tensor tDgD = cta_mma.partition_C(gD);                                                         // (MMA,MMA_M,MMA_N)

    // Partition source and destination tiles according to tmem copy T2R partitioning (tTR_)
    using ElementAccumulator = typename AccEngine::value_type;
    auto tiled_t2r = make_tmem_copy(Copy_Atom<CopyOpT2R, ElementAccumulator>{}, accumulators);
    auto thread_t2r = tiled_t2r.get_slice(threadIdx.x % size(tiled_t2r));
    Tensor tTR_tAcc = thread_t2r.partition_S(accumulators);                                // (T2R,T2R_MMA,T2R_M,T2R_N)
    Tensor tTR_gC   = thread_t2r.partition_D(tCgC);                                        // (T2R,T2R_MMA,T2R_M,T2R_N)
    Tensor tTR_gD   = thread_t2r.partition_D(tDgD);                                        // (T2R,T2R_MMA,T2R_M,T2R_N)
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(take<0,1>(shape(tTR_gD)));           // (T2R)

    // Get matching coordinate tensors
    Tensor mD_crd_whdnc = make_identity_tensor(shape_WHDNC);                                   // ((W,H,D,N),C)
    Tensor mD_crd = outer_partition(mD_crd_whdnc, traversal_stride, act_offset);                      // ((W/V,H/U,D/O,N), C)
    Tensor cD = local_tile(mD_crd, take<0,2>(mma_tile_mnk), mma_coord_mn);                            // (TILE_M,TILE_N)
    Tensor tDcD = cta_mma.partition_C(cD);                                                            // (MMA,MMA_M,MMA_N)
    Tensor tTR_cD = thread_t2r.partition_D(tDcD);                                                     // (T2R,T2R_MMA,T2R_M,T2R_N)

    // Initialize fusion params
    ThreadEpilogueOp epilogue_op{params.thread};
    // backprop zeros is passed as k_coord
    auto backprop_zeros = get<2>(cta_coord_mnkl);

    // Wait for mma warp to fill tmem buffer with accumulator results
    acc_pipeline.consumer_wait(acc_pipe_consumer_state);

    CUTLASS_PRAGMA_UNROLL
    for (auto t2r_iter = make_coord_iterator(take<1,4>(shape(tTR_gD))); // iterates over (T2R_MMA,T2R_M,T2R_N)
         t2r_iter != ForwardCoordIteratorSentinel{}; ++t2r_iter) {
      // 1. Load accumulators into register from tmem
      if (backprop_zeros) {
        fill(tTR_rAcc, 0);
      }
      else {
        auto t2r_slice = prepend(*t2r_iter, _); // (_,t2r_mma,t2r_m,t2r_n)
        copy(tiled_t2r, tTR_tAcc(t2r_slice), tTR_rAcc);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); ++i) {
        auto t2r_coord = prepend(*t2r_iter, i);
        if (elem_less(tTR_cD(t2r_coord), shape_WHDNC)) {
          tTR_gD(t2r_coord) = epilogue_op(tTR_rAcc(i), tTR_gC(t2r_coord));
        }
      }
    }

    // Let mma warp know tmem buffer is consumed and empty
    cutlass::arch::fence_view_async_tmem_load();
    acc_pipeline.consumer_release(acc_pipe_consumer_state);
    ++acc_pipe_consumer_state;
    ++load_pipe_consumer_state;
    ++store_pipe_producer_state;

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state, acc_pipe_consumer_state);
  }

  template <class CtaTileMNK>
  CUTLASS_DEVICE void
  store_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      [[maybe_unused]] LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      [[maybe_unused]] StorePipelineState store_pipe_producer_state,
      [[maybe_unused]] CtaTileMNK cta_tile_mnk) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
