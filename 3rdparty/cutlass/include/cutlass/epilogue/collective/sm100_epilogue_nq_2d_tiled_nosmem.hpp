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

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// For Nq 2d tiled sm100 kernels
template <
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class CopyOpT2R_
>
class CollectiveEpilogue<
    Sm100NoSmemNq2dTiledWarpSpecialized,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ThreadEpilogueOp_,
    CopyOpT2R_>
{
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator
  using DispatchPolicy = Sm100NoSmemNq2dTiledWarpSpecialized;
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;     //? TODO: allow this to be its own type too?
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_; // Useless since we construct strideC within kernel
  using ElementD = ElementD_;
  using StrideD = StrideD_; // Useless since we construct strideD within kernel
  using CopyOpT2R = CopyOpT2R_;

  using LoadPipeline = cutlass::PipelineTransactionAsync<0>; // 0 stage to disable smem alloc
  using LoadPipelineState = cutlass::PipelineState<0>;

  using ElementTmem = uint32_t;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");  // TODO: Don't know the reason why it's rank 3, need to scrub
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");  // TODO: Don't know the reason why it's rank 3, need to scrub

  // Residual is not supported
  constexpr static bool IsSourceSupported = false;

  constexpr static int ThreadCount = 128;
  constexpr static int kOutputAlignment = ThreadEpilogueOp::kCount;
  constexpr static bool isEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::value;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  struct SharedStorage {
    struct TensorStorage : aligned_struct<128, _0> { } tensors;
    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{}; // TODO: make sure the stride is not linearized, currently assigned by make_cute_packed_stride
    ElementD* ptr_D = nullptr;
    StrideD dD{}; // TODO: make sure the stride is not linearized, currently assigned by make_cute_packed_stride
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    // TODO: problem_shape_MNKL is useless here so far
    // Here is used for TMA desc construction. We don't have TMA for store but we need the load.
    return args;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    // TODO: delegate to ThreadEpilogueOp // {$nv-release-never}
    return true;
  }

  template<class TensorStorage>
  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params,
                     [[maybe_unused]] TensorStorage& shared_tensors)
                     : params(params) {
  };

  template <
    class ProblemShape,
    class TileShape,
    class Params
  >
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape const& problem_shape,
      TileShape const& tile_shape,
      Params const& params)
  {
    // Implement this fcn if residual loading is needed
    return 0;
  }

  template <
    class ProblemShape,
    class TileShape,
    class Params
  >
  CUTLASS_DEVICE auto
  store_init(
      ProblemShape const& problem_shape,
      TileShape const& tile_shape,
      Params const& params)
  {
    using namespace cute;
    using X = Underscore;

    auto M = get<0>(problem_shape);               // K
    auto N = shape<1>(problem_shape);             // (Q,N)
    auto K = get<0>(shape<2>(problem_shape));     // (C,S,R,T)
    auto P = get<1>(shape<2>(problem_shape));     // P
    auto Z = get<3>(problem_shape);               // Z

    // Unpack problem shape and compute stride by ourselves
    auto [stride_tile_q, 
          stride_tile_p,
          stride_tile_z,
          stride_tile_n] = get<0>(params.dD);          // (q, p, z, n)

    // Construct linearized mD then tile to gD_mn
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M, N, Z, P),
      make_stride(_1{}, make_stride(stride_tile_q, stride_tile_n), stride_tile_z, stride_tile_p));         // (M,N,Z,P)
    Tensor mD_linear = make_identity_tensor(make_shape(M, size(N), Z, P));                                 // (M,N,Z,P)
    Tensor mD_linearized = make_tensor(mD.data(),
      composition(mD.layout(), mD_linear(_0{}), mD_linear.layout()));                                      // (M,N,Z,P)
    Tensor gD_mnl = local_tile(
      mD_linearized, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                       // (TILE_M,TILE_N,m,n,z,p)

    return cute::make_tuple(gD_mnl);
  }

  template<
    class Params,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class EpiLoadPipeline,
    class EpiLoadPipelineState,
    class GTensorD,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TensorStorage
  >
  CUTLASS_DEVICE auto
  store(
      [[maybe_unused]] Params const& params,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState accumulator_pipe_consumer_state,
      [[maybe_unused]] EpiLoadPipeline epi_load_pipeline,
      [[maybe_unused]] EpiLoadPipelineState epi_load_pipe_consumer_state,
      cute::tuple<GTensorD> const& store_inputs,
      TileCoordMNKL tile_coord_mnkl,
      [[maybe_unused]] int p_pixels_start,
      [[maybe_unused]] int p_pixels_end,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      [[maybe_unused]] TensorStorage& shared_tensors)
  {
    using namespace cute;
    using X = Underscore;
    using Converter = cutlass::NumericConverter<ElementOutput, ElementAccumulator>;
    Converter converter{};

    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(AccLayout{}) == 4, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N,STAGE)");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    const int epi_thread_idx = threadIdx.x % ThreadCount;
    ThreadEpilogueOp epilogue_op{params.thread};
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    GTensorD gD_mnl = get<0>(store_inputs);                                                  // (TILE_M,TILE_N,m,n,l,p)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord,_);                                             // (TILE_M,TILE_N,p)

    Tensor tC = accumulators(_,0,0,_);

    //
    // Epilogue streaming loop
    //

    CUTE_NO_UNROLL
    for (int p_ = 0; p_ < get<2>(gD.shape()) /* conv_P */; ++p_) {
      // Wait for mma warp to fill tmem buffer with accumulator results
      acc_pipeline.consumer_wait(accumulator_pipe_consumer_state);

      // TMEM_LOAD
      Tensor tC_slice = make_tensor(tC(_,accumulator_pipe_consumer_state.index()).data(), get<0>(tC(_,accumulator_pipe_consumer_state.index()).layout()));
      // Load acc element in TMEM to reg
      auto ldtm = make_tmem_copy(Copy_Atom<CopyOpT2R, ElementAccumulator>{}, tC_slice);
      auto thr_ldtm = ldtm.get_slice(epi_thread_idx);
      Tensor tDtC = thr_ldtm.partition_S(tC_slice);                                             // (TMEM_LOAD,TMEM_LOAD_M,TMEM_LOAD_N)
      // Transpose Layout gD => Transpose Layout tDgC
      Tensor tDgC = thr_ldtm.partition_D(gD(make_coord(_,_,p_)));
      Tensor tDrC = make_tensor<ElementAccumulator>(shape(tDgC));
      copy(ldtm, tDtC, tDrC);
      cutlass::arch::fence_view_async_tmem_load();

      acc_pipeline.consumer_release(accumulator_pipe_consumer_state);
      ++accumulator_pipe_consumer_state;

      // Pred and output tensor
      auto tDgC_layout = tDgC.layout();
      auto tDgC_coord_layout = make_composed_layout(
          make_layout(tDgC_layout.layout_a().shape(), make_basis_like(tDgC_layout.layout_a().shape())),
          tDgC_layout.offset(),
          tDgC_layout.layout_b());
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tDrC); ++i) {
        auto cD = tDgC_coord_layout(i);
        if (elem_less(cD, tDgC_coord_layout.layout_a().shape())) {
          tDgC(i) = converter(tDrC(i));
        }
      }
    }

    return cute::make_tuple(accumulator_pipe_consumer_state, epi_load_pipe_consumer_state);
  }

protected:
  Params const & params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
