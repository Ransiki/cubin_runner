/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/detail/sm100_tmem_helper.hpp"

#include "cutlass/conv/detail.hpp"
#include "cute/tensor.hpp"
#include "cute/numeric/int.hpp"

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
  class ElementAccumulator_,
  class ElementCompute_,
  class FusionOp_,
  class AccumulatorCopyOpT2R_,
  class ShuffleCopyOpR2T_,
  class ShuffleCopyOpT2R_,
  class OffsetCopyOpR2T_,
  class OffsetCopyOpT2R_,
  class TmemShuffleLayout_,
  class TmemOffsetLayout_
>
class CollectiveEpilogue<
    Sm100TmemShuffleNq2dTiledWarpSpecialized,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ElementAccumulator_,
    ElementCompute_,
    FusionOp_,
    AccumulatorCopyOpT2R_,
    ShuffleCopyOpR2T_,
    ShuffleCopyOpT2R_,
    OffsetCopyOpR2T_,
    OffsetCopyOpT2R_,
    TmemShuffleLayout_,
    TmemOffsetLayout_
    >
{
public:
  // derived types of output thread level operator
  using DispatchPolicy = Sm100TmemShuffleNq2dTiledWarpSpecialized;
  using ThreadEpilogueOp = FusionOp_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD_, ElementC_>;
  using StrideC = StrideC_; // Useless since we construct strideC within kernel
  using ElementD = ElementD_;
  using StrideD = StrideD_; // Useless since we construct strideD within kernel

  static_assert(DisableSource, "Residual C is unsupported since it has poor performance");

  using AccumulatorCopyOpT2R = AccumulatorCopyOpT2R_;
  // Shuffle by TMEM to get a transposed output
  using ShuffleCopyOpR2T = ShuffleCopyOpR2T_;
  using ShuffleCopyOpT2R = ShuffleCopyOpT2R_;
  // Pre-compute gmem address and store to tmem to reduce ALU and Register pressure
  using OffsetCopyOpR2T = OffsetCopyOpR2T_;
  using OffsetCopyOpT2R = OffsetCopyOpT2R_;

  using TmemShuffleLayout   = TmemShuffleLayout_;
  using TmemOffsetLayout    = TmemOffsetLayout_;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static constexpr auto ScaleType = thread::ScaleType::OnlyAlphaScaling;
  static constexpr auto PerChannelScaleType = thread::ScaleType::PerChannelScaling;
  static constexpr auto RoundStyle = FloatRoundStyle::round_to_nearest;
  // Revisit FragmentSize setting in perf tuning
  static constexpr int FragmentSize = 4;

  static_assert((!FusionOp_::IsPerRowScaleSupported) ||
    cute::is_same_v<typename FusionOp_::ActivationFn, thread::Identity<ElementCompute>>,
    "Activation is not supported by per-chanel scaling epilogue."
  );
  static_assert(!FusionOp_::IsPerColScaleSupported, "Per-column scaling is not supported.");
  static_assert(!FusionOp_::IsPerColBiasSupported, "Per-column bias is not supported.");

  // Select ThreadEpiOp based on FusionOp
  using ThreadEpiOp = cute::conditional_t<
    FusionOp_::IsPerRowScaleSupported,
    // Per-channel scaling
    thread::LinearCombination<
        ElementD, FragmentSize, ElementAccumulator, ElementCompute,
        PerChannelScaleType, RoundStyle, ElementC>,
    cute::conditional_t<
      FusionOp_::IsEltActSupported,
      // Scaling / bias / activation
      thread::LinearCombinationBiasElementwise<
          ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, FragmentSize,
          typename FusionOp_::ActivationFn, cutlass::plus<ElementCompute>,
          false, typename FusionOp_::ElementBias>,
      cute::conditional_t<
        FusionOp_::IsPerRowBiasSupported,
        // Scaling / bias
        thread::LinearCombinationBiasElementwise<
            ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, FragmentSize,
            thread::Identity<ElementCompute>, cutlass::plus<ElementCompute>,
            false, typename FusionOp_::ElementBias>,
        // Scaling
        thread::LinearCombination<
            ElementD, FragmentSize, ElementAccumulator, ElementCompute,
            ScaleType, RoundStyle, ElementC>
      >
    >
  >;

  constexpr static int ThreadCount = 128;
  constexpr static int kOutputAlignment = ThreadEpilogueOp::kCount;
  static constexpr bool IsActHasArgs = detail::sm100_act_has_arguments<typename FusionOp_::ActivationFn>::value;
  constexpr static bool IsSourceSupported = false; // Unsupported Residual C since it has poor performance

  constexpr static bool IsEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpiOp>::value;
  constexpr static bool IsPerChannelScalingSupported = detail::IsThreadEpilogueOpWithPerChannelScaling<ThreadEpiOp>::value;

  using ElementOutput = typename ThreadEpiOp::ElementOutput;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpiOp>::type;
  using ElementAlpha = ElementCompute;
  using ElementBeta = ElementCompute;
  using LoadPipeline = cutlass::PipelineTransactionAsync<0>; // 0 stage to disable smem alloc
  using LoadPipelineState = cutlass::PipelineState<0>;

  struct SharedStorage {
    struct TensorStorage : aligned_struct<128, _0> { } tensors;
    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Tmem shuffle
  using TmemStride = Layout<Shape <               _128, _16384>,
                            Stride< TMEM::DP<uint32_t>,     _1>>;
  using AccumalatorMapping  = Layout<Shape <Shape <_8,   _2,    _2,  _4>, Shape <_4, Shape < _16,    _2>>>,
                                     Stride<Stride<_4, _128, _4096, _32>, Stride<_1, Stride<_256, _8192>>>>;

  // Reuse accumulator when the 64-column TMEM is not enough
  static constexpr bool ReuseAccumulator = shape<2>(TmemShuffleLayout{}) >= Int<2>{};

  // Args without activation arguments
  template <class ActivationFn, class = void>
  struct EpilogueOpArgs {
    ElementAlpha alpha{0};
    ElementBeta beta{0};
    ElementAlpha const* alpha_ptr = nullptr;
    ElementBeta const* beta_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
  };

  // Args with activation arguments
  template <class ActivationFn>
  struct EpilogueOpArgs<ActivationFn, cute::enable_if_t<detail::sm100_act_has_arguments<ActivationFn>::value>> {
    ElementAlpha alpha{0};
    ElementBeta beta{0};
    ElementAlpha const* alpha_ptr = nullptr;
    ElementBeta const* beta_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    typename ActivationFn::Arguments activation{};
  };

  // Host side epilogue arguments
  struct Arguments {
    EpilogueOpArgs<typename FusionOp_::ActivationFn> thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Params without activation arguments
  template <class ActivationFn, class = void>
  struct ParamsType {
    typename ThreadEpiOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementBias const* ptr_Bias = nullptr;
  };

  // Params with activation arguments
  template <class ActivationFn>
  struct ParamsType<ActivationFn, cute::enable_if_t<detail::sm100_act_has_arguments<ActivationFn>::value>> {
    typename ThreadEpiOp::Params thread{};
    typename ActivationFn::Arguments activation{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementBias const* ptr_Bias = nullptr;
  };

  // Device side epilogue params
  using Params = ParamsType<typename FusionOp_::ActivationFn>;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {

    auto thread_op_args = [=]() {
      if constexpr (IsPerChannelScalingSupported) {
        return typename ThreadEpiOp::Params {args.thread.alpha_ptr, args.thread.beta_ptr};
      } else {
        return typename ThreadEpiOp::Params {args.thread.alpha, args.thread.beta};
      }
    }();

    if constexpr (IsActHasArgs) {
      return {
        thread_op_args,
        args.thread.activation,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        args.thread.bias_ptr
      };
    }
    else {
      return {
        thread_op_args,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        args.thread.bias_ptr
      };
    }
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    static constexpr int NumPacked = 4;
    bool implementable = true;

    auto problem_shape_transformed = cutlass::conv::detail::get_transformed_problem_shape_MNK_nq_2d_tiled(problem_shape);
    auto M = shape<0>(problem_shape_transformed);               // K
    implementable &= (M % NumPacked == 0);

    return implementable;
  }

  template<class TensorStorage>
  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params,
                     [[maybe_unused]] TensorStorage& shared_tensors)
                     : params(params), epilogue_op(params.thread) {
  };

  CUTLASS_DEVICE bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

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

  template<
    class EpiLoadPipeline,
    class EpiLoadPipelineState,
    class GTensorC,
    class CTensorC,
    class TileCoordMNKL
  >
  CUTLASS_DEVICE auto
  load(
      EpiLoadPipeline epi_load_pipeline,
      EpiLoadPipelineState epi_load_pipe_producer_state,
      cute::tuple<GTensorC,CTensorC> const& load_inputs,
      TileCoordMNKL tile_coord_mnkl,
      int p_pixels_start, int p_pixels_end,
      TensorStorage& shared_tensors)
  {}

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

    auto [gD_mnl, cD_mnl] = construct_C_D_tensors(params.ptr_D, params.dD, problem_shape, tile_shape);

    auto [ptr_alpha, ptr_beta] = [&]() {
      if constexpr (IsPerChannelScalingSupported) {
        return cute::make_tuple(params.thread.alpha_ptr, params.thread.beta_ptr);
      }
      else {
        return cute::make_tuple(&params.thread.alpha, &params.thread.beta);
      }}();
    auto [gAlpha_mnl, cAlpha_mnl] = construct_bias_alpha_beta_tensors(ptr_alpha,       problem_shape, tile_shape);
    auto [gBias_mnl,  cBias_mnl ] = construct_bias_alpha_beta_tensors(params.ptr_Bias, problem_shape, tile_shape);

    return cute::make_tuple(gD_mnl, cD_mnl, gBias_mnl, cBias_mnl, gAlpha_mnl, cAlpha_mnl);
  }

  template<
    class Params,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class EpiLoadPipeline,
    class EpiLoadPipelineState,
    class GTensorD,
    class CTensorD,
    class GTensorBias,
    class CTensorBias,
    class GTensorAlpha,
    class CTensorAlpha,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TensorStorage
  >
  CUTLASS_DEVICE auto
  store(
      Params const& params,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState accumulator_pipe_consumer_state,
      EpiLoadPipeline epi_load_pipeline,
      EpiLoadPipelineState epi_load_pipe_consumer_state,
      cute::tuple<GTensorD,CTensorD,GTensorBias,CTensorBias,GTensorAlpha,CTensorAlpha> const& store_inputs,
      TileCoordMNKL tile_coord_mnkl,
      int p_pixels_start, int p_pixels_end,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      [[maybe_unused]] TensorStorage& shared_tensors)
  {
    using namespace cute;
    using X = Underscore;

    static constexpr int NumPacked = 4;
    using PackedType = uint_bit_t<sizeof_bits_v<ElementD> * NumPacked>;

    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(AccLayout{}) == 4, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N,STAGE)");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    const int epi_thread_idx = threadIdx.x % ThreadCount;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    Tensor tC = accumulators(_,0,0,_);

    // Get output tensor (GMEM) for current CTA
    GTensorD mD     = get<0>(store_inputs);                                                //  (M,N,Z,P)
    // Get output coordinate tensor for current CTA
    CTensorD cD_mnl = get<1>(store_inputs);                                                //  (TILE_M,TILE_N,m,n,l,p)
    Tensor   cD     = cD_mnl(_,_,m_coord,n_coord,l_coord,_);                               //  (TILE_M,TILE_N,      p)

    // Get bias tensor (GMEM) without slicing
    GTensorBias mBias     = get<2>(store_inputs);                                          //  (M,N,Z,P)
    // Get bias coordinate tensor for current CTA
    CTensorBias cBias_mnl = get<3>(store_inputs);                                          //  (TILE_M,TILE_N,m,n,l,p)
    Tensor      cBias     = cBias_mnl(_,_,m_coord,n_coord,l_coord,_);                      //  (TILE_M,TILE_N,      p)

    // Get Alpha tensor (GMEM) for current CTA
    GTensorAlpha mAlpha     = get<4>(store_inputs);                                        //  (M,N,Z,P)
    // Get Alpha coordinate tensor for current CTA
    CTensorAlpha cAlpha_mnl = get<5>(store_inputs);                                        //  (TILE_M,TILE_N,m,n,l,p)
    Tensor       cAlpha     = cAlpha_mnl(_,_,m_coord,n_coord,l_coord,_);                   //  (TILE_M,TILE_N,      p)

    // Epilogue slicing outside loop p
    Tensor cD_slice = cD(_,_,_0{});                                                        //  (TILE_M,TILE_N)
    Tensor cBias_slice = cBias(_,_,_0{});                                                  //  (TILE_M,TILE_N)
    Tensor cAlpha_slice = cAlpha(_,_,_0{});                                                //  (TILE_M,TILE_N)

    //
    // Epilogue streaming loop
    //

    uint32_t stride_p = stride<3>(mD);
    auto [tC_shuffle, tD_offset, tC_shuffle_extra] = deduce_offset_predicate_tensors(tC);

    // Read Accumulator after math
    // TMEM_LOAD.32dp {$nv-internal-release}
    // (t)hread-partition for (t)mem to (r)egister copy (tTR_)
    Tensor tC_slice = make_tensor(tC(_,Int<0>{}).data(), get<0>(tC(_,Int<0>{}).layout()));
    auto tiled_t2r = make_tmem_copy(Copy_Atom<AccumulatorCopyOpT2R, ElementAccumulator>{}, tC_slice);
    auto thread_t2r = tiled_t2r.get_slice(epi_thread_idx);
    Tensor tTR_cAcc = thread_t2r.partition_D(cD_slice);
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_cAcc));

    // For data shuffle
    // TMEM_STORE.16dp {$nv-internal-release}
    // (t)hread-partition for (r)egister to (t)mem shuffle (tRT_S_)
    auto shuffle_r2t = make_tmem_copy(Copy_Atom<ShuffleCopyOpR2T, ElementAccumulator>{}, tC_shuffle(_,_,Int<0>{}));
    auto thread_shuffle_r2t = shuffle_r2t.get_slice(epi_thread_idx);
    Tensor tRT_S_gAcc = thread_shuffle_r2t.partition_S(tC_shuffle);
    Tensor tRT_S_tAcc = thread_shuffle_r2t.partition_D(tC_shuffle(_,_,Int<0>{}));
    Tensor tRT_S_rAcc = make_tensor<ElementAccumulator>(shape(tRT_S_gAcc));
    // TMEM_LOAD.32dp {$nv-internal-release}
    // (t)hread-partition for (t)mem to (r)egister shuffle (tTR_S_)
    auto shuffle_t2r = make_tmem_copy(Copy_Atom<ShuffleCopyOpT2R, ElementAccumulator>{}, tC_shuffle(_,_,Int<0>{}));
    auto thread_shuffle_t2r = shuffle_t2r.get_slice(epi_thread_idx);
    Tensor tTR_S_tAcc = thread_shuffle_t2r.partition_S(tC_shuffle(_,_,Int<0>{}));
    Tensor tTR_S_gAcc = thread_shuffle_t2r.partition_D(tC_shuffle(_,_,Int<0>{}));
    Tensor tTR_S_rAcc = make_tensor<ElementAccumulator>(shape(tTR_S_gAcc));
    // Output tensor
    Tensor tTR_S_rD = make_tensor<ElementD>(shape(tTR_S_gAcc));

    // For gmem offset
    // TMEM_STORE.32dp {$nv-internal-release}
    // (t)hread-partition for (r)egister to (t)mem copy (tRT_)
    auto offset_r2t = make_tmem_copy(Copy_Atom<OffsetCopyOpR2T, uint32_t>{}, tD_offset);
    auto thread_offset_r2t = offset_r2t.get_slice(epi_thread_idx);
    Tensor tRT_gOffset = thread_offset_r2t.partition_S(tD_offset);
    Tensor tRT_tOffset = thread_offset_r2t.partition_D(tD_offset);
    Tensor tRT_rOffset = make_tensor<uint32_t>(shape(tRT_gOffset));
    // TMEM_LOAD.32dp {$nv-internal-release}
    // (t)hread-partition for (t)mem to (r)egister copy (tTR)
    auto offset_t2r = make_tmem_copy(Copy_Atom<OffsetCopyOpT2R, uint32_t>{}, tD_offset(_,_,Int<0>{}));
    auto thread_offset_t2r = offset_t2r.get_slice(epi_thread_idx);
    Tensor tTR_tOffset = thread_offset_t2r.partition_S(tD_offset);
    Tensor tTR_gOffset = thread_offset_t2r.partition_D(tD_offset(_,_,Int<0>{}));
    Tensor tTR_rOffset = make_tensor<uint32_t>(shape(tTR_gOffset));

    // Pre-compute the address and coordinate
    auto cD_mapping = deduce_shuffled_D_tensors(tiled_t2r, shuffle_r2t, shuffle_t2r, cD_slice, tC_shuffle);
    auto cBias_mapping = deduce_shuffled_D_tensors(tiled_t2r, shuffle_r2t, shuffle_t2r, cBias_slice, tC_shuffle);
    auto cAlpha_mapping = deduce_shuffled_D_tensors(tiled_t2r, shuffle_r2t, shuffle_t2r, cAlpha_slice, tC_shuffle);

    auto cD_thr_slice     = cD_mapping(epi_thread_idx,_,_);
    auto cBias_thr_slice  = cBias_mapping(epi_thread_idx,_,_);
    auto cAlpha_thr_slice = cAlpha_mapping(epi_thread_idx,_,_);

    // Construct the tensor in epilogue op
    // rBias covers a epilogue tile
    Tensor tTR_S_rBias  = make_tensor<ElementBias>(shape(cBias_thr_slice));     // (T2R,T2R_M,T2R_N)
    // rAlpha covers a epilogue tile
    Tensor tTR_S_rAlpha = make_tensor<ElementAlpha>(shape(cAlpha_thr_slice));   // (T2R,T2R_M,T2R_N)

    // Vectorized fragment view
    Tensor tTR_S_rAcc_frg   = recast<typename ThreadEpiOp::FragmentAccumulator>(coalesce(tTR_S_rAcc));
    Tensor tTR_S_rAlpha_frg = recast<typename ThreadEpiOp::FragmentCompute>    (coalesce(tTR_S_rAlpha));
    Tensor tTR_S_rD_frg     = recast<typename ThreadEpiOp::FragmentOutput>     (coalesce(tTR_S_rD));

    ElementD *dst_base_ptr = recast_ptr<ElementD>(mD.data().get());
    auto mOffset = make_tensor(counting_iterator<uint32_t>(1), mD.layout());

    static constexpr int NumPreds = 32;
    bool preds[NumPreds];
    // Pre-compute the offset and the predicate
    for (int ii = 0; ii < size(tRT_rOffset); ++ii) {
      auto coord_mnzp = cD_thr_slice(ii * NumPacked);
      tRT_rOffset(ii) = mOffset(coord_mnzp);
      preds[ii] = elem_less(coord_mnzp, cD_thr_slice.layout().layout_a().shape());
    }

    for (int ii = 0; ii < size(tRT_rOffset); ++ii) {
      if (!preds[ii]) {
        tRT_rOffset(ii) = static_cast<uint32_t>(0x00u);
      }
    }

    // Store the offset and predicate
    copy(offset_r2t, tRT_rOffset, tRT_tOffset);
    cutlass::arch::fence_view_async_tmem_store();

    // load bias
    if constexpr (IsEpilogueBiasSupported) {
      if (params.ptr_Bias) {
        // vector load
        using PackedType = uint_bit_t<sizeof_bits_v<ElementBias> * NumPacked>;
        auto coord_mnzp = cBias_thr_slice(Int<0>{});
        auto bias_ptr = &mBias(coord_mnzp);
        bool pred = elem_less(coord_mnzp, cBias_thr_slice.layout().layout_a().shape());
        PackedType bias;
        cutlass::arch::global_load<PackedType, sizeof(PackedType)>(bias, (void*)bias_ptr, pred);
        // broadcast
        auto tDrBias_packed = recast<PackedType>(tTR_S_rBias);
        for (int ii = 0; ii < size(tDrBias_packed); ii++) {
          tDrBias_packed(ii) = bias;
        }
      }
    }

    // load valpha
    if constexpr (IsPerChannelScalingSupported) {
      // vector load
      using PackedType = uint_bit_t<sizeof_bits_v<ElementBias> * NumPacked>;
      auto coord_mnzp = cAlpha_thr_slice(Int<0>{});
      auto alpha_ptr = &mAlpha(coord_mnzp);
      bool pred = elem_less(coord_mnzp, cAlpha_thr_slice.layout().layout_a().shape());
      PackedType alpha;
      cutlass::arch::global_load<PackedType, sizeof(PackedType)>(alpha, (void*)alpha_ptr, pred);
      // broadcast
      auto tDrAlpha_packed = recast<PackedType>(tTR_S_rAlpha);
      for (int ii = 0; ii < size(tDrAlpha_packed); ii++) {
        tDrAlpha_packed(ii) = alpha;
      }
    }

    // Use uint32_t instead of `int` as the index type, to avoid extra instructions
    // due to conversion from int32_t to uint64_t (for address).
    CUTLASS_PRAGMA_NO_UNROLL
    for (uint32_t p = static_cast<uint32_t>(p_pixels_start); p < static_cast<uint32_t>(p_pixels_end); ++p) {

      uint32_t offset = p * stride_p;

      // Wait for mma warp to fill tmem buffer with accumulator results
      acc_pipeline.consumer_wait(accumulator_pipe_consumer_state);

      // TMEM_LOAD
      Tensor tC_slice = make_tensor(tC(_,accumulator_pipe_consumer_state.index()).data(), get<0>(tC(_,accumulator_pipe_consumer_state.index()).layout()));
      // Load acc element in TMEM to reg
      Tensor tTR_tAcc = thread_t2r.partition_S(tC_slice);
      // Load TMEM to Register
      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);
      cutlass::arch::fence_view_async_tmem_load();

      // Epilogue op
      copy(tTR_rAcc, tRT_S_rAcc);

      Tensor tRT_S_tAcc_extra = thread_shuffle_r2t.partition_D(flatten(tC_shuffle_extra(_,accumulator_pipe_consumer_state.index())));
      Tensor tTR_S_tAcc_extra = thread_shuffle_t2r.partition_S(flatten(tC_shuffle_extra(_,accumulator_pipe_consumer_state.index())));

      if constexpr (ReuseAccumulator) {
        copy(shuffle_r2t, tRT_S_rAcc(_,_,_,Int<1>{}), tRT_S_tAcc_extra);
        cutlass::arch::fence_view_async_tmem_store();
        copy(shuffle_t2r, tTR_S_tAcc_extra, tTR_S_rAcc);
        cutlass::arch::fence_view_async_tmem_load();
      }

      acc_pipeline.consumer_release(accumulator_pipe_consumer_state);
      ++accumulator_pipe_consumer_state;

      copy(shuffle_r2t, tRT_S_rAcc(_,_,_,Int<0>{}), tRT_S_tAcc);

      if constexpr (ReuseAccumulator) {
        // epilogue op
        if constexpr (IsEpilogueBiasSupported) {
          cutlass::NumericArrayConverter<typename ThreadEpiOp::ElementCompute, ElementBias,
            ThreadEpiOp::FragmentBias::kElements> bias_converter;
          Tensor tTR_S_rBias_frg = recast<typename ThreadEpiOp::FragmentBias>  (coalesce(tTR_S_rBias)); // (EPI_V x EPI_M x EPI_N)
          // D = alpha * accumulator + bias
          CUTLASS_PRAGMA_UNROLL
          for (int ii = 0; ii < size(tTR_S_rD_frg); ++ii) {
            typename ThreadEpiOp::FragmentCompute converted_bias = bias_converter(tTR_S_rBias_frg(ii));
            if constexpr (IsActHasArgs) {
              epilogue_op(tTR_S_rD_frg(ii), tTR_S_rD_frg(ii), tTR_S_rAcc_frg(ii), converted_bias, params.activation);
            }
            else {
              epilogue_op(tTR_S_rD_frg(ii), tTR_S_rD_frg(ii), tTR_S_rAcc_frg(ii), converted_bias);
            }
          }
        }
        else {
          if constexpr (IsPerChannelScalingSupported) {
            // D = valpha * accumulator
            CUTLASS_PRAGMA_UNROLL
            for (int ii = 0; ii < size(tTR_S_rD_frg); ++ii) {
              tTR_S_rD_frg(ii) = epilogue_op(tTR_S_rAcc_frg(ii), tTR_S_rAlpha_frg(ii));
            }
          }
          else {
            // D = alpha * accumulator
            CUTLASS_PRAGMA_UNROLL
            for (int ii = 0; ii < size(tTR_S_rD_frg); ++ii) {
              tTR_S_rD_frg(ii) = epilogue_op(tTR_S_rAcc_frg(ii));
            }
          }
        }
        auto tTR_S_rD_packed = recast<PackedType>(tTR_S_rD);
        copy(offset_t2r, tTR_tOffset(_,_,_,Int<1>{}), tTR_rOffset);

        // Pre-compute to reduce the ALU operation and reduce the register usage of predicate
        // offset = real_offset + 1
        // if (!xxx_pred) offset == 0
        // ...
        // if (offset != 0) store (src, dst_ptr + offset - 1)

        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < size(tTR_rOffset); ++ii) {
          PackedType src = tTR_S_rD_packed(ii);
          ElementD* dst = dst_base_ptr + tTR_rOffset(ii) + offset - 1;
          cutlass::arch::global_store<PackedType, sizeof(PackedType)>(src, (void*)dst, tTR_rOffset(ii) != 0);
        }
      }

      cutlass::arch::fence_view_async_tmem_store();
      // TMEM_LOAD data
      copy(shuffle_t2r, tTR_S_tAcc, tTR_S_rAcc);
      cutlass::arch::fence_view_async_tmem_load();
      // epilogue op
      if constexpr (IsEpilogueBiasSupported) {
        cutlass::NumericArrayConverter<typename ThreadEpiOp::ElementCompute, ElementBias,
          ThreadEpiOp::FragmentBias::kElements> bias_converter;
        Tensor tTR_S_rBias_frg = recast<typename ThreadEpiOp::FragmentBias>  (coalesce(tTR_S_rBias)); // (EPI_V x EPI_M x EPI_N)
        // D = alpha * accumulator + bias
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < size(tTR_S_rD_frg); ++ii) {
          typename ThreadEpiOp::FragmentCompute converted_bias = bias_converter(tTR_S_rBias_frg(ii));
          if constexpr (IsActHasArgs) {
            epilogue_op(tTR_S_rD_frg(ii), tTR_S_rD_frg(ii), tTR_S_rAcc_frg(ii), converted_bias, params.activation);
          }
          else {
            epilogue_op(tTR_S_rD_frg(ii), tTR_S_rD_frg(ii), tTR_S_rAcc_frg(ii), converted_bias);
          }
        }
      }
      else {
        if constexpr (IsPerChannelScalingSupported) {
          // D = valpha * accumulator
          CUTLASS_PRAGMA_UNROLL
          for (int ii = 0; ii < size(tTR_S_rD_frg); ++ii) {
            tTR_S_rD_frg(ii) = epilogue_op(tTR_S_rAcc_frg(ii), tTR_S_rAlpha_frg(ii));
          }
        }
        else {
          // D = alpha * accumulator
          CUTLASS_PRAGMA_UNROLL
          for (int ii = 0; ii < size(tTR_S_rD_frg); ++ii) {
            tTR_S_rD_frg(ii) = epilogue_op(tTR_S_rAcc_frg(ii));
          }
        }
      }
      // Recast to PackedType
      auto tTR_S_rD_packed = recast<PackedType>(tTR_S_rD);
      // Load the offset and predicate
      copy(offset_t2r, tTR_tOffset(_,_,_,Int<0>{}), tTR_rOffset);

      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < size(tTR_rOffset); ++ii) {
        PackedType src = tTR_S_rD_packed(ii);
        ElementD* dst = dst_base_ptr + tTR_rOffset(ii) + offset - 1;
        cutlass::arch::global_store<PackedType, sizeof(PackedType)>(src, (void*)dst, tTR_rOffset(ii) != 0);
      }
    }

    return cute::make_tuple(accumulator_pipe_consumer_state, epi_load_pipe_consumer_state);
  }

private:

  template <
    class Element,
    class Stride,
    class ProblemShape,
    class TileShape
  >
  CUTLASS_DEVICE auto
  construct_C_D_tensors(
    Element *ptr,
    Stride stride,
    ProblemShape const& problem_shape,
    TileShape const& tile_shape)
  {
    using namespace cute;
    using X = Underscore;

    auto M = shape<0>(problem_shape);               // K
    auto N = shape<1>(problem_shape);               // (Q,N)
    auto P = shape<2,1>(problem_shape);             // P
    auto Z = shape<3>(problem_shape);               // Z

    // Unpack problem shape and compute stride by ourselves
    auto [stride_tile_q,
          stride_tile_p,
          stride_tile_z,
          stride_tile_n] = get<0>(stride);          // (q, p, z, n)

    // Construct linearized mTensor then tile to gTensor_mnl
    Tensor mTensor = make_tensor(make_gmem_ptr(ptr), make_shape(M, N, Z, P),
      make_stride(_1{}, make_stride(stride_tile_q, stride_tile_n), stride_tile_z, stride_tile_p));         // (M,N,Z,P)
    Tensor mTensor_linear = make_identity_tensor(make_shape(M, size(N), Z, P));                            // (M,N,Z,P)
    Tensor mTensor_linearized = make_tensor(mTensor.data(),
      composition(mTensor.layout(), mTensor_linear(_0{}), mTensor_linear.layout()));                       // (M,N,Z,P)
    Tensor gTensor_mnl = local_tile(
      mTensor_linearized, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                  // (TILE_M,TILE_N,m,n,z,p)

    // Construct linearized cTensor (coordinate tensor for copy predication) then tile to cTensor_mnl
    auto mTensor_linearized_layout = mTensor_linearized.layout();                                          // (M,N,Z,P)
    auto cTensor_layout = make_composed_layout(
      make_layout(
        mTensor_linearized_layout.layout_a().shape(),
        make_basis_like(mTensor_linearized_layout.layout_a().shape())),
      mTensor_linearized_layout.offset(),
      mTensor_linearized_layout.layout_b());                                                               // (M,N,Z,P)
    Tensor cTensor = make_tensor(make_inttuple_iter(coprofile(cTensor_layout.layout_a())), cTensor_layout);
    auto cTensor_mnl = local_tile(
      cTensor, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                             // (TILE_M,TILE_N,m,n,z,p)

    return cute::make_tuple(mTensor, cTensor_mnl);
  }

  template <
    class Element,
    class ProblemShape,
    class TileShape
  >
  CUTLASS_DEVICE auto
  construct_bias_alpha_beta_tensors(
    Element *ptr,
    ProblemShape const& problem_shape,
    TileShape const& tile_shape)
  {
    using namespace cute;
    using X = Underscore;

    auto M = shape<0>(problem_shape);               // K
    auto N = shape<1>(problem_shape);               // (Q,N)
    auto P = shape<2,1>(problem_shape);             // P
    auto Z = shape<3>(problem_shape);               // Z

    // Construct linearized mTensor then tile to gTensor_mnl
    Tensor mTensor = make_tensor(make_gmem_ptr(ptr), make_shape(M, N, Z, P),
      make_stride(_1{}, repeat_like(N, _0{}), repeat_like(Z, _0{}), repeat_like(P, _0{})));                // (M,N,Z,P)
    static_assert(is_integral<decltype(M)>::value, "M should be single mode.");
    Tensor mTensor_linear = make_identity_tensor(make_shape(M, size(N), Z, P));                            // (M,N,Z,P)
    Tensor mTensor_linearized = make_tensor(mTensor.data(),
      composition(mTensor.layout(), mTensor_linear(_0{}), mTensor_linear.layout()));                       // (M,N,Z,P)
    Tensor gTensor_mnl = local_tile(
      mTensor_linearized, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                  // (TILE_M,TILE_N,m,n,z,p)

    // Construct linearized cTensor (coordinate tensor for copy predication) then tile to cTensor_mnl
    auto mTensor_linearized_layout = mTensor_linearized.layout();                                          // (M,N,Z,P)
    auto cTensor_layout = make_composed_layout(
      make_layout(
        mTensor_linearized_layout.layout_a().shape(),
        make_basis_like(mTensor_linearized_layout.layout_a().shape())),
      mTensor_linearized_layout.offset(),
      mTensor_linearized_layout.layout_b());                                                               // (M,N,Z,P)
    Tensor cTensor = make_tensor(make_inttuple_iter(coprofile(cTensor_layout.layout_a())), cTensor_layout);
    auto cTensor_mnl = local_tile(
      cTensor, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                             // (TILE_M,TILE_N,m,n,z,p)

    return cute::make_tuple(mTensor, cTensor_mnl);
  }

  template <
    class TiledCopyT2R,
    class ShuffleTiledCopyR2T,
    class ShuffleTiledCopyT2R,
    class CoordTensorD,
    class TmemTensorC
  >
  CUTLASS_DEVICE auto
  deduce_shuffled_D_tensors(
    TiledCopyT2R ldtm_t2r,
    [[maybe_unused]] ShuffleTiledCopyR2T shuffle_r2t,
    [[maybe_unused]] ShuffleTiledCopyT2R shuffle_t2r,
    CoordTensorD cD_slice,
    TmemTensorC  tC_slice)
  {
    auto tRT_shuffle   = ShuffleTiledCopyR2T::tidfrg_S(tC_slice);
    auto tTR_shuffle   = ShuffleTiledCopyT2R::tidfrg_D(tC_slice);
    auto cTR_unshuffle = TiledCopyT2R::tidfrg_D(cD_slice);

    // TV -> unshuffle TV
    auto acc_mapping = cute::composition(cute::left_inverse(tRT_shuffle.layout()), tTR_shuffle.layout());
    // TV -> unshuffle TV -> gmem
    auto cD_mapping = cTR_unshuffle.compose(acc_mapping);

    return cD_mapping;
  }

  template <
    class AccumulatorTensor
  >
  CUTLASS_DEVICE auto
  deduce_offset_predicate_tensors(
    AccumulatorTensor accumulator)
  {
    // Tmem management (total 128 columns)
    // The address will be aligned with 16
    // | metadata | offset |     shuffle     |
    // 0         32       64                128
    uint32_t tmem_base_ptr = accumulator.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulator);
    static constexpr uint32_t tmem_shuffle    = 64;
    static constexpr uint32_t tmem_offset     = 32;

    Tensor tC_shuffle   = make_tensor(make_tmem_ptr<uint32_t>(tmem_base_ptr + tmem_shuffle  ), TmemShuffleLayout{});
    Tensor tD_offset    = make_tensor(make_tmem_ptr<uint32_t>(tmem_base_ptr + tmem_offset   ), TmemOffsetLayout{});

    // Reuse the Accumulator TMEM once the 64 columns tmem is not enough
    auto tC_shuffle_extra = accumulator.compose(take<0,2>(shape(tC_shuffle)),_);

    return cute::make_tuple(tC_shuffle, tD_offset, tC_shuffle_extra);
  }

private:
  Params const & params;
  ThreadEpiOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
