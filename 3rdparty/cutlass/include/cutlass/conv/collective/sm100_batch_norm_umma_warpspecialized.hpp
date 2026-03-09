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
//
// {$nv-internal-release file}
//

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/trace.h"

#if (! defined(__CUDA_ARCH__)) && (CUTLASS_DEBUG_TRACE_LEVEL > 0)
#  include <sstream>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// Both DMA Load and MMA methods of this class must be run by a single thread that's picked by elect_one
template <
  conv::Operator ConvOp,
  int Stages,
  int BatchNormApplyStages,
  int SchedulerPipelineStages,
  int AccumulatorStages,
  int NumSpatialDims,
  class ClusterShape,    // Static cluster shape or dynamic (int, int, _1)
  class TileShape_,      // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class TileTraitsA_,
  class TileTraitsB_,
  class CopyAtomR2T_,
  class CopyAtomS2R_,
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_,
  int AlignmentAAlpha_,
  int AlignmentABias_>
struct CollectiveConv<
    MainloopSm100TmaBatchNormUmmaWarpSpecializedImplicitGemm<
        ConvOp, Stages, BatchNormApplyStages, SchedulerPipelineStages, AccumulatorStages, NumSpatialDims, CopyAtomR2T_, CopyAtomS2R_,
        ElementAAlpha_, ElementABias_, ActivationFunctor_, ClusterShape, AlignmentAAlpha_, AlignmentABias_>,
    TileShape_,
    ElementA_,
    ElementB_,
    TiledMma_,
    TileTraitsA_,
    TileTraitsB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm100TmaBatchNormUmmaWarpSpecializedImplicitGemm<
        ConvOp, Stages, BatchNormApplyStages, SchedulerPipelineStages, AccumulatorStages, NumSpatialDims, CopyAtomR2T_, CopyAtomS2R_,
        ElementAAlpha_, ElementABias_, ActivationFunctor_, ClusterShape, AlignmentAAlpha_, AlignmentABias_>;

  template <class T>
  using ActivationFunctor = ActivationFunctor_<T>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementAAlpha = ElementAAlpha_;
  using ElementABias = ElementABias_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = typename TileTraitsA_::GmemTiledCopy;
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;

  // Only fprop will use GmemTiledCopyAAlpha & GmemTiledCopyABias
  using GmemTiledCopyAAlpha = SM90_TMA_LOAD;
  using GmemTiledCopyABias = SM90_TMA_LOAD;

  using SmemLayoutAtomA = typename TileTraitsA_::SmemLayoutAtom;
  using TmemLayoutAtomA = typename TileTraitsA_::TmemLayoutAtom;
  using SmemLayoutAtomB = typename TileTraitsB_::SmemLayoutAtom;
  using CopyAtomR2T = CopyAtomR2T_;
  using CopyAtomS2R = CopyAtomS2R_;

  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int BatchNormApplyThreadCount = 128;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;

  // deduce the kernel-facing stride tuple types based on the dispatch policy (spatial dim, algo, etc.)
  using StrideA = cute::conditional_t<ConvOp == conv::Operator::kFprop,
                                      decltype(detail::sm100_dispatch_policy_to_stride_A<DispatchPolicy>()),
                                      decltype(detail::sm100_dispatch_policy_to_stride_B<DispatchPolicy>())>;
  using StrideB = cute::conditional_t<ConvOp == conv::Operator::kFprop,
                                      decltype(detail::sm100_dispatch_policy_to_stride_B<DispatchPolicy>()),
                                      decltype(detail::sm100_dispatch_policy_to_stride_A<DispatchPolicy>())>;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using TmaInternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, ElementA>;
  using TmaInternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, ElementB>;

  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // Determine MMA type: MMA_1SM vs MMA_2SM
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;

  using TmemWarpShape = cute::conditional_t<size<0>(TileShape{}) == 128 && size(AtomThrShapeMNK{}) == 2,
      Shape<_2,_2>, Shape<_4,_1>>;

  using TMALoadAPipeline = cutlass::PipelineTmaTransformAsync<
                             DispatchPolicy::Stages,
                             AtomThrShapeMNK>;
  using TMALoadAPipelineState = typename TMALoadAPipeline::PipelineState;

  using TMALoadBPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::Stages,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using TMALoadBPipelineState = typename TMALoadBPipeline::PipelineState;

  using BatchNormApplyPipeline = cutlass::PipelineUmmaConsumerAsync<
                                  DispatchPolicy::BatchNormApplyStages,
                                  AtomThrShapeMNK>;
  using BatchNormApplyPipelineState = typename BatchNormApplyPipeline::PipelineState;

  CUTE_STATIC_ASSERT_V(evenly_divides(shape<0>(TileShape{}), tile_size<0>(TiledMma{})) || (ConvOp == conv::Operator::kWgrad), "TileShape_M should be evenly divided by TiledMma_M");
  CUTE_STATIC_ASSERT_V(evenly_divides(shape<1>(TileShape{}), tile_size<1>(TiledMma{})), "TileShape_N should be evenly divided by TiledMma_N");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));
  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  static_assert((not IsDynamicCluster) or (size<2>(ClusterShape{}) == _1{}),
      "Dynamic ClusterShape must be (int, int, _1)");
  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(MmaShapeA_MK{}) * size<1>(MmaShapeA_MK{})) % size<0>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeA_MK{}) * size<2>(MmaShapeA_MK{})) % size<1>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");

  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(MmaShapeB_NK{}) * size<1>(MmaShapeB_NK{})) % size<0>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeB_NK{}) * size<2>(MmaShapeB_NK{})) % size<1>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));
  using TmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      TmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::BatchNormApplyStages>{}),
      Step<_2,_1,_3>{}));

  // Tile a MN-logical layout atom to an MMA Tile Shape ((MMA_M,MMA_N),M_MMAs,N_MMAs,...) with Strides of M are _0s.
  template <class LayoutAtom, class MMATileShape>
  CUTLASS_HOST_DEVICE static constexpr auto
  col_vector_tile_to_mma_shape(LayoutAtom const& atom, MMATileShape const& mma_tile_shape)
  {
    constexpr int R = decltype(rank(mma_tile_shape))::value;
    auto mn_shape = cute::tuple_cat(zip(shape<0>(mma_tile_shape), take<1,3>(mma_tile_shape)), take<3,R>(mma_tile_shape));
    constexpr int R1 = decltype(rank(tile_to_shape(atom, mn_shape)))::value;
    auto mn_tiled = coalesce(tile_to_shape(atom, mn_shape), tuple_repeat<R1>(Int<1>{}));           // (BLK_M,BLK_N,...)
    auto col_mn_tiled = make_layout(shape(mn_tiled), make_stride(_0{}, _1{}, size<1>(mn_tiled)));
    return tiled_divide(col_mn_tiled, product_each(shape<0>(mma_tile_shape)));     // ((MMA_M,MMA_N),M_MMAs,N_MMAs,...)
  }

  using SmemLayoutAtomAAlpha = cute::conditional_t<ConvOp == conv::Operator::kFprop, SmemLayoutAtomA, TmemLayoutAtomA>;
  using SmemLayoutAtomABias = cute::conditional_t<ConvOp == conv::Operator::kFprop, SmemLayoutAtomA, TmemLayoutAtomA>;
  using SmemLayoutAAlpha = decltype(col_vector_tile_to_mma_shape(SmemLayoutAtomAAlpha{}, append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{})));
  using SmemLayoutABias = decltype(col_vector_tile_to_mma_shape(SmemLayoutAtomABias{}, append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::tmem_frg_base, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must have A operand from TMEM and B operand from SMEM for this mainloop.");

  // Batchnorm kernels fprop and wgrad
  // are im2col for tensor A and tiled for tensor B.
  static_assert(
      ((cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_IM2COL_MULTICAST>)),
      "GmemTiledCopyA - invalid TMA copy atom specified.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        ((cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>))) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        ((cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD_MULTICAST>))),
      "GmemTiledCopyB -  invalid TMA copy atom specified.");

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;

      static constexpr auto SizeAAlpha = ConvOp == conv::Operator::kFprop ? cute::cosize_v<SmemLayoutAAlpha> : 0;
      static constexpr auto SizeABias = ConvOp == conv::Operator::kFprop ? cute::cosize_v<SmemLayoutABias> : 0;

      cute::array_aligned<ElementAAlpha, SizeAAlpha> smem_A_alpha;
      cute::array_aligned<ElementABias, SizeABias> smem_A_bias;
    } tensors;

    using TMALoadAPipelineStorage = typename TMALoadAPipeline::SharedStorage;
    using TMALoadBPipelineStorage = typename TMALoadBPipeline::SharedStorage;
    using BatchNormApplyPipelineStorage = typename BatchNormApplyPipeline::SharedStorage;

    TMALoadAPipelineStorage tma_load_a_pipeline;
    TMALoadBPipelineStorage tma_load_b_pipeline;
    BatchNormApplyPipelineStorage batch_norm_apply_pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using TMALoadAPipelineStorage = typename SharedStorage::TMALoadAPipelineStorage;
  using TMALoadBPipelineStorage = typename SharedStorage::TMALoadBPipelineStorage;
  using BatchNormApplyPipelineStorage = typename SharedStorage::BatchNormApplyPipelineStorage;

  static constexpr uint32_t TmaATransactionBytes =
    (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * size<2>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA)))
    + (ConvOp == conv::Operator::kWgrad ? 0 : size<0, 1>(SmemLayoutAAlpha{}) * size<2>(SmemLayoutAAlpha{}) * static_cast<uint32_t>(sizeof(ElementAAlpha))
    + size<0, 1>(SmemLayoutABias{}) * size<2>(SmemLayoutABias{}) * static_cast<uint32_t>(sizeof(ElementABias)));
  static constexpr uint32_t TmaBTransactionBytes =
    size(AtomThrShapeMNK{}) * (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * size<2>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB)));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    ElementB const* ptr_B{nullptr};
    ElementAAlpha const* ptr_A_alpha{nullptr};
    ElementABias const* ptr_A_bias{nullptr};
  };

  template <class ProblemShapeMNKL>
  static constexpr auto
  get_implemented_problem_shape_MNKL(ProblemShapeMNKL const& problem_shape_mnkl) {
    if constexpr (DispatchPolicy::ConvOp == conv::Operator::kFprop) {
      return problem_shape_mnkl;
    }
    else if constexpr (DispatchPolicy::ConvOp == conv::Operator::kWgrad) {
      // For bn_fusion wgrad, we need to swap M & N
      return select<1,0,2,3>(problem_shape_mnkl);
    }
    else { // ConvOp == conv::Operator::kDgrad
      static_assert(dependent_false<ProblemShapeMNKL>, "Unsupported ConvOp");
    }
  }

  template <class TensorA, class ClusterShapeVMNK>
  static constexpr auto
  get_tma_load_a_instance(TensorA const& tensor_a, ProblemShape const& problem_shape,
      ClusterShapeVMNK const& cluster_shape_vmnk) {
    // compute the upper and lower corners based on the conv padding
    auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
    auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
    auto lower_srt = detail::compute_lower_srt(problem_shape);
    cute::array<int32_t, NumSpatialDimensions> stride_srt{};
    for (int i = 0; i < NumSpatialDimensions; ++i) {
      stride_srt[i] = problem_shape.dilation[NumSpatialDimensions-1-i];
    }
    return make_im2col_tma_atom_A_sm100(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_shape_vmnk,
        shape(lower_corner_whd),
        shape(upper_corner_whd),
        cute::reverse(shape(problem_shape.lower_padding)),
        cute::reverse(shape(problem_shape.upper_padding)),
        cute::reverse(shape(problem_shape.traversal_stride)),
        shape(lower_srt),
        shape(stride_srt),
        // Enable nan-fill when OOB
        cute::TMA::DescriptorAuxParams{cute::TMA::OOBFill::CONSTANT, cute::TMA::L2Promotion::DISABLE});
  }

  template <class TensorB, class ClusterShapeVMNK>
  static constexpr auto
  get_tma_load_b_instance(TensorB const& tensor_b, ClusterShapeVMNK const& cluster_shape_vmnk) {
    return make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_shape_vmnk);
  }

  // Only needed by Fprop
  template <class TensorAAlpha, class ClusterShapeVMNK>
  static constexpr auto
  get_tma_load_a_alpha_instance(TensorAAlpha const& tensor_a_alpha, ClusterShapeVMNK const& cluster_shape_vmnk) {
    return make_tma_atom_A_sm100(
        GmemTiledCopyAAlpha{},
        tensor_a_alpha,
        SmemLayoutAAlpha{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_shape_vmnk);
  }

  // Only needed by Fprop
  template <class TensorABias, class ClusterShapeVMNK>
  static constexpr auto
  get_tma_load_a_bias_instance(TensorABias const& tensor_a_bias, ClusterShapeVMNK const& cluster_shape_vmnk) {
    return make_tma_atom_A_sm100(
        GmemTiledCopyABias{},
        tensor_a_bias,
        SmemLayoutABias{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_shape_vmnk);
  }

  // Device-side kernel params
  //
  // Arguments has the untransformed problem shape from the user.
  // Params will have the transformed problem shape.
  struct Params {
    using _Submode = decltype(take<0,NumTensorDimensions-1>(typename ProblemShape::TensorExtent{}));
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(
        make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
        make_tile(typename TiledMma::AtomThrID{})));

    // Assumption: StrideA is congruent with Problem_MK
    // Select TMA load type according to convolution operator.
    using TensorShapeA = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(make_shape(int32_t(0), _Submode{})),
        decltype(make_shape(_Submode{}, int32_t(0)))>;

    using TensorShapeB = decltype(repeat_like(StrideB{}, int32_t(0)));

    using TensorStrideAAlpha = cute::conditional_t<ConvOp == conv::Operator::kFprop,
        Stride<_0, _1>,
        Stride<_1, _0>>;

    using TensorStrideABias = TensorStrideAAlpha;

    using TensorShapeAAlpha = decltype(repeat_like(TensorStrideAAlpha{}, int32_t(0)));

    using TensorShapeABias = decltype(repeat_like(TensorStrideABias{}, int32_t(0)));

    using TMA_A = decltype(get_tma_load_a_instance(
            make_tensor(
                make_gmem_ptr(recast_ptr<TmaInternalElementA>(nullptr)),
                make_layout(TensorShapeA{}, StrideA{})),
            ProblemShape{},
            ClusterLayout_VMNK{}));

    using TMA_B = decltype(get_tma_load_b_instance(
        make_tensor(
            make_gmem_ptr(recast_ptr<TmaInternalElementB>(nullptr)),
            make_layout(TensorShapeB{}, StrideB{})),
        ClusterLayout_VMNK{}));

    struct FpropAuxParams {
      using TMA_A_alpha =
          decltype(get_tma_load_a_alpha_instance(
            make_tensor(
                make_gmem_ptr(recast_ptr<ElementAAlpha>(nullptr)),
                make_layout(TensorShapeAAlpha{}, TensorStrideAAlpha{})),
            ClusterLayout_VMNK{})
          );

      using TMA_A_bias =
          decltype(get_tma_load_a_bias_instance(
            make_tensor(
                make_gmem_ptr(recast_ptr<ElementABias>(nullptr)),
                make_layout(TensorShapeABias{}, TensorStrideABias())
            ),
            ClusterLayout_VMNK{})
          );

      TMA_A_alpha tma_load_a_alpha;
      TMA_A_bias tma_load_a_bias;
      TMA_A_alpha tma_load_a_alpha_fallback;
      TMA_A_bias tma_load_a_bias_fallback;
    };

    struct WgradAuxParams {
      ElementAAlpha const* ptr_A_alpha = nullptr;
      ElementABias const* ptr_A_bias = nullptr;
    };

    static_assert(ConvOp != conv::Operator::kDgrad, "No supports for dgrad kernels");

    using AuxParams = cute::conditional_t<ConvOp == conv::Operator::kFprop, FpropAuxParams, WgradAuxParams>;

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    AuxParams aux_params;

    dim3 cluster_shape_fallback;
  };

  CUTLASS_DEVICE
  CollectiveConv(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
      if constexpr (ConvOp == conv::Operator::kFprop) {
        observed_tma_load_a_alpha_ = is_fallback_cluster ? &params.aux_params.tma_load_a_alpha_fallback : &params.aux_params.tma_load_a_alpha;
        observed_tma_load_a_bias_ = is_fallback_cluster ? &params.aux_params.tma_load_a_bias_fallback : &params.aux_params.tma_load_a_bias;
      }
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
      if constexpr (ConvOp == conv::Operator::kFprop) {
        observed_tma_load_a_alpha_ = &params.aux_params.tma_load_a_alpha;
        observed_tma_load_a_bias_ = &params.aux_params.tma_load_a_bias;
      }
    }
  }

  //
  // Methods
  //

  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;

    // From the flat problem shape arrays of ConvProblemShape<N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents. For Wgrad, we use B in problem_shape (i.e., activation) as gemm A.
    auto shape_A_orig = [&] () {
                          if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kFprop) {
                            return problem_shape.get_shape_A();
                          }
                          else { // Wgrad
                            return problem_shape.get_shape_B();
                          }
                        } ();
    // B extents. For Wgrad, we use A in problem_shape (i.e., xformed activation) as gemm B.
    auto shape_B_orig = [&] () {
                          if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kFprop) {
                            return problem_shape.get_shape_B();
                          }
                          else { // Wgrad
                            return problem_shape.get_shape_A();
                          }
                        } ();

    // Fill inferred cute strides from flat stride arrays
    auto dA = make_cute_packed_stride(StrideA{}, ConvOp == conv::Operator::kFprop ? problem_shape.stride_A : problem_shape.stride_B, ConvOp);
    auto dB = make_cute_packed_stride(StrideB{}, ConvOp == conv::Operator::kFprop ? problem_shape.stride_B : problem_shape.stride_A, ConvOp);

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(make_gmem_ptr(ptr_A), make_layout(shape_A_orig, dA));
    Tensor tensor_b = make_tensor(make_gmem_ptr(ptr_B), make_layout(shape_B_orig, dB));
    Tensor tensor_a_alpha = make_tensor(make_gmem_ptr(args.ptr_A_alpha), make_layout(make_shape(size<0>(shape_A_orig), size<1>(shape_A_orig)), typename Params::TensorStrideAAlpha{}));
    Tensor tensor_a_bias = make_tensor(make_gmem_ptr(args.ptr_A_bias), make_layout(make_shape(size<0>(shape_A_orig), size<1>(shape_A_orig)), typename Params::TensorStrideABias{}));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    auto tma_load_a = get_tma_load_a_instance(tensor_a, problem_shape, cluster_layout_vmnk);
    auto tma_load_b = get_tma_load_b_instance(tensor_b, cluster_layout_vmnk);
    auto tma_load_a_fallback = get_tma_load_a_instance(tensor_a, problem_shape, cluster_layout_vmnk_fallback);
    auto tma_load_b_fallback = get_tma_load_b_instance(tensor_b, cluster_layout_vmnk_fallback);

    if constexpr (ConvOp == conv::Operator::kFprop) {
      auto tma_load_a_alpha = get_tma_load_a_alpha_instance(tensor_a_alpha, cluster_layout_vmnk);
      auto tma_load_a_bias = get_tma_load_a_bias_instance(tensor_a_bias, cluster_layout_vmnk);
      auto tma_load_a_alpha_fallback = get_tma_load_a_alpha_instance(tensor_a_alpha, cluster_layout_vmnk_fallback);
      auto tma_load_a_bias_fallback = get_tma_load_a_bias_instance(tensor_a_bias, cluster_layout_vmnk_fallback);
      return {
        tma_load_a,
        tma_load_b,
        tma_load_a_fallback,
        tma_load_b_fallback,
        {
          tma_load_a_alpha,
          tma_load_a_bias,
          tma_load_a_alpha_fallback,
          tma_load_a_bias_fallback
        },
        hw_info.cluster_shape_fallback,
      };
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return {
        tma_load_a,
        tma_load_b,
        tma_load_a_fallback,
        tma_load_b_fallback,
        {
          args.ptr_A_alpha,
          args.ptr_A_bias
        },
        hw_info.cluster_shape_fallback,
      };
    }
    else { // ConvOp == conv::Operator::kDgrad
      static_assert(dependent_false<Arguments>, "Unsupported ConvOp");
    }
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      Arguments const& args) {
    // Activation and Filter channel mode extents much match
    bool implementable = true;
    // channel mode is major
    {
      const bool check = problem_shape.stride_A[NumTensorDimensions-1] == 1;
#if (! defined(__CUDA_ARCH__)) && (CUTLASS_DEBUG_TRACE_LEVEL > 0)
      if (not check) {
        const auto offending_stride =
          problem_shape.stride_A[NumTensorDimensions-1];
        std::ostringstream os;
        os << "CollectiveConv::can_implement: "
          "problem_shape.stride_A[NumTensorDimensions-1 = "
          << (NumTensorDimensions-1) << "] = "
          << offending_stride << " != 1";
        CUTLASS_TRACE_HOST( os.str() );
      }
#endif
      implementable &= check;
    }

    {
      const bool check = problem_shape.stride_B[NumTensorDimensions-1] == 1;
#if (! defined(__CUDA_ARCH__)) && (CUTLASS_DEBUG_TRACE_LEVEL > 0)
      if (not check) {
        const auto offending_stride =
          problem_shape.stride_B[NumTensorDimensions-1];
        std::ostringstream os;
        os << "CollectiveConv::can_implement: "
          "problem_shape.stride_B[NumTensorDimensions-1 = "
          << (NumTensorDimensions-1) << "] = "
          << offending_stride << " != 1\n";
        CUTLASS_TRACE_HOST( os.str() );
      }
#endif
      implementable &= check;
    }

    {
      const auto & traversal_stride  = problem_shape.traversal_stride;
      for (auto stride: traversal_stride) {
       implementable &= (stride >= 1 && stride <= 8);
      }
    }

    if constexpr (ConvOp == conv::Operator::kDgrad) {
      // batch_norm kernel only support fprop & wgrad
      return false;
    }

    constexpr int tma_alignment_bits = 128;
    // A extents. For Wgrad, we use B in problem_shape (i.e., activation) as gemm A.
    auto shape_A_orig = [&] () {
                          if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kFprop) {
                            return problem_shape.get_shape_A();
                          } else { // Wgrad
                            return problem_shape.get_shape_B();
                          }
                        } ();
    // B extents. For Wgrad, we use A in problem_shape (i.e., xformed activation) as gemm B.
    auto shape_B_orig = [&] () {
                          if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kFprop) {
                            return problem_shape.get_shape_B();
                          } else { // Wgrad
                            return problem_shape.get_shape_A();
                          }
                        } ();

    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    {
      const bool check = cutlass::detail::check_alignment<min_tma_aligned_elements_A>(shape_A_orig, StrideA{});
      if (not check) {
        CUTLASS_TRACE_HOST("A shape and/or strides have alignment issue.");
      }
      implementable &= check;
    }

    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    {
      const bool check = cutlass::detail::check_alignment<min_tma_aligned_elements_B>(shape_B_orig, StrideB{});
      if (not check) {
        CUTLASS_TRACE_HOST("B shape and/or strides have alignment issue.");
      }
      implementable &= check;
    }

    if (not implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
      return false;
    }

    // Check valid padding values for TMA_LOAD_IM2COL
    constexpr int padding_limit = (ProblemShape::RankS == 1) ? 65536 : (ProblemShape::RankS == 2 ? 256 : 16);
    for (int i = 0; i < problem_shape.RankS; ++i) {
      implementable = implementable && problem_shape.lower_padding[i] <= padding_limit && problem_shape.lower_padding[i] >= 0;
      implementable = implementable && problem_shape.upper_padding[i] <= padding_limit && problem_shape.upper_padding[i] >= 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Padding values don't meet requirements for TMA LOAD IM2COL.\n");
      return false;
    }

    constexpr bool is_im2col_A = detail::is_im2col_load<GmemTiledCopyA>::value;
    constexpr bool is_im2col_B = detail::is_im2col_load<GmemTiledCopyB>::value;

    // Check valid corner values for TMA_LOAD_IM2COL, signed int ranging from [-corner_limit, corner_limit - 1]
    if (is_im2col_A || is_im2col_B) {
      constexpr int32_t corner_limit = 1 << (16 / NumSpatialDimensions - 1);
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      for (int i = 0; i < problem_shape.RankS; ++i) {
        implementable = implementable && lower_corner_whd[i] >= -corner_limit && lower_corner_whd[i] <= (corner_limit - 1);
      }
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      for (int i = 0; i < problem_shape.RankS; ++i) {
        implementable = implementable && upper_corner_whd[i] >= -corner_limit && upper_corner_whd[i] <= (corner_limit - 1);
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Padding values don't meet requirements for TMA LOAD IM2COL.\n");
        return false;
      }
    }

    // Check valid filter offsets for TMA_LOAD_IM2COL, unsigned int ranging from [0, offset_limit]
    if (is_im2col_A || is_im2col_B) {
      constexpr int32_t offset_limit = (1 << (16 / NumSpatialDimensions)) - 1;
      auto flt_data = (ConvOp == conv::Operator::kWgrad) ? problem_shape.shape_C : problem_shape.shape_B;
      for (int i = 0; i < problem_shape.RankS; ++i) {
        // flt_data array contains [K, T, R, S, C], so pure filter [T, R, S] starts from the second position in the array
        implementable = implementable && ((flt_data[i+1] - 1) * problem_shape.dilation[i] >= 0)
                                      && ((flt_data[i+1] - 1) * problem_shape.dilation[i] <= offset_limit);
      }

      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: tensor coordinate offset values don't meet requirements for TMA LOAD IM2COL.\n");
        return false;
      }
    }

    // The extents of linearized problem shape should be int32_t type(maximum is 2^31-1).
    if (is_im2col_A || is_im2col_B) {
      auto [M, N, K, L] = get_implemented_problem_shape_MNKL(cutlass::conv::detail::get_transformed_problem_shape_MNKL(problem_shape));
      auto to_64b = [](auto S) { return transform_leaf(S, [](auto s) { return static_cast<int64_t>(s); }); };
      if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kWgrad) {
        implementable &= (cute::product(to_64b(K)) <= cutlass::platform::numeric_limits<int32_t>::max());
      }
      else {
        implementable &= (cute::product(to_64b(M)) <= cutlass::platform::numeric_limits<int32_t>::max());
      }
      if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: the linearized extent exceeds the maximum number.\n");
        return false;
      }
    }

    return true;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
    if constexpr (ConvOp == conv::Operator::kFprop) {
      cute::prefetch_tma_descriptor(observed_tma_load_a_alpha_->get_tma_descriptor());
      cute::prefetch_tma_descriptor(observed_tma_load_a_bias_->get_tma_descriptor());
    }
  }

  // {$nv-release-never begin}
  CUTLASS_DEVICE static void
  prefetch_tma_descriptors(Params const& params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == params.cluster_shape_fallback.x && cs.y == params.cluster_shape_fallback.y);
      if (is_fallback_cluster) {
        cute::prefetch_tma_descriptor(params.tma_load_a_fallback.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b_fallback.get_tma_descriptor());
        if constexpr (ConvOp == conv::Operator::kFprop) {
          cute::prefetch_tma_descriptor(params.aux_params.tma_load_a_alpha_fallback.get_tma_descriptor());
          cute::prefetch_tma_descriptor(params.aux_params.tma_load_a_bias_fallback.get_tma_descriptor());
        }
      }
      else {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
        if constexpr (ConvOp == conv::Operator::kFprop) {
          cute::prefetch_tma_descriptor(params.aux_params.tma_load_a_alpha.get_tma_descriptor());
          cute::prefetch_tma_descriptor(params.aux_params.tma_load_a_bias.get_tma_descriptor());
        }
      }
    }
    else {
      cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
      if constexpr (ConvOp == conv::Operator::kFprop) {
        cute::prefetch_tma_descriptor(params.aux_params.tma_load_a_alpha.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.aux_params.tma_load_a_bias.get_tma_descriptor());
      }
    }
  }
  // {$nv-release-never end}

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE static auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)
    return acc_shape;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class InputTuples,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
      Params const& params,
      TMALoadAPipeline tma_load_a_pipeline,
      TMALoadAPipelineState tma_load_a_pipe_producer_state,
      TMALoadBPipeline tma_load_b_pipeline,
      TMALoadBPipelineState tma_load_b_pipe_producer_state,
      InputTuples const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {
    [[maybe_unused]] auto& [unused_gA, unused_gB,
          tAgA_mk, tBgB_nk, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b,
          aux_inputs] = load_inputs;
    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mk(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _);
    Tensor tBgB = tBgB_nk(_, get<1>(cta_coord_mnkl), _);

    auto tma_load_a_barrier_token = tma_load_a_pipeline.producer_try_acquire(tma_load_a_pipe_producer_state);
    auto tma_load_b_barrier_token = tma_load_b_pipeline.producer_try_acquire(tma_load_b_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK tma_load_a_pipe_producer_state & tma_load_b_pipe_producer_state for _writing_
      tma_load_a_pipeline.producer_acquire(tma_load_a_pipe_producer_state, tma_load_a_barrier_token);
      tma_load_b_pipeline.producer_acquire(tma_load_b_pipe_producer_state, tma_load_b_barrier_token);
      using TMALoadABarrierType = typename TMALoadAPipeline::ProducerBarrierType;
      TMALoadABarrierType* tma_a_barrier = tma_load_a_pipeline.producer_get_barrier(tma_load_a_pipe_producer_state);
      using TMALoadBBarrierType = typename TMALoadBPipeline::ProducerBarrierType;
      TMALoadBBarrierType* tma_b_barrier = tma_load_b_pipeline.producer_get_barrier(tma_load_b_pipe_producer_state);

      int write_a_stage = tma_load_a_pipe_producer_state.index();
      int write_b_stage = tma_load_b_pipe_producer_state.index();
      ++tma_load_a_pipe_producer_state;
      ++tma_load_b_pipe_producer_state;

      tma_load_a_barrier_token = tma_load_a_pipeline.producer_try_acquire(tma_load_a_pipe_producer_state);
      tma_load_b_barrier_token = tma_load_b_pipeline.producer_try_acquire(tma_load_b_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_a_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_a_stage));
      }
      if constexpr (ConvOp == conv::Operator::kFprop) {
        auto& [tAgA_alpha_mk, tAgA_bias_mk, tAsA_alpha, tAsA_bias] = aux_inputs;
        Tensor tAgA_alpha = tAgA_alpha_mk(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _);
        Tensor tAgA_bias = tAgA_bias_mk(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _);
        if (cute::elect_one_sync()) {
          copy(observed_tma_load_a_alpha_->with(*tma_a_barrier), tAgA_alpha(_,get<0>(*k_tile_iter)), tAsA_alpha(_,write_a_stage));
          copy(observed_tma_load_a_bias_->with(*tma_a_barrier), tAgA_bias(_,get<0>(*k_tile_iter)), tAsA_bias(_,write_a_stage));
        }
      }
      if (cute::elect_one_sync()) {
        copy(observed_tma_load_b_->with(*tma_b_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_b_stage));
      }

      --k_tile_count;
      ++k_tile_iter;
    }
    return cute::make_tuple(tma_load_a_pipe_producer_state, tma_load_b_pipe_producer_state, k_tile_iter);
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mk - The tiled tma tensor for input A
  /// gB_nk - The tiled tma tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mk = observed_tma_load_a_->get_tma_tensor(make_shape(M, size(K)));
    Tensor mB_nk = observed_tma_load_b_->get_tma_tensor(make_shape(N, K));

    // Tile the tensors and defer the slice
    Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M, BLK_K, m, k)
    Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});         // (BLK_N, BLK_K, n, k)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mk = cta_mma.partition_A(gA_mk);                                           // (MMA, MMA_M, MMA_K, m, k)
    Tensor tCgB_nk = cta_mma.partition_B(gB_nk);                                           // (MMA, MMA_N, MMA_K, n, k)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});     // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});     // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mk, tAsA] = tma_partition(*observed_tma_load_a_,
        get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
        group_modes<0,3>(sA), group_modes<0,3>(tCgA_mk));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nk, tBsB] = tma_partition(*observed_tma_load_b_,
        get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
        group_modes<0,3>(sB), group_modes<0,3>(tCgB_nk));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    if constexpr (ConvOp == conv::Operator::kFprop) {
      Tensor mA_alpha_mk = observed_tma_load_a_alpha_->get_tma_tensor(make_shape(M, get<0>(K)));
      Tensor mA_bias_mk = observed_tma_load_a_bias_->get_tma_tensor(make_shape(M, get<0>(K)));;

      Tensor gA_alpha_mk = local_tile(mA_alpha_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
      Tensor gA_bias_mk = local_tile(mA_bias_mk,   TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

      Tensor tCgA_alpha_mk = cta_mma.partition_A(gA_alpha_mk);
      Tensor tCgA_bias_mk = cta_mma.partition_A(gA_bias_mk);

      Tensor sA_alpha = make_tensor(make_smem_ptr(shared_tensors.smem_A_alpha.begin()), SmemLayoutAAlpha{});
      Tensor sA_bias = make_tensor(make_smem_ptr(shared_tensors.smem_A_bias.begin()), SmemLayoutABias{});

      // For alpha & bias, no needs for multicast
      auto [tAgA_alpha_mk, tAsA_alpha] = tma_partition(*observed_tma_load_a_alpha_,
          0, make_layout(size<2>(cta_layout_vmnk)),
          group_modes<0,3>(sA_alpha), group_modes<0,3>(tCgA_alpha_mk));
      auto [tAgA_bias_mk, tAsA_bias] = tma_partition(*observed_tma_load_a_bias_,
          0, make_layout(size<2>(cta_layout_vmnk)),
          group_modes<0,3>(sA_bias), group_modes<0,3>(tCgA_bias_mk));

      return cute::make_tuple(
        gA_mk, gB_nk,                                        // for scheduler
        tAgA_mk, tBgB_nk, tAsA, tBsB,                        // for input tensor values
        mcast_mask_a, mcast_mask_b,                          // multicast masks
        cute::make_tuple(
          tAgA_alpha_mk, tAgA_bias_mk, tAsA_alpha, tAsA_bias // aux tensors for bn_apply
        )
      );
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return cute::make_tuple(
          gA_mk, gB_nk,                        // for scheduler
          tAgA_mk, tBgB_nk, tAsA, tBsB,        // for input tensor values
          mcast_mask_a, mcast_mask_b,          // multicast masks
          cute::make_tuple());
    }
    else { // Dgrad
      static_assert(dependent_false<ProblemShape_MNKL>, "Unsupported ConvOp");
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_tail(TMALoadAPipeline tma_load_a_pipeline, TMALoadAPipelineState tma_load_a_pipe_producer_state,
            TMALoadBPipeline tma_load_b_pipeline, TMALoadBPipelineState tma_load_b_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    tma_load_a_pipeline.producer_tail(tma_load_a_pipe_producer_state);
    tma_load_b_pipeline.producer_tail(tma_load_b_pipe_producer_state);
  }

  template <
    class ProblemShape,
    class FrgEngine, class FrgLayout
  >
  CUTLASS_DEVICE auto
  batch_norm_apply_init(
    ProblemShape const& problem_shape,
    Params const& params,
    cute::Tensor<FrgEngine, FrgLayout>& accumulators,
    TensorStorage& shared_tensors
  ) {
    auto get_tensor = [&] (auto tensor) constexpr {
      // For M=128 with 2CTA MMA atoms, the TMEM tensor for A has a duplicated allocation.
      // Instead of allocation a 64x16 TMEM tensor, we have a 128x16 allocation
      // See: TmemAllocMode::Duplicated.
      // M Tile for wgrad is multimodal
      if constexpr (decltype(size<0>(TileShape{}) == Int<128>{} && shape<0>(typename TiledMma_::ThrLayoutVMNK{}) == Int<2>{})::value) {
        return make_tensor(tensor.data(),
                           logical_product(tensor.layout(),
                                           make_tile(make_tile(Layout<_2,_0>{},_),_,_,_)));      // ((128,16),m,k,PIPE)
      }
      else {
        return tensor;
      }
    };
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    TiledMma tiled_mma;
    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = tiled_mma.make_fragment_A(shape(TmemLayoutA{}));                         // (MMA,MMA_M,MMA_K,PIPE_BN)

    // HACK: Change the starting address of A in TMEM -- use the last TMEM COL of accumulators
    tCrA.data() = accumulators.data().get() + size<0, 1>(accumulators.layout()) * size<2>(accumulators.layout())
        * size<3>(accumulators.layout()) * sizeof(ElementAccumulator) / sizeof(float) / size<1>(TmemWarpShape{});

    int thread_idx = threadIdx.x % BatchNormApplyThreadCount;

    auto tiled_sttm = make_tmem_copy(CopyAtomR2T{}, tCrA(_,_,_,0));

    auto thr_sttm = tiled_sttm.get_slice(thread_idx);
    auto tiled_s2r = make_tiled_copy_S(Copy_Atom<CopyAtomS2R, ElementA>{}, tiled_sttm);

    auto thr_s2r = tiled_s2r.get_slice(thread_idx);

    Tensor tAsA_s2r = thr_s2r.partition_S(get_tensor(sA));                                 // (TMEM_STORE,TMEM_STORE_M,TMEM_STORE_K,PIPE)

    Tensor tAtA_r2t = thr_sttm.partition_D(tCrA);                                       // (TMEM_STORE,TMEM_STORE_M,TMEM_STORE_K,PIPE_BN)
    Tensor tArA_r2t = make_tensor<ElementA>(shape(thr_sttm.partition_S(get_tensor(sA))(_,_,_,_,0)));

    //
    // Load of A_alpha and A_bias specifics
    //

    if constexpr (ConvOp == conv::Operator::kFprop) {
      // Fprop smem -> rmem
      auto col_tv_layout = [&] (auto layout) constexpr {
        static_assert(cute::is_same_v<CopyAtomR2T, SM100_TMEM_STORE_16dp128b8x>, "CopyAtomR2T - invalid R2T copy atom specified.");
        auto ori_shape = shape(layout);
        // To present duplicate loading, this function customizes the tv_layout for these column vectors
        auto col_shape = replace<1>(ori_shape, replace<0>(get<1>(ori_shape), replace<1>(get<1,0>(ori_shape),_1{})));
        return make_layout(col_shape, stride(layout));
      }(tiled_sttm.get_layoutS_TV());
      auto tiled_col_s2r = make_tiled_copy_impl(Copy_Atom<CopyAtomS2R, ElementA>{}, col_tv_layout, typename decltype(tiled_sttm)::Tiler_MN{});
      auto thr_col_s2r = tiled_col_s2r.get_slice(thread_idx);

      Tensor sA_alpha = make_tensor(make_smem_ptr(shared_tensors.smem_A_alpha.data()), SmemLayoutAAlpha{});
      Tensor sA_bias = make_tensor(make_smem_ptr(shared_tensors.smem_A_bias.data()), SmemLayoutABias{});
      Tensor tAsA_alpha_s2r = thr_col_s2r.partition_S(get_tensor(sA_alpha));
      Tensor tAsA_bias_s2r = thr_col_s2r.partition_S(get_tensor(sA_bias));
      auto s2r_layout = make_layout_like(thr_col_s2r.partition_D(get_tensor(sA_alpha))(_,_,_,_,0).layout());

      Tensor tArA_alpha = make_tensor_like(thr_sttm.partition_S(get_tensor(sA_alpha))(_,_,_,_,0));
      Tensor tArA_bias = make_tensor_like(thr_sttm.partition_S(get_tensor(sA_bias))(_,_,_,_,0));

      // For fprop, TMA will check OOB. No need to pass tArA_cVec. To maintain consistency, pass X instead.
      auto aux_tuples = cute::make_tuple(tiled_col_s2r, s2r_layout, tAsA_alpha_s2r, tAsA_bias_s2r, tArA_alpha, tArA_bias, X{});
      return cute::make_tuple(tiled_s2r, thr_s2r, tAsA_s2r, tiled_sttm, tAtA_r2t, tArA_r2t, aux_tuples);
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) { // ConvOp == conv::Operator::kWgrad
      // Wgrad gmem -> rmem
      auto M = get<0>(problem_shape);
      auto K = size<2>(problem_shape);

      Tensor mA_alpha = make_tensor(params.aux_params.ptr_A_alpha, make_shape(size<0>(M), K), make_stride(_1{}, _0{}));
      Tensor mA_bias = make_tensor(params.aux_params.ptr_A_bias, make_shape(size<0>(M), K), make_stride(_1{}, _0{}));

      auto tile_coord = make_coord(_,_,_);
      Tensor gA_alpha_m = local_tile(mA_alpha, CtaShape_MNK{}, tile_coord, Step<_1, X,_1>{});    // (TILE_M,TILE_K,m,k)
      Tensor gA_bias_m = local_tile(mA_bias, CtaShape_MNK{}, tile_coord, Step<_1, X,_1>{});      // (TILE_M,TILE_K,m,k)

      Tensor gA_alpha_m_ = tiled_divide(gA_alpha_m, get<0>(shape(TmemLayoutA{})));
      Tensor gA_bias_m_ = tiled_divide(gA_bias_m, get<0>(shape(TmemLayoutA{})));
      Tensor tAgA_alpha_g2r = thr_sttm.partition_S(get_tensor(gA_alpha_m_));
      Tensor tAgA_bias_g2r = thr_sttm.partition_S(get_tensor(gA_bias_m_));

      Tensor tArA_alpha = make_tensor_like(tAgA_alpha_g2r(_,_,_,_,0,0));
      Tensor tArA_bias = make_tensor_like(tAgA_bias_g2r(_,_,_,_,0,0));

      Tensor cVec = make_coord_tensor(get_tensor(gA_alpha_m_).layout());
      Tensor tArA_cVec = thr_sttm.partition_S(cVec)(_,_,_,_,0,0);

      auto aux_tuples = cute::make_tuple(tiled_s2r, thr_s2r, tAgA_alpha_g2r, tAgA_bias_g2r, tArA_alpha, tArA_bias, tArA_cVec);

      return cute::make_tuple(tiled_s2r, thr_s2r, tAsA_s2r, tiled_sttm, tAtA_r2t, tArA_r2t, aux_tuples);
    }
    else { // ConvOp == conv::Operator::kDgrad
      static_assert(dependent_false<FrgLayout>, "Unsupported ConvOp");
    }
  }

  template <
    class ProblemShape,
    class CtaCoordMNKL, class InputsTuple
  >
  CUTLASS_DEVICE void
  batch_norm_load(
    [[maybe_unused]] ProblemShape const& problem_shape,
    [[maybe_unused]] CtaCoordMNKL cta_coord_mnkl,
    [[maybe_unused]] InputsTuple&& batch_norm_apply_inputs
  ) {
    if constexpr (ConvOp == conv::Operator::kWgrad) {
      auto [M, N, K, L] = problem_shape;
      auto [m, n, k, l] = cta_coord_mnkl;
      // For Wgrad, we directly load alpha & bias from gmem to rmem
      [[maybe_unused]] auto& [unused_tiled_s2r, unused_thr_s2r, unused_tAsA_s2r, unused_tiled_sttm, unused_tAtA_r2t, unused_tArA_r2t, aux_tuples] = batch_norm_apply_inputs;
      [[maybe_unused]] auto& [unused_tiled_col_s2r, unused_s2r_layout, tAgA_alpha_g2r, tAgA_bias_g2r, tArA_alpha, tArA_bias, tArA_cVec] = aux_tuples;
      Tensor tAgA_alpha_g2r_flt = filter_zeros(tAgA_alpha_g2r(_,_,_,_,get<0>(m),_0{}));
      Tensor tAgA_bias_g2r_flt = filter_zeros(tAgA_bias_g2r(_,_,_,_,get<0>(m),_0{}));
      Tensor tArA_alpha_flt = filter_zeros(tArA_alpha);
      Tensor tArA_bias_flt = filter_zeros(tArA_bias);
      Tensor tArA_cVec_flt = filter_zeros(tArA_cVec, tAgA_alpha_g2r(_,_,_,_,get<0>(m),_0{}).stride());

      Tensor mVec_crd = make_identity_tensor(make_shape(get<0>(M),size(K)));
      Tensor cVec = local_tile(mVec_crd, select<0,2>(CtaShape_MNK{}), make_coord(get<0>(m),_0{}));

      auto residue_vec = make_coord(get<0>(M),size(K)) - cVec(_0{});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tAgA_alpha_g2r_flt); ++i) {
        if (elem_less(make_coord(tArA_cVec_flt(i),_0{}), residue_vec)) {
          tArA_alpha_flt(i) = tAgA_alpha_g2r_flt(i);
          tArA_bias_flt(i) = tAgA_bias_g2r_flt(i);
        }
      }
    }
    else {
      // For other ConvOp, we load alpha & bias by TMA in the main_load warp
      return;
    }
  }

  template <
    class InputsTuple
  >
  CUTLASS_DEVICE auto
  batch_norm_apply(
    TMALoadAPipeline tma_load_A_pipeline,
    TMALoadAPipelineState tma_load_A_pipe_consumer_state,
    BatchNormApplyPipeline batch_norm_apply_pipeline,
    BatchNormApplyPipelineState batch_norm_apply_pipe_producer_state,
    InputsTuple&& batch_norm_apply_inputs,
    int k_tile_count
  ) {
    auto& [tiled_s2r, thr_s2r, tAsA_s2r, tiled_sttm, tAtA_r2t, tArA_r2t, aux_tuples] = batch_norm_apply_inputs;
    [[maybe_unused]] auto& [tiled_col_s2r, s2r_layout, tAsA_alpha_s2r, tAsA_bias_s2r, tArA_alpha, tArA_bias, tArA_cVec] = aux_tuples;

    uint32_t skip_wait = k_tile_count <= 0;
    auto tma_load_A_barrier_token = tma_load_A_pipeline.consumer_try_wait(tma_load_A_pipe_consumer_state, skip_wait);
    auto batch_norm_apply_barrier_token = batch_norm_apply_pipeline.producer_try_acquire(batch_norm_apply_pipe_producer_state, skip_wait);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      tma_load_A_pipeline.consumer_wait(tma_load_A_pipe_consumer_state, tma_load_A_barrier_token);
      int load_A_stage = tma_load_A_pipe_consumer_state.index();

      Tensor tArA_s2r = thr_s2r.retile_D(tArA_r2t);
      copy(tiled_s2r, tAsA_s2r(_,_,_,_,load_A_stage), tArA_s2r);                                       // smem -> rmem

      if constexpr (ConvOp == conv::Operator::kFprop) {
        // Load A_alpha and A_bias from smem to rmem
        Tensor tArA_alpha_flt = make_tensor(tArA_alpha.data(), s2r_layout);
        Tensor tArA_bias_flt = make_tensor(tArA_bias.data(), s2r_layout);

        copy(tiled_col_s2r, tAsA_alpha_s2r(_,_,_,_,load_A_stage), tArA_alpha_flt);                // smem -> rmem
        copy(tiled_col_s2r, tAsA_bias_s2r(_,_,_,_,load_A_stage), tArA_bias_flt);                  // smem -> rmem
      }
      tma_load_A_pipeline.consumer_release(tma_load_A_pipe_consumer_state);

      int apply_A_stage = batch_norm_apply_pipe_producer_state.index();
      auto curr_batch_norm_apply_pipe_producer_state = batch_norm_apply_pipe_producer_state;

      // Next pipeline stage
      ++tma_load_A_pipe_consumer_state;
      ++batch_norm_apply_pipe_producer_state;
      --k_tile_count;
      skip_wait = (k_tile_count <= 0);

      // Peek the next pipeline stage's barriers
      tma_load_A_barrier_token = tma_load_A_pipeline.consumer_try_wait(tma_load_A_pipe_consumer_state, skip_wait);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA_r2t.size(); ++i) {
        if constexpr (cute::is_same_v<ActivationFunctor<ElementA>, cutlass::epilogue::thread::ReLu<ElementA>>) {
          cutlass::guarded_multiply_add_relu0<ElementA, ElementAAlpha, ElementABias> guarded_multiply_add_relu0;
          tArA_r2t(i) = guarded_multiply_add_relu0(tArA_r2t(i), tArA_alpha(i), tArA_bias(i));
        }
        else { // For the other activations
          cutlass::guarded_multiply_add<ElementA, ElementAAlpha, ElementABias> guarded_multiply_add;
          ActivationFunctor<ElementA> activation;
          tArA_r2t(i) = activation(guarded_multiply_add(tArA_r2t(i), tArA_alpha(i), tArA_alpha(i)));
        }
      }

      batch_norm_apply_pipeline.producer_acquire(curr_batch_norm_apply_pipe_producer_state, batch_norm_apply_barrier_token);

      // Peek the next pipeline stage's barriers
      batch_norm_apply_barrier_token = batch_norm_apply_pipeline.producer_try_acquire(batch_norm_apply_pipe_producer_state, skip_wait);
      // Store A from rmem to tmem
      copy(tiled_sttm, tArA_r2t, tAtA_r2t(_,_,_,_,apply_A_stage));

      // Ensure tmem loads are complete
      cutlass::arch::fence_view_async_tmem_store();
      // Let the MMA know we are done batch norm apply
      batch_norm_apply_pipeline.producer_commit(curr_batch_norm_apply_pipe_producer_state);
    }
    return cute::make_tuple(tma_load_A_pipe_consumer_state, batch_norm_apply_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgEngine, class FrgLayout,
    class FragmentA, class FragmentB
  >
  CUTLASS_DEVICE auto
  mma(BatchNormApplyPipeline batch_norm_apply_pipeline,
      BatchNormApplyPipelineState batch_norm_apply_pipe_consumer_state,
      TMALoadBPipeline tma_load_B_pipeline,
      TMALoadBPipelineState tma_load_B_pipe_consumer_state,
      cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      cute::tuple<TiledMma, FragmentA, FragmentB> const& mma_inputs,
      int k_tile_count)
  {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    uint32_t skip_wait = k_tile_count <= 0;
    auto batch_norm_apply_barrier_token = batch_norm_apply_pipeline.consumer_try_wait(batch_norm_apply_pipe_consumer_state, skip_wait);
    auto tma_load_B_barrier_token = tma_load_B_pipeline.consumer_try_wait(tma_load_B_pipe_consumer_state, skip_wait);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // WAIT on batch_norm_apply_pipe_consumer_state, tma_load_B_pipe_consumer_state until its data are available
      // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
      batch_norm_apply_pipeline.consumer_wait(batch_norm_apply_pipe_consumer_state, batch_norm_apply_barrier_token);
      tma_load_B_pipeline.consumer_wait(tma_load_B_pipe_consumer_state, tma_load_B_barrier_token);
      // Compute on k_tile
      int read_A_stage = batch_norm_apply_pipe_consumer_state.index();
      int read_B_stage = tma_load_B_pipe_consumer_state.index();

      // Save current mainlop pipeline read state
      auto curr_batch_norm_apply_pipe_consumer_state = batch_norm_apply_pipe_consumer_state;
      auto curr_tma_load_B_pipe_consumer_state = tma_load_B_pipe_consumer_state;

      // Advance mainloop_pipe
      ++batch_norm_apply_pipe_consumer_state;
      ++tma_load_B_pipe_consumer_state;
      --k_tile_count;

      skip_wait = k_tile_count <= 0;
      // Peek at next iteration
      batch_norm_apply_barrier_token = batch_norm_apply_pipeline.consumer_try_wait(batch_norm_apply_pipe_consumer_state, skip_wait);
      tma_load_B_barrier_token = tma_load_B_pipeline.consumer_try_wait(tma_load_B_pipe_consumer_state, skip_wait);

      // Unroll the K mode manually so we can set scale C to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_A_stage), tCrB(_,_,k_block,read_B_stage), accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      batch_norm_apply_pipeline.consumer_release(curr_batch_norm_apply_pipe_consumer_state);
      tma_load_B_pipeline.consumer_release(curr_tma_load_B_pipe_consumer_state);
    }
    return cute::make_tuple(batch_norm_apply_pipe_consumer_state, tma_load_B_pipe_consumer_state);
  }

  template <
    class FrgEngine, class FrgLayout
  >
  CUTLASS_DEVICE auto
  mma_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TensorStorage& shared_tensors) const {
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    TiledMma tiled_mma;

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = tiled_mma.make_fragment_A(shape(TmemLayoutA{}));                         // (MMA,MMA_M,MMA_K,PIPE_BN)
    Tensor tCrB = tiled_mma.make_fragment_B(sB);                                              // (MMA,MMA_N,MMA_K,PIPE)

    // HACK: Change the starting address of A in TMEM -- use the last TMEM COL of accumulators
    tCrA.data() = accumulators.data().get() + size<0, 1>(accumulators.layout()) * size<2>(accumulators.layout())
        * size<3>(accumulators.layout()) * sizeof(ElementAccumulator) / sizeof(float) / size<1>(TmemWarpShape{});
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::BatchNormApplyStages>{} == size<3>(shape(TmemLayoutA{})));      // PIPE_BN
    return cute::make_tuple(tiled_mma, tCrA, tCrB);
  }

private:
  typename Params::TMA_A const* observed_tma_load_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_b_ = nullptr;
  typename Params::FpropAuxParams::TMA_A_alpha const* observed_tma_load_a_alpha_ = nullptr;
  typename Params::FpropAuxParams::TMA_A_bias const* observed_tma_load_a_bias_ = nullptr;

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
