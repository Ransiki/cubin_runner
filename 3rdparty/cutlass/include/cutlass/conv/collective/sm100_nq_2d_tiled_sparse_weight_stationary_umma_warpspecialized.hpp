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
//
// {$nv-internal-release file}
//
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/atom/copy_traits_sm100_w_tma.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/trace.h"
#include "cutlass/arch/memory.h"

#include "cutlass/conv/collective/builders/sm100_sparse_config.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  conv::Operator ConvOp,
  int AStages,
  int BStages,
  int NumSpatialDims,
  class ClusterShape,   // Static cluster shape or dynamic (int, int, _1)
  class KernelSchedule,
  class TileShape_,     // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class TileTraitsAE_,
  class TileTraitsB_>
struct CollectiveConv<
    MainloopSm100TmaSparseWeightStationaryUmmaWarpSpecializedNq2dTiled<
      ConvOp, AStages, BStages, NumSpatialDims, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    ElementB_,
    TiledMma_,
    TileTraitsAE_,
    TileTraitsB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm100TmaSparseWeightStationaryUmmaWarpSpecializedNq2dTiled<
      ConvOp, AStages, BStages, NumSpatialDims, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using TileTraitsAE = TileTraitsAE_;
  using TileTraitsA = typename TileTraitsAE_::TileTraitsA;
  using TileTraitsE = typename TileTraitsAE_::TileTraitsE;
  // GmemTiledCopyA represents Tiled Copy for Filter
  using GmemTiledCopyA = typename TileTraitsA::GmemTiledCopy;
  using SmemLayoutFlt = typename TileTraitsA::SmemLayoutAtom;
  // GmemTiledCopyB represents Tiled Copy for Activation
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;
  // Smem layout that considers halo in both shape and stride, this is used to derive SMEM buffer size
  using SmemLayoutActFull = typename TileTraitsB_::SmemLayoutFullAtom;
  // Smem layout that considers halo in stride but not in shape, this is used to derive UMMA desc
  using SmemLayoutActMma = typename TileTraitsB_::SmemLayoutMmaAtom;
  // Smem layout that doesn't consider halo in shape or stride, this is used to make TileCopy
  using SmemLayoutActTma = typename TileTraitsB_::SmemLayoutTmaAtom;
  using GmemTiledCopyE = typename TileTraitsE::GmemTiledCopy;
  using SmemLayoutMeta = typename TileTraitsE::SmemLayoutAtom;
  using ArchTag = typename DispatchPolicy::ArchTag;

  // Filter shape
  using FilterShape_TRS = typename KernelSchedule::FilterShapeTRS;
  // Traversal stride
  using StrideDHW = typename KernelSchedule::TraveralStrideDHW;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  // sparse element A/E
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  static constexpr int TensorASparsity = ElementAMma::sparsity;
  static constexpr int TensorESparsity = ElementEMma::sparsity;

  using SparseConfig = cutlass::Sm100ConvSparseConfig<ElementAMma, ElementEMma, TileShape>;
  using TileShapeMeta = typename SparseConfig::TileShapeE;
  using ExpandFactor = typename SparseConfig::ExpandFactor;
  using GmemReorderedAtom = typename SparseConfig::GmemReorderedAtom;
  using SmemReorderedAtom = typename SparseConfig::SmemReorderedAtom;

  using SmemCopyAtomE = cute::conditional_t<(decltype(size<0>(TileShape{}) == Int<128>{})::value), cute::SM100_UTCCP_128dp128bit_1cta,
                        cute::conditional_t<(decltype(size<0>(TileShape{}) ==  Int<64>{})::value), cute::SM100_UTCCP_2x64dp128bitlw0213_1cta,
                                                                                                   cute::SM100_UTCCP_4x32dp128bit_1cta>>;

  static constexpr int ActStages = DispatchPolicy::BStages;
  static constexpr int FltStages = DispatchPolicy::AStages;
  static constexpr int MetaStages = FltStages;

  static constexpr int Flt_T = size<0>(FilterShape_TRS{});
  static constexpr int Flt_R = size<1>(FilterShape_TRS{});
  static constexpr int Flt_S = size<2>(FilterShape_TRS{});

  // Accumulator buffer count, align with Flt_R
  static constexpr uint32_t AccumulatorPipelineStageCount = Flt_R;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;
  // W_Halo for TMALDG.W depends on Flt_S
  static constexpr int W_Halo_Size = Flt_S - 1;

  // Deduce the kernel facing stride tuple types based on the dispatch policy (spatial dim, algo, etc.)
  // The dispatch_policy_to_stride_A is default as activation for implicitGEMM use
  // The dispatch_policy_to_stride_B is default as filter for implicitGEMM use
  using StrideAct = decltype(detail::sm100_dispatch_policy_to_stride_A<DispatchPolicy>());
  using StrideFlt = decltype(detail::sm100_dispatch_policy_to_stride_B<DispatchPolicy>());
  using LayoutFlt  = typename SparseConfig::LayoutA;
  using LayoutMeta = typename SparseConfig::LayoutE;

  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using TmaInternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>>;
  using TmaInternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>>;
  using TmaInternalElementE = uint8_t;

  // Determine MMA type: MMA_1SM vs MMA_2SM
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  // Single-sided usage of the PipelineTmaUmmaAsync class, unlike GEMMs and Implicit GEMMs
  using MainloopFltPipeline = cutlass::PipelineTmaUmmaAsync<
                                FltStages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopPipelineFltParams = typename MainloopFltPipeline::Params;
  using MainloopFltPipelineState = typename cutlass::PipelineState<FltStages>;

  using MainloopActPipeline = cutlass::PipelineTmaUmmaAsync<
                                ActStages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopPipelineActParams = typename MainloopActPipeline::Params;
  using MainloopActPipelineState = typename cutlass::PipelineState<ActStages>;

  using MainloopMetaPipeline = cutlass::PipelineTmaUmmaAsync<
                                MetaStages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopPipelineMetaParams = typename MainloopMetaPipeline::Params;
  using MainloopMetaPipelineState = typename cutlass::PipelineState<MetaStages>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  // Check consistency of the 3 Act SMEM layouts
  // Consider Halo in shape
  static_assert(shape<0,0>(SmemLayoutActFull{}) >= shape<0,0>(SmemLayoutActTma{}) + W_Halo_Size, "SmemLayoutActFull considers halos in the shape.");
  static_assert(shape<0,0>(SmemLayoutActFull{}) % (1 << get_swizzle_portion(SmemLayoutActFull{}).num_bits) == 0, "SmemLayoutActFull's shape should be aligned with the swizzle.");
  static_assert(shape(SmemLayoutActMma{}) == shape(SmemLayoutActTma{}), "SmemLayoutActMma doesn't consider halo in shape.");
  // Consider Halo in stride
  static_assert(cosize(SmemLayoutActFull{}) == cosize(make_layout(shape(SmemLayoutActFull{}))), "SmemLayoutActFull should have packed shape and stride.");
  static_assert(cosize(SmemLayoutActTma{})  == cosize(make_layout(shape(SmemLayoutActTma{}))), "SmemLayoutActTma should have packed shape and stride.");
  static_assert(stride(SmemLayoutActMma{}.layout_b()) == stride(SmemLayoutActFull{}.layout_b()), "SmemLayoutActMma considers halo in stride.");

  static_assert(rank(SmemLayoutFlt{}) == 4, "SmemLayout must be rank 4");
  static_assert((size<0>(TileShape{}) == size<0>(get<0>(SmemLayoutFlt{}))), "SmemLayout must be compatible with the tile shape.");
  static_assert(rank(SmemLayoutActMma{}) == 4, "SmemLayout must be rank 4");
  static_assert((size<1>(TileShape{}) == size<0>(get<0>(SmemLayoutActMma{}))),
      "SmemLayout must be compatible with the tile shape.");

  static_assert(ActStages >= 2, "The number of A stages must be at least 2.");
  static_assert(FltStages >= 2, "The number of B stages must be at least 2.");

  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>
             || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopyA - invalid Filter TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM100_TMA_LOAD_W>,
      "GmemTiledCopyB - invalid Activation TMA copy atom specified.");

  struct FltSharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutFlt>> smem_Flt;
    } tensors;

    using PipelineStorage = typename MainloopFltPipeline::SharedStorage;
    PipelineStorage flt_pipeline;
  };

  using FltTensorStorage = typename FltSharedStorage::TensorStorage;
  using FltPipelineStorage = typename FltSharedStorage::PipelineStorage;

  struct ActSharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutActFull>> smem_Act;
    } tensors;

    using PipelineStorage = typename MainloopActPipeline::SharedStorage;
    PipelineStorage act_pipeline;
  };

  using ActTensorStorage = typename ActSharedStorage::TensorStorage;
  using ActPipelineStorage = typename ActSharedStorage::PipelineStorage;

  // metadata
  struct MetaSharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      // cute::array_aligned<typename TiledMma::ElementEVal, cute::cosize_v<SmemLayoutMeta>> smem_Meta;
      cute::ArrayEngine<sparse_elem<TensorESparsity, uint8_t>, cute::cosize_v<SmemLayoutMeta>> smem_Meta;
    } tensors;

    using PipelineStorage = typename MainloopMetaPipeline::SharedStorage;
    PipelineStorage meta_pipeline;
  };

  using MetaTensorStorage = typename MetaSharedStorage::TensorStorage;
  using MetaPipelineStorage = typename MetaSharedStorage::PipelineStorage;

  // SmemLayoutAct here for TransactionBytes only need to consider wHalo, don't count it rounded up smem rows.
  static constexpr uint32_t ActTransactionBytes =
      ((size<0>(SmemLayoutActTma{}) + (W_Halo_Size * size<1>(get<0>(SmemLayoutActTma{})))) * size<1>(SmemLayoutActTma{}) * size<2>(SmemLayoutActTma{}) *
      static_cast<uint32_t>(sizeof(TmaInternalElementB)));
  static constexpr uint32_t FltTransactionBytes =
      (size<0>(SmemLayoutFlt{}) * size<1>(SmemLayoutFlt{}) * size<2>(SmemLayoutFlt{}) *
      static_cast<uint32_t>(sizeof(TmaInternalElementA)) / TensorASparsity);
  static constexpr uint32_t MetaTransactionBytes =
      (size<0>(SmemLayoutMeta{}) * size<1>(SmemLayoutMeta{}) * size<2>(SmemLayoutMeta{}) *
      static_cast<uint32_t>(sizeof(TmaInternalElementE)) / TensorESparsity);

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    ElementB const* ptr_B;
    LayoutFlt    layout_b;
    ElementE const* ptr_E;
    LayoutMeta   layout_e;
  };

  // Device side kernel params
  struct Params {

    using ClusterLayout_VMNK = decltype(tiled_divide(
      make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
      make_tile(typename TiledMma::AtomThrID{}))
    );

    // Assumption: StrideFlt is congruent with Problem_MK
    using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(
            recast_ptr<sparse_elem<TensorASparsity, TmaInternalElementA>>(nullptr),
            LayoutFlt{}),
        SmemLayoutFlt{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{}));

    // Assumption: StrideAct is congruent with Problem_NK
    using TMA_B = decltype(make_w_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(
          make_gmem_ptr(static_cast<TmaInternalElementB const*>(nullptr)),
          make_layout(repeat_like(StrideAct{}, int32_t(0)), StrideAct{})),
        SmemLayoutActTma{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cute::array<int32_t, 1>{},
        cute::array<int32_t, 1>{},
        cute::array<int32_t, 1>{},
        cute::array<int32_t, 1>{},
        cute::array<int32_t, NumSpatialDimensions>{},
        cute::array<int32_t, NumSpatialDimensions>{})
        );

    using TMA_E = decltype(make_tma_atom_A_sm100<TmaInternalElementE>(
        GmemTiledCopyA{},
        // tensor_Meta,
        make_tensor(
          recast_ptr<sparse_elem<TensorESparsity, TmaInternalElementE>>(nullptr),
          LayoutMeta{}),
        SmemLayoutMeta{}(_,_,_,cute::Int<0>{}),
        TileShapeMeta{},
        TiledMma{},
        ClusterLayout_VMNK{}));

    TMA_A tma_load_flt_a;
    TMA_B tma_load_act_b;
    TMA_E tma_load_meta_e;
    // act w mode TMA does not support mcast, so we don't need act_b_fallback.
    TMA_A tma_load_flt_a_fallback;
    TMA_E tma_load_meta_e_fallback;
    dim3 cluster_shape_fallback;
    // Note: The is problem shape MNK, not the problem shape of conv input args.
    // The number of pixels we need to skip on the left.
    // This will control the data flow and kernel implementation details.
    int32_t num_pixels_skip_left;
    LayoutFlt   layout_flt_a;
    LayoutMeta  layout_meta_e;
  };

  //
  // Constructor
  //
  CUTLASS_DEVICE
  CollectiveConv(Params const& params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == params.cluster_shape_fallback.x && cs.y == params.cluster_shape_fallback.y);
      observed_tma_load_flt_a_ = is_fallback_cluster ? &params.tma_load_flt_a_fallback : &params.tma_load_flt_a;
      observed_tma_load_meta_e_ = is_fallback_cluster ? &params.tma_load_meta_e_fallback : &params.tma_load_meta_e;
    }
    else {
      observed_tma_load_flt_a_ = &params.tma_load_flt_a;
      observed_tma_load_meta_e_ = &params.tma_load_meta_e;
    }
    observed_tma_load_act_b_ = &params.tma_load_act_b;
  }

  //
  // Methods
  //

  // Lowers the host-side user-facing arguments to the kernel-facing launch parameters
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace,
                          cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;
    // from the flat problem shape arrays of ConvProblemShape<N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents.
    auto shape_Act_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_Flt_orig = problem_shape.get_shape_B();

    // Fill inferred cute strides from flat stride arrays
    auto dFlt = make_cute_packed_stride(StrideFlt{}, problem_shape.stride_B, ConvOp);
    auto dAct = make_cute_packed_stride(StrideAct{}, problem_shape.stride_A, ConvOp);

    auto ptr_Flt = recast_ptr<sparse_elem<TensorASparsity, TmaInternalElementA>>(args.ptr_B);
    auto ptr_Act = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_Meta = recast_ptr<sparse_elem<TensorESparsity, TmaInternalElementE>>(args.ptr_E);

    Tensor tensor_Flt  = make_tensor(make_gmem_ptr(ptr_Flt), args.layout_b);
    Tensor tensor_Act  = make_tensor(make_gmem_ptr(ptr_Act), make_layout(shape_Act_orig, dAct));
    Tensor tensor_Meta = make_tensor(make_gmem_ptr(ptr_Meta), args.layout_e);

    auto lower_srt = detail::compute_lower_srt(problem_shape);

    // Only lower_d is used in act tensor iteration.
    // lower_w is calculated in below compte_corner_w().
    // lower_h is not used.
    auto lower_whd = detail::compute_lower_corner_whd(problem_shape);
    cute::array<int32_t, 1> lower_d{lower_whd[2]};

    auto upper_whd = detail::compute_upper_corner_whd(problem_shape);

    cute::array<int32_t, 1> start_coord_w;
    int32_t wOffset = 0;
    start_coord_w[0] = -1 * int32_t(problem_shape.lower_padding[NumSpatialDimensions-1]) - wOffset;
    int32_t num_pixels_skip_left = cute::min(int32_t(problem_shape.lower_padding[NumSpatialDimensions-1]), 2);

    auto [lower_w, upper_w] = detail::compute_corner_w(problem_shape);

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    typename Params::TMA_A tma_load_flt_a = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_Flt,
        SmemLayoutFlt{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_flt_a_fallback = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_Flt,
        SmemLayoutFlt{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_act_b = make_w_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_Act,
        SmemLayoutActTma{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        start_coord_w,
        lower_w,
        upper_w,
        lower_d,
        lower_srt,
        problem_shape.traversal_stride);

    typename Params::TMA_E tma_load_meta_e = make_tma_atom_A_sm100<TmaInternalElementE>(
        GmemTiledCopyA{},
        tensor_Meta,
        SmemLayoutMeta{}(_,_,_,cute::Int<0>{}),
        TileShapeMeta{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_E tma_load_meta_e_fallback = make_tma_atom_A_sm100<TmaInternalElementE>(
        GmemTiledCopyA{},
        tensor_Meta,
        SmemLayoutMeta{}(_,_,_,cute::Int<0>{}),
        TileShapeMeta{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    return {
      tma_load_flt_a,
      tma_load_act_b,
      tma_load_meta_e,
      tma_load_flt_a_fallback,
      tma_load_meta_e_fallback,
      hw_info.cluster_shape_fallback,
      num_pixels_skip_left,
      args.layout_b,
      args.layout_e
    };
  }

  // Returns true if the collective can run successfully with the arguments, else false.
  static constexpr bool
  can_implement(
    ProblemShape const& problem_shape,
    Arguments const& args) {
    // Alias
    const auto & activation_shape  = problem_shape.shape_A;          // [N,D,H,W,C]
    const auto & filter_shape      = problem_shape.shape_B;          // [K,T,R,S,C]
    const auto & activation_stride = problem_shape.stride_A;
    const auto & filter_stride     = problem_shape.stride_B;
    const auto & traversal_stride  = problem_shape.traversal_stride;

    bool implementable = true;

    // Activation and Filter channel mode extents must match
    if constexpr (ConvOp == conv::Operator::kFprop) {
      // channel count is equal
      implementable &= (activation_shape[NumTensorDimensions-1] == filter_shape[NumTensorDimensions-1]);
    }
    else {
      // Not support Dgrad yet
      implementable &= false;
    }
    // channel mode is major
    implementable &= (activation_stride[NumTensorDimensions-1] == 1);
    implementable &= (filter_stride[NumTensorDimensions-1]     == 1);

    // Filter size
    implementable &= (filter_shape[NumTensorDimensions-3] == Flt_R && filter_shape[NumTensorDimensions-2] == Flt_S);
    if constexpr (NumTensorDimensions == 5) {
      implementable &= filter_shape[NumTensorDimensions-4] == Flt_T;
      implementable &= (Flt_T == 1 || Flt_T == 3);
    }

    // Stride
    {
      for (auto stride: traversal_stride) {
       implementable &= (stride >= 1 && stride <= 8);
      }
    }
    implementable &= (traversal_stride[NumSpatialDimensions-2] == get<1>(StrideDHW{}) && traversal_stride[NumSpatialDimensions-1] == get<2>(StrideDHW{}));
    if constexpr (NumTensorDimensions == 5) {
      implementable &= (traversal_stride[NumSpatialDimensions-3] == get<0>(StrideDHW{}));
      implementable &= (get<0>(StrideDHW{}) == 1);
    }

    // Dilation not supported
    for ( auto & d : problem_shape.dilation ) { implementable &= (d == 1); }

    // Group conv not supported
    implementable &= (problem_shape.groups == 1);

    // Conv mode
    implementable &= (problem_shape.mode == cutlass::conv::Mode::kCrossCorrelation);

    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == mainloop_params.cluster_shape_fallback.x && cs.y == mainloop_params.cluster_shape_fallback.y);
      if (is_fallback_cluster) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_flt_a_fallback.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_meta_e_fallback.get_tma_descriptor());
      }
      else {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_flt_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_meta_e.get_tma_descriptor());
      }
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_act_b.get_tma_descriptor());
    }
    else {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_flt_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_act_b.get_tma_descriptor());
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_meta_e.get_tma_descriptor());
    }
  }

  /// Prepare for a collective-scoped tensor loading
  /// Must be called once before load()
  /// Producer Perspective
  template <class FilterShape>
  CUTLASS_DEVICE auto
  load_flt_init(
    FilterShape const& filter_shape,
    FltTensorStorage& shared_tensors) const {

    using X = Underscore;

    // mFlt
    Tensor mFlt = observed_tma_load_flt_a_->get_tma_tensor(filter_shape);
    // Tile the tensors and defer the slice
    Tensor gFlt = local_tile(mFlt, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (TILE_M,TILE_K)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));
    Tensor tCgFlt = cta_mma.partition_A(gFlt);                           // (MMA,MMA_M,MMA_K,k)

    int block_rank_in_cluster = cute::block_rank_in_cluster();

    Tensor sFlt = make_tensor(make_smem_ptr(shared_tensors.smem_Flt.begin()), SmemLayoutFlt{});  // (MMA,MMA_M,MMA_K,PIPE)

    // Define the CTA-in-CGA Layout and Coord
    auto cluster_shape  = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    Layout cta_layout_mnk  = make_layout(cluster_shape);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    uint16_t mcast_mask_flt_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);

    auto [tAgFlt_mk, tAsFlt] = tma_partition(*observed_tma_load_flt_a_,
                                     get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                     group_modes<0,3>(sFlt), group_modes<0,3>(tCgFlt)
    );

    return cute::make_tuple(tAgFlt_mk, tAsFlt, mcast_mask_flt_a);
  }

  // In NQ Tiling we xcorrelate filter matrix.
  // As we unfold all loops mode explicitly, it is trival to give coord (c,s,r,t) in fprop,
  // and give coord(c,S-s,R-r,T-t) in dgrad in kernel level.
  CUTLASS_HOST_DEVICE
  constexpr auto
  get_flt_coord(int32_t c_iter, int32_t s_iter, int32_t r_iter, int32_t t_iter) const {

    if constexpr (ConvOp == conv::Operator::kFprop) {
      return make_coord(c_iter, s_iter, r_iter, t_iter);
    }
    else if constexpr (ConvOp == conv::Operator::kDgrad) {
      return make_coord(c_iter,
                        Flt_S - 1 - s_iter,
                        Flt_R - 1 - r_iter,
                        Flt_T - 1 - t_iter);
    }
    else {
      static_assert(ConvOp == conv::Operator::kWgrad, "Wgrad is not yet supported.");
    }
  }

  template <
    class GTensorPartitionedA, class STensorA,
    class GTensorPartitionedE, class STensorE,
    class TileCoordMNKL
  >
  CUTLASS_DEVICE auto
  load_flt_and_meta(MainloopFltPipeline flt_pipeline,
                    MainloopFltPipelineState flt_smem_pipe_producer_state,
                    cute::tuple<GTensorPartitionedA, STensorA, uint16_t> const& load_flt_a,
                    cute::tuple<GTensorPartitionedE, STensorE> const& load_meta_e,
                    TileCoordMNKL const& cta_coord_mnkl,
                    int c_chunks, int h_pixels_start, int h_pixels_end,
                    int lane_predicate) {

    asm volatile(".pragma \"set knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}

    auto [tAgFlt_mk, tAsFlt, mcast_mask_flt_a] = load_flt_a;
    auto [tEgMeta_mk, tEsMeta] = load_meta_e;
    Tensor tAgFlt = tAgFlt_mk(_, get<0>(cta_coord_mnkl), _);
    Tensor tEgMeta = tEgMeta_mk(_, get<0>(cta_coord_mnkl), _);

    //
    // Load filter
    //
    static constexpr int Ramp_State = Flt_R - 1;
    int ss_start_idx = h_pixels_start + (Flt_R - 1);
    int ss_end_idx = h_pixels_end - (Flt_R - 1);
    int Steady_State_Iters = ss_end_idx - ss_start_idx;

    auto barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state);

    // Ramp up states
    CUTLASS_PRAGMA_UNROLL
    for (int ramp_iter = 0; ramp_iter < Ramp_State; ++ramp_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int t_iter = 0; t_iter < Flt_T; ++t_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          CUTLASS_PRAGMA_UNROLL
          for (int s_iter = 0; s_iter < Flt_S; ++s_iter) {
            CUTLASS_PRAGMA_UNROLL
            for (int r_iter = ramp_iter; r_iter >= 0; --r_iter) {
              flt_pipeline.producer_acquire(flt_smem_pipe_producer_state, barrier_token);

              using BarrierType = typename MainloopFltPipeline::ProducerBarrierType;
              BarrierType* tma_barrier = flt_pipeline.producer_get_barrier(flt_smem_pipe_producer_state);

              int write_stage = flt_smem_pipe_producer_state.index();
              ++flt_smem_pipe_producer_state;
              barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state);

              auto loop_coords = get_flt_coord(c_iter, s_iter, r_iter, t_iter);

              if (lane_predicate) {
                copy(observed_tma_load_flt_a_->with(*tma_barrier, mcast_mask_flt_a), tAgFlt(_,loop_coords), tAsFlt(_,write_stage));
                copy(observed_tma_load_meta_e_->with(*tma_barrier, mcast_mask_flt_a), tEgMeta(_,loop_coords), tEsMeta(_,write_stage));
              }
            } // r iter
          } // s iter
        } // c iter
      } // t iter
    } // ramp state

    // Steady states
    CUTLASS_PRAGMA_NO_UNROLL
    for (int ss_iter = 0; ss_iter < Steady_State_Iters; ++ss_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int t_iter = 0; t_iter < Flt_T; ++t_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          CUTLASS_PRAGMA_UNROLL
          for (int s_iter = 0; s_iter < Flt_S; ++s_iter) {
            CUTLASS_PRAGMA_UNROLL
            for (int r_iter = Flt_R - 1; r_iter >= 0; --r_iter) {
              flt_pipeline.producer_acquire(flt_smem_pipe_producer_state, barrier_token);

              using BarrierType = typename MainloopFltPipeline::ProducerBarrierType;
              BarrierType* tma_barrier = flt_pipeline.producer_get_barrier(flt_smem_pipe_producer_state);

              int write_stage = flt_smem_pipe_producer_state.index();
              ++flt_smem_pipe_producer_state;
              barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state);

              auto loop_coords = get_flt_coord(c_iter, s_iter, r_iter, t_iter);

              if (lane_predicate) {
                copy(observed_tma_load_flt_a_->with(*tma_barrier, mcast_mask_flt_a), tAgFlt(_,loop_coords), tAsFlt(_,write_stage));
                copy(observed_tma_load_meta_e_->with(*tma_barrier, mcast_mask_flt_a), tEgMeta(_,loop_coords), tEsMeta(_,write_stage));
              }
            } // r iter
          } // s iter
        } // c iter
      } // t iter
    } // steady state counter

    // Ramp down states
    CUTLASS_PRAGMA_UNROLL
    for (int ramp_iter = Ramp_State - 1; ramp_iter >= 0; --ramp_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int t_iter = 0; t_iter < Flt_T; ++t_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          CUTLASS_PRAGMA_UNROLL
          for (int s_iter = 0; s_iter < Flt_S; ++s_iter) {
            CUTLASS_PRAGMA_UNROLL
            for (int r_iter = Flt_R - 1; r_iter >= Flt_R - 1 - ramp_iter; --r_iter) {
              flt_pipeline.producer_acquire(flt_smem_pipe_producer_state, barrier_token);

              using BarrierType = typename MainloopFltPipeline::ProducerBarrierType;
              BarrierType* tma_barrier = flt_pipeline.producer_get_barrier(flt_smem_pipe_producer_state);

              int write_stage = flt_smem_pipe_producer_state.index();
              ++flt_smem_pipe_producer_state;
              barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state);

              auto loop_coords = get_flt_coord(c_iter, s_iter, r_iter, t_iter);

              if (lane_predicate) {
                copy(observed_tma_load_flt_a_->with(*tma_barrier, mcast_mask_flt_a), tAgFlt(_,loop_coords), tAsFlt(_,write_stage));
                copy(observed_tma_load_meta_e_->with(*tma_barrier, mcast_mask_flt_a), tEgMeta(_,loop_coords), tEsMeta(_,write_stage));
              }
            } // r iter
          } // s iter
        } // c iter
      } // t iter
    } // ramp state

    asm volatile(".pragma \"reset knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}

    return cute::make_tuple(flt_smem_pipe_producer_state);
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_flt_and_meta_tail(MainloopFltPipeline flt_pipeline, MainloopFltPipelineState flt_smem_pipe_producer_state,
                int lane_predicate) {
    // Issue the epilogue waits
    // Free filter - tma_load_a
    if (lane_predicate) {
      flt_pipeline.producer_tail(flt_smem_pipe_producer_state);
    }
  }

  /// Prepare for a collective-scoped tensor loading
  /// Must be called once before load()
  /// Producer Perspective
  template <class MetadataShape>
  CUTLASS_DEVICE auto
  load_meta_init(
    MetadataShape const& metadata_shape,
    MetaTensorStorage& shared_tensors) const {

    using X = Underscore;

    // mMeta
    Tensor mMeta = observed_tma_load_meta_e_->get_tma_tensor(metadata_shape);
    // Tile the tensors and defer the slice
    Tensor gMeta = local_tile(mMeta, TileShapeMeta{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (TILE_M,TILE_K)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));
    Tensor tCgMeta = cta_mma.partition_A(gMeta);                           // (MMA,MMA_M,MMA_K,k)

    int block_rank_in_cluster = cute::block_rank_in_cluster();

    Tensor sMeta = make_tensor(make_smem_ptr(shared_tensors.smem_Meta.begin()), SmemLayoutMeta{});  // (MMA,MMA_M,MMA_K,PIPE)

    // Define the CTA-in-CGA Layout and Coord
    auto cluster_shape  = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    Layout cta_layout_mnk  = make_layout(cluster_shape);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    auto [tEgMeta_mk, tEsMeta] = tma_partition(*observed_tma_load_meta_e_,
                                     get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                     group_modes<0,3>(sMeta), group_modes<0,3>(tCgMeta));

    return cute::make_tuple(tEgMeta_mk, tEsMeta);
  }

  /// Prepare for a collective-scoped tensor loading
  /// Must be called once before load()
  /// Producer Perspective
  template <class ActivationShape>
  CUTLASS_DEVICE auto
  load_act_init(
    ActivationShape const& activation_shape,
    ActTensorStorage& shared_tensors) {

    using X = Underscore;

    Tensor mAct = observed_tma_load_act_b_->get_tma_tensor(activation_shape);  // ((QN),C,P,Z)
    // Tile the tensors and defer the slice
    Tensor gAct = local_tile(mAct, TileShape{}, make_coord(_,_,_), Step<X,_1,_1>{});  // (TILE_N,TILE_K,n,k,h,d)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));
    Tensor tCgAct = cta_mma.partition_B(gAct);                           // (MMA,MMA_N,MMA_K,n,k,h,d)

    int block_rank_in_cluster = cute::block_rank_in_cluster();

    Tensor sAct = make_tensor(make_smem_ptr(shared_tensors.smem_Act.data()), SmemLayoutActMma{});  // (BLK_N,BLK_K,PIPE)

    // Define the CTA-in-CGA Layout and Coord
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    Layout cta_layout_mnk  = make_layout(cluster_shape);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    auto [tBgAct_nk, tBsAct] = tma_partition(*observed_tma_load_act_b_,
                                     get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                     group_modes<0,3>(sAct), group_modes<0,3>(tCgAct));

    return cute::make_tuple(tBgAct_nk, tBsAct);
  }

  template <class GTensorPartitionedB, class STensorB, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  load_act(MainloopActPipeline act_pipeline,
           MainloopActPipelineState act_smem_pipe_producer_state,
           cute::tuple<GTensorPartitionedB, STensorB> const& load_inputs,
           TileCoordMNKL const& cta_coord_mnkl,
           int c_chunks, int h_pixels_start, int h_pixels_end, int z_idx,
           int lane_predicate) {

    constexpr int w_Halo = Flt_S - 1;
    constexpr int w_Offset = 0;

    //
    // Load Activation
    //

    auto [tBgAct_nk, tBsAct] = load_inputs;
    Tensor tBgAct_ = tBgAct_nk(_, get<1>(cta_coord_mnkl), _, _, _); // (TMA,k,h,d)
    // basic tensor
    auto tBgAct_linear = make_coord_tensor(tBgAct_.layout().layout_b());
    auto tBgAct = make_tensor(
      // deduce the nq offset at the begining to reduce redundant alu.
      ArithmeticTupleIterator(tBgAct_(Int<0>{})),
      composition(tBgAct_.layout().layout_a(),
                  tBgAct_linear(Int<0>{}),
                  tBgAct_linear.layout())
    );

    auto barrier_token = act_pipeline.producer_try_acquire(act_smem_pipe_producer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int h_iter = h_pixels_start; h_iter < h_pixels_end; ++h_iter) {
      asm volatile(".pragma \"set knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}
      CUTLASS_PRAGMA_UNROLL
      for (int d_iter = z_idx ; d_iter < z_idx + Flt_T; ++d_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          act_pipeline.producer_acquire(act_smem_pipe_producer_state, barrier_token);

          using BarrierType = typename MainloopActPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = act_pipeline.producer_get_barrier(act_smem_pipe_producer_state);

          int write_stage = act_smem_pipe_producer_state.index();

          if (lane_predicate) {
            copy(observed_tma_load_act_b_->with(*tma_barrier, w_Halo, w_Offset), tBgAct(_,c_iter,h_iter,d_iter), tBsAct(_,write_stage));
          }

          // just want to skip the try-wait.
          ++act_smem_pipe_producer_state;
          barrier_token = act_pipeline.producer_try_acquire(act_smem_pipe_producer_state);
        } // c iter
      } // d iter
      asm volatile(".pragma \"reset knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}
    } // h iter

    return cute::make_tuple(act_smem_pipe_producer_state);
  }

  CUTLASS_DEVICE void
  load_act_tail(MainloopActPipeline act_pipeline, MainloopActPipelineState act_smem_pipe_producer_state,
                int lane_predicate) {
    // Issue the epilogue waits
    // Free activation - tma_load_b
    if (lane_predicate) {
      act_pipeline.producer_tail(act_smem_pipe_producer_state);
    }
  }

  CUTLASS_DEVICE auto
  mma_init(Params const& params, FltTensorStorage& flt_shared_tensors, ActTensorStorage& act_shared_tensors, MetaTensorStorage& meta_shared_tensors, uint32_t meta_base_addr) {
    Tensor sFlt  = make_tensor(make_smem_ptr( flt_shared_tensors.smem_Flt.begin()), SmemLayoutFlt{});          // (BLK_M,BLK_K,PIPE)
    Tensor sAct  = make_tensor(make_smem_ptr( act_shared_tensors.smem_Act.data()), SmemLayoutActMma{});       // (BLK_N,BLK_K,PIPE)
    Tensor sMeta = make_tensor(make_smem_ptr(meta_shared_tensors.smem_Meta.begin()), SmemLayoutMeta{});

    [[maybe_unused]] TiledMma tiled_mma; // suppress spurious MSVC "unused variable" warning

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = tiled_mma.make_fragment_A(sFlt);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = tiled_mma.make_fragment_B(sAct);                                           // (MMA,MMA_N,MMA_K,PIPE)

    using X = Underscore;
    Tensor mE = params.tma_load_meta_e.get_tma_tensor(params.layout_meta_e.shape());
    auto tile_coord = make_coord(_,_,_);
    Tensor gE       = local_tile(mE, TileShape{}, tile_coord, Step<_1, X,_1>{});
    Tensor gE_utccp = local_tile(mE, TileShapeMeta{}, tile_coord, Step<_1, X,_1>{});

    ThrMMA cta_mma = TiledMma{}.get_slice(cute::block_rank_in_cluster());
    Tensor tCgE = cta_mma.partition_A(gE);
    Tensor tEgE = cta_mma.partition_A(gE_utccp);

    static constexpr int stages = 2;

    // Expand the metadata with HMMA
    Tensor tCtE = make_tensor<typename TiledMma::FrgTypeE>(append(insert<2>(take<0,3>(shape(tCgE)), ExpandFactor{}), Int<stages>{}));
    // used for constructing the utccp_copy
    Tensor tE  = make_tensor<typename TiledMma::FrgTypeE>(append(take<0,3>(shape(tEgE)), Int<stages>{}));
    tCtE.data().get() = meta_base_addr;
    tE.data().get() = meta_base_addr;

    [[maybe_unused]] auto tiledcpy_utccp = make_utccp_copy(SmemCopyAtomE{}, recast<uint8_t>(tE(_,_,_,Int<0>{})));
    auto thrcpy_utccp = tiledcpy_utccp.get_slice(0);
    Tensor tEsE_tmp = thrcpy_utccp.partition_S(recast<uint8_t>(sMeta));
    Tensor tEtE     = thrcpy_utccp.partition_D(recast<uint8_t>(tE));
    Tensor tEsE     = get_utccp_smem_desc_tensor<SmemCopyAtomE>(tEsE_tmp);

    return cute::make_tuple(
      tCrA, tCrB, tCtE(_,_,Int<0>{},_,_),     // mma oprands
      tEsE, tEtE,                             // utccp oprands
      tiledcpy_utccp);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgEngine, class FrgLayout,
    class Tensor_tCrA, class Tensor_tCrB, class Tensor_tCtE, // mma operands
    class Tensor_tEsE, class Tensor_tEtE,                    // utccp_operands
    class TiledCopyUtccp
  >
  CUTLASS_DEVICE auto
  mma(MainloopFltPipeline flt_pipeline, MainloopFltPipelineState flt_smem_pipe_consumer_state,
      MainloopActPipeline act_pipeline, MainloopActPipelineState act_smem_pipe_consumer_state,
      AccumulatorPipeline acc_pipeline, AccumulatorPipelineState acc_tmem_pipe_producer_state,
      cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      int c_chunks, int h_pixels_start, int h_pixels_end,
      cute::tuple<Tensor_tCrA, Tensor_tCrB, Tensor_tCtE, Tensor_tEsE, Tensor_tEtE, TiledCopyUtccp>& operands,
      int lane_predicate) {

    asm volatile(".pragma \"set knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}
    asm volatile(".pragma \"global knob AdvancedSBDiffXBlockRdSb\";\n" : : : "memory");     // {$nv-internal-release}

    TiledMma tiled_mma;

    cute::Tensor tCrA = cute::get<0>(operands);        // (MMA, MMA_M, MMA_K, PIPE)
    cute::Tensor tCtE = cute::get<2>(operands);        // (MMA, MMA_M, MMA_K, PIPE_E)
    cute::Tensor tCrB = cute::get<1>(operands);        // (MMA, MMA_N, MMA_K, PIPE)

    Tensor tEsE = get<3>(operands);
    Tensor tEtE = get<4>(operands);
    auto tiled_cpy_utccp = get<5>(operands);

    static constexpr int Ramp_State = Flt_R - 1;
    int ss_start_idx = h_pixels_start + (Flt_R - 1);
    int ss_end_idx = h_pixels_end - (Flt_R - 1);
    int Steady_State_Iters = ss_end_idx - ss_start_idx;

    auto flt_barrier_token = flt_pipeline.consumer_try_wait(flt_smem_pipe_consumer_state);
    auto act_barrier_token = act_pipeline.consumer_try_wait(act_smem_pipe_consumer_state);
    auto acc_barrier_token = acc_pipeline.producer_try_acquire(acc_tmem_pipe_producer_state);
    auto acc_tmem_pipe_producer_release_state = acc_tmem_pipe_producer_state;

    // output_acc_tmem_pipe_state is to track the begin of tmem circular buffer index to store MMA result for each output P mode (each ss_iter).
    // Within r_iter, we use round-robin tmem circular buffer to store MMA result by current_acc_tmem_pipe_state.
    // We will go back to output_acc_tmem_pipe_state when we finish all r_iter in each s_iter.
    // output_acc_tmem_pipe_state and current_acc_tmem_pipe_state are not involved in any pipeline sync logic, just to track buffer index.

    // acc_tmem_pipe_producer_state is to track the pipeline between MMA store tmem acc and Epilogue load tmem acc.
    // acc_tmem_pipe_producer_state and output_acc_tmem_pipe_state/current_acc_tmem_pipe_state have the same pipeline stage as buffer num,
    // but they are different in function, and their accumulating steps during loops are also different.
    auto output_acc_tmem_pipe_state = acc_tmem_pipe_producer_state;

    int meta_pipe = 0;

    // Ramp up states
    CUTLASS_PRAGMA_UNROLL
    for (int ramp_iter = 0; ramp_iter < Ramp_State; ++ramp_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int t_iter = 0; t_iter < Flt_T; ++t_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          // Wait on Act loading - NQ slice with Halo
          act_pipeline.consumer_wait(act_smem_pipe_consumer_state, act_barrier_token);
          int read_stage_act = act_smem_pipe_consumer_state.index();
          auto curr_act_smem_pipe_consumer_state = act_smem_pipe_consumer_state;

          CUTLASS_PRAGMA_UNROLL
          for (int s_iter = 0; s_iter < Flt_S; ++s_iter) {
            set_czm_and_bshift(tiled_mma, s_iter);
            auto current_acc_tmem_pipe_state = output_acc_tmem_pipe_state;
            CUTLASS_PRAGMA_UNROLL
            for (int r_iter = ramp_iter; r_iter >= 0; --r_iter) {
              // Acquire acc_buffer for coming math
              if (s_iter == 0 && r_iter == 0 && c_iter == 0 && t_iter == 0) {
                acc_pipeline.producer_acquire(acc_tmem_pipe_producer_state, acc_barrier_token);
                // setup UPp to ignore C.
                tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
              }
              if (s_iter == 0 && r_iter == 1 && c_iter == 0 && t_iter == 0) {
                ++acc_tmem_pipe_producer_state;
                acc_barrier_token = acc_pipeline.producer_try_acquire(acc_tmem_pipe_producer_state);
              }
              if (s_iter == Flt_S - 1 && r_iter == 0) {
                ++act_smem_pipe_consumer_state;
                act_barrier_token = act_pipeline.consumer_try_wait(act_smem_pipe_consumer_state);
              }
              auto act_buffer_status = get_act_buffer_status_for_rampup(s_iter, r_iter, ramp_iter);

              flt_pipeline.consumer_wait(flt_smem_pipe_consumer_state, flt_barrier_token);
              int read_stage_flt = flt_smem_pipe_consumer_state.index();
              int &read_stage_meta = read_stage_flt;

              if (lane_predicate) {
                copy(tiled_cpy_utccp, tEsE(_,_,_,_,read_stage_meta), tEtE(_,_,_,_,meta_pipe));
              }

              auto curr_flt_smem_pipe_consumer_state = flt_smem_pipe_consumer_state;
              ++flt_smem_pipe_consumer_state;

              // Do gemm
              // Unroll the K mode manually so we can set scale C to 1
              CUTLASS_PRAGMA_UNROLL
              for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                if (k_block == 1) {
                  flt_barrier_token = flt_pipeline.consumer_try_wait(flt_smem_pipe_consumer_state);
                }
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(
                  tiled_mma.with(
                    static_cast<typename UMMA::BMatrixBufferId>(k_block),
                    act_buffer_status,
                    tCtE(_,_,k_block,meta_pipe)),
                  tCrA(_,_,k_block,read_stage_flt),
                  tCrB(_,_,k_block,read_stage_act),
                  accumulators(_,_,_,current_acc_tmem_pipe_state.index()));
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
              }
              ++current_acc_tmem_pipe_state;
              meta_pipe ^=1;
              flt_pipeline.consumer_release(curr_flt_smem_pipe_consumer_state);
            } // r iter
          } // s iter
          act_pipeline.consumer_release(curr_act_smem_pipe_consumer_state);
        } // c iter
      } // t iter
    } // ramp state

    // Steady states
    CUTLASS_PRAGMA_NO_UNROLL
    for (int ss_iter = 0; ss_iter < Steady_State_Iters; ++ss_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int t_iter = 0; t_iter < Flt_T; ++t_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          // Wait on Act loading - NQ slice with Halo
          act_pipeline.consumer_wait(act_smem_pipe_consumer_state, act_barrier_token);
          int read_stage_act = act_smem_pipe_consumer_state.index();
          auto curr_act_smem_pipe_consumer_state = act_smem_pipe_consumer_state;

          CUTLASS_PRAGMA_UNROLL
          for (int s_iter = 0; s_iter < Flt_S; ++s_iter) {
            set_czm_and_bshift(tiled_mma, s_iter);
            auto current_acc_tmem_pipe_state = output_acc_tmem_pipe_state;
            CUTLASS_PRAGMA_UNROLL
            for (int r_iter = Flt_R - 1; r_iter >= 0; --r_iter) {
              // Acquire acc_buffer for coming math
              if (s_iter == 0 && r_iter == 0 && c_iter == 0 && t_iter == 0) {
                acc_pipeline.producer_acquire(acc_tmem_pipe_producer_state, acc_barrier_token);
                // setup UPp to ignore C.
                tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
              }

              if (s_iter == 0 && r_iter == 1 && c_iter == 0 && t_iter == 0) {
                ++acc_tmem_pipe_producer_state;
                acc_barrier_token = acc_pipeline.producer_try_acquire(acc_tmem_pipe_producer_state);
              }

              if (s_iter == Flt_S - 1 && r_iter == 0) {
                ++act_smem_pipe_consumer_state;
                act_barrier_token = act_pipeline.consumer_try_wait(act_smem_pipe_consumer_state);
              }

              auto act_buffer_status = get_act_buffer_status(s_iter, r_iter);

              flt_pipeline.consumer_wait(flt_smem_pipe_consumer_state, flt_barrier_token);
              int read_stage_flt = flt_smem_pipe_consumer_state.index();
              int &read_stage_meta = read_stage_flt;

              if (lane_predicate) {
                copy(tiled_cpy_utccp, tEsE(_,_,_,_,read_stage_meta), tEtE(_,_,_,_,meta_pipe));
              }

              auto curr_flt_smem_pipe_consumer_state = flt_smem_pipe_consumer_state;
              ++flt_smem_pipe_consumer_state;

              // Do gemm
              // Unroll the K mode manually so we can set scale C to 1
              CUTLASS_PRAGMA_UNROLL
              for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                if (k_block == 1) {
                  flt_barrier_token = flt_pipeline.consumer_try_wait(flt_smem_pipe_consumer_state);
                }
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(
                  tiled_mma.with(
                    static_cast<typename UMMA::BMatrixBufferId>(k_block),
                    act_buffer_status,
                    tCtE(_,_,k_block,meta_pipe)),
                  tCrA(_,_,k_block,read_stage_flt),
                  tCrB(_,_,k_block,read_stage_act),
                  accumulators(_,_,_,current_acc_tmem_pipe_state.index()));
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
              }
              ++current_acc_tmem_pipe_state;
              meta_pipe ^=1;
              flt_pipeline.consumer_release(curr_flt_smem_pipe_consumer_state);

              // Arrive acc_buffer_full
              if (r_iter == (Flt_R - 1) && s_iter == (Flt_S - 1) && c_iter == (c_chunks - 1)
                  && t_iter == (Flt_T -1)) {
                acc_pipeline.producer_commit(acc_tmem_pipe_producer_release_state);
                ++acc_tmem_pipe_producer_release_state;
              } // The earliest slot to output acc and the last c chunk
            } // r iter
          } // s iter
          act_pipeline.consumer_release(curr_act_smem_pipe_consumer_state);
        } // c iter
      } // t iter
      // Increment the output acc index
      ++output_acc_tmem_pipe_state;
    } // steady state counter

    ++acc_tmem_pipe_producer_state;

    // Ramp down states
    CUTLASS_PRAGMA_UNROLL
    for (int ramp_iter = Ramp_State - 1; ramp_iter >= 0; --ramp_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int t_iter = 0; t_iter < Flt_T; ++t_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          // Wait on Act loading - NQ slice with Halo
          act_pipeline.consumer_wait(act_smem_pipe_consumer_state, act_barrier_token);
          int read_stage_act = act_smem_pipe_consumer_state.index();
          auto curr_act_smem_pipe_consumer_state = act_smem_pipe_consumer_state;

          CUTLASS_PRAGMA_UNROLL
          for (int s_iter = 0; s_iter < Flt_S; ++s_iter) {
            set_czm_and_bshift(tiled_mma, s_iter);
            auto current_acc_tmem_pipe_state = output_acc_tmem_pipe_state;
            CUTLASS_PRAGMA_UNROLL
            for (int r_iter = Flt_R - 1; r_iter >= Flt_R - 1 - ramp_iter; --r_iter) {

              auto act_buffer_status = get_act_buffer_status_for_rampdown(s_iter, r_iter, ramp_iter);

              if (s_iter == Flt_S - 1 && r_iter == Flt_R - 1 - ramp_iter) {
                uint32_t skip_wait_act = (ramp_iter == 0) && (t_iter == Flt_T - 1) && (c_iter == c_chunks - 1);
                ++act_smem_pipe_consumer_state;
                act_barrier_token = act_pipeline.consumer_try_wait(act_smem_pipe_consumer_state, skip_wait_act);
              }

              flt_pipeline.consumer_wait(flt_smem_pipe_consumer_state, flt_barrier_token);
              int read_stage_flt = flt_smem_pipe_consumer_state.index();
              int &read_stage_meta = read_stage_flt;

              if (lane_predicate) {
                copy(tiled_cpy_utccp, tEsE(_,_,_,_,read_stage_meta), tEtE(_,_,_,_,meta_pipe));
              }

              auto curr_flt_smem_pipe_consumer_state = flt_smem_pipe_consumer_state;
              ++flt_smem_pipe_consumer_state;
              uint32_t skip_wait_flt =
                (ramp_iter == 0) && (t_iter == Flt_T - 1) && (c_iter == c_chunks - 1) &&
                (s_iter == Flt_S - 1) && (r_iter == Flt_R - 1);

              // Do gemm
              // Unroll the K mode manually so we can set scale C to 1
              CUTLASS_PRAGMA_UNROLL
              for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                if (k_block == 1) {
                  flt_barrier_token = flt_pipeline.consumer_try_wait(flt_smem_pipe_consumer_state, skip_wait_flt);
                }
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(
                  tiled_mma.with(
                    static_cast<typename UMMA::BMatrixBufferId>(k_block),
                    act_buffer_status,
                    tCtE(_,_,k_block,meta_pipe)),
                  tCrA(_,_,k_block,read_stage_flt),
                  tCrB(_,_,k_block,read_stage_act),
                  accumulators(_,_,_,current_acc_tmem_pipe_state.index()));
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
              }
              ++current_acc_tmem_pipe_state;
              meta_pipe ^=1;
              flt_pipeline.consumer_release(curr_flt_smem_pipe_consumer_state);

              // Arrive acc_buffer_full
              if (r_iter == (Flt_R - 1) && s_iter == (Flt_S - 1) && c_iter == (c_chunks - 1)
                  && t_iter == (Flt_T - 1)) {
                acc_pipeline.producer_commit(acc_tmem_pipe_producer_release_state);
                ++acc_tmem_pipe_producer_release_state;
              } // The earliest slot to output acc and the last c chunk
            } // r iter
          } // s iter
          act_pipeline.consumer_release(curr_act_smem_pipe_consumer_state);
        } // c iter
      } // t iter
      // Increment the output acc index
      ++output_acc_tmem_pipe_state;
    } // ramp down state

    asm volatile(".pragma \"reset knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}
    return cute::make_tuple(flt_smem_pipe_consumer_state, act_smem_pipe_consumer_state, acc_tmem_pipe_producer_state);
  }

  CUTLASS_DEVICE void
  set_czm_and_bshift(TiledMma& tiled_mma, int s_iter) {
    // ColumnZeroMask and b_shift number in Uri for UMMA
    tiled_mma.mask_and_shift_b_ = column_zero_masks[s_iter];
  }

  CUTLASS_DEVICE auto
  get_act_buffer_status_for_rampup(int s_iter, int r_iter, int ramp_iter) {

    // First rs iter needs:
    //  only use b_keep modifier of UMMA
    // Last rs iter needs:
    //  only use b_reuse modifier of UMMA
    // Other rs iter needs:
    //  both b_keep and b_reuse modifier of UMMA
    auto act_buffer_status = UMMA::BMatrixBufferReuse::Keep;
    if (s_iter == 0 && r_iter == ramp_iter) {
      act_buffer_status = UMMA::BMatrixBufferReuse::Keep;
    }
    else if (s_iter == (Flt_S - 1) && r_iter == 0) {
      act_buffer_status = UMMA::BMatrixBufferReuse::Reuse;
    }
    else {
      act_buffer_status = UMMA::BMatrixBufferReuse::ReuseAndKeep;
    }
    return act_buffer_status;
  }

  CUTLASS_DEVICE auto
  get_act_buffer_status(int s_iter, int r_iter) {

    // First rs iter needs:
    //  only use b_keep modifier of UMMA
    // Last rs iter needs:
    //  only use b_reuse modifier of UMMA
    // Other rs iter needs:
    //  both b_keep and b_reuse modifier of UMMA
    auto act_buffer_status = UMMA::BMatrixBufferReuse::Keep;
    if (s_iter == 0 && r_iter == (Flt_R - 1)) {
      act_buffer_status = UMMA::BMatrixBufferReuse::Keep;
    }
    else if (s_iter == (Flt_S - 1) && r_iter == 0) {
      act_buffer_status = UMMA::BMatrixBufferReuse::Reuse;
    }
    else {
      act_buffer_status = UMMA::BMatrixBufferReuse::ReuseAndKeep;
    }
    return act_buffer_status;
  }

  CUTLASS_DEVICE auto
  get_act_buffer_status_for_rampdown(int s_iter, int r_iter, int ramp_iter) {

    // First rs iter needs:
    //  only use b_keep modifier of UMMA
    // Last rs iter needs:
    //  only use b_reuse modifier of UMMA
    // Other rs iter needs:
    //  both b_keep and b_reuse modifier of UMMA
    auto act_buffer_status = UMMA::BMatrixBufferReuse::Keep;
    if (s_iter == 0 && r_iter == (Flt_R - 1)) {
      act_buffer_status = UMMA::BMatrixBufferReuse::Keep;
    }
    else if (s_iter == (Flt_S - 1) && r_iter == (Flt_R - 1 - ramp_iter)) {
      act_buffer_status = UMMA::BMatrixBufferReuse::Reuse;
    }
    else {
      act_buffer_status = UMMA::BMatrixBufferReuse::ReuseAndKeep;
    }
    return act_buffer_status;
  }

  // Calculate CZM in the local warp(e.g. MMA warp) and store the result directly in local warp's reg. No need to load from smem.
  // It is used for the first wave.
  template <typename ShapeType>
  CUTLASS_DEVICE void
  update_column_zero_mask(ShapeType conv_q, int32_t cta_coord_q, int32_t num_pixels_skip_left) {
    auto column_zero_masks_ =
      UMMA::make_column_zero_mask<ShapeType, Flt_S, size<0>(TileShape{}), size<1>(TileShape{})>
      (conv_q, cta_coord_q, num_pixels_skip_left);

    CUTLASS_PRAGMA_UNROLL
    for (int s_iter = 0; s_iter < Flt_S; s_iter += 1) {
      column_zero_masks[s_iter] = column_zero_masks_[s_iter];
    }
  }

  // Calculate CZM in remote warp(e.g. Sched warp) and store the result into smem. MMA warp needs to load result from smem.
  // It is used for waves except the first one.
  template <typename ShapeType, class CZMPipeline, class CZMPipelineState>
  CUTLASS_HOST_DEVICE auto
  update_column_zero_mask(ShapeType conv_q, int32_t cta_coord_q, int32_t num_pixels_skip_left,
                          UMMA::MaskAndShiftB* czm_ptr, CZMPipeline czm_pipeline, CZMPipelineState czm_pipe_producer_state) {
    auto column_zero_masks_ =
      UMMA::make_column_zero_mask<ShapeType, Flt_S, size<0>(TileShape{}), size<1>(TileShape{})>
      (conv_q, cta_coord_q, num_pixels_skip_left);

    czm_pipeline.producer_acquire(czm_pipe_producer_state);

    CUTLASS_PRAGMA_UNROLL
    for (int s_iter = 0; s_iter < Flt_S; s_iter += 1) {
      uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void*>(&czm_ptr[s_iter + Flt_S * czm_pipe_producer_state.index()]));
      // Store 64-bit czm into smem.
      cutlass::arch::shared_store<sizeof(UMMA::MaskAndShiftB)>(smem_ptr, &column_zero_masks_[s_iter]);
    }
    // Ensure STS is completed before next memory instruction.
    #if defined(__CUDA_ARCH__)
    cutlass::arch::fence_view_async_shared();
    #endif
    czm_pipeline.producer_commit(czm_pipe_producer_state);
    ++czm_pipe_producer_state;
    return czm_pipe_producer_state;
  }

  // Load CZM from smem that Sched warp calculated and stored.
  template<
    class CZMPipeline,
    class CZMPipelineState
  >
  CUTLASS_HOST_DEVICE auto
  load_column_zero_mask(UMMA::MaskAndShiftB* czm_ptr, CZMPipeline czm_pipeline, CZMPipelineState czm_pipe_consumer_state) {
    czm_pipeline.consumer_wait(czm_pipe_consumer_state);
    CUTLASS_PRAGMA_UNROLL
    for (int s_iter = 0; s_iter < Flt_S; s_iter += 1) {
      uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void*>(&czm_ptr[s_iter + Flt_S * czm_pipe_consumer_state.index()]));
      // Load 64-bit czm from smem.
      cutlass::arch::shared_load<sizeof(UMMA::MaskAndShiftB)>(&column_zero_masks[s_iter], smem_ptr);
    }
    // Ensure LDS is completed before next memory instruction.
    #if defined(__CUDA_ARCH__)
    cutlass::arch::fence_view_async_shared();
    #endif
    czm_pipeline.consumer_release(czm_pipe_consumer_state);
    ++czm_pipe_consumer_state;
    return czm_pipe_consumer_state;
  }

private:
  // Column Zero Mask
  UMMA::MaskAndShiftB column_zero_masks [Flt_S] = {0};

  typename Params::TMA_A const* observed_tma_load_flt_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_act_b_ = nullptr;
  typename Params::TMA_E const* observed_tma_load_meta_e_ = nullptr;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
