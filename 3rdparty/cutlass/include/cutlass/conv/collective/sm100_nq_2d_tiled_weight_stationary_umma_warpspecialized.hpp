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
  class TileTraitsA_,
  class TileTraitsB_>
struct CollectiveConv<
    MainloopSm100TmaWeightStationaryUmmaWarpSpecializedNq2dTiled<
      ConvOp, AStages, BStages, NumSpatialDims, ClusterShape, KernelSchedule>,
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
  using DispatchPolicy = MainloopSm100TmaWeightStationaryUmmaWarpSpecializedNq2dTiled<
      ConvOp, AStages, BStages, NumSpatialDims, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  // GmemTiledCopyA represents Tiled Copy for Filter
  using GmemTiledCopyA = typename TileTraitsA_::GmemTiledCopy;
  using SmemLayoutFlt = typename TileTraitsA_::SmemLayoutAtom;
  // GmemTiledCopyB represents Tiled Copy for Activation
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;
  // Smem layout that considers halo in both shape and stride, this is used to derive SMEM buffer size
  using SmemLayoutActFull = typename TileTraitsB_::SmemLayoutFullAtom;
  // Smem layout that considers halo in stride but not in shape, this is used to derive UMMA desc
  using SmemLayoutActMma = typename TileTraitsB_::SmemLayoutMmaAtom;
  // Smem layout that doesn't consider halo in shape or stride, this is used to make TileCopy
  using SmemLayoutActTma = typename TileTraitsB_::SmemLayoutTmaAtom;
  using ArchTag = typename DispatchPolicy::ArchTag;

  // Filter shape
  using FilterShape_TRS = typename KernelSchedule::FilterShapeTRS;
  // Traversal stride
  using StrideDHW = typename KernelSchedule::TraveralStrideDHW;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  static constexpr int Act_Stages = DispatchPolicy::BStages;
  static constexpr int Flt_Stages = DispatchPolicy::AStages;

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

  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using TmaInternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>>;
  using TmaInternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>>;

  // Determine MMA type: MMA_1SM vs MMA_2SM
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  // Single-sided usage of the PipelineTmaUmmaAsync class, unlike GEMMs and Implicit GEMMs
  using MainloopFltPipeline = cutlass::PipelineTmaUmmaAsync<
                                Flt_Stages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopPipelineFltParams = typename MainloopFltPipeline::Params;
  using MainloopFltPipelineState = typename cutlass::PipelineState<Flt_Stages>;

  using MainloopActPipeline = cutlass::PipelineTmaUmmaAsync<
                                Act_Stages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopPipelineActParams = typename MainloopActPipeline::Params;
  using MainloopActPipelineState = typename cutlass::PipelineState<Act_Stages>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  // Check consistency of the 3 Act SMEM layouts
  // Consider Halo in shape
  static_assert(shape<0,0>(SmemLayoutActFull{}) >= shape<0,0>(SmemLayoutActTma{}) +  W_Halo_Size, "SmemLayoutActFull considers halos in the shape.");
  static_assert(shape<0,0>(SmemLayoutActFull{}) % (1 << get_swizzle_portion(SmemLayoutActFull{}).num_bits) == 0, "SmemLayoutActFull's shape should be aligned with the swizzle.");
  static_assert(shape(SmemLayoutActMma{}) == shape(SmemLayoutActTma{}), "SmemLayoutActMma doesn't consider halo in shape.");
  // Consider Halo in stride
  static_assert(cosize(SmemLayoutActFull{}) == cosize(make_layout(shape(SmemLayoutActFull{}))), "SmemLayoutActFull should have packed shape and stride.");
  static_assert(cosize(SmemLayoutActTma{})  == cosize(make_layout(shape(SmemLayoutActTma{}))), "SmemLayoutActTma should have packed shape and stride.");
  static_assert(stride(SmemLayoutActMma{}.layout_b()) == stride(SmemLayoutActFull{}.layout_b()), "SmemLayoutActMma considers halo in stride.");

  // TODO: move pipeline mode tiling into the collective setup phase instead https://jirasw.nvidia.com/browse/CFK-22323 // {$nv-release-never}
  static_assert(rank(SmemLayoutFlt{}) == 4, "SmemLayout must be rank 4");
  static_assert((size<0>(TileShape{}) == size<0>(get<0>(SmemLayoutFlt{}))), "SmemLayout must be compatible with the tile shape.");
  static_assert(rank(SmemLayoutActMma{}) == 4, "SmemLayout must be rank 4");
  static_assert((size<1>(TileShape{}) == size<0>(get<0>(SmemLayoutActMma{}))),
      "SmemLayout must be compatible with the tile shape.");

  static_assert(Act_Stages >= 2, "The number of A stages must be at least 2.");
  static_assert(Flt_Stages >= 2, "The number of B stages must be at least 2.");

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
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutFlt>> smem_Flt;
    } tensors;

    using PipelineStorage = typename MainloopFltPipeline::SharedStorage;
    PipelineStorage flt_pipeline;
  };

  using FltTensorStorage = typename FltSharedStorage::TensorStorage;
  using FltPipelineStorage = typename FltSharedStorage::PipelineStorage;

  struct ActSharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutActFull>> smem_Act;
    } tensors;

    using PipelineStorage = typename MainloopActPipeline::SharedStorage;
    PipelineStorage act_pipeline;
  };

  using ActTensorStorage = typename ActSharedStorage::TensorStorage;
  using ActPipelineStorage = typename ActSharedStorage::PipelineStorage;

  // SmemLayoutAct here for TransactionBytes only need to consider wHalo, don't count it rounded up smem rows.
  static constexpr uint32_t ActTransactionBytes =
      ((size<0>(SmemLayoutActTma{}) + (W_Halo_Size * size<1>(get<0>(SmemLayoutActTma{})))) * size<1>(SmemLayoutActTma{}) * size<2>(SmemLayoutActTma{}) *
      static_cast<uint32_t>(sizeof(TmaInternalElementB)));
  static constexpr uint32_t FltTransactionBytes =
      (size<0>(SmemLayoutFlt{}) * size<1>(SmemLayoutFlt{}) * size<2>(SmemLayoutFlt{}) *
      static_cast<uint32_t>(sizeof(TmaInternalElementA)));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    ElementB const* ptr_B{nullptr};
  };

  // Device side kernel params
  struct Params {

    // Assumption: StrideFlt is congruent with Problem_MK
    using ClusterLayout_VMNK = decltype(tiled_divide(
      make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
      make_tile(typename TiledMma::AtomThrID{}))
    );
    using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
      GmemTiledCopyA{},
      make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideFlt{}, int32_t(0)), StrideFlt{}),
      SmemLayoutFlt{}(_,_,_,cute::Int<0>{}),
      TileShape{},
      TiledMma{},
      ClusterLayout_VMNK{})
    );

    // Assumption: StrideAct is congruent with Problem_NK
    using TMA_B = decltype(make_w_tma_atom_B_sm100<TmaInternalElementB>(
      GmemTiledCopyB{},
      make_tensor(
        make_gmem_ptr(recast_ptr<TmaInternalElementB>(nullptr)),
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

    TMA_A tma_load_flt_a;
    TMA_B tma_load_act_b;
    // act w mode TMA does not support mcast, so we don't need act_b_fallback.
    TMA_A tma_load_flt_a_fallback;
    dim3 cluster_shape_fallback;
    // The number of pixels we need to skip on the left.
    // This will control the data flow and kernel implementation details.
    int32_t num_pixels_skip_left;
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
    }
    else {
      observed_tma_load_flt_a_ = &params.tma_load_flt_a;
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

    auto ptr_Flt = recast_ptr<TmaInternalElementB>(args.ptr_B);
    auto ptr_Act = recast_ptr<TmaInternalElementA>(args.ptr_A);

    Tensor tensor_Flt = make_tensor(make_gmem_ptr(ptr_Flt), make_layout(shape_Flt_orig, dFlt));
    Tensor tensor_Act = make_tensor(make_gmem_ptr(ptr_Act), make_layout(shape_Act_orig, dAct));

    auto lower_srt = detail::compute_lower_srt(problem_shape);

    // Only lower_d is used in act tensor iteration.
    // lower_w is calculated in below compte_corner_w().
    // lower_h is not used.
    auto lower_whd = detail::compute_lower_corner_whd(problem_shape);
    cute::array<int32_t, 1> lower_d{lower_whd[2]};

    auto upper_whd = detail::compute_upper_corner_whd(problem_shape);

    int32_t wOffset = 0;
    cute::array<int32_t, 1> start_coord_w{
      lower_whd[0] - wOffset
    };
    int32_t num_pixels_skip_left = cute::min(int32_t(-1 * lower_whd[0]), 2);

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

    typename Params::TMA_A tma_load_flt_a_fallback = make_tma_atom_A_sm100<TmaInternalElementA>(
      GmemTiledCopyA{},
      tensor_Flt,
      SmemLayoutFlt{}(_,_,_,cute::Int<0>{}),
      TileShape{},
      TiledMma{},
      cluster_layout_vmnk_fallback);

    return {
      tma_load_flt_a,
      tma_load_act_b,
      tma_load_flt_a_fallback,
      hw_info.cluster_shape_fallback,
      num_pixels_skip_left
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
    const auto & output_shape      = problem_shape.shape_C;          // [N,Z,P,Q,K]
    const auto & activation_stride = problem_shape.stride_A;
    const auto & filter_stride     = problem_shape.stride_B;
    const auto & traversal_stride  = problem_shape.traversal_stride;

    bool implementable = true;

    // Activation and Filter channel mode extents must match
    if constexpr (ConvOp == conv::Operator::kFprop) {
      // channel count is equal
      implementable &= (activation_shape[NumTensorDimensions-1] == filter_shape[NumTensorDimensions-1]);
    }
    else if constexpr (ConvOp == conv::Operator::kDgrad) {
      // channel count is equal
      implementable &= (activation_shape[NumTensorDimensions-1] == filter_shape[0]);
    }
    // channel mode is major
    implementable &= (activation_stride[NumTensorDimensions-1] == 1);
    implementable &= (filter_stride[NumTensorDimensions-1]     == 1);

    // Filter size
    // Only support 1x3x3 or 3x3x3
    implementable &= (filter_shape[NumTensorDimensions-3] == Flt_R && filter_shape[NumTensorDimensions-2] == Flt_S);
    implementable &= (Flt_R == 3 && Flt_S == 3);
    if constexpr (NumTensorDimensions == 5) {
      implementable &= filter_shape[NumTensorDimensions-4] == Flt_T;
      implementable &= (Flt_T == 1 || Flt_T == 3);
    }

    // Stride
    // Only support 1x1x1
    implementable &= (traversal_stride[NumSpatialDimensions-2] == get<1>(StrideDHW{}) && traversal_stride[NumSpatialDimensions-1] == get<2>(StrideDHW{}));
    implementable &= (get<1>(StrideDHW{}) == 1 && get<2>(StrideDHW{}) == 1);
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

    // Padding limit due to czm.
    implementable &= detail::corner_w_can_implement(problem_shape);

    // H or P dim limit
    // Steady_State = h_pixels_end - h_pixels_start >= 0, as we unroll the whole ramp_up/steady/ramp_down states.
    auto lower_whd = detail::compute_lower_corner_whd(problem_shape);
    auto upper_whd = detail::compute_upper_corner_whd(problem_shape);
    int32_t h_pixels_start = lower_whd[1];
    int32_t h_pixels_end = upper_whd[1] + (Flt_R - 1) + activation_shape[NumTensorDimensions - 3];
    implementable &= (h_pixels_end >= h_pixels_start);

    // Alignment rule
    constexpr int32_t tma_alignment_bits = 128;
    auto shape_Act_orig = problem_shape.get_shape_A();
    auto shape_Flt_orig = problem_shape.get_shape_B();
    constexpr int32_t min_tma_aligned_elements_Act = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable &= cutlass::detail::check_alignment<min_tma_aligned_elements_Act>(shape_Act_orig, StrideAct{});
    constexpr int32_t min_tma_aligned_elements_Flt = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_tma_aligned_elements_Flt>(shape_Flt_orig, StrideFlt{});

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
      }
      else {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_flt_a.get_tma_descriptor());
      }
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_act_b.get_tma_descriptor());
    }
    else {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_flt_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_act_b.get_tma_descriptor());
    }
  }

  /// Prepare for a collective-scoped tensor loading
  /// Must be called once before load()
  /// Producer Perspective
  template <class ProblemShape_MNKZP>
  CUTLASS_DEVICE auto
  load_flt_init(
    ProblemShape_MNKZP const& problem_shape_MNKZP,
    Params const& params,
    FltTensorStorage& shared_tensors) const {

    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M, N, K, Z, P] = problem_shape_MNKZP;
    // mFlt
    Tensor mFlt = observed_tma_load_flt_a_->get_tma_tensor(make_shape(M,K)); // (K,(C,S,R,T))
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

  template <class GTensorPartitionedA, class STensorA, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  load_flt(
    Params const& params,
    MainloopFltPipeline flt_pipeline,
    MainloopFltPipelineState flt_smem_pipe_producer_state,
    cute::tuple<GTensorPartitionedA, STensorA, uint16_t> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    int c_chunks, int h_pixels_start, int h_pixels_end) {

    auto [tAgFlt_mk, tAsFlt, mcast_mask_flt_a] = load_inputs;
    Tensor tAgFlt = tAgFlt_mk(_, get<0>(cta_coord_mnkl), _);

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

              auto loop_coords = get_flt_coord(c_iter, s_iter, r_iter, t_iter);

              copy(observed_tma_load_flt_a_->with(*tma_barrier, mcast_mask_flt_a), tAgFlt(_,loop_coords), tAsFlt(_,write_stage));

              // {$nv-internal-release begin}
              // Before, copy pipeline was: TRYWAIT X+1 -> COPY X -> WAIT X+1 -> TRYWAIT X+2 -> COPY X+1 ...
              // but current compiler hit a false dependency issue on TRYWAIT's scoreboard sometimes,
              // and we need to wait TRYWAIT return between TRYWAIT X+1 and COPY X.
              // To avoid perf issue due to COPY X being issued too late, we move TRYWAIT X+1 after COPY X.
              // Now, copy pipeline is: COPY X -> TRYWATI X+1 -> WAIT X+1 -> COPY X+1...
              // But the LDG latency may potentially exposed due to TRYWATI X+1 is too close to WAIT X+1.
              // {$nv-internal-release end}
              ++flt_smem_pipe_producer_state;
              barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state);
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

              auto loop_coords = get_flt_coord(c_iter, s_iter, r_iter, t_iter);

              copy(observed_tma_load_flt_a_->with(*tma_barrier, mcast_mask_flt_a), tAgFlt(_,loop_coords), tAsFlt(_,write_stage));

              ++flt_smem_pipe_producer_state;
              barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state);
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

              auto loop_coords = get_flt_coord(c_iter, s_iter, r_iter, t_iter);

              copy(observed_tma_load_flt_a_->with(*tma_barrier, mcast_mask_flt_a), tAgFlt(_,loop_coords), tAsFlt(_,write_stage));

              ++flt_smem_pipe_producer_state;
              uint32_t skip_wait = (ramp_iter == 0) && (t_iter == Flt_T - 1) && (c_iter == c_chunks - 1) &&
                                   (s_iter == Flt_S - 1) && (r_iter == Flt_R - 1);
              barrier_token = flt_pipeline.producer_try_acquire(flt_smem_pipe_producer_state, skip_wait);
            } // r iter
          } // s iter
        } // c iter
      } // t iter
    } // ramp state
    return cute::make_tuple(flt_smem_pipe_producer_state);
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_flt_tail(MainloopFltPipeline flt_pipeline, MainloopFltPipelineState flt_smem_pipe_producer_state) {
    flt_pipeline.producer_tail(flt_smem_pipe_producer_state);
  }

  /// Prepare for a collective-scoped tensor loading
  /// Must be called once before load()
  /// Producer Perspective
  template <class ProblemShape_MNKZP>
  CUTLASS_DEVICE auto
  load_act_init(
    ProblemShape_MNKZP const& problem_shape_MNKZP,
    Params const& params,
    ActTensorStorage& shared_tensors) const {

    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M, N, K, Z, P] = problem_shape_MNKZP;

    // mAct
    Tensor mAct = observed_tma_load_act_b_->get_tma_tensor(
      make_shape(N, get<0>(K), P, Z));  // ((QN),C,P,Z)
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
  load_act(
    Params const& params,
    MainloopActPipeline act_pipeline,
    MainloopActPipelineState act_smem_pipe_producer_state,
    cute::tuple<GTensorPartitionedB, STensorB> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    int c_chunks, int h_pixels_start, int h_pixels_end, int z_idx) {

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

    constexpr int w_Halo = Flt_S - 1;
    constexpr int w_Offset = 0;

    //
    // Load Activation
    //

    auto barrier_token = act_pipeline.producer_try_acquire(act_smem_pipe_producer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int h_iter = h_pixels_start; h_iter < h_pixels_end; ++h_iter) {
      CUTLASS_PRAGMA_UNROLL
      for (int d_iter = z_idx ; d_iter < z_idx + Flt_T; ++d_iter) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int c_iter = 0; c_iter < c_chunks; ++c_iter) {
          act_pipeline.producer_acquire(act_smem_pipe_producer_state, barrier_token);

          using BarrierType = typename MainloopActPipeline::ProducerBarrierType;
          BarrierType* tma_barrier = act_pipeline.producer_get_barrier(act_smem_pipe_producer_state);

          int write_stage = act_smem_pipe_producer_state.index();

          copy(observed_tma_load_act_b_->with(*tma_barrier, w_Halo, w_Offset), tBgAct(_,c_iter,h_iter,d_iter), tBsAct(_,write_stage));

          // {$nv-internal-release begin}
          // Before, copy pipeline was: TRYWAIT X+1 -> COPY X -> WAIT X+1 -> TRYWAIT X+2 -> COPY X+1 ...
          // but current compiler hit a false dependency issue on TRYWAIT's scoreboard sometimes,
          // and we need to wait TRYWAIT return between TRYWAIT X+1 and COPY X.
          // To avoid perf issue due to COPY X being issued too late, we move TRYWAIT X+1 after COPY X.
          // Now, copy pipeline is: COPY X -> TRYWATI X+1 -> WAIT X+1 -> COPY X+1...
          // But the LDG latency may potentially exposed due to TRYWATI X+1 is too close to WAIT X+1.
          // {$nv-internal-release end}
          ++act_smem_pipe_producer_state;
          uint32_t skip_wait = (c_iter == c_chunks - 1) && (h_iter == h_pixels_end - 1) && (d_iter == z_idx + Flt_T - 1);
          barrier_token = act_pipeline.producer_try_acquire(act_smem_pipe_producer_state, skip_wait);
        } // c iter
      } // d iter
    } // h iter
    return cute::make_tuple(act_smem_pipe_producer_state);
  }

  CUTLASS_DEVICE void
  load_act_tail(MainloopActPipeline act_pipeline, MainloopActPipelineState act_smem_pipe_producer_state) {
    act_pipeline.producer_tail(act_smem_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  mma_init(FltTensorStorage& flt_shared_tensors, ActTensorStorage& act_shared_tensors) {
    Tensor sFlt = make_tensor(make_smem_ptr(flt_shared_tensors.smem_Flt.data()), SmemLayoutFlt{});          // (BLK_M,BLK_K,PIPE)
    Tensor sAct = make_tensor(make_smem_ptr(act_shared_tensors.smem_Act.data()), SmemLayoutActMma{});       // (BLK_N,BLK_K,PIPE)

    [[maybe_unused]] TiledMma tiled_mma; // suppress spurious MSVC "unused variable" warning

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = tiled_mma.make_fragment_A(sFlt);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = tiled_mma.make_fragment_B(sAct);                                           // (MMA,MMA_N,MMA_K,PIPE)

    return cute::make_tuple(tCrA, tCrB);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgEngine, class FrgLayout,
    class FragmentA, class FragmentB
  >
  CUTLASS_DEVICE auto
  mma(MainloopFltPipeline flt_pipeline, MainloopFltPipelineState flt_smem_pipe_consumer_state,
      MainloopActPipeline act_pipeline, MainloopActPipelineState act_smem_pipe_consumer_state,
      AccumulatorPipeline acc_pipeline, AccumulatorPipelineState acc_tmem_pipe_producer_state,
      cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      int c_chunks, int h_pixels_start, int h_pixels_end,
      cute::tuple<FragmentA, FragmentB> const& mma_inputs) {

    asm volatile(".pragma \"set knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory"); // {$nv-internal-release}
    asm volatile(".pragma \"global knob AdvancedSBDiffXBlockRdSb\";\n" : : : "memory");     // {$nv-internal-release}

    TiledMma tiled_mma;

    auto [tCrA, tCrB] = mma_inputs;

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
                cute::gemm(tiled_mma.with(static_cast<typename UMMA::BMatrixBufferId>(k_block), act_buffer_status),
                  tCrA(_,_,k_block,read_stage_flt), tCrB(_,_,k_block,read_stage_act), accumulators(_,_,_,current_acc_tmem_pipe_state.index()));
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
              }
              ++current_acc_tmem_pipe_state;
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
                cute::gemm(tiled_mma.with(static_cast<typename UMMA::BMatrixBufferId>(k_block), act_buffer_status),
                  tCrA(_,_,k_block,read_stage_flt), tCrB(_,_,k_block,read_stage_act), accumulators(_,_,_,current_acc_tmem_pipe_state.index()));
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
              }
              ++current_acc_tmem_pipe_state;
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
                cute::gemm(tiled_mma.with(static_cast<typename UMMA::BMatrixBufferId>(k_block), act_buffer_status),
                  tCrA(_,_,k_block,read_stage_flt), tCrB(_,_,k_block,read_stage_act), accumulators(_,_,_,current_acc_tmem_pipe_state.index()));
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
              }
              ++current_acc_tmem_pipe_state;
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
  // It is used for waves except for the first one.
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
  UMMA::MaskAndShiftB column_zero_masks [Flt_S] = {};

  typename Params::TMA_A const* observed_tma_load_flt_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_act_b_ = nullptr;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
