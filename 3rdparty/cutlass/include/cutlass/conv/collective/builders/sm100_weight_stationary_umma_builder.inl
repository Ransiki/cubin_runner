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

#include "cutlass/conv/collective/builders/sm100_common.inl"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM100_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

namespace detail {

// Returns the maximum number of filter smem tiles that can be used with a given smem capacity.
template<int CapacityBytes, class ElementFlt, class TileShapeMNK, int ActSmemBytes, bool IsSparse, int carveout_bytes>
constexpr int
compute_weight_stationary_flt_stage_or_override(StageCountAutoCarveout<carveout_bytes> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage);
  constexpr auto flt_bits = cute::sizeof_bits_v<ElementFlt>;
  if constexpr (IsSparse) {
    constexpr int TensorFltSparsity = 2;
    constexpr int flt_stage_bytes =
      // sparse tensor storage
      cutlass::bits_to_bytes(flt_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{}) / TensorFltSparsity) + 
      // metadata storage
      cutlass::bits_to_bytes(size<0>(TileShapeMNK{}) * 128) + 
      // mbarrier
      static_cast<int>(mainloop_pipeline_bytes);
    return (CapacityBytes - ActSmemBytes - carveout_bytes) / flt_stage_bytes;
  } else {
    constexpr int flt_stage_bytes =
      cutlass::bits_to_bytes(flt_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
      static_cast<int>(mainloop_pipeline_bytes);
    return (CapacityBytes - ActSmemBytes - carveout_bytes) / flt_stage_bytes;
  }
}

// Overides stage with manual count.
template<int CapacityBytes, class ElementFlt, class TileShapeMNK, int ActSmemBytes, bool IsSparse, int stages>
constexpr int
compute_weight_stationary_flt_stage_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Overides stage with manual count.
template<int CapacityBytes, class ElementFlt, class TileShapeMNK, int ActSmemBytes, bool IsSparse, int stages>
constexpr int
compute_weight_stationary_flt_stage_or_override(cute::Int<stages> stage_count) {
  return stages;
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////

// UMMA_TMA_WEIGHT_STATIONARY_WS_SS
template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,      // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK,   // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelNq2dTiledTmaWarpSpecializedStride1x1x1Sm100<cute::Shape<_1,_3,_3>>> ||
      cute::is_same_v<KernelScheduleType, KernelNq2dTiledTmaWarpSpecializedStride1x1x1Sm100<cute::Shape<_3,_3,_3>>>>
> {
  static_assert(is_static<TileShape_MNK>::value);
#ifndef CUTLASS_SM100_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM100 Collective Builder\n");
#endif
  static_assert(cutlass::gemm::collective::detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, cutlass::gemm::collective::detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr int Flt_T = cute::get<0>(typename KernelScheduleType::FilterShapeTRS{});
  static constexpr int Flt_R = cute::get<1>(typename KernelScheduleType::FilterShapeTRS{});
  static constexpr int Flt_S = cute::get<2>(typename KernelScheduleType::FilterShapeTRS{});

  // For fprop, major A = K,  major B = K;
  // For dgrad, major A = MN, major B = K;
  static constexpr cute::UMMA::Major UmmaMajorA = (ConvOp == conv::Operator::kFprop) ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
  static constexpr cute::UMMA::Major UmmaMajorB = cute::UMMA::Major::K;

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = decltype(detail::sm100_make_dense_weight_stationary_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                                                                 TileShape_MNK, ClusterShape_MNK,
                                                                                 UmmaMajorA, UmmaMajorB>());

  using AtomThrID = typename TiledMma::AtomThrID;

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using GmemTiledCopyFlt = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));
  using GmemTiledCopyAct = decltype(cutlass::conv::collective::detail::sm100_cluster_shape_to_w_tma_atom(
      ClusterShape_MNK{}, AtomThrID{}));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
    UmmaMajorA, ElementAMma, BlockTileA_M, BlockTileA_K>());

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
    UmmaMajorB, ElementBMma, BlockTileB_N, BlockTileB_K>());

  static constexpr int w_Halo = Flt_S - 1;
  static constexpr int SwizzleRow = get<0>((SmemLayoutAtomB{}).layout_b().shape());
  static constexpr int Pixels_w_Halo = cute::size<1>(TileShape_MNK{}) + w_Halo;
  static constexpr int RoundedShapeB_N = ((Pixels_w_Halo + SwizzleRow - 1) / SwizzleRow) * SwizzleRow;
  static constexpr auto RoundedMmaShapeB_NK_mode0 = replace<0>(get<0>(MmaShapeB_NK{}), Int<RoundedShapeB_N>{});
  static constexpr auto RoundedMmaShapeB_NK = replace<0>(MmaShapeB_NK{}, RoundedMmaShapeB_NK_mode0);

  // Fixed activation buffer stages and set equal to Flt_R
  static constexpr uint32_t ActBPipelineStages = Flt_R;
  // Smem layout that considers halo in both shape and stride; this is used to derive SMEM buffer size
  using SmemLayoutActFull = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(RoundedMmaShapeB_NK, Int<ActBPipelineStages>{}),
      Step<_2,_1,_3>{}));
  // Smem layout that considers halo in stride but not in shape; this is used to derive UMMA desc
  static constexpr auto mocked_layout = replace<0>((SmemLayoutActFull{}).layout_b(),
                                                   make_layout(get<0>(MmaShapeB_NK{}), get<0>((SmemLayoutActFull{}).layout_b().stride())));
  using SmemLayoutActMma = ComposedLayout<decltype((SmemLayoutActFull{}).layout_a()),
                                          decltype((SmemLayoutActFull{}).offset()),
                                          decltype(mocked_layout)>;
  // Smem layout that doesn't consider halo in shape or stride; this is used to make TileCopy
  using SmemLayoutActTma = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<ActBPipelineStages>{}),
      Step<_2,_1,_3>{}));

  static constexpr uint32_t AccumulatorPipelineStageCount = Flt_R;
  static constexpr uint32_t SchedulerPipelineStageCount = 2;
  static constexpr uint32_t CLCResponseSize = 16;
  static constexpr uint32_t TmemAllocationCount = 1;

  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;
  // CZM pipeline storage
  static constexpr auto CZMPipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = TmemAllocationCount * sizeof(uint32_t);
  // CZM size align to 16B
  static constexpr uint32_t ColumnZeroMasksStorage = (Flt_S * sizeof(UMMA::MaskAndShiftB) * SchedulerPipelineStageCount + 15) / 16 * 16;

  // Smem usage that's not part of SharedStorage for tensor
  static constexpr int KernelSmemCarveout = AccumulatorPipelineStorage +
                                            CLCPipelineStorage + 
                                            CLCResponseStorage +
                                            TmemBasePtrsStorage +
                                            ColumnZeroMasksStorage +
                                            CZMPipelineStorage;

  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;
  // Activation SMEM space
  static constexpr int ActSmemBytes = cute::cosize_v<SmemLayoutActFull> * sizeof(ElementB) + 32 /* stage_barrier_bytes */ * ActBPipelineStages;
  static constexpr int FltAPipelineStages = detail::compute_weight_stationary_flt_stage_or_override<
      Sm100ReducedSmemCapacityBytes, ElementAMma, TileShape_MNK, ActSmemBytes, false>(StageCountType{});
  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutFlt = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<FltAPipelineStages>{}),
      Step<_2,_1,_3>{}));

  // GmemLayoutB is activation
  // GmemLayoutA is filter
  constexpr static int NumSpatialDimensions = detail::gmem_layout_tags_to_spatial_dims<GmemLayoutB, GmemLayoutA>();

  using DispatchPolicy = cutlass::conv::MainloopSm100TmaWeightStationaryUmmaWarpSpecializedNq2dTiled<
      ConvOp, FltAPipelineStages, ActBPipelineStages, NumSpatialDimensions, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = cutlass::conv::collective::CollectiveConv<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      ElementB,
      TiledMma,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyFlt, SmemLayoutFlt>,
      detail::Sm100NqTwodTiledWithHaloTileTraits<GmemTiledCopyAct,
                                                 SmemLayoutActFull, SmemLayoutActMma, SmemLayoutActTma>
    >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
