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
#include "cutlass/conv/collective/builders/sm90_gmma_builder.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<conv::Operator ConvOp, int CapacityBytes, class ElementA, class ElementB, class ElementAAlpha, class ElementABias, class TileShapeMNK, int carveout_bytes>
constexpr int
sm100_compute_stage_count_or_override_batch_norm(StageCountAutoCarveout<carveout_bytes> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto a_alpha_bits = cute::sizeof_bits_v<ElementAAlpha>;
  constexpr auto a_bias_bits = cute::sizeof_bits_v<ElementABias>;
  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    (ConvOp == conv::Operator::kFprop ?
    cutlass::bits_to_bytes(a_alpha_bits * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(a_bias_bits * size<2>(TileShapeMNK{}))
    : 0) +
    static_cast<int>(mainloop_pipeline_bytes) * 2 /* 2 for tma_load_A_pipeline & tma_load_B_pipeline */;

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
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
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cutlass::detail::is_batch_norm_builder_tag_of_v<KernelScheduleType, KernelBatchNormTmaWarpSpecialized1SmSm100> ||
       cutlass::detail::is_batch_norm_builder_tag_of_v<KernelScheduleType, KernelBatchNormTmaWarpSpecialized2SmSm100> ||
       cutlass::detail::is_batch_norm_builder_tag_of_v<KernelScheduleType, KernelBatchNormScheduleAuto>) &&
      ((sizeof(ElementA) * AlignmentA) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0) &&
      ((sizeof(ElementB) * AlignmentB) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0)
    >
> {

  using ElementAAlpha = typename KernelScheduleType::ElementAAlpha;
  using ElementABias = typename KernelScheduleType::ElementABias;

  template<class T>
  using ActivationFunctor = typename KernelScheduleType::template ActivationFunctor<T>;

  static constexpr int AlignmentAAlpha = KernelScheduleType::AlignmentAAlpha;
  static constexpr int AlignmentABias = KernelScheduleType::AlignmentABias;

  static_assert((sizeof(ElementAAlpha) * AlignmentAAlpha) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0, "Unaligned tensor for TMA load");
  static_assert((sizeof(ElementABias) * AlignmentABias) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0, "Unaligned tensor for TMA load");

  // For wgrad, GmemMajorA = MN; For fprop, GmemMajorA = K; For dgrad, it's not supported
  static constexpr cute::UMMA::Major GmemMajorA =
    (ConvOp == conv::Operator::kWgrad) ? cute::UMMA::Major::MN : cute::UMMA::Major::K;
  // Tensor in tmem must be K-major
  static constexpr cute::UMMA::Major UmmaMajorA = cute::UMMA::Major::K;

  // For fprop, GmemMajorB = K; For wgrad, GmemMajorB = MN; For dgrad, it's not supported
  // For B, keep UmmaMajor the same as GmemMajor to avoid transpose on smem, so we only define UmmaMajorB, not GmemMajorB
  static constexpr cute::UMMA::Major UmmaMajorB =
    (ConvOp == conv::Operator::kFprop) ? cute::UMMA::Major::K : cute::UMMA::Major::MN;

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  using TiledMma = decltype(detail::sm100_make_batch_norm_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                                                    TileShape_MNK, ClusterShape_MNK,
                                                                    UmmaMajorA, UmmaMajorB, KernelScheduleType>());

  using AtomThrID = typename TiledMma::AtomThrID;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  // For batch_norm both fprop & wgrad kernel, tensor A uses tma im2col mode and tensor B uses tma tiled mode.
  // Input with transformation can not use TMA 2SM instructions.
  using GmemTiledCopyA = decltype(cutlass::conv::collective::detail::sm100_cluster_shape_to_im2col_tma_atom_A(ClusterShape_MNK{}, Layout<_1>{}));
  // Input without transformation can use TMA 2SM instructions.
  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{}));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      GmemMajorA, ElementAMma, BlockTileA_M, BlockTileA_K>());
  using TmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma, BlockTileA_M, BlockTileA_K>());

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma, BlockTileB_N, BlockTileB_K>());

  // For Fprop, alpha & bias are column vectors, 16dp could reduce usage of register & pressure of smem access
  using CopyAtomR2T = SM100_TMEM_STORE_16dp128b8x;
  using CoptAtomS2R = cute::conditional_t<ConvOp == conv::Operator::kFprop, SM75_U32x4_LDSM_N, SM75_U16x8_LDSM_T>;

  // Calculate SMEM matrix A and B buffers' pipeline stages
  static constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size<0>(AtomThrShapeMNK{});
  static constexpr int MMA_N = cute::size<1>(TileShape_MNK{}) / cute::size<1>(AtomThrShapeMNK{});
  static constexpr uint32_t AccumulatorPipelineStageCount = ((MMA_M >= 128 || (MMA_M == 64 && cute::size<0>(AtomThrShapeMNK{}) == 1))
      && MMA_N * (sizeof(ElementAccumulator) / sizeof(float)) >= 256 ? 1 : 2);
  static constexpr uint32_t SchedulerPipelineStageCount = 1;
  static constexpr uint32_t CLCResponseSize = 16;
  static constexpr uint32_t BatchNormApplyStageCount = 4;

  // BatchNormApplyPipeline = PipelineUmmaConsumerAsync
  static constexpr auto BatchNormApplyPipelineStorage = sizeof(typename cutlass::PipelineUmmaConsumerAsync<BatchNormApplyStageCount>::SharedStorage);
  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrStorage = sizeof(uint32_t);

  static constexpr int KernelSmemCarveout = static_cast<int>(BatchNormApplyPipelineStorage +
                                                             AccumulatorPipelineStorage +
                                                             CLCPipelineStorage +
                                                             LoadOrderBarrierStorage +
                                                             TmemDeallocStorage +
                                                             CLCResponseStorage +
                                                             TmemBasePtrStorage);

  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  static constexpr int PipelineStages = sm100_compute_stage_count_or_override_batch_norm<ConvOp,
      Sm100ReducedSmemCapacityBytes, ElementAMma, ElementBMma, ElementAAlpha, ElementABias, SmemTileShape>(StageCountType{});

  constexpr static int NumSpatialDimensions = detail::gmem_layout_tags_to_spatial_dims<GmemLayoutA, GmemLayoutB>();

  using DispatchPolicy = cutlass::conv::MainloopSm100TmaBatchNormUmmaWarpSpecializedImplicitGemm<
      ConvOp, PipelineStages, BatchNormApplyStageCount, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, NumSpatialDimensions,
      CopyAtomR2T, CoptAtomS2R, ElementAAlpha, ElementABias, ActivationFunctor, ClusterShape_MNK, AlignmentAAlpha, AlignmentABias>;

  using CollectiveOp = cutlass::conv::collective::CollectiveConv<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      ElementB,
      TiledMma,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyA, SmemLayoutAtomA, TmemLayoutAtomA>,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyB, SmemLayoutAtomB>
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
