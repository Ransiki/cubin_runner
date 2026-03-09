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
#pragma once
//
// {$nv-internal-release file}
//

// All shared header file should be added in sm100_common.inl instead of builder {$nv-release-never}
#include "cutlass/gemm/collective/builders/sm100_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////
// dense BgradA GEMM: General dense GEMM + input A reduction
template <
  class ArchTag,
  class ElementPairA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    ArchTag,
    arch::OpClassTensorOp,
    ElementPairA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      (cute::is_same_v<ArchTag, arch::Sm100> 
      || cute::is_same_v<ArchTag, arch::Sm107> // {$nv-internal-release}
      ) &&
      (cute::is_tuple<ElementPairA>::value && (not cute::is_tuple<ElementB>::value)) &&
      (cute::is_base_of_v<KernelScheduleSm100BgradAGemm, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) &&
      ((sizeof(remove_cvref_t<decltype(get<0>(ElementPairA{}))>) * AlignmentA) % detail::tma_alignment_bytes == 0) &&
      ((sizeof(ElementB) * AlignmentB) % detail::tma_alignment_bytes == 0)>>
{
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;

  using TiledMma = decltype(detail::sm100_make_trivial_tiled_mma<ElementA, ElementB, ElementAccumulator,
                                                         TileShape_MNK, ClusterShape_MNK,
                                                         UmmaMajorA, UmmaMajorB, BuilderScheduleTag>());
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  using AtomThrID = typename TiledMma::AtomThrID;

  using Min_TileShape_N = std::conditional_t<
      (decltype(cute::size(AtomThrID{}) == Int<2>{})::value),
      decltype(Int<32>{}), decltype(Int<16>{})>;
  using TileShapeBgradA = decltype(make_shape(cute::size<0>(TileShape_MNK{}), Min_TileShape_N{}, cute::size<2>(TileShape_MNK{}))); 
  using TiledMma_BgradA = decltype(detail::sm100_make_trivial_tiled_mma<ElementA, ElementB, ElementAccumulator,
                                                         TileShapeBgradA, ClusterShape_MNK,
                                                         UmmaMajorA, UmmaMajorB, BuilderScheduleTag>());
  using TiledMmaPair = cutlass::gemm::collective::detail::Sm100CollectiveBgradATiledMmaPair<
    TiledMma,
    TiledMma_BgradA
  >;
  using TileShapePair = cutlass::gemm::collective::detail::Sm100CollectiveBgradATileShapePair<TileShape_MNK, TileShapeBgradA>;

  // Define A and B block shapes for reduced size TMA_LOADs
  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}), 
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}), 
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  // This SMEM_B_constant1 provided for the ACC of BgradA and only have 1 k phase to be shared among different k_block loop
  using MmaShapeB_const1_NK = decltype(partition_shape_B(TiledMma_BgradA{}, make_shape(tile_size<1>(TiledMma_BgradA{}), tile_size<2>(TiledMma_BgradA{}))));

  using GmemTiledCopyA = decltype(detail::sm100_cluster_shape_to_tma_atom_A(ClusterShape_MNK{}, AtomThrID{}));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementA, BlockTileA_M, BlockTileA_K>());

  using GmemTiledCopyB = decltype(detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{}));

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementB, BlockTileB_N, BlockTileB_K>());
  using BlockTileB_const1_N = decltype(cute::size<0,0>(MmaShapeB_const1_NK{}) * cute::size<1>(MmaShapeB_const1_NK{}));
  using BlockTileB_1phase_K = decltype(cute::size<0,1>(MmaShapeB_const1_NK{}) * cute::size<2>(MmaShapeB_const1_NK{}));
  using SmemLayoutAtomB_const1 = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementB, BlockTileB_const1_N, BlockTileB_1phase_K>());

  using SmemLayoutAtomPairB = cutlass::gemm::collective::detail::Sm100CollectiveBgradASmemLayoutAtomBPair<SmemLayoutAtomB, 
                                SmemLayoutAtomB_const1>;

  // Calculate SMEM matrix A and B buffers' pipeline stages
  // We need a separate smem space to store 1 for the second MMA tile
  // And this only have 1 stage
  static constexpr int b_const1_bytes_one_stage = static_cast<int>(sizeof(ElementB)) * BlockTileB_const1_N{} * BlockTileB_1phase_K{};

  // static constexpr uint32_t AccumulatorStageCount = 2;
  // Since we have the second group of MMA, not all UTCHMMA tile can support 2 stage ACC.
  // For 2sm version, 256x256 clusterMMA only can handle 1 stage:
  // per CTA: general TMEM: 256 / 2 * 256 * 4B = 128KB and TMEM for reduction A: 256 / 2 * 32 * 4B = 16KB
  // For 1sm version, 128x256 clusterMMA only can handle 1 stage:
  // per CTA: general TMEM: 128 * 256 * 4B = 128KB and TMEM for reduction A: 128 * 16 * 4B = 8KB
  static constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size<0>(AtomThrShapeMNK{});
  static constexpr int MMA_N = cute::size<1>(TileShape_MNK{}) / cute::size<1>(AtomThrShapeMNK{});
  static constexpr bool fail2tmemstage = (MMA_M >= 128) && (MMA_N >= 256);
  static constexpr uint32_t AccumulatorPipelineStageCount = fail2tmemstage ? 1 : 2;

  // Calculate scheduler pipeline stages. Having one more stage than the accumulator allows more latency hiding.
  static constexpr uint32_t SchedulerPipelineStageCount = AccumulatorPipelineStageCount + 1;
  static constexpr bool IsArrayOfPointersGemm = false;
  static constexpr uint32_t KernelSmemCarveout = detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      detail::CLCResponseSize,
      IsArrayOfPointersGemm
    >::KernelSmemCarveout;

  // Reduce SMEM capacity available for buffers considering barrier allocations.
  
  static constexpr int ReducedSmemCapacityBytes = 
    cute::is_same_v<ArchTag, arch::Sm107> ? // {$nv-internal-release}
    cutlass::gemm::collective::detail::sm107_smem_capacity_bytes - (KernelSmemCarveout + b_const1_bytes_one_stage) : // {$nv-internal-release}
    cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - (KernelSmemCarveout + b_const1_bytes_one_stage);

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  using MainloopPipelineStorage = typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage;
  static constexpr int PipelineStages = detail::sm100_compute_stage_count_or_override<
      ReducedSmemCapacityBytes, ElementA, ElementB, SmemTileShape, MainloopPipelineStorage>(StageCountType{});

  // The reduction A acc only need 1x 32dp32bit TMEM_LOAD per thread to load 1st register of column.
  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaWarpSpecializedBgradA<PipelineStages, 
                                                           SchedulerPipelineStageCount, 
                                                           AccumulatorPipelineStageCount,
                                                           ClusterShape_MNK>,
      TileShapePair,
      ElementPairA,
      cutlass::gemm::TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>,
      TiledMmaPair,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomPairB,
      void,
      cute::identity
    >;
};

} // cutlass::gemm::collective