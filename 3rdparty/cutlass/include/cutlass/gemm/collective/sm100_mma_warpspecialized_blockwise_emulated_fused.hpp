/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * {$nv-internal-release file}
 * @file sm100_mma_warpspecialized_blockwise_emulated_fused.hpp
 * @brief Derived from sm100_mma_warpspecialized_emulated.hpp.
 *     Used for 4xFP16 on non-transformed inputs (inputs are original FP32 numbers).
 *     This mainloop expects input-transform kernel scheduling.
 */


#pragma once
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/cluster.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm100.hpp"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Dual pipeline structure for 4xFP16 2SM kernels where each CTA needs to release the computed scaling factors to its peer CTA.
//
// Specifically for this pipeline, it's possible for us to completely skip local pipeline operations:
// - "Stage full" side: SFs are produced the same time as (if not earlier than) ABCompute.
//   By the time accum is ready, SFs must be ready thus no need for waiting.
//
// - "Stage empty" side: Each producer task acquire is equivalent to waiting for (pipeline.count - pipeline.stages) to be empty.
//   Here, the Accumulator task ensures:
//
//   (largest empty mma2accum_pipeline.count) <= AccumPerTile * (largest empty sf_pipeline.count) + AccumPerTile - 1.
//
//   Also, the Mma task (apparently) ensures that:
//   (largest full mma2accum_pipeline.count) <= AccumPerTile * (largest empty transform2mma_pipeline.count) + AccumPerTile - 1;
//
//   Considering that for Mma task we also have acquire equivalent to wait-empty (count - stages):
//   (largest empty mma2accum_pipeline.count) > (largest full mma2accum_pipeline.count) - mma2accum_pipeline.stages
//
//   Connecting the 3 equations we get:
//   (largest empty sf_pipeline.count) > (largest empty mma2accum_pipeline.count) - ceil(mma2accum_pipeline.stages / AccumPerTile)
//
//   This equation means that if:
//   sf_pipeline.stage >= transform2mma_pipe.stages + ceil(mma2accum_pipeline.stages / AccumPerTile),
//   then each time we successfully acquires a stage for transform2mma_pipeline, we would also have implicity also acquired sufficient stages to produce on sf_pipeline.
template <int StageCount, bool SkipLocal>
struct PipelineSm100Async2mEmu4xFP162 {
  static constexpr int StageCountLocal = SkipLocal ? 1 : StageCount;
  using ImplLocal = cutlass::PipelineAsync<StageCountLocal>;
  using ImplRemote = cutlass::PipelineTransactionAsync<StageCount>;
  using Params = typename ImplLocal::Params;
  using ParamsRemote = typename ImplRemote::Params;
  using PipelineState = cutlass::PipelineState<StageCount>;
  using SharedStorage = cute::tuple<typename ImplLocal::SharedStorage, typename ImplRemote::SharedStorage>;
  using ThreadCategory = typename ImplLocal::ThreadCategory;
  using ThreadCategoryRemote = typename ImplRemote::ThreadCategory;
  using ProducerBarrierType = typename ImplRemote::ProducerBarrierType;

  CUTLASS_DEVICE
  cute::tuple<ProducerToken, ProducerToken> producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    ProducerToken local_flag{BarrierStatus::WaitDone};
    if constexpr (!SkipLocal) {
      local_flag = local_pipe.producer_try_acquire(state, skip_wait);
    }
    auto remote_flag = remote_pipe.producer_try_acquire(state, skip_wait);
    return make_tuple(local_flag, remote_flag);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, cute::tuple<ProducerToken, ProducerToken> token) {
    if constexpr (!SkipLocal) {
      local_pipe.producer_acquire(state, get<0>(token));
    }
    remote_pipe.producer_acquire(state, get<1>(token));
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    if constexpr (!SkipLocal) {
      local_pipe.producer_acquire(state);
    }
    remote_pipe.producer_acquire(state);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    if constexpr (!SkipLocal) {
      local_pipe.producer_commit(state);
    }
  }

  // By default, cutlass::PipelineTransactionAsync doesn't emit expect-tx on .shared::cluster
  // This 2SM pipe expect directly on remote mbarrier
  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    auto *barrier = reinterpret_cast<typename ImplRemote::FullBarrier*>(producer_get_barrier(state));
    barrier->arrive_and_expect_tx(transaction_bytes, !blockid);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return remote_pipe.producer_get_barrier(state);
  }

  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    if constexpr (!SkipLocal) {
      local_pipe.producer_tail(state);
    }
    remote_pipe.producer_tail(state);
  }

  CUTLASS_DEVICE
  cute::tuple<ConsumerToken, ConsumerToken> consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    ConsumerToken local_flag{BarrierStatus::WaitDone};
    if constexpr (!SkipLocal) {
      local_flag = local_pipe.consumer_try_wait(state, skip_wait);
    }
    auto remote_flag = remote_pipe.consumer_try_wait(state, skip_wait);
    return make_tuple(local_flag, remote_flag);
  }

  CUTLASS_DEVICE
  cute::tuple<ConsumerToken, ConsumerToken> consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    ConsumerToken local_flag{BarrierStatus::WaitDone};
    if constexpr (!SkipLocal) {
      local_flag = local_pipe.consumer_test_wait(state, skip_wait);
    }
    auto remote_flag = remote_pipe.consumer_test_wait(state, skip_wait);
    return make_tuple(local_flag, remote_flag);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, tuple<ConsumerToken, ConsumerToken> token) {
    if constexpr (!SkipLocal) {
      local_pipe.consumer_wait(state, get<0>(token));
    }
    remote_pipe.consumer_wait(state, get<1>(token));
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    if constexpr (!SkipLocal) {
      local_pipe.consumer_wait(state);
    }
    remote_pipe.consumer_wait(state);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    if constexpr (!SkipLocal) {
      local_pipe.consumer_release(state);
    }
    remote_pipe.consumer_release(state);
  }

  // Constructs two pipelines underneath
  template<class InitBarriers>
  CUTLASS_DEVICE
  PipelineSm100Async2mEmu4xFP162(
    SharedStorage& storage,
    Params const& params,
    InitBarriers = {}) :
      blockid(params.dst_blockid),
      local_pipe(get<0>(storage), params, cute::bool_constant<!SkipLocal>{}),
      remote_pipe(get<1>(storage), translate_params_to_remote(params), true_type{}) {

    static_assert(cute::is_same_v<InitBarriers, cute::true_type>);
  }

protected:
  uint32_t blockid;
  ImplLocal local_pipe;
  ImplRemote remote_pipe;

  // Translates local pipeline params to peer CTA pipeline params
  CUTLASS_DEVICE
  static auto translate_params_to_remote(Params const& params) {
    ParamsRemote params_remote;
    if (params.role == ThreadCategory::Producer) {
      params_remote.role = ThreadCategoryRemote::Producer;
    }
    if (params.role == ThreadCategory::Consumer) {
      params_remote.role = ThreadCategoryRemote::Consumer;
    }
    params_remote.producer_arv_count = 1;
    params_remote.consumer_arv_count = params.consumer_arv_count;
    params_remote.initializing_warp = params.initializing_warp;
    params_remote.dst_blockid = !params.dst_blockid;
    return params_remote;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Compute reciprocal for an input with zero mantissa. Can trace a MUFU cycle with an ALU cycle
template <typename T>
CUTLASS_HOST_DEVICE auto reciprocal_for_sf() {
  if constexpr (is_same_v<T, float>) {
    // Reciprocal function for a float power of 2 can be expressed as:
    return [] (float x) {
      uint32_t out_bits = 0x7f000000 - *reinterpret_cast<uint32_t*>(&x);
      return *reinterpret_cast<float*>(&out_bits);
    };
  }
  else {
    return cutlass::reciprocal_approximate_ftz<T>{};
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop for FastF32 Kernels that needs blockwise exponent aligning (scaling)
template <
  int Load2TransformPipelineStageCount_,
  int Transform2MmaPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  int NumBandsToCompute_,
  int ScalingFactor_,
  int AccPromotionInterval_,
  class AccumulatorCopyAtom_,
  class ClusterShape,
  class TileShape_,
  class StrideA_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomsA_,
  class CopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomsB_,
  class CopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecializedFastF32<
      Load2TransformPipelineStageCount_,
      Transform2MmaPipelineStageCount_,
      SchedulerPipelineStageCount_,
      AccumulatorPipelineStageCount_,
      NumBandsToCompute_,
      ScalingFactor_,
      AccPromotionInterval_,
      half_t,
      ClusterShape,
      AccumulatorCopyAtom_>,
    TileShape_,
    float,
    StrideA_,
    float,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomsA_,
    CopyAtomsA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomsB_,
    CopyAtomsB_,
    TransformB_>
{
  //
  // Type Aliases
  //

  // Determine MMA type: MMA_1SM vs MMA_2SM
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;
  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedFastF32<
                            Load2TransformPipelineStageCount_,
                            Transform2MmaPipelineStageCount_,
                            SchedulerPipelineStageCount_,
                            AccumulatorPipelineStageCount_,
                            NumBandsToCompute_,
                            ScalingFactor_,
                            AccPromotionInterval_,
                            half_t,
                            ClusterShape,
                            AccumulatorCopyAtom_>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using CtaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using CtaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  using ElementA = float;
  using PackedElementA = float2;
  using StrideA = StrideA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using PackedElementAMma = uint32_t;
  using ElementB = float;
  using PackedElementB = float2;
  using StrideB = StrideB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using PackedElementBMma = uint32_t;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;
  using CopyAtomsA = CopyAtomsA_;
  using CopyAtomsB = CopyAtomsB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(cute::is_same_v<ElementA, float>, "Input type A should be float");
  static_assert(cute::is_same_v<ElementB, float>, "Input type B should be float");
  static_assert(cute::is_same_v<ElementAMma, cutlass::half_t>, "Compute type A should be cutlass::half_t");
  static_assert(cute::is_same_v<ElementBMma, cutlass::half_t>, "Compute type A should be cutlass::half_t");

  using Load2TransformPipeline = cutlass::PipelineTmaTransformAsync<
                             DispatchPolicy::Load2TransformPipelineStageCount,
                             AtomThrShapeMNK>;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;

  using Transform2MmaPipeline = cutlass::PipelineUmmaConsumerAsync<
                              DispatchPolicy::Transform2MmaPipelineStageCount,
                              AtomThrShapeMNK>;
  using Transform2MmaPipelineState = typename Transform2MmaPipeline::PipelineState;

  // Allocate enough stages for SF so that we can skip local pipeline waits
  /// NOTE: Keeping regular syncs & waits for 1Sm cases. Potentially we can improve them as well
  constexpr static int SFPipelineStageCount = DispatchPolicy::Transform2MmaPipelineStageCount +
      ceil_div(DispatchPolicy::Schedule::AccumulatorPipelineStageCount, size<2>(CtaShapeA_MK{}) / DispatchPolicy::AccPromotionInterval);
  using SFPipeline = conditional_t<size(AtomThrShapeMNK{})==2, detail::PipelineSm100Async2mEmu4xFP162<SFPipelineStageCount, true>, cutlass::PipelineAsync<SFPipelineStageCount>>;
  using SFPipelineState = typename SFPipeline::PipelineState;

  using Mma2AccumPipeline =  cutlass::PipelineUmmaAsync<
                              DispatchPolicy::Schedule::AccumulatorPipelineStageCount,
                              AtomThrShapeMNK>;
  using Mma2AccumPipelineState = typename Mma2AccumPipeline::PipelineState;

  // Thread Counts
  static constexpr uint32_t NumTransformationThreads = 128;
  static constexpr uint32_t NumAccumThreads = 128;

  // Get the Algorithm parameters
  constexpr static int NumComputeMtxs = 2;
  constexpr static int NumBandsToCompute = DispatchPolicy::NumBandsToCompute;
  constexpr static int ScalingFactor = DispatchPolicy::ScalingFactor;
  constexpr static int AccPromotionInterval = DispatchPolicy::AccPromotionInterval;
  constexpr static int AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  constexpr static int StagesPerTile = size<2>(CtaShapeA_MK{}) / DispatchPolicy::AccPromotionInterval;
  constexpr static int NumBandsMax = 3;
  static_assert(NumBandsToCompute <= NumBandsMax && NumBandsToCompute >= NumComputeMtxs, "NumBandsToCompute should be less than maximum number of bands");
  static_assert(StagesPerTile * AccPromotionInterval == size<2>(CtaShapeA_MK{}), "PromotionInterval doesn't evenly divide CTA shape");

  // Copy atom for Accumulator
  /// NOTE: This is not actually used. Proper TmemCopyAtom is picked from epilogue instead.
  using AccumulatorCopyAtom = typename DispatchPolicy::AccumulatorCopyAtom;

  static_assert(NumBandsToCompute == 3 || NumBandsToCompute == 2, "4xFP16 with 3/2 Bands are supported");

  using SmemLayoutAtomA = typename SmemLayoutAtomsA::InputLayoutAtom;
  using SmemLayoutAtomACompute = typename SmemLayoutAtomsA::ComputeLayoutAtom;
  using SmemLayoutAtomB = typename SmemLayoutAtomsB::InputLayoutAtom;
  using SmemLayoutAtomBCompute = typename SmemLayoutAtomsB::ComputeLayoutAtom;

  using InputCopyAtomA = typename CopyAtomsA::InputCopyAtom;
  using ComputeCopyAtomA = typename CopyAtomsA::ComputeCopyAtom;
  using InputCopyAtomB = typename CopyAtomsB::InputCopyAtom;
  using ComputeCopyAtomB = typename CopyAtomsB::ComputeCopyAtom;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(CtaShapeA_MK{}) * size<1>(CtaShapeA_MK{})) % size<0>(SmemLayoutAtomACompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");
  static_assert(((size<0,1>(CtaShapeA_MK{}) * size<2>(CtaShapeA_MK{})) % size<1>(SmemLayoutAtomACompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(CtaShapeB_NK{}) * size<1>(CtaShapeB_NK{})) % size<0>(SmemLayoutAtomBCompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");
  static_assert(((size<0,1>(CtaShapeB_NK{}) * size<2>(CtaShapeB_NK{})) % size<1>(SmemLayoutAtomBCompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(CtaShapeA_MK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutACompute = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomACompute{},
      append(append(CtaShapeA_MK{}, Int<NumComputeMtxs>{}), Int<DispatchPolicy::Transform2MmaPipelineStageCount>{})));

  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(CtaShapeB_NK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutBCompute = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomBCompute{},
      append(append(CtaShapeB_NK{}, Int<NumComputeMtxs>{}), Int<DispatchPolicy::Transform2MmaPipelineStageCount>{})));

  // Scaling factor layout:
  template <class Tile, class Layout>
  static constexpr auto
  sf_layout_at_transform(Tile mn_tile, Layout input_layout) {
    auto mn_mma = shape<0,0>(input_layout);
    auto st_mn_mma = conditional_t<is_same_v<_1, decltype(shape<1>(input_layout))>, _0, decltype(mn_mma)>{};
    auto st_stages = composition(make_layout(size<3>(input_layout), mn_tile), make_layout(shape<3>(input_layout))).stride();
    auto stride = make_stride(make_tuple(_1{}, _0{}), st_mn_mma, _0{}, st_stages);
    auto sf_layout = make_layout(shape(input_layout), stride);
    return sf_layout;
  }

  // Input layout with proper stride-0 modes:
  using SmemLayoutSFA = decltype(sf_layout_at_transform(get<0>(CtaShape_MNK{}),
      UMMA::tile_to_mma_shape(SmemLayoutAtomACompute{}.layout_b(), append(CtaShapeA_MK{}, Int<SFPipelineStageCount>{}))));

  using SmemLayoutSFB = decltype(sf_layout_at_transform(get<1>(CtaShape_MNK{}),
      UMMA::tile_to_mma_shape(SmemLayoutAtomBCompute{}.layout_b(), append(CtaShapeB_NK{}, Int<SFPipelineStageCount>{}))));

  // Accum layout with proper stride-0 modes: (M, fake-N) / (fake-M, N)
  using SmemAccumLayoutSFA = decltype(make_layout(
      append(take<0,2>(CtaShape_MNK{}),Int<SFPipelineStageCount>{}), make_stride(_1{},_0{},get<0>(CtaShape_MNK{}))));
  using SmemAccumLayoutSFB = decltype(make_layout(
      append(take<0,2>(CtaShape_MNK{}),Int<SFPipelineStageCount>{}), make_stride(_0{},_1{},get<1>(CtaShape_MNK{}))));

  static_assert(DispatchPolicy::Load2TransformPipelineStageCount >= 2 && DispatchPolicy::Load2TransformPipelineStageCount >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert((cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value ||
                 cute::is_base_of<cute::UMMA::tmem_frg_base,      typename TiledMma::FrgTypeA>::value  ) &&
                 cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                 "MMA atom must A operand from SMEM or TMEM and B operand from SMEM for this mainloop.");
  static_assert((cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>),
                 "GmemTiledCopyA - invalid TMA copy atom specified.");
  static_assert((cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>),
                 "GmemTiledCopyB -  invalid TMA copy atom specified.");

  struct PipelineStorage {
    using Load2TransformPipelineStorage = typename Load2TransformPipeline::SharedStorage;
    alignas(16) Load2TransformPipelineStorage load2transform_pipeline;
    using Transform2MmaPipelineStorage = typename Transform2MmaPipeline::SharedStorage;
    alignas(16) Transform2MmaPipelineStorage transform2mma_pipeline;
    using SFPipelineStorage = typename SFPipeline::SharedStorage;
    alignas(16) SFPipelineStorage sf_pipeline;
    using Mma2AccumPipelineStorage = typename Mma2AccumPipeline::SharedStorage;
    alignas(16) Mma2AccumPipelineStorage mma2accum_pipeline;
  };

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      struct TensorStorageUntransformed {
        cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
        cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
      };

      struct TensorStorageTransformedAinSmem {
        alignas(1024) cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutACompute>> smem_ACompute;
        alignas(1024) cute::ArrayEngine<ElementBMma, cute::cosize_v<SmemLayoutBCompute>> smem_BCompute;
      };

      union TensorStorageTransformedAinTmem {
        alignas(1024) cute::ArrayEngine<ElementAMma, 1> smem_ACompute;  // No smem_ACompute
        alignas(1024) cute::ArrayEngine<ElementBMma, cute::cosize_v<SmemLayoutBCompute>> smem_BCompute;
      };

      using TensorStorageTransformed = cute::conditional_t<
                                      cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value,
                                      TensorStorageTransformedAinSmem,
                                      TensorStorageTransformedAinTmem>;

      TensorStorageUntransformed input;
      TensorStorageTransformed compute;

      alignas(128) cute::array<ElementA, cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(128) cute::array<ElementB, cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Different from other GEMM kernels, both CTAs should be aware of loads. Both CTAs will work on
  // loaded input A and B matrices to convert the data type
  static constexpr uint32_t TmaTransactionBytes =
    cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * size<2>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value))+
    cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * size<2>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<ElementA>(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );
    using TMA_B = decltype(make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor tensor_a = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    // Cluster layout for TMA construction
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<ElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<ElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    return {
      tma_load_a,
      tma_load_b,
      tma_load_a_fallback,
      tma_load_b_fallback,
      hw_info.cluster_shape_fallback
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE static void
  prefetch_tma_descriptors(Params const& params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == params.cluster_shape_fallback.x && cs.y == params.cluster_shape_fallback.y);
      if (is_fallback_cluster) {
        cute::prefetch_tma_descriptor(params.tma_load_a_fallback.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b_fallback.get_tma_descriptor());
      }
      else {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
      }
    }
    else {
      cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    }
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  /// Produce the inputs to the transform threads by loading inputs from gmem -> smem
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
      Params const& params,
      Load2TransformPipeline pipeline,
      Load2TransformPipelineState load2xform_pipeline_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b] = load_inputs;

    // slice out the work coord from tiled tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    uint32_t skip_wait = (k_tile_count <= 0);
    auto pipeline_flag = pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK mainloop_load2xform_pipeline_state for _writing_
      pipeline.producer_acquire(load2xform_pipeline_state, pipeline_flag);
      int write_stage = load2xform_pipeline_state.index();

      using BarrierType = typename Load2TransformPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(load2xform_pipeline_state);

      // Advance mainloop_pipe
      ++load2xform_pipeline_state;
      skip_wait = (k_tile_count <= 1);
      pipeline_flag = pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

      copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
      copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
      ++k_tile_iter;
    }
    return cute::make_tuple(load2xform_pipeline_state, k_tile_iter);
  }

  /// Set up the data needed by this collective for load.
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tiled tensor for input A
  /// gB_nkl - The tiled tensor for input B
  // Other inputs needed for load(): partitioned AB tensors for gmem and smem, and mcast masks
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_storage) const {
    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);          // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});  // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    return cute::make_tuple(
        gA_mkl, gB_nkl,                        // for scheduler
        tAgA_mkl, tBgB_nkl, tAsA, tBsB,        // for input tensor values
        mcast_mask_a, mcast_mask_b);           // multicast masks
  }

  template<
    class KTileIterator, class Accumulator,
    class GTensorA, class DstCopyA, class SrcTensorA, class DstTensorA, class TensorSFA, class ThrKA,
    class GTensorB,                 class SrcTensorB, class DstTensorB, class TensorSFB, class ThrKB
  >
  CUTLASS_DEVICE auto
  transform(
      Load2TransformPipeline load2transform_pipeline,
      Load2TransformPipelineState load2transform_pipeline_consumer_state,
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_producer_state,
      SFPipeline sf_pipeline,
      SFPipelineState sf_pipeline_producer_state,
      Accumulator accumulators,
      cute::tuple<GTensorA, DstCopyA, SrcTensorA, DstTensorA, TensorSFA, ThrKA,
                  GTensorB,           SrcTensorB, DstTensorB, TensorSFB, ThrKB> input_operands,
      KTileIterator k_tile_iter, int k_tile_count) {

    static_assert(cute::is_same_v<ElementA, ElementB>, "ElementA and ElementB types should be the same.");
    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma types should be the same.");

    cutlass::arch::NamedBarrier transform_bar(NumTransformationThreads, cutlass::arch::ReservedNamedBarriers::TransformBarrier);

    // tAsA : (Copy,#Copy),MMA_Rest,MMA_M_Rest,MMA_K_Rest, SmemStages (In SMEM)
    // tAdA : (Copy,#Copy),MMA_Rest,MMA_M_Rest,MMA_K_Rest, NumComputeMtxs, SmemStages (In SMEM or TMEM)
    // tBsB : (Copy,#Copy),MMA_Rest,MMA_N_Rest,MMA_K_Rest, SmemStages (In SMEM)
    // tBsB : (Copy,#Copy),MMA_Rest,MMA_N_Rest,MMA_K_Rest, NumComputeMtxs, SmemStages (In SMEM)
    auto [unused_tAgA, dst_copy_A, tAsA, tAdACompute, tAsSFA, thrK_A,
          unused_tBgB,             tBsB, tBsBCompute, tBsSFB, thrK_B] = input_operands;

    // Create the tensors in registers
    auto tArA = make_tensor<ElementA>(tAsA(_,_,_,_,0).shape());
    auto tArA_temp = make_tensor<ElementA>(tAsA(_,_,_,_,0).shape());
    auto tArACompute = make_tensor<ElementAMma>(tAsA(_,_,_,_,0).shape());

    auto tBrB = make_tensor<ElementB>(tBsB(_,_,_,_,0).shape());
    auto tBrB_temp = make_tensor<ElementB>(tBsB(_,_,_,_,0).shape());
    auto tBrBCompute = make_tensor<ElementBMma>(tBsB(_,_,_,_,0).shape());

    auto tArA_x2 = recast<Array<ElementA,2>>(tArA);
    auto tArA_temp_x2 = recast<Array<ElementA,2>>(tArA_temp);
    auto tArACompute_x2 = recast<Array<ElementAMma,2>>(tArACompute);

    auto tBrB_x2 = recast<Array<ElementB,2>>(tBrB);
    auto tBrB_temp_x2 = recast<Array<ElementB,2>>(tBrB_temp);
    auto tBrBCompute_x2 = recast<Array<ElementBMma,2>>(tBrBCompute);

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
    auto transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);
    auto sf_flag = sf_pipeline.producer_try_acquire(sf_pipeline_producer_state, skip_wait);

    /// NOTE: In principle, tArA can be elementwise-scaled with tAsSFA. Recasting & blocking here is for saving register space.
    /// Assumption is exploited here that Smem is always K-major. Hence any stride-0 mode in tSF should be broadcasted.
    /// For that pupose, it's fine to use tSF in a filtered (reinterpreted) way.
    auto tArSFA = make_tensor<ElementA>(filter(tAsSFA(_,_,_,_,0)).shape());
    auto tBrSFB = make_tensor<ElementB>(filter(tBsSFB(_,_,_,_,0)).shape());
    using ArrayA_xN = Array<ElementA, decltype(size(tArA) / size(tArSFA)){}>;
    using ArrayB_xN = Array<ElementB, decltype(size(tBrB) / size(tBrSFB)){}>;
    // Group elements according stride-0 modes -> Vector scaling according to each group
    auto tArA_xN = recast<ArrayA_xN>(tArA);
    auto tBrB_xN = recast<ArrayB_xN>(tBrB);
    cutlass::maximum_absolute_value_zero_mantissa_reduction<ArrayA_xN> reducer_A;
    cutlass::maximum_absolute_value_zero_mantissa_reduction<ArrayB_xN> reducer_B;
    cutlass::multiplies<ArrayA_xN> rescaler_A;
    cutlass::multiplies<ArrayB_xN> rescaler_B;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      load2transform_pipeline.consumer_wait(load2transform_pipeline_consumer_state, load2transform_flag);
      transform2mma_pipeline.producer_acquire(transform2mma_pipeline_producer_state, transform2mma_flag);
      sf_pipeline.producer_acquire(sf_pipeline_producer_state, sf_flag);

      int load2transform_consumer_index = load2transform_pipeline_consumer_state.index();
      int transform2mma_producer_index = transform2mma_pipeline_producer_state.index();
      int sf_stage_idx = sf_pipeline_producer_state.index();

      auto curr_load2transform_pipeline_consumer_state = load2transform_pipeline_consumer_state;
      auto curr_transform2mma_pipeline_producer_state = transform2mma_pipeline_producer_state;
      auto curr_sf_pipeline_producer_state = sf_pipeline_producer_state;

      // Copy the input B matrix from SMEM
      copy(AutoVectorizingCopy{}, tBsB(_,_,_,_,load2transform_consumer_index), tBrB);
      // Copy the input A matrix from SMEM
      copy(AutoVectorizingCopy{}, tAsA(_,_,_,_,load2transform_consumer_index), tArA);

      { // Gather max exponents from loaded input
        // Init & intra-thread reduction
        cute::transform(tBrSFB, [&] (auto x) { return static_cast<ElementB>(0); });
        cute::transform(tArSFA, [&] (auto x) { return static_cast<ElementA>(0); });
        cute::transform(tBrSFB, tBrB_xN, tBrSFB, reducer_B);
        cute::transform(tArSFA, tArA_xN, tArSFA, reducer_A);

        int32_t FULL_MASK = 0xffffffff;
        if (thrK_B > 1) {
          if constexpr (sizeof_bits_v<ElementB> == 32 && decltype(size(tBrSFB)){} > 1) {
            auto tBrSFB_x2 = recast<Array<ElementB,2>>(tBrSFB);
            // Shuffling reduction -- vectorized
            CUTE_UNROLL
            for (int delta_tid = 1; delta_tid < thrK_B; delta_tid *= 2) {
              cute::transform(tBrSFB_x2, tBrSFB_x2, [&] (auto x2) -> Array<ElementB,2> {
                  int64_t x2i = *reinterpret_cast<int64_t*>(&x2);
                  int64_t y2i = __shfl_xor_sync(FULL_MASK, x2i, delta_tid);
                  auto y2 = *reinterpret_cast<Array<ElementB,2>*>(&y2i);
                  return { fast_max(x2[0], y2[0]), fast_max(x2[1], y2[1]) };
              });
            }
          }
          else {
            // Shuffling reduction
            CUTE_UNROLL
            for (int delta_tid = 1; delta_tid < thrK_B; delta_tid *= 2) {
              cute::transform(tBrSFB, tBrSFB, [&] (auto x) { return fast_max(x, __shfl_xor_sync(FULL_MASK, x, delta_tid)); });
            }
          }
          jetfire::warp_switch(); // {$nv-internal-release}
        }
        if (thrK_A > 1) {
          // Shuffling reduction
          CUTE_UNROLL
          for (int delta_tid = 1; delta_tid < thrK_A; delta_tid *= 2) {
            cute::transform(tArSFA, tArSFA, [&] (auto x) { return fast_max(x, __shfl_xor_sync(FULL_MASK, x, delta_tid)); });
          }
        }

        // Release to Smem
        copy(tArSFA, filter(tAsSFA(_,_,_,_,sf_stage_idx)));
        copy(tBrSFB, filter(tBsSFB(_,_,_,_,sf_stage_idx)));

        // Release to peer Smem
        if constexpr (size(AtomThrShapeMNK{}) == 2) {
          if (threadIdx.x % NumTransformationThreads == 0) {
            sf_pipeline.producer_expect_transaction(sf_pipeline_producer_state,
                                                    NumTransformationThreads * size(tBrSFB) * sizeof_bits_v<ElementB> / 8);
          }
          uint32_t dst_blockid = !cute::block_rank_in_cluster();
          uint32_t barrier_ptr = cute::cast_smem_ptr_to_uint(sf_pipeline.producer_get_barrier(sf_pipeline_producer_state));
          Tensor tBsSFB_u32 = recast<uint32_t>(filter(tBsSFB(_,_,_,_,sf_stage_idx)));
          Tensor tBrSFB_u32 = recast<uint32_t>(tBrSFB);

          CUTE_UNROLL
          for (int is = 0; is < size(tBrSFB_u32); ++is) {
            uint32_t dst_smem_addr = cute::cast_smem_ptr_to_uint(&tBsSFB_u32(is));
            cute::store_shared_remote(tBrSFB_u32(is), dst_smem_addr, barrier_ptr, dst_blockid);
          }
        }

        // Reciprocal
        cute::transform(tArSFA, tArSFA, detail::reciprocal_for_sf<ElementA>());
        cute::transform(tBrSFB, tBrSFB, detail::reciprocal_for_sf<ElementB>());
      }

      // Rescaling
      cute::transform(tBrB_xN, tBrSFB, tBrB_xN, rescaler_B);

      CUTE_UNROLL
      for (int comp_mtx_index = 0; comp_mtx_index < NumComputeMtxs; ++comp_mtx_index) {
        // Convert from fp32 -> fp16
        // {$nv-internal-release begin}
        /// NOTE: cutlass::NumericArrayConverter<half_t, float, 2, cutlass::FloatRoundStyle::round_to_nearest_satfinite> fails.
        /// Either of the following works:
        /// - cutlass::NumericArrayConverter<half_t, float, 2, cutlass::FloatRoundStyle::round_to_nearest> // no saturation
        /// - cutlass::NumericConverter<half_t, float, cutlass::FloatRoundStyle::round_to_nearest_satfinite> // no array conversion
        /// CuTe bug? No such instruction?
        // {$nv-internal-release end}
        cute::transform(tBrB_x2, tBrBCompute_x2, cutlass::NumericArrayConverter<ElementBMma, ElementB, 2, cutlass::FloatRoundStyle::round_to_nearest>::convert);
        copy(AutoVectorizingCopy{}, tBrBCompute, tBsBCompute(_,_,_,_,comp_mtx_index,transform2mma_producer_index));

        // if it is not the last compute matrix, scale and substract
        if (comp_mtx_index < NumComputeMtxs - 1) {
          // Convert from fp16 -> fp32 to substract
          cute::transform(tBrBCompute_x2, tBrB_temp_x2, cutlass::NumericArrayConverter<ElementB, ElementBMma, 2, cutlass::FloatRoundStyle::round_to_nearest>::convert);
          cute::transform(tBrB_x2, tBrB_temp_x2, tBrB_x2, cutlass::minus<Array<ElementB,2>>{});
          if constexpr (DispatchPolicy::ScalingFactor != 0) {
            cute::transform(tBrB_x2, tBrB_x2, cutlass::scale<Array<ElementB,2>>{(1 << DispatchPolicy::ScalingFactor)});
          }
        }
      }

      // Loads from SMEM are done. Signal the mainloop load as early as possible
      transform_bar.sync();
      load2transform_pipeline.consumer_release(curr_load2transform_pipeline_consumer_state);

      // Rescaling
      cute::transform(tArA_xN, tArSFA, tArA_xN, rescaler_A);

      CUTE_UNROLL
      for (int comp_mtx_index = 0; comp_mtx_index < NumComputeMtxs; ++comp_mtx_index) {
        // Convert from fp32 -> fp16
        /// NOTE: Same as above {$nv-internal-release}
        cute::transform(tArA_x2, tArACompute_x2, cutlass::NumericArrayConverter<ElementAMma, ElementA, 2, cutlass::FloatRoundStyle::round_to_nearest>::convert);
        copy(dst_copy_A, tArACompute, tAdACompute(_,_,_,_,comp_mtx_index,transform2mma_producer_index));

        // if it is not the last compute matrix, scale and substract
        if (comp_mtx_index < NumComputeMtxs - 1) {
          // Convert from fp16 -> fp32 to substract
          cute::transform(tArACompute_x2, tArA_temp_x2, cutlass::NumericArrayConverter<ElementA, ElementAMma, 2, cutlass::FloatRoundStyle::round_to_nearest>::convert);
          cute::transform(tArA_x2, tArA_temp_x2, tArA_x2, cutlass::minus<Array<ElementA,2>>{});
          if constexpr (DispatchPolicy::ScalingFactor != 0) {
            cute::transform(tArA_x2, tArA_x2, cutlass::scale<Array<ElementA,2>>{(1 << DispatchPolicy::ScalingFactor)});
          }
        }
      }

      // fence for SMEM writes
      cutlass::arch::fence_view_async_shared();
      if constexpr (is_tmem<decltype(tAdACompute)>::value) {
        // fence for TMEM writes if A operand is coming from TMEM
        cutlass::arch::fence_view_async_tmem_store();
      }

      // Let the MMA know we are done transforming
      transform2mma_pipeline.producer_commit(curr_transform2mma_pipeline_producer_state);
      sf_pipeline.producer_commit(curr_sf_pipeline_producer_state);
      // Next pipeline stage
      ++load2transform_pipeline_consumer_state;
      ++transform2mma_pipeline_producer_state;
      ++sf_pipeline_producer_state;

      skip_wait = (k_tile_count <= 1);
      // Peek the next pipeline stage's barriers
      load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
      transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);
      sf_flag = sf_pipeline.producer_try_acquire(sf_pipeline_producer_state, skip_wait);
    }
    return cute::make_tuple(load2transform_pipeline_consumer_state, transform2mma_pipeline_producer_state, sf_pipeline_producer_state);
  }

  template<class ProblemShape_MNKL, class Accumulator>
  CUTLASS_DEVICE auto
  transform_init(
      Params const& params,
      ProblemShape_MNKL const& problem_shape_MNKL,
      Accumulator accumulators,
      TensorStorage& shared_storage) {
    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    Tensor sA_orig = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});
    Tensor sA = as_position_independent_swizzle_tensor(sA_orig);
    Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});

    Tensor sB_orig = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});
    Tensor sB = as_position_independent_swizzle_tensor(sB_orig);
    Tensor sBCompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_BCompute.begin()), SmemLayoutBCompute{});

    // 2Sm collaboratively transform B, while accum access the entire N-tile.
    // Hence, each CTA holds a complete copy of SFB, but transform() only access a half.
    auto offset_SFB = cute::block_rank_in_cluster() * cosize_v<decltype(SmemLayoutSFB{}(_,_,_,0))>;
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_SFB.data() + offset_SFB), SmemLayoutSFB{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_SFA.data()), SmemLayoutSFA{});

    // Map input, compute, and fragment tensors to
    //   Copy strategies and partitioned tensors. These will become the input
    //   operands of the transform function. Depending on MMA atom type, the
    //   operands can reside in SMEM or TMEM
    auto setup_copy_ops = [&] (
        auto tensor_input,
        auto input_copy_atom,
        auto tensor_compute,
        auto tensor_sf,
        auto make_fragment,
        auto compute_copy_atom) constexpr {
      auto fragment_compute = make_fragment(tensor_compute);
      if constexpr (cute::is_tmem<cute::remove_cvref_t<decltype(fragment_compute)>>::value) {
        // For M=128 with 2CTA MMA atoms, the TMEM tensor for A has a duplicated allocation.
        // Instead of allocation a 64x16 TMEM tensor, we have a 128x16 allocation
        // See: TmemAllocMode::Duplicated.
        auto tensor2x = [&] (auto &tensor) constexpr {
        if constexpr (decltype(size<0,0>(fragment_compute) == Int<128>{} && size<0,0>(tensor) == Int<64>{})::value) {
          return make_tensor(tensor.data(),
                             logical_product(tensor.layout(),
                                             make_tile(make_tile(Layout<_2,_0>{},_),_,_,_)));   // ((128,16),m,k,PIPE)
          }
          else {
            return tensor;
          }
        };
        Tensor tensor_input2x = tensor2x(tensor_input);
        Tensor tensor_sf2x = tensor2x(tensor_sf);

        fragment_compute.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
        // If operand comes from TMEM, create the TMEM_STORE based copy // {$nv-internal-release}
        auto reg2tmem_tiled_copy = make_tmem_copy(compute_copy_atom, fragment_compute(_,_,_,0,0));
        auto thr_reg2tmem_tiled_copy = reg2tmem_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
        auto partitioned_tensor_input = thr_reg2tmem_tiled_copy.partition_S(tensor_input2x);
        auto partitioned_tensor_sf = thr_reg2tmem_tiled_copy.partition_S(tensor_sf2x);
        auto partitioned_tensor_compute = thr_reg2tmem_tiled_copy.partition_D(fragment_compute);
        // Identify how the TV-layout divides into the K mode of SF tensors
        auto thr_layout_sf = get<0>(reg2tmem_tiled_copy.tidfrg_S(tensor_sf2x.layout()));
        auto thrcnt_k = Int<NumTransformationThreads / cosize_v<decltype(thr_layout_sf)>>{};
        static_assert(thr_layout_sf(0) == thr_layout_sf(thrcnt_k-1), "Unexpected thread partitioning for scale factors.");
        return cute::make_tuple(reg2tmem_tiled_copy, partitioned_tensor_input, partitioned_tensor_compute, partitioned_tensor_sf, thrcnt_k);
      }
      else {
        // If the operand comes from SMEM, create STS based copy. // {$nv-internal-release}
        auto tensor_compute_ind_sw = as_position_independent_swizzle_tensor(tensor_compute);
        auto reg2smem_tiled_copy = make_cotiled_copy(compute_copy_atom, Layout<Shape <_128,_8>, Stride<  _8,_1>>{},
                                                     tensor_compute(_,_,_,0,0).layout());

        // Source copy is based on the source operand of STS copy. // {$nv-internal-release}
        auto thr_reg2smem_tiled_copy = reg2smem_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
        auto partitioned_tensor_input = thr_reg2smem_tiled_copy.partition_S(tensor_input);
        auto partitioned_tensor_sf = thr_reg2smem_tiled_copy.partition_S(tensor_sf);
        auto partitioned_tensor_compute = thr_reg2smem_tiled_copy.partition_D(tensor_compute_ind_sw);
        // Identify how the TV-layout divides into the K mode of SF tensors
        auto thr_layout_sf = get<0>(reg2smem_tiled_copy.tidfrg_S(tensor_sf.layout()));
        auto thrcnt_k = Int<NumTransformationThreads / cosize_v<decltype(thr_layout_sf)>>{};
        static_assert(thr_layout_sf(0) == thr_layout_sf(thrcnt_k-1), "Unexpected thread partitioning for scale factors.");

        return cute::make_tuple(AutoVectorizingCopy{}, partitioned_tensor_input, partitioned_tensor_compute, partitioned_tensor_sf, thrcnt_k);
      }
    };

    auto [dst_copy_A, tAsA, tAsACompute, tAsSFA, thrK_A] =
        setup_copy_ops(sA, InputCopyAtomA{}, sACompute, sSFA, [&](auto &arg) {return TiledMma::make_fragment_A(arg);}, ComputeCopyAtomA{});

    auto [dst_copy_B, tBsB, tBsBCompute, tBsSFB, thrK_B] =
        setup_copy_ops(sB, InputCopyAtomB{}, sBCompute, sSFB, [&](auto &arg) {return TiledMma::make_fragment_B(arg);}, ComputeCopyAtomB{});

    return cute::make_tuple(gA_mkl, dst_copy_A, tAsA, tAsACompute, tAsSFA, thrK_A,
                            gB_nkl,             tBsB, tBsBCompute, tBsSFB, thrK_B);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgEngine, class FrgLayout,
    class TensorA, class TensorB
  >
  CUTLASS_DEVICE auto
  mma(
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_consumer_state,
      Mma2AccumPipeline mma2accum_pipeline,
      Mma2AccumPipelineState mma2accum_pipeline_producer_state,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      cute::tuple<TensorA, TensorB> const& input_operands,
      int k_tile_count
      // {$nv-internal-release begin}
      , bool zero_accumulator = true
      , bool a_negate = false
      // {$nv-internal-release end}
  ) {
    TiledMma tiled_mma;

    auto curr_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;
    auto next_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;
    uint32_t skip_wait = (k_tile_count <= 0);
    auto transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);
    ++next_transform2mma_pipeline_consumer_state;

    // tCrA : (MMA), MMA_M, MMA_K, NumComputeMtxs, SmemStage  (In SMEM or TMEM)
    //      We use SMEM stages to match #buffers in Load <-> Convert
    // tCrB : (MMA), MMA_N, MMA_K, NumComputeMtxs, SmemStages (In SMEM)
    auto const [tCrA, tCrB] = input_operands;

    using ZeroScaler = cute::integral_constant<uint32_t, 0>;
    using Scaler = cute::integral_constant<uint32_t, ScalingFactor>;

    int remaining_accum_promotions = k_tile_count * StagesPerTile;
    uint32_t mma2accum_skip_wait = (remaining_accum_promotions <= 0);
    auto mma2accum_flag = mma2accum_pipeline.producer_try_acquire(mma2accum_pipeline_producer_state, mma2accum_skip_wait);

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      transform2mma_pipeline.consumer_wait(curr_transform2mma_pipeline_consumer_state, transform2mma_flag);

      int transform2mma_pipeline_consumer_state_index = curr_transform2mma_pipeline_consumer_state.index();

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); k_block += DispatchPolicy::AccPromotionInterval, --remaining_accum_promotions) {
        mma2accum_pipeline.producer_acquire(mma2accum_pipeline_producer_state, mma2accum_flag);

        int mma2accum_pipeline_producer_state_index = mma2accum_pipeline_producer_state.index();
        auto tCtC = accumulators(_,_,_,mma2accum_pipeline_producer_state_index);
        auto curr_mma2accum_pipeline_producer_state = mma2accum_pipeline_producer_state;

        ++mma2accum_pipeline_producer_state;
        mma2accum_skip_wait = (remaining_accum_promotions <= 1);
        mma2accum_flag = mma2accum_pipeline.producer_try_acquire(mma2accum_pipeline_producer_state, mma2accum_skip_wait);

        auto tCrA0 = tCrA(_,_,_,0,transform2mma_pipeline_consumer_state_index);
        auto tCrA1 = tCrA(_,_,_,1,transform2mma_pipeline_consumer_state_index);

        auto tCrB0 = tCrB(_,_,_,0,transform2mma_pipeline_consumer_state_index);
        auto tCrB1 = tCrB(_,_,_,1,transform2mma_pipeline_consumer_state_index);

        // MMA instructions Emulation
        auto accumulate = UMMA::ScaleOut::Zero;
        // {$nv-internal-release begin}
        if (!zero_accumulator) {
          accumulate = UMMA::ScaleOut::One;
        }
        if (a_negate) {
          tiled_mma.idesc_.a_negate_ ^= 1;
        }
        // {$nv-internal-release end}

        // First set of GEMMs that we need to perform for each band are unrolled to set compile-time constant
        // scaling parameter. Scaled GEMM operations are only needed for the first MMA operation of each band.

        // Band 3
        if constexpr (NumBandsToCompute >= 3) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block), tCrB1(_,_,k_block), tCtC);         // A[1]*B[1]
          accumulate = UMMA::ScaleOut::One;
          CUTLASS_PRAGMA_UNROLL
          for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
            cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block+s), tCrB1(_,_,k_block+s), tCtC);   // A[1]*B[1]
          }
        }
        // Band 2
        cute::gemm(tiled_mma.with(accumulate, Scaler{}), tCrA0(_,_,k_block), tCrB1(_,_,k_block), tCtC);               // A[0]*B[1]
        accumulate = UMMA::ScaleOut::One;
        cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block), tCrB0(_,_,k_block), tCtC);           // A[1]*B[0]
        CUTLASS_PRAGMA_UNROLL
        for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA0(_,_,k_block+s), tCrB1(_,_,k_block+s), tCtC);     // A[0]*B[1]
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block+s), tCrB0(_,_,k_block+s), tCtC);     // A[1]*B[0]
        }
        // Band 1
        cute::gemm(tiled_mma.with(accumulate, Scaler{}), tCrA0(_,_,k_block), tCrB0(_,_,k_block), tCtC);               // A[0]*B[0]
        CUTLASS_PRAGMA_UNROLL
        for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA0(_,_,k_block+s), tCrB0(_,_,k_block+s), tCtC);     // A[0]*B[0]
        }
        mma2accum_pipeline.producer_commit(curr_mma2accum_pipeline_producer_state);
      }

      transform2mma_pipeline.consumer_release(curr_transform2mma_pipeline_consumer_state);

      skip_wait = (k_tile_count <= 1);
      transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);

      curr_transform2mma_pipeline_consumer_state = next_transform2mma_pipeline_consumer_state;
      ++next_transform2mma_pipeline_consumer_state;
    }
    return cute::make_tuple(curr_transform2mma_pipeline_consumer_state, mma2accum_pipeline_producer_state);
  }

  template<class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  mma_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TensorStorage& shared_storage) const {
    TiledMma tiled_mma;

    auto get_tCrA = [&] () constexpr {
      if constexpr (cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value) {
        Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});
        return tiled_mma.make_fragment_A(sACompute);
      }
      else {
        auto tCrA = tiled_mma.make_fragment_A(shape(SmemLayoutACompute{}));
        tCrA.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
        return tCrA;
      }
    };

    Tensor tCrA = get_tCrA();
    Tensor sBCompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_BCompute.begin()), SmemLayoutBCompute{});
    Tensor tCrB = tiled_mma.make_fragment_B(sBCompute);
    return cute::make_tuple(tCrA, tCrB);
  }

  template<class FrgEngine, class FrgLayout, class TmemCopyAtom, class EpilogueTile>
  CUTLASS_DEVICE auto
  accum_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TmemCopyAtom tmem_cp_atom, EpilogueTile epilogue_tile, TensorStorage &shared_storage) {
    // Obtain a single accumulator
    Tensor tAcc = tensor<0>(accumulators(_,_,_,_0{}));
    // Apply epilogue subtiling
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    // Create the TMEM copy for single EpilogueTile.
    // Note that EpilogueTile = CtaTile for NoSmem epilogue
    auto tiled_t2r = make_tmem_copy(tmem_cp_atom, tAcc_epi(_,_,_0{},_0{}));
    auto thread_t2r = tiled_t2r.get_slice(threadIdx.x % size(tiled_t2r));
    Tensor tTR_gC   = thread_t2r.partition_D(tAcc_epi);
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_gC));                               // (T2R,T2R_M,T2R_N)
    Tensor tTR_rGlobAcc = make_tensor<ElementAccumulator>(shape(tTR_gC));                           // (T2R,T2R_M,T2R_N)
    static_assert(is_same_v<decltype(shape<2>(tAcc_epi)), _1>, "EpilogueTile must match CtaTile in M-mode to enable TMEM access.");

    Tensor sSFA = make_tensor(make_smem_ptr(shared_storage.smem_SFA.data()), SmemAccumLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_storage.smem_SFB.data()), SmemAccumLayoutSFB{});

    // Map scaling factor tensors to accumulation layouts
    auto tiled_s2r = make_tiled_copy_D(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementB>{}, tiled_t2r);
    Tensor tBsSFB = thread_t2r.partition_D(flat_divide(sSFB, EpilogueTile{}));
    Tensor tAsSFA = thread_t2r.partition_D(flat_divide(sSFA, EpilogueTile{}));
    Tensor tBrSFB = make_tensor<ElementB>(shape(tTR_rAcc));                                      // (T2R,T2R_M,T2R_N)
    static_assert(cosize_v<decltype(tAsSFA(_,_,_,_,_,0).layout())> == 1, "SFA should be thread-scalar.");

    // Apply epilogue subtiling to bulk accumulator
    // We need to tile the whole bulk_tmem allocation with EpilogueTile.
    // The accumulation should be aware of the AccumulatorPipelineStages
    Tensor tBulkAcc_epi = flat_divide(accumulators(make_coord(_,_),_0{},_0{}, _), EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N,PIPE)
    Tensor tTR_tBulkAcc = thread_t2r.partition_S(tBulkAcc_epi);                                           // (T2R,T2R_M,T2R_N,EPI_M,EPI_N,PIPE)
    return cute::make_tuple(tiled_t2r, thread_t2r, tiled_s2r, tTR_tBulkAcc, tTR_rAcc, tTR_rGlobAcc, tAsSFA, tBsSFB, tBrSFB);
  }

  template<class TiledCopy, class ThrCopy, class TiledCopySF, class AccumulatorTensor, class LocalAccFrg, class GlobalAccFrg, class TensorSFA, class TensorSFB, class FrgSFB>
  CUTLASS_DEVICE auto
  accum(cute::tuple<TiledCopy, ThrCopy, TiledCopySF, AccumulatorTensor, LocalAccFrg, GlobalAccFrg, TensorSFA, TensorSFB, FrgSFB> accum_inputs,
        Mma2AccumPipeline mma2accum_pipeline,
        Mma2AccumPipelineState mma2accum_pipeline_consumer_state,
        SFPipeline sf_pipeline,
        SFPipelineState sf_pipeline_consumer_state,
        int k_tile_count) {
    auto [tiled_t2r, thread_t2r, tiled_s2r, tTR_tBulkAcc,
          tTR_rAcc, tTR_rGlobAcc, tAsSFA, tBsSFB, tBrSFB] = accum_inputs;


    Tensor tTR_rAcc_float2 = recast<Array<ElementAccumulator,2>>(tTR_rAcc);                       // (T2R/2,T2R_M,T2R_N)
    Tensor tTR_rGlobAcc_float2 = recast<Array<ElementAccumulator,2>>(tTR_rGlobAcc);               // (T2R/2,T2R_M,T2R_N)
    Tensor tBrSFB_float2 = recast<Array<ElementB,2>>(tBrSFB);                                     // (T2R/2,T2R_M,T2R_N)
    Tensor tBsSFB_u32 = recast<uint32_t>(tBsSFB);
    cutlass::plus<Array<ElementAccumulator,2>> add_float2{};
    cutlass::multiplies<Array<ElementAccumulator,2>> mul_float2{};

    // Clear the global accumulator
    CUTE_UNROLL
    for (int i = 0; i<size(tTR_rGlobAcc); i++) {
      tTR_rGlobAcc(i) = ElementAccumulator(0);
    }

    uint32_t skip_wait = 0;
    auto mma2accum_flag = mma2accum_pipeline.consumer_try_wait(mma2accum_pipeline_consumer_state, skip_wait);
    auto sf_flag = sf_pipeline.consumer_try_wait(sf_pipeline_consumer_state, skip_wait);

    // 1. Global periodic accumulation in registers
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      int sf_stage_idx = sf_pipeline_consumer_state.index();
      sf_pipeline.consumer_wait(sf_pipeline_consumer_state, sf_flag);

      // Thread-scalar SFA
      ElementA tArSFA = tAsSFA(0,0,0,0,0,sf_stage_idx);
      Array<ElementA,2> tArSFA_float2{ tArSFA, tArSFA };

      // The stage is limited to a CTA tile
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = 0; k_block<StagesPerTile; k_block++) {
        int mma2accum_pipeline_consumer_state_index = mma2accum_pipeline_consumer_state.index();
        mma2accum_pipeline.consumer_wait(mma2accum_pipeline_consumer_state, mma2accum_flag);
        auto prev_state = mma2accum_pipeline_consumer_state;

        copy(tiled_t2r, tTR_tBulkAcc(_,_,_,_,_,mma2accum_pipeline_consumer_state_index), tTR_rAcc);
        copy(tiled_s2r, tBsSFB(_,_,_,_,_,sf_stage_idx), tBrSFB);
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < size(tTR_rAcc_float2); ++ii) {
          tBrSFB_float2(ii) = mul_float2(tBrSFB_float2(ii), tArSFA_float2);
          tTR_rAcc_float2(ii) = mul_float2(tTR_rAcc_float2(ii), tBrSFB_float2(ii));
          tTR_rGlobAcc_float2(ii) = add_float2(tTR_rGlobAcc_float2(ii), tTR_rAcc_float2(ii));
          jetfire::warp_switch(); // {$nv-release-internal}
        }

        cutlass::arch::fence_view_async_tmem_load(); // Need a fence bw TMEM_LOAD and arrive
        mma2accum_pipeline.consumer_release(mma2accum_pipeline_consumer_state);

        ++mma2accum_pipeline_consumer_state;
        skip_wait = ((k_tile_count <= 1) && (k_block >= (StagesPerTile-1)));
        mma2accum_flag = mma2accum_pipeline.consumer_try_wait(mma2accum_pipeline_consumer_state, skip_wait);
      }
      sf_pipeline.consumer_release(sf_pipeline_consumer_state);
      ++sf_pipeline_consumer_state;
      sf_flag = sf_pipeline.consumer_try_wait(sf_pipeline_consumer_state, skip_wait);
    }
    return cute::make_tuple(mma2accum_pipeline_consumer_state, sf_pipeline_consumer_state, tTR_rGlobAcc);
  }

protected:

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  constexpr auto
  tile_input_tensors(Params const& params, ProblemShape_MNKL const& problem_shape_MNKL) const {
    using X = cute::Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  typename Params::TMA_A const* observed_tma_load_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_b_ = nullptr;

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
