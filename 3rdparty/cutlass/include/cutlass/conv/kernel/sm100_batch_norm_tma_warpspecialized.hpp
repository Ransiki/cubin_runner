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

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/cluster_sm90.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/conv/detail.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_
>
class ConvUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<cutlass::detail::is_batch_norm_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, KernelScheduleBatchNormTmaWarpSpecializedSm100>>>
{
public:
  //
  // Type Aliases
  //

  // Mainloop derived types
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;

  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  static constexpr int NumSpatialDimensions = CollectiveMainloop::NumSpatialDimensions;
  static constexpr bool IsComplex = false;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;
  // TileID scheduler
  // CLC pipeline depth determines how many waves (stages-1) the scheduler can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = CollectiveMainloop::DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = CollectiveMainloop::DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  using TileSchedulerTag = TileSchedulerTag_;
  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Whether CGA is dynamic or not.
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads          = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMMAThreads            = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads   = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads   = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueThreads       = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps         = NumEpilogueThreads / NumThreadsPerWarp;
  static constexpr uint32_t NumBatchNormApplyThreads = CollectiveMainloop::BatchNormApplyThreadCount;
  static constexpr uint32_t NumBatchNormApplyWarps   = NumBatchNormApplyThreads / NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads +
                                                 NumBatchNormApplyThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumFixupBarriers = 1;

  // Pipelines and pipeline states
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Pipeline and pipeline state types
  using TMALoadAPipeline = typename CollectiveMainloop::TMALoadAPipeline;
  using TMALoadAPipelineState = typename CollectiveMainloop::TMALoadAPipelineState;

  using TMALoadBPipeline = typename CollectiveMainloop::TMALoadBPipeline;
  using TMALoadBPipelineState = typename CollectiveMainloop::TMALoadBPipelineState;

  using BatchNormApplyPipeline = typename CollectiveMainloop::BatchNormApplyPipeline;
  using BatchNormApplyPipelineState = typename CollectiveMainloop::BatchNormApplyPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using TMALoadAPipelineStorage = typename CollectiveMainloop::TMALoadAPipelineStorage;
      using TMALoadBPipelineStorage = typename CollectiveMainloop::TMALoadBPipelineStorage;
      using BatchNormApplyPipelineStorage = typename CollectiveMainloop::BatchNormApplyPipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

      alignas(16) TMALoadAPipelineStorage tma_load_a;
      alignas(16) TMALoadBPipelineStorage tma_load_b;
      alignas(16) BatchNormApplyPipelineStorage batch_norm_apply;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) AccumulatorPipelineStorage accumulator;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];

    // {$nv-internal-release begin}
    // NOTE:
    // Also change CollectiveBuilder's KernelSmemCarveout computation when adopting tmem ptx and remove C++ tmem here
    // This includes all Kernel/CollectiveBuilder that use `KernelTmaWarpSpecializedSm100`
    //   (MainloopSm100TmaUmmaWarpSpecialized, MainloopSm100TmaUmmaWarpSpecializedPlanarComplex)
    // {$nv-internal-release end}
    uint32_t tmem_base_ptr;

    // {$nv-release-never begin}
    // GDC (a.k.a. FDL or "Fast Dependent Launch") MainloopTensorStorage for a dependent kernel,
    // so we move it to the end of SharedStorage.
    // {$nv-release-never end}
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");

  // Host facing host arguments
  struct Arguments {
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    using ProblemShapeMNKL = decltype(CollectiveMainloop::get_implemented_problem_shape_MNKL(cutlass::conv::detail::get_transformed_problem_shape_MNKL(declval<ProblemShape>())));
    ProblemShapeMNKL problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    KernelHardwareInfo hw_info{}; // {$nv-internal-release}
  };

  enum class WarpCategory : int32_t {
    MMA          =   0,
    Sched        =   1,
    MainloopLoad =   2,
    EpilogueLoad =   3,
    Epilogue     =   4,
    BatchNormApply = 8
  };

  struct IsParticipant {
    uint32_t mma              = false;
    uint32_t sched            = false;
    uint32_t main_load        = false;
    uint32_t epi_load         = false;
    uint32_t epilogue         = false;
    uint32_t batch_norm_apply = false;
  };

  template <class ProblemShapeMNKL>
  CUTLASS_HOST_DEVICE
  static auto
  get_linear_problem_shape_MNKL(ProblemShapeMNKL const& problem_shape_mnkl) {
    if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kWgrad) {
      return make_shape(shape<0>(problem_shape_mnkl), shape<1>(problem_shape_mnkl), size<2>(problem_shape_mnkl), Int<1>{});
    }
    else {
      return make_shape(size<0>(problem_shape_mnkl), shape<1>(problem_shape_mnkl), shape<2>(problem_shape_mnkl), Int<1>{});
    }
  }

  // Map user facing arguments to device facing params
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;

    auto problem_shape_mnkl = CollectiveMainloop::get_implemented_problem_shape_MNKL(cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape));

    auto mainloop_params = CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace, args.hw_info);

    auto linear_problem_shape_MNKL = get_linear_problem_shape_MNKL(problem_shape_mnkl);
    auto problem_shape_MNKL = append<4>(problem_shape_mnkl, Int<1>{});

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    void* epilogue_workspace =  workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(linear_problem_shape_MNKL, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ElementAccumulator>(
      args.scheduler, problem_shape_mnkl, TileShape{}, AtomThrShapeMNK{}, ClusterShape{}, args.hw_info, NumFixupBarriers,
      CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    return {
      problem_shape_MNKL,
      mainloop_params,
      CollectiveEpilogue::to_underlying_arguments(problem_shape_MNKL, args.epilogue, epilogue_workspace),
      TileScheduler::to_underlying_arguments(
          linear_problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
          args.hw_info, args.scheduler, scheduler_workspace),
      args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(CollectiveMainloop::get_implemented_problem_shape_MNKL(
        cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape)), args.epilogue);
    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    auto problem_shape_mnkl = CollectiveMainloop::get_implemented_problem_shape_MNKL(cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape));
    auto linear_problem_shape_MNKL = get_linear_problem_shape_MNKL(problem_shape_mnkl);

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(linear_problem_shape_MNKL, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<ElementAccumulator>(
      args.scheduler, linear_problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{}, args.hw_info, NumFixupBarriers,
      CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    auto problem_shape_mnkl = CollectiveMainloop::get_implemented_problem_shape_MNKL(cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape));
    auto linear_problem_shape_MNKL = get_linear_problem_shape_MNKL(problem_shape_mnkl);
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(linear_problem_shape_MNKL, args.epilogue,
        workspace_ptr + workspace_offset, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<ElementAccumulator>(
        args.scheduler, workspace_ptr + workspace_offset, stream, linear_problem_shape_MNKL,
        TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, NumFixupBarriers, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // The CONV mainloop params problem shape will be the cute::Shape<> rank-3 MNK tuple we want for grid planning
    // Although conv problems do not have an L mode, we add it here to comply with the scheduler API
    // {$nv-internal-release begin}
    // XXX: [vht 2023/07] this tiling and linearization is begging to be moved into the collectives and get_tma_tensor
    // and for the mainloop problem shape to be the linearized shape
    // {$nv-internal-release end}
    auto linear_problem_shape_MNKL = get_linear_problem_shape_MNKL(params.problem_shape);
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);

    return TileScheduler::get_grid_shape(
        params.scheduler,
        linear_problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape
        ,params.hw_info // {$nv-internal-release}
       );
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  template <class WorkTileShapeM>
  CUTLASS_DEVICE
  static constexpr auto
  work_tile_to_cta_coord(TileScheduler& scheduler, typename TileScheduler::WorkTileInfo& work_tile_info, WorkTileShapeM work_tile_shape_m) {
    // Need to convert to hierarchal cluster coord before converting to CTA coord for wgrad
    // otherwise linear CTA idx may map CTAs in the same cluster to a different TRS pixel

    if constexpr (DispatchPolicy::ConvOp == cutlass::conv::Operator::kWgrad) {
      auto [m_coord_, n_coord, _, l_coord] = scheduler.work_tile_to_cta_coord(work_tile_info);
      auto m_coord = idx2crd(m_coord_ / size<0>(typename TiledMma::AtomThrID{}), work_tile_shape_m);
      get<0>(m_coord) = get<0>(m_coord) * size<0>(typename TiledMma::AtomThrID{}) + m_coord_ % size<0>(typename TiledMma::AtomThrID{});
      return make_coord(m_coord, n_coord, _, l_coord);
    }
    else {
      return scheduler.work_tile_to_cta_coord(work_tile_info);
    }
  }

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

// {$nv-release-never begin}
    // Inline PTX knobs
    global_knob_elect_one_r2ur_placement();
    global_knob_trywait_sel_sb_dep();
    global_knob_demote_to_pred_blockidx_limit();
    global_knob_mbarrier_init_mapping();
// {$nv-release-never end}

    using namespace cute;
    using X = Underscore;

    // output strides are coalesced so we linearize the output shape to match the shape/stride profiles
    auto linear_problem_shape_MNKL = get_linear_problem_shape_MNKL(params.problem_shape);
    // Separate out problem shape for convenience
    auto [M, N, K, L] = linear_problem_shape_MNKL;

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();

    auto warp_category = warp_idx < static_cast<int>(WarpCategory::BatchNormApply) ?
        (warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx) : WarpCategory::Epilogue) :
        WarpCategory::BatchNormApply;
    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_category == WarpCategory::Sched) && lane_predicate) {
      collective_mainloop.prefetch_tma_descriptors();
    }
    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      collective_epilogue.prefetch_tma_descriptors(params.epilogue);
    }

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
      (warp_category == WarpCategory::Epilogue),                            // epilogue
      (warp_category == WarpCategory::BatchNormApply)                       // batch_norm_apply
    };

    // TMA Load A pipeline
    typename TMALoadAPipeline::Params tma_load_a_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      tma_load_a_pipeline_params.role = TMALoadAPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::BatchNormApply == warp_category) {
      tma_load_a_pipeline_params.role = TMALoadAPipeline::ThreadCategory::Consumer;
    }
    tma_load_a_pipeline_params.is_leader = lane_predicate && is_participant.main_load;
    tma_load_a_pipeline_params.num_consumers = NumBatchNormApplyThreads;
    tma_load_a_pipeline_params.transaction_bytes = CollectiveMainloop::TmaATransactionBytes;
    tma_load_a_pipeline_params.initializing_warp = 0;
    TMALoadAPipeline tma_load_a_pipeline(shared_storage.pipelines.tma_load_a,
                                         tma_load_a_pipeline_params,
                                         cluster_shape);

    // TMA Load B pipeline
    typename TMALoadBPipeline::Params tma_load_b_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      tma_load_b_pipeline_params.role = TMALoadBPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      tma_load_b_pipeline_params.role = TMALoadAPipeline::ThreadCategory::Consumer;
    }
    tma_load_b_pipeline_params.is_leader = lane_predicate && is_participant.main_load && is_mma_leader_cta;
    tma_load_b_pipeline_params.transaction_bytes = CollectiveMainloop::TmaBTransactionBytes;
    tma_load_b_pipeline_params.initializing_warp = 3;
    TMALoadBPipeline tma_load_b_pipeline(shared_storage.pipelines.tma_load_b,
                                         tma_load_b_pipeline_params,
                                         cluster_shape,
                                         cute::true_type{},   // Perform barrier init
                                         cute::false_type{}); // Delay mask calculation

    // Batch Norm Apply pipeline
    typename BatchNormApplyPipeline::Params batch_norm_apply_pipeline_params;
    if (WarpCategory::BatchNormApply == warp_category) {
      batch_norm_apply_pipeline_params.role = BatchNormApplyPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      batch_norm_apply_pipeline_params.role = BatchNormApplyPipeline::ThreadCategory::Consumer;
    }
    batch_norm_apply_pipeline_params.consumer_arv_count = 1;
    batch_norm_apply_pipeline_params.producer_arv_count = size(AtomThrShapeMNK{}) * NumBatchNormApplyThreads;
    batch_norm_apply_pipeline_params.initializing_warp = 7;
    BatchNormApplyPipeline batch_norm_apply_pipeline(shared_storage.pipelines.batch_norm_apply,
                                                     batch_norm_apply_pipeline_params,
                                                     cluster_shape,
                                                     cute::true_type{},  // Perform barrier init
                                                     cute::false_type{}  // Delay mask calculation
                                                     );

    // Epilogue Load pipeline
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.initializing_warp = 4;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Load order barrier
    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = 1;
    load_order_barrier_params.initializing_warp = 5;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    }
    else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_blockid = 0;
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                 (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads + NumBatchNormApplyThreads);
    if (is_epi_load_needed) {
      clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
    }
    clc_pipeline_params.transaction_bytes = CLCResponseSize;
    clc_pipeline_params.initializing_warp = 1;
    CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);

    // Mainloop-Epilogue pipeline
    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    }
    // Only one producer thread arrives on this barrier.
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    accumulator_pipeline_params.initializing_warp = 2;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator,
                                             accumulator_pipeline_params,
                                             cluster_shape,
                                             cute::true_type{},
                                             cute::false_type{});

    // Tmem allocator
    #if defined(CUTE_USE_CPP_SM100_TMEM_ALLOCATION) // {$nv-internal-release begin}
    TmemAllocator tmem_allocator(shared_storage.tmem_allocator_state, cta_rank_in_cluster);
    #else  // CUTE_USE_CPP_SM100_TMEM_ALLOCATION // {$nv-internal-release end}
    TmemAllocator tmem_allocator{};
    #endif // CUTE_USE_CPP_SM100_TMEM_ALLOCATION // {$nv-internal-release}
    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumBatchNormApplyThreads + NumMMAThreads + NumEpilogueThreads,
                                                      cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if (WarpCategory::MMA == warp_category && has_mma_peer_cta && lane_predicate) {
      tmem_deallocation_result_barrier.init(NumMMAThreads);
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    TMALoadAPipelineState tma_load_a_pipe_consumer_state;
    TMALoadAPipelineState tma_load_a_pipe_producer_state = cutlass::make_producer_start_state<TMALoadAPipelineState>();

    TMALoadBPipelineState tma_load_b_pipe_consumer_state;
    TMALoadBPipelineState tma_load_b_pipe_producer_state = cutlass::make_producer_start_state<TMALoadBPipelineState>();

    BatchNormApplyPipelineState batch_norm_apply_pipe_consumer_state;
    BatchNormApplyPipelineState batch_norm_apply_pipe_producer_state = cutlass::make_producer_start_state<BatchNormApplyPipelineState>();

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    // {$nv-internal-release begin}
    // The barrier to ensure all threads are ready to commit Shared memory resizing
    cutlass::arch::SmemEarlyReleaseManager<SharedStorage> smem_early_release_manager;
    // {$nv-internal-release end}

    dim3 block_id_in_cluster = cute::block_id_in_cluster();

    // Calculate mask after cluster barrier arrival
    tma_load_b_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    batch_norm_apply_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, linear_problem_shape_MNKL, TileShape{}, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);

    auto load_inputs = collective_mainloop.load_init(
        linear_problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop);
    Tensor gA_mk = get<0>(load_inputs);                                                          // (TILE_N,TILE_K,n,k)
    auto work_tile_shape_m = shape<2>(gA_mk);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

    auto acc_shape = collective_mainloop.partition_accumulator_shape();

    auto bulk_tmem = TiledMma::make_fragment_C(append(acc_shape,
                                                      Int<AccumulatorPipelineStageCount>{}));

    // __syncthreads() need to be executed at the same PC. // {$nv-release-never}
    pipeline_init_wait(cluster_size);

    // bar.sync for the same id need to be executed in the same branch. // {$nv-release-never}
    if (is_participant.epilogue || is_participant.batch_norm_apply) {
      // Wait for tmem allocation
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem.data() = tmem_base_ptr;
    }

    if (is_participant.main_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;

      do {
        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, linear_problem_shape_MNKL, TileShape{}, shape<3>(gA_mk));
        auto k_tile_count = scheduler.get_work_k_tile_count(work_tile_info, linear_problem_shape_MNKL, TileShape{});
        auto k_tile_prologue = min(min(TMALoadAPipeline::Stages, TMALoadBPipeline::Stages), k_tile_count);

        auto [tma_load_a_pipe_producer_state_next, tma_load_b_pipe_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
          params.mainloop,
          tma_load_a_pipeline,
          tma_load_a_pipe_producer_state,
          tma_load_b_pipeline,
          tma_load_b_pipe_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter, k_tile_prologue
        );
        tma_load_a_pipe_producer_state = tma_load_a_pipe_producer_state_next;
        tma_load_b_pipe_producer_state = tma_load_b_pipe_producer_state_next;

        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }

        auto [tma_load_a_pipe_producer_state_next_, tma_load_b_pipe_producer_state_next_, unused_] = collective_mainloop.load(
          params.mainloop,
          tma_load_a_pipeline,
          tma_load_a_pipe_producer_state,
          tma_load_b_pipeline,
          tma_load_b_pipe_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter_next, k_tile_count - k_tile_prologue
        );
        tma_load_a_pipe_producer_state = tma_load_a_pipe_producer_state_next_;
        tma_load_b_pipe_producer_state = tma_load_b_pipe_producer_state_next_;

        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());
      collective_mainloop.load_tail(tma_load_a_pipeline, tma_load_a_pipe_producer_state,
                                    tma_load_b_pipeline, tma_load_b_pipe_producer_state);
      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the end of mainloop DMA
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}
    }

    else if (is_participant.batch_norm_apply) {
      // M coord should be multimodal for wgrad
      auto cta_coord_mnkl_ = work_tile_to_cta_coord(scheduler, work_tile_info, work_tile_shape_m);
      auto batch_norm_apply_inputs = collective_mainloop.batch_norm_apply_init(params.problem_shape, params.mainloop, bulk_tmem, shared_storage.tensors.mainloop);
      do {
        // Directly load tensors from gmem -> reg
        collective_mainloop.batch_norm_load(params.problem_shape, cta_coord_mnkl_, batch_norm_apply_inputs);
        auto k_tile_count = scheduler.get_work_k_tile_count(work_tile_info, linear_problem_shape_MNKL, TileShape{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl_ = work_tile_to_cta_coord(scheduler, work_tile_info, work_tile_shape_m);

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        auto [tma_load_a_pipe_consumer_state_next, batch_norm_apply_pipe_producer_state_next] = collective_mainloop.batch_norm_apply(
          tma_load_a_pipeline, tma_load_a_pipe_consumer_state,
          batch_norm_apply_pipeline, batch_norm_apply_pipe_producer_state,
          batch_norm_apply_inputs, k_tile_count
        );
        tma_load_a_pipe_consumer_state = tma_load_a_pipe_consumer_state_next;
        batch_norm_apply_pipe_producer_state = batch_norm_apply_pipe_producer_state_next;
      } while (work_tile_info.is_valid());

      batch_norm_apply_pipeline.producer_tail(batch_norm_apply_pipe_producer_state);
      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the end of mainloop DMA
      // This needs to be after the tail to ensure that all MMAs are finished with SMEM.
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}
    }

    else if (is_participant.sched) {
      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the beginning of scheduler warp
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

      // Whether a new CLC query must be performed.
      // See comment below where this variable is updated for a description of
      // why this variable is needed.
      bool requires_clc_query = true;

      do {
        if (requires_clc_query) {
          // Query next clcID and update producer state
          clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
        }

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        // Only perform a new CLC query if we consumed a new CLC query result in
        // `fetch_next_work`. An example of a case in which CLC `fetch_next_work` does
        // not consume a new CLC query response is when processing stream-K units.
        // The current stream-K scheduler uses single WorkTileInfo to track multiple
        // (potentially-partial) tiles to be computed via stream-K. In this case,
        // `fetch_next_work` simply performs in-place updates on the existing WorkTileInfo,
        // rather than consuming a CLC query response.
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
      clc_pipeline.producer_tail(clc_pipe_producer_state);
    }

    else if (is_participant.mma) {
      // Tmem allocation sequence
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem.data() = tmem_base_ptr;
      tmem_allocation_result_barrier.arrive();

      auto mma_inputs = collective_mainloop.mma_init(bulk_tmem, shared_storage.tensors.mainloop);

      do {
        auto k_tile_count = scheduler.get_work_k_tile_count(work_tile_info, linear_problem_shape_MNKL, TileShape{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        Tensor accumulators = bulk_tmem(_,_,_, accumulator_pipe_producer_state.index());

        if (is_mma_leader_cta) {
          // Wait for tmem accumulator buffer to become empty with a flipped phase
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
          auto [batch_norm_apply_pipe_consumer_state_next, tma_load_b_pipe_consumer_state_next] = collective_mainloop.mma(
            batch_norm_apply_pipeline,
            batch_norm_apply_pipe_consumer_state,
            tma_load_b_pipeline,
            tma_load_b_pipe_consumer_state,
            accumulators,
            mma_inputs,
            k_tile_count
          );

          batch_norm_apply_pipe_consumer_state = batch_norm_apply_pipe_consumer_state_next;
          tma_load_b_pipe_consumer_state = tma_load_b_pipe_consumer_state_next;

          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
        }
        ++accumulator_pipe_producer_state;
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // {$nv-internal-release begin}
      // Release shared memory at the end of MMA.
      // Since the persistent loop of MMA is the last place where the shared memory reserved
      // for A and B tiles is consumed, we let the barrier wait here and then commit
      // This will wait on the DMA warp to observe all MMAs, meaning this warp is done using SMEM.
      smem_early_release_manager.resize_for_all_blocking();
      // {$nv-internal-release end}

      if (is_mma_leader_cta) {
        accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
      }

      // Signal to peer MMA that entire tmem allocation can be deallocated
      if constexpr (has_mma_peer_cta) {
        // Leader does wait + arrive, follower does arrive + wait
        tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
        tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
      }

      // Free entire tmem allocation
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (is_participant.epi_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the beginning of epilogue DMA
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

      bool do_load_order_wait = true;
      bool do_tail_load = false;

      do {
        bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

        // Get current work tile and fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        if (compute_epilogue) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            linear_problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            shared_storage.tensors.epilogue
          );

          do_tail_load = true;
        }

        // Calculate the cta coordinates of the next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Only perform a tail load if one of the work units processed performed
      // an epilogue load. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_load) {
        collective_epilogue.load_tail(
          epi_load_pipeline, epi_load_pipe_producer_state,
          epi_store_pipeline, epi_store_pipe_producer_state);
      }
    }
    else if (is_participant.epilogue) {
      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the beginning of epilogue
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

      auto cta_coord_mnkl_ = work_tile_to_cta_coord(scheduler, work_tile_info, work_tile_shape_m);
      bool do_tail_store = false;
      do {
        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        // Accumulator stage slice after making sure allocation has been performed
        Tensor accumulators = bulk_tmem(_,_,_,accumulator_pipe_consumer_state.index()); // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)
        accumulator_pipe_consumer_state = scheduler.template fixup<IsComplex>(
          TiledMma{},
          work_tile_info,
          accumulators,
          accumulator_pipeline,
          accumulator_pipe_consumer_state,
          typename CollectiveEpilogue::CopyOpT2R{}
        );

        //
        // Epilogue and write to gD
        //
        if (scheduler.compute_epilogue(work_tile_info)) {
          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            accumulator_pipeline,
            accumulator_pipe_consumer_state,
            linear_problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl_,
            TileShape{},
            TiledMma{},
            accumulators,
            shared_storage.tensors.epilogue
          );
          epi_load_pipe_consumer_state = load_state_next;
          epi_store_pipe_producer_state = store_state_next;
          accumulator_pipe_consumer_state = acc_state_next;
          do_tail_store = true;
        }
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl_ = work_tile_to_cta_coord(scheduler, work_tile_info, work_tile_shape_m);
      } while (work_tile_info.is_valid());

      // Only perform a tail store if one of the work units processed performed
      // an epilogue. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_store) {
        collective_epilogue.store_tail(
          epi_load_pipeline, epi_load_pipe_consumer_state,
          epi_store_pipeline, epi_store_pipe_producer_state,
          CtaShape_MNK{});
      }
    }

    else {
      // {$nv-internal-release begin}
      // Some warps may not be invoked in warp-specialized branches.
      // These warps do not touch A/B tiles so they hint to resize shared memory directly.
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
