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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/cuda_ptx_global_knobs.h" // {$nv-release-never}
#include "cutlass/detail/cluster.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_atom.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                        KernelSparseTmaWarpSpecializedSm100> ||
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                        KernelSparseTmaWarpSpecializedBlockScaledSm100>>
  >
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  // [AG] This is for the Cutlass compatability, but this is very dangerous because of the StrideA is not similar to other kernels'. {$nv-release-never}
  using LayoutA   = typename CollectiveMainloop::LayoutA;
  using StrideA   = remove_cvref_t<decltype(LayoutA{}.stride())>;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using ElementE  = typename CollectiveMainloop::ElementE;
  using LayoutE   = typename CollectiveMainloop::LayoutE;
  using LayoutSFA = typename cutlass::detail::LayoutSFAType<CollectiveMainloop>::type;
  using LayoutSFB = typename cutlass::detail::LayoutSFBType<CollectiveMainloop>::type;
  using ElementSF = typename cutlass::detail::ElementSFType<CollectiveMainloop>::type;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  static constexpr bool IsBlockscaled = cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                                                            KernelSparseTmaWarpSpecializedBlockScaledSm100>;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool IsComplex = CollectiveEpilogue::NumAccumulatorMtxs == 2;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using TileSchedulerTag = TileSchedulerTag_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsModsEnabled = TileScheduler::IsModsEnabled; // {$nv-release-never}
  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps       = NumEpilogueThreads / NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_load_pipe_increment(CtaShape_MNK{});

  // Fixup performed for split-/stream-K is done across warps in different CTAs
  // at epilogue subtile granularity. Thus, there must be one barrier per sub-tile per
  // epilogue warp.
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Pipeline and pipeline state types
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  // Kernel level shared memory storage
  struct SharedStorage {
    // {$nv-release-never begin}
    // * NOTE:
    // If you change `SharedStorage`
    //   Also change `sm100_sparse_umma_builder.inl`'s `sm100_compute_stage_count_or_override_sparse()`
    // {$nv-release-never end}

    // Barriers should be allocated in lower 8KB of SMEM for SM100
    // See: https://nvbugswb.nvidia.com/NvBugs5/ArchBug.aspx?bugid=4336796  // {$nv-release-never}
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      // {$nv-release-never begin}
      // * Padding Notice
      // MainloopPipelineStorage is 24 bytes for each Stage
      // If Stage is odd, 8 bytes is padded after MainloopPipelineStorage & before EpiLoadPipelineStorage
      //   to satisfy struct alignment requirement
      // This 8 bytes padding is considered in `sm100_compute_stage_count_or_override_sparse()`
      // {$nv-release-never end}
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) AccumulatorPipelineStorage accumulator;
      alignas(16) CLCThrottlePipelineStorage clc_throttle;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;
    // {$nv-release-never begin}
    // * Padding Notice
    // TensorStorage Alignment = max(128, alignof(EpilogueTensorStorage), alignof(MainloopTensorStorage))
    //    `TensorStorage tensors` start from TensorStorage Alignment.
    // This padding is **NOT** considered in `sm100_compute_stage_count_or_override_sparse()`
    // {$nv-release-never end}

    // {$nv-release-never begin}
    // GDC (a.k.a. FDL or "Fast Dependent Launch") MainloopTensorStorage for a dependent kernel,
    // so we move it to the end of SharedStorage.
    // {$nv-release-never end}
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

      EpilogueTensorStorage epilogue;
      // {$nv-release-never begin}
      // * Padding Notice
      // If alignof(MainloopTensorStorage) > alignof(EpilogueTensorStorage), padding will be added between epilogue and mainloop
      //    `MainloopTensorStorage mainloop` start from MainloopTensorStorage Alignment
      // `sm100_compute_stage_count_or_override_sparse()` **ONLY** assume 128 alignment
      // {$nv-release-never end}
      MainloopTensorStorage mainloop;
      // {$nv-release-never begin}
      // * Padding Notice
      // If alignof(EpilogueTensorStorage) > alignof(MainloopTensorStorage), padding will be added after mainloop
      // `sm100_compute_stage_count_or_override_sparse()` **ONLY** assume 128 alignment
      // {$nv-release-never end}
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Host facing host arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{}; 
  };

  enum class WarpCategory : int32_t {
    MMA          = 0,
    Sched        = 1,
    MainloopLoad = 2,
    EpilogueLoad = 3,
    Epilogue     = 4
  };

  struct IsParticipant {
    uint32_t mma       = false;
    uint32_t sched     = false;
    uint32_t main_load = false;
    uint32_t epi_load  = false;
    uint32_t epilogue  = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count != 0) {
      CUTLASS_TRACE_HOST("  WARNING: SM100 tile scheduler does not allow for user specified SM counts.\n"
          "  To restrict a kernel's resource usage, consider using CUDA driver APIs instead (green contexts).");
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    const uint32_t ktile_start_alignment_count = 2u;

    // Epilogue
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::StreamKScheduler> && not IsBlockscaled) {
      workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers,
        /*epilogue_subtile=*/1, /*num_accumulator_mtx=*/1,
        ktile_start_alignment_count);
    }
    else {
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    }
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    auto scheduler_params = [&]() {
      if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::StreamKScheduler> && not IsBlockscaled) {
        // SM100 Sparse Gemm requires ktile start from even number {$nv-release-never}
        return TileScheduler::to_underlying_arguments(
            problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
            args.hw_info, args.scheduler, scheduler_workspace,
            cutlass::gemm::kernel::detail::operand_sizes<ElementA, ElementB, ElementD, CollectiveMainloop::ElementAMmaSparsity>(problem_shape_MNKL, TileShape{}), // {$nv-release-never}
            ktile_start_alignment_count
            );
      }
      else {
        return TileScheduler::to_underlying_arguments(
            problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
            args.hw_info, args.scheduler, scheduler_workspace
            , cutlass::gemm::kernel::detail::operand_sizes<ElementA, ElementB, ElementD, CollectiveMainloop::ElementAMmaSparsity, ElementSF, LayoutSFA, ElementSF, LayoutSFB>(problem_shape_MNKL, TileShape{}) // {$nv-release-never}
          );
      }
    }();

    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace, args.hw_info),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      scheduler_params
      ,args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }
    
    if constexpr (IsBlockscaled) {
      if constexpr (IsDynamicCluster) {
        implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
        // Special cluster shape check for scale factor multicasts. Due to limited size of scale factors, we can't multicast among
        // more than 4 CTAs
        implementable &= (args.hw_info.cluster_shape.x <= 4 && args.hw_info.cluster_shape.y <= 4 &&
                          args.hw_info.cluster_shape_fallback.x <= 4 && args.hw_info.cluster_shape_fallback.y <= 4);
      }
      else {
        // Special cluster shape check for scale factor multicasts. Due to limited size of scale factors, we can't multicast among
        // more than 4 CTAs
        implementable &= ((size<0>(ClusterShape{}) <= 4) && (size<1>(ClusterShape{}) <= 4));
      }
    }

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Tile scheduler
    if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::StreamKScheduler> && not IsBlockscaled) {
      // SM100 Sparse Gemm requires ktile start from even number {$nv-release-never}
      const uint32_t ktile_start_alignment_count = 2u;
      workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers,
        /*epilogue_subtile=*/1, /*num_accumulator_mtx=*/1,
        ktile_start_alignment_count);
    }
    else {
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    }
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::StreamKScheduler> && not IsBlockscaled) {
      // SM100 Sparse Gemm requires ktile start from even number {$nv-release-never}
      const uint32_t ktile_start_alignment_count = 2u;
      status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter, ktile_start_alignment_count);
      workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers,
        /*epilogue_subtile=*/1, /*num_accumulator_mtx=*/1,
        ktile_start_alignment_count);
    }
    else {
    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    }
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // NOTE cluster_shape here is the major cluster shape, not fallback one
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_grid_shape(
        params.scheduler,
        problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
  }

  static constexpr
  dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

// {$nv-release-never begin}
    // Inline PTX knobs
    global_knob_elect_one_r2ur_placement();
    global_knob_trywait_sel_sb_dep();
    global_knob_demote_to_pred_blockidx_limit();
    global_knob_ldc_ldcu_hoisting();
    global_knob_mbarrier_init_mapping();
// {$nv-release-never end}

    using namespace cute;
    using X = Underscore;

#if defined(CUTLASS_ARCH_MMA_SM107A_ENABLED) // {$nv-internal-release begin}
    static_assert(SharedStorageSize <= cutlass::arch::sm107_smem_capacity_bytes, "SMEM usage exceeded capacity.");
#else // {$nv-internal-release end}
    static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");
#endif // {$nv-internal-release}

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;

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
    // TODO: consider dynamic shared storage configs so we don't unnecessarily allocate C smem {$nv-release-never}
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
      (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    // Mainloop Load pipeline
    typename MainloopPipeline::Params mainloop_pipeline_params;
    typename MainloopPipeline::ParamsMetadata mainloop_pipeline_params_metadata;
    if (WarpCategory::MainloopLoad == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load;
    mainloop_pipeline_params_metadata.transaction_bytes = CollectiveMainloop::MainLoadTmaTransactionBytes;
    mainloop_pipeline_params_metadata.metadata_transaction_bytes = CollectiveMainloop::MetadataTmaTransactionBytes;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       mainloop_pipeline_params_metadata,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

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
    epi_load_pipeline_params.initializing_warp = 1;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Load order barrier
    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = NumMainloopLoadThreads;
    load_order_barrier_params.initializing_warp = 3;
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
                                                 (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads);
    if (is_epi_load_needed) {
      clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
    }
    clc_pipeline_params.transaction_bytes = CLCResponseSize;
    clc_pipeline_params.initializing_warp = 4;
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
    accumulator_pipeline_params.initializing_warp = 5;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator,
                                             accumulator_pipeline_params,
                                             cluster_shape,
                                             cute::true_type{},   // Perform barrier init
                                             cute::false_type{}); // Delay mask calculation

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Sched == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
    }
    clc_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
    clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
    clc_throttle_pipeline_params.dst_blockid = 0;
    clc_throttle_pipeline_params.initializing_warp = 3;
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_pipeline_params);
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    // {$nv-internal-release begin}
    // For Narrow-Precision Dense, Narrow-Precision Sparse, Legacy-Precision Sparse kernels
    //  (where we need to allocate TMEM space for auxillary data such as scale factors, sparse metadata)
    //  TMEM allocation mechanism is different than Dense GEMMs.
    // These 3 kernels require overlapping accumulator WAR to enable 2 virtual accumulator buffers when N=256.
    // The TMEM allocation mechanism for these 3 kernels works as follows:
    // Case (1): CTA-N=128 or CTA-N=192 (TMEM has enough capacity to accommodate 2 accumulator buffers):
    //    Ex: N=192, with 32word auxillary data
    //      TMEM cols[0-191]: accum[0],
    //      TMEM_cols[192:383]: accum[1]
    //      TMEM_cols[384:415]: auxillary data
    // Case (2) CTA-N=256 (we need overlapping accumulators SW WAR)
    //    We make a single 512 words (full TMEM capacity) allocations and deallocations.
    //    To enable this WAR, we need a consecutive (256*2 - # overlapping TMEM words) allocation. TMEM allocation ptx instrs can't
    //    guarantee consecutive allocation if this allocation was done in multiple stages. Therefore, we need a single large allocation.
    //    To prevent potential fragmentation and hang issues for the next kernel, we also perform a single deallocation.
    //    Ex: N=256 with with 64 words overlap, 32word auxillary data
    //       TMEM_cols[0-447]:accum[0] and accum[1], TMEM_cols[448-479]: auxillary data, TMEM_cols[480-511]: allocated but unused
    // {$nv-internal-release end}
    if (WarpCategory::MMA == warp_category) {
      if constexpr(!IsOverlappingAccum) {
        // If overlapping accumulator WAR is NOT used, whether we can deallocate a given buffer can be determined by MMA warp // {$nv-internal-release}
        if (has_mma_peer_cta && lane_predicate) {
          tmem_deallocation_result_barrier.init(NumMMAThreads);
        }
      }
      else {
      // {$nv-internal-release begin}
      // If overlapping accumulator WAR is used, we need to wait until both peer CTAs epilogues are done in order to
      // deallocate full TMEM.
      // {$nv-internal-release end}
        if (has_mma_peer_cta && lane_predicate) {
          tmem_deallocation_result_barrier.init(NumEpilogueThreads*2);
        }
        else if (lane_predicate) {
          tmem_deallocation_result_barrier.init(NumEpilogueThreads);
        }
      }
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    // {$nv-release-never begin}
    // Moving load_init() forward is a prologue perf optimization as this can help
    //   reduce the no instruction cycles between UCGABAR_WAIT to UTMALDG.
    // When load_init() is placed in WS of mainload_warp, mainload_warp will encountered
    //   no instruction issue as there will be lots of instructions between UCGABAR_WAIT to UTMALDG.
    // When mainload warp issue those instruction, other warps
    //   are also requesting their own instruction from GCC, which lead to GCC peak BW usage
    //   and increase GCC -> ICC response time (referred as dynamic latency).
    // This means mainload warp's will wait for this longer GCC -> ICC latency and expose
    //   no instruction miss.
    // When there's no much instruction between UCGBAR_WAIT and UTMALDG, mainload warp can relies on
    //   the instruction that resident in ICC cache (ICC prefetch current hit stride + 3 cache line on blackwell)
    //   and don't need to request GCC.
    // {$nv-release-never end}
    auto load_inputs = collective_mainloop.load_init(
        problem_shape_MNKL, shared_storage.tensors.mainloop);

    MainloopPipelineState mainloop_pipe_consumer_state;
    MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();

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
    mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    // TODO https://jirasw.nvidia.com/browse/CFK-13829 Consider introducing partition_tmem_layout function to manage all TMEM related tensors' layout // {$nv-internal-release}
    //
    // TMEM "Allocation"
    //
    auto tmem_storage = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});

    // {$nv-release-never begin}
    const int32_t mods_mainloop_count = IsModsEnabled
                                      ? params.scheduler.mods.looping_controls.mainloop_count | 1 // Must be odd.
                                      : 1;
    // {$nv-release-never end}
    // __syncthreads() need to be executed at the same PC. // {$nv-release-never}
    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;
      bool requires_clc_query = true;

      do {
        // {$nv-release-never begin}
        if constexpr (IsModsEnabled) {
          if (lane_predicate) {
            scheduler.report_smid(params.scheduler, work_tile_info, params.problem_shape, CtaShape_MNK{}, cluster_shape);
          }
        }
        // {$nv-release-never end}

        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, load_inputs.k_tiles);
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        // {$nv-release-never begin}
        for (int32_t mods_mainloop_index = 0; mods_mainloop_index < mods_mainloop_count; mods_mainloop_index++) {
          if constexpr (IsModsEnabled) {
            scheduler.mods_throttle(params.scheduler.mods);
          }
        // {$nv-release-never end}

        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }

        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
        auto [mainloop_producer_state_next, unused_] = collective_mainloop.load(
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter, k_tile_count
        );
        mainloop_pipe_producer_state = mainloop_producer_state_next;

        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }
        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();
        } // mods_mainloop_index {$nv-release-never}

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());
      collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

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

      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        // {$nv-internal-release begin}
        // The following wait on dependent grids is done for load balancing.
        // There is no need to wait for dependency resolution for correctness, but if
        // we wait until all writes are visible, we are likely to find that all SMs are
        // occupied at this point fetching more work IDs will result in load balancing.
        // For example, with one wave it is possible for a single CTA to be launched after
        // PREEXIT and fetch multiple work IDs turning the grid into 2 waves.
        // {$nv-internal-release end}
        cutlass::arch::wait_on_dependent_grids();

        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;

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
    }

    else if (is_participant.mma) {
      // Tmem allocation sequence
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

      auto mma_inputs = collective_mainloop.mma_init(
        tmem_storage,
        shared_storage.tensors.mainloop);

      do {
        // {$nv-release-never begin}
        if constexpr (IsModsEnabled) {
          if (is_mma_leader_cta && lane_predicate) {
            scheduler.report_mainloop_start_time(params.scheduler, work_tile_info, params.problem_shape, CtaShape_MNK{}, cluster_shape, size<0>(AtomThrShapeMNK{}));
          }
        }
        // {$nv-release-never end}

        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        // {$nv-release-never begin}
        // When we apply N=256 WAR, accumulator_pipe_producer_state has a single stage. We use the phase of the single
        // barrier to detect odd/even iterations of the accumulator pipeline
        // XXX: [vht 2023/11/01] maybe `overlapping_accum` can be absorbed into a template parameter of the accumulator pipeline class then?
        // {$nv-release-never end}

        // Accumulator stage slice
        int acc_stage = [&] () {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_producer_state.phase() ^ 1;
          }
          else {
            return accumulator_pipe_producer_state.index();
          }
        }();

        if (is_mma_leader_cta) {
          // {$nv-release-never begin}
          for (int32_t mods_mainloop_index = 0; mods_mainloop_index < mods_mainloop_count; mods_mainloop_index++) {
            bool zero_accumulator = !IsModsEnabled || mods_mainloop_index == 0;
            bool a_negate = IsModsEnabled && (mods_mainloop_index & 1) != 0;
          // {$nv-release-never end}
          mainloop_pipe_consumer_state = collective_mainloop.mma(
            cute::make_tuple(mainloop_pipeline, accumulator_pipeline),
            cute::make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state),
            collective_mainloop.slice_accumulator(tmem_storage, acc_stage),
            mma_inputs,
            cta_coord_mnkl,
            k_tile_count
            , zero_accumulator // {$nv-release-never}
            , a_negate         // {$nv-release-never}
            );
          } // mods_mainloop_index {$nv-release-never}
          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
        }
        ++accumulator_pipe_producer_state;
        // {$nv-release-never begin}
        if constexpr (IsModsEnabled) {
          if (is_mma_leader_cta && lane_predicate) {
              scheduler.report_mainloop_end_time(params.scheduler, work_tile_info, params.problem_shape, CtaShape_MNK{}, cluster_shape, size<0>(AtomThrShapeMNK{}));
          }
        }
        // {$nv-release-never end}

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Release the right to allocate before deallocations so that the next CTA can rasterize
      tmem_allocator.release_allocation_lock();

      // {$nv-internal-release begin}
      // Release shared memory at the end of MMA.
      // Since the persistent loop of MMA is the last place where the shared memory reserved
      // for A and B tiles is consumed, we let the barrier wait here and then commit
      // This will wait on the DMA warp to observe all MMAs, meaning this warp is done using SMEM.
      smem_early_release_manager.resize_for_all_blocking();
      // {$nv-internal-release end}

      if constexpr (!IsOverlappingAccum) {
        // Leader MMA waits for leader + peer epilogues to release accumulator stage
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
      }
      else {
        // Overlapping accumulator WAR, wait for epilogue to finish. // {$nv-internal-release}
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
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
      // {$nv-internal-release begin}
      // With Stream-K epi_load_pipe_producer_state alone is not enough to keep track of
      // even/odd output tiles. In order to reverse epilogue's order of processing for epi_tiles
      // with N=256 WAR, we need to keep track of the current wave id and pass this information (reverse_epi_n) to
      // load function.
      // {$nv-internal-release end}
      int current_wave = 0;

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

          bool reverse_epi_n = IsOverlappingAccum && (current_wave % 2 == 0);
          epi_load_pipe_producer_state = collective_epilogue.template load<IsOverlappingAccum>(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            shared_storage.tensors.epilogue,
            reverse_epi_n
          );

          do_tail_load = true;
        }
        current_wave++;

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

      // Wait for tmem allocate here
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

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

        // Accumulator stage slice
        int acc_stage = [&] () {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_consumer_state.phase();
          }
          else {
            return accumulator_pipe_consumer_state.index();
          }
        }();

        auto accumulator = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
        accumulator_pipe_consumer_state = scheduler.template fixup<IsComplex>(
          TiledMma{},
          work_tile_info,
          accumulator,
          accumulator_pipeline,
          accumulator_pipe_consumer_state,
          typename CollectiveEpilogue::CopyOpT2R{}
        );

        //
        // Epilogue and write to gD
        //
        if (scheduler.compute_epilogue(work_tile_info)) {
          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.template store<IsOverlappingAccum>(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            accumulator_pipeline,
            accumulator_pipe_consumer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            accumulator,
            shared_storage.tensors.epilogue
          );
          epi_load_pipe_consumer_state = load_state_next;
          epi_store_pipe_producer_state = store_state_next;
          accumulator_pipe_consumer_state = acc_state_next;
          do_tail_store = true;
        }
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

      } while (work_tile_info.is_valid());

      if constexpr (IsOverlappingAccum) {
        // Signal to peer MMA that Full TMEM alloc can be deallocated
        if constexpr (has_mma_peer_cta) {
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank);
        }
        tmem_deallocation_result_barrier.arrive();
      }

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
