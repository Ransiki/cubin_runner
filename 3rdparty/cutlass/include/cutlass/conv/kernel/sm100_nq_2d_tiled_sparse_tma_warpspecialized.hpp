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
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/grid_dependency_control.h"

#include "cutlass/conv/detail.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/conv/kernel/sm100_tile_scheduler_nq_2d_tiled.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"

#include "cute/arch/tmem_allocator_sm100.hpp"

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
  cute::enable_if_t<
    cute::is_base_of_v<KernelSparseNq2dTiledTmaWarpSpecializedStride1x1x1Sm100<cute::Shape<_1,_3,_3>>,
                        typename CollectiveMainloop_::DispatchPolicy::Schedule> ||
    cute::is_base_of_v<KernelSparseNq2dTiledTmaWarpSpecializedStride1x1x1Sm100<cute::Shape<_3,_3,_3>>,
                        typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //

  // Mainloop derived types
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TileShapeMeta = typename CollectiveMainloop::TileShapeMeta;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideFlt = typename CollectiveMainloop::StrideFlt;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideAct = typename CollectiveMainloop::StrideAct;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static constexpr int NumSpatialDimensions = CollectiveMainloop::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  using ElementE = typename CollectiveMainloop::ElementE;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = 2;
  static_assert(cute::is_void_v<TileSchedulerTag_>, "Customizing the tile scheduler is not supported yet.");
  using TileSchedulerTag = cutlass::gemm::PersistentScheduler;
  using TileSchedulerBase = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileScheduler = cutlass::conv::kernel::detail::Nq2dTiledSchedulerSm100<
    TileSchedulerBase, typename CollectiveMainloop::FilterShape_TRS>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  // Warp specialization warp count
  static constexpr uint32_t NumActivationLoadThreads = 1 * NumThreadsPerWarp;
  static constexpr uint32_t NumFilterLoadThreads     = 1 * NumThreadsPerWarp;
  static constexpr uint32_t NumSchedThreads          = 1 * NumThreadsPerWarp;
  static constexpr uint32_t NumMmaThreads            = 1 * NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueThreads       = CollectiveEpilogue::ThreadCount;

  static constexpr uint32_t MaxThreadsPerBlock = NumActivationLoadThreads +
                                                 NumFilterLoadThreads +
                                                 NumSchedThreads +
                                                 NumMmaThreads +
                                                 NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Accumulator buffer count
  static constexpr uint32_t AccumulatorPipelineStageCount = CollectiveMainloop::AccumulatorPipelineStageCount;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Pipeline and pipeline state types
  using MainloopActPipeline = typename CollectiveMainloop::MainloopActPipeline;
  using MainloopActPipelineState = typename CollectiveMainloop::MainloopActPipelineState;

  using MainloopFltPipeline = typename CollectiveMainloop::MainloopFltPipeline;
  using MainloopFltPipelineState = typename CollectiveMainloop::MainloopFltPipelineState;

  using MainloopMetaPipeline = typename CollectiveMainloop::MainloopMetaPipeline;
  using MainloopMetaPipelineState = typename CollectiveMainloop::MainloopMetaPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using AccumulatorPipeline = typename CollectiveMainloop::AccumulatorPipeline;
  using AccumulatorPipelineState = typename CollectiveMainloop::AccumulatorPipelineState;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CZMPipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CZMPipelineState = typename CZMPipeline::PipelineState;

  // 2d-tiled kernel only support 1SM version.
  using TmemAllocator = cute::TMEM::Allocator1Sm;

  using FilterShape_TRS = typename CollectiveMainloop::FilterShape_TRS;
  static constexpr int Flt_S = size<2>(FilterShape_TRS{});

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopActPipelineStorage = typename CollectiveMainloop::ActPipelineStorage;
      using MainloopFltPipelineStorage = typename CollectiveMainloop::FltPipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using CZMPipelineStorage = typename CZMPipeline::SharedStorage;

      alignas(16) MainloopActPipelineStorage mainloop_act;
      alignas(16) MainloopFltPipelineStorage mainloop_flt;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) AccumulatorPipelineStorage accumulator;
      alignas(16) CZMPipelineStorage czm;
    } pipelines;
    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    alignas(16) typename UMMA::MaskAndShiftB column_zero_masks[Flt_S * SchedulerPipelineStageCount];

    // NOTE: Also change CollectiveBuilder's KernelSmemCarveout computation when adopting tmem ptx and remove C++ tmem here  // {$nv-internal-release}
    alignas(16) uint32_t tmem_base_ptr;

    // {$nv-internal-release begin}
    // GDC (a.k.a. FDL or "Fast Dependent Launch") mainloop buffers for a dependent kernel,
    // so we move them to the end of SharedStorage.
    // {$nv-internal-release end}
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopActTensorStorage = typename CollectiveMainloop::ActTensorStorage;
      using MainloopFltTensorStorage = typename CollectiveMainloop::FltTensorStorage;
      using MainloopMetaTensorStorage = typename CollectiveMainloop::MetaTensorStorage;

      EpilogueTensorStorage epilogue;
      struct MainloopTensorStorage : cute::aligned_struct<128, _1> {
        MainloopActTensorStorage act;
        MainloopFltTensorStorage flt;
        MainloopMetaTensorStorage meta;
      } mainloop;
    } tensors;

  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");

  // User facing host arguments
  struct Arguments {
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    
    private:
      // Fprop: ((C,S,R,T),P)
      // Dgrad: ((K,S,R,T),H)
      using _Submode = decltype(make_shape(take<0,NumTensorDimensions-1>(typename ProblemShape::TensorExtent{}), 0));
      // Fprop: (Q,N)
      // Dgrad: (W,N)
      using _Submode_QN = decltype(take<0,2>(typename ProblemShape::TensorExtent{}));
      using _FastDivmod_Submode_QN = decltype(cute::repeat_like(_Submode_QN{}, cutlass::FastDivmod{}));
    public:
      // Apply FastDivmod for all NQ-tiled kernels
      using ProblemShapeMNKL = Shape<int, _FastDivmod_Submode_QN, _Submode, int>;
      ProblemShapeMNKL problem_shape;
      MainloopParams mainloop;
      EpilogueParams epilogue;
      TileSchedulerParams scheduler;
      KernelHardwareInfo hw_info;
  };

  enum class WarpCategory : int32_t {
    MMA           = 0,
    Sched_EpiLoad = 1,
    ActLoad       = 2,
    FltLoad       = 3,
    Epilogue      = 4
  };

  //
  // Methods
  //

  // Map user facing arguments to device facing params
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    using namespace cutlass::conv::detail;
    auto [M, N, K, L] = get_transformed_problem_shape_MNK_nq_2d_tiled(args.problem_shape);
    auto FastDivmod_N = cute::transform_leaf(
                                N, 
                                [](auto const& s) { return conditional_return(is_static<decltype(s)>{}, s, FastDivmod(s)); });

    auto problem_shape_mnkl = make_shape(M,FastDivmod_N,K,L);
    auto mainloop_params = CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace, args.hw_info);
    // problem_shape_mnkl is not used in epilogue actually.
    // mainloop_params.problem_shape is (K,(Q,N),((C,S,R,T),P),Z)
    auto linear_problem_shape_MNKL = make_shape(
        shape<0>(problem_shape_mnkl),
        size<1>(problem_shape_mnkl), // QN mode is linearized.
        shape<2>(problem_shape_mnkl),
        shape<3>(problem_shape_mnkl));
    auto lower_corner_h = (cutlass::conv::collective::detail::compute_lower_corner_whd(args.problem_shape))[1];

    return {
      problem_shape_mnkl,
      mainloop_params,
      CollectiveEpilogue::to_underlying_arguments(linear_problem_shape_MNKL, args.epilogue, workspace),
      TileScheduler::to_underlying_arguments(
        lower_corner_h, linear_problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, workspace),
      args.hw_info
    };
  }

  // Returns true if the kernel can run successfully with the given arguments, else false.
  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
  
    // Alias
    auto const& output_shape = args.problem_shape.shape_C; // [N,Z,P,Q,K]

    // P in each slice cannot be less than 3
    implementable &= ((output_shape[NumTensorDimensions-3] / args.scheduler.split_p_slices) >= 3);

    implementable = implementable && CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable = implementable && CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= (args.hw_info.cluster_shape.x == 1 && args.hw_info.cluster_shape.y > 0 && args.hw_info.cluster_shape.z == 1);

      implementable &= (args.hw_info.cluster_shape_fallback.x == 1 && args.hw_info.cluster_shape_fallback.y > 0 && args.hw_info.cluster_shape_fallback.z == 1);

      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }
    return implementable;
  }

  static int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);
    auto linear_problem_shape_MNKL = make_shape(
        shape<0>(params.problem_shape),
        size<1>(params.problem_shape), // QN mode is linearized.
        shape<2>(params.problem_shape),
        shape<3>(params.problem_shape));
    return TileScheduler::get_grid_shape(
        params.scheduler,
        linear_problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info
       );

  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {

// {$nv-release-never begin}
    // Inline PTX knobs
    global_knob_elect_one_r2ur_placement();
    global_knob_demote_to_pred_blockidx_limit();
    // This knob will cause perf drop
    // https://nvbugspro.nvidia.com/bug/4970686
    // global_knob_ldc_ldcu_hoisting();
    global_knob_mbarrier_init_mapping();
// {$nv-release-never end}

    using namespace cute;
    using X = Underscore;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop(params.mainloop);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Warp index and role
    int thread_idx = int(threadIdx.x);
    int lane_predicate = cute::elect_one_sync();
    auto warp_category = (WarpCategory(canonical_warp_idx_sync()) < WarpCategory::Epilogue) ?
                            WarpCategory(canonical_warp_idx_sync()) : WarpCategory::Epilogue;
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    int cluster_size = size(cluster_shape);
    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    int cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    bool is_mma_leader_cta = cta_rank_in_cluster % size<0>(TiledMma{}) == 0;  // XXX: Assumes MMA has contiguous CTAs

    // Issue Tma Descriptor Prefetch from a single thread
    if ((canonical_warp_idx_sync() == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    // Mainloop activation load pipeline
    typename MainloopActPipeline::Params mainloop_act_pipeline_params;
    if (WarpCategory::ActLoad == warp_category) {
      mainloop_act_pipeline_params.role = MainloopActPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_act_pipeline_params.role = MainloopActPipeline::ThreadCategory::Consumer;
    }
    mainloop_act_pipeline_params.is_leader = lane_predicate && (warp_category == WarpCategory::ActLoad);
    mainloop_act_pipeline_params.num_consumers = 1;
    mainloop_act_pipeline_params.transaction_bytes = CollectiveMainloop::ActTransactionBytes;
    mainloop_act_pipeline_params.initializing_warp = 0;
    // Use single-sided constructor for PipelineTmaUmmaAsync
    MainloopActPipeline mainloop_act_pipeline(
      shared_storage.pipelines.mainloop_act, mainloop_act_pipeline_params, cluster_shape, MainloopActPipeline::McastDirection::kCol);
    MainloopActPipelineState mainloop_act_pipe_consumer_state;
    MainloopActPipelineState mainloop_act_pipe_producer_state = cutlass::make_producer_start_state<MainloopActPipeline>();

    // Mainloop filter load pipeline
    typename MainloopFltPipeline::Params mainloop_flt_pipeline_params;
    if (WarpCategory::FltLoad == warp_category) {
      mainloop_flt_pipeline_params.role = MainloopFltPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_flt_pipeline_params.role = MainloopFltPipeline::ThreadCategory::Consumer;
    }
    mainloop_flt_pipeline_params.is_leader = lane_predicate && (warp_category == WarpCategory::FltLoad);
    mainloop_flt_pipeline_params.num_consumers = 1;
    mainloop_flt_pipeline_params.transaction_bytes = CollectiveMainloop::FltTransactionBytes + CollectiveMainloop::MetaTransactionBytes;
    mainloop_flt_pipeline_params.initializing_warp = 3;
    // Use single-sided constructor for PipelineTmaUmmaAsync
    MainloopFltPipeline mainloop_flt_pipeline(
      shared_storage.pipelines.mainloop_flt, mainloop_flt_pipeline_params, cluster_shape, MainloopFltPipeline::McastDirection::kRow);
    MainloopFltPipelineState mainloop_flt_pipe_consumer_state;
    MainloopFltPipelineState mainloop_flt_pipe_producer_state = cutlass::make_producer_start_state<MainloopFltPipeline>();

    // Epilogue load pipeline
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::Sched_EpiLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumSchedThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.initializing_warp = 4;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);
    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched_EpiLoad == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    }
    else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_blockid = 0;
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.consumer_arv_count = cluster_size * NumSchedThreads +
                                                 cluster_size * NumMmaThreads +              // mainloop mma threads
                                                 cluster_size * NumActivationLoadThreads +      // mainloop act warp threads
                                                 cluster_size * NumFilterLoadThreads +          // mainloop flt warp threads
                                                 cluster_size * NumEpilogueThreads;
    clc_pipeline_params.transaction_bytes = CLCResponseSize;
    clc_pipeline_params.initializing_warp = 1;
    CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);
    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    // Mainloop-Epilogue pipeline
    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    }
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    accumulator_pipeline_params.initializing_warp = 2;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator, accumulator_pipeline_params, cluster_shape);

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    // CZM pipeline
    typename CZMPipeline::Params czm_pipeline_params;
    if (WarpCategory::Sched_EpiLoad == warp_category) {
      czm_pipeline_params.role = CZMPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      czm_pipeline_params.role = CZMPipeline::ThreadCategory::Consumer;
    }
    czm_pipeline_params.dst_blockid = cta_rank_in_cluster;
    czm_pipeline_params.producer_arv_count = NumSchedThreads;
    czm_pipeline_params.consumer_arv_count = NumMmaThreads;
    czm_pipeline_params.initializing_warp = 5;
    CZMPipeline czm_pipeline(shared_storage.pipelines.czm, czm_pipeline_params);
    CZMPipelineState czm_pipe_consumer_state;
    CZMPipelineState czm_pipe_producer_state = cutlass::make_producer_start_state<CZMPipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMmaThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

    // {$nv-internal-release begin}
    // The barrier to ensure all threads are ready to commit Shared memory resizing
    cutlass::arch::SmemEarlyReleaseManager<SharedStorage> smem_early_release_manager;
    // {$nv-internal-release end}

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    if (cluster_size > 1) {
      cute::cluster_arrive_relaxed();
    }
    else {
      __syncthreads();
    }

    // TileID scheduler
    // First piece of work is just blockIdx
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info{scheduler.initial_work_tile_info(cluster_shape)};

    // Define problem shape here.
    auto M = get<0>(params.problem_shape);               // K
    auto N = shape<1>(params.problem_shape);             // (Q,N)
    auto K = get<0>(shape<2>(params.problem_shape));     // (C,S,R,T)
    auto P = get<1>(shape<2>(params.problem_shape));     // P
    auto Z = get<3>(params.problem_shape);               // Z

    // Number of c chunks and h pixels
    // This assumes the same gAct as our example
    auto c_chunks = size(ceil_div(get<0>(K), get<0>(get<2>(CtaShape_MNK{}))));

    //
    // TMEM "Allocation"
    //
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(CtaShape_MNK{}));
    Tensor accumulator_tmem = TiledMma::make_fragment_C(append(acc_shape, Int<AccumulatorPipelineStageCount>{}));

    //
    // END PROLOGUE
    //

    if (cluster_size > 1) {
      cute::cluster_wait();
    } else {
      __syncthreads();
    }

    if (WarpCategory::ActLoad == warp_category) {

      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      auto load_act_inputs = collective_mainloop.load_act_init(
          make_shape(N, get<0>(K), P, Z), shared_storage.tensors.mainloop.act);

      do {
        auto [cta_coord_mnkl, h_pixels_start, h_pixels_end] = scheduler.work_tile_to_cta_coord_and_h_range(work_tile_info);
        auto z_idx = get<3>(cta_coord_mnkl);

        auto [mainloop_act_producer_state_next] = collective_mainloop.load_act(
          mainloop_act_pipeline,
          mainloop_act_pipe_producer_state,
          load_act_inputs,
          cta_coord_mnkl,
          c_chunks, h_pixels_start, h_pixels_end, z_idx,
          lane_predicate
        );
        mainloop_act_pipe_producer_state = mainloop_act_producer_state_next;

        __syncwarp();

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());

      collective_mainloop.load_act_tail(mainloop_act_pipeline, mainloop_act_pipe_producer_state, lane_predicate);

      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the end of mainloop DMA
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

    }
    else if (WarpCategory::FltLoad == warp_category) {

      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      auto load_flt_inputs = collective_mainloop.load_flt_init(
          shape(params.mainloop.layout_flt_a), shared_storage.tensors.mainloop.flt);
      auto load_meta_inputs = collective_mainloop.load_meta_init(
          shape(params.mainloop.layout_meta_e), shared_storage.tensors.mainloop.meta);

      do {
        auto [cta_coord_mnkl, h_pixels_start, h_pixels_end] = scheduler.work_tile_to_cta_coord_and_h_range(work_tile_info);

        auto [mainloop_flt_producer_state_next] = collective_mainloop.load_flt_and_meta(
          mainloop_flt_pipeline,
          mainloop_flt_pipe_producer_state,
          load_flt_inputs,
          load_meta_inputs,
          cta_coord_mnkl,
          c_chunks, h_pixels_start, h_pixels_end,
          lane_predicate
        );
        mainloop_flt_pipe_producer_state = mainloop_flt_producer_state_next;

        __syncwarp();

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());

      collective_mainloop.load_flt_and_meta_tail(mainloop_flt_pipeline, mainloop_flt_pipe_producer_state, lane_predicate);

      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the end of mainloop DMA
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

    }
    else if (WarpCategory::Sched_EpiLoad == warp_category) {
      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the beginning of scheduler warp
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

      auto load_inputs = collective_epilogue.load_init(params.problem_shape, CtaShape_MNK{}, params.epilogue);
      // Whether a new CLC query must be performed.
      // See comment below where this variable is updated for a description of
      // why this variable is needed.
      bool requires_clc_query = true;
      do {
        if (requires_clc_query && is_first_cta_in_cluster) {
          // Query next clcID and update producer state
          clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
       }

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        // Skip the last update when the tile info is invalid.
        if (next_work_tile_info.is_valid()) {
          // Update column_zero_mask for next_work_tile and store the result into smem for mma warp to load.
          // Add CZM pipeline for Sched warp's CZM STS and Math warp's CZM LDS.
          // In both warps, CZM pipeline's wait() and release() are after CLC pipeline's release().
          // So CZM pipeline will not block CLC pipeline anytime.
          auto [tile_coord_mnkl, h_pixels_start, h_pixels_end] = scheduler.work_tile_to_cta_coord_and_h_range(next_work_tile_info);
          auto conv_q = get<0>(N);
          auto cta_coord_q = get<1>(tile_coord_mnkl);
          auto czm_pipe_producer_state_next = collective_mainloop.update_column_zero_mask(conv_q, cta_coord_q, params.mainloop.num_pixels_skip_left,
                                                      &shared_storage.column_zero_masks[0],
                                                      czm_pipeline, czm_pipe_producer_state);
          czm_pipe_producer_state = czm_pipe_producer_state_next;
        }

        // Load residual
        if constexpr (CollectiveEpilogue::IsSourceSupported) {
          if (collective_epilogue.is_source_needed()) {
            auto [cta_coord_mnkl, p_pixels_start, p_pixels_end] = scheduler.work_tile_to_cta_coord_and_p_range(work_tile_info);
            auto [epi_load_pipe_producer_state_next] = collective_epilogue.load(
              epi_load_pipeline,
              epi_load_pipe_producer_state,
              load_inputs,
              cta_coord_mnkl,
              p_pixels_start,
              p_pixels_end,
              shared_storage.tensors.epilogue
            );
            epi_load_pipe_producer_state = epi_load_pipe_producer_state_next;
          }
        }

        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

      if (is_first_cta_in_cluster) {
        clc_pipeline.producer_tail(clc_pipe_producer_state);
      }
    }
    else if (WarpCategory::MMA == warp_category && is_mma_leader_cta) {
      // Allocate all tmem
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      accumulator_tmem.data() = tmem_base_ptr;
      uint32_t meta_base_addr = accumulator_tmem.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulator_tmem);

      auto mma_input_operands = collective_mainloop.mma_init(params.mainloop,
                                                             shared_storage.tensors.mainloop.flt,
                                                             shared_storage.tensors.mainloop.act,
                                                             shared_storage.tensors.mainloop.meta,
                                                             meta_base_addr);

      // Update column_zero_mask for each work_tile
      // Calculate first/initialized work_tile's czm.
      auto [tile_coord_mnkl, h_pixels_start, h_pixels_end] = scheduler.work_tile_to_cta_coord_and_h_range(work_tile_info);
      auto conv_q = get<0>(N);
      auto cta_coord_q = get<1>(tile_coord_mnkl);
      // Use the function without smem_ptr args and store the result in mma warp's own reg.
      collective_mainloop.update_column_zero_mask(conv_q, cta_coord_q, params.mainloop.num_pixels_skip_left);
      // Only first work_tile's czm is calculated by mma warp itself. After that, czm is calculated by sched warp.
      bool is_first_work_tile = true;

      do {
        // Get current work tile cta coord
        auto [cta_coord_mnkl, h_pixels_start, h_pixels_end] = scheduler.work_tile_to_cta_coord_and_h_range(work_tile_info);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        // Load CZM from smem for current work_tile
        if (!is_first_work_tile) {
          auto czm_pipe_consumer_state_next = collective_mainloop.load_column_zero_mask(&shared_storage.column_zero_masks[0],
                                                                                          czm_pipeline, czm_pipe_consumer_state);
          czm_pipe_consumer_state = czm_pipe_consumer_state_next;
        }

        auto [mainloop_flt_pipe_consumer_state_next, 
              mainloop_act_pipe_consumer_state_next, 
              accumulator_pipe_producer_state_next] = collective_mainloop.mma(
          mainloop_flt_pipeline, mainloop_flt_pipe_consumer_state,
          mainloop_act_pipeline, mainloop_act_pipe_consumer_state,
          accumulator_pipeline, accumulator_pipe_producer_state,
          accumulator_tmem, c_chunks, h_pixels_start, h_pixels_end,
          mma_input_operands, lane_predicate
        );
        mainloop_flt_pipe_consumer_state = mainloop_flt_pipe_consumer_state_next;
        mainloop_act_pipe_consumer_state = mainloop_act_pipe_consumer_state_next;
        accumulator_pipe_producer_state  = accumulator_pipe_producer_state_next;

        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        is_first_work_tile = false;
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);

      // {$nv-internal-release begin}
      // Release shared memory at the end of MMA.
      // Since the persistent loop of MMA is the last place where the shared memory reserved
      // for A and B tiles is consumed, we let the barrier wait here and then commit
      
      smem_early_release_manager.resize_for_all_blocking();
      // {$nv-internal-release end}

    }
    else if (WarpCategory::Epilogue == warp_category) {

      // {$nv-internal-release begin}
      // Release shared memory for A and B tiles at the beginning of epilogue
      smem_early_release_manager.resize_for_self_non_blocking();
      // {$nv-internal-release end}

      auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(CollectiveEpilogue::ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
      const int epi_thread_idx = thread_idx % CollectiveEpilogue::ThreadCount;
      const int epi_warp_idx = epi_thread_idx / NumThreadsPerWarp;
      auto store_inputs = collective_epilogue.store_init(params.problem_shape, CtaShape_MNK{}, params.epilogue);

      // Wait for tmem allocation to be done in all epilogue warps
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      accumulator_tmem.data() = tmem_base_ptr;

      // Epilogue tile switching loop
      do {
        auto [cta_coord_mnkl, p_pixels_start, p_pixels_end] = scheduler.work_tile_to_cta_coord_and_p_range(work_tile_info);

        auto [accumulator_pipe_consumer_state_next, epi_load_pipe_consumer_state_next] = collective_epilogue.store(
          params.epilogue,
          accumulator_pipeline,
          accumulator_pipe_consumer_state,
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          store_inputs,
          cta_coord_mnkl,
          p_pixels_start,
          p_pixels_end,
          accumulator_tmem,
          shared_storage.tensors.epilogue
        );

        accumulator_pipe_consumer_state = accumulator_pipe_consumer_state_next;
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());

      // All ldtm have finished
      synchronize();
      // Deallocate TMEM
      if (epi_warp_idx == 0) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }
    }
    // {$nv-internal-release begin}
    // Some warps may not be invoked in warp-specialized branches.
    // These warps do not touch A/B tiles so they hint to resize shared memory directly. 
    else {
      smem_early_release_manager.resize_for_self_non_blocking();
    }
    // {$nv-internal-release end}
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::kernel
