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
#include "cutlass/gemm/kernel/static_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp" // {$nv-release-never}
#include "cutlass/arch/mods.h"  // {$nv-release-never}


namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

template <auto... Options> // {$nv-release-never}
class StaticPersistentTileScheduler100:
public StaticPersistentTileScheduler<
  StaticPersistentTileScheduler100
    <Options...> // {$nv-release-never}
  > {

public:
  #if 0 // {$nv-release-never}
  using BaseScheduler = StaticPersistentTileScheduler<StaticPersistentTileScheduler100>;
  // {$nv-release-never begin}
  #else
  using BaseScheduler = StaticPersistentTileScheduler<StaticPersistentTileScheduler100<Options...>>;
  #endif
  // {$nv-release-never end}

public:
  using BaseScheduler::StaticPersistentTileScheduler;
  using Params = PersistentTileSchedulerSm90Params;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  struct CLCResponse { uint32_t data[4] = {0}; };

  // {$nv-release-never begin}
  static constexpr bool IsModsEnabled = cute::get<0>(cute::make_tuple(Options..., cutlass::mods::InstrumentationCategoryClass::None)) == cutlass::mods::InstrumentationCategoryClass::Mods;
  using ModsInstrumentation = cutlass::mods::ModsInstrumentation;
  using ModsParams = ModsInstrumentation::Params;
  using ModsArguments = ModsInstrumentation::Arguments;
  // {$nv-release-never end}

  static constexpr bool IsDynamicPersistent = false;
  using Pipeline = PipelineEmpty;
  using PipelineStorage = typename Pipeline::SharedStorage;
  using ThrottlePipeline = PipelineEmpty;
  using ThrottlePipelineStorage = typename ThrottlePipeline::SharedStorage;

  class SharedStorage {
  public:
    CUTLASS_DEVICE PipelineStorage pipeline() { return PipelineStorage{}; }
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return ThrottlePipelineStorage{}; }
    CUTLASS_DEVICE CLCResponse* data() { return nullptr; }
  };

  using WorkTileInfo = typename BaseScheduler::WorkTileInfo;
  using Arguments = typename BaseScheduler::Arguments;

  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {

    // {$nv-release-never begin}
    //
    // This could be changed to get_hier_coord with FastDivmodU64s
    //
    // The underlying approach is:
    // We have our blocks which we rasterize in a given direction
    //    e.g., (M, N) : (1, M)
    // We then tile this with our CGA
    //    e.g., for a 2x2 CGA the shape is ((M / 2, 2), (N / 2, 2))
    //
    // Finally we interleave blocks
    //    e.g., for a 1x1 CGA we may have (M, (2, N / 2)) : (N, (1, M * 2))
    //          This is M major interleaved
    //
    // To get from our linear index (blk_per_grid_dim) to our coordinates
    // we do the following:
    //
    // Calculate the cluster_id within the grid:
    //    cluster_id = linear index / CGA size major
    //    cluster_major_offset = linear index % CGA size major
    // This gives us the cluster we are operating in and the offset within it
    // We also know based on the way we launch the grid that our cluster minor
    // offset is blockIdx.x or blockIdx.y since we only launch CGA_M or CGA_N
    // CTAs in that dimension of the grid.
    //
    // Next we can calculate the cluster minor index and cluster major index
    // First we calculate our swizzle offset, which is
    //   offset_into_swizzle = cluster_id % (2 ^ log_swizzle_size)
    //   extra = cluster_id / 2^log_swizzle_size
    // Then we get our minor index divided by the swizzle and our major index
    //   cluster_idx_minor_div_swizzle = extra /
    //       (Problem Size in Major Dimension / CGA size in Major Dimension)
    //   cluster_idx_major = extra %
    //       (Problem Size in Major Dimension / CGA size in Major Dimension)
    // Then we calculate the minor index from the swizzle
    //   cluster_idx_minor = cluster_idx_minor_div_swizzle * 2^log_swizzle_size + offset
    // We can then get the index by multiplying the indexes by the cluster shape and adding
    // our offsets
    //
    // {$nv-release-never end}

    uint64_t cluster_id, cluster_major_offset = 0 ;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;
    int32_t minor_work_idx, major_work_idx;

    minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor);
    major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }
  }

  // clc_response_ptr is a placeholder; it is just to make the StaticPersistentTileScheduler100 and PersistentTileScheduler100 constructor interfaces consistent
  CUTLASS_DEVICE explicit
  StaticPersistentTileScheduler100(CLCResponse* /* clc_response_ptr */, Params const& params, dim3 block_id_in_cluster)
    : BaseScheduler(params) {}

  // The basic tile scheduler does not require any additional workspace
  // Additional workspace is only needed if MODS instrumentation is enabled // {$nv-release-never}
  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const&args, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    size_t workspace_size  = 0;
    // {$nv-release-never begin}
    // Mods workspace
    if constexpr (IsModsEnabled) {
      workspace_size += ModsInstrumentation::get_workspace_size(args.mods_args);
      workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);
    }
    // {$nv-release-never end}
    return workspace_size;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace_ptr, cudaStream_t stream, ProblemShape problem_shape, KernelHardwareInfo const&,
    uint32_t, const uint32_t = 1, uint32_t = 1, CudaHostAdapter *cuda_adapter = nullptr) {

    // {$nv-release-never begin}
    // Initialize Mods workspace
    if constexpr (IsModsEnabled) {
#if !defined(__CUDACC_RTC__)
    uint8_t* workspace = reinterpret_cast<uint8_t*>(workspace_ptr);
    size_t workspace_size = ModsInstrumentation::get_workspace_size(args.mods_args);
    return zero_workspace(workspace, workspace_size, stream, cuda_adapter);
#endif
    }
    // {$nv-release-never end}

    return Status::kSuccess;
  }

  template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1
      , OperandSizeInfo const& operand_sizes = {} // {$nv-release-never}
      ) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = BaseScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk,
                                                                 atom_thr_shape_mnk, cluster_shape_mnk);
    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape_mnk),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );

    // {$nv-release-never begin}
    // MODS arguments
    if constexpr (IsModsEnabled) {
      params.mods = ModsInstrumentation::to_underlying_arguments(arguments.mods_args, workspace);
    }
    // {$nv-release-never end}

    return params;
  }

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    [[maybe_unused]] KernelHardwareInfo const& hw_info,
    Arguments const& arguments,
    [[maybe_unused]] void* workspace=nullptr,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1,
    OperandSizeInfo const& operand_sizes = {}, // {$nv-release-never}
    [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = BaseScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );

    // {$nv-release-never begin}
    // MODS arguments
    if constexpr (IsModsEnabled) {
      params.mods = ModsInstrumentation::to_underlying_arguments(arguments.mods_args, workspace);
    }
    // {$nv-release-never end}

    return params;
  }

  // {$nv-release-never begin}
  // Adapters needed for CAGE integration
  template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace,
      OperandSizeInfo const& operand_sizes) {
    return to_underlying_arguments(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk,
      cluster_shape_mnk, hw_info, arguments, workspace, /*epilogue_subtile=*/1, operand_sizes);
  }

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    [[maybe_unused]] KernelHardwareInfo const& hw_info,
    Arguments const& arguments,
    [[maybe_unused]] void* workspace,
    OperandSizeInfo const& operand_sizes) {
    return to_underlying_arguments(problem_shape_mnkl, tile_shape, cluster_shape,
      hw_info, arguments, workspace, /*epilogue_subtile=*/1, operand_sizes);
  }
  // {$nv-release-never end}

  template <
    bool IsComplex,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class CopyOpT2R
  >
  CUTLASS_DEVICE
  AccumulatorPipelineState
  fixup(
      TiledMma const& ,
      WorkTileInfo const&,
      cute::Tensor<AccEngine, AccLayout>&,
      AccumulatorPipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      CopyOpT2R) const {
    return acc_pipe_consumer_state;
  }

  // Performs the reduction across splits for a given output tile.
  // Used by SM90 kernel layers // {$nv-internal-release}
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(
      Params const& params,
      WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {
  }

  // {$nv-release-never begin}
  // Methods needed for MODS
  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  CUTLASS_DEVICE
  int block_rank_in_grid(Params const& params, WorkTileInfo const& work_tile_info,  ProblemShapeMNKL const& problem_shape, TileShape tile_shape, ClusterShape cluster_shape) {
    auto output_grid_shape = this->get_tiled_cta_shape_mnl(append<4>(problem_shape, Int<1>{}), tile_shape, cluster_shape);
    dim3 cta_id_in_cluster = cute::block_id_in_cluster();
    auto tile_coord = make_coord(work_tile_info.M_idx + cta_id_in_cluster.x, work_tile_info.N_idx + cta_id_in_cluster.y, work_tile_info.L_idx + cta_id_in_cluster.z);
    auto grid_shape = make_shape(output_grid_shape.x, output_grid_shape.y, output_grid_shape.z);
    auto cluster_MNL = append(take<0,2>(cluster_shape), Int<1>{});
    if (params.raster_order_ == RasterOrder::AlongM) {
      auto cta_layout_in_grid = tile_to_shape(make_layout(cluster_MNL), grid_shape, Step<_0,_1,_2>{});
      return cta_layout_in_grid(tile_coord);
    }
    else
    {
      auto cta_layout_in_grid = tile_to_shape(make_layout(cluster_MNL), grid_shape, Step<_1,_0,_2>{});
      return cta_layout_in_grid(tile_coord);
    }
  }

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  CUTLASS_DEVICE
  void
  report_smid(Params const& params, WorkTileInfo const& work_tile_info,  ProblemShapeMNKL const& problem_shape, TileShape tile_shape, ClusterShape cluster_shape) {
    if (this->scheduler_params.mods.reporting_enabled()) {
      int linear_index = this->block_rank_in_grid(params, work_tile_info, problem_shape, tile_shape, cluster_shape);
      this->scheduler_params.mods.tile_info[linear_index].sm_id = cutlass::arch::SmId();
    }
  }

  // Mods feature: Report mainloop start time
  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  CUTLASS_DEVICE
  void
  report_mainloop_start_time(Params const& params, WorkTileInfo const& work_tile_info,  ProblemShapeMNKL const& problem_shape, TileShape tile_shape, ClusterShape cluster_shape, int AtomThrShapeMNK) {
    if (this->scheduler_params.mods.reporting_enabled()) {
      int linear_index = this->block_rank_in_grid(params, work_tile_info, problem_shape, tile_shape, cluster_shape);
      auto start_ts = clock64();
      this->scheduler_params.mods.tile_info[linear_index].mainloop_start_ts = start_ts;
      if (AtomThrShapeMNK == 2) {
        this->scheduler_params.mods.tile_info[linear_index + 1].mainloop_start_ts = start_ts;
      }
    }
  }

  // Mods feature: Report mainloop end time
  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  CUTLASS_DEVICE
  void
  report_mainloop_end_time(Params const& params, WorkTileInfo const& work_tile_info,  ProblemShapeMNKL const& problem_shape, TileShape tile_shape, ClusterShape cluster_shape, int AtomThrShapeMNK) {
    if (this->scheduler_params.mods.reporting_enabled()) {
      int linear_index = this->block_rank_in_grid(params, work_tile_info, problem_shape, tile_shape, cluster_shape);
      auto end_ts = clock64();
      this->scheduler_params.mods.tile_info[linear_index].mainloop_end_ts = end_ts;
      if (AtomThrShapeMNK == 2) {
        this->scheduler_params.mods.tile_info[linear_index + 1].mainloop_end_ts = end_ts;
      }
    }
  }
  // {$nv-release-never end}

  // {$nv-release-never begin}
  template<class ModsArguments>
  CUTLASS_DEVICE
  void
  mods_throttle(ModsArguments const& mods) {
    while (IsModsEnabled && mods.looping_controls.throttling_enabled()) {
      uint32_t mods_cur_time_ns = cutlass::arch::detail::globaltimer_lo();
      uint32_t mods_masked_time_ns = mods_cur_time_ns % mods.looping_controls.period_ns;
      if (mods_masked_time_ns <= mods.looping_controls.pulse_ns) {
        break;
      }
      __nanosleep(mods.looping_controls.min_sleep);
    }
  }
  // {$nv-release-never end}
};
}
