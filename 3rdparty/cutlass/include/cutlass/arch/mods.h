/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

// {$nv-release-never file}

#pragma once

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"

#include "cutlass/trace.h"
#include "cutlass/workspace.h"

#endif

namespace cutlass {
namespace mods {


class ModsInstrumentation {

public:
  // report the following information for each output tile
  struct OutputTileInfo {
    int sm_id                 = -1;
    int64_t mainloop_start_ts = -1;
    int64_t mainloop_end_ts   = -1;
  };

  // control the mainloop execution based the following arguments
  struct SyntheticLoopingControls {
    size_t period_ns       = 0;
    size_t pulse_ns        = 0;
    size_t min_sleep       = 0;
    int32_t mainloop_count = 0;

    // Returns whether MODS instrumentation is set up to control the time between mainloop repetitions
    CUTLASS_HOST_DEVICE
    bool
    throttling_enabled() const {
      return period_ns > 0;
    }
  };

  struct Arguments {
    int32_t num_tiles = 0;
    SyntheticLoopingControls looping_controls{};
  };

  struct Params {
    OutputTileInfo *tile_info = nullptr;
    int32_t num_tiles = 0;
    SyntheticLoopingControls looping_controls{};

    // Returns whether the MODS instrumentation is set up to enable reporting timing information
    CUTLASS_HOST_DEVICE
    bool
    reporting_enabled() const {
      // Don't attempt to report if the `num_tiles` parameter hasn't been initialized.
      // It is not safe to assume that `tile_info` is nullptr in cases in which
      // reporting is not enabled because a workspace pointer is typically passed in to the
      // MODS `to_underlying_arguments` method based on incrementing an existing workspace
      // pointer passed to another part of the kernel. If the workspace pointer for the kernel
      // is not nullptr, then the pointer passed to `to_underlying_arguments` will also not
      // be nulltpr.
      return num_tiles > 0;
    }
  };

  static Params
  to_underlying_arguments(
      Arguments const& args,
      void* workspace = nullptr) {
    Params params;
    params.tile_info = reinterpret_cast<OutputTileInfo*>(workspace);
    params.num_tiles = args.num_tiles;
    params.looping_controls = args.looping_controls;
    return params;
  }

  static cutlass::Status
  initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
#if !defined(__CUDACC_RTC__)
    return zero_workspace(workspace, get_workspace_size(args), stream);
#endif
    return Status::kSuccess;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return sizeof(OutputTileInfo) * args.num_tiles;
  }

  template<typename Kernel, typename ProblemShapeType>
  static int
  get_num_output_tiles(ProblemShapeType problem_shape_mnkl) {
    using CtaShape_MNK = typename Kernel::CollectiveMainloop::CtaShape_MNK;
    using ClusterShape = typename Kernel::CollectiveMainloop::DispatchPolicy::ClusterShape;
    auto output_grid = Kernel::TileScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, CtaShape_MNK{}, ClusterShape{});
    return output_grid.x * output_grid.y * output_grid.z;
  }
};

enum class InstrumentationCategoryClass {
  None,
  Mods
};

} // namespace mods
} // namespace cutlass
