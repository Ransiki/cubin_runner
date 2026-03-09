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

#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/conv/collective/detail.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::conv::kernel::detail {

template <class TileSchedulerBase, class FilterShape_TRS>
class Nq2dTiledSchedulerSm100 : public TileSchedulerBase {

private:

  using Base = TileSchedulerBase;

  struct SplitPParams {
    int slices;
    FastDivmod fast_divmod_N;
    int conv_p;
    int p_step;
    int h_pixels_start_offset;
    int h_pixels_end_offset;
    int p_residual;
  };

public:

  using Base::IsModsEnabled; // {$nv-release-never}
  using typename Base::CLCResponse;
  using typename Base::WorkTileInfo;

  static constexpr int FltR = size<1>(FilterShape_TRS{});

  struct Arguments : Base::Arguments {
    int split_p_slices = 1;
  };

  struct Params {
    typename Base::Params base_params;
    SplitPParams split_p_params;
  };

  CUTLASS_DEVICE
  Nq2dTiledSchedulerSm100(CLCResponse* clc_response_ptr, Params const& params)
   : Base(clc_response_ptr, params.base_params), split_p_params(params.split_p_params) {}

  CUTLASS_DEVICE
  Nq2dTiledSchedulerSm100(CLCResponse* clc_response_ptr, Params const& params, dim3 block_id_in_cluster)
   : Base(clc_response_ptr, params.base_params, block_id_in_cluster), split_p_params(params.split_p_params) {}

  template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      int lower_corner_h,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      [[maybe_unused]] ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace = nullptr) {

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    constexpr auto cta_shape_mnk = shape_div(TileShape{}, AtomThrShape{});
    SplitPParams split_p_params;
    split_p_params.slices = args.split_p_slices;
    auto num_n_tile = int(size<1>(cs) * ceil_div(size<1>(problem_shape_mnkl), size<1>(cta_shape_mnk) * size<1>(cs)));
    split_p_params.fast_divmod_N = FastDivmod(num_n_tile);
    split_p_params.conv_p = shape<2,1>(problem_shape_mnkl);
    split_p_params.p_step = split_p_params.conv_p / split_p_params.slices;
    split_p_params.p_residual = split_p_params.conv_p - split_p_params.p_step * split_p_params.slices;
    split_p_params.h_pixels_start_offset = lower_corner_h;
    split_p_params.h_pixels_end_offset = lower_corner_h + (FltR - 1);

    // Force AlongM mode to be used for the split-P scheduler
    Arguments new_args = args;
    new_args.raster_order = Base::RasterOrderOptions::AlongM;

    return {
      Base::to_underlying_arguments(
          problem_shape_mnkl,
          tile_shape_mnk,
          atom_thr_shape_mnk,
          cs,
          hw_info,
          static_cast<typename Base::Arguments>(new_args),
          workspace),
      split_p_params
    };
  }

  template<class ProblemShapeMNKL, class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
      Params const& params,
      ProblemShapeMNKL problem_shape_mnk,
      BlockShape cta_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo hw_info) {
    auto grid = Base::get_grid_shape(params.base_params, problem_shape_mnk, cta_shape, cluster_shape, hw_info);
    return {grid.x, grid.y * params.split_p_params.slices, grid.z};
  }

  template<class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
      Params const& params,
      ProblemShapeMNKL problem_shape_mnk,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo hw_info) {
    auto grid = Base::get_grid_shape(
        params.base_params,
        problem_shape_mnk,
        tile_shape_mnk,
        atom_thr_shape_mnk,
        cluster_shape_mnk,
        hw_info);
    return {grid.x, grid.y * params.split_p_params.slices, grid.z};
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord_and_p_range(WorkTileInfo work_tile_info) {
    auto [N_idx, p_pixels_start, p_pixels_end] = get_n_idx_and_p_range(split_p_params, work_tile_info.N_idx);
    work_tile_info.N_idx = N_idx;
    auto tile_coord_mnkl = Base::work_tile_to_cta_coord(work_tile_info);
    return cute::make_tuple(tile_coord_mnkl, p_pixels_start, p_pixels_end);
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord_and_h_range(WorkTileInfo work_tile_info) {
    auto [N_idx, h_pixels_start, h_pixels_end] = get_n_idx_and_h_range(split_p_params, work_tile_info.N_idx);
    work_tile_info.N_idx = N_idx;
    auto tile_coord_mnkl = Base::work_tile_to_cta_coord(work_tile_info);
    return cute::make_tuple(tile_coord_mnkl, h_pixels_start, h_pixels_end);
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cluster_coord_mnkl_and_h_range(WorkTileInfo work_tile_info) {
    auto [N_idx, h_pixels_start, h_pixels_end] = get_n_idx_and_h_range(split_p_params, work_tile_info.N_idx);
    work_tile_info.N_idx = N_idx;
    auto tile_coord_mnkl = Base::work_tile_to_cluster_coord_mnkl(work_tile_info);
    return cute::make_tuple(tile_coord_mnkl, h_pixels_start, h_pixels_end);
  }

private:

  SplitPParams split_p_params;

  CUTLASS_DEVICE
  static auto
  get_n_idx_and_p_range(SplitPParams const& split_p, int32_t worktile_N_idx) {
    int N_idx, p_idx;
    // Dispatch entire N_tile then dispatch next P_tile
    split_p.fast_divmod_N(p_idx, N_idx, worktile_N_idx);

    // One more p workload if p_idx < p_residual
    int p_pixels_start, p_pixels_end;
    if (p_idx < split_p.p_residual) {
      p_pixels_start = p_idx * (split_p.p_step + 1);
      p_pixels_end   = p_pixels_start + (split_p.p_step + 1);
    }
    else {
      p_pixels_start = p_idx * split_p.p_step + split_p.p_residual;
      p_pixels_end   = p_pixels_start + split_p.p_step;
    }

    return cute::make_tuple(N_idx, p_pixels_start, p_pixels_end);
  }

  CUTLASS_DEVICE
  static auto
  get_n_idx_and_h_range(SplitPParams const& split_p, int32_t worktile_N_idx) {
    auto [N_idx, p_pixels_start, p_pixels_end] = get_n_idx_and_p_range(split_p, worktile_N_idx);
    auto h_pixels_start = p_pixels_start + split_p.h_pixels_start_offset;
    auto h_pixels_end   = p_pixels_end   + split_p.h_pixels_end_offset;
    return cute::make_tuple(N_idx, h_pixels_start, h_pixels_end);
  }

};


////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cutlass::conv::kernel::detail
