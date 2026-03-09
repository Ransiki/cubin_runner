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

/*! \file
    \brief Base scheduler for grouped problems in which problems in the overall group are placed
    into "sub-groups" with idential problem size and other parameters.
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace detail {

///
/// Leightweight wrappers around different modes for passing in problem sizes
///

/// Problem sizes passed in as an array of GemmCoords
struct ProblemSizeParamsCombined {
  GemmCoord* problem_sizes;
  bool transposed;

  CUTLASS_HOST_DEVICE
  ProblemSizeParamsCombined() : problem_sizes(nullptr), transposed(false) {}

  CUTLASS_HOST_DEVICE
  ProblemSizeParamsCombined(GemmCoord* problem_sizes_) : problem_sizes(problem_sizes_) {}

  CUTLASS_HOST_DEVICE
  void transpose_problem() {
    transposed = !transposed;
  }

  CUTLASS_HOST_DEVICE
  GemmCoord problem_size(int idx) const {
    GemmCoord problem = problem_sizes[idx];

    if (transposed) {
      return GemmCoord(problem.n(), problem.m(), problem.k());
    }

    return problem;
  }
};

/// Problem sizes passed in via arrays of individual M, N, and K sizes
struct ProblemSizeParamsSplit {
  int* Ms;
  int* Ns;
  int* Ks;

  CUTLASS_HOST_DEVICE
  ProblemSizeParamsSplit() : Ms(nullptr), Ns(nullptr), Ks(nullptr) {}

  CUTLASS_HOST_DEVICE
  ProblemSizeParamsSplit(int* Ms_, int* Ns_, int* Ks_) : Ms(Ms_), Ns(Ns_), Ks(Ks_) {}

  CUTLASS_HOST_DEVICE
  void transpose_problem() {
    cutlass::swap(Ms, Ns);
  }

  CUTLASS_HOST_DEVICE
  GemmCoord problem_size(int idx) const {
    return GemmCoord{Ms[idx], Ns[idx], Ks[idx]};
  }
};

} // detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles
template <
  typename ProblemSizeHelper,
  typename ProblemSizeParams_
>
struct BaseGroupedProblemVisitorSubGrouped {
  using ProblemSizeParams = ProblemSizeParams_;

  struct ProblemInfo {
    static int32_t const kNoPrefetchEntry = -1;
    int32_t problem_idx;
    int32_t problem_start;

    CUTLASS_DEVICE
    ProblemInfo() : problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

    CUTLASS_DEVICE
    ProblemInfo(int32_t problem_idx_, int32_t problem_start_) :
      problem_idx(problem_idx_), problem_start(problem_start_) {}
  };

  struct Params {
    ProblemSizeParams problem_size_params;
    int32_t           group_count;
    int32_t          *group_sizes;
    void const       *workspace;
    int32_t           tile_count;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(): workspace(nullptr), tile_count(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(
      ProblemSizeParams problem_sizes,
      int32_t          *group_sizes,
      int32_t           group_count,
      void const       *workspace = nullptr,
      int32_t           tile_count = 0
    ):
      problem_size_params(problem_sizes),
      group_sizes(group_sizes),
      group_count(group_count),
      workspace(workspace),
      tile_count(tile_count)
    {}

  };

  Params const &params;
  int32_t tile_idx;
  int32_t problem_tile_start;
  int32_t problem_idx;
  int32_t group_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  BaseGroupedProblemVisitorSubGrouped(
    Params const &params_,
    int32_t block_idx
  ):
  params(params_),
  tile_idx(block_idx),
  problem_tile_start(0),
  problem_idx(0),
  group_idx(0)
  {}

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return ProblemSizeHelper::grid_shape(problem);
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int32_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  /// Gets the index of the group
  CUTLASS_HOST_DEVICE
  int32_t group_index() const {
    return group_idx;
  }

  CUTLASS_HOST_DEVICE
  int32_t threadblock_idx() const {
    return tile_idx - problem_tile_start;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem) {
    ProblemSizeHelper::possibly_transpose_problem(problem);
  }

  /// Returns a given problem size
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size(int32_t idx) const {
    GemmCoord problem = params.problem_size_params.problem_size(idx);
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    return problem_size(group_idx);
  }

  /// Returns a given group size
  CUTLASS_HOST_DEVICE
  int group_size(int32_t idx) const {
    return params.group_sizes[idx];
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord& grid) {
    return ProblemSizeHelper::tile_count(grid);
  }

  static int32_t group_tile_count(
    const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
    int32_t group_count,
    int32_t* group_sizes=nullptr) {
    int32_t total_tiles = 0;
    for (int32_t i = 0; i < group_count; ++i) {
      auto problem = host_problem_sizes_ptr[i];
      possibly_transpose_problem(problem);
      auto grid = grid_shape(problem);
      total_tiles += (tile_count(grid) * group_sizes[i]);
    }

    return total_tiles;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that performs all scheduling on device
//
template <
  typename ProblemSizeHelper,
  typename ProblemSizeParams_ = detail::ProblemSizeParamsCombined
>
struct GroupedProblemVisitorSubGrouped : public BaseGroupedProblemVisitorSubGrouped<ProblemSizeHelper, ProblemSizeParams_> {
  using Base = BaseGroupedProblemVisitorSubGrouped<ProblemSizeHelper, ProblemSizeParams_>;
  using Params = typename Base::Params;
  static bool const kRequiresPrecomputation = false;
  static int const kThreadsPerWarp = 32;

  struct SharedStorage {};

  // Final tile of the problem loaded by this thread. Each thread will hold
  // a separate value.
  int32_t problem_ending_tile;

  SharedStorage &shared_storage;

  int32_t group_tile_start;
  int32_t group_ending_tile;
  int32_t group_problem_start;
  int32_t group_ending_problem;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitorSubGrouped(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  problem_ending_tile(0),
  group_tile_start(0),
  group_problem_start(0),
  group_ending_tile(0),
  group_ending_problem(0),
  shared_storage(shared_storage_)
  {
    this->problem_idx = -1 * kThreadsPerWarp;
    this->group_idx = -1 * kThreadsPerWarp;
    this->problem_tile_start = 0;
  }

  CUTLASS_DEVICE
  void set_problem_from_group(int32_t group_tile_end, int32_t group_problem_end) {
    int32_t tiles_in_group = group_tile_end - this->group_tile_start;
    int32_t problems_in_group = group_problem_end - this->group_problem_start;
    int32_t tiles_per_problem = tiles_in_group / problems_in_group;
    int32_t problem_idx_in_group = (this->tile_idx - this->group_tile_start) / tiles_per_problem;
    this->problem_idx = group_problem_start + problem_idx_in_group;
    this->problem_tile_start = this->group_tile_start + (problem_idx_in_group * tiles_per_problem);
  }

  CUTLASS_DEVICE
  bool next_tile() {
    // Check whether the tile to compute is within the range of the current group.
    int32_t group_tile_end = __shfl_sync(0xffffffff, group_ending_tile, this->group_idx % kThreadsPerWarp);

    if (this->tile_idx < group_tile_end) {
      int32_t group_problem_end = __shfl_sync(0xffffffff, group_ending_problem, this->group_idx % kThreadsPerWarp);
      set_problem_from_group(group_tile_end, group_problem_end);
      return true;
    }

    // Check whether the tile to compute is within the current set of groups fetched by the warp.
    // The last tile for this set is the final tile of the group held by the final thread in the warp.
    int32_t set_tile_end = __shfl_sync(0xffffffff, group_ending_tile, kThreadsPerWarp-1);

    // Keep the starting group for this set in `group_idx`. This is done to reduce
    // register pressure. The starting group for this set is simply the first problem
    // in the set most recently fetched by the warp.
    int32_t &set_group_start = this->group_idx;
    set_group_start = (this->group_idx / kThreadsPerWarp) * kThreadsPerWarp;

    // Keep the starting tile for this set in `group_tile_start`. This is done to reduce
    // register pressure.
    int32_t &set_tile_start = this->group_tile_start;
    int32_t &set_problem_start = this->group_problem_start;
    int32_t set_problem_end = __shfl_sync(0xffffffff, group_ending_problem, kThreadsPerWarp-1);

    // Each thread in the warp processes a separate group to advance until
    // reaching a group whose starting tile is less less than tile_idx.
    while (set_tile_end <= this->tile_idx) {
      set_group_start += kThreadsPerWarp;
      if (set_group_start > this->params.group_count) {
        return false;
      }

      // Since `set_tile_start` is a reference to `this->group_tile_start`, this
      // also sets `this->group_tile_start`. The fact that `this->group_tile_start`
      // is also set here is used later..
      set_tile_start = set_tile_end;
      set_problem_start = set_problem_end;

      // The problem ID following the final problem in this group
      group_ending_problem = 0;
      group_ending_tile = 0;

      int lane_idx = threadIdx.x % kThreadsPerWarp;
      int32_t lane_group = set_group_start + lane_idx;

      if (lane_group < this->params.group_count) {
        // Temporarily set group_ending_problem to the number of problems in the group.
        // We will later perform a  prefix sum over these values to obtain the true ending
        // problem in the group. We do similarly for group_ending_tile.
        cutlass::gemm::GemmCoord problem = this->problem_size(lane_group);
        cutlass::gemm::GemmCoord grid = this->grid_shape(problem);
        int group_size = this->group_size(lane_group);
        group_ending_tile = this->tile_count(grid) * group_size;
        group_ending_problem = group_size;
      }

      // Compute warp-wide inclusive prefix sum to compute the ending problem and tile of each
      // thread's group.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
        int32_t val_problem = __shfl_up_sync(0xffffffff, group_ending_problem, i);
        int32_t val_tile = __shfl_up_sync(0xffffffff, group_ending_tile, i);
        if (lane_idx >= i) {
          group_ending_problem += val_problem;
          group_ending_tile += val_tile;
        }
      }

      // The total tile and problem count for this set is now in the final position of the prefix sum
      int32_t tiles_in_set = __shfl_sync(0xffffffff, group_ending_tile, kThreadsPerWarp-1);
      int32_t problems_in_set = __shfl_sync(0xffffffff, group_ending_problem, kThreadsPerWarp-1);

      group_ending_tile += set_tile_start;
      group_ending_problem += set_problem_start;
      set_tile_end += tiles_in_set;
      set_problem_end += problems_in_set;
    }

    // The next group to process is the first one that does not have ending tile position
    // that is greater than or equal to tile index.
    int32_t group_idx_in_set =
        __popc(__ballot_sync(0xffffffff, group_ending_tile <= this->tile_idx));

    this->group_idx = set_group_start + group_idx_in_set;

    // The starting tile for this group is the ending tile of the previous problem. In cases
    // where `group_idx_in_set` is the first group in the set, we do not need to reset
    // `group_tile_start`, because it is set to the previous set's ending tile in the while
    // loop above.
    if (group_idx_in_set > 0) {
      this->group_tile_start = __shfl_sync(0xffffffff, group_ending_tile, group_idx_in_set - 1);
      this->group_problem_start = __shfl_sync(0xffffffff, group_ending_problem, group_idx_in_set - 1);
    }
    int32_t final_tile_in_group = __shfl_sync(0xffffffff, group_ending_tile, group_idx_in_set);
    int32_t group_problem_end = __shfl_sync(0xffffffff, group_ending_problem, group_idx_in_set);

    set_problem_from_group(final_tile_in_group, group_problem_end);
    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    return 0;
  }

  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////