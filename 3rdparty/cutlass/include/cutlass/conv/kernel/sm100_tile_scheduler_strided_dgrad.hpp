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
// {$nv-internal-release file}

#pragma once

#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/conv/collective/detail.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::kernel::detail {

//////////////////// Blackwell Scheduler /////////////////////////

template<
  int NumSpatialDims,
  class ClusterShape_,
  int Stages_
>
class PersistentTileSchedulerSm100StridedDgrad {

public:
  using ClusterShape = ClusterShape_;

private:
  using ProblemShape = cutlass::conv::ConvProblemShape<conv::Operator::kDgrad, NumSpatialDims>;

  using UnderlyingTileScheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape_,Stages_>;
  static constexpr int MaxTraversalStride = 8; // TMA limitation

  // Stateful iterator that does:
  //   for (int ti = t_offset, zi = z_offset; ti < T; ti += t_stride, zi -= z_stride) {
  //   for (int ri = r_offset, pi = p_offset; ri < R; ri += r_stride, pi -= p_stride) {
  //   for (int si = s_offset, qi = q_offset; si < S; si += s_stride, qi -= q_stride) {
  //   for (int ki = 0;                       ki < ceil_div(K, TILE_K); ++ki) {
  struct KTileIterator {
    using SpatialCoord = array<int, NumSpatialDims>;

    int       k_coord; // k
    int const k_shape; // ceil_div(K, TILE_K)

    SpatialCoord       flt_coord;  // (s,r,t)
    SpatialCoord const flt_offset; // (s,r,t)
    SpatialCoord const flt_stride; // stride (s,r,t)
    SpatialCoord const flt_shape;  // (S,R,T)

    SpatialCoord       out_coord;  // (q,p,z)
    SpatialCoord const out_offset; // (q,p,z)
    SpatialCoord const out_stride; // stride (q,p,z)

    CUTLASS_DEVICE
    KTileIterator(int k_shape,
                  SpatialCoord flt_offset,
                  SpatialCoord flt_stride,
                  SpatialCoord flt_shape,
                  SpatialCoord out_offset,
                  SpatialCoord out_stride)
      : k_coord(0),
        k_shape(k_shape),
        flt_coord(flt_offset),
        flt_offset(flt_offset),
        flt_stride(flt_stride),
        flt_shape(flt_shape),
        out_coord(out_offset),
        out_offset(out_offset),
        out_stride(out_stride) {}

    CUTLASS_DEVICE
    KTileIterator&
    operator++() {
      k_coord++;

      bool increment_next = k_coord == k_shape;
      if (increment_next) {
        k_coord = 0;
        flt_coord[0] += flt_stride[0];
        out_coord[0] -= out_stride[0];
      }

      CUTLASS_PRAGMA_UNROLL
      for (int mode = 0; mode < NumSpatialDims-1; mode++) {
        increment_next &= flt_coord[mode] >= flt_shape[mode];
        if (increment_next) {
          flt_coord[mode] = flt_offset[mode];
          out_coord[mode] = out_offset[mode];
          flt_coord[mode+1] += flt_stride[mode+1];
          out_coord[mode+1] -= out_stride[mode+1];
        }
      }

      return *this;
    }

    CUTLASS_DEVICE
    auto
    operator*() {
      return cute::make_tuple(k_coord, flt_coord, out_coord);
    }
  };


public:
  using RasterOrder = typename UnderlyingTileScheduler::RasterOrder;
  using RasterOrderOptions = typename UnderlyingTileScheduler::RasterOrderOptions;  // {$nv-internal-release}
  static constexpr bool IsModsEnabled = false; // {$nv-release-never}

  static constexpr uint32_t Stages = Stages_;

  using CLCResponse = typename UnderlyingTileScheduler::CLCResponse;

  using Pipeline = typename UnderlyingTileScheduler::Pipeline;

  using WorkTileInfo = typename UnderlyingTileScheduler::WorkTileInfo;

  using Arguments = typename UnderlyingTileScheduler::Arguments;

  struct Params : UnderlyingTileScheduler::Params {
    // Precomputed decomposition strides + offsets
    int8_t flt_stride[NumSpatialDims]; // (s, r, t)
    int8_t out_stride[NumSpatialDims]; // (q, p, z)
    int8_t out_offset[NumSpatialDims][MaxTraversalStride]; // (q, p, z) for each (s,r,t)
    int8_t act_offset[NumSpatialDims][MaxTraversalStride]; // (w, h, d) for each (s,r,t)
    // Some decompositions may backpropagate zeros because there is no corresponding output
    bool backprop_zeros[NumSpatialDims][MaxTraversalStride]; // (s, r, t) for each (s,r,t)
  };

  template <class TileShape, class AtomThrShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShape problem_shape,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace = nullptr) { 
    
    Params params;

    // Initialize underlying params
    static_cast<typename UnderlyingTileScheduler::Params&>(params) =
      UnderlyingTileScheduler::to_underlying_arguments(
        problem_shape,
        tile_shape_mnk,
        atom_thr_shape_mnk,
        cluster_shape_mnk,
        hw_info,
        args,
        workspace
      );

    // Compute decomposition strides + offsets for each spatial mode
    for (int mode = 0; mode < NumSpatialDims; ++mode) {
      int xform_mode = NumSpatialDims - 1 - mode;
      int traversal_stride = problem_shape.traversal_stride[mode];
      int dilation = problem_shape.dilation[mode];
      int flt_shape = problem_shape.shape_B[mode+1];
      int lower_pad = problem_shape.lower_padding[mode];

      // Compute decomposition strides
      if (traversal_stride % dilation == 0) {
        params.flt_stride[xform_mode] = int8_t(traversal_stride / dilation);
        params.out_stride[xform_mode] = 1;
      }
      else if (dilation % traversal_stride == 0) {
        params.flt_stride[xform_mode] = 1;
        params.out_stride[xform_mode] = int8_t(dilation / traversal_stride);
      }
      else {
        params.flt_stride[xform_mode] = int8_t(traversal_stride);
        params.out_stride[xform_mode] = int8_t(dilation);
      }

      // Compute activation + output decomposition offsets for each possible filter decomp offset (up to activation traversal stride)
      for (int flt_offset = 0; flt_offset < traversal_stride; ++flt_offset) {
        int dilated_flt_offset = flt_offset * dilation;
        int act_offset_ = -lower_pad;
        if (traversal_stride % dilation == 0) {
          act_offset_ += dilated_flt_offset + dilated_flt_offset / traversal_stride;
          params.backprop_zeros[xform_mode][flt_offset] = flt_offset >= params.flt_stride[xform_mode];
        }
        else if (dilation % traversal_stride == 0) {
          act_offset_ += flt_offset;
          params.backprop_zeros[xform_mode][flt_offset] = flt_offset > 0;
        }
        else {
          act_offset_ += dilated_flt_offset;
          params.backprop_zeros[xform_mode][flt_offset] = false;
        }
        act_offset_ = ((act_offset_ % traversal_stride) + traversal_stride) % traversal_stride; // modulo
        params.act_offset[xform_mode][flt_offset] = int8_t(act_offset_);
        params.out_offset[xform_mode][flt_offset] = int8_t((act_offset_ + lower_pad - dilated_flt_offset) / traversal_stride);
        // Filter decomposition offset can exceed filter shape, in which case we backprop zeros
        params.backprop_zeros[xform_mode][flt_offset] |= flt_offset >= flt_shape;
      }
    }

    return params;
  }

  CUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const& args) {
    return true;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const& args, ProblemShape problem_shape, KernelHardwareInfo const& hw_info, uint32_t, uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream,
    ProblemShape const& problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t,
    uint32_t = 1,
    uint32_t = 1,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShape>
  static dim3
  get_grid_shape(
      Params const& params,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo hw_info) {
    return UnderlyingTileScheduler::get_grid_shape(
      params, problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cluster_shape_mnk, hw_info);
  }

  //
  // Constructors
  //
  template <class ProblemShapeMNKL, class TileShape>
  CUTLASS_DEVICE
  PersistentTileSchedulerSm100StridedDgrad(CLCResponse* clc_response_ptr, Params const& params, ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape, dim3 block_id_in_cluster)
    : impl_(UnderlyingTileScheduler(clc_response_ptr, params, problem_shape_mnkl, tile_shape, block_id_in_cluster)),
      params(params),
      block_id_in_cluster_(block_id_in_cluster),
      k_shape_(ceil_div(get<2,0>(problem_shape_mnkl), get<2,0>(tile_shape))),
      flt_stride_(apply(reinterpret_cast<cute::array<int8_t,NumSpatialDims> const&>(params.flt_stride), to_int_array{})),
      flt_shape_(apply(take<1,NumSpatialDims+1>(get<2>(problem_shape_mnkl)), to_int_array{})),
      out_stride_(apply(reinterpret_cast<cute::array<int8_t,NumSpatialDims> const&>(params.out_stride), to_int_array{})),
      traversal_stride_(apply(get<3>(problem_shape_mnkl), to_int_array{}))
      {}

private:
  //
  // Data members
  //
  UnderlyingTileScheduler impl_;
  Params const& params;
  dim3 block_id_in_cluster_;

  // Save constants in RF
  int const k_shape_;                                 // ceil_div(K, TILE_K))
  array<int, NumSpatialDims> const flt_stride_;       // stride (s,r,t)
  array<int, NumSpatialDims> const flt_shape_;        // (S,R,T)
  array<int, NumSpatialDims> const out_stride_;       // stride (q,p,z)
  array<int, NumSpatialDims> const traversal_stride_; // stride (w,h,d) == (V,U,O)

  // Decomposition offset changes with each call to fetch_next_work
  array<int, NumSpatialDims> flt_offset_;             // offset (s,r,t)
  bool backprop_zeros_;

  template <class L_idx>
  CUTLASS_DEVICE
  void
  update_decomposition_offset(L_idx l_idx) {
    flt_offset_ = apply(idx2crd(l_idx, traversal_stride_), to_int_array{});
    backprop_zeros_ = false;
    CUTLASS_PRAGMA_UNROLL
    for (int dim = 0; dim < NumSpatialDims; dim++) {
      backprop_zeros_ |= params.backprop_zeros[dim][flt_offset_[dim]];
    }
  }

public:

  //
  // Work Tile API
  //

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    WorkTileInfo work_tile_info = impl_.initial_work_tile_info(cluster_shape);

    update_decomposition_offset(work_tile_info.L_idx);

    return work_tile_info;
  }

  CUTLASS_DEVICE
  PipelineState<Stages> 
  advance_to_next_work(Pipeline& clc_pipeline, PipelineState<Stages> clc_pipe_producer_state) const {
    return impl_.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
  }

  // Kernel helper function to get next CLC ID
  template <class CLCPipeline, class CLCPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
      WorkTileInfo work_tile_info,
      CLCPipeline& clc_pipeline,
      CLCPipelineState clc_pipe_consumer_state) {
    auto next_work_tuple = impl_.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
    WorkTileInfo const& next_work_tile_info = get<0>(next_work_tuple);

    update_decomposition_offset(next_work_tile_info.L_idx);

    return next_work_tuple;
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord(WorkTileInfo const& work_tile_info) {
    // Get every cta coord in three dimensions of the cluster
    auto [cta_m_in_cluster, cta_n_in_cluster, cta_l_in_cluster] = block_id_in_cluster_;
    
    // coordinate tricks for epilogue in lieu of writing a new kernel layer
    // pass backprop_zeros as k_coord for the epilogue
    auto k_coord = backprop_zeros_;
    // Convert filter decomposition offset to activation decomposition offset, pass as l_coord to epilogue
    array<int,NumSpatialDims> act_offset;
    CUTLASS_PRAGMA_UNROLL
    for (int dim = 0; dim < NumSpatialDims; dim++) {
      act_offset[dim] = params.act_offset[dim][flt_offset_[dim]];
    }
    auto l_coord = act_offset; // (w,h,d)

    return make_coord(work_tile_info.M_idx, work_tile_info.N_idx, k_coord, l_coord);
  }

  //
  // K Tile API
  //

  // update_decomposition_offset must be called before this is called, via call to fetch_next_work
  template <class ProblemShapeMNKL, class TileShape, class Shape>
  CUTLASS_DEVICE auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShapeMNKL, TileShape, Shape) {
    // Convert filter decomposition offset to output decomposition offset
    array<int,NumSpatialDims> out_offset;
    for (int dim = 0; dim < NumSpatialDims; dim++) {
      out_offset[dim] = params.out_offset[dim][flt_offset_[dim]];
    }

    return KTileIterator(k_shape_, flt_offset_, flt_stride_, flt_shape_, out_offset, out_stride_);
  }

  // update_decomposition_offset must be called before this is called, via call to fetch_next_work
  template <class ProblemShapeMNKL, class TileShape>
  CUTLASS_DEVICE
  int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape) {
    if (backprop_zeros_) {
      return 0;
    }

    auto tiles_srt = ceil_div((as_arithmetic_tuple(flt_shape_) - as_arithmetic_tuple(flt_offset_)), flt_stride_); // (s,r,t)

    return k_shape_ * size(tiles_srt);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&) {
    return true;
  }

  // Returns whether fixup is needed for `work_tile_info`. None of the work units returned by
  // this scheduler require fixup, since none of the work units partition the reduction extent.
  CUTLASS_HOST_DEVICE
  static bool
  requires_fixup(Params const& params, WorkTileInfo const work_tile_info) {
    return false;
  }

  // Performs the reduction across splits for a given output tile. No fixup is required for
  // work units returned by this scheduler.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  fixup(WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t, uint32_t = 1) const { }

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
      cute::Tensor<AccEngine, AccLayout> ,
      AccumulatorPipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      CopyOpT2R) const {
    return acc_pipe_consumer_state;
  }

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo&) {
    return false;
  }

private:
  //
  // Helpers
  //
  struct to_int_array {
    template <class... T>
    CUTLASS_DEVICE
    cute::array<int,sizeof...(T)>
    operator()(T&&... x) const {
      return {static_cast<int>(x)...}; 
    }
  };

};

///////////////////////////////////////////////////////////////////////////////

} // end namespace cutlass::gemm::kernel::detail
