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

/* \file
   \brief Template for BatchNormStatEnding kernel
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace kernel {
//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                   BatchNormStatEnding
//
// This kernel aims to do some finishing touches after convolution-batchnorm-fusion kernels.
// The procedure is described as follows:
//
// step1: sum reduction (num_channels, num_partial_results) --> (num_channels)
//        for partial results (only needed when IsBatchNormStatFinal == false)
//
// (1) sum = sum_reduce(partial_sum)
// (2) sum_of_square = sum_reduce(partial_sum_of_square) 
// (3) second_sum_of_square = sum_reduce(partial_second_sum_of_square)
//
// note that: 
//        (3) only needed when ConvolutionOperator == conv::Operator::kDgrad 
//                                & IsDualBatchNormSupported == true
//
// step2: calculation among vectors with shape (num_channels)
// (1) For ConvolutionOperator == conv::Operator::kFprop:
//                              mean = sum / (N * D * H * W)
//                     var = sum_of_square / (N * D * H * W) - mean * mean
//                      inv_stddev = inverse_square_root(var + epsilon)
//
// (2) For ConvolutionOperator == conv::Operator::kDgrad:
//                   sum_of_square = (sum_of_square - mean * sum) * inv_stddev
//                          dbna_eq_dy_scale = alpha * inv_stddev
//        dbna_eq_x_scale = -dbna_eq_dy_scale / (N * D * H * W) * sum_of_square * inv_stddev
// dbna_eq_bias_scale = dbna_eq_dy_scale / (N * D * H * W) * (mean * sum_of_square * inv_stddev - sum)
//
//     For IsDualBatchNormSupported == true, there're some extra computations:
//   second_sum_of_square = (second_sum_of_square - second_mean * sum) * second_inv_stddev
//                 second_dbna_eq_dy_scale = second_alpha * second_inv_stddev
//               second_dbna_eq_x_scale = -second_dbna_eq_dy_scale / (N * D * H * W)
//                         * second_sum_of_square * second_inv_stddev
//             second_dbna_eq_bias_scale = second_dbna_eq_dy_scale / (N * D * H * W)
//               * (second_mean * second_sum_of_square * second_inv_stddev - sum)
//
//////////////////////////////////////////////////////////////////////////////////////////////////////
template <
  conv::Operator ConvolutionOperator_,
  class BatchNormStatOperation_,
  int NumStatsPerThread_ = 1,
  int DimY_ = 32
>
struct BatchNormStatEnding {
  using BatchNormStatOperation = BatchNormStatOperation_;
  using ElementBatchNormApply = typename BatchNormStatOperation::ElementScalar;
  using ElementBatchNormStat = typename BatchNormStatOperation::ElementBatchNormStat;
  static constexpr conv::Operator ConvolutionOperator = ConvolutionOperator_;
  static constexpr bool IsDualBatchNormSupported = BatchNormStatOperation::IsDualBatchNormSupported;
  static constexpr bool IsBatchNormStatFinal = BatchNormStatOperation::IsBatchNormStatFinal;
  static constexpr int NumStatsPerThread = NumStatsPerThread_;
  static constexpr int DimY = DimY_;

  static constexpr uint32_t MaxThreadsPerBlock = NumThreadsPerWarp * DimY;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  using ArchTag = arch::Sm100;

  template <
    class ElementBatchNormApply,
    class ElementBatchNormStat,
    conv::Operator ConvolutionOperator,
    bool IsDualBatchNormSupported
  >
  struct BatchNormStatEndingArguments {
    static_assert(cutlass::detail::dependent_false<ElementBatchNormStat>,
      "Could not find a specialization of BatchNormStatEndingArguments.");
  };

  template <
    class ElementBatchNormApply,
    class ElementBatchNormStat
  >
  struct BatchNormStatEndingArguments<
    ElementBatchNormApply,
    ElementBatchNormStat,
    conv::Operator::kFprop,
    false
  > {
    using StridePartialResult = typename cute::Stride<_1, int64_t>;

    int num_partial_results = 0;
    int num_channels = 0;

    StridePartialResult stride_partial_result = {};

    ElementBatchNormStat inv_count = ElementBatchNormStat(0);
    ElementBatchNormStat epsilon = ElementBatchNormStat(0);

    // Input: partial results of batchnorm stat
    ElementBatchNormStat const* partial_sum = nullptr;
    ElementBatchNormStat const* partial_sum_of_square = nullptr;

    // Output: results of sum, sum_of_square, mean, inv_stddev
    ElementBatchNormStat* sum = nullptr;
    ElementBatchNormStat* sum_of_square = nullptr;
    ElementBatchNormStat* mean = nullptr;
    ElementBatchNormStat* inv_stddev = nullptr;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments() { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments(BatchNormStatEndingArguments const& args):
      num_partial_results(args.num_partial_results),
      num_channels(args.num_channels),
      stride_partial_result(args.stride_partial_result),
      inv_count(args.inv_count),
      epsilon(args.epsilon),
      partial_sum(args.partial_sum),
      partial_sum_of_square(args.partial_sum_of_square),
      sum(args.sum),
      sum_of_square(args.sum_of_square),
      mean(args.mean),
      inv_stddev(args.inv_stddev) { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments& operator=(BatchNormStatEndingArguments const&) = default;

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments(BatchNormStatEndingArguments const& args, int* workspace):
      BatchNormStatEndingArguments(args) { }
  };

  template <
    class ElementBatchNormApply,
    class ElementBatchNormStat
  >
  struct BatchNormStatEndingArguments<
    ElementBatchNormApply,
    ElementBatchNormStat,
    conv::Operator::kDgrad,
    false
  > {
    using StridePartialResult = typename cute::Stride<_1, int64_t>;

    int num_partial_results = 0;
    int num_channels = 0;

    StridePartialResult stride_partial_result = {};

    ElementBatchNormStat inv_count = ElementBatchNormStat(0);
    ElementBatchNormStat epsilon = ElementBatchNormStat(0);

    // Input: partial results of batchnorm stat
    ElementBatchNormStat const* partial_sum = nullptr;
    ElementBatchNormStat const* partial_sum_of_square = nullptr;

    // Input: parameters of batchnorm apply
    ElementBatchNormApply const* fprop_alpha = nullptr;
    ElementBatchNormApply const* fprop_mean = nullptr;
    ElementBatchNormApply const* fprop_inv_stddev = nullptr;

    // Output: results of sum, sum_of_square, mean, inv_stddev
    ElementBatchNormStat* sum = nullptr;
    ElementBatchNormStat* sum_of_square = nullptr;
    ElementBatchNormStat* dbna_eq_dy_scale = nullptr;
    ElementBatchNormStat* dbna_eq_x_scale = nullptr;
    ElementBatchNormStat* dbna_eq_bias_scale = nullptr;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments() { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments(BatchNormStatEndingArguments const& args):
      num_partial_results(args.num_partial_results),
      num_channels(args.num_channels),
      stride_partial_result(args.stride_partial_result),
      inv_count(args.inv_count),
      epsilon(args.epsilon),
      partial_sum(args.partial_sum),
      partial_sum_of_square(args.partial_sum_of_square),
      fprop_alpha(args.fprop_alpha),
      fprop_mean(args.fprop_mean),
      fprop_inv_stddev(args.fprop_inv_stddev),
      sum(args.sum),
      sum_of_square(args.sum_of_square),
      dbna_eq_dy_scale(args.dbna_eq_dy_scale),
      dbna_eq_x_scale(args.dbna_eq_x_scale),
      dbna_eq_bias_scale(args.dbna_eq_bias_scale) { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments& operator=(BatchNormStatEndingArguments const&) = default;

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments(BatchNormStatEndingArguments const& args, int* workspace):
      BatchNormStatEndingArguments(args) { }
  };

  template <
    class ElementBatchNormApply,
    class ElementBatchNormStat
  >
  struct BatchNormStatEndingArguments<
    ElementBatchNormApply,
    ElementBatchNormStat,
    conv::Operator::kDgrad,
    true
  > {
    using StridePartialResult = typename cute::Stride<_1, int64_t>;

    int num_partial_results = 0;
    int num_channels = 0;

    StridePartialResult stride_partial_result = {};

    ElementBatchNormStat inv_count = ElementBatchNormStat(0);
    ElementBatchNormStat epsilon = ElementBatchNormStat(0);

    // Input: partial results of batchnorm stat
    ElementBatchNormStat const* partial_sum = nullptr;
    ElementBatchNormStat const* partial_sum_of_square = nullptr;

    // Input: parameters of batchnorm apply
    ElementBatchNormApply const* fprop_alpha = nullptr;
    ElementBatchNormApply const* fprop_mean = nullptr;
    ElementBatchNormApply const* fprop_inv_stddev = nullptr;

    // Output: results of sum, sum_of_square, mean, inv_stddev
    ElementBatchNormStat* sum = nullptr;
    ElementBatchNormStat* sum_of_square = nullptr;
    ElementBatchNormStat* dbna_eq_dy_scale = nullptr;
    ElementBatchNormStat* dbna_eq_x_scale = nullptr;
    ElementBatchNormStat* dbna_eq_bias_scale = nullptr;

    // Input: partial results of the second batchnorm stat
    ElementBatchNormStat const* second_partial_sum_of_square = nullptr;

    // Input: parameters of the second batchnorm apply
    ElementBatchNormApply const* second_fprop_alpha = nullptr;
    ElementBatchNormApply const* second_fprop_mean = nullptr;
    ElementBatchNormApply const* second_fprop_inv_stddev = nullptr;

    // Output: results of the second sum, sum_of_square, mean, inv_stddev
    ElementBatchNormStat* second_sum = nullptr;
    ElementBatchNormStat* second_sum_of_square = nullptr;
    ElementBatchNormStat* second_dbna_eq_dy_scale = nullptr;
    ElementBatchNormStat* second_dbna_eq_x_scale = nullptr;
    ElementBatchNormStat* second_dbna_eq_bias_scale = nullptr;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments() { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments(BatchNormStatEndingArguments const& args):
      num_partial_results(args.num_partial_results),
      num_channels(args.num_channels),
      stride_partial_result(args.stride_partial_result),
      inv_count(args.inv_count),
      epsilon(args.epsilon),
      partial_sum(args.partial_sum),
      partial_sum_of_square(args.partial_sum_of_square),
      fprop_alpha(args.fprop_alpha),
      fprop_mean(args.fprop_mean),
      fprop_inv_stddev(args.fprop_inv_stddev),
      sum(args.sum),
      sum_of_square(args.sum_of_square),
      dbna_eq_dy_scale(args.dbna_eq_dy_scale),
      dbna_eq_x_scale(args.dbna_eq_x_scale),
      dbna_eq_bias_scale(args.dbna_eq_bias_scale),
      second_partial_sum_of_square(args.second_partial_sum_of_square),
      second_fprop_alpha(args.second_fprop_alpha),
      second_fprop_mean(args.second_fprop_mean),
      second_fprop_inv_stddev(args.second_fprop_inv_stddev),
      second_sum(args.second_sum),
      second_sum_of_square(args.second_sum_of_square),
      second_dbna_eq_dy_scale(args.second_dbna_eq_dy_scale),
      second_dbna_eq_x_scale(args.second_dbna_eq_x_scale),
      second_dbna_eq_bias_scale(args.second_dbna_eq_bias_scale) { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments& operator=(BatchNormStatEndingArguments const&) = default;

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingArguments(BatchNormStatEndingArguments const& args, int* workspace):
      BatchNormStatEndingArguments(args) { }
  };

  template <
    class BatchNormStatEndingArguments,
    int NumStatsPerThread = 1
  >
  struct BatchNormStatEndingParams : BatchNormStatEndingArguments {

    /// Default ctor
    CUTLASS_HOST_DEVICE
    BatchNormStatEndingParams() { }

    CUTLASS_HOST_DEVICE
    BatchNormStatEndingParams(
      BatchNormStatEndingArguments const& args,
      void *workspace = nullptr
    ): BatchNormStatEndingArguments(args) {
      // From user's view, the base element is one ElementBatchNormStat, but indeed we use FragmentBatchNormStat as the base element
      this->stride_partial_result = make_coord(_1{}, size<1>(args.stride_partial_result) / NumStatsPerThread);
      this->num_channels = args.num_channels / NumStatsPerThread;
    }
  };

  using FragmentBatchNormStat = Array<ElementBatchNormStat, NumStatsPerThread>;
  using FragmentBatchNormApply = Array<ElementBatchNormApply, NumStatsPerThread>;
  using ConvertApply = NumericArrayConverter<ElementBatchNormStat, ElementBatchNormApply, NumStatsPerThread>;
  using Arguments = BatchNormStatEndingArguments<ElementBatchNormApply, ElementBatchNormStat, ConvolutionOperator, IsDualBatchNormSupported>;
  using StridePartialResult = typename Arguments::StridePartialResult;
  using Params = BatchNormStatEndingParams<Arguments, NumStatsPerThread>;

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params(args, workspace);
  }

  /// Shared memory storage structure
  struct SharedStorage {
    FragmentBatchNormStat smem_partial_stat[DimY * NumThreadsPerWarp];
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  //
  // Methods
  //

  static dim3 get_grid_shape(Params const& params) {
    return dim3(ceil_div(params.num_channels, NumThreadsPerWarp), 1, 1);
  }

  static dim3 get_block_shape() {
    return dim3(NumThreadsPerWarp, DimY, 1);
  }

  CUTLASS_HOST_DEVICE
  BatchNormStatEnding() { }

  static cutlass::Status
  can_implement(Arguments const& args) {
    if (args.num_channels % NumStatsPerThread == 0 && size<1>(args.stride_partial_result) % NumStatsPerThread == 0) {
      return cutlass::Status::kSuccess;
    };
    return cutlass::Status::kErrorInvalidProblem;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  // Sum reduction: (N_TRD, N_PR) --(phase1)--> (N_TRD, DIM_Y) --(phase2)--> (N_TRD)
  template <class PTensor, class ArgsTuple>
  CUTLASS_DEVICE
  FragmentBatchNormStat reduce(PTensor const& mP, ArgsTuple const& args_tuple, Params const& params, char *smem_buf) {
    auto& [blk_shape, blk_coord, init_residue_coord, tPcP, thread_idx] = args_tuple;
    auto gP = local_tile(mP, blk_shape, blk_coord);                                            // (N_TRD, DIM_Y, N_BLK)
    auto tCgP = local_partition(gP, make_layout(make_shape(Int<NumThreadsPerWarp>{}, Int<DimY>{})),  // (_1, _1, N_BLK)
      thread_idx);
    auto tCrP = make_tensor_like(take<0,2>(tCgP));
    FragmentBatchNormStat frg_result;
    frg_result.clear();

    // Sum reduction phase1: (N_TRD, N_PR) ----> (N_TRD, DIM_Y)
    auto residue_coord = init_residue_coord;
    for (int n_blk = 0; n_blk < size<2>(gP)/*N_BLK*/; ++n_blk) {
      // check OOB
      if (elem_less(tPcP(_0{},_0{}), residue_coord)) {
        copy_aligned(tCgP(_,_,n_blk), tCrP);
        frg_result = frg_result + tCrP(_0{});
      }
      // The residue_coord will be reduced by blk_shape in the second mode each iteration
      residue_coord = [] (auto const& a, auto const& b) { 
        return cute::transform(a, b, [] (auto const& a, auto const& b) {return a - b;});
      } (residue_coord, make_coord(0, get<1>(blk_shape)));
    }

    // Sum reduction phase2: (N_TRD, DIM_Y) ----> (N_TRD)
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto sP = make_tensor(reinterpret_cast<FragmentBatchNormStat*>(shared_storage.smem_partial_stat), // (N_TRD, DIM_Y)
      blk_shape);
    sP(threadIdx.x, threadIdx.y) = frg_result;

    // Make sure the data is in shared memory.
    __syncthreads();

    CUTLASS_PRAGMA_UNROLL
    for (int offset = DimY / 2; offset > 0; offset /= 2) {
      if (threadIdx.y < offset) {
        frg_result = sP(threadIdx.x, threadIdx.y + offset) + frg_result;
        sP(threadIdx.x, threadIdx.y) = frg_result;
      }
      __syncthreads();
    }

    return sP(threadIdx.x, 0);
  }

  /// Executes one BatchNormStatEnding
  CUTLASS_DEVICE
  void operator()(Params const& params, char *smem_buf) {
    // Ensure that the prefetched kernel does not touch
    // unflushed global memory prior to this instruction
    cutlass::arch::wait_on_dependent_grids();

    // The position of the thread.
    int channel_idx = blockIdx.x * NumThreadsPerWarp + threadIdx.x;

    auto shape_partial_result = make_shape(params.num_channels, params.num_partial_results);            // (N_CN, N_PR)
    auto shape_result = make_shape(params.num_channels);                                                      // (N_CN)

    auto blk_shape = make_shape(Int<NumThreadsPerWarp>{}, Int<DimY>{});                               // (N_TRD, DIM_Y)
    auto blk_coord = make_coord(blockIdx.x, _);

    // Coordinate tensors and residue for tile quantization
    int n_cn_max_coord = get<0>(shape_partial_result) - get<0>(blk_shape) * get<0>(blk_coord);
    int n_pr_max_coord = get<1>(shape_partial_result);
    auto residue_coord = make_coord(n_cn_max_coord, n_pr_max_coord);

    int thread_idx = threadIdx.x + threadIdx.y * DimY;
    auto cP = make_identity_tensor(blk_shape);                                                        // (N_TRD, DIM_Y)
    auto tPcP = local_partition(cP, make_layout(make_shape(Int<NumThreadsPerWarp>{}, Int<DimY>{})),         // (_1, _1)
      thread_idx);

    auto args_tuple = make_tuple(blk_shape, blk_coord, residue_coord, tPcP, thread_idx);

    Tensor mPartialSum = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat const*>(      // (N_CN, N_PR)
      params.partial_sum)), shape_partial_result, params.stride_partial_result);
    Tensor mPartialSumOfSquare = make_tensor(make_gmem_ptr(reinterpret_cast<                            // (N_CN, N_PR)
      FragmentBatchNormStat const*>(params.partial_sum_of_square)), shape_partial_result, params.stride_partial_result);
    Tensor mSum = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(params.sum)),            // (N_CN)
      shape_result);
    Tensor mSumOfSquare = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(                 // (N_CN)
      params.sum_of_square)), shape_result);

    FragmentBatchNormStat sum;
    FragmentBatchNormStat sum_of_square;

    // Step1: sum reduction from partial results
    if constexpr (!IsBatchNormStatFinal) {
      sum = reduce(mPartialSum, args_tuple, params, smem_buf);
      // make sure smem buf has been loaded successfully before reusing the same smem buf
      __syncthreads();
      sum_of_square = reduce(mPartialSumOfSquare, args_tuple, params, smem_buf);
    }

    bool do_output = (threadIdx.y == 0 && channel_idx < params.num_channels);

    // Step2: calculation among vectors
    if constexpr (ConvolutionOperator == conv::Operator::kFprop) {
      Tensor mMean = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(params.mean)),        // (N_CN)
        shape_result);
      Tensor mInvStddev = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(                 // (N_CN)
        params.inv_stddev)), shape_result);
      inverse_square_root<FragmentBatchNormStat> inverse_square_root;

      if (do_output) {
        if constexpr (IsBatchNormStatFinal) {
          if (params.sum != nullptr) {
            sum = mSum(channel_idx);
          }
          if (params.sum_of_square != nullptr) {
            sum_of_square = mSumOfSquare(channel_idx);
          }
        } else {
          if (params.sum != nullptr) {
            mSum(channel_idx) = sum;
          }
          if (params.sum_of_square != nullptr) {
             mSumOfSquare(channel_idx) = sum_of_square;
          }
        }
        FragmentBatchNormStat mean = sum * params.inv_count;
        if (params.mean != nullptr) {
          mMean(channel_idx) = mean;
        }
        if (params.inv_stddev != nullptr) {
          auto var = sum_of_square * params.inv_count - mean * mean;
          mInvStddev(channel_idx) = inverse_square_root(var + params.epsilon);
        }
      }
    }

    if constexpr (ConvolutionOperator == conv::Operator::kDgrad) {
      Tensor mMean = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormApply const*>(               // (N_CN)
        params.fprop_mean)), shape_result);
      Tensor mInvStddev = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormApply const*>(          // (N_CN)
        params.fprop_inv_stddev)), shape_result);
      Tensor mAlpha = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormApply const*>(              // (N_CN)
        params.fprop_alpha)), shape_result);
      Tensor mDbnaEqDyScale = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(             // (N_CN)
        params.dbna_eq_dy_scale)), shape_result);
      Tensor mDbnaEqXScale = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(              // (N_CN)
        params.dbna_eq_x_scale)), shape_result);
      Tensor mDbnaEqBiasScale = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(           // (N_CN)
        params.dbna_eq_bias_scale)), shape_result);
      ConvertApply convert_apply{}; // FragmentBatchNormApply -> FragmentBatchNormStat
      if (do_output) {
        if constexpr (IsBatchNormStatFinal) {
          if (params.sum != nullptr) {
            sum = mSum(channel_idx);
          }
          if (params.sum_of_square != nullptr) {
            sum_of_square = mSumOfSquare(channel_idx);
          }
        }
        FragmentBatchNormStat mean = convert_apply(mMean(channel_idx));
        FragmentBatchNormStat alpha = convert_apply(mAlpha(channel_idx));
        FragmentBatchNormStat inv_stddev = convert_apply(mInvStddev(channel_idx));
        sum_of_square = (sum_of_square - mean * sum) * inv_stddev;
        FragmentBatchNormStat dbna_eq_dy_scale = alpha * inv_stddev;
        FragmentBatchNormStat dbna_eq_x_scale = -1.f * dbna_eq_dy_scale * params.inv_count
          * sum_of_square * inv_stddev;
        FragmentBatchNormStat dbna_eq_bias_scale = dbna_eq_dy_scale * params.inv_count
          * (mean * sum_of_square * inv_stddev - sum);
        if constexpr (!IsBatchNormStatFinal) {
          if (params.sum != nullptr) {
            mSum(channel_idx) = sum;
          }
        }
        if (params.sum_of_square != nullptr) {
          mSumOfSquare(channel_idx) = sum_of_square;
        }
        if (params.dbna_eq_dy_scale != nullptr) {
          mDbnaEqDyScale(channel_idx) = dbna_eq_dy_scale;
        }
        if (params.dbna_eq_x_scale != nullptr) {
          mDbnaEqXScale(channel_idx) = dbna_eq_x_scale;
        }
        if (params.dbna_eq_bias_scale != nullptr) {
          mDbnaEqBiasScale(channel_idx) = dbna_eq_bias_scale;
        }
      }
      if constexpr (IsDualBatchNormSupported) {
        Tensor mSecondPartialSumOfsquare = make_tensor(make_gmem_ptr(reinterpret_cast<                  // (N_CN, N_PR)
          FragmentBatchNormStat const*>(params.second_partial_sum_of_square)), shape_partial_result,
          params.stride_partial_result);
        FragmentBatchNormStat second_sum_of_square;

        Tensor mSecondSum = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(               // (N_CN)
          params.second_sum)), shape_result);
        Tensor mSecondSumOfSquare = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(       // (N_CN)
          params.second_sum_of_square)), shape_result);
        Tensor mSecondMean = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormApply const*>(       // (N_CN)
          params.second_fprop_mean)), shape_result);
        Tensor mSecondInvStddev = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormApply const*>(  // (N_CN)
          params.second_fprop_inv_stddev)), shape_result);
        Tensor mSecondAlpha = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormApply const*>(      // (N_CN)
          params.second_fprop_alpha)), shape_result);
        Tensor mSecondDbnaEqDyScale = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(     // (N_CN)
          params.second_dbna_eq_dy_scale)), shape_result);
        Tensor mSecondDbnaEqXScale = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(      // (N_CN)
          params.second_dbna_eq_x_scale)), shape_result);
        Tensor mSecondDbnaEqBiasScale = make_tensor(make_gmem_ptr(reinterpret_cast<FragmentBatchNormStat*>(   // (N_CN)
          params.second_dbna_eq_bias_scale)), shape_result);
        if constexpr (!IsBatchNormStatFinal) {
          // make sure smem buf has been loaded successfully before reusing the same smem buf
          __syncthreads();
          second_sum_of_square = reduce(mSecondPartialSumOfsquare, args_tuple, params, smem_buf);
        }
        if (do_output) {
          if constexpr (IsBatchNormStatFinal) {
            second_sum_of_square = mSecondSumOfSquare(channel_idx);
          }
          FragmentBatchNormStat second_sum = sum;
          FragmentBatchNormStat second_mean = convert_apply(mSecondMean(channel_idx));
          FragmentBatchNormStat second_alpha = convert_apply(mSecondAlpha(channel_idx));
          FragmentBatchNormStat second_inv_stddev = convert_apply(mSecondInvStddev(channel_idx));
          second_sum_of_square = (second_sum_of_square - second_mean * second_sum) * second_inv_stddev;
          FragmentBatchNormStat second_dbna_eq_dy_scale = second_alpha * second_inv_stddev;
          FragmentBatchNormStat second_dbna_eq_x_scale = -1.f * second_dbna_eq_dy_scale * params.inv_count
            * second_sum_of_square * second_inv_stddev;
          FragmentBatchNormStat second_dbna_eq_bias_scale = second_dbna_eq_dy_scale * params.inv_count
            * (second_mean * second_sum_of_square * second_inv_stddev - second_sum);
          if (params.second_sum != nullptr) {
            mSecondSum(channel_idx) = second_sum;
          }
          if (params.second_sum_of_square != nullptr) {
            mSecondSumOfSquare(channel_idx) = second_sum_of_square;
          }
          if (params.second_dbna_eq_dy_scale != nullptr) {
            mSecondDbnaEqDyScale(channel_idx) = second_dbna_eq_dy_scale;
          }
          if (params.second_dbna_eq_x_scale != nullptr) {
            mSecondDbnaEqXScale(channel_idx) = second_dbna_eq_x_scale;
          }
          if (params.second_dbna_eq_bias_scale != nullptr) {
            mSecondDbnaEqBiasScale(channel_idx) = second_dbna_eq_bias_scale;
          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace transform
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////////
