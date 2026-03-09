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
    \brief High-level interface for running a grouped version of a CUTLASS kernel
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace detail {

template <
  typename BaseKernel_   ///! Kernel-scoped matrix multiply-accumulate
>
struct GroupedKernelBase {
  // Many of these types need to be exported to work properly with device::BaseGrouped
  using BaseKernel = BaseKernel_;
  using Epilogue = typename BaseKernel::Epilogue;
  using EpilogueOutputOp = typename BaseKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;
  using ProblemVisitor = typename ThreadblockSwizzle::ProblemVisitor;
  static_assert(!ProblemVisitor::kRequiresPrecomputation,
    "Only scheduling mode kDeviceOnly is available for the grouped interface");

  using Mapper = typename ThreadblockSwizzle::template Mapper<BaseKernel>;

  using ElementA = typename Mapper::ElementA;
  using LayoutA = typename Mapper::LayoutA;
  static ComplexTransform const kTransformA = Mapper::kTransformA;
  static int const kAlignmentA = Mapper::kAlignmentA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;

  using ElementB = typename Mapper::ElementB;
  using LayoutB = typename Mapper::LayoutB;
  static ComplexTransform const kTransformB = Mapper::kTransformB;
  static int const kAlignmentB = Mapper::kAlignmentB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;

  using ElementC = typename Mapper::ElementC;
  using LayoutC = typename Mapper::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  static int const kAlignmentC = Mapper::kAlignmentC;

  using ElementAccumulator = typename Mapper::ElementAccumulator;

  using Operator = typename Mapper::Operator;
  using WarpMmaOperator = typename Mapper::WarpMmaOperator;

  using ArchMmaOperator = typename Mapper::ArchMmaOperator;
  using MathOperator = typename Mapper::MathOperator;
  using OperatorClass = typename Mapper::OperatorClass;
  using ArchTag = typename Mapper::ArchTag;
  using ThreadblockShape = typename Mapper::ThreadblockShape;
  using WarpShape = typename Mapper::WarpShape;
  using InstructionShape = typename Mapper::InstructionShape;
  static int const kStages = Mapper::kStages;

  using Mma = typename Mapper::Mma;

  using Arguments = typename BaseKernel::GroupedArguments;

  static int const kThreadCount = BaseKernel::kThreadCount;
};

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// High-level interface for running a grouped version of a CUTLASS kernel
template <
  typename BaseKernel_   ///! Kernel-scoped matrix multiply-accumulate
>
struct GroupedKernel : public detail::GroupedKernelBase<BaseKernel_> {
public:
  using Base = detail::GroupedKernelBase<BaseKernel_>;
  using Params = typename Base::BaseKernel::GroupedParams;
  using Arguments = typename Base::BaseKernel::GroupedArguments;
  using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;

  /// Shared memory storage structure
  struct SharedStorage {
    typename Base::BaseKernel::SharedStorage kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename Base::ProblemVisitor::SharedStorage problem_visitor;
  };

  CUTLASS_DEVICE
  GroupedKernel() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  /// Executes a kernel in a loop
  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {

    ThreadblockSwizzle swizzle(params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    if (ThreadblockSwizzle::kTransposed) {
      params.transpose();
    }

    typename Base::BaseKernel mma;

    // Outer 'persistent' loop to iterate over tiles
    while (swizzle.problem_visitor.next_tile()) {

      typename Base::BaseKernel::Params mma_params = params.to_single_params(swizzle.problem_visitor);
      mma.run_with_swizzle(mma_params, shared_storage.kernel, swizzle);

      // Next tile
      swizzle.problem_visitor.advance(gridDim.x);
    }
  }
};

/// High-level interface for dispatching to multiple different kernels depending on the layouts of operands
template <
  typename BaseKernelNN_,
  typename BaseKernelNT_,
  typename BaseKernelTN_,
  typename BaseKernelTT_
>
struct GroupedKernelMulti : public detail::GroupedKernelBase<BaseKernelNN_> {
public:
  using Base = detail::GroupedKernelBase<BaseKernelNN_>;
  using BaseKernelNN = BaseKernelNN_;
  using BaseKernelNT = BaseKernelNT_;
  using BaseKernelTN = BaseKernelTN_;
  using BaseKernelTT = BaseKernelTT_;
  using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;
  using ProblemVisitor = typename ThreadblockSwizzle::ProblemVisitor;
  using BaseParams = typename Base::BaseKernel::GroupedParams;

public:
  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename BaseKernelNN::SharedStorage nn;
      typename BaseKernelNT::SharedStorage nt;
      typename BaseKernelTN::SharedStorage tn;
      typename BaseKernelTT::SharedStorage tt;
    } kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename Base::ProblemVisitor::SharedStorage problem_visitor;
  };

  struct Arguments {
    typename Base::Arguments grouped_args;
    GemmUniversalMode mode;
    int* transpose_A;
    int* transpose_B;

    // Arguments needed to satisfy device::BaseGrouped
    int& threadblock_count;
    int& group_count;
    int* group_sizes;
    bool transposed;

    CUTLASS_HOST_DEVICE
    Arguments(
      typename Base::Arguments grouped_args_,
      int* transpose_A_,
      int* transpose_B_
    ) :
      grouped_args(grouped_args_),
      mode(grouped_args_.mode),
      transpose_A(transpose_A_),
      transpose_B(transpose_B_),
      threadblock_count(grouped_args.threadblock_count),
      group_count(grouped_args.group_count),
      group_sizes(grouped_args_.group_sizes),
      transposed(false) {}

    Arguments transposed_problem() const {
      Arguments args(*this);

      args.grouped_args = args.grouped_args.transposed_problem();

      // Flip the transposed bit
      args.transposed = !args.transposed;

      return args;
    }
  };

  struct Params {
    BaseParams grouped_params{};
    int* transpose_A{nullptr};
    int* transpose_B{nullptr};
    typename ProblemVisitor::Params problem_visitor{};
    int threadblock_count{0};
    bool transposed{false};

    Params() {}

    Params(Arguments const& args,
      void *workspace = nullptr,
      int tile_count = 0) :
      grouped_params(args.grouped_args, workspace, tile_count),
      transpose_A(args.transpose_A),
      transpose_B(args.transpose_B),
      problem_visitor(grouped_params.problem_visitor),
      threadblock_count(args.threadblock_count),
      transposed(args.transposed) {}

    Params(Arguments const& args,
      int device_sms,
      int sm_occupancy):
      grouped_params(args.grouped_args, nullptr, 0),
      transpose_A(args.transpose_A),
      transpose_B(args.transpose_B),
      problem_visitor(grouped_params.problem_visitor),
      threadblock_count(args.threadblock_count),
      transposed(args.transposed) {}

    Status init_workspace(void* workspace, cudaStream_t stream = nullptr) {
      grouped_params = BaseParams(grouped_params, workspace);
      problem_visitor = grouped_params.problem_visitor;
      return Status::kSuccess;
    }

    size_t get_workspace_size() const {
      // Currently, only kernels with no workspace are supported
      return 0;
    }

    dim3 get_grid_dims() const {
      return {static_cast<uint32_t>(threadblock_count), 1, 1};
    }
  };

private:
  template <typename Kernel>
  CUTLASS_DEVICE
  void run(
    ThreadblockSwizzle& swizzle,
    typename Kernel::SharedStorage& shared_storage,
    typename Kernel::GroupedParams* grouped_params
  ) const {
    typename Kernel::Params mma_params = grouped_params->to_single_params(swizzle.problem_visitor);
    Kernel mma;
    mma.run_with_swizzle(mma_params, shared_storage, swizzle);
  }

public:

  CUTLASS_DEVICE
  GroupedKernelMulti() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  /// Executes a kernel in a loop
  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {

    BaseParams& grouped_params = params.grouped_params;
    ThreadblockSwizzle swizzle(grouped_params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    if (ThreadblockSwizzle::kTransposed) {
      grouped_params.transpose_problems();
    }

    // In order to call the `to_single_params()` arguments of each of the different row/column
    // versions, we need to reinterpret the baseline grouped parameters into those for each row/column type.
    // This could be avoided by removing the definitions of `Arguments` and `Params` from within the
    // kernel::-scoped operation and removing any layout-specific arguments.
    typename BaseKernelNN::GroupedParams* nn_params = reinterpret_cast<typename BaseKernelNN::GroupedParams*>(&params);
    typename BaseKernelNT::GroupedParams* nt_params = reinterpret_cast<typename BaseKernelNT::GroupedParams*>(&params);
    typename BaseKernelTN::GroupedParams* tn_params = reinterpret_cast<typename BaseKernelTN::GroupedParams*>(&params);
    typename BaseKernelTT::GroupedParams* tt_params = reinterpret_cast<typename BaseKernelTT::GroupedParams*>(&params);

    // Outer 'persistent' loop to iterate over tiles
    while (swizzle.problem_visitor.next_tile()) {
      int group_idx = swizzle.problem_visitor.group_index();
      bool tA, tB;
      if (!params.transposed) {
        tA = params.transpose_A[group_idx] == 1;
        tB = params.transpose_B[group_idx] == 1;
      }
      else {
        // See internal transposition done in cutlass::gemm::kernel::detail::MapArguments.
        // We must swap the A and B transpositions and flip each transposition itself.
        tA = params.transpose_B[group_idx] == 0;
        tB = params.transpose_A[group_idx] == 0;
      }

      if (!tA && !tB) {
        run<BaseKernelNN>(swizzle, shared_storage.kernel.nn, nn_params);
      }
      else if (!tA && tB) {
        run<BaseKernelNT>(swizzle, shared_storage.kernel.nt, nt_params);
      }
      else if (tA && !tB) {
        run<BaseKernelTN>(swizzle, shared_storage.kernel.tn, tn_params);
      }
      else {
        run<BaseKernelTT>(swizzle, shared_storage.kernel.tt, tt_params);
      }

      // Next tile
      swizzle.problem_visitor.advance(gridDim.x);
    }
  }

  // Factory invocation
  CUTLASS_DEVICE
  static void invoke(
    Params &params,
    SharedStorage &shared_storage)
  {
    GroupedKernelMulti op;
    op(params, shared_storage);
  }
};

template <typename BaseKernel>
int blocks_per_device(int available_sm_count=-1) {
  // Determine the number of blocks that would be launched to fill up a single
  // wave on the GPU with each SM having maximum occupancy.
  cudaDeviceProp properties;
  int device_idx;
  cudaError_t result = cudaGetDevice(&device_idx);
  if (result != cudaSuccess) {
    // Call cudaGetLastError() to clear the error bit
    result = cudaGetLastError();
    CUTLASS_TRACE_HOST("  cudaGetDevice() returned error "
        << cudaGetErrorString(result));
    return 0;
  }

  result = cudaGetDeviceProperties(&properties, device_idx);
  if (result != cudaSuccess) {
    // Call cudaGetLastError() to clear the error bit
    result = cudaGetLastError();
    CUTLASS_TRACE_HOST("  cudaGetDeviceProperties() returned error "
        << cudaGetErrorString(result));
    return 0;
  }

  bool override_sm_count = (available_sm_count < 0 || available_sm_count > properties.multiProcessorCount);
  if (override_sm_count) {
    available_sm_count = properties.multiProcessorCount;
  }

  int smem_size = int(sizeof(typename BaseKernel::SharedStorage));

  if (smem_size > (48 << 10)) {
    result = cudaFuncSetAttribute(Kernel<BaseKernel>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size);

    if (result != cudaSuccess) {
      // Call cudaGetLastError() to clear the error bit
      result = cudaGetLastError();
      CUTLASS_TRACE_HOST(
        "  cudaFuncSetAttribute() returned error "
        << cudaGetErrorString(result));
      return -1;
    }
  }

  int max_active_blocks = -1;
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks,
      Kernel<BaseKernel>,
      BaseKernel::kThreadCount,
      smem_size);

  if (result != cudaSuccess) {
    // Call cudaGetLastError() to clear the error bit
    result = cudaGetLastError();
    CUTLASS_TRACE_HOST(
      "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error "
      << cudaGetErrorString(result));
    return -1;
  }

  if (max_active_blocks <= 0) {
    return 0;
  }

  return available_sm_count * max_active_blocks;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////