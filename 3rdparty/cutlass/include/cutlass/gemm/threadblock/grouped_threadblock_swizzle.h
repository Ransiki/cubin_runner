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
    \brief Implements the "grouped" threadblock swizzling functor. This functor leverages a
    grouped "problem visitor" internally to determine the mapping of tiles to CTAs.
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/blas3.h"
#include "cutlass/gemm/kernel/grouped_problem_visitor_subgrouped.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/rank_2k_grouped_problem_visitor.h"
#include "cutlass/gemm/kernel/rank_2k_transpose_operands.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {
namespace detail {

// Helper for mapping exposing the data types needed by a higher-level
// grouped kernel when the underlying kernel is a GEMM.
template <
  typename BaseKernel,
  bool Transposed
>
struct GemmTypeMapper {
  // Optional transpose
  using MapArguments = kernel::detail::MapArguments<
    typename BaseKernel::ElementA,
    typename BaseKernel::LayoutA,
    BaseKernel::kTransformA,
    BaseKernel::kAlignmentA,
    typename BaseKernel::ElementB,
    typename BaseKernel::LayoutB,
    BaseKernel::kTransformB,
    BaseKernel::kAlignmentB,
    typename BaseKernel::LayoutC,
    Transposed
  >;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notation.
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static int const kAlignmentA = MapArguments::kAlignmentA;

  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;
  static int const kAlignmentB = MapArguments::kAlignmentB;

  using ElementC = typename BaseKernel::ElementC;
  using LayoutC = typename MapArguments::LayoutC;
  static int const kAlignmentC = BaseKernel::kAlignmentC;

  using ElementAccumulator = typename BaseKernel::Mma::Policy::Operator::ElementC;

  using Operator = typename BaseKernel::Operator;
  using WarpMmaOperator = typename BaseKernel::Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;
  using ThreadblockShape = typename BaseKernel::Mma::Shape;
  using WarpShape = typename BaseKernel::WarpShape;
  using InstructionShape = typename BaseKernel::InstructionShape;
  static int const kStages = BaseKernel::Mma::kStages;

  using Mma = typename BaseKernel::Mma;
};

struct GroupedWithSubGroupsThreadblockSwizzleBase {};

/// Helper for determining if a swizzling function is specialized for grouped operation
template <typename ThreadblockSwizzle>
struct IsGroupedWithSubGroupsSwizzle {
  static bool const value = platform::is_base_of<GroupedWithSubGroupsThreadblockSwizzleBase, ThreadblockSwizzle>::value;
};

/// Utility struct for returning the type of the problem visitor used by the swizzling function,
/// if it is a grouped swizzling function, or a default visitor. This is used only for defining
/// the parameters of the problem visitor used in GroupedParams.
template <
  typename Mma_,
  typename ThreadblockSwizzle_,
  typename Enable = void
>
struct ProblemVisitorOrDefault;

/// Return a generic problem visitor for GEMM problems
template <
  typename Mma_,
  typename ThreadblockSwizzle_
>
struct ProblemVisitorOrDefault<
  Mma_,
  ThreadblockSwizzle_,
  typename platform::enable_if< !IsGroupedWithSubGroupsSwizzle<ThreadblockSwizzle_>::value >::type> {
  using value = cutlass::gemm::kernel::GemmGroupedProblemVisitorSubGrouped<
    typename Mma_::Shape, platform::is_same<typename Mma_::LayoutC, cutlass::layout::ColumnMajor>::value>;
};

/// Return the problem visitor specified by the swizzling function
template <
  typename Mma_,
  typename ThreadblockSwizzle_
>
struct ProblemVisitorOrDefault<
  Mma_,
  ThreadblockSwizzle_,
  typename platform::enable_if<IsGroupedWithSubGroupsSwizzle<ThreadblockSwizzle_>::value >::type> {
  using value = typename ThreadblockSwizzle_::ProblemVisitor;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Swizzling function for grouped kernels
template <
  typename ProblemVisitor_,
  bool Transposed,
  template <typename BaseKernel, bool T> class Mapper_
>
struct GroupedWithSubGroupsThreadblockSwizzle : detail::GroupedWithSubGroupsThreadblockSwizzleBase {

  using ProblemVisitor = ProblemVisitor_;
  static bool const kTransposed = Transposed;
  ProblemVisitor problem_visitor;

  template <typename BaseKernel>
  using Mapper = Mapper_<BaseKernel, Transposed>;

  CUTLASS_HOST_DEVICE
  GroupedWithSubGroupsThreadblockSwizzle(
    typename ProblemVisitor::Params& params,
    typename ProblemVisitor::SharedStorage& shared_storage,
    int block_idx) : problem_visitor(params, shared_storage, block_idx) { }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int /*log_tile*/) const {
    GemmCoord problem_size = problem_visitor.problem_size();
    int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());
    GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

    return problem_visitor.threadblock_offset(threadblock_idx, grid_shape);
  }

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static GemmCoord get_tiled_shape(GemmCoord problem_size, GemmCoord /*tile_size*/, int /*batch_count*/) {
    return ProblemVisitor::grid_shape(problem_size);
  }

  /// Dummy method to satisfy API for threadblock swizzling functions
  CUTLASS_HOST_DEVICE
  static int get_log_tile(GemmCoord /*tiled_shape*/) {
    return 0;
  }
};

// Specialization for grouped GEMM problems
template <
  typename ThreadblockShape,
  typename LayoutC,
  typename ProblemSizeParams = typename cutlass::gemm::kernel::detail::ProblemSizeParamsCombined>
using GemmGroupedThreadblockSwizzle = GroupedWithSubGroupsThreadblockSwizzle<
                                          cutlass::gemm::kernel::GemmGroupedProblemVisitorSubGrouped<
                                            ThreadblockShape,
                                            platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value,
                                            ProblemSizeParams
                                          >,
                                          platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value,
                                          detail::GemmTypeMapper
                                        >;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass