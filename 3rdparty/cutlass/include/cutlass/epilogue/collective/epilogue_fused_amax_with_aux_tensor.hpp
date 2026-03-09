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

/*! \file
  \brief Functor performing elementwise operations used by epilogues and calculating the absolute max
  output of the output tensor. Meanwhile, it would return Aux tensor and calculate absolute max
  output of the Aux tensor.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {
/////////////////////////////////////////////////////////////////////////////////////////////////

/// 1. Applies an element wise operation to all elements within the fragment
/// and writes output and aux tensor out to destination storage.
///
/// 2. Writes out the absolute max output value of the output tensor and the Aux tensor if the user needs
/// them.
///
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_
>
class EpilogueFusedAmaxWithAuxTensor {
public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;
  
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  using ElementAuxOutput = typename ThreadEpilogueOp::ElementAuxOutput;
  
  static constexpr int kOutputAlignment = ThreadEpilogueOp::kCount;
  static constexpr bool isEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::value;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  struct SharedStorage { };

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementAccumulator* ptr_abs_max_D = nullptr;
    // Aux tensor related
    ElementAuxOutput* ptr_Aux = nullptr;
    StrideD dAux{};
    ElementAccumulator* ptr_abs_max_Aux = nullptr;
    ElementBias* ptr_Bias = nullptr;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    // TODO: ThreadEpilogueOp::to_underlying_arguments(args) here for visitor pattern epilogues // {$nv-release-never}
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueFusedAmaxWithAuxTensor(Params const& params_)
      : params(params_), epilogue_op(params_.thread) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      [[maybe_unused]] char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 4");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto stride_c    = detail::get_epilogue_stride<EpilogueSchedule>(params.dC);
    auto stride_d    = detail::get_epilogue_stride<EpilogueSchedule>(params.dD);
    auto stride_aux  = detail::get_epilogue_stride<EpilogueSchedule>(params.dAux);
    auto stride_bias = detail::get_epilogue_stride<EpilogueSchedule>(Stride<_1, _0, _0>{});

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), stride_c);                   // (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), stride_d);                   // (m,n,l)
    Tensor mAux_mnl = make_tensor(make_gmem_ptr(params.ptr_Aux), make_shape(M,N,L), stride_aux);             // (m,n,l)
    Tensor mBias_mnl = make_tensor(make_gmem_ptr(params.ptr_Bias), make_shape(M,N,L), stride_bias);          // (m,n,l)
    
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gAux_mnl = local_tile(mAux_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});  // (BLK_M,BLK_N,m,n,l)
    Tensor gBias_mnl = local_tile(mBias_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});// (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gAux = gAux_mnl(_,_,m_coord,n_coord,l_coord);                                               // (BLK_M,BLK_N)
    Tensor gBias = gBias_mnl(_,_,m_coord,n_coord,l_coord);                                             // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                                         // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC);                                                         // (VEC,THR_M,THR_N)
    Tensor tCgAux = thr_mma.partition_C(gAux);                                                     // (VEC,THR_M,THR_N)
    Tensor tCgBias = thr_mma.partition_C(gBias);                                                   // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value,
        "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");
    CUTE_STATIC_ASSERT_V(size(tCgAux) == size(tCgD),
        "Aux and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(size(tCgBias) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    auto cD   = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    const bool is_aux_output_needed = params.ptr_Aux != nullptr;

    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          if constexpr (isEpilogueBiasSupported) {
            ElementBias bias = ElementBias(0);
            if (params.ptr_Bias) {
              bias = tCgBias(i);
            }
            auto [out_D, out_Aux] = epilogue_op(accumulators(i), tCgC(i), bias);
            tCgD(i) = out_D;
            // Write out Aux tensor if user needs it.
            if (is_aux_output_needed) {
              tCgAux(i) = out_Aux;
            }
          }
          else {
            auto [out_D, out_Aux] = epilogue_op(accumulators(i), tCgC(i));
            tCgD(i) = out_D;
            // Write out Aux tensor if user needs it.
            if (is_aux_output_needed) {
              tCgAux(i) = out_Aux;
            }
          }
        }
      }
    }

    // source is not needed, avoid load
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          if constexpr (isEpilogueBiasSupported) {   
            ElementBias bias = ElementBias(0);
            if (params.ptr_Bias) {
              bias = tCgBias(i);
            }
            auto [out_D, out_Aux] = epilogue_op(accumulators(i), bias);
            tCgD(i) = out_D;
            // Write out Aux tensor if user needs it.
            if (is_aux_output_needed) {
              tCgAux(i) = out_Aux;
            }
          } 
          else {
            auto [out_D, out_Aux] = epilogue_op(accumulators(i));
            tCgD(i) = out_D;
            // Write out Aux tensor if user needs it.
            if (is_aux_output_needed) {
              tCgAux(i) = out_Aux;
            }
          }
        }
      }
    }

    // Store the amax_Aux output if needed
    if (params.ptr_abs_max_Aux != nullptr) {
      detail::atomic_maximum<ElementAccumulator>{}(
          params.ptr_abs_max_Aux, epilogue_op.get_aux_output_abs_max());
    }

    // Store the amax_d output if needed
    if (params.ptr_abs_max_D != nullptr) {
      detail::atomic_maximum<ElementAccumulator>{}(
          params.ptr_abs_max_D, epilogue_op.get_output_abs_max());
    }
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
