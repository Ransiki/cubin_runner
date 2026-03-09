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

/*! \file
    \brief block-scaled GEMV with Epilogue Visitor Tree
*/

#pragma once

#include "cutlass/arch/cache_operation.h"  /// cutlass::arch::CacheOperation
#include "cutlass/arch/memory.h"           // cutlass::arch::global_load
#include "cutlass/complex.h"               // cutlass::ComplexTransform:
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"             // cutlass::fast_max
#include "cutlass/layout/matrix.h"         // cutlass::layout::RowMajor
#include "cutlass/matrix_coord.h"          // cutlass::MatrixCoord
#include "cutlass/numeric_conversion.h"    // cutlass::FloatRoundStyle, cutlass::NumericConverter
#include "cutlass/numeric_types.h"         // cutlass::float_e4m3_t
#include "cutlass/platform/platform.h"     // cutlass::is_same_v
#include "cutlass/tensor_ref.h"            // cutlass::TensorRef
#include "cutlass/semaphore.h"             // split-k

#include <cute/tensor.hpp>
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitor_load_gemv.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename ElementAccumulator_,
  typename Callbacks_,
  int kElementsPerAccess_ = 1,
  int kThreadCount_ = 0,
  int kThreadsPerRow_ = 0,
  typename ElementSFA_ = cutlass::float_e4m3_t,
  typename ElementSFB_ = cutlass::float_e4m3_t,
  int kSFVecSize_ = 16
>
struct GemvBlockScaledwithVisitor;


// GEMV for row-major A matrix
template <typename ElementA_,
          typename ElementB_,
          typename ElementAccumulator_,
          typename Epilogue_,
          int kElementsPerAccess_,
          int kThreadCount_,
          int kThreadsPerRow_,
          typename ElementSFA_,
          typename ElementSFB_,
          int kSFVecSize_>
struct GemvBlockScaledwithVisitor<
            ElementA_,
            cutlass::layout::RowMajor,
            ElementB_,
            ElementAccumulator_,
            Epilogue_,
            kElementsPerAccess_,
            kThreadCount_,
            kThreadsPerRow_,
            ElementSFA_,
            ElementSFB_,
            kSFVecSize_>
{
public:
  using ElementA = ElementA_;
  using ElementSFA = ElementSFA_;
  using LayoutA = cutlass::layout::RowMajor;
  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
  static_assert(cutlass::sizeof_bits<ElementSFA>::value == 8, "ElementSFA should be FP8 type");

  using ElementB = ElementB_;
  using ElementSFB = ElementSFB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  static_assert(cutlass::sizeof_bits<ElementSFB>::value == 8, "ElementSFB should be FP8 type");
  
  // Placeholder to adapt to the device API
  using ElementC = ElementA; 
  using EpilogueOutputOp = Epilogue_;

  using ElementAccumulator = ElementAccumulator_;

  static constexpr cutlass::ComplexTransform kTransformA = cutlass::ComplexTransform::kNone;
  static constexpr cutlass::ComplexTransform kTransformB = cutlass::ComplexTransform::kNone;

  static constexpr FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest;

  // number of return elements in a global access
  static constexpr int kElementsPerAccess = kElementsPerAccess_;
  static constexpr int kSFVecSize = kSFVecSize_;
  static constexpr int kSFPerAccess = cutlass::const_max(1, kElementsPerAccess / kSFVecSize);

  static_assert(kSFVecSize == 16, "Only SFVecSize = 16 is supported");
  // Hardcode some check for easier debug
  static_assert(kElementsPerAccess == 32, "for fp4 kernel, 32 elt per access");
  static_assert(kSFPerAccess == 2, "fpr fp4 kernel, 2 sf read per thread");

  static constexpr bool kDequantizeA = cutlass::sizeof_bits<ElementA>::value == 4;
  static constexpr bool kDequantizeB = cutlass::sizeof_bits<ElementB>::value == 4;
  static constexpr int kPackedElementsA = cutlass::sizeof_bits<ElementA>::value == 4 ? 2 : 1;
  static constexpr int kPackedElementsB = cutlass::sizeof_bits<ElementB>::value == 4 ? 2 : 1;
  static constexpr int kPackedElements = cutlass::const_max(kPackedElementsA, kPackedElementsB);

  using FragmentA = cutlass::Array<ElementA, kElementsPerAccess>;
  using FragmentB = cutlass::Array<ElementB, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentSFA = cutlass::Array<ElementSFA, kSFPerAccess>;
  using FragmentSFB = cutlass::Array<ElementSFB, kSFPerAccess>;
  using FragmentPackedA = cutlass::Array<ElementA, kPackedElements>;
  using FragmentPackedB = cutlass::Array<ElementB, kPackedElements>;

  // // thread block shape (kThreadsPerRow, kThreadCount / kThreadsPerRow, 1)
  static constexpr int kThreadCount = (kThreadCount_ <= 0) ? 128 : kThreadCount_;
  static constexpr int kThreadsPerRow = (kThreadsPerRow_ <= 0) ? 
                                        cutlass::const_min(static_cast<int>(kThreadCount / cutlass::bits_to_bytes(kElementsPerAccess * cutlass::sizeof_bits<ElementA>::value)), 16) :
                                        kThreadsPerRow_;
  static constexpr int kThreadsPerCol = kThreadCount / kThreadsPerRow;

  using AccReorder = cutlass::epilogue::threadblock::VisitorAccReorderGemv<gemm::GemmShape<16,8>, ElementAccumulator>;
  using Epilogue = Epilogue_;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments
  {
    MatrixCoord problem_size;
    int32_t batch_count{0};
    typename AccReorder::Params acc_reorder;
    typename Epilogue::Params epilogue;

    TensorRefA ref_A;

    ElementB const *ptr_B{nullptr};

    ElementSFA const *ptr_SFA{nullptr};
    ElementSFB const *ptr_SFB{nullptr};

    int64_t stride_A{0};
    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};

    int64_t batch_stride_SFA{0};
    int64_t batch_stride_SFB{0};
  };

  using Params = Arguments;

  /// Shared memory storage structure
  struct SharedStorage
  {
    using EpilogueStorage = typename Epilogue::SharedStorage;
    EpilogueStorage epilogue;
    typename AccReorder::SharedStorage acc_reorder;
  };

public:
  //
  // Methods
  //

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::MatrixCoord const &problem_size)
  {
    if (problem_size.column() % kElementsPerAccess != 0) {
      return Status::kErrorMisalignedOperand;
    }
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args)
  {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMV
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage)
  {
    // XXX: use FDL to overlap LDG.128 w/ pervious kernel epilogue (gemm)

    Epilogue epilogue(params.epilogue, shared_storage.epilogue);
    AccReorder acc_reorder(params.acc_reorder, shared_storage.acc_reorder);

    NumericArrayConverter<ElementAccumulator, ElementA, kElementsPerAccess, Round> srcA_converter;
    NumericArrayConverter<ElementAccumulator, ElementB, kElementsPerAccess, Round> srcB_converter;
    NumericConverter<ElementAccumulator, ElementSFA, Round> SFA_converter;
    NumericConverter<ElementAccumulator, ElementSFB, Round> SFB_converter;

    const int32_t gemm_m = params.problem_size.row();
    [[maybe_unused]] static constexpr int32_t gemm_n = 1;
    const int32_t gemm_k = params.problem_size.column();
    const int32_t gemm_batch = params.batch_count;

    // Construct the epilogue visitor callbacks
    int thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
    auto callbacks = epilogue.get_callbacks(make_shape(gemm_m, Int<gemm_n>{}, gemm_batch), thread_idx);
    auto acc_reorder_callbacks = acc_reorder.get_callbacks(make_shape(gemm_m, Int<gemm_n>{}, gemm_batch), thread_idx);

    // Loop over batch indices
    for (int batch_idx = blockIdx.z; batch_idx < gemm_batch; batch_idx += gridDim.z) {
      int idx_col_k = threadIdx.x;
      int idx_row_m = blockIdx.x * blockDim.y + threadIdx.y;

      // Filter the active tensor
      if (acc_reorder_callbacks.is_active())
        callbacks.begin_epilogue(batch_idx);

      if (idx_row_m < gemm_m) {
        // problem_size (row = m, column = k)
        // matrix A (batch, m, k)
        // vector B (batch, k, 1)

        // move in the batch dimension
        ElementA const *ptr_A = params.ref_A.data() + batch_idx * params.batch_stride_A / kPackedElementsA;
        ElementB const *ptr_B = params.ptr_B + batch_idx * params.batch_stride_B / kPackedElementsB;

        // move in the k dimension
        ptr_A += idx_col_k * kElementsPerAccess / kPackedElementsA;
        ptr_B += idx_col_k * kElementsPerAccess / kPackedElementsB;

        // move in the m dimension
        ptr_A += idx_row_m * params.stride_A / kPackedElementsA;

        ElementSFA const *ptr_SF_A{nullptr};
        ElementSFB const *ptr_SF_B{nullptr};
        int global_k{0};

        if constexpr (kDequantizeA || kDequantizeB) {
          int SF_blocks_by_M = (gemm_m + 127) >> 7;
          int SF_blocks_by_K = (gemm_k / kSFVecSize + 3) >> 2;

          // move in the batch dimension
          ptr_SF_A = params.ptr_SFA + batch_idx * SF_blocks_by_M * SF_blocks_by_K * 512;
          ptr_SF_B = params.ptr_SFB + batch_idx * SF_blocks_by_K * 512;

          // move in the m dimension
          ptr_SF_A +=
              (((idx_row_m >> 7) * SF_blocks_by_K) << 9) + ((idx_row_m & 0x1f) << 4) + ((idx_row_m & 0x7f) >> 5 << 2);

          global_k = idx_col_k * kElementsPerAccess;
        }

        ElementAccumulator accum = ElementAccumulator(0);

        FragmentA fragA;
        FragmentB fragB;
        FragmentSFA fragSFA;
        FragmentSFB fragSFB;

        int unroll_col_k = 0;

        // rows of the rolling tile
        // tileA_k will access 128 Bytes
        static constexpr int tileA_k = kThreadsPerRow * kElementsPerAccess;

        // XXX: add double buffer for data loading. overlap math computation w/ LDG.
        // XXX: add force no unroll here to avoid instruction miss. Need perf study to verify this
        for (; unroll_col_k < gemm_k / tileA_k * tileA_k; unroll_col_k += tileA_k) {
          if constexpr (kDequantizeA || kDequantizeB) {
            int SF_idx = global_k / kSFVecSize;
            int SF_offset_by_k = ((SF_idx >> 2) << 9) + (SF_idx & 0x3);

            if constexpr (kDequantizeA) {
              arch::global_load<FragmentSFA, sizeof(FragmentSFA), arch::CacheOperation::LastUse>(
                  fragSFA,
                  (ptr_SF_A + SF_offset_by_k),
                  true);
            }

            if constexpr (kDequantizeB) {
              arch::global_load<FragmentSFB, sizeof(FragmentSFB), arch::CacheOperation::Always>(
                  fragSFB,
                  (ptr_SF_B + SF_offset_by_k),
                  true);
            }

            global_k += tileA_k;
          }

          // L2 prefetch next LDG.128 chunk is enabled by setting `CUTLASS_CUDA_INTERNAL_L2_PREFETCH_ENABLED` CMake flag

          // fetch from matrix A
          cutlass::arch::global_load<FragmentA, sizeof(FragmentA), arch::CacheOperation::LastUse>(
              fragA,
              (ptr_A + unroll_col_k / kPackedElementsA),
              true);

          // fetch from vector B
          cutlass::arch::global_load<FragmentB, sizeof(FragmentB), arch::CacheOperation::Always>(
              fragB,
              (ptr_B + unroll_col_k / kPackedElementsB),
              true);

          FragmentCompute fragB_Compute = srcB_converter(fragB);
          FragmentCompute fragA_Compute = srcA_converter(fragA);

          // Math
          // Blockscaled GEMV
          if constexpr (kDequantizeA || kDequantizeB) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kSFPerAccess; i++) {
              ElementAccumulator accum_SF_block = 0.f;

              int local_k_offset = i * kSFVecSize;
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < min(kElementsPerAccess, kSFVecSize); e++) {
                accum_SF_block += fragA_Compute.at(e + local_k_offset) * fragB_Compute.at(e + local_k_offset);
              }

              if constexpr (kDequantizeA) {
                accum_SF_block *= SFA_converter(fragSFA.at(i));
              }

              if constexpr (kDequantizeB) {
                accum_SF_block *= SFB_converter(fragSFB.at(i));
              }

              accum += accum_SF_block;
            }
          } 
          // Regular GEMV
          else {
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < kElementsPerAccess; e++) {
              accum += fragA_Compute.at(e) * fragB_Compute.at(e);
            }
          }
        } // end of one cta_tile computation, move along k-dim for next cta_tile

        // calculate the rest of K elements
        // each thread fetch 1 element each time
        for (int k = unroll_col_k + idx_col_k * kPackedElementsA; k < gemm_k;
             k += kThreadsPerRow * kPackedElementsA) {
          // blockscaled GEMV
          if constexpr (kDequantizeA || kDequantizeB) {
            int SF_idx = k / kSFVecSize;
            int SF_offset_by_k = ((SF_idx >> 2) << 9) + (SF_idx & 0x3);

            ElementSFA sfa = *(ptr_SF_A + SF_offset_by_k);
            ElementSFB sfb = *(ptr_SF_B + SF_offset_by_k);

            FragmentPackedA fragA;
            FragmentPackedB fragB;

            // fetch from matrix A
            arch::global_load<FragmentPackedA, sizeof(FragmentPackedA), arch::CacheOperation::Always>(
                fragA,
                ptr_A - (idx_col_k * kElementsPerAccess - k) / kPackedElementsA,
                true);

            // fetch from vector B
            arch::global_load<FragmentPackedB, sizeof(FragmentPackedB), arch::CacheOperation::Always>(
                fragB,
                ptr_B - (idx_col_k * kElementsPerAccess - k) / kPackedElementsB,
                true);

            ElementAccumulator accum_SF_packed = 0.f;

            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < kPackedElements; e++) {
              accum_SF_packed += ElementAccumulator(fragA.at(e)) * ElementAccumulator(fragB.at(e));
            }

            if constexpr (kDequantizeA) {
              accum_SF_packed *= SFA_converter(sfa);
            }

            if constexpr (kDequantizeB) {
              accum_SF_packed *= SFB_converter(sfb);
            }

            accum += accum_SF_packed;
          } 
          // regular GEMV
          else {
            ElementA a = *(ptr_A - idx_col_k * kElementsPerAccess + k);
            ElementB b = *(ptr_B - idx_col_k * kElementsPerAccess + k);

            accum += ElementAccumulator(a) * ElementAccumulator(b);
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mask = (kThreadsPerRow >> 1); mask > 0; mask >>= 1) {
          accum += __shfl_xor_sync(0xFFFFFFFF, accum, mask, 32);
        }

        auto frag_acc = static_cast<ElementAccumulator>(accum);

        // Rearrange the accumulator
        frag_acc = acc_reorder_callbacks.visit(batch_idx, blockIdx.x, frag_acc);
        
        // Applying blockscaled epilogue
        if (acc_reorder_callbacks.is_active())
          callbacks.visit(batch_idx, blockIdx.x, frag_acc);
      }
      if (acc_reorder_callbacks.is_active())
        callbacks.end_epilogue(batch_idx);
    }
  } //end of operator()
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
