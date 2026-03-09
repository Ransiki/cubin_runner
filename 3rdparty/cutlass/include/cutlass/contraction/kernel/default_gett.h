/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief 
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.
  
      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#pragma once



#include "cutlass/gemm/device/gemm.h"

#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_with_access_size.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/contraction/threadblock/threadblock_swizzle.h"
#include "cutlass/contraction/kernel/gett.h"
#include "cutlass/contraction/threadblock/dynamic_epilogue.h"
#include "cutlass/conv/threadblock/implicit_gemm_pipelined.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/contraction/threadblock/fused_ndim_tile_access_iterator_optimized.h"
#include "cutlass/contraction/threadblock/fused_tensor_ndim_predicated_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contraction {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Elementwise transform on A
  typename TransformA,
  /// Vectorized Loads for A
  int kElementsPerAccessA,
  /// Layout type for A matrix operand
  bool TransA_,
  /// Determines if all modes of A have a non-1 stride.
  bool StridedLoadsA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Elementwise transform on B
  typename TransformB,
  /// Vectorized Loads for B
  int kElementsPerAccessB,
  /// Layout type for A matrix operand
  bool TransB_,
  /// Determines if all modes of B have a non-1 stride.
  bool StridedLoadsB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Vectorized Loads for C
  int kElementsPerAccessC,
  /// Element type of alpha and beta
  typename ElementScalar,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Operator class tag
  typename OperatorClass,
  /// Tag indicating architecture to tune for
  typename ArchTag,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Threadblock-level tile size in k-dimension (concept: IntTuple)
  typename ShapeK,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Maximal number of modes of a tensor
  int kMaxRank,
  /// Number of blocked M modes (at the threadblock level)
  int kBlockedModesM,
  /// Number of blocked N modes (at the threadblock level)
  int kBlockedModesN,
  /// target compute capability that this kernel will be compiled for
  int ccTarget,
  typename MathOperatorTag = typename arch::OpMultiplyAdd,
  bool SplitKSerial = false,
  bool StreamK = false
>
struct DefaultGett;

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Hopper Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Elementwise transform on A
    typename TransformA,
    /// Vectorized Loads for A
    int kElementsPerAccessA,
    /// Layout type for A matrix operand
    bool TransA,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Elementwise transform on B
    typename TransformB,
    /// Vectorized Loads for B
    int kElementsPerAccessB,
    /// Layout type for B matrix operand
    bool TransB,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Vectorized Loads for C
    int kElementsPerAccessC,
    /// Element type of alpha and beta
    typename ElementScalar,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Maximal number of modes of a tensor
    int kMaxRank,
    int kBlockedModesM,
    int kBlockedModesN,
    /// target compute capability that this kernel will be compiled for
    int ccTarget,
    typename MathOperatorTag,
    bool SplitKSerial,
    bool StreamK
    >
struct DefaultGett<ElementA, TransformA, kElementsPerAccessA, TransA, StridedLoadsA,
                   ElementB, TransformB, kElementsPerAccessB, TransB, StridedLoadsB,
                   ElementC, kElementsPerAccessC,
                   ElementScalar, ElementAccumulator,
                   arch::OpClassTensorOp,
                   arch::Sm90,
                   ThreadblockShape, ShapeK, WarpShape, InstructionShape,
                   kMaxRank, kBlockedModesM, kBlockedModesN,
                   ccTarget, MathOperatorTag, SplitKSerial, StreamK
                   > {
    using ElementOutput = ElementC;

    using LayoutA = typename platform::conditional<TransA, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutB = typename platform::conditional<TransB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutAccumulator = cutlass::layout::RowMajor;

    using TensorCoreInputTypeAB = typename platform::conditional<sizeof(ElementA) < sizeof(ElementB), ElementA, ElementB>::type;
    using TensorCoreInputType = typename platform::conditional<sizeof(TensorCoreInputTypeAB) < sizeof(ElementAccumulator), TensorCoreInputTypeAB, ElementAccumulator>::type;

    using SharedmemoryElementA = typename platform::conditional<sizeof(ElementA) < sizeof(TensorCoreInputType), ElementA, TensorCoreInputType>::type;
    using SharedmemoryElementB = typename platform::conditional<sizeof(ElementB) < sizeof(TensorCoreInputType), ElementB, TensorCoreInputType>::type;

    static const int kStages = 3;
    static bool const kIsComplex = (is_complex<SharedmemoryElementA>::value || is_complex<SharedmemoryElementB>::value);

    static_assert( platform::is_same<TransformA, transform::thread::UnaryTransform::Conjugate>::value ||
                   platform::is_same<TransformA, transform::thread::UnaryTransform::Identity>::value, "TransformA must be valid.");
    static_assert( platform::is_same<TransformB, transform::thread::UnaryTransform::Conjugate>::value ||
                   platform::is_same<TransformB, transform::thread::UnaryTransform::Identity>::value, "TransformB must be valid.");
    // remap to complex transform
    static const ComplexTransform complexTransformA = platform::is_same<TransformA, transform::thread::UnaryTransform::Conjugate>::value ? ComplexTransform::kConjugate : ComplexTransform::kNone;
    static const ComplexTransform complexTransformB = platform::is_same<TransformB, transform::thread::UnaryTransform::Conjugate>::value ? ComplexTransform::kConjugate : ComplexTransform::kNone;

    static const cutlass::arch::CacheOperation::Kind kCacheOpA = sizeof(ElementA) * kElementsPerAccessA >= 16 ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;
    static const cutlass::arch::CacheOperation::Kind kCacheOpB = sizeof(ElementB) * kElementsPerAccessB >= 16 ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, SharedmemoryElementA, LayoutA,
        SharedmemoryElementB, LayoutB, ElementAccumulator, LayoutAccumulator,
        arch::OpClassTensorOp, kStages, MathOperatorTag, false,
        kCacheOpA, kCacheOpB,
        complexTransformA, complexTransformB,
        kIsComplex
    >;

    //
    // Iterators
    // 

    using FreeAccessLayoutM = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
    using FreeAccessLayoutN = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;
    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA = cutlass::contraction::threadblock::FusedTensorNDimPredicatedTileAccessIterator<
          ElementA,
          kElementsPerAccessA,
          ThreadMapA,
          TransA, ShapeK,
          FreeAccessLayoutM,
          kMaxRank
    >;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB = cutlass::contraction::threadblock::FusedTensorNDimPredicatedTileAccessIterator<
          ElementB,
          kElementsPerAccessB,
          ThreadMapB,
          !TransB, ShapeK,
          FreeAccessLayoutN,
          kMaxRank
    >;

    //
    // MMA
    //

    using MmaPolicy = typename MmaCore::MmaPolicy;
    using Mma = cutlass::gemm::threadblock::MmaMultistage<
        ThreadblockShape,
        IteratorA,
        typename MmaCore::SmemIteratorA, kCacheOpA,
        IteratorB,
        typename MmaCore::SmemIteratorB, kCacheOpB,
        ElementC, layout::RowMajor,
        MmaPolicy, kStages, cutlass::gemm::SharedMemoryClearOption::kZfill
            >;

    //
    // Epilogue
    //

    static const int kPartitionsK = 1;

    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccessC,
        ElementAccumulator,
        ElementScalar>;

    /// This can be used to select the epilogue rank. The first half of the modes are mapped to the 
    /// linearized GEMM M dimension. The second half of the modes are mapped to the GEMM N dimension.
    static int const kEpilogueAffineRank = kBlockedModesN + kBlockedModesM;
    static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");

    /// Construct an AffineLayoutRankN epilogue
    using EpilogueFastAffineLayoutRankN = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOpAffineLayoutRankN<
        kEpilogueAffineRank,
        FreeAccessLayoutM,
        FreeAccessLayoutN,
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    /// Assemble the epilogue components
    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<
      EpilogueFastAffineLayoutRankN,
      EpilogueFast
    >;
    
    //
    // Threadblock Swizzle
    //

    static_assert( 2048 % ThreadblockShape::kM == 0, "" );
    static const int kBlockRows = 2048 / ThreadblockShape::kM;
    using ThreadblockSwizzle = typename cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzle<
        kBlockRows, kMaxRank, ShapeK::kRank, FreeAccessLayoutM, FreeAccessLayoutN,
        Mma::Base::WarpCount::kM < Mma::Base::WarpCount::kN ? true :
          (Mma::Base::WarpCount::kM > Mma::Base::WarpCount::kN ? false :
            Mma::IteratorA::isContractedContiguous)>;

    using TensorCoord = typename ThreadblockSwizzle::TensorCoord;

    /// Define the kernel-level GEMM operator.
    using GettKernel = kernel::Gett<Mma, Epilogue, ThreadblockSwizzle, ccTarget, SplitKSerial, StreamK>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Elementwise transform on A
    typename TransformA,
    /// Vectorized Loads for A
    int kElementsPerAccessA,
    /// Layout type for A matrix operand
    bool TransA,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Elementwise transform on B
    typename TransformB,
    /// Vectorized Loads for B
    int kElementsPerAccessB,
    /// Layout type for B matrix operand
    bool TransB,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Vectorized Loads for C
    int kElementsPerAccessC,
    /// Element type of alpha and beta
    typename ElementScalar,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Maximal number of modes of a tensor
    int kMaxRank,
    int kBlockedModesM,
    int kBlockedModesN,
    /// target compute capability that this kernel will be compiled for
    int ccTarget,
    typename MathOperatorTag,
    bool SplitKSerial,
    bool StreamK
    >
struct DefaultGett<ElementA, TransformA, kElementsPerAccessA, TransA, StridedLoadsA,
                   ElementB, TransformB, kElementsPerAccessB, TransB, StridedLoadsB,
                   ElementC, kElementsPerAccessC,
                   ElementScalar, ElementAccumulator,
                   arch::OpClassTensorOp,
                   arch::Sm80,
                   ThreadblockShape, ShapeK, WarpShape, InstructionShape,
                   kMaxRank, kBlockedModesM, kBlockedModesN,
                   ccTarget, MathOperatorTag, SplitKSerial, StreamK
                   > {
    using ElementOutput = ElementC;

    using LayoutA = typename platform::conditional<TransA, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutB = typename platform::conditional<TransB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutAccumulator = cutlass::layout::RowMajor;

    using TensorCoreInputTypeAB = typename platform::conditional<sizeof(ElementA) < sizeof(ElementB), ElementA, ElementB>::type;
    using TensorCoreInputType = typename platform::conditional<sizeof(TensorCoreInputTypeAB) < sizeof(ElementAccumulator), TensorCoreInputTypeAB, ElementAccumulator>::type;

    using SharedmemoryElementA = typename platform::conditional<sizeof(ElementA) < sizeof(TensorCoreInputType), ElementA, TensorCoreInputType>::type;
    using SharedmemoryElementB = typename platform::conditional<sizeof(ElementB) < sizeof(TensorCoreInputType), ElementB, TensorCoreInputType>::type;

    static const int kStages = 3;
    static bool const kIsComplex = (is_complex<SharedmemoryElementA>::value || is_complex<SharedmemoryElementB>::value);

//    using MathOperatorTag = typename platform::conditional<
//      kIsComplex,
//      arch::OpMultiplyAddComplex,  // TODO - indicate Gaussian complex 
//      arch::OpMultiplyAdd
//    >::type;

    static_assert( platform::is_same<TransformA, transform::thread::UnaryTransform::Conjugate>::value ||
                   platform::is_same<TransformA, transform::thread::UnaryTransform::Identity>::value, "TransformA must be valid.");
    static_assert( platform::is_same<TransformB, transform::thread::UnaryTransform::Conjugate>::value ||
                   platform::is_same<TransformB, transform::thread::UnaryTransform::Identity>::value, "TransformB must be valid.");
    // remap to complex transform
    static const ComplexTransform complexTransformA = platform::is_same<TransformA, transform::thread::UnaryTransform::Conjugate>::value ? ComplexTransform::kConjugate : ComplexTransform::kNone;
    static const ComplexTransform complexTransformB = platform::is_same<TransformB, transform::thread::UnaryTransform::Conjugate>::value ? ComplexTransform::kConjugate : ComplexTransform::kNone;

    static const cutlass::arch::CacheOperation::Kind kCacheOpA = sizeof(ElementA) * kElementsPerAccessA >= 16 ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;
    static const cutlass::arch::CacheOperation::Kind kCacheOpB = sizeof(ElementB) * kElementsPerAccessB >= 16 ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, SharedmemoryElementA, LayoutA,
        SharedmemoryElementB, LayoutB, ElementAccumulator, LayoutAccumulator,
        arch::OpClassTensorOp, kStages, MathOperatorTag, false,
        kCacheOpA, kCacheOpB,
        complexTransformA, complexTransformB,
        kIsComplex
    >;

    //
    // Iterators
    // 

    using FreeAccessLayoutM = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
    using FreeAccessLayoutN = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;
    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA = cutlass::contraction::threadblock::FusedTensorNDimPredicatedTileAccessIterator<
          ElementA,
          kElementsPerAccessA,
          ThreadMapA,
          TransA, ShapeK,
          FreeAccessLayoutM,
          kMaxRank
    >;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB = cutlass::contraction::threadblock::FusedTensorNDimPredicatedTileAccessIterator<
          ElementB,
          kElementsPerAccessB,
          ThreadMapB,
          !TransB, ShapeK,
          FreeAccessLayoutN,
          kMaxRank
    >;

    //
    // MMA
    //
//    using OperatorA = typename cutlass::transform::thread::UnaryOp<
//                         typename IteratorA::Fragment,
//                         typename cutlass::Array<SharedmemoryElementA, IteratorA::Fragment::kElements>,
//                         TransformA>;
//    using OperatorB = typename cutlass::transform::thread::UnaryOp<
//                         typename IteratorB::Fragment,
//                         typename cutlass::Array<SharedmemoryElementB, IteratorB::Fragment::kElements>,
//                         TransformB>;

    // Define the threadblock-scoped pipelined matrix multiply
    // using Mma = cutlass::contractionFused::threadblock::MmaMultistage<
    //              typename MmaCore::Shape, 
    //              IteratorA, 
    //              typename MmaCore::SmemIteratorA,
    //              kCacheOpA,
    //              //OperatorA,
    //              IteratorB, 
    //              typename MmaCore::SmemIteratorB,
    //              kCacheOpA,
    //              //OperatorB,
    //              ElementAccumulator, 
    //              LayoutAccumulator,
    //              typename MmaCore::MmaPolicy,
    //              kStages>;

    using MmaPolicy = typename MmaCore::MmaPolicy;
    // using Mma = cutlass::conv::threadblock::ImplicitGemmMultistage<
    //     ThreadblockShape,
    //     IteratorA,
    //     typename MmaCore::SmemIteratorA, kCacheOpA,
    //     IteratorB,
    //     typename MmaCore::SmemIteratorB, kCacheOpB,
    //     MmaPolicy, kStages
    //         >;
    using Mma = cutlass::gemm::threadblock::MmaMultistage<
        ThreadblockShape,
        IteratorA,
        typename MmaCore::SmemIteratorA, kCacheOpA,
        IteratorB,
        typename MmaCore::SmemIteratorB, kCacheOpB,
        ElementC, layout::RowMajor,
        MmaPolicy, kStages, cutlass::gemm::SharedMemoryClearOption::kZfill
            >;

    //
    // Epilogue
    //

    static const int kPartitionsK = 1;

    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccessC,
        ElementAccumulator,
        ElementScalar>;

    /// This can be used to select the epilogue rank. The first half of the modes are mapped to the 
    /// linearized GEMM M dimension. The second half of the modes are mapped to the GEMM N dimension.
    static int const kEpilogueAffineRank = kBlockedModesN + kBlockedModesM;
    static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");

    /// Construct an AffineLayoutRankN epilogue
    using EpilogueFastAffineLayoutRankN = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOpAffineLayoutRankN<
        kEpilogueAffineRank,
        FreeAccessLayoutM,
        FreeAccessLayoutN,
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    /// Assemble the epilogue components
    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<
      EpilogueFastAffineLayoutRankN,
      EpilogueFast
    >;
    
    //
    // Threadblock Swizzle
    //

    static_assert( 2048 % ThreadblockShape::kM == 0, "" );
    static const int kBlockRows = 2048 / ThreadblockShape::kM;
    using ThreadblockSwizzle = typename cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzle<
        kBlockRows, kMaxRank, ShapeK::kRank, FreeAccessLayoutM, FreeAccessLayoutN,
        Mma::Base::WarpCount::kM < Mma::Base::WarpCount::kN ? true :
          (Mma::Base::WarpCount::kM > Mma::Base::WarpCount::kN ? false :
            Mma::IteratorA::isContractedContiguous)>;

    using TensorCoord = typename ThreadblockSwizzle::TensorCoord;

    /// Define the kernel-level GEMM operator.
    using GettKernel = kernel::Gett<Mma, Epilogue, ThreadblockSwizzle, ccTarget, SplitKSerial, StreamK>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Turing Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Elementwise transform on A
    typename TransformA,
    /// Vectorized Loads for A
    int kElementsPerAccessA,
    /// Layout type for A matrix operand
    bool TransA,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Elementwise transform on B
    typename TransformB,
    /// Vectorized Loads for B
    int kElementsPerAccessB,
    /// Layout type for B matrix operand
    bool TransB,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Vectorized Loads for C
    int kElementsPerAccessC,
    /// Element type of alpha and beta
    typename ElementScalar,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Maximal number of modes of a tensor
    int kMaxRank,
    int kBlockedModesM,
    int kBlockedModesN,
    /// target compute capability that this kernel will be compiled for
    int ccTarget,
    typename MathOperatorTag,
    bool SplitKSerial,
    bool StreamK
    >
struct DefaultGett<ElementA, TransformA, kElementsPerAccessA, TransA, StridedLoadsA,
                   ElementB, TransformB, kElementsPerAccessB, TransB, StridedLoadsB,
                   ElementC, kElementsPerAccessC,
                   ElementScalar, ElementAccumulator,
                   arch::OpClassTensorOp,
                   arch::Sm75,
                   ThreadblockShape, ShapeK, WarpShape, InstructionShape,
                   kMaxRank, kBlockedModesM, kBlockedModesN,
                   ccTarget, MathOperatorTag, SplitKSerial, StreamK
                   > {
    using ElementOutput = ElementC;

    using LayoutA = typename platform::conditional<TransA, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutB = typename platform::conditional<TransB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutAccumulator = cutlass::layout::RowMajor;

    //using TensorCoreInputType = cutlass::half_t;//getTensorCoreInputType<ElementA, ElementB>(); // TODO
    using TensorCoreInputTypeAB = typename platform::conditional<sizeof(ElementA) < sizeof(ElementB), ElementA, ElementB>::type;
    //using TensorCoreInputType = typename platform::conditional<sizeof(TensorCoreInputTypeAB) < sizeof(ElementAccumulator), TensorCoreInputTypeAB, ElementAccumulator>::type;
    using TensorCoreInputType = cutlass::half_t;

    // OpClassTensorOp Sm75
    // Should be able to handle TF32 / BF16
    using SharedmemoryElementA = typename platform::conditional<sizeof(ElementA) < sizeof(TensorCoreInputType), ElementA, TensorCoreInputType>::type;
    using SharedmemoryElementB = typename platform::conditional<sizeof(ElementB) < sizeof(TensorCoreInputType), ElementB, TensorCoreInputType>::type;
    // typename DE<TensorCoreInputType, SharedmemoryElementA, SharedmemoryElementB>::lol a;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, SharedmemoryElementA, LayoutA,
        SharedmemoryElementB, LayoutB, ElementAccumulator, LayoutAccumulator,
        arch::OpClassTensorOp, 2, MathOperatorTag>;

    //
    // Iterators
    // 

    using FreeAccessLayoutM = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
    using FreeAccessLayoutN = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;
    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA =
      cutlass::contraction::threadblock::MyIterator<
          ElementA,
          kElementsPerAccessA,
          ThreadMapA,
          TransA, ShapeK, FreeAccessLayoutM, kMaxRank
      >;
    static_assert(IteratorA::kNumModesContracted == ShapeK::kRank, "num contracted don't match");

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB =
      cutlass::contraction::threadblock::MyIterator<
          ElementB,
          kElementsPerAccessB,
          ThreadMapB,
          !TransB, ShapeK, FreeAccessLayoutN, kMaxRank
        >;


    //
    // MMA
    //
    using OperatorA = typename cutlass::NumericArrayConverter<
            SharedmemoryElementA,
            ElementA,
            IteratorA::Fragment::kElements, FloatRoundStyle::round_to_nearest, TransformA>;

    using OperatorB = typename cutlass::NumericArrayConverter<
            SharedmemoryElementB,
            ElementB,
            IteratorB::Fragment::kElements, FloatRoundStyle::round_to_nearest, TransformB>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPolicy = typename MmaCore::MmaPolicy;
    using Mma = cutlass::conv::threadblock::ImplicitGemmPipelined<
        ThreadblockShape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        ElementAccumulator,
        LayoutAccumulator,
        MmaPolicy,
        OperatorA, OperatorB
            >;

    //
    // Epilogue
    //

    static const int kPartitionsK = 1;

    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccessC,
        ElementAccumulator,
        ElementScalar>;

    /// This can be used to select the epilogue rank. The first half of the modes are mapped to the 
    /// linearized GEMM M dimension. The second half of the modes are mapped to the GEMM N dimension.
    static int const kEpilogueAffineRank = kBlockedModesN + kBlockedModesM;
    static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");

    /// Construct an AffineLayoutRankN epilogue
    using EpilogueFastAffineLayoutRankN = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOpAffineLayoutRankN<
        kEpilogueAffineRank,
        FreeAccessLayoutM,
        FreeAccessLayoutN,
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    /// Assemble the epilogue components
    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<
      EpilogueFastAffineLayoutRankN,
      EpilogueFast
    >;

    //
    // Threadblock Swizzle
    //

    static_assert( 2048 % ThreadblockShape::kM == 0, "" );
    static const int kBlockRows = 2048 / ThreadblockShape::kM;
    using ThreadblockSwizzle = typename cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzle<
        kBlockRows, kMaxRank, ShapeK::kRank, FreeAccessLayoutM, FreeAccessLayoutN,
        Mma::Base::WarpCount::kM < Mma::Base::WarpCount::kN ? true :
          (Mma::Base::WarpCount::kM > Mma::Base::WarpCount::kN ? false :
            Mma::IteratorA::isContractedContiguous)>;

    using TensorCoord = typename ThreadblockSwizzle::TensorCoord;


    /// Define the kernel-level GEMM operator.
    using GettKernel = kernel::Gett<Mma, Epilogue, ThreadblockSwizzle, ccTarget, SplitKSerial, StreamK>;
};

////////////////////////////////////////////////////////////////////////////////
// Partial specialization for Volta Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Elementwise transform on A
    typename TransformA,
    /// Vectorized Loads for A
    int kElementsPerAccessA,
    /// Layout type for A matrix operand
    bool TransA,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Elementwise transform on B
    typename TransformB,
    /// Vectorized Loads for B
    int kElementsPerAccessB,
    /// Layout type for B matrix operand
    bool TransB,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Vectorized Loads for C
    int kElementsPerAccessC,
    /// Element type of alpha and beta
    typename ElementScalar,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Maximal number of modes of a tensor
    int kMaxRank,
    int kBlockedModesM,
    int kBlockedModesN,
    /// target compute capability that this kernel will be compiled for
    int ccTarget,
    typename MathOperatorTag,
    bool SplitKSerial,
    bool StreamK
    >
struct DefaultGett<ElementA, TransformA, kElementsPerAccessA, TransA, StridedLoadsA,
                   ElementB, TransformB, kElementsPerAccessB, TransB, StridedLoadsB,
                   ElementC, kElementsPerAccessC,
                   ElementScalar, ElementAccumulator,
                   arch::OpClassTensorOp,
                   arch::Sm70,
                   ThreadblockShape, ShapeK, WarpShape, InstructionShape,
                   kMaxRank, kBlockedModesM, kBlockedModesN,
                   ccTarget, MathOperatorTag, SplitKSerial, StreamK
                   > {
    using ElementOutput = ElementC;

    using LayoutA = typename platform::conditional<TransA, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutB = typename platform::conditional<TransB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutAccumulator = cutlass::layout::RowMajor;

    using TensorCoreInputType = cutlass::half_t;//getTensorCoreInputType<ElementA, ElementB>(); // TODO

    using SharedmemoryElementA = typename platform::conditional<sizeof(ElementA) < sizeof(TensorCoreInputType), ElementA, TensorCoreInputType>::type;
    using SharedmemoryElementB = typename platform::conditional<sizeof(ElementB) < sizeof(TensorCoreInputType), ElementB, TensorCoreInputType>::type;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, SharedmemoryElementA, LayoutA,
        SharedmemoryElementB, LayoutB, ElementAccumulator, LayoutAccumulator,
        arch::OpClassTensorOp>;

    //
    // Iterators
    // 

    using FreeAccessLayoutM = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
    using FreeAccessLayoutN = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;
    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA =
      cutlass::contraction::threadblock::MyIterator<
          ElementA,
          kElementsPerAccessA,
          ThreadMapA,
          TransA, ShapeK, FreeAccessLayoutM, kMaxRank
      >;
    static_assert(IteratorA::kNumModesContracted == ShapeK::kRank, "num contracted don't match");

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB =
      cutlass::contraction::threadblock::MyIterator<
          ElementB,
          kElementsPerAccessB,
          ThreadMapB,
          !TransB, ShapeK, FreeAccessLayoutN, kMaxRank
        >;

    //
    // MMA
    //
    using OperatorA = typename cutlass::NumericArrayConverter<
            SharedmemoryElementA,
            ElementA,
            IteratorA::Fragment::kElements, FloatRoundStyle::round_to_nearest, TransformA>;

    using OperatorB = typename cutlass::NumericArrayConverter<
            SharedmemoryElementB,
            ElementB,
            IteratorB::Fragment::kElements, FloatRoundStyle::round_to_nearest, TransformB>;

    // Define the threadblock-scoped pipelined matrix multiply
    using MmaPolicy = typename MmaCore::MmaPolicy;
    using Mma = cutlass::conv::threadblock::ImplicitGemmPipelined<
        ThreadblockShape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        ElementAccumulator,
        LayoutAccumulator,
        MmaPolicy,
        OperatorA, OperatorB
            >;
    //
    // Epilogue
    //

    static const int kPartitionsK = 1;

    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccessC,
        ElementAccumulator,
        ElementScalar>;


    /// This can be used to select the epilogue rank. The first half of the modes are mapped to the 
    /// linearized GEMM M dimension. The second half of the modes are mapped to the GEMM N dimension.
    static int const kEpilogueAffineRank = kBlockedModesN + kBlockedModesM;
    static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");

    /// Construct an AffineLayoutRankN epilogue
    using EpilogueFastAffineLayoutRankN = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOpAffineLayoutRankN<
        kEpilogueAffineRank,
        FreeAccessLayoutM,
        FreeAccessLayoutN,
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
        ThreadblockShape,
        typename Mma::Operator,
        kPartitionsK,
        OutputOp,
        kElementsPerAccessC
      >::Epilogue;

    /// Assemble the epilogue components
    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<
      EpilogueFastAffineLayoutRankN,
      EpilogueFast
    >;

    //
    // Threadblock Swizzle
    //

    static_assert( 2048 % ThreadblockShape::kM == 0, "" );
    static const int kBlockRows = 2048 / ThreadblockShape::kM;
    using ThreadblockSwizzle = typename cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzle<
        kBlockRows, kMaxRank, ShapeK::kRank, FreeAccessLayoutM, FreeAccessLayoutN,
        Mma::Base::WarpCount::kM < Mma::Base::WarpCount::kN ? true :
          (Mma::Base::WarpCount::kM > Mma::Base::WarpCount::kN ? false :
            Mma::IteratorA::isContractedContiguous)>;

    using TensorCoord = typename ThreadblockSwizzle::TensorCoord;


    /// Define the kernel-level GEMM operator.
    using GettKernel = kernel::Gett<Mma, Epilogue, ThreadblockSwizzle, ccTarget, SplitKSerial, StreamK>;
};
//////////////////////////////////////////////////////////////////////////////////
//
/// Partial specialization for SIMT
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Elementwise transform on A
    typename TransformA,
    /// Vectorized Loads for A
    int kElementsPerAccessA,
    /// Layout type for A matrix operand
    bool TransA,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Elementwise transform on B
    typename TransformB,
    /// Vectorized Loads for B
    int kElementsPerAccessB,
    /// Layout type for B matrix operand
    bool TransB,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Vectorized Loads for C
    int kElementsPerAccessC,
    /// Element type of alpha and beta
    typename ElementScalar,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Maximal number of modes of a tensor
    int kMaxRank,
    int kBlockedModesM,
    int kBlockedModesN,
    /// target compute capability that this kernel will be compiled for
    int ccTarget,
    typename MathOperatorTag,
    bool SplitKSerial,
    bool StreamK
    >
struct DefaultGett<ElementA, TransformA, kElementsPerAccessA, TransA, StridedLoadsA,
                   ElementB, TransformB, kElementsPerAccessB, TransB, StridedLoadsB,
                   ElementC, kElementsPerAccessC,
                   ElementScalar, ElementAccumulator,
                   arch::OpClassSimt,
                   arch::Sm50,
                   ThreadblockShape, ShapeK, WarpShape, InstructionShape,
                   kMaxRank, kBlockedModesM, kBlockedModesN,
                   ccTarget, MathOperatorTag, SplitKSerial, StreamK
                   > {
    using ElementOutput = ElementC;

    using LayoutA = typename platform::conditional<TransA, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutB = typename platform::conditional<TransB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutAccumulator = cutlass::layout::RowMajor;

    using SharedmemoryElementA = typename platform::conditional<sizeof(ElementA) < sizeof(ElementAccumulator), ElementA, ElementAccumulator>::type;
    using SharedmemoryElementB = typename platform::conditional<sizeof(ElementB) < sizeof(ElementAccumulator), ElementB, ElementAccumulator>::type;

    static const int kStages = 2;
    // Define the core components from GEMM

    static const int kAccessSizeInBits = kElementsPerAccessA > 1 &&
                  kElementsPerAccessA == kElementsPerAccessB &&
                  sizeof(ElementA) == sizeof(ElementB) ?  kElementsPerAccessB * static_cast<int>(sizeof(ElementB)) * 8: -1;

    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCoreWithAccessSize<
      ThreadblockShape, WarpShape, InstructionShape, SharedmemoryElementA, LayoutA,
      SharedmemoryElementB, LayoutB, ElementAccumulator, LayoutAccumulator,
      arch::OpClassSimt, kAccessSizeInBits, kStages>;

    static_assert(ThreadblockShape::kK == cutlass::contraction::Product<ShapeK>::value, "product of k-tilings doesn't match ThreadblockShape::kK");
    //
    // Iterators
    // 

    using FreeAccessLayoutM = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
    using FreeAccessLayoutN = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;
    // Define iterators over tiles from the B operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA =
      cutlass::contraction::threadblock::MyIterator<
          ElementA,
          kElementsPerAccessA,
          ThreadMapA,
          TransA, ShapeK, FreeAccessLayoutM, kMaxRank
      >;
    static_assert(IteratorA::kNumModesContracted == ShapeK::kRank, "num contracted don't match");

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB =
      cutlass::contraction::threadblock::MyIterator<
          ElementB,
          kElementsPerAccessB,
          ThreadMapB,
          !TransB, ShapeK, FreeAccessLayoutN, kMaxRank
        >;

    //
    // MMA
    //
//    using OperatorA = typename cutlass::transform::thread::UnaryOp<
//                         typename IteratorA::Fragment,
//                         typename cutlass::Array<SharedmemoryElementA, IteratorA::Fragment::kElements>,
//                         TransformA>;
    using OperatorA = typename cutlass::NumericArrayConverter<
            SharedmemoryElementA,
            ElementA,
            IteratorA::Fragment::kElements, FloatRoundStyle::round_to_nearest, TransformA>;
//    using OperatorB = typename cutlass::transform::thread::UnaryOp<
//                         typename IteratorB::Fragment,
//                         typename cutlass::Array<SharedmemoryElementB, IteratorB::Fragment::kElements>,
//                         TransformB>; // TODO use
    using OperatorB = typename cutlass::NumericArrayConverter<
            SharedmemoryElementB,
            ElementB,
            IteratorB::Fragment::kElements, FloatRoundStyle::round_to_nearest, TransformB>;
//    // Define the threadblock-scoped pipelined matrix multiply
//    using Mma = cutlass::contraction::threadblock::MmaPipelined<
//        typename MmaCore::Shape, 
//                 IteratorA, 
//                 typename MmaCore::SmemIteratorA,
//                 OperatorA,
//                 IteratorB, 
//                 typename MmaCore::SmemIteratorB, 
//                 OperatorB,
//                 ElementAccumulator, 
//                 LayoutAccumulator,
//                 typename MmaCore::MmaPolicy>;

    // Warp-level GEMM components
    using MmaPolicy = typename MmaCore::MmaPolicy;

    // Define the Mma
    using Mma = cutlass::conv::threadblock::ImplicitGemmPipelined<
        ThreadblockShape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        ElementAccumulator,
        LayoutAccumulator,
        MmaPolicy,
        OperatorA, OperatorB
            >;

    //
    // Epilogue
    //

    static_assert(kElementsPerAccessC == 1, "simt epilogue must operate on scalars");
    static const int kPartitionsK = 1;

    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccessC,
        ElementAccumulator,
        ElementScalar>;

    #if 0
    using EpilogueNaive = typename cutlass::contraction::threadblock::NaiveEpilogue4DSimt<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      typename Mma::Operator,
      kPartitionsK,
      ElementC,
      ElementAccumulator,
      OutputOp,
      kElementsPerAccessC,kMaxRank,
      typename MmaCore::Policy>;
    #endif

    /// This can be used to select the epilogue rank. The first half of the modes are mapped to the 
    /// linearized GEMM M dimension. The second half of the modes are mapped to the GEMM N dimension.
    static int const kEpilogueAffineRank = kBlockedModesN + kBlockedModesM;
    static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");


    /// Define the epilogue
    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      OutputOp,
      kElementsPerAccessC
      >::Epilogue;

    /// Construct an AffineLayoutRankN epilogue
    using EpilogueFastAffineLayoutRankN = typename cutlass::epilogue::threadblock::DefaultEpilogueSimtAffineLayoutRankN<
      kEpilogueAffineRank,
      FreeAccessLayoutM,
      FreeAccessLayoutN,
      ThreadblockShape,
      typename Mma::Operator,
      OutputOp,
      kElementsPerAccessC
      >::Epilogue;
      
    /// Assemble the epilogue components
    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<
      EpilogueFastAffineLayoutRankN,
      EpilogueFast
    >;

    //
    // Threadblock Swizzle
    //

    static_assert( 2048 % ThreadblockShape::kM == 0, "" );
    static const int kBlockRows = 2048 / ThreadblockShape::kM;
    using ThreadblockSwizzle = typename cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzle<
        kBlockRows, kMaxRank, ShapeK::kRank, FreeAccessLayoutM, FreeAccessLayoutN,
        Mma::Base::WarpCount::kM < Mma::Base::WarpCount::kN ? true :
          (Mma::Base::WarpCount::kM > Mma::Base::WarpCount::kN ? false :
            Mma::IteratorA::isContractedContiguous)>;

    using TensorCoord = typename ThreadblockSwizzle::TensorCoord;

    /// Define the kernel-level GEMM operator.
    using GettKernel = kernel::Gett<Mma, Epilogue, ThreadblockSwizzle, ccTarget, SplitKSerial, StreamK>;
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Ampere Architecture (SIMT)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Elementwise transform on A
    typename TransformA,
    /// Vectorized Loads for A
    int kElementsPerAccessA,
    /// Layout type for A matrix operand
    bool TransA,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Elementwise transform on B
    typename TransformB,
    /// Vectorized Loads for B
    int kElementsPerAccessB,
    /// Layout type for B matrix operand
    bool TransB,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Vectorized Loads for C
    int kElementsPerAccessC,
    /// Element type of alpha and beta
    typename ElementScalar,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Maximal number of modes of a tensor
    int kMaxRank,
    int kBlockedModesM,
    int kBlockedModesN,
    /// target compute capability that this kernel will be compiled for
    int ccTarget,
    typename MathOperatorTag_,
    bool SplitKSerial,
    bool StreamK
    >
struct DefaultGett<ElementA, TransformA, kElementsPerAccessA, TransA, StridedLoadsA,
                   ElementB, TransformB, kElementsPerAccessB, TransB, StridedLoadsB,
                   ElementC, kElementsPerAccessC,
                   ElementScalar, ElementAccumulator,
                   arch::OpClassSimt,
                   arch::Sm80,
                   ThreadblockShape, ShapeK, WarpShape, InstructionShape,
                   kMaxRank, kBlockedModesM, kBlockedModesN,
                   ccTarget, MathOperatorTag_, SplitKSerial, StreamK
                   > {
    using ElementOutput = ElementC;

    using LayoutA = typename platform::conditional<TransA, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutB = typename platform::conditional<TransB, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
    using LayoutAccumulator = cutlass::layout::RowMajor;

    using TensorCoreInputTypeAB = typename platform::conditional<sizeof(ElementA) < sizeof(ElementB), ElementA, ElementB>::type;
    using TensorCoreInputType = typename platform::conditional<sizeof(TensorCoreInputTypeAB) < sizeof(ElementAccumulator), TensorCoreInputTypeAB, ElementAccumulator>::type;

    using SharedmemoryElementA = typename platform::conditional<sizeof(ElementA) < sizeof(TensorCoreInputType), ElementA, TensorCoreInputType>::type;
    using SharedmemoryElementB = typename platform::conditional<sizeof(ElementB) < sizeof(TensorCoreInputType), ElementB, TensorCoreInputType>::type;

    static const int kStages = 3;
    static bool const kIsComplex = (is_complex<SharedmemoryElementA>::value || is_complex<SharedmemoryElementB>::value);

    using MathOperatorTag = typename platform::conditional<
      kIsComplex,
      arch::OpMultiplyAddComplex,  // TODO - indicate Gaussian complex 
      arch::OpMultiplyAdd
    >::type;

    static_assert( platform::is_same<TransformA, transform::thread::UnaryTransform::Conjugate>::value ||
                   platform::is_same<TransformA, transform::thread::UnaryTransform::Identity>::value, "TransformA must be valid.");
    static_assert( platform::is_same<TransformB, transform::thread::UnaryTransform::Conjugate>::value ||
                   platform::is_same<TransformB, transform::thread::UnaryTransform::Identity>::value, "TransformB must be valid.");
    // remap to complex transform
    static const ComplexTransform complexTransformA = platform::is_same<TransformA, transform::thread::UnaryTransform::Conjugate>::value ? ComplexTransform::kConjugate : ComplexTransform::kNone;
    static const ComplexTransform complexTransformB = platform::is_same<TransformB, transform::thread::UnaryTransform::Conjugate>::value ? ComplexTransform::kConjugate : ComplexTransform::kNone;

    static const cutlass::arch::CacheOperation::Kind kCacheOpA = sizeof(ElementA) * kElementsPerAccessA >= 16 ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;
    static const cutlass::arch::CacheOperation::Kind kCacheOpB = sizeof(ElementB) * kElementsPerAccessB >= 16 ? cutlass::arch::CacheOperation::Global : cutlass::arch::CacheOperation::Always;

    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, SharedmemoryElementA, LayoutA,
        SharedmemoryElementB, LayoutB, ElementAccumulator, LayoutAccumulator,
        arch::OpClassSimt, kStages, MathOperatorTag, false,
        kCacheOpA, kCacheOpB,
        complexTransformA, complexTransformB,
        kIsComplex
    >;

    //
    // Iterators
    // 

    using FreeAccessLayoutM = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
    using FreeAccessLayoutN = cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;
    // Define iterators over tiles from the A operand
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA = cutlass::contraction::threadblock::FusedTensorNDimPredicatedTileAccessIterator<
          ElementA,
          kElementsPerAccessA,
          ThreadMapA,
          TransA, ShapeK, FreeAccessLayoutM, 
          kMaxRank
    >;

    // Define iterators over tiles from the B operand
    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB = cutlass::contraction::threadblock::FusedTensorNDimPredicatedTileAccessIterator<
          ElementB,
          kElementsPerAccessB,
          ThreadMapB,
          !TransB, ShapeK, FreeAccessLayoutN,
          kMaxRank
    >;

    using TensorCoord = typename IteratorA::TensorCoord;

    //
    // MMA
    //

    using MmaPolicy = typename MmaCore::MmaPolicy;
    // using Mma = cutlass::conv::threadblock::ImplicitGemmMultistage<
    //     ThreadblockShape,
    //     IteratorA,
    //     typename MmaCore::SmemIteratorA, kCacheOpA,
    //     IteratorB,
    //     typename MmaCore::SmemIteratorB, kCacheOpB,
    //     MmaPolicy, kStages
    //         >;
    using Mma = cutlass::gemm::threadblock::MmaMultistage<
        ThreadblockShape,
        IteratorA,
        typename MmaCore::SmemIteratorA, kCacheOpA,
        IteratorB,
        typename MmaCore::SmemIteratorB, kCacheOpB,
        ElementC, layout::RowMajor,
        MmaPolicy, kStages, cutlass::gemm::SharedMemoryClearOption::kZfill
            >;

    //
    // Epilogue
    //

    static const int kPartitionsK = 1;

    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccessC,
        ElementAccumulator,
        ElementScalar>;

//    using EpilogueNaive = typename cutlass::contractionFused::threadblock::NaiveEpilogue4DTensorOp<
//      ThreadblockShape,
//      WarpShape,
//      InstructionShape,
//      typename Mma::Operator,
//      kPartitionsK,
//      ElementC,
//      ElementAccumulator,
//      OutputOp,
//      kElementsPerAccessC>;
//
//    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
//        ThreadblockShape,
//        typename Mma::Operator,
//        kPartitionsK,
//        OutputOp,
//        kElementsPerAccessC
//            >::Epilogue;
//
//    //using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<EpilogueNaive, EpilogueFast>;
//    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<EpilogueNaive, EpilogueFast>;
//    using EpilogueNaive = typename cutlass::contraction::threadblock::NaiveEpilogue4DSimt<
//      ThreadblockShape,
//      WarpShape,
//      InstructionShape,
//      typename Mma::Operator,
//      kPartitionsK,
//      ElementC,
//      ElementAccumulator,
//      OutputOp,
//      kElementsPerAccessC,
//      typename MmaCore::Policy>;

    /// This can be used to select the epilogue rank. The first half of the modes are mapped to the 
    /// linearized GEMM M dimension. The second half of the modes are mapped to the GEMM N dimension.
    static int const kEpilogueAffineRank = kBlockedModesN + kBlockedModesM;
    static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");


    /// Construct an AffineLayoutRankN epilogue
    using EpilogueFastAffineLayoutRankN = typename cutlass::epilogue::threadblock::DefaultEpilogueSimtAffineLayoutRankN<
      kEpilogueAffineRank,
      FreeAccessLayoutM,
      FreeAccessLayoutN,
      ThreadblockShape,
      typename Mma::Operator,
      OutputOp,
      kElementsPerAccessC
      >::Epilogue;

    /// Define the epilogue
    using EpilogueFast = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      OutputOp,
      kElementsPerAccessC
      >::Epilogue;

    /// Assemble the epilogue components
    using Epilogue = typename cutlass::contraction::threadblock::DynamicEpilogue<
      EpilogueFastAffineLayoutRankN,
      EpilogueFast
    >;

    //
    // Threadblock Swizzle
    //

    static_assert( 2048 % ThreadblockShape::kM == 0, "" );
    static const int kBlockRows = 2048 / ThreadblockShape::kM;
    using ThreadblockSwizzle = typename cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzle<
        kBlockRows, kMaxRank, ShapeK::kRank, FreeAccessLayoutM, FreeAccessLayoutN,
        Mma::Base::WarpCount::kM < Mma::Base::WarpCount::kN ? true :
          (Mma::Base::WarpCount::kM > Mma::Base::WarpCount::kN ? false :
            Mma::IteratorA::isContractedContiguous)>;


    /// Define the kernel-level GEMM operator.
    using GettKernel = kernel::Gett<Mma, Epilogue, ThreadblockSwizzle, ccTarget, SplitKSerial, StreamK>;
};

}  // namespace kernel
}  // namespace contraction
}  // namespace cutlass
