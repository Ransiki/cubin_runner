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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/contraction/device_kernel.h"
#include "cutlass/coord.h"
#include "cutlass/transform/thread/unary_op.h"

#ifdef CUTENSOR_USE_LOGGER
#include <cutensor/defines.h>
#endif

#ifdef DEBUG
#include <iostream>
#endif

#include "cutlass/contraction/kernel/default_gett.h"

#include "cutlass/cuda_host_adapter.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contraction {
namespace device {

////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_CONTRACTION_KERNEL_RENAME

template<typename T>
void dispatch_cutlass_kernel(int threadblockCount, int threadCount, int smem_size, cudaStream_t stream, const typename T::Params &params);

template<typename T>
void* lookup_cutlass_kernel();

#else

template<typename T>
void dispatch_cutlass_kernel(int threadblockCount, int threadCount, int smem_size, cudaStream_t stream, const typename T::Params &params) {
    cutlass::arch::synclog_setup();
    cutlass::KernelSpecialization<T><<<threadblockCount, threadCount, smem_size, stream>>>(params);
}

template<typename T>
void* lookup_cutlass_kernel() {
    return (void*) cutlass::KernelSpecialization<T>;
}

#endif

/*! Gett device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using CUTLASS components at the threadblock, warp, 
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for
      typename ArchTag,
      
      /// Threadblock-level tile size (concept: GemmShape)
      typename ThreadblockShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename WarpShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename InstructionShape,
      
      /// Epilogue output operator
      typename EpilogueOutputOp,
      
      /// Number of stages used in the pipelined mainloop
      int Stages
    >
    class Gett;
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Vectorized Loads for A
    int kElementsPerAccessA_,
    /// Layout type for A matrix operand
    bool TransA_,
    /// Determines if all modes of A have a non-1 stride.
    bool StridedLoadsA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Vectorized Loads for B
    int kElementsPerAccessB_,
    /// Layout type for B matrix operand
    bool TransB_,
    /// Determines if all modes of B have a non-1 stride.
    bool StridedLoadsB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Vectorized Loads for C
    int kElementsPerAccessC_,
    /// Element type of alpha and beta
    typename ElementScalar_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Number of blocked M modes (at the threadblock level)
    int kBlockedModesM_,
    /// Number of blocked N modes (at the threadblock level)
    int kBlockedModesN_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// target compute capability that this kernel will be compiled for
    int ccTarget_,
    int kMaxRank,
    /// Elementwise transform on A
    typename TransformA_ = typename cutlass::transform::thread::UnaryTransform::Identity,
    /// Elementwise transform on B
    typename TransformB_ = typename cutlass::transform::thread::UnaryTransform::Identity,
    typename MathOperatorTag = typename arch::OpMultiplyAdd,
    bool SplitKSerial = true,
    bool StreamK = false
    >
class Gett {
  
  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

 public:

  static const int kMajorVersion = 0;
  static const int kMinorVersion = 3;
  static const int kPatchVersion = 0;

  static_assert(kMaxRank >= ShapeK::kRank, "Too many contracted dimentsion.");

  using ElementA = ElementA_;
  using TransformA = TransformA_;
  static const bool TransA = TransA_;
  static const bool StridedLoadsA = StridedLoadsA_;
  static const int  kElementsPerAccessA = kElementsPerAccessA_;
  using ElementB = ElementB_;
  using TransformB = TransformB_;
  static const bool TransB = TransB_;
  static const bool StridedLoadsB = StridedLoadsB_;
  static const int  kElementsPerAccessB = kElementsPerAccessB_;
  using ElementC = ElementC_;
  static const int  kElementsPerAccessC = kElementsPerAccessC_;
  using ElementScalar = ElementScalar_;
  using ElementAccumulator = ElementAccumulator_;
  static const int kBlockedModesM = kBlockedModesM_;
  static const int kBlockedModesN = kBlockedModesN_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  static const int ccTarget = ccTarget_;
  using TensorCoord = Coord<kMaxRank>;
  using LongIndex = int64_t;
  using TensorStrideCoord = Coord<kMaxRank, LongIndex>;
  static_assert(kBlockedModesM == kBlockedModesN, "Affine layout needs to be symmetric in terms of M and N");
  static const bool kSplitKSerial = SplitKSerial;
  static const bool kStreamK = StreamK;

  /// Define the kernel
  using GettKernel = typename kernel::DefaultGett<
    ElementA,
    TransformA,
    kElementsPerAccessA,
    TransA,
    StridedLoadsA,
    ElementB,
    TransformB,
    kElementsPerAccessB,
    TransB,
    StridedLoadsB,
    ElementC,
    kElementsPerAccessC,
    ElementScalar,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    ShapeK,
    WarpShape,
    InstructionShape,
    kMaxRank, kBlockedModesM, kBlockedModesN,
    ccTarget, MathOperatorTag, kSplitKSerial, kStreamK>::GettKernel;

  using TensorCoordK = typename GettKernel::TensorCoordK;

private:

public:

  /// Constructs the GETT.
  Gett() {}

  Status handleError(cudaError_t error) const
  {
    if (error == cudaSuccess) {
      return Status::kSuccess;
    } else if (error == cudaErrorInsufficientDriver) {
#ifdef CUTENSOR_USE_LOGGER
      LOG_ERROR("CUDA error: {}", cudaGetErrorString(error));
#endif
      return Status::kErrorInsufficientDriver;
    } else if (error == cudaErrorInvalidDeviceFunction) {
#ifdef CUTENSOR_USE_LOGGER
      LOG_ERROR("CUDA error: {}", cudaGetErrorString(error));
#endif
      return Status::kErrorArchMismatch;
    } else if (error == cudaErrorMemoryAllocation) {
#ifdef CUTENSOR_USE_LOGGER
      LOG_ERROR("CUDA error: {}", cudaGetErrorString(error));
#endif
      return Status::kErrorMemoryAllocation;
    } else {
#ifdef CUTENSOR_USE_LOGGER
      LOG_ERROR("CUDA error: {}", cudaGetErrorString(error));
#endif
      return Status::kErrorInternal;
    }
  }

  Status run(typename GettKernel::Params &params, 
             cudaStream_t stream = nullptr,
             CudaHostAdapter *cuda_adapter = nullptr) const
  {
    static int const kWarpCount = 
          (ThreadblockShape::kM / WarpShape::kM) * 
          (ThreadblockShape::kN / WarpShape::kN);
    static int const kThreadCount = 32 * kWarpCount;

    // typename GettKernel::ThreadblockSwizzle threadblock_swizzle; 
    // int kThreadblockCount = threadblock_swizzle.get_grid_shape(params.extent,
    //                                                            params.startM,
    //                                                            params.startN,
    //                                                            params.startL,
    //                                                            params.partitions);
    int kThreadblockCount(params.gridSize);
    
    if (params.splittingActive()) {
      // Only do memset on semaphore. Scratchpad is too large.
      if constexpr (kEnableCudaHostAdapter) {
        auto result = cuda_adapter->memsetDevice<int>(params.semaphore, 0, size_t(params.output_tile_num), stream);
        if (result != Status::kSuccess) {
          return result;
        }
      }
      else {
        auto result = cudaMemsetAsync(params.semaphore, 0, sizeof(int) * size_t(params.output_tile_num), stream);
        if (result != cudaSuccess) {
          return handleError(result);
        }
      }
    }

    int smem_size = int(sizeof(typename GettKernel::SharedStorage));
    if (kEnableCudaHostAdapter) {
      void* kernel_params[] = { &params };
      Status launch_result;

      if (GettKernel::ccTarget >= 90 && kThreadblockCount % 2 == 0) {
        launch_result = 
            cuda_adapter->launch(dim3(kThreadblockCount),
                                 dim3(2, 1, 1),
                                 dim3(kThreadCount),
                                 smem_size,
                                 stream,
                                 kernel_params,
                                 0);
      } 
      else {
        launch_result = 
            cuda_adapter->launch(dim3(kThreadblockCount),
                                 dim3(kThreadCount),
                                 smem_size,
                                 stream,
                                 kernel_params,
                                 0);
      }
      if (launch_result != Status::kSuccess) {
        CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << static_cast<int>(launch_result));
        return Status::kErrorInternal;
      }
    }
    else {
      if (smem_size > (48 << 10)) {
        auto result = cudaFuncSetAttribute(lookup_cutlass_kernel<GettKernel>(),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
        if (result != cudaSuccess) {
          return handleError(result);
        }
      }

#if __CUDACC_VER_MAJOR__ >= 12
      if (GettKernel::ccTarget >= 90 && kThreadblockCount % 2 == 0) {
        // Launch with CGA options.
        cudaLaunchConfig_t config {
          dim3(kThreadblockCount),
          dim3(kThreadCount),
          size_t(smem_size),
          stream,
          nullptr,
          0
        };
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, (void (*)(typename GettKernel::Params))lookup_cutlass_kernel<GettKernel>(),
                          params);
      } else
#endif
      dispatch_cutlass_kernel<GettKernel>(kThreadblockCount, kThreadCount, smem_size, stream, params);
      return handleError(cudaGetLastError());
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(typename GettKernel::Params const &params) {
    return GettKernel::get_workspace_size_kernel(params);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
