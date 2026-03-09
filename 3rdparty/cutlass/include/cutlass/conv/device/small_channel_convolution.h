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
   \brief Template for device-level Small Channel Convolution
*/

// {$nv-internal-release file}

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename SmallChannelConvolutionKernel_>
class SmallChannelConvolution {
public:
  using UnderlyingKernel = GetUnderlyingKernel_t<SmallChannelConvolutionKernel_>;

  using ElementA = typename UnderlyingKernel::ElementA;
  using LayoutA = typename UnderlyingKernel::LayoutA;
  using ElementB = typename UnderlyingKernel::ElementB;
  using LayoutB = typename UnderlyingKernel::LayoutB;
  using ElementC = typename UnderlyingKernel::ElementC;
  using LayoutC = typename UnderlyingKernel::LayoutC;
  using ElementAccumulator = typename UnderlyingKernel::ElementAccumulator;
  using ElementCompute = typename UnderlyingKernel::ElementCompute;

  using ArchTag = typename UnderlyingKernel::ArchTag;
  using InstructionShape = typename UnderlyingKernel::InstructionShape;
  using Conv2dFilterShape = typename UnderlyingKernel::Conv2dFilterShape;

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  static cutlass::conv::Operator const kConvolutionalOperator = UnderlyingKernel::kConvolutionalOperator;
  static constexpr uint32_t WarpsPerCTA = UnderlyingKernel::WarpsPerCTA;

  /// Argument structure
  using Arguments = typename UnderlyingKernel::Arguments;

private:

  /// Kernel parameters object
  typename UnderlyingKernel::Params params_;

public:

  /// Constructs Small Channel Convolution
  SmallChannelConvolution() { }

  /// Determines whether the Small Channel Convolution can execute the given problem.
  static Status can_implement(Arguments const &args) {
    constexpr int kFltK = Conv2dFilterShape::kN;
    constexpr int kFltR = Conv2dFilterShape::kH;
    constexpr int kFltS = Conv2dFilterShape::kW;
    constexpr int kFltC = Conv2dFilterShape::kC;
    if constexpr (kConvolutionalOperator == conv::Operator::kWgrad && kFltC != 4) {
      return Status::kErrorInvalidProblem;
    }
    if constexpr (kFltC != 4 && kFltC != 8) {
      return Status::kErrorInvalidProblem;
    }
    if constexpr (kFltK != 16 && kFltK != 32 && kFltK != 64 && !(kFltK == 48 && kConvolutionalOperator == conv::Operator::kFprop)) {
      return Status::kErrorInvalidProblem;
    }
    if constexpr (kFltR != kFltS || (kFltR != 3 && kFltR != 5 && kFltR != 7)) {
      return Status::kErrorInvalidProblem;
    }
    auto check_tensor_maximum_size = [](auto& tensor_shape, auto element_size) {
      auto tensor_size = static_cast<int64_t>(tensor_shape.n()) * static_cast<int64_t>(tensor_shape.h())
          * static_cast<int64_t>(tensor_shape.w()) * static_cast<int64_t>(tensor_shape.c()) * element_size;
      if (tensor_size >= (1ull << 31)) {
        return false;
      }
      return true;
    };
    if (!check_tensor_maximum_size(args.input_tensor_size, sizeof(ElementA)) ||
        !check_tensor_maximum_size(args.conv_filter_size, kConvolutionalOperator == conv::Operator::kFprop ? sizeof(ElementB) : sizeof(ElementC)) ||
        !check_tensor_maximum_size(args.output_tensor_size, kConvolutionalOperator == conv::Operator::kFprop ? sizeof(ElementC) : sizeof(ElementB))) {
      return Status::kErrorInvalidProblem;
    }
    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    if constexpr (kConvolutionalOperator == conv::Operator::kWgrad) {
      constexpr int kFltK = Conv2dFilterShape::kN;
      constexpr int kFltR = Conv2dFilterShape::kH;
      constexpr int kFltS = Conv2dFilterShape::kW;
      // 16 bytes alignment for reduction buffer stg
      constexpr int alignment = 16;
      constexpr int kFltKPerCTA = kFltK;
      constexpr int kCtasPerK = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;

      int kSplitsInP = cutlass::platform::max(args.split_k.slices, 1);
      int kSplitsInQ = (args.output_tensor_size.w() + 31) / 32 * kCtasPerK;
      int kSplitsInN = args.output_tensor_size.n();
      int kTotalSplits = kSplitsInP * kSplitsInQ * kSplitsInN;
      auto find_log2 = [](int x) {
        int clz = 32;
        for (int i = 31; i >= 0; --i) {
          if ((1 << i) & x) {
            clz = 31 - i;
            break;
          }
        }
        int result = 31- clz;
        result += (x & (x - 1)) != 0; // Roundup, add 1 if not a power of 2.
        return result;
      };
      int kNumLocks = args.split_k.buffers > 0 ? args.split_k.buffers
          : (1 << ((find_log2(kTotalSplits) + 1) >> 1));

      size_t kLocksSize = ((kNumLocks + 1) * kCtasPerK * kFltR * sizeof(int) + alignment - 1)
        / alignment * alignment;
      size_t kReductionBufSize =
        kFltR * kFltS * 4 * kFltKPerCTA * kCtasPerK * kNumLocks * sizeof(uint16_t);

      return kLocksSize + kReductionBufSize;
    } else { // kFprop
      return 0;
    }
  }

  /// Initializes Small Channel Convolution state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("SmallChannelConvolution::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    if constexpr (kConvolutionalOperator == conv::Operator::kWgrad) {
      size_t workspace_bytes = get_workspace_size(args);
      CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);
      if (workspace_bytes) {
        if (!workspace) {
          CUTLASS_TRACE_HOST("  error: device workspace must not be null");
          return Status::kErrorWorkspaceNull;
        }
        CUTLASS_TRACE_HOST("  clearing device workspace");
        cudaError_t result = cudaMemsetAsync(workspace, 0, workspace_bytes, stream);
        if (result != cudaSuccess) {
          result = cudaGetLastError(); // to clear the error bit
          CUTLASS_TRACE_HOST("Failed to clear locks and reduction buffer: " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }

    // Initialize the Params structure from the arguments
    params_ = typename UnderlyingKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );

    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      // account for dynamic smem capacity if needed
      int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      if (smem_size >= (48 << 10)) {
        cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<UnderlyingKernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

        if (cudaSuccess != result) {
          result = cudaGetLastError(); // to clear the error bit
          CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }
    return Status::kSuccess;
  }

  /// Initializes Small Channel Convolution state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {
    CUTLASS_TRACE_HOST("SmallChannelConvolution::update() - workspace: " << workspace);
    params_.input_tensor_size = args.input_tensor_size;
    params_.conv_filter_size = args.conv_filter_size;
    params_.output_tensor_size = args.output_tensor_size;
    params_.ref_A = args.ref_A;
    params_.ref_B = args.ref_B;
    params_.ref_C = args.ref_C;
    params_.kPadTop = (Conv2dFilterShape::kH - 1) / 2;
    params_.kPadLeft = (Conv2dFilterShape::kW - 1) / 2;

    if constexpr (kConvolutionalOperator == conv::Operator::kFprop) {
      params_.bias_tensor_size = args.bias_tensor_size;
      params_.ref_bias = args.ref_bias;
      params_.kAlpha = args.kAlpha;
      params_.kLowerBound = args.kLowerBound;
      params_.kUpperBound = args.kUpperBound;
    }

    if constexpr (kConvolutionalOperator == conv::Operator::kWgrad) {
      constexpr int kFltK = Conv2dFilterShape::kN;
      constexpr int kFltR = Conv2dFilterShape::kH;
      constexpr int kFltS = Conv2dFilterShape::kW;
      // 16 bytes alignment for reduction buffer stg
      constexpr int alignment = 16;
      constexpr int kFltKPerCTA = kFltK;
      constexpr int kCtasPerK = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;

      int kSplitsInP = cutlass::platform::max(args.split_k.slices, 1);
      int kSplitsInQ = (args.output_tensor_size.w() + 31) / 32 * kCtasPerK;
      int kSplitsInN = args.output_tensor_size.n();
      int kTotalSplits = kSplitsInP * kSplitsInQ * kSplitsInN;
      auto find_log2 = [](int x) {
        int clz = 32;
        for (int i = 31; i >= 0; --i) {
          if ((1 << i) & x) {
            clz = 31 - i;
            break;
          }
        }
        int result = 31- clz;
        result += (x & (x - 1)) != 0; // Roundup, add 1 if not a power of 2.
        return result;
      };
      params_.kNumLocks = args.split_k.buffers > 0 ? args.split_k.buffers
          : (1 << ((find_log2(kTotalSplits) + 1) >> 1));

      size_t kLocksSize = ((params_.kNumLocks + 1) * kCtasPerK * kFltR * sizeof(int) + alignment - 1)
        / alignment * alignment;
      size_t kReductionBufSize =
        kFltR * kFltS * 4 * kFltKPerCTA * kCtasPerK * params_.kNumLocks * sizeof(uint16_t);
      size_t workspace_bytes = get_workspace_size(args);
      if (workspace_bytes > 0) {
        if (nullptr == workspace) {
          return Status::kErrorWorkspaceNull;
        }
        params_.gmem_locks = reinterpret_cast<int *>(workspace);
        params_.gmem_retired_ctas = &params_.gmem_locks[params_.kNumLocks * kCtasPerK * kFltR];
        params_.gmem_red = reinterpret_cast<uint16_t *>((char *)workspace + kLocksSize);
      }
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    CUTLASS_TRACE_HOST("SmallChannelConvolution::run()");
    dim3 grid = UnderlyingKernel::get_grid_shape(params_);
    dim3 block(32 * WarpsPerCTA, 1, 1);

    // configure smem size and carveout
    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
    Status launch_result = Status::kSuccess;

    if constexpr (kEnableCudaHostAdapter) {
      //
      // Use the cuda host adapter
      //
      CUTLASS_ASSERT(cuda_adapter);
      if (cuda_adapter) {
        void* kernel_params[] = {&params_};

        launch_result = cuda_adapter->launch(
            grid, block, smem_size, stream, kernel_params, 0
            );
      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      CUTLASS_ASSERT(cuda_adapter == nullptr);
      cutlass::arch::synclog_setup();
      cutlass::device_kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);
    }
    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
