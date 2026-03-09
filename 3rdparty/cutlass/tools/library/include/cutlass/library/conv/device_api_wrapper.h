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
/* \file
   \brief Defines a simple function schema
*/
#pragma once

#include "cutlass/conv/device/implicit_gemm.h"
#include "cutlass/ir/function.h"


namespace cutlass {
namespace library {
namespace conv {

/// FunctionType enclosing Argument structure
template <
  /// Element type for Activation tensor operand
  typename ElementActivation_,
  /// Layout type for Activation tensor operand
  typename LayoutActivation_,
  /// Element type for Filter tensor operand
  typename ElementFilter_,
  /// Layout type for Filter tensor operand
  typename LayoutFilter_,
  /// Element type for Output tensor operand
  typename ElementOutput_,
  /// Layout type for Output matrix operands
  typename LayoutOutput_,
  /// Epilogue compute operator
  typename EpilogueComputeType_
  >
class FunctionType : public ir::DeviceFunction {

  //
  // Type definitions
  //
  using ElementActivation = ElementActivation_;
  using LayoutActivation = LayoutActivation_;
  using ElementFilter = ElementFilter_;
  using LayoutFilter = LayoutFilter_;
  using ElementOutput = ElementOutput_;
  using LayoutOutput = LayoutOutput_;
  using EpilogueComputeType = EpilogueComputeType_;

public:
  /// Parameterized constructor
  FunctionType(ir::Function *function) : DeviceFunction(function) {}

  /// Virtual destructor
  ~FunctionType() {
    ir::DeviceFunction::~DeviceFunction();
  }

public:
  /// Argument struct
  struct Arguments {
    cutlass::Tensor4DCoord                          input_tensor_size;
    cutlass::Tensor4DCoord                          filter_tensor_size;
    TensorRef<ElementActivation, LayoutActivation>  ref_Activation;
    TensorRef<ElementFilter, LayoutFilter>          ref_Filter;
    TensorRef<ElementOutput, LayoutOutput>          ref_Output;
    EpilogueComputeType                             alpha;
    EpilogueComputeType                             beta;
    int                                             split_count;
  };

};

/// FunctionWrapper enclosing cutlass::gemm::device::ImplicitGemm
template <
  typename ImplicitGemm_
>
class FunctionWrapper : public FunctionType<  
  typename ImplicitGemm_::ElementA, 
  typename layout::TensorNHWC, 
  typename ImplicitGemm_::ElementB, 
  typename layout::TensorNHWC, 
  typename ImplicitGemm_::ElementOutput,
  typename layout::TensorNHWC, 
  typename ImplicitGemm_::ElementCompute> {

  //
  // Type definitions
  //

  using ImplicitGemm = ImplicitGemm_;
  
  // FunctionWrapper Arguments
  using FunctionType = FunctionType<
                        typename ImplicitGemm::ElementA, 
                        typename layout::TensorNHWC, 
                        typename ImplicitGemm::ElementB, 
                        typename layout::TensorNHWC, 
                        typename ImplicitGemm::ElementOutput, 
                        typename layout::TensorNHWC, 
                        typename ImplicitGemm::ElementCompute
                        >;
  using Arguments = typename FunctionType::Arguments;

public:
  /// Parameterized constructor
  FunctionWrapper(ir::Function *function) : FunctionType(function) {}

  /// Destructor
  ~FunctionWrapper() {}

  /// returns true if a device function can implement the solution to the given problem.
  virtual Status can_implement(void const *arguments) const {

    return Status::kSuccess;
  }

  
  /// Gets the host-side workspace size in bytes given a pointer to the argument structure
  virtual Status get_host_workspace_size(size_t *size, void const *arguments) const {
    *size = sizeof(ImplicitGemm);
    return Status::kSuccess;
  }

  
  /// Gets the device-side workspace size in bytes given a pointer to the argument structure
  virtual Status get_device_workspace_size(size_t *size, void const *arguments) const {
    
    Arguments const *args = reinterpret_cast<Arguments const *>(arguments);
    
    //
    // Get device workspace for the cutlass::gemm::device::ImplicitGemm[Fprop|Dgrad|Wgrad] operator
    //
    typename ImplicitGemm::Arguments implicit_gemm_arguments{
      args->input_tensor_size,
      args->filter_tensor_size,
      args->ref_Activation,
      args->ref_Filter,
      args->ref_Output,
      {args->alpha, args->beta},
      args->split_count
    };

    *size = ImplicitGemm::get_workspace_size(implicit_gemm_arguments); 

    return Status::kSuccess;
  }

  
  /// Initializes the host-side workspace given the kernel argument structure
  virtual Status initialize_workspace(void const *arguments, void *host_workspace, void *device_workspace) const {
    
    Arguments const *args = reinterpret_cast<Arguments const *>(arguments);

    //
    // Get device workspace for the cutlass::gemm::device::ImplicitGemm[Fprop|Dgrad|Wgrad] operator
    //
    typename ImplicitGemm::Arguments implicit_gemm_arguments{
      args->input_tensor_size,
      args->filter_tensor_size,
      args->ref_Activation,
      args->ref_Filter,
      args->ref_Output,
      {args->alpha, args->beta},
      args->split_count
    };

    // Placement new operator (https://en.cppreference.com/w/cpp/language/new#Placement_new)
    ImplicitGemm *implicit_gemm_op = new (host_workspace) ImplicitGemm;
    
    return implicit_gemm_op->initialize(implicit_gemm_arguments, device_workspace);
  }

  /// Updates the workspace
  virtual Status update_workspace(void const *arguments, void *host_workspace, void *device_workspace) const {

    return initialize_workspace(arguments, host_workspace, device_workspace);
  
  }
  
  // TODO: add cudastream_t to ir::DeviceFunction::run() along with cutlass::conv::device::ImplicitGemm
  // pass cudastrea_t starting from profiler to enable profiler run test on any stream
  /// Runs the kernel
  virtual Status run(
    void const *arguments, 
    void *host_workspace, 
    void *device_workspace) const {

    ImplicitGemm *implicit_gemm_op = static_cast<ImplicitGemm *>(host_workspace);

    return implicit_gemm_op->run();
  }
  
};


///////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace library
} // namespace cutlass
