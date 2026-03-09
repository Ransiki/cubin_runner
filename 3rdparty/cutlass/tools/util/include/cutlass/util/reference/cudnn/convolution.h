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
// {$nv-internal-release file}
/*! \file
    \brief CUDNN reference implementation for convolution.
*/

#pragma once

#include "cutlass/conv/convolution.h"
#include "cutlass/coord.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_view.h"

#include <cudnn.h>

namespace cutlass {
namespace reference {
namespace cudnn {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline const char *Error() { return "\033[1;31mError:\033[0m "; }

template <typename T>
inline cudnnDataType_t cuDNN_DataType() {
  return CUDNN_DATA_FLOAT;
}

template <>
inline cudnnDataType_t cuDNN_DataType<cutlass::half_t>() {
  return CUDNN_DATA_HALF;
}

template <>
inline cudnnDataType_t cuDNN_DataType<float>() {
  return CUDNN_DATA_FLOAT;
}

template <typename Tensor>
cudnnStatus_t set_tensor_descriptor(cudnnTensorDescriptor_t desc, Tensor const &tensor) {
  cudnnStatus_t status =
      cudnnSetTensor4dDescriptor(desc,
                                 CUDNN_TENSOR_NHWC,
                                 cuDNN_DataType<typename Tensor::Element>(),  // dataType
                                 tensor.extent(0),                            // N
                                 tensor.extent(3),                            // C
                                 tensor.extent(1),                            // H
                                 tensor.extent(2)                             // W
      );
  if (status != CUDNN_STATUS_SUCCESS) {
    std::cerr << Error() << "cudnnSetTensor4dDescriptor():\n"
              << " size: " << tensor.extent() << "\n"
              << " stride: " << tensor.stride() << "\n";
  }
  return status;
}

template <typename Tensor>
cudnnStatus_t set_filter_descriptor(cudnnFilterDescriptor_t desc, Tensor const &tensor) {
  cudnnStatus_t status =
      cudnnSetFilter4dDescriptor(desc,
                                 cuDNN_DataType<typename Tensor::Element>(),  // dataType
                                 CUDNN_TENSOR_NHWC,
                                 tensor.extent(0),  // K
                                 tensor.extent(3),  // C
                                 tensor.extent(1),  // R
                                 tensor.extent(2)   // S
      );
  return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TensorLayout,
          typename ElementA,
          typename ElementB,
          typename ElementOutput,
          typename ElementCompute,
          cutlass::conv::Operator kOperator>
class CudnnConv {};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Forward Propagation
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TensorLayout,
          typename ElementA,
          typename ElementB,
          typename ElementOutput,
          typename ElementCompute>
class CudnnConv<TensorLayout,
                ElementA,
                ElementB,
                ElementOutput,
                ElementCompute,
                cutlass::conv::Operator::kFprop> 
        : public cutlass::util::reference::Conv {
  

  cutlass::TensorView<ElementA, TensorLayout> tensor_view_A_;
  cutlass::TensorView<ElementB, TensorLayout> tensor_view_B_;
  cutlass::TensorView<ElementOutput, TensorLayout> output_view_;
  ElementCompute alpha_;
  ElementCompute beta_;
  cutlass::Tensor4DCoord padding_;
  cutlass::Coord<2> conv_stride_;
  cutlass::Coord<2> conv_dilation_;
  int groups_;
  cutlass::conv::Mode mode_;

  cudnnHandle_t handle_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t activation_desc_;
  cudnnTensorDescriptor_t output_desc_;

  size_t workspace_size_in_bytes_ = 0;
  cutlass::device_memory::allocation<char> workspace_;

  static cudnnConvolutionFwdAlgo_t const algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  cudnnStatus_t initialize() {

    // Create convolution descriptor object

    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&conv_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateConvolutionDescriptor() - " << status << std::endl;
      return status;
    }

    // Configure convolution operator
    status = cudnnSetConvolution2dDescriptor(
        conv_desc_,
        padding_[0],
        padding_[2],
        conv_stride_[0],
        conv_stride_[1],
        conv_dilation_[0],
        conv_dilation_[1],
        mode_ == cutlass::conv::Mode::kCrossCorrelation ? CUDNN_CROSS_CORRELATION
                                                        : CUDNN_CONVOLUTION,
        cuDNN_DataType<ElementCompute>()  // dataType
    );

    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolution2dDescriptor() - " << status << std::endl;
      return status;
    }

    // set groups
    status = cudnnSetConvolutionGroupCount(conv_desc_, groups_);
    if(status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolutionGroupCount() - " << status << std::endl;
      return status;
    }

    //
    // Initialize tensor descriptors
    //

    // Create filter descriptor object
    status = cudnnCreateFilterDescriptor(&filter_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateFilterDescriptor() - " << status << std::endl;
      return status;
    }

    // Set filter tensor as cuDNN filter descriptor
    status = set_filter_descriptor(filter_desc_, tensor_view_B_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_filter_descriptor() - " << status << std::endl;
      return status;
    }

    // Create activation descriptor object
    status = cudnnCreateTensorDescriptor(&activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateTensorDescriptor(activation_desc) - " << status
                << std::endl;
      return status;
    }

    // Set activation tensor as cuDNN tensor descriptor
    status = set_tensor_descriptor(activation_desc_, tensor_view_A_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_tensor_descriptor(activation) - " << status << std::endl;
      return status;
    }

    // Create output descriptor object
    status = cudnnCreateTensorDescriptor(&output_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateTensorDescriptor(output_desc) - " << status << std::endl;
      return status;
    }

    // Set output tensor as cuDNN output descriptor
    status = set_tensor_descriptor(output_desc_, output_view_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_tensor_descriptor(output) - " << status << std::endl;
      return status;
    }

    //
    // Obtain algorithm
    //

    status = cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolutionMathType - " << status << std::endl;
      return status;
    }

    //
    // Initialize workspace
    //

    status = cudnnGetConvolutionForwardWorkspaceSize(handle_,
                                                     activation_desc_,
                                                     filter_desc_,
                                                     conv_desc_,
                                                     output_desc_,
                                                     algo_,
                                                     &workspace_size_in_bytes_);

    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnGetConvolutionForwardWorkspaceSize - " << status << std::endl;

      std::cerr << " activation: " << activation_desc_ << ", size: (" << tensor_view_A_.extent()
                << ")\n"
                << "     filter: " << filter_desc_ << ", size: (" << tensor_view_B_.extent() << ")\n"
                << "       conv: " << conv_desc_ << "\n"
                << "     output: " << output_desc_ << ", size: (" << output_view_.extent() << ")"
                << std::endl;

      return status;
    }

    workspace_ = cutlass::device_memory::allocation<char>(workspace_size_in_bytes_);

    return CUDNN_STATUS_SUCCESS;
  }

 public:
  CudnnConv(cutlass::TensorView<ElementA, TensorLayout> tensor_view_A,
            cutlass::TensorView<ElementB, TensorLayout> tensor_view_B,
            cutlass::TensorView<ElementOutput, TensorLayout> output_view,
            ElementCompute alpha,
            ElementCompute beta,
            cutlass::Tensor4DCoord padding,
            cutlass::MatrixCoord conv_stride,
            cutlass::MatrixCoord conv_dilation,
            int groups = 1,
            cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation)
      : tensor_view_A_(tensor_view_A),
        tensor_view_B_(tensor_view_B),
        output_view_(output_view),
        alpha_(alpha),
        beta_(beta),
        padding_(padding),
        conv_stride_(conv_stride),
        conv_dilation_(conv_dilation),
        groups_(groups),
        mode_(mode) {
    cudnnStatus_t status_ = cudnnCreate(&handle_);
    if (status_ != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreate() - " << status_ << std::endl;
      return;
    }
    initialize();
  }

  ~CudnnConv() {
    cudnnStatus_t status = cudnnDestroy(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroy() - " << status << std::endl;
    }

    status = cudnnDestroyTensorDescriptor(activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }

    status = cudnnDestroyTensorDescriptor(output_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }

    status = cudnnDestroyFilterDescriptor(filter_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }
  }

  void run() {
    
    float alpha_float = static_cast<float>(alpha_);
    float beta_float = static_cast<float>(beta_);

    cudnnStatus_t status = cudnnConvolutionForward(handle_,
                                                   &alpha_float,
                                                   activation_desc_,
                                                   tensor_view_A_.ref().data(),
                                                   filter_desc_,
                                                   tensor_view_B_.ref().data(),
                                                   conv_desc_,
                                                   algo_,
                                                   workspace_.get(),
                                                   workspace_size_in_bytes_,
                                                   &beta_float,
                                                   output_desc_,
                                                   output_view_.ref().data());

  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dgrad
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TensorLayout,
          typename ElementA,
          typename ElementB,
          typename ElementOutput,
          typename ElementCompute>
class CudnnConv<TensorLayout,
                ElementA,
                ElementB,
                ElementOutput,
                ElementCompute,
                cutlass::conv::Operator::kDgrad> 
        : public cutlass::util::reference::Conv {
  

  cutlass::TensorView<ElementA, TensorLayout> tensor_view_A_;
  cutlass::TensorView<ElementB, TensorLayout> tensor_view_B_;
  cutlass::TensorView<ElementOutput, TensorLayout> output_view_;
  ElementCompute alpha_;
  ElementCompute beta_;
  cutlass::Tensor4DCoord padding_;
  cutlass::Coord<2> conv_stride_;
  cutlass::Coord<2> conv_dilation_;
  int groups_;
  cutlass::conv::Mode mode_;

  cudnnHandle_t handle_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t activation_desc_;
  cudnnTensorDescriptor_t output_desc_;

  size_t workspace_size_in_bytes_ = 0;
  cutlass::device_memory::allocation<char> workspace_;

  static cudnnConvolutionBwdDataAlgo_t const algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  cudnnStatus_t initialize() {

    // Create convolution descriptor object

    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&conv_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateConvolutionDescriptor() - " << status << std::endl;
      return status;
    }

    // Configure convolution operator
    status = cudnnSetConvolution2dDescriptor(
        conv_desc_,
        padding_[0],
        padding_[2],
        conv_stride_[0],
        conv_stride_[1],
        conv_dilation_[0],
        conv_dilation_[1],
        mode_ == cutlass::conv::Mode::kCrossCorrelation ? CUDNN_CROSS_CORRELATION
                                                        : CUDNN_CONVOLUTION,
        cuDNN_DataType<ElementCompute>()  // dataType
    );

    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolution2dDescriptor() - " << status << std::endl;
      return status;
    }

    // set groups
    status = cudnnSetConvolutionGroupCount(conv_desc_, groups_);
    if(status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolutionGroupCount() - " << status << std::endl;
      return status;
    }

    //
    // Initialize tensor descriptors
    //

    // Create filter descriptor object
    status = cudnnCreateFilterDescriptor(&filter_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateFilterDescriptor() - " << status << std::endl;
      return status;
    }

    // Set filter tensor as cuDNN filter descriptor
    status = set_filter_descriptor(filter_desc_, tensor_view_B_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_filter_descriptor() - " << status << std::endl;
      return status;
    }

    // Create activation descriptor object
    status = cudnnCreateTensorDescriptor(&activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateTensorDescriptor(activation_desc) - " << status
                << std::endl;
      return status;
    }

    // Set activation tensor as cuDNN tensor descriptor
    status = set_tensor_descriptor(activation_desc_, output_view_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_tensor_descriptor(activation) - " << status << std::endl;
      return status;
    }

    // Create output descriptor object
    status = cudnnCreateTensorDescriptor(&output_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateTensorDescriptor(output_desc) - " << status << std::endl;
      return status;
    }

    // Set output tensor as cuDNN output descriptor
    status = set_tensor_descriptor(output_desc_, tensor_view_A_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_tensor_descriptor(output) - " << status << std::endl;
      return status;
    }

    //
    // Obtain algorithm
    //

    status = cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolutionMathType - " << status << std::endl;
      return status;
    }

    //
    // Initialize workspace
    //

    status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
                                                     filter_desc_,
                                                     output_desc_,
                                                     conv_desc_,
                                                     activation_desc_,
                                                     algo_,
                                                     &workspace_size_in_bytes_);

    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnGetConvolutionBackwardDataWorkspaceSize - " << status << std::endl;

      std::cerr << " activation: " << activation_desc_ << ", size: (" << output_view_.extent()
                << ")\n"
                << "     filter: " << filter_desc_ << ", size: (" << tensor_view_B_.extent() << ")\n"
                << "       conv: " << conv_desc_ << "\n"
                << "     output: " << output_desc_ << ", size: (" << tensor_view_A_.extent() << ")"
                << std::endl;

      return status;
    }

    workspace_ = cutlass::device_memory::allocation<char>(workspace_size_in_bytes_);

    return CUDNN_STATUS_SUCCESS;
  }

 public:
  CudnnConv(cutlass::TensorView<ElementA, TensorLayout> tensor_view_A,
            cutlass::TensorView<ElementB, TensorLayout> tensor_view_B,
            cutlass::TensorView<ElementOutput, TensorLayout> output_view,
            ElementCompute alpha,
            ElementCompute beta,
            cutlass::Tensor4DCoord padding,
            cutlass::MatrixCoord conv_stride,
            cutlass::MatrixCoord conv_dilation,
            int groups = 1,
            cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation)
      : tensor_view_A_(tensor_view_A),
        tensor_view_B_(tensor_view_B),
        output_view_(output_view),
        alpha_(alpha),
        beta_(beta),
        padding_(padding),
        conv_stride_(conv_stride),
        conv_dilation_(conv_dilation),
        groups_(groups),
        mode_(mode) {
    cudnnStatus_t status_ = cudnnCreate(&handle_);
    if (status_ != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreate() - " << status_ << std::endl;
      return;
    }

    initialize();
  }

  ~CudnnConv() {
    cudnnStatus_t status = cudnnDestroy(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroy() - " << status << std::endl;
    }

    status = cudnnDestroyTensorDescriptor(activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }

    status = cudnnDestroyTensorDescriptor(output_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }

    status = cudnnDestroyFilterDescriptor(filter_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }
  }

  void run() {
      cudnnStatus_t status = cudnnConvolutionBackwardData(handle_,
                                                     &alpha_,
                                                     filter_desc_,
                                                     tensor_view_B_.ref().data(),
                                                     output_desc_,
                                                     tensor_view_A_.ref().data(),
                                                     conv_desc_,
                                                     algo_,
                                                     workspace_.get(),
                                                     workspace_size_in_bytes_,
                                                     &beta_,
                                                     activation_desc_,
                                                     output_view_.ref().data());

  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Wgrad
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TensorLayout,
          typename ElementA,
          typename ElementB,
          typename ElementOutput,
          typename ElementCompute>
class CudnnConv<TensorLayout,
                ElementA,
                ElementB,
                ElementOutput,
                ElementCompute,
                cutlass::conv::Operator::kWgrad> 
        : public cutlass::util::reference::Conv {
  

  cutlass::TensorView<ElementA, TensorLayout> tensor_view_A_;
  cutlass::TensorView<ElementB, TensorLayout> tensor_view_B_;
  cutlass::TensorView<ElementOutput, TensorLayout> output_view_;
  ElementCompute alpha_;
  ElementCompute beta_;
  cutlass::Tensor4DCoord padding_;
  cutlass::Coord<2> conv_stride_;
  cutlass::Coord<2> conv_dilation_;
  int groups_;
  cutlass::conv::Mode mode_;

  cudnnHandle_t handle_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t activation_desc_;
  cudnnTensorDescriptor_t output_desc_;

  size_t workspace_size_in_bytes_ = 0;
  cutlass::device_memory::allocation<char> workspace_;

  static cudnnConvolutionBwdFilterAlgo_t const algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

  cudnnStatus_t initialize() {

    // Create convolution descriptor object

    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&conv_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateConvolutionDescriptor() - " << status << std::endl;
      return status;
    }

    // Configure convolution operator
    status = cudnnSetConvolution2dDescriptor(
        conv_desc_,
        padding_[0],
        padding_[2],
        conv_stride_[0],
        conv_stride_[1],
        conv_dilation_[0],
        conv_dilation_[1],
        mode_ == cutlass::conv::Mode::kCrossCorrelation ? CUDNN_CROSS_CORRELATION
                                                        : CUDNN_CONVOLUTION,
        cuDNN_DataType<ElementCompute>()  // dataType
    );

    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolution2dDescriptor() - " << status << std::endl;
      return status;
    }

    // set groups
    status = cudnnSetConvolutionGroupCount(conv_desc_, groups_);
    if(status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolutionGroupCount() - " << status << std::endl;
      return status;
    }

    //
    // Initialize tensor descriptors
    //

    // Create filter descriptor object
    status = cudnnCreateFilterDescriptor(&filter_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateFilterDescriptor() - " << status << std::endl;
      return status;
    }

    // Set filter tensor as cuDNN filter descriptor
    status = set_filter_descriptor(filter_desc_, output_view_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_filter_descriptor() - " << status << std::endl;
      return status;
    }

    // Create activation descriptor object
    status = cudnnCreateTensorDescriptor(&activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateTensorDescriptor(activation_desc) - " << status
                << std::endl;
      return status;
    }

    // Set activation tensor as cuDNN tensor descriptor
    status = set_tensor_descriptor(activation_desc_, tensor_view_B_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_tensor_descriptor(activation) - " << status << std::endl;
      return status;
    }

    // Create output descriptor object
    status = cudnnCreateTensorDescriptor(&output_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreateTensorDescriptor(output_desc) - " << status << std::endl;
      return status;
    }

    // Set output tensor as cuDNN output descriptor
    status = set_tensor_descriptor(output_desc_, tensor_view_A_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "set_tensor_descriptor(output) - " << status << std::endl;
      return status;
    }

    //
    // Obtain algorithm
    //

    status = cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnSetConvolutionMathType - " << status << std::endl;
      return status;
    }

    //
    // Initialize workspace
    //

    status = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
                                                     activation_desc_,
                                                     output_desc_,
                                                     conv_desc_,
                                                     filter_desc_,
                                                     algo_,
                                                     &workspace_size_in_bytes_);

    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnGetConvolutionForwardWorkspaceSize - " << status << std::endl;

      std::cerr << " activation: " << activation_desc_ << ", size: (" << tensor_view_B_.extent()
                << ")\n"
                << "     filter: " << filter_desc_ << ", size: (" << output_view_.extent() << ")\n"
                << "       conv: " << conv_desc_ << "\n"
                << "     output: " << output_desc_ << ", size: (" << tensor_view_A_.extent() << ")"
                << std::endl;

      return status;
    }

    workspace_ = cutlass::device_memory::allocation<char>(workspace_size_in_bytes_);

    return CUDNN_STATUS_SUCCESS;
  }

 public:
  CudnnConv(cutlass::TensorView<ElementA, TensorLayout> tensor_view_A,
            cutlass::TensorView<ElementB, TensorLayout> tensor_view_B,
            cutlass::TensorView<ElementOutput, TensorLayout> output_view,
            ElementCompute alpha,
            ElementCompute beta,
            cutlass::Tensor4DCoord padding,
            cutlass::MatrixCoord conv_stride,
            cutlass::MatrixCoord conv_dilation,
            int groups = 1,
            cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation)
      : tensor_view_A_(tensor_view_A),
        tensor_view_B_(tensor_view_B),
        output_view_(output_view),
        alpha_(alpha),
        beta_(beta),
        padding_(padding),
        conv_stride_(conv_stride),
        conv_dilation_(conv_dilation),
        groups_(groups),
        mode_(mode) {

    cudnnStatus_t status_ = cudnnCreate(&handle_);
    if (status_ != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnCreate() - " << status_ << std::endl;
      return;
    }
    initialize();
  }

  ~CudnnConv() {
    cudnnStatus_t status = cudnnDestroy(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroy() - " << status << std::endl;
    }

    status = cudnnDestroyTensorDescriptor(activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }

    status = cudnnDestroyTensorDescriptor(output_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }

    status = cudnnDestroyFilterDescriptor(filter_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
      std::cerr << Error() << "cudnnDestroyTensorDescriptor() - " << status << std::endl;
    }
  }

  void run() {
      cudnnStatus_t status = cudnnConvolutionBackwardFilter(handle_,
                                                     &alpha_,
                                                     activation_desc_,
                                                     tensor_view_B_.ref().data(),
                                                     output_desc_,
                                                     tensor_view_A_.ref().data(),
                                                     conv_desc_,
                                                     algo_,
                                                     workspace_.get(),
                                                     workspace_size_in_bytes_,
                                                     &beta_,
                                                     filter_desc_,
                                                     output_view_.ref().data());

  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace cudnn
}  // namespace reference
}  // namespace cutlass
////////////////////////////////////////////////////////////////////////////////////////////////////
