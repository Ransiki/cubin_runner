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
    \brief Emulation of CUDA runtime for TMA descriptors
  
  This file emulates setting TMA descriptors as described in the xlsx file below // {$nv-release-never}
    https://p4viewer.nvidia.com/get///hw/doc/gpu/hopper/hopper/design/Functional_Descriptions/Hopper-189-190_TMA_tensor_descriptor.xlsx // {$nv-release-never}

*/
#pragma once
#include "cuda_runtime.h"

namespace cutlass {
namespace arch {
namespace emu {
namespace cuda_rt {

template <
  int Start,      /// Starting bit in the descriptor
  int Width,      /// Bitwidth per item
  int N = 1       /// Number of items
  >
struct ParamBitInfo {
  // start bit within 512b descriptor
  static int const kStart = Start;

  // bit width of parameter
  static int const kWidth = Width;

  // number of maximum elements of parameter type
  static int const kElements = N;

  // bit mask per element of bitwidth = kWidith
  static uint64_t const kBitMask = ((1ull << kWidth) - 1);

  // start idx within 8x64b
  static int const kStartIdx = kStart / 64;

  // start bit within 64b
  static int const kStartBit = kStart % 64;
};

struct TmaDescriptorBitInfo {
  /// Storage type
  using Storage = uint64_t;

  /// Number of total storage bits
  static size_t const kTotalBits = 512;

  static size_t const kStroageBits = sizeof_bits<Storage>::value;

  /// Number of Storage elements
  static size_t const kStorageElements = kTotalBits / kStroageBits;

  ///
  /// Descriptor params info
  ///
  using Zeros = ParamBitInfo<0, 4>;
  using TensorGlobalAddress = ParamBitInfo<4, 53>;
  // reserved bits
  using Type = ParamBitInfo<64, 1>;
  using Version = ParamBitInfo<65, 3>;
  using Dimensionality = ParamBitInfo<68, 3>;
  using Format = ParamBitInfo<71, 4>;
  using Interleave = ParamBitInfo<75, 2>;
  using SmemSwizzle = ParamBitInfo<77, 2>;
  using OobFillMode = ParamBitInfo<79, 1>;
  using F32toTF32 = ParamBitInfo<80, 1>;
  using L2sectorPromotion = ParamBitInfo<81, 2>;
  // reserved bits
  using TensorStride = ParamBitInfo<96, 36, 4>;
  // reserved bits
  using TensorSize = ParamBitInfo<256, 32, 5>;
  using TraversalStride = ParamBitInfo<416, 3, 5>;
  // reserved bits
  using BoxSize = ParamBitInfo<440, 8, 5>;
};

struct cudaTmaDescriptor {  

  //
  // Data members
  //

  // 512b of internal storage
  uint64_t data[8];

  //
  // Methods
  //

  // ctor
  cudaTmaDescriptor() {
    // initialize all bits to zero
    for (int i = 0; i < 8; i++) {
      data[i] = 0;
    }
  }

  // Set bits within 64b boundaries
  void set_bits(int idx, int start_bit, uint64_t mask, uint64_t bits) {
    data[idx] = (data[idx] & ~(mask << start_bit)) | ((bits & mask) << start_bit);
  }

  // Set bits for scalar (N=1) descriptor params 
  // (all scalar params stay within 64b boundary)  
  template <typename Type>
  void set_param(uint64_t const bits) {
    
    set_bits(
      Type::kStartIdx, 
      Type::kStartBit, 
      Type::kBitMask, 
      bits);
  }

  // Set bits for array (N>1) descriptor params 
  // (bits[i] might go across 64b boundary)  
  template <typename Type>
  void set_param_vector(uint64_t const *bits, int num_items) {

    for (int i = 0; i < num_items; i++) {
      // start idx within 8x64b
      int start_idx = 
        (Type::kStart + i * Type::kWidth) / 64;

      // start bit within 64b
      int start_bit = 
        (Type::kStart + i * Type::kWidth) % 64;

      // end bit may go across 64b boundary
      int end_bit = start_bit + Type::kWidth;

      set_bits(
        start_idx, 
        start_bit, 
        Type::kBitMask, 
        bits[i]);
    
      // tensor_stride split accross 64b uint64_t internal storage
      if (end_bit >= 64) {

        int right_shift = 64 - start_bit;

        set_bits(
          start_idx + 1, 
          0, 
          (Type::kBitMask >> right_shift), 
          (bits[i] >> right_shift));
      }

    }

  }

  // Debug print the TmaDescriptor 
  void print() {
    std::cout << "uint64_t data[8] = {\n";
    for (int i = 0; i < 8; i++) {
      std::cout << std::hex << " data[" << i << "] = " 
                << "0x" << data[i] << ((i < 7) ? ",\n " : " }") ;
    }
    std::cout << std::endl;
  }
};

// Set Tma descriptor bits in device memory pointed by device_ptr
cudaError_t cudaSetTmaDescriptor(
    void *device_ptr,
    void const *global_addr,
    TMAOperation::Type const desc_type,
    int const rank,
    TMAOperation::Format const format,
    TMAOperation::Interleave const interleave,
    TMAOperation::SmemSwizzle const smem_swizzle,
    TMAOperation::OobFillMode const oob_fill_mode,
    TMAOperation::F32toTF32 const f32_to_tf32,
    TMAOperation::L2sectorPromotion const l2_sector_promotion,
    uint64_t const *tensor_stride,
    uint32_t const *tensor_size,
    uint8_t const *traversal_stride,
    uint8_t const *box_size,
    int version = 1
  ) {

  cudaTmaDescriptor desc;

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Set scalar parameters
  //////////////////////////////////////////////////////////////////////////////////////////////////////  
  // 4 LSBs are not specified since address is 16B aligned
  uint64_t addr_bits = reinterpret_cast<uint64_t>(global_addr) >> 4;
  desc.set_param<TmaDescriptorBitInfo::TensorGlobalAddress>(addr_bits);
  desc.set_param<TmaDescriptorBitInfo::Type>(static_cast<uint64_t>(desc_type));
  desc.set_param<TmaDescriptorBitInfo::Version>(static_cast<uint64_t>(version));
  desc.set_param<TmaDescriptorBitInfo::Dimensionality>(static_cast<uint64_t>(rank - 1));
  desc.set_param<TmaDescriptorBitInfo::Format>(static_cast<uint64_t>(format));
  desc.set_param<TmaDescriptorBitInfo::Interleave>(static_cast<uint64_t>(interleave));
  desc.set_param<TmaDescriptorBitInfo::SmemSwizzle>(static_cast<uint64_t>(smem_swizzle));
  desc.set_param<TmaDescriptorBitInfo::OobFillMode>(static_cast<uint64_t>(oob_fill_mode));  
  desc.set_param<TmaDescriptorBitInfo::F32toTF32>(static_cast<uint64_t>(f32_to_tf32));
  desc.set_param<TmaDescriptorBitInfo::L2sectorPromotion>(static_cast<uint64_t>(l2_sector_promotion));
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Set vector parameters (vector size based on rank)
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  // set tensor strides of length rank - 1
  int num_strides = rank - 1;
  std::vector<uint64_t> tensor_stride_bits(num_strides, 0);
  for (int i = 0; i < num_strides; i++) {
    // 4 LSBs are not specified since strides are 16B aligned
    tensor_stride_bits[i] = tensor_stride[i] >> 4;
  }
  desc.set_param_vector<TmaDescriptorBitInfo::TensorStride>(tensor_stride_bits.data(), num_strides);

  // set tensor size of length rank
  std::vector<uint64_t> tensor_size_bits(rank, 0);
  for (int i = 0; i < rank; i++) {
    // values range [1:2^32]. Thus subtract `1` to map to [0:2^32 - 1]
    tensor_size_bits[i] = static_cast<uint64_t>(tensor_size[i] - 1);
  }
  desc.set_param_vector<TmaDescriptorBitInfo::TensorSize>(tensor_size_bits.data(), rank);

  // set traversal stride of length rank
  std::vector<uint64_t> traversal_stride_bits(rank, 0);
  for (int i = 0; i < rank; i++) {
    traversal_stride_bits[i] = static_cast<uint64_t>(traversal_stride[i]);
  }
  desc.set_param_vector<TmaDescriptorBitInfo::TraversalStride>(traversal_stride_bits.data(), rank);

  // set box size of length rank
  std::vector<uint64_t> box_size_bits(rank, 0);
  for (int i = 0; i < rank; i++) {
    // values range [1:256]. Thus subtract `1` to map to [0:255]
    box_size_bits[i] = static_cast<uint64_t>(box_size[i] - 1);
  }
  desc.set_param_vector<TmaDescriptorBitInfo::BoxSize>(box_size_bits.data(), rank);

  // debug print tma descriptor
  // desc.print();

  /// Copy Tma bits to user provided device allocation
  cudaError_t status = cudaMemcpy(device_ptr, 
                                  &desc, 
                                  sizeof(cudaTmaDescriptor), 
                                  cudaMemcpyHostToDevice);
  
  if (status != cudaSuccess) {
    std::cerr << "Failed create/set cudaTmaDescriptor "
      << cudaGetErrorString(status) << std::endl;

    return status;
  }

  return cudaSuccess;
}

} // namespace cuda_rt
} // namespace emu
} // namespace arch
} // namespace cutlass



