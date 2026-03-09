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

/*!
  \file This file contains extra information that reflects implementation details of a kernel.
*/

#pragma once

#include <array>
#include <cuda.h>

#include <cutlass/gemm_coord.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

using Coord = cutlass::gemm::GemmCoord;

struct LaunchConfigurationPayload {
  Coord     block_dim;              // Thread configuration within each thread-block
  int32_t   dynamic_shared_memory;  // Dynamic shared memory required for launching the kernel
};

// The copy engine used for tensor's memory access.
enum class CopyEngineID {
  kDereference = 0,           // Conventional pointer dereference
  kCopyAsync,                 // Asynchronous Copy
  kTma,                       // Tensor Memory Accelerator unit
  kInvalid
};

struct TmaOperandPayload {
  // Box size
  std::array<int32_t, 5>    box_size;

  // Box stride
  std::array<int32_t, 5>    box_stride;

  // Basis permutation function
  std::array<int32_t, 5>    basis_permutation;
  
#if defined (CUDA_VERSION) && CUDA_VERSION >= 12000
  // L2 Sector promotion handling
  CUtensorMapL2promotion    l2_promotion;

  // Out of bounds memory access fill policy
  CUtensorMapFloatOOBfill   oob_fill;

  // Shared Memory swizzled layout enumerant
  CUtensorMapSwizzle        smem_swizzle;
#else
  // L2 Sector promotion handling
  int32_t                   l2_promotion;

  // Out of bounds memory access fill policy
  int32_t                   oob_fill;

  // Shared Memory swizzled layout enumerant
  int32_t                   smem_swizzle;
#endif

  TmaOperandPayload() {}

#if defined (CUDA_VERSION) && CUDA_VERSION >= 12000
  TmaOperandPayload(
    std::array<int32_t, 5>    box_size,
    std::array<int32_t, 5>    box_stride,
    std::array<int32_t, 5>    basis_permutation,
    CUtensorMapL2promotion    l2_promotion,
    CUtensorMapFloatOOBfill   oob_fill,
    CUtensorMapSwizzle        smem_swizzle) :
    box_size(box_size),
    box_stride(box_stride),
    basis_permutation(basis_permutation),
    l2_promotion(l2_promotion),
    oob_fill(oob_fill),
    smem_swizzle(smem_swizzle) {}
#else
  TmaOperandPayload(
    std::array<int32_t, 5>    box_size,
    std::array<int32_t, 5>    box_stride,
    std::array<int32_t, 5>    basis_permutation,
    int32_t                   l2_promotion,
    int32_t                   oob_fill,
    int32_t                   smem_swizzle) :
    box_size(box_size),
    box_stride(box_stride),
    basis_permutation(basis_permutation),
    l2_promotion(l2_promotion),
    oob_fill(oob_fill),
    smem_swizzle(smem_swizzle) {}
#endif
};

struct TensorOperandPayload {
  CopyEngineID copy_engine;   // Copy engine used for the tensor
  TmaOperandPayload tma;

  TensorOperandPayload() : copy_engine(CopyEngineID::kInvalid) {}
  TensorOperandPayload(CopyEngineID cp_engine) : copy_engine(cp_engine) {}
  TensorOperandPayload(
    CopyEngineID cp_engine,
    TmaOperandPayload tma_payload) : copy_engine(cp_engine), tma(tma_payload) {}
  TensorOperandPayload(TmaOperandPayload tma_payload) : copy_engine(CopyEngineID::kTma), tma(tma_payload) {}
};


enum class MainloopScheduleType {
  kMainloopScheduleAuto = 0,        // Automatic procedure for deriving the best scheduling strategy
  kTmaWarpSpecialized1SmSm100 = 1,  // Kernel schedule type specific for .1CTA kernels.
  kTmaWarpSpecialized2SmSm100 = 2,  // Kernel schedule type specific for .2CTA kernels.
};

enum class EpilogueScheduleType {
  kEpilogueScheduleAuto = 0,        // Automatic procedure for deriving the best scheduling startegy
};

enum class EpilogueTileType {
  kEpilogueTileAuto = 0,            // Automatic tile computation for the epilogue
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmPayload {
  LaunchConfigurationPayload  launch_config;     // Launch configuration information
  TensorOperandPayload        A;
  TensorOperandPayload        B;
  TensorOperandPayload        C;
  TensorOperandPayload        D;
  MainloopScheduleType        mainloop_schedule_type;
  EpilogueScheduleType        epilogue_schedule_type;
  EpilogueTileType            epilogue_tile_type;

  GemmPayload() {}

  GemmPayload(
    LaunchConfigurationPayload  launch_config_,
    TensorOperandPayload        A_,
    TensorOperandPayload        B_,
    TensorOperandPayload        C_,
    TensorOperandPayload        D_,
    MainloopScheduleType        mainloop_schedule_type_,
    EpilogueScheduleType        epilogue_schedule_type_,
    EpilogueTileType            epilogue_tile_type_
  ):
    launch_config(launch_config_),
    A(A_),
    B(B_),
    C(C_),
    D(D_),
    mainloop_schedule_type(mainloop_schedule_type_),
    epilogue_schedule_type(epilogue_schedule_type_),
    epilogue_tile_type(epilogue_tile_type_) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmInstance {
  GemmDescription     description;  // Description of the operation
  GemmPayload         payload;      // Payload information with respect to this kernel

  GemmInstance(
    GemmDescription   desc_,
    GemmPayload       payload_
  ):
  description(desc_), payload(payload_) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
