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
    \brief Architecture-specific operators on memory added for SM75
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/layout/matrix.h"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/util.hpp"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Layout of destination matrix (column-major implies transpose)
  typename Layout,
  /// .x1, .x2, or .x4
  int MatrixCount
>
CUTLASS_DEVICE void ldsm(Array<unsigned, MatrixCount> & D, void const* ptr);

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Determine the appropriate way to target PTX's "ldmatrix" instruction.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}

// NVVM LDSM intrinsics are an internal-only feature (never exposed publicly). If feature detection
// determines LDSM is supported, it will set CUDA_NVVM_LDSM_SUPPORTED, otherwise it is assumed 
// not supported.

#if ! defined(CUDA_NVVM_LDSM_SUPPORTED)
  #define CUDA_NVVM_LDSM_SUPPORTED (((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 1)) || (__CUDACC_VER_MAJOR__ > 10)) && (CUTLASS_ENABLE_INTERNAL_NVVM)
#endif

#if ! defined(CUDA_NVVM_LDSM_ENABLED)
  #define CUDA_NVVM_LDSM_ENABLED (CUDA_NVVM_LDSM_SUPPORTED)
#endif

#if CUDA_NVVM_LDSM_ENABLED
extern "C" {
  __device__ int __nvvm_ldsm_b16_m8n8_n1(int4 const *ptr);
  __device__ int2 __nvvm_ldsm_b16_m8n8_n2(int4 const *ptr);
  __device__ int4 __nvvm_ldsm_b16_m8n8_n4(int4 const *ptr);
  
  __device__ int __nvvm_ldsm_b16_m8n8_t1(int4 const *ptr);
  __device__ int2 __nvvm_ldsm_b16_m8n8_t2(int4 const *ptr);
  __device__ int4 __nvvm_ldsm_b16_m8n8_t4(int4 const *ptr);
}
#endif

#if (CUDA_NVVM_LDSM_ENABLED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  #define CUDA_NVVM_LDSM_ACTIVATED 1
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// {$nv-internal-release end}

/// CUTLASS helper to get SMEM pointer
CUTLASS_HOST_DEVICE unsigned cutlass_get_smem_pointer(void *ptr) {
  return cute::cast_smem_ptr_to_uint(ptr);
}

/// CUTLASS helper to get SMEM pointer
CUTLASS_DEVICE unsigned cutlass_get_smem_pointer(void const *ptr) {
  return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  // {$nv-internal-release begin}
  #elif CUDA_NVVM_LDSM_ACTIVATED

    reinterpret_cast<int &>(D) = __nvvm_ldsm_b16_m8n8_n1(reinterpret_cast<int4 const *>(ptr));

  // {$nv-internal-release end}
  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y;
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    reinterpret_cast<int2 &>(D) = make_int2(x, y);

  // {$nv-internal-release begin}
  #elif CUDA_NVVM_LDSM_ACTIVATED

    reinterpret_cast<int2 &>(D) = __nvvm_ldsm_b16_m8n8_n2(reinterpret_cast<int4 const *>(ptr));

  // {$nv-internal-release end}
  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y, z, w;
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);

  // {$nv-internal-release begin}
  #elif CUDA_NVVM_LDSM_ACTIVATED

    reinterpret_cast<int4 &>(D) = __nvvm_ldsm_b16_m8n8_n4(reinterpret_cast<int4 const *>(ptr));

  // {$nv-internal-release end}
  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Transpose on 16b granularity
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::ColumnMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  // {$nv-internal-release begin}
  #elif CUDA_NVVM_LDSM_ACTIVATED

    reinterpret_cast<int &>(D) = __nvvm_ldsm_b16_m8n8_t1(reinterpret_cast<int4 const *>(ptr));

  // {$nv-internal-release end}
  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::ColumnMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y;
    asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    reinterpret_cast<int2 &>(D) = make_int2(x, y);

  // {$nv-internal-release begin}
  #elif CUDA_NVVM_LDSM_ACTIVATED

    reinterpret_cast<int2 &>(D) = __nvvm_ldsm_b16_m8n8_t2(reinterpret_cast<int4 const *>(ptr));

  // {$nv-internal-release end}
  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_DEVICE void ldsm<layout::ColumnMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y, z, w;
    asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);

  // {$nv-internal-release begin}
  #elif CUDA_NVVM_LDSM_ACTIVATED

    reinterpret_cast<int4 &>(D) = __nvvm_ldsm_b16_m8n8_t4(reinterpret_cast<int4 const *>(ptr));

  // {$nv-internal-release end}
  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct shared_load_op {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    D = *reinterpret_cast<AccessType const *>(ptr);  
  }
};

template <typename AccessType>
CUTLASS_DEVICE void shared_load(AccessType &D, void const *ptr) {
  shared_load_op<AccessType, int(sizeof(AccessType))>(D, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 16> {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);

    uint4 v;
    asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : 
      "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));

    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 8> {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);

    uint2 v;
    asm volatile ("ld.shared.v2.b32 {%0, %1}, [%2];" : 
      "=r"(v.x), "=r"(v.y) : "r"(addr));

    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
