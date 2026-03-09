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
/*! \file
    \brief Random synchronization delays for race condition debugging.
*/

#pragma once

// {$nv-internal-release file}

#include "cutlass/detail/helper_macros.hpp" // CUTLASS_DEVICE
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif

#if defined(CUTLASS_ENABLE_STARVATION) && !defined(CUTLASS_ENABLE_RANDOM_DELAYS)
#define CUTLASS_ENABLE_RANDOM_DELAYS 1
#endif

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

CUTLASS_DEVICE
uint32_t globaltimer_lo() {
  // MSVC requires protecting use of CUDA-specific nonstandard syntax,
  // like "asm volatile", with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
  uint32_t ret;
  asm volatile (
    "mov.u32 %0, %%globaltimer_lo;\n"
    : "=r"(ret) :
  );
  return ret;
#else
  return 0u;
#endif
}

CUTLASS_DEVICE
uint32_t globaltimer_hi() {
  // MSVC requires protecting CUDA-specific nonstandard syntax with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
  uint32_t ret;

  asm volatile (
    "mov.u32 %0, %%globaltimer_hi;\n"
    : "=r"(ret) :
  );

  return ret;
#else
  return 0u;
#endif
}

CUTLASS_DEVICE
uint32_t clock_lo() {
  // MSVC requires protecting CUDA-specific nonstandard syntax with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
  uint32_t ret;

  asm volatile (
    "mov.u32 %0, %%clock;\n"
    : "=r"(ret) :
  );

  return ret;
#else
  return 0u;
#endif
}

CUTLASS_DEVICE
uint32_t clock_hi() {
  // MSVC requires protecting CUDA-specific nonstandard syntax with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
  uint32_t ret;

  asm volatile (
    "mov.u32 %0, %%clock_hi;\n"
    : "=r"(ret) :
  );

  return ret;
#else
  return 0u;
#endif
}

CUTLASS_DEVICE
uint32_t murmurhash3_insert(uint32_t hash, uint32_t part) {
  part *= 0xcc9e2d51;
  part = part << 15 | part >> 17;
  hash ^= part * 0x1b873593;
  hash = hash << 13 | hash >> 19;
  hash *= 5;
  hash += 0xe6546b64;
  return hash;
}

CUTLASS_DEVICE
uint32_t murmurhash3_finish(uint32_t hash) {
  hash ^= hash >> 16;
  hash *= 0x85ebca6b;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35;
  hash ^= hash >> 16;
  return hash;
}

CUTLASS_DEVICE
uint32_t state_hash() {
  uint32_t ret = 0;

  #if defined(__CUDA_ARCH__)

  ret = murmurhash3_insert(ret, blockIdx.x << 16 | threadIdx.x);
  ret = murmurhash3_insert(ret, blockIdx.y << 16 | threadIdx.y);
  ret = murmurhash3_insert(ret, blockIdx.z << 16 | threadIdx.z);
  ret = murmurhash3_insert(ret, globaltimer_lo());
  ret = murmurhash3_insert(ret, globaltimer_hi());
  ret = murmurhash3_insert(ret, clock_lo());
  ret = murmurhash3_insert(ret, clock_hi());
  ret = murmurhash3_finish(ret);

  #endif

  return ret;
}

#if defined(CUTLASS_ENABLE_STARVATION)
static CUTLASS_DEVICE uint32_t starvation_victim;
#endif

CUTLASS_DEVICE
uint32_t starvation_id() {
  uint32_t ret = 0;

  #if defined(__CUDA_ARCH__)

  // This computation is a prefix of the random delay
  // state_hash() so the compiler will not repeat it.
  ret = murmurhash3_insert(ret, blockIdx.x << 16 | threadIdx.x);
  ret = murmurhash3_insert(ret, blockIdx.y << 16 | threadIdx.y);
  ret = murmurhash3_insert(ret, blockIdx.z << 16 | threadIdx.z);

  #endif

  return ret;
}

CUTLASS_DEVICE
bool starvation_condition() {
  #if defined(CUTLASS_ENABLE_STARVATION)
  return atomicAdd(&starvation_victim, 0) == starvation_id();
  #else
  return true;
  #endif
}

CUTLASS_DEVICE
void starvation_setup() {
  #if defined(CUTLASS_ENABLE_STARVATION) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

  // Each wave of CTAs chooses a thread to starve.
  uint32_t id = starvation_id();
  uint32_t r = 0;

  CUTLASS_PRAGMA_NO_UNROLL
  for (uint32_t repeat = 1 << 13; repeat > 0; repeat--) {
    r = murmurhash3_insert(r, cutlass::arch::detail::state_hash());
    r = murmurhash3_finish(r);

    // Shorter delay than used in the rest of the kernel.
    // pow(2, 18) nanoseconds is roughly a quarter millisecond.
    uint32_t ns = r >> (32 - 18);

    __nanosleep(ns);

    if (((r >> 8) & 63) == 0) {
      // Swap this thread's id into global memory.
      atomicExch(&starvation_victim, id);
    }
  }

  #endif
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE
void random_fill(char* smem) {
  #if defined(CUTLASS_ENABLE_RANDOM_DELAYS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

  uint32_t smem_size = 0;
  asm volatile (
    "mov.u32 %0, %%dynamic_smem_size;\n"
    : "=r"(smem_size) :
  );

  for (uint32_t i = threadIdx.x; i < smem_size; i += blockDim.x) {
    smem[i] = cutlass::arch::detail::state_hash();
  }

  __syncthreads();

  cutlass::arch::detail::starvation_setup();

  __syncthreads();
  #else
  CUTLASS_UNUSED(smem);
  #endif
}

CUTLASS_DEVICE
void random_delay() {
  #if defined(CUTLASS_ENABLE_RANDOM_DELAYS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

  if (!cutlass::arch::detail::starvation_condition()) return;

  uint32_t r = cutlass::arch::detail::state_hash();

  #if defined(CUTLASS_ENABLE_STARVATION)
  // pow(2, 26) nanoseconds is about 64 milliseconds.
  uint32_t ns = r >> (32 - 26);
  #else
  // pow(2, 21) nanoseconds is about 2 milliseconds.
  uint32_t ns = r >> (32 - 21);
  #endif

  __nanosleep(ns);

  #endif
}

#if defined(CUTLASS_ENABLE_RANDOM_DELAYS)

#define __syncthreads() do {\
  __syncthreads();\
  cutlass::arch::random_delay();\
} while (0)

#define __syncwarp(...) do {\
  __syncwarp(__VA_ARGS__);\
  cutlass::arch::random_delay();\
} while (0)

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
