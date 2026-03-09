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
    \brief Predicate handling using NVVM IR intrinsics.
*/

#pragma once

#include "cutlass/cutlass.h"

// {$nv-internal-release begin}
#if ! defined(CUTLASS_CUDA_RP2RP_ENABLED)
#define CUTLASS_CUDA_RP2RP_ENABLED 0
#endif
// {$nv-internal-release end}

namespace cutlass {
////////////////////////////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}
#if CUTLASS_CUDA_RP2RP_ENABLED

/// Pack predicates using NVVM intrinsic function (p2r)
template <int kIterationCount, 
          int kPredicatesPerByte>
CUTLASS_DEVICE
static void pack_predicates(bool *predicates, unsigned int *predicate_reg) {
  // Make sure the predicates fit in 1 single register.
  static_assert(kIterationCount <= kPredicatesPerByte * 4, "Too many predicates for 1 register");

  // Pack the predicates that use a "full" byte.
  const int kCompleteBytes = kIterationCount / kPredicatesPerByte;
  const int kMask = (1 << kPredicatesPerByte) - 1;
  CUTLASS_PRAGMA_UNROLL
  for(int i = 0; i < kCompleteBytes; ++i) {
    if(i == 0) {
      __nv_p2r(0, &predicates[i * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(i == 1) {
      __nv_p2r(1, &predicates[i * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(i == 2) {
      __nv_p2r(2, &predicates[i * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(i == 3) {
      __nv_p2r(3, &predicates[i * kPredicatesPerByte], kMask, predicate_reg);
    }
  }

  // Deal with the remainder.
  const int kRemainder = kIterationCount % kPredicatesPerByte;
  if( kRemainder != 0 ) {
    const int kMask = (1 << kRemainder) - 1;
    if( kCompleteBytes == 0 ) {
      __nv_p2r(0, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(kCompleteBytes == 1) {
      __nv_p2r(1, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(kCompleteBytes == 2) {
      __nv_p2r(2, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(kCompleteBytes == 3) {
      __nv_p2r(3, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Extract prediates using NVVM intrinsic function (r2p)
template <int kIterationCount, 
         int kPredicatesPerByte>
CUTLASS_DEVICE
static void extract_predicates(bool *predicates, unsigned int predicate_reg) {
  // Make sure the predicates fit in 1 single register.
  static_assert(kIterationCount <= kPredicatesPerByte * 4, "Too many predicates for 1 register");

  // Pack the predicates that use a "full" byte.
  const int kCompleteBytes = kIterationCount / kPredicatesPerByte;
  const int kMask = (1 << kPredicatesPerByte) - 1;
  CUTLASS_PRAGMA_UNROLL
  for(int i = 0; i < kCompleteBytes; ++i) {
    if(i == 0) {
      __nv_r2p(0, &predicates[i*kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(i == 1) {
      __nv_r2p(1, &predicates[i*kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(i == 2) {
      __nv_r2p(2, &predicates[i*kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(i == 3) {
      __nv_r2p(3, &predicates[i*kPredicatesPerByte], kMask, predicate_reg);
    }
  }

  // Deal with the remainder.
  const int kRemainder = kIterationCount % kPredicatesPerByte;
  if(kRemainder != 0) {
    const int kMask = (1 << kRemainder) - 1;
    if(kCompleteBytes == 0) {
      __nv_r2p(0, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(kCompleteBytes == 1) {
      __nv_r2p(1, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(kCompleteBytes == 2) {
      __nv_r2p(2, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    } 
    else if(kCompleteBytes == 3) {
      __nv_r2p(3, &predicates[kCompleteBytes * kPredicatesPerByte], kMask, predicate_reg);
    }
  }
}

#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

} // namespace cutlass
