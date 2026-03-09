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
    \brief Directives related to TMA operations
    // {$nv-internal-release file}

*/
#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Controls PTX TMA operations
struct TMAOperation {
  enum class Type {
    /// Tiled load mode
    kTiled,
    /// Image-to-column load mode
    kIm2Col,
  };

  // {No Multicast, Multicast}
  enum class Broadcast {
    /// Single CTA load/store operation 
    //  Only one CTA that issues TMA sends/receives data
    kSinglecast,
    /// Multicast mode 
    //  Up to 16 destination CTA IDs are encoded 16-bit mask provided
    kMulticast
  };

  // {u8, u16, u32, s32, u64, s64, f16.rn, f32.rn,  f32.ftz.rn,  f64.rn, bf16.rn}
  enum class Format {
    kU8,
    kU16,
    kU32,
    kS8,
    kS32,
    kS64,
    kF16_RN,
    kF32_RN,
    kF32_FTZ_RN,
    kF64_RN,
    kBF16_RN
  };

  // {disable, interleave_16B, interleave_32B}
  enum class Interleave {
    kDisable,
    kInterleave16B,
    kInterleave32B
  };

  // {disable, swizzle_32B, swizzle_64B, swizzle_128B}  
  enum class SmemSwizzle {
    kDisable,
    kSwizzle32B,
    kSwizzle64B,
    kSwizzle128B
  };
  
  // {zero, constant}
  enum class OobFillMode {
    kZeroFill,
    kConstantFill
  };

  //F32toTF32 {disable, enable}           
  enum class F32toTF32 {
    kDisable,
    kEnable
  };
  
  // L2sectorPromotion {disable, L2_64B, L2_128B, L2_256B}     
  enum class L2sectorPromotion {
    kDisable,
    kL2_64B,
    kL2_128B,
    kL2_256B
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass