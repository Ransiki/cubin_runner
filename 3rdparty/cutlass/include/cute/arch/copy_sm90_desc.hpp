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
#pragma once

#include "cutlass/numeric_types.h"

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#include <cinttypes>
#endif

#include <cute/config.hpp>

#include <cute/arch/util.hpp>   // cute::cast_smem_ptr_to_uint
#include <cute/arch/config.hpp> // CUTE_ARCH_TMA_SMxx_ENABLED
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90.hpp>

#include <cute/container/alignment.hpp>
#include <cute/container/bit_field.hpp>
#include <cute/container/array.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute
{

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Barriers are 64-bit of user-managed information used in broadly two types syncronization patterns
/// 1) arrive/wait on threads (usage: cp.async and warp-specialized kernels)
/// 2) transaction-based (usage: TMA transaction where a CTA issues one transaction)
//////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize barrier present in shared memory
CUTE_HOST_DEVICE
void
initialize_barrier(uint64_t& smem_barrier,                 // 64 bits user-manged barrier in smem
                   int thread_count = 1)                   // Thread count expected to arrive/wait on this barrier
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(thread_count));
#endif
}

// Set the number of bytes transfered per transaction and perform an arrive operation as well
CUTE_HOST_DEVICE
void
set_barrier_transaction_bytes(uint64_t& smem_barrier,      // 64 bits user-manged barrier in smem
                              uint32_t bytes)              // Number of bytes transfered by per TMA transaction
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(bytes));
#endif
}

// Barrier wait
CUTE_HOST_DEVICE
void
wait_barrier(uint64_t& smem_barrier,                       // 64 bits user-manged barrier in smem
             int phase_bit)                                // Current phase bit the barrier waiting to flip
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  asm volatile(".pragma \"set knob DontInsertYield\";\n" : : : "memory" );  // {$nv-internal-release}

  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
    "{\n"
    ".reg .pred                P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    "@P1                       bra DONE;\n"
    "bra                   LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :: "r"(smem_int_ptr),
       "r"(phase_bit));

  asm volatile(".pragma \"reset knob DontInsertYield\";\n" : : : "memory" );  // {$nv-internal-release}
#endif
}

// Barrier arrive
CUTE_HOST_DEVICE
void
arrive_barrier(uint64_t& smem_barrier)                      // 64 bits user-manged barrier in smem
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
    "{\n"
    ".reg .b64 state; \n"
    "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
    "}\n"
    :: "r"(smem_int_ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TMA Descriptor and utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TMA {

enum class SmemSwizzleBits : uint8_t {
  DISABLE = 0,
  B32 = 1,
  B64 = 2,
  B128 = 3,
};

enum class SmemSwizzleBase : uint8_t {
  SWIZZLE_BASE_16B         = 0,
  
  SWIZZLE_BASE_32B         = 1,
  SWIZZLE_BASE_32B_FLIP_8B = 2,
  SWIZZLE_BASE_64B         = 3,
  
};

enum class OOBFill : uint8_t {
  ZERO = 0,
  CONSTANT = 1,
};

CUTE_HOST_DEVICE char const* to_string(OOBFill const& t) {
  switch (t) {
    case OOBFill::ZERO:     return "ZERO";
    case OOBFill::CONSTANT: return "CONSTANT";
  }
  return nullptr;
}

enum class L2Promotion : uint8_t {
  DISABLE = 0,
  B64 = 1,
  B128 = 2,
  B256 = 3,
};

CUTE_HOST_DEVICE char const* to_string(L2Promotion const& t) {
  switch (t) {
    case L2Promotion::DISABLE: return "DISABLE";
    case L2Promotion::B64:     return "B64";
    case L2Promotion::B128:    return "B128";
    case L2Promotion::B256:    return "B256";
  }
  return nullptr;
}

// Aux parameters which are independent with the problem size
struct DescriptorAuxParams {
  OOBFill     oobfill_     = OOBFill::ZERO;
  L2Promotion l2promo_     = L2Promotion::DISABLE;
};

// {$nv-internal-release begin}
enum class OobAddrGenMode : uint8_t {
  OOB_ADDR_GEN_MODE_LIB_4kB    = 0,  // Hopper legacy behavior (last-inbound mode + 4kB spread)
  OOB_ADDR_GEN_MODE_BASE_128kB = 1,  // New mode that can hit all GB102 L2 slices
};
// {$nv-internal-release end}

enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};


enum class CacheHintSm100 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};


#if (__CUDACC_VER_MAJOR__ >= 12)

#if !defined(__CUDACC_RTC__)
/// @return The TMA descriptor datatype enum corresponding to T.
template <class T>
inline CUtensorMapDataType
to_CUtensorMapDataType() {
  if constexpr (is_same_v<T,       int8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T,      uint8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T,     mxint8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else // {$nv-internal-release}
  if constexpr (is_same_v<T, float_e4m3_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_e5m2_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
  if constexpr (is_same_v<T, float_e3m4_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else // {$nv-internal-release}
  if constexpr (is_same_v<T, float_ue8m0_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;   } else
  if constexpr (is_same_v<T, type_erased_dynamic_float8_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;} else 
  if constexpr (is_same_v<T,     uint16_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT16;   } else
  if constexpr (is_same_v<T,     uint32_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT32;   } else
  if constexpr (is_same_v<T,     uint64_t>) { return CU_TENSOR_MAP_DATA_TYPE_UINT64;   } else
  if constexpr (is_same_v<T,      int32_t>) { return CU_TENSOR_MAP_DATA_TYPE_INT32;    } else
  if constexpr (is_same_v<T,      int64_t>) { return CU_TENSOR_MAP_DATA_TYPE_INT64;    } else
  if constexpr (is_same_v<T,       half_t>) { return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;  } else
  if constexpr (is_same_v<T,        float>) { return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;  } else
  if constexpr (is_same_v<T,       double>) { return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;  } else
  if constexpr (is_same_v<T,   bfloat16_t>) { return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; } else
  if constexpr (is_same_v<T,   tfloat32_t>) { return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32; } else
  // if constexpr () { return CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ;  } else // {$nv-internal-release}
  // if constexpr () { return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ; } else // {$nv-internal-release}

  #if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ > 6)))
  if constexpr (is_same_v<T, float_e2m1_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;  } else
  if constexpr (is_same_v<T, float_e0m3_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;  } else // {$nv-internal-release}
  if constexpr (is_same_v<T, float_e2m3_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, float_e3m2_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, type_erased_dynamic_float4_t>)    { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;  } else
  if constexpr (is_same_v<T, type_erased_dynamic_float6_t>)    { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e2m1_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e2m3_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e3m2_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::type_erased_dynamic_float4_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::type_erased_dynamic_float6_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  #endif

  { static_assert(sizeof(T) < 0, "Unknown TMA Format!"); }
}

inline CUtensorMapSwizzle
to_CUtensorMapSwizzle(SmemSwizzleBits const& t, SmemSwizzleBase const& b) {
  switch (t) {
    default: throw std::runtime_error("Unsupported pair of SmemSwizzleBits and SmemSwizzleBase!");
    case SmemSwizzleBits::DISABLE: 
      assert((b == SmemSwizzleBase::SWIZZLE_BASE_16B) && "Expected 16B swizzle base for 0B swizzle bits.");
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case SmemSwizzleBits::B32:
      assert((b == SmemSwizzleBase::SWIZZLE_BASE_16B) && "Expected 16B swizzle base for 32B swizzle bits.");
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case SmemSwizzleBits::B64:
      assert((b == SmemSwizzleBase::SWIZZLE_BASE_16B) && "Expected 16B swizzle base for 64B swizzle bits.");
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case SmemSwizzleBits::B128:
      switch (b) {
        default: throw std::runtime_error("Unsupported pair of SmemSwizzleBits and SmemSwizzleBase!");
        case SmemSwizzleBase::SWIZZLE_BASE_16B: return CU_TENSOR_MAP_SWIZZLE_128B;
        
        #if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ > 6)))
        case SmemSwizzleBase::SWIZZLE_BASE_32B: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
        case SmemSwizzleBase::SWIZZLE_BASE_64B: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
        #endif
      }
  }
}

inline CUtensorMapFloatOOBfill
to_CUtensorMapFloatOOBfill(OOBFill const& t) {
  switch(t) {
    default:                throw std::runtime_error("Unknown OOBFill!");
    case OOBFill::ZERO:     return CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case OOBFill::CONSTANT: return CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  }
}

inline CUtensorMapL2promotion
to_CUtensorMapL2promotion(L2Promotion const& t) {
  switch(t) {
    default: throw std::runtime_error("Unknown L2Promotion!");
    case L2Promotion::DISABLE: return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case L2Promotion::B64:     return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case L2Promotion::B128:    return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case L2Promotion::B256:    return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }
}

#endif // !defined(__CUDACC_RTC__)

#endif // (__CUDACC_VER_MAJOR__ >= 12)

} // end namespace TMA

// {$nv-internal-release begin}
// INTERNAL TMA Descriptor and utilities -- preserved for debugging and printing

namespace TMA {

#if !defined(__CUDACC_RTC__)
// Output operator for all enums in this namespace
template <class T>
CUTE_HOST std::ostream& operator<<(std::ostream& os, T const& t) {
  char const* s = to_string(t);
  if (s) {
    std::operator<<(os, s);  // Explicit call to avoid ambiguity
  } else {
    os.setstate(std::ios_base::failbit);
  }
  return os;
}
#endif // !defined(__CUDACC_RTC__)

CUTE_HOST_DEVICE char const* to_string(SmemSwizzleBits const& t) {
  switch (t) {
    case SmemSwizzleBits::DISABLE: return "SWIZZLE_BITS_DISABLE";
    case SmemSwizzleBits::B32:     return "SWIZZLE_BITS_32b";
    case SmemSwizzleBits::B64:     return "SWIZZLE_BITS_64b";
    case SmemSwizzleBits::B128:    return "SWIZZLE_BITS_128b";
  }
  return nullptr;
}

CUTE_HOST_DEVICE char const* to_string(SmemSwizzleBase const& t) {
  switch (t) {
    case SmemSwizzleBase::SWIZZLE_BASE_16B:          return "SWIZZLE_BASE_16B";
    case SmemSwizzleBase::SWIZZLE_BASE_32B:          return "SWIZZLE_BASE_32B";
    case SmemSwizzleBase::SWIZZLE_BASE_32B_FLIP_8B:  return "SWIZZLE_BASE_32B_FLIP_8B";
    case SmemSwizzleBase::SWIZZLE_BASE_64B:          return "SWIZZLE_BASE_64B";
  }
  return nullptr;
}

CUTE_HOST_DEVICE char const* to_string(OobAddrGenMode const& t) {
  switch (t) {
    case OobAddrGenMode::OOB_ADDR_GEN_MODE_LIB_4kB:          return "OOB_ADDR_GEN_MODE_LIB_4kB";
    case OobAddrGenMode::OOB_ADDR_GEN_MODE_BASE_128kB:       return "OOB_ADDR_GEN_MODE_BASE_128kB";
  }
  return nullptr;
}

enum class Type : uint8_t {
  TILED = 0,
  IM2COL = 1,
};

CUTE_HOST_DEVICE char const* to_string(Type const& t) {
  switch (t) {
    case Type::TILED:  return "TILED";
    case Type::IM2COL: return "IM2COL";
  }
  return nullptr;
}

enum class Dimension : uint8_t {
  ONE_D = 0,
  TWO_D = 1,
  THREE_D = 2,
  FOUR_D = 3,
  FIVE_D = 4,
};

CUTE_HOST_DEVICE char const* to_string(Dimension const& t) {
  switch (t) {
    case Dimension::ONE_D:   return "ONE_D";
    case Dimension::TWO_D:   return "TWO_D";
    case Dimension::THREE_D: return "THREE_D";
    case Dimension::FOUR_D:  return "FOUR_D";
    case Dimension::FIVE_D:  return "FIVE_D";
  }
  return nullptr;
}

enum class Format : uint8_t {
  U8 = 0,
  U16 = 1,
  U32 = 2,
  S32 = 3,
  U64 = 4,
  S64 = 5,
  F16_RN = 6,
  F32_RN = 7,
  F32_FTZ_RN = 8,
  F64_RN = 9,
  BF16_RN = 10,
  // {$nv-internal-release begin}
  U4  = 11,
  U4_UNPACK_U8 = 12,
  U6_UNPACK_U8 = 13,
  // {$nv-internal-release end}

  // {$nv-internal-release begin}
  // XXX: Both Hopper and Blackwell does not define TF32 in TensorElementFormat_t
  // {$nv-internal-release end}
  TF32_RN = 14,
  TF32_FTZ_RN = 15,
};

CUTE_HOST_DEVICE char const* to_string(Format const& t) {
  switch (t) {
    case Format::U8:         return "U8";
    case Format::U16:        return "U16";
    case Format::U32:        return "U32";
    case Format::S32:        return "S32";
    case Format::U64:        return "U64";
    case Format::S64:        return "S64";
    case Format::F16_RN:     return "F16_RN";
    case Format::F32_RN:     return "F32_RN";
    case Format::F64_RN:     return "F64_RN";
    case Format::BF16_RN:    return "BF16_RN";
    case Format::F32_FTZ_RN: return "F32_FTZ_RN";
    // {$nv-internal-release begin}
    case Format::U4:           return "U4";
    case Format::U4_UNPACK_U8: return "U4_UNPACK_U8";
    case Format::U6_UNPACK_U8: return "U6_UNPACK_U8";
    // {$nv-internal-release end}
    case Format::TF32_RN:     return "TF32_RN";
    case Format::TF32_FTZ_RN: return "TF32_FTZ_RN";
  }
  return nullptr;
}

CUTE_HOST_DEVICE int sizeof_bits_format(Format const& t) {
  switch (t) {
    case Format::U4: // {$nv-internal-release}
      return 4;      // {$nv-internal-release}
    case Format::U8:
    case Format::U4_UNPACK_U8: // {$nv-internal-release}
    case Format::U6_UNPACK_U8: // {$nv-internal-release}
      return 8;
    case Format::U16:
    case Format::F16_RN:
    case Format::BF16_RN:
      return 16;
    case Format::U32:
    case Format::S32:
    case Format::F32_RN:
    case Format::F32_FTZ_RN:
    case Format::TF32_RN:
    case Format::TF32_FTZ_RN:
      return 32;
    case Format::U64:
    case Format::S64:
    case Format::F64_RN:
      return 64;
  }
  return 0;
}

CUTE_HOST_DEVICE int sizeof_format(Format const& t) {
    return sizeof_bits_format(t) * 8;
}

template <class T>
inline Format to_Format() {
  
  // TMALDG TMASTG U4 : `| 4bit | 4bit | 4bit | ... | -> | 4bit | 4bit | 4bit | ... |` {$nv-internal-release}
  if constexpr (is_same<T, type_erased_dynamic_float4_t>::value) { return Format::U4; } else
  if constexpr (is_same<T, float_e2m1_t>::value) { return Format::U4; } else
  if constexpr (is_same<T, float_e0m3_t>::value) { return Format::U4; } else
  // TMALDG U4x16P64 : `| 4bit | 4bit | 4bit | ... | -> | 64bit | 64bit_padding_0 |` {$nv-internal-release}
  if constexpr (is_same<T, detail::type_erased_dynamic_float4_unpacksmem_t>::value) { return Format::U4_UNPACK_U8; } else
  if constexpr (is_same<T, detail::float_e2m1_unpacksmem_t>::value) { return Format::U4_UNPACK_U8; } else
  // TMALDG U6x16P32 : `| 6bit | 6bit | 6bit | ... | -> | 96bit | 32bit_padding_0 |` {$nv-internal-release}
  if constexpr (is_same<T, detail::type_erased_dynamic_float6_unpacksmem_t>::value) { return Format::U6_UNPACK_U8; } else
  if constexpr (is_same<T, detail::float_e2m3_unpacksmem_t>::value) { return Format::U6_UNPACK_U8; } else
  if constexpr (is_same<T, detail::float_e3m2_unpacksmem_t>::value) { return Format::U6_UNPACK_U8; } else
  // TMASTG U6P2x16 : `| 6bit + 2bit0 | 6bit + 2bit0 | 6bit + 2bit0 | ... | -> | 6bit | 6bit | 6bit | ... |` {$nv-internal-release}
  // TMASTG TMALDG's U6_UNPACK_U8 means different thing {$nv-internal-release}
  if constexpr (is_same<T, type_erased_dynamic_float6_t>::value) { return Format::U6_UNPACK_U8; } else
  if constexpr (is_same<T, float_e2m3_t>::value) { return Format::U6_UNPACK_U8; } else
  if constexpr (is_same<T, float_e3m2_t>::value) { return Format::U6_UNPACK_U8; } else
  if constexpr (is_same<T, type_erased_dynamic_float8_t>::value) { return Format::U8; } else
  if constexpr (is_same<T, float_e4m3_t>::value) { return Format::U8; } else
  if constexpr (is_same<T, float_e5m2_t>::value) { return Format::U8; } else
  if constexpr (is_same<T, float_e3m4_t>::value) { return Format::U8; } else
  if constexpr (is_same<T,float_ue8m0_t>::value) { return Format::U8; } else
  if constexpr (is_same<T,float_ue4m3_t>::value) { return Format::U8; } else
  if constexpr (is_same<T,     mxint8_t>::value) { return Format::U8; } else
  
  if constexpr (is_same<T,       int8_t>::value) { return Format::U8; } else
  if constexpr (is_same<T,      uint8_t>::value) { return Format::U8; } else
  if constexpr (is_same<T,     uint16_t>::value) { return Format::U16; } else
  if constexpr (is_same<T,     uint32_t>::value) { return Format::U32; } else
  if constexpr (is_same<T,      int32_t>::value) { return Format::S32; } else
  if constexpr (is_same<T,     uint64_t>::value) { return Format::U64; } else
  if constexpr (is_same<T,      int64_t>::value) { return Format::S64; } else
  if constexpr (is_same<T,       half_t>::value) { return Format::F16_RN; } else
  if constexpr (is_same<T,        float>::value) { return Format::F32_RN; } else
  if constexpr (is_same<T,       double>::value) { return Format::F64_RN; } else
  if constexpr (is_same<T,   bfloat16_t>::value) { return Format::BF16_RN; } else
  //if constexpr () { return Format::F32_FTZ_RN; } else   // ??? {$nv-internal-release}
  if constexpr (is_same<T,   tfloat32_t>::value) { return Format::TF32_RN; } else
  //if constexpr () { return Format::TF32_FTZ_RN; } else   // ??? {$nv-internal-release}
  { static_assert(sizeof(T) < 0, "Unknown TMA Format!"); }

  CUTE_GCC_UNREACHABLE;
}

enum class Interleave : uint8_t {
  DISABLE = 0,
  B16 = 1,
  B32 = 2,
};

CUTE_HOST_DEVICE char const* to_string(Interleave const& t) {
  switch (t) {
    case Interleave::DISABLE: return "DISABLE";
    case Interleave::B16:     return "B16";
    case Interleave::B32:     return "B32";
  }
  return nullptr;
}

enum class F32toTF32 : uint8_t {
  DISABLE = 0,
  ENABLE = 1,
};

CUTE_HOST_DEVICE char const* to_string(F32toTF32 const& t) {
  switch (t) {
    case F32toTF32::DISABLE: return "DISABLE";
    case F32toTF32::ENABLE:  return "ENABLE";
  }
  return nullptr;
}

// Helper struct for split strides
template <int lower_bit_start, int lower_bit_size, int upper_bit_start, int upper_bit_size>
union split_stride {
  bit_field<lower_bit_start, lower_bit_size> lower_;
  bit_field<upper_bit_start, upper_bit_size> upper_;

  CUTE_HOST_DEVICE constexpr
  operator uint64_t() const { return uint64_t(lower_) + (uint64_t(upper_) << lower_bit_size); }

  CUTE_HOST_DEVICE constexpr split_stride
  operator=(uint64_t const& stride) {
    assert(stride < (uint64_t(1) << (lower_bit_size + upper_bit_size)));   // Stride must be 36 bits without 4 LSB
    lower_ = uint32_t(stride);
    upper_ = uint8_t (stride >> lower_bit_size);
    return *this;
  }
};

} // end namespace TMA

////////////////////////////////////////////////////////////////////////////////////////////////////

union CUTE_ALIGNAS(64) TmaDescriptorInternal
{
  uint64_t data[16];

  // Global memory address, 4LSB not included
  bit_field<4, 53> start_address_;

  // Common parameters
  bit_field<64, 1, TMA::Type>            type_;         // Tiled or Im2Col mode
  bit_field<65, 3>                       version_;      // Derived from CUDA_ARCH
  bit_field<68, 3, TMA::Dimension>       dim_;          // Dimension of tensors
  bit_field<71, 4, TMA::Format>          format_;       // Datatype of tensors
  bit_field<75, 2, TMA::Interleave>      interleaved_;  // Global memory interleaved state
  bit_field<77, 2, TMA::SmemSwizzleBits> swizzle_bits_; // Shared memory swizzle bits
  bit_field<79, 1, TMA::OOBFill>         oobfill_;      // Shared memory out-of-bounds fill state
  bit_field<80, 1, TMA::F32toTF32>       toTF32_;       // Shared memory conversion state
  bit_field<81, 2, TMA::L2Promotion>     l2promo_;      // L2 cache sector promotion
  bit_field<83, 2, TMA::SmemSwizzleBase> swizzle_base_; // Shared memory swizzle base [BLACKWELL ONLY]
  bit_field<85, 1, TMA::OobAddrGenMode>  oob_addr_gen_mode_; // oob get L2 data mode [BLACKWELL ONLY]

  // Global memory tensor strides in bytes, 4 LSBs not included (units of [uint64_t] 16 bytes)
  TMA::split_stride< 96, 32,
                    224,  4> stride0_;
  TMA::split_stride<128, 32,
                    228,  4> stride1_;
  TMA::split_stride<160, 32,
                    232,  4> stride2_;
  TMA::split_stride<192, 32,
                    236,  4> stride3_;

  // Global memory tensor sizes in elements minus one for [1:2^32]
  bit_field<256, 32> size0_;
  bit_field<288, 32> size1_;
  bit_field<320, 32> size2_;
  bit_field<352, 32> size3_;
  bit_field<384, 32> size4_;

  // Shared memory traversal strides in elements minus one for [1:8]
  bit_field<416, 3> tstride0_;
  bit_field<419, 3> tstride1_;
  bit_field<422, 3> tstride2_;
  bit_field<425, 3> tstride3_;
  bit_field<428, 3> tstride4_;

  // Shared memory box sizes in elements minus one for [1:256]
  bit_field<440, 8> bsize0_;
  bit_field<448, 8> bsize1_;
  bit_field<456, 8> bsize2_;
  bit_field<464, 8> bsize3_;
  bit_field<472, 8> bsize4_;

  CUTE_HOST_DEVICE constexpr
  TmaDescriptorInternal()
    : data{} {}

  CUTE_HOST_DEVICE constexpr
  TmaDescriptorInternal(uint64_t const& d0, uint64_t const& d1, uint64_t const& d2, uint64_t const& d3,
                        uint64_t const& d4, uint64_t const& d5, uint64_t const& d6, uint64_t const& d7)
    : data{d0,d1,d2,d3,d4,d5,d6,d7} {}

  CUTE_HOST_DEVICE friend void
  print(TmaDescriptorInternal const& tma_desc)
  {
    #if !defined(__CUDACC_RTC__)
    printf("DESC_TMA512: 0x%016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n"
           "               %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
           tma_desc.data[7], tma_desc.data[6], tma_desc.data[5], tma_desc.data[4],
           tma_desc.data[3], tma_desc.data[2], tma_desc.data[1], tma_desc.data[0]);
    printf("  start_address: 0x%014" PRIx64 "\n", uint64_t(tma_desc.start_address_));
    printf("\n");
    printf("  type:          0x%01x  [%s]\n", uint8_t(tma_desc.type_),        to_string(tma_desc.type_));
    printf("  version:       0x%01x\n",      uint8_t(tma_desc.version_));
    printf("  dim:           0x%01x  [%s]\n", uint8_t(tma_desc.dim_),         to_string(tma_desc.dim_));
    printf("  format:        0x%01x  [%s]\n", uint8_t(tma_desc.format_),      to_string(tma_desc.format_));
    printf("  interleaved:   0x%01x  [%s]\n", uint8_t(tma_desc.interleaved_), to_string(tma_desc.interleaved_));
    printf("  swizzle:       0x%01x  [%s]\n", uint8_t(tma_desc.swizzle_bits_),to_string(tma_desc.swizzle_bits_));
    printf("  oobfill:       0x%01x  [%s]\n", uint8_t(tma_desc.oobfill_),     to_string(tma_desc.oobfill_));
    printf("  toTF32:        0x%01x  [%s]\n", uint8_t(tma_desc.toTF32_),      to_string(tma_desc.toTF32_));
    printf("  L2promotion:   0x%01x  [%s]\n", uint8_t(tma_desc.l2promo_),     to_string(tma_desc.l2promo_));
    printf("  swizzle_atom:  0x%01x  [%s]\n", uint8_t(tma_desc.swizzle_base_),to_string(tma_desc.swizzle_base_));
    printf("  OOB addr mode: 0x%01x  [%s]\n", uint8_t(tma_desc.oob_addr_gen_mode_), to_string(tma_desc.oob_addr_gen_mode_));
    printf("\n");
    printf("  stride0:       0x%01x 0x%08" PRIx64 "  (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride0_.upper_), uint64_t(tma_desc.stride0_.lower_), uint64_t(tma_desc.stride0_), (uint64_t(tma_desc.stride0_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
    printf("  stride1:       0x%01x 0x%08" PRIx64 "  (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride1_.upper_), uint64_t(tma_desc.stride1_.lower_), uint64_t(tma_desc.stride1_), (uint64_t(tma_desc.stride1_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
    printf("  stride2:       0x%01x 0x%08" PRIx64 "  (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride2_.upper_), uint64_t(tma_desc.stride2_.lower_), uint64_t(tma_desc.stride2_), (uint64_t(tma_desc.stride2_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
    printf("  stride3:       0x%01x 0x%08" PRIx64 "  (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride3_.upper_), uint64_t(tma_desc.stride3_.lower_), uint64_t(tma_desc.stride3_), (uint64_t(tma_desc.stride3_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
    printf("\n");
    printf("  size0:         0x%08x  (%u) [%u %s]\n", uint32_t(tma_desc.size0_), uint32_t(tma_desc.size0_), uint32_t(tma_desc.size0_)+1, to_string(tma_desc.format_));
    printf("  size1:         0x%08x  (%u) [%u %s]\n", uint32_t(tma_desc.size1_), uint32_t(tma_desc.size1_), uint32_t(tma_desc.size1_)+1, to_string(tma_desc.format_));
    printf("  size2:         0x%08x  (%u) [%u %s]\n", uint32_t(tma_desc.size2_), uint32_t(tma_desc.size2_), uint32_t(tma_desc.size2_)+1, to_string(tma_desc.format_));
    printf("  size3:         0x%08x  (%u) [%u %s]\n", uint32_t(tma_desc.size3_), uint32_t(tma_desc.size3_), uint32_t(tma_desc.size3_)+1, to_string(tma_desc.format_));
    printf("  size4:         0x%08x  (%u) [%u %s]\n", uint32_t(tma_desc.size4_), uint32_t(tma_desc.size4_), uint32_t(tma_desc.size4_)+1, to_string(tma_desc.format_));
    printf("\n");
    printf("  tstride0:      0x%01x  (%u) [%u %s]\n", uint8_t(tma_desc.tstride0_), uint8_t(tma_desc.tstride0_), uint8_t(tma_desc.tstride0_)+1, to_string(tma_desc.format_));
    printf("  tstride1:      0x%01x  (%u) [%u %s]\n", uint8_t(tma_desc.tstride1_), uint8_t(tma_desc.tstride1_), uint8_t(tma_desc.tstride1_)+1, to_string(tma_desc.format_));
    printf("  tstride2:      0x%01x  (%u) [%u %s]\n", uint8_t(tma_desc.tstride2_), uint8_t(tma_desc.tstride2_), uint8_t(tma_desc.tstride2_)+1, to_string(tma_desc.format_));
    printf("  tstride3:      0x%01x  (%u) [%u %s]\n", uint8_t(tma_desc.tstride3_), uint8_t(tma_desc.tstride3_), uint8_t(tma_desc.tstride3_)+1, to_string(tma_desc.format_));
    printf("  tstride4:      0x%01x  (%u) [%u %s]\n", uint8_t(tma_desc.tstride4_), uint8_t(tma_desc.tstride4_), uint8_t(tma_desc.tstride4_)+1, to_string(tma_desc.format_));
    printf("\n");
    printf("  bsize0:        0x%02x  (%u) [%u %s]\n", uint8_t(tma_desc.bsize0_), uint8_t(tma_desc.bsize0_), uint32_t(tma_desc.bsize0_)+1, to_string(tma_desc.format_));
    printf("  bsize1:        0x%02x  (%u) [%u %s]\n", uint8_t(tma_desc.bsize1_), uint8_t(tma_desc.bsize1_), uint32_t(tma_desc.bsize1_)+1, to_string(tma_desc.format_));
    printf("  bsize2:        0x%02x  (%u) [%u %s]\n", uint8_t(tma_desc.bsize2_), uint8_t(tma_desc.bsize2_), uint32_t(tma_desc.bsize2_)+1, to_string(tma_desc.format_));
    printf("  bsize3:        0x%02x  (%u) [%u %s]\n", uint8_t(tma_desc.bsize3_), uint8_t(tma_desc.bsize3_), uint32_t(tma_desc.bsize3_)+1, to_string(tma_desc.format_));
    printf("  bsize4:        0x%02x  (%u) [%u %s]\n", uint8_t(tma_desc.bsize4_), uint8_t(tma_desc.bsize4_), uint32_t(tma_desc.bsize4_)+1, to_string(tma_desc.format_));
    #endif // !defined(__CUDACC_RTC__)
  }
};
static_assert(sizeof(TmaDescriptorInternal) * 8 == 1024, "Expected TmaDescriptor to have size 1024 bits");

union CUTE_ALIGNAS(64) Im2ColTmaDescriptorInternal
{
  uint64_t data[16];

  // Global memory address, 4LSB not included
  bit_field<4, 53> start_address_;

  // Common parameters
  bit_field<64, 1, TMA::Type>            type_;         // Tiled or Im2Col mode
  bit_field<65, 3>                       version_;      // Derived from CUDA_ARCH
  bit_field<68, 3, TMA::Dimension>       dim_;          // Dimension of tensors
  bit_field<71, 4, TMA::Format>          format_;       // Datatype of tensors
  bit_field<75, 2, TMA::Interleave>      interleaved_;  // Global memory interleaved state
  bit_field<77, 2, TMA::SmemSwizzleBits> swizzle_bits_; // Shared memory swizzle bits
  bit_field<79, 1, TMA::OOBFill>         oobfill_;      // Shared memory out-of-bounds fill state
  bit_field<80, 1, TMA::F32toTF32>       toTF32_;       // Shared memory conversion state
  bit_field<81, 2, TMA::L2Promotion>     l2promo_;      // L2 cache sector promotion
  bit_field<83, 2, TMA::SmemSwizzleBase> swizzle_base_; // Shared memory swizzle base [BLACKWELL ONLY]
  bit_field<85, 1, TMA::OobAddrGenMode>  oob_addr_gen_mode_; // oob get L2 data mode [BLACKWELL ONLY]

  // Global memory tensor strides in bytes, 4 LSBs not included (units of [uint64_t] 16 bytes)
  TMA::split_stride< 96, 32,
                    224,  4> stride0_;
  TMA::split_stride<128, 32,
                    228,  4> stride1_;
  TMA::split_stride<160, 32,
                    232,  4> stride2_;
  TMA::split_stride<192, 32,
                    236,  4> stride3_;

  // Global memory tensor sizes in elements minus one for [1:2^32]
  bit_field<256, 32> size0_;
  bit_field<288, 32> size1_;
  bit_field<320, 32> size2_;
  bit_field<352, 32> size3_;
  bit_field<384, 32> size4_;

  // Shared memory traversal strides in elements minus one for [1:8]
  bit_field<416, 3> tstride0_;
  bit_field<419, 3> tstride1_;
  bit_field<422, 3> tstride2_;
  bit_field<425, 3> tstride3_;
  bit_field<428, 3> tstride4_;

  // number of elements to load along C dimension minus 1 for [1:256]
  bit_field<440,  8> range_c_;

  // bounding box top corner {W} coordinate offsets, signed int
  bit_field<448, 16> lower_corner_3d_w_;
  // bounding box top corner {W, H} coordinate offsets, signed int
  bit_field<448,  8> lower_corner_4d_w_;
  bit_field<456,  8> lower_corner_4d_h_;
  // bounding box top corner {W, H, D} coordinate offsets, signed int
  bit_field<448,  5> lower_corner_5d_w_;
  bit_field<453,  5> lower_corner_5d_h_;
  bit_field<458,  5> lower_corner_5d_d_;

  // bounding box bottom corner {W} coordinate offsets, signed int
  bit_field<464, 16> upper_corner_3d_w_;
  // bounding box bottom corner {W, H} coordinate offsets, signed int
  bit_field<464,  8> upper_corner_4d_w_;
  bit_field<472,  8> upper_corner_4d_h_;
  // bounding box bottom corner {W, H, D} coordinate offsets, signed int // XXX: verify the bit offsets?
  bit_field<464,  5> upper_corner_5d_w_;
  bit_field<469,  5> upper_corner_5d_h_;
  bit_field<474,  5> upper_corner_5d_d_;

  // number of elements to load along NDHW dimensions minus 1 for [1:1024]
  bit_field<480, 10> range_ndhw_;

  CUTE_HOST_DEVICE constexpr
  Im2ColTmaDescriptorInternal()
    : data{} {}

  CUTE_HOST_DEVICE constexpr
  Im2ColTmaDescriptorInternal(
      uint64_t const& d0, uint64_t const& d1, uint64_t const& d2, uint64_t const& d3,
      uint64_t const& d4, uint64_t const& d5, uint64_t const& d6, uint64_t const& d7)
    : data{d0,d1,d2,d3,d4,d5,d6,d7} {}
};
static_assert(sizeof(Im2ColTmaDescriptorInternal) * 8 == 1024, "Expected 1024b Im2ColTmaDescriptor");

// Printer
CUTE_HOST_DEVICE void
print(Im2ColTmaDescriptorInternal const& tma_desc)
{
#if !defined(__CUDACC_RTC__)
  printf("IM2COL_DESC_TMA512: 0x%016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n"
         "                      %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
          tma_desc.data[7], tma_desc.data[6], tma_desc.data[5], tma_desc.data[4],
          tma_desc.data[3], tma_desc.data[2], tma_desc.data[1], tma_desc.data[0]);
  printf("  start_address: 0x%014" PRIx64 "\n", uint64_t(tma_desc.start_address_));
  printf("\n");
  printf("  type:          0x%01x (%s)\n", uint8_t(tma_desc.type_),        to_string(tma_desc.type_));
  printf("  version:       0x%01x\n",      uint8_t(tma_desc.version_));
  printf("  dim:           0x%01x (%s)\n", uint8_t(tma_desc.dim_),         to_string(tma_desc.dim_));
  printf("  format:        0x%01x (%s)\n", uint8_t(tma_desc.format_),      to_string(tma_desc.format_));
  printf("  interleaved:   0x%01x (%s)\n", uint8_t(tma_desc.interleaved_), to_string(tma_desc.interleaved_));
  printf("  swizzle:       0x%01x (%s)\n", uint8_t(tma_desc.swizzle_bits_),to_string(tma_desc.swizzle_bits_));
  printf("  oobfill:       0x%01x (%s)\n", uint8_t(tma_desc.oobfill_),     to_string(tma_desc.oobfill_));
  printf("  toTF32:        0x%01x (%s)\n", uint8_t(tma_desc.toTF32_),      to_string(tma_desc.toTF32_));
  printf("  L2promotion:   0x%01x (%s)\n", uint8_t(tma_desc.l2promo_),     to_string(tma_desc.l2promo_));
  printf("  swizzle_atom:  0x%01x (%s)\n", uint8_t(tma_desc.swizzle_base_),to_string(tma_desc.swizzle_base_));
  printf("  OOB addr mode: 0x%01x  [%s]\n", uint8_t(tma_desc.oob_addr_gen_mode_), to_string(tma_desc.oob_addr_gen_mode_));
  printf("\n");
  printf("  stride0:       0x%01x 0x%08" PRIx64 " (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride0_.upper_), uint64_t(tma_desc.stride0_.lower_), uint64_t(tma_desc.stride0_), (uint64_t(tma_desc.stride0_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
  printf("  stride1:       0x%01x 0x%08" PRIx64 " (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride1_.upper_), uint64_t(tma_desc.stride1_.lower_), uint64_t(tma_desc.stride1_), (uint64_t(tma_desc.stride1_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
  printf("  stride2:       0x%01x 0x%08" PRIx64 " (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride2_.upper_), uint64_t(tma_desc.stride2_.lower_), uint64_t(tma_desc.stride2_), (uint64_t(tma_desc.stride2_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
  printf("  stride3:       0x%01x 0x%08" PRIx64 " (%" PRIu64 ") [%" PRIu64 " %s]\n", uint8_t(tma_desc.stride3_.upper_), uint64_t(tma_desc.stride3_.lower_), uint64_t(tma_desc.stride3_), (uint64_t(tma_desc.stride3_) << 4) * 8 / sizeof_bits_format(tma_desc.format_), to_string(tma_desc.format_));
  printf("\n");
  printf("  size0:         0x%08x (%u) [%u]\n", uint32_t(tma_desc.size0_), uint32_t(tma_desc.size0_), uint32_t(tma_desc.size0_)+1);
  printf("  size1:         0x%08x (%u) [%u]\n", uint32_t(tma_desc.size1_), uint32_t(tma_desc.size1_), uint32_t(tma_desc.size1_)+1);
  printf("  size2:         0x%08x (%u) [%u]\n", uint32_t(tma_desc.size2_), uint32_t(tma_desc.size2_), uint32_t(tma_desc.size2_)+1);
  printf("  size3:         0x%08x (%u) [%u]\n", uint32_t(tma_desc.size3_), uint32_t(tma_desc.size3_), uint32_t(tma_desc.size3_)+1);
  printf("  size4:         0x%08x (%u) [%u]\n", uint32_t(tma_desc.size4_), uint32_t(tma_desc.size4_), uint32_t(tma_desc.size4_)+1);
  printf("\n");
  printf("  tstride0:      0x%01x  (%u) [%u]\n", uint8_t(tma_desc.tstride0_), uint8_t(tma_desc.tstride0_), uint8_t(tma_desc.tstride0_)+1);
  printf("  tstride1:      0x%01x  (%u) [%u]\n", uint8_t(tma_desc.tstride1_), uint8_t(tma_desc.tstride1_), uint8_t(tma_desc.tstride1_)+1);
  printf("  tstride2:      0x%01x  (%u) [%u]\n", uint8_t(tma_desc.tstride2_), uint8_t(tma_desc.tstride2_), uint8_t(tma_desc.tstride2_)+1);
  printf("  tstride3:      0x%01x  (%u) [%u]\n", uint8_t(tma_desc.tstride3_), uint8_t(tma_desc.tstride3_), uint8_t(tma_desc.tstride3_)+1);
  printf("  tstride4:      0x%01x  (%u) [%u]\n", uint8_t(tma_desc.tstride4_), uint8_t(tma_desc.tstride4_), uint8_t(tma_desc.tstride4_)+1);
  printf("\n");
  printf("  range_c:          0x%01x  (%u) [%u]\n", uint8_t(tma_desc.range_c_), uint8_t(tma_desc.range_c_), uint8_t(tma_desc.range_c_)+1);
  if (3 == tma_desc.dim_ + 1) {
    printf("  lower_corner_3d_w: 0x%01x  (%d) [%d]\n", int16_t(tma_desc.lower_corner_3d_w_), int16_t(tma_desc.lower_corner_3d_w_), int16_t(tma_desc.lower_corner_3d_w_));
    printf("  upper_corner_3d_w: 0x%01x  (%d) [%d]\n", int16_t(tma_desc.upper_corner_3d_w_), int16_t(tma_desc.upper_corner_3d_w_), int16_t(tma_desc.upper_corner_3d_w_));
  }
  else if (4 == tma_desc.dim_ + 1) {
    printf("  lower_corner_4d_w: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.lower_corner_4d_w_), int8_t(tma_desc.lower_corner_4d_w_), int8_t(tma_desc.lower_corner_4d_w_));
    printf("  lower_corner_4d_h: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.lower_corner_4d_h_), int8_t(tma_desc.lower_corner_4d_h_), int8_t(tma_desc.lower_corner_4d_h_));
    printf("  upper_corner_4d_w: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.upper_corner_4d_w_), int8_t(tma_desc.upper_corner_4d_w_), int8_t(tma_desc.upper_corner_4d_w_));
    printf("  upper_corner_4d_h: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.upper_corner_4d_h_), int8_t(tma_desc.upper_corner_4d_h_), int8_t(tma_desc.upper_corner_4d_h_));
  }
  else {
    printf("  lower_corner_5d_w: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.lower_corner_5d_w_), int8_t(tma_desc.lower_corner_5d_w_), int8_t(tma_desc.lower_corner_5d_w_));
    printf("  lower_corner_5d_h: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.lower_corner_5d_h_), int8_t(tma_desc.lower_corner_5d_h_), int8_t(tma_desc.lower_corner_5d_h_));
    printf("  lower_corner_5d_d: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.lower_corner_5d_d_), int8_t(tma_desc.lower_corner_5d_d_), int8_t(tma_desc.lower_corner_5d_d_));
    printf("  upper_corner_5d_w: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.upper_corner_5d_w_), int8_t(tma_desc.upper_corner_5d_w_), int8_t(tma_desc.upper_corner_5d_w_));
    printf("  upper_corner_5d_h: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.upper_corner_5d_h_), int8_t(tma_desc.upper_corner_5d_h_), int8_t(tma_desc.upper_corner_5d_h_));
    printf("  upper_corner_5d_d: 0x%01x  (%d) [%d]\n", int8_t(tma_desc.upper_corner_5d_d_), int8_t(tma_desc.upper_corner_5d_d_), int8_t(tma_desc.upper_corner_5d_d_));
  }
  printf("  range_ndhw:       0x%01x  (%u) [%u]\n", uint8_t(tma_desc.range_ndhw_), uint8_t(tma_desc.range_ndhw_), uint8_t(tma_desc.range_ndhw_)+1);
#endif // !defined(__CUDACC_RTC__)
}

// TMA MemBar utility for inspecting uint64_t smem_barrier objects
union TmaMemBar
{
  uint64_t data_;

  bit_field< 0, 1> RESERVED;
  bit_field< 1,20> expectedArvCnt_;
  bit_field<21,21> transactionCnt_;
  bit_field<42, 1> lock_;
  bit_field<43,20> arrivalCnt_;
  bit_field<63, 1> phase_bit_;
};
// {$nv-internal-release end}    // Internal TMA and TMA utilities guard

#if defined(CUTE_USE_PUBLIC_TMA_DESCRIPTOR)                // {$nv-internal-release}
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
#else                                                      // {$nv-internal-release begin}
  using TmaDescriptor = TmaDescriptorInternal;
  using Im2ColTmaDescriptor = Im2ColTmaDescriptorInternal;
  #if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  static_assert(
    sizeof(TmaDescriptorInternal) == sizeof(CUtensorMap),
    "CUTLASS 3x expects consistent binary layout for TMA Descriptors.");
  static_assert(
    sizeof(Im2ColTmaDescriptorInternal) == sizeof(CUtensorMap),
    "CUTLASS 3x expects consistent binary layout for TMA Descriptors.");
  #endif // (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
#endif                                                     // {$nv-internal-release end}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Initiates a TensorMap Prefetch
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
prefetch_tma_descriptor(TmaDescriptor const* desc_ptr)
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state space: const or param)
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// {$nv-internal-release begin}

// XXX: This overload should always remain release guarded.
// Builds with CUTE_USE_PUBLIC_TMA_DESCRIPTOR=1 treat both IM2COL and Tiled mode desc as type cuTensorMap
// Builds with CUTE_USE_PUBLIC_TMA_DESCRIPTOR=0 treat them as separate types
// So this overload must only ever be enabled in non-release-guarded code with CUTE_USE_PUBLIC_TMA_DESCRIPTOR=0
#if ! defined(CUTE_USE_PUBLIC_TMA_DESCRIPTOR)
CUTE_HOST_DEVICE
void
prefetch_tma_descriptor(Im2ColTmaDescriptorInternal const* desc_ptr)
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state space: const or param)
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}
#endif // ! defined(CUTE_USE_PUBLIC_TMA_DESCRIPTOR

// {$nv-internal-release end}

// {$nv-internal-release begin}    // Internal TMA and TMA utilities guard
////////////////////////////////////////////////////////////////////////////////////////////////////
/// UTMACCTL (TMA descriptor cache control)
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {
  // TMA descriptor cache control
  __device__ void __nv_ptx_builtin_ocg_cache_tensor_iv(uint64_t tma_desc_ptr);
}

CUTE_HOST_DEVICE
void
invalidate_tma_descriptor(uint64_t const* tma_desc)
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  __nv_ptx_builtin_ocg_cache_tensor_iv(reinterpret_cast<uint64_t>(tma_desc));
#endif
}
// {$nv-internal-release end}    // Internal TMA and TMA utilities guard

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a TensorMap modification (by each field)
////////////////////////////////////////////////////////////////////////////////////////////////////

// Replace tensor pointer directly in GMEM
CUTE_HOST_DEVICE
void
tma_descriptor_replace_addr_in_global_mem(TmaDescriptor const* desc_ptr,
                                          void const* const new_tensor_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint64_t const new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);
  asm volatile (
    "tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
    :: "l"(gmem_int_desc), "l"(new_desc_addr));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

// Replace tensor pointer by bringing the tensormap from GMEM into the shared memory
CUTE_HOST_DEVICE
void
tma_descriptor_replace_addr_in_shared_mem(TmaDescriptor& smem_desc,
                                          void const* const new_tensor_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const new_desc_addr = reinterpret_cast<uint64_t>(new_tensor_ptr);
  asm volatile (
    "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
    :: "r"(smem_int_desc), "l"(new_desc_addr));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

// Replace tensor dims and strides for GEMMs by bringing the tensormap from GMEM into the shared memory
CUTE_HOST_DEVICE
void
tma_descriptor_replace_dims_strides_in_shared_mem(TmaDescriptor                 & smem_desc,
                                                  cute::array<uint32_t, 5> const& prob_shape,
                                                  cute::array<uint64_t, 5> const& prob_stride)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const smem_int64_desc = 0;
  asm volatile (
    "cvt.u64.u32 %0, %1;"
    :: "l"(smem_int64_desc), "r"(smem_int_desc));
  // The operations below can be adjusted to be called by a warp-wide collective if per-thread perf. is an issue {$nv-internal-release}
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[0]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[1]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 2, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[2]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 3, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[3]));
  asm volatile (
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 4, %1;"
    :: "l"(smem_int64_desc), "r"(prob_shape[4]));
  // Strides must be a multiple of 16. Also, stride for the intermost dimension is implicitly 1
  #if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 5)))
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[1]));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[2]));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 2, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[3]));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 3, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[4]));
  #else
  // 4 LSBs are not included
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[1] >> 4));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 1, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[2] >> 4));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 2, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[3] >> 4));
  asm volatile (
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 3, %1;"
    :: "l"(smem_int64_desc), "l"(prob_stride[4] >> 4));
  #endif
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a fused copy and fence operation (needed when modifying tensormap in shared memory)
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
tma_descriptor_cp_fence_release(TmaDescriptor const* gmem_desc_ptr, TmaDescriptor& smem_desc)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
  uint32_t smem_int_desc = cast_smem_ptr_to_uint(&smem_desc);
  asm volatile (
    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
    :: "l"(gmem_int_desc), "r"(smem_int_desc));
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a release fence operation (needed when modifying tensormap directly in GMEM)
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
tma_descriptor_fence_release()
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  asm volatile ("fence.proxy.tensormap::generic.release.gpu;");
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perform a acquire fence operation
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
tma_descriptor_fence_acquire(TmaDescriptor const* desc_ptr)
{
#if defined(CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
    :
    : "l"(gmem_int_desc)
    : "memory");
#else
  CUTE_INVALID_CONTROL_PATH("Using TMA Descriptor modification without CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED and CUDA 12.3");
#endif
}

///////////////////////////////////////////////////////////////////////////////

} // end namespace cute
