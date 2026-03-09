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
//
// {$nv-internal-release file}
//
#include <cute/config.hpp>    // CUTE_HOST_DEVICE

# include <cute/numeric/numeric_types.hpp>
// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#   define CUTE_ARCH_MMA_SPARSE_SM80_ENABLED
#endif

// {$nv-internal-release begin}
////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION NOTES
////////////////////////////////////////////////////////////////////////////////
//
// * do we need mma.sp thread selector 0x2, 0x3?
// * don't want id2 selector to be a dynamic parameter of these Ops.
//     Make congruent with the PTX and consider creation of "entagled atoms"
//     that use the full warp/registers across multiple PTX calls.
// * S4 specializations have not been tested
// * are there any advantages of using extended PTX _mma.sp?
//     the only difference is .spformat that can be set to .thread or .regoffset
//     .thread:    which thread in a quad provides the e0 register
//     .regoffset: which byte provides the e0 register
// {$nv-internal-release end}

namespace cute {

////////////////////////////////////////////////////////////////////////////////
// F16
////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x16 TN
struct SM80_SPARSE_16x8x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1,
      uint32_t  a0, uint32_t  a1,
      uint32_t  b0, uint32_t  b1,
      uint32_t  c0, uint32_t  c1,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0, %1},"
          "{%2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "%8, 0x0;\n"
          : "=r"(d0), "=r"(d1)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0, %1},"
          "{%2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "%8, 0x1;\n"
          : "=r"(d0), "=r"(d1)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x16_F16F16F16F16_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x32 TN
struct SM80_SPARSE_16x8x32_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1,
      uint32_t  a0, uint32_t  a1, uint32_t  a2, uint32_t  a3,
      uint32_t  b0, uint32_t  b1, uint32_t  b2, uint32_t  b3,
      uint32_t  c0, uint32_t  c1,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
          "{%0, %1},"
          "{%2, %3, %4, %5},"
          "{%6, %7, %8, %9},"
          "{%10, %11},"
          "%12, 0x0;\n"
          : "=r"(d0), "=r"(d1)
          : "r"(a0),  "r"(a1), "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1), "r"(b2),  "r"(b3),
            "r"(c0),  "r"(c1),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
          "{%0, %1},"
          "{%2, %3, %4, %5},"
          "{%6, %7, %8, %9},"
          "{%10, %11},"
          "%12, 0x1;\n"
          : "=r"(d0), "=r"(d1)
          : "r"(a0),  "r"(a1), "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1), "r"(b2),  "r"(b3),
            "r"(c0),  "r"(c1),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x32_F16F16F16F16_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x16 TN
struct SM80_SPARSE_16x8x16_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(float&   d0, float&   d1, float& d2, float& d3,
      uint32_t a0, uint32_t a1,
      uint32_t b0, uint32_t b1,
      float    c0, float    c1, float  c2, float  c3,
      uint32_t e0,
      int      id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
    assert(id2 == 0 || id2 == 1);
    if (id2 == 0) {
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5},"
        "{%6, %7},"
        "{%8, %9, %10, %11},"
        "%12, 0x0;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0),  "r"(a1),
          "r"(b0),  "r"(b1),
          "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
          "r"(e0));
    }
    else {
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5},"
        "{%6, %7},"
        "{%8, %9, %10, %11},"
        "%12, 0x1;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0),  "r"(a1),
          "r"(b0),  "r"(b1),
          "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
          "r"(e0));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use "
                              "SM80_SPARSE_16x8x16_F32F16F16F32_TN "
                              "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x32 TN
struct SM80_SPARSE_16x8x32_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(float&   d0, float&   d1, float&   d2, float&   d3,
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
      uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
      float    c0, float    c1, float    c2, float    c3,
      uint32_t e0,
      int      id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
          "{%0,   %1,  %2,  %3},"
          "{%4,   %5,  %6,  %7},"
          "{%8,   %9, %10, %11},"
          "{%12, %13, %14, %15},"
          "%16, 0x0;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
          "{%0,   %1,  %2,  %3},"
          "{%4,   %5,  %6,  %7},"
          "{%8,   %9, %10, %11},"
          "{%12, %13, %14, %15},"
          "%16, 0x1;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x32_F32F16F16F32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// BF16
////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x16 TN
struct SM80_SPARSE_16x8x16_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(float&   d0, float&   d1, float& d2, float& d3,
      uint32_t a0, uint32_t a1,
      uint32_t b0, uint32_t b1,
      float    c0, float    c1, float  c2, float  c3,
      uint32_t e0,
      int      id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x0;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x1;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x16_F32BF16BF16F32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x32 TN
struct SM80_SPARSE_16x8x32_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(float&   d0, float&   d1, float&   d2, float&   d3,
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
      uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
      float    c0, float    c1, float    c2, float    c3,
      uint32_t e0,
      int      id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
          "{%0,   %1,  %2,  %3},"
          "{%4,   %5,  %6,  %7},"
          "{%8,   %9, %10, %11},"
          "{%12, %13, %14, %15},"
          "%16, 0x0;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
          "{%0,   %1,  %2,  %3},"
          "{%4,   %5,  %6,  %7},"
          "{%8,   %9, %10, %11},"
          "{%12, %13, %14, %15},"
          "%16, 0x1;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x32_F32BF16BF16F32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// TF32
////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x8 TN
struct SM80_SPARSE_16x8x8_F32TF32TF32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(float&   d0, float&   d1, float& d2, float& d3,
      uint32_t a0, uint32_t a1,
      uint32_t b0, uint32_t b1,
      float    c0, float    c1, float  c2, float  c3,
      uint32_t e0,
      int      id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x0;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x1;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x8_F32TF32TF32F32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x16 TN
struct SM80_SPARSE_16x8x16_F32TF32TF32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(float&   d0, float&   d1, float&   d2, float&   d3,
      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
      uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
      float    c0, float    c1, float    c2, float    c3,
      uint32_t e0,
      int      id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
          "{%0, %1, %2, %3},"
          "{%4, %5, %6, %7},"
          "{%8, %9, %10, %11},"
          "{%12, %13, %14, %15},"
          "%16, 0x0;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1), "r"(a2),   "r"(a3),
            "r"(b0),  "r"(b1), "r"(b2),   "r"(b3),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
          "{%0, %1, %2, %3},"
          "{%4, %5, %6, %7},"
          "{%8, %9, %10, %11},"
          "{%12, %13, %14, %15},"
          "%16, 0x1;\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
            "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
            "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x16_F32TF32TF32F32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// INTEGER TYPES (S8)
////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x32 TN
struct SM80_SPARSE_16x8x32_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1,
      uint32_t  b0, uint32_t  b1,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x0;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x1;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x32_S32S8S8S32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x32 TN
struct SM80_SPARSE_16x8x32_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1,
      uint32_t  b0, uint32_t  b1,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x0;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x1;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x32_S32S8S8S32_TN_SATURATE "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x64 TN
struct SM80_SPARSE_16x8x64_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1, uint32_t  a2, uint32_t  a3,
      uint32_t  b0, uint32_t  b1, uint32_t  b2, uint32_t  b3,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
    assert(id2 == 0);
    asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9, %10, %11},"
      "{%12, %13, %14, %15},"
      "%16, 0x0;\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
      "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
      "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
      "r"(e0));
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x64_S32S8S8S32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x64 TN
struct SM80_SPARSE_16x8x64_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1, uint32_t  a2, uint32_t  a3,
      uint32_t  b0, uint32_t  b1, uint32_t  b2, uint32_t  b3,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0);
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9, %10, %11},"
        "{%12, %13, %14, %15},"
        "%16, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
          "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
          "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
          "r"(e0));
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x64_S32S8S8S32_TN_SATURATE "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// INTEGER TYPES (S4)
////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x64 TN
struct SM80_SPARSE_16x8x64_S32S4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1,
      uint32_t  b0, uint32_t  b1,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x0;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x1;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x64_S32S4S4S32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x32 TN
struct SM80_SPARSE_16x8x64_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[2];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1,
      uint32_t  b0, uint32_t  b1,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0 || id2 == 1);
      if (id2 == 0) {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x0;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
      else {
        asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6, %7},"
          "{%8, %9, %10, %11},"
          "%12, 0x1;\n"
          : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
          : "r"(a0),  "r"(a1),
            "r"(b0),  "r"(b1),
            "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
            "r"(e0));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x64_S32S4S4S32_TN_SATURATE "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x64 TN
struct SM80_SPARSE_16x8x128_S32S4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1, uint32_t  a2, uint32_t  a3,
      uint32_t  b0, uint32_t  b1, uint32_t  b2, uint32_t  b3,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0);
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9, %10, %11},"
        "{%12, %13, %14, %15},"
        "%16, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
          "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
          "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
          "r"(e0));
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x128_S32S4S4S32_TN "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

// Sparse MMA 16x8x64 TN
struct SM80_SPARSE_16x8x128_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using ERegisters = uint32_t[1];
  using BRegisters = uint32_t[4];
  using CRegisters = DRegisters;

  CUTE_HOST_DEVICE static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t  a0, uint32_t  a1, uint32_t  a2, uint32_t  a3,
      uint32_t  b0, uint32_t  b1, uint32_t  b2, uint32_t  b3,
      uint32_t  c0, uint32_t  c1, uint32_t  c2, uint32_t  c3,
      uint32_t  e0,
      int       id2 = 0)
  {
#if defined(CUTE_ARCH_MMA_SPARSE_SM80_ENABLED)
      assert(id2 == 0);
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.satfinite.s32.s4.s4.s32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9, %10, %11},"
        "{%12, %13, %14, %15},"
        "%16, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
          "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
          "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3),
          "r"(e0));
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use "
                                "SM80_SPARSE_16x8x128_S32S4S4S32_TN_SATURATE "
                                "without CUTE_ARCH_MMA_SPARSE_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cute
