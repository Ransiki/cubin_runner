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
//
// Since SM90's MMA_Atom defines ScaleIn as template constants,
// The C <- C - A * B subvariant cannot be directly obtained from MMA_Atom for C <- C + A * B
// Hence, this file defines a type MMA_Atom_PlanarComplex2Complex to warp two MMA Atoms into one class
//
// make_tiled_mma is concequently defined over this new type MMA_Atom_PlanarComplex2Complex as
// make_composite_tiled_mma with exactly the same behavior as in mma_atom.hpp

#pragma once

#include "cutlass/cutlass.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM90::GMMA {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Overload MMA Ops to do complex MMAs

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 C32+=F16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x128x16_C32F16F16_SS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint64_t[2];
  using BRegisters = uint64_t[2];
  using CRegisters = float[128];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a_re, uint64_t const& desc_a_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      float& d16_re, float& d16_im, float& d17_re, float& d17_im, float& d18_re, float& d18_im, float& d19_re, float& d19_im,
      float& d20_re, float& d20_im, float& d21_re, float& d21_im, float& d22_re, float& d22_im, float& d23_re, float& d23_im,
      float& d24_re, float& d24_im, float& d25_re, float& d25_im, float& d26_re, float& d26_im, float& d27_re, float& d27_im,
      float& d28_re, float& d28_im, float& d29_re, float& d29_im, float& d30_re, float& d30_im, float& d31_re, float& d31_im,
      float& d32_re, float& d32_im, float& d33_re, float& d33_im, float& d34_re, float& d34_im, float& d35_re, float& d35_im,
      float& d36_re, float& d36_im, float& d37_re, float& d37_im, float& d38_re, float& d38_im, float& d39_re, float& d39_im,
      float& d40_re, float& d40_im, float& d41_re, float& d41_im, float& d42_re, float& d42_im, float& d43_re, float& d43_im,
      float& d44_re, float& d44_im, float& d45_re, float& d45_im, float& d46_re, float& d46_im, float& d47_re, float& d47_im,
      float& d48_re, float& d48_im, float& d49_re, float& d49_im, float& d50_re, float& d50_im, float& d51_re, float& d51_im,
      float& d52_re, float& d52_im, float& d53_re, float& d53_im, float& d54_re, float& d54_im, float& d55_re, float& d55_im,
      float& d56_re, float& d56_im, float& d57_re, float& d57_im, float& d58_re, float& d58_im, float& d59_re, float& d59_im,
      float& d60_re, float& d60_im, float& d61_re, float& d61_im, float& d62_re, float& d62_im, float& d63_re, float& d63_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x128x16_F32F16F16_SS<tnspA, tnspB>;
    using Neg = MMA_64x128x16_F32F16F16_SS<tnspA, tnspB, GMMA::ScaleIn::Neg>;

    Pos::fma(
        desc_a_re, desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        d48_re, d49_re, d50_re, d51_re, d52_re, d53_re, d54_re, d55_re,
        d56_re, d57_re, d58_re, d59_re, d60_re, d61_re, d62_re, d63_re,
        scale_D);
    Pos::fma(
        desc_a_re, desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        d48_im, d49_im, d50_im, d51_im, d52_im, d53_im, d54_im, d55_im,
        d56_im, d57_im, d58_im, d59_im, d60_im, d61_im, d62_im, d63_im,
        scale_D);
    Neg::fma(
        desc_a_im, desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        d48_re, d49_re, d50_re, d51_re, d52_re, d53_re, d54_re, d55_re,
        d56_re, d57_re, d58_re, d59_re, d60_re, d61_re, d62_re, d63_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        desc_a_im, desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        d48_im, d49_im, d50_im, d51_im, d52_im, d53_im, d54_im, d55_im,
        d56_im, d57_im, d58_im, d59_im, d60_im, d61_im, d62_im, d63_im,
        GMMA::ScaleOut::One);
  }
};

// GMMA 64x128x16 C32+=C16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x128x16_C32C16F16_RS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint32_t[8];
  using BRegisters = uint64_t[2];
  using CRegisters = float[128];

  static_assert(tnspA == GMMA::Major::K, "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a00_re, uint32_t const& a00_im, uint32_t const& a01_re, uint32_t const& a01_im,
      uint32_t const& a02_re, uint32_t const& a02_im, uint32_t const& a03_re, uint32_t const& a03_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      float& d16_re, float& d16_im, float& d17_re, float& d17_im, float& d18_re, float& d18_im, float& d19_re, float& d19_im,
      float& d20_re, float& d20_im, float& d21_re, float& d21_im, float& d22_re, float& d22_im, float& d23_re, float& d23_im,
      float& d24_re, float& d24_im, float& d25_re, float& d25_im, float& d26_re, float& d26_im, float& d27_re, float& d27_im,
      float& d28_re, float& d28_im, float& d29_re, float& d29_im, float& d30_re, float& d30_im, float& d31_re, float& d31_im,
      float& d32_re, float& d32_im, float& d33_re, float& d33_im, float& d34_re, float& d34_im, float& d35_re, float& d35_im,
      float& d36_re, float& d36_im, float& d37_re, float& d37_im, float& d38_re, float& d38_im, float& d39_re, float& d39_im,
      float& d40_re, float& d40_im, float& d41_re, float& d41_im, float& d42_re, float& d42_im, float& d43_re, float& d43_im,
      float& d44_re, float& d44_im, float& d45_re, float& d45_im, float& d46_re, float& d46_im, float& d47_re, float& d47_im,
      float& d48_re, float& d48_im, float& d49_re, float& d49_im, float& d50_re, float& d50_im, float& d51_re, float& d51_im,
      float& d52_re, float& d52_im, float& d53_re, float& d53_im, float& d54_re, float& d54_im, float& d55_re, float& d55_im,
      float& d56_re, float& d56_im, float& d57_re, float& d57_im, float& d58_re, float& d58_im, float& d59_re, float& d59_im,
      float& d60_re, float& d60_im, float& d61_re, float& d61_im, float& d62_re, float& d62_im, float& d63_re, float& d63_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x128x16_F32F16F16_RS<tnspA, tnspB>;
    using Neg = MMA_64x128x16_F32F16F16_RS<tnspA, tnspB, GMMA::ScaleIn::One, GMMA::ScaleIn::Neg>;

    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        d48_re, d49_re, d50_re, d51_re, d52_re, d53_re, d54_re, d55_re,
        d56_re, d57_re, d58_re, d59_re, d60_re, d61_re, d62_re, d63_re,
        scale_D);
    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        d48_im, d49_im, d50_im, d51_im, d52_im, d53_im, d54_im, d55_im,
        d56_im, d57_im, d58_im, d59_im, d60_im, d61_im, d62_im, d63_im,
        scale_D);
    Neg::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        d48_re, d49_re, d50_re, d51_re, d52_re, d53_re, d54_re, d55_re,
        d56_re, d57_re, d58_re, d59_re, d60_re, d61_re, d62_re, d63_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        d48_im, d49_im, d50_im, d51_im, d52_im, d53_im, d54_im, d55_im,
        d56_im, d57_im, d58_im, d59_im, d60_im, d61_im, d62_im, d63_im,
        GMMA::ScaleOut::One);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 C32+=F16*F1 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x96x16_C32F16F16_SS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint64_t[2];
  using BRegisters = uint64_t[2];
  using CRegisters = float[96];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a_re, uint64_t const& desc_a_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      float& d16_re, float& d16_im, float& d17_re, float& d17_im, float& d18_re, float& d18_im, float& d19_re, float& d19_im,
      float& d20_re, float& d20_im, float& d21_re, float& d21_im, float& d22_re, float& d22_im, float& d23_re, float& d23_im,
      float& d24_re, float& d24_im, float& d25_re, float& d25_im, float& d26_re, float& d26_im, float& d27_re, float& d27_im,
      float& d28_re, float& d28_im, float& d29_re, float& d29_im, float& d30_re, float& d30_im, float& d31_re, float& d31_im,
      float& d32_re, float& d32_im, float& d33_re, float& d33_im, float& d34_re, float& d34_im, float& d35_re, float& d35_im,
      float& d36_re, float& d36_im, float& d37_re, float& d37_im, float& d38_re, float& d38_im, float& d39_re, float& d39_im,
      float& d40_re, float& d40_im, float& d41_re, float& d41_im, float& d42_re, float& d42_im, float& d43_re, float& d43_im,
      float& d44_re, float& d44_im, float& d45_re, float& d45_im, float& d46_re, float& d46_im, float& d47_re, float& d47_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x96x16_F32F16F16_SS<tnspA, tnspB>;
    using Neg = MMA_64x96x16_F32F16F16_SS<tnspA, tnspB, GMMA::ScaleIn::Neg>;

    Pos::fma(
        desc_a_re, desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        scale_D);
    Pos::fma(
        desc_a_re, desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        scale_D);
    Neg::fma(
        desc_a_im, desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        desc_a_im, desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        GMMA::ScaleOut::One);
  }
};

// GMMA 64x96x16 C32+=C16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x96x16_C32C16F16_RS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint32_t[8];
  using BRegisters = uint64_t[2];
  using CRegisters = float[96];

  static_assert(tnspA == GMMA::Major::K, "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a00_re, uint32_t const& a00_im, uint32_t const& a01_re, uint32_t const& a01_im,
      uint32_t const& a02_re, uint32_t const& a02_im, uint32_t const& a03_re, uint32_t const& a03_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      float& d16_re, float& d16_im, float& d17_re, float& d17_im, float& d18_re, float& d18_im, float& d19_re, float& d19_im,
      float& d20_re, float& d20_im, float& d21_re, float& d21_im, float& d22_re, float& d22_im, float& d23_re, float& d23_im,
      float& d24_re, float& d24_im, float& d25_re, float& d25_im, float& d26_re, float& d26_im, float& d27_re, float& d27_im,
      float& d28_re, float& d28_im, float& d29_re, float& d29_im, float& d30_re, float& d30_im, float& d31_re, float& d31_im,
      float& d32_re, float& d32_im, float& d33_re, float& d33_im, float& d34_re, float& d34_im, float& d35_re, float& d35_im,
      float& d36_re, float& d36_im, float& d37_re, float& d37_im, float& d38_re, float& d38_im, float& d39_re, float& d39_im,
      float& d40_re, float& d40_im, float& d41_re, float& d41_im, float& d42_re, float& d42_im, float& d43_re, float& d43_im,
      float& d44_re, float& d44_im, float& d45_re, float& d45_im, float& d46_re, float& d46_im, float& d47_re, float& d47_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x96x16_F32F16F16_RS<tnspA, tnspB>;
    using Neg = MMA_64x96x16_F32F16F16_RS<tnspA, tnspB, GMMA::ScaleIn::One, GMMA::ScaleIn::Neg>;

    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        scale_D);
    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        scale_D);
    Neg::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        d32_re, d33_re, d34_re, d35_re, d36_re, d37_re, d38_re, d39_re,
        d40_re, d41_re, d42_re, d43_re, d44_re, d45_re, d46_re, d47_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        d32_im, d33_im, d34_im, d35_im, d36_im, d37_im, d38_im, d39_im,
        d40_im, d41_im, d42_im, d43_im, d44_im, d45_im, d46_im, d47_im,
        GMMA::ScaleOut::One);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 C32+=F16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x64x16_C32F16F16_SS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint64_t[2];
  using BRegisters = uint64_t[2];
  using CRegisters = float[64];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a_re, uint64_t const& desc_a_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      float& d16_re, float& d16_im, float& d17_re, float& d17_im, float& d18_re, float& d18_im, float& d19_re, float& d19_im,
      float& d20_re, float& d20_im, float& d21_re, float& d21_im, float& d22_re, float& d22_im, float& d23_re, float& d23_im,
      float& d24_re, float& d24_im, float& d25_re, float& d25_im, float& d26_re, float& d26_im, float& d27_re, float& d27_im,
      float& d28_re, float& d28_im, float& d29_re, float& d29_im, float& d30_re, float& d30_im, float& d31_re, float& d31_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x64x16_F32F16F16_SS<tnspA, tnspB>;
    using Neg = MMA_64x64x16_F32F16F16_SS<tnspA, tnspB, GMMA::ScaleIn::Neg>;

    Pos::fma(
        desc_a_re, desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        scale_D);
    Pos::fma(
        desc_a_re, desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        scale_D);
    Neg::fma(
        desc_a_im, desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        desc_a_im, desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        GMMA::ScaleOut::One);
  }
};

// GMMA 64x64x16 C32+=C16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x64x16_C32C16F16_RS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint32_t[8];
  using BRegisters = uint64_t[2];
  using CRegisters = float[64];

  static_assert(tnspA == GMMA::Major::K, "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a00_re, uint32_t const& a00_im, uint32_t const& a01_re, uint32_t const& a01_im,
      uint32_t const& a02_re, uint32_t const& a02_im, uint32_t const& a03_re, uint32_t const& a03_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      float& d16_re, float& d16_im, float& d17_re, float& d17_im, float& d18_re, float& d18_im, float& d19_re, float& d19_im,
      float& d20_re, float& d20_im, float& d21_re, float& d21_im, float& d22_re, float& d22_im, float& d23_re, float& d23_im,
      float& d24_re, float& d24_im, float& d25_re, float& d25_im, float& d26_re, float& d26_im, float& d27_re, float& d27_im,
      float& d28_re, float& d28_im, float& d29_re, float& d29_im, float& d30_re, float& d30_im, float& d31_re, float& d31_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x64x16_F32F16F16_RS<tnspA, tnspB>;
    using Neg = MMA_64x64x16_F32F16F16_RS<tnspA, tnspB, GMMA::ScaleIn::One, GMMA::ScaleIn::Neg>;

    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        scale_D);
    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        scale_D);
    Neg::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        d16_re, d17_re, d18_re, d19_re, d20_re, d21_re, d22_re, d23_re,
        d24_re, d25_re, d26_re, d27_re, d28_re, d29_re, d30_re, d31_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        d16_im, d17_im, d18_im, d19_im, d20_im, d21_im, d22_im, d23_im,
        d24_im, d25_im, d26_im, d27_im, d28_im, d29_im, d30_im, d31_im,
        GMMA::ScaleOut::One);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 C32+=F16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x32x16_C32F16F16_SS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint64_t[2];
  using BRegisters = uint64_t[2];
  using CRegisters = float[32];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a_re, uint64_t const& desc_a_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x32x16_F32F16F16_SS<tnspA, tnspB>;
    using Neg = MMA_64x32x16_F32F16F16_SS<tnspA, tnspB, GMMA::ScaleIn::Neg>;

    Pos::fma(
        desc_a_re, desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        scale_D);
    Pos::fma(
        desc_a_re, desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        scale_D);
    Neg::fma(
        desc_a_im, desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        desc_a_im, desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        GMMA::ScaleOut::One);
  }
};

// GMMA 64x32x16 C32+=C16*F16 (PLANAR)
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
struct MMA_64x32x16_C32C16F16_RS_PLANAR2C
{
  using DRegisters = void;
  using ARegisters = uint32_t[8];
  using BRegisters = uint64_t[2];
  using CRegisters = float[32];

  static_assert(tnspA == GMMA::Major::K, "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a00_re, uint32_t const& a00_im, uint32_t const& a01_re, uint32_t const& a01_im,
      uint32_t const& a02_re, uint32_t const& a02_im, uint32_t const& a03_re, uint32_t const& a03_im,
      uint64_t const& desc_b_re, uint64_t const& desc_b_im,
      float& d00_re, float& d00_im, float& d01_re, float& d01_im, float& d02_re, float& d02_im, float& d03_re, float& d03_im,
      float& d04_re, float& d04_im, float& d05_re, float& d05_im, float& d06_re, float& d06_im, float& d07_re, float& d07_im,
      float& d08_re, float& d08_im, float& d09_re, float& d09_im, float& d10_re, float& d10_im, float& d11_re, float& d11_im,
      float& d12_re, float& d12_im, float& d13_re, float& d13_im, float& d14_re, float& d14_im, float& d15_re, float& d15_im,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
    using Pos = MMA_64x32x16_F32F16F16_RS<tnspA, tnspB>;
    using Neg = MMA_64x32x16_F32F16F16_RS<tnspA, tnspB, GMMA::ScaleIn::One, GMMA::ScaleIn::Neg>;

    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_re,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        scale_D);
    Pos::fma(
        a00_re, a01_re, a02_re, a03_re,
        desc_b_im,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        scale_D);
    Neg::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_im,
        d00_re, d01_re, d02_re, d03_re, d04_re, d05_re, d06_re, d07_re,
        d08_re, d09_re, d10_re, d11_re, d12_re, d13_re, d14_re, d15_re,
        GMMA::ScaleOut::One);
    Pos::fma(
        a00_im, a01_im, a02_im, a03_im,
        desc_b_re,
        d00_im, d01_im, d02_im, d03_im, d04_im, d05_im, d06_im, d07_im,
        d08_im, d09_im, d10_im, d11_im, d12_im, d13_im, d14_im, d15_im,
        GMMA::ScaleOut::One);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Real_Op,
  GMMA::Major MajorA = GMMA::Major::K,
  GMMA::Major MajorB = GMMA::Major::K
>
CUTE_HOST_DEVICE constexpr
auto
complex_mma_op_lookup() {
  if constexpr (is_same_v<Real_Op, MMA_64x128x16_F32F16F16_SS<MajorA, MajorB>>) {
    return MMA_64x128x16_C32F16F16_SS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x128x16_F32F16F16_RS<MajorA, MajorB>>) {
    return MMA_64x128x16_C32C16F16_RS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x96x16_F32F16F16_SS<MajorA, MajorB>>) {
    return MMA_64x96x16_C32F16F16_SS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x96x16_F32F16F16_RS<MajorA, MajorB>>) {
    return MMA_64x96x16_C32C16F16_RS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x64x16_F32F16F16_SS<MajorA, MajorB>>) {
    return MMA_64x64x16_C32F16F16_SS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x64x16_F32F16F16_RS<MajorA, MajorB>>) {
    return MMA_64x64x16_C32C16F16_RS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x32x16_F32F16F16_SS<MajorA, MajorB>>) {
    return MMA_64x32x16_C32F16F16_SS_PLANAR2C<MajorA, MajorB>{};
  }
  else if constexpr (is_same_v<Real_Op, MMA_64x32x16_F32F16F16_RS<MajorA, MajorB>>) {
    return MMA_64x32x16_C32C16F16_RS_PLANAR2C<MajorA, MajorB>{};
  }
  else {
    static_assert(sizeof(Real_Op) == 1024, "No eligible complex GMMA operator.");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // SM90::GMMA

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x128x16_C32F16F16_SS_PLANAR2C = SM90::GMMA::MMA_64x128x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x128x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x128x16_C32C16F16_RS_PLANAR2C = SM90::GMMA::MMA_64x128x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x128x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  // complex<half_t> should be reordered as complex<Array<half_t,2>>, declare Array<half_t,2> to avoid size conflict. {$nv-internal-release}
  using ValTypeA = cutlass::Array<half_t, 2>;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_128,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<128, 16>;
  using CLayout = GMMA::CLayout_64x128;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x96x16_C32F16F16_SS_PLANAR2C = SM90::GMMA::MMA_64x96x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x96x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x96x16_C32C16F16_RS_PLANAR2C = SM90::GMMA::MMA_64x96x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x96x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  // complex<half_t> should be reordered as complex<Array<half_t,2>>, declare Array<half_t,2> to avoid size conflict. {$nv-internal-release}
  using ValTypeA = cutlass::Array<half_t, 2>;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_96,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<96, 16>;
  using CLayout = GMMA::CLayout_64x96;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x64x16_C32F16F16_SS_PLANAR2C = SM90::GMMA::MMA_64x64x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x64x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout<64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x64x16_C32C16F16_RS_PLANAR2C = SM90::GMMA::MMA_64x64x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x64x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  // complex<half_t> should be reordered as complex<Array<half_t,2>>, declare Array<half_t,2> to avoid size conflict. {$nv-internal-release}
  using ValTypeA = cutlass::Array<half_t, 2>;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_64,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<64, 16>;
  using CLayout = GMMA::CLayout_64x64;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x32x16_C32F16F16_SS_PLANAR2C = SM90::GMMA::MMA_64x32x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x32x16_C32F16F16_SS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using BLayout = GMMA::ABLayout< 32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <
  GMMA::Major tnspA,
  GMMA::Major tnspB
>
using SM90_64x32x16_C32C16F16_RS_PLANAR2C = SM90::GMMA::MMA_64x32x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>;

template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x32x16_C32C16F16_RS_PLANAR2C<tnspA, tnspB>>
{
  using ValTypeD = cutlass::complex<float>;
  // complex<half_t> should be reordered as complex<Array<half_t,2>>, declare Array<half_t,2> to avoid size conflict. {$nv-internal-release}
  using ValTypeA = cutlass::Array<half_t, 2>;
  using ValTypeB = half_t;
  using ValTypeC = cutlass::complex<float>;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_32,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using BLayout = GMMA::ABLayout<32, 16>;
  using CLayout = GMMA::CLayout_64x32;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cute

/////////////////////////////////////////////////////////////////////////////////////////////////
