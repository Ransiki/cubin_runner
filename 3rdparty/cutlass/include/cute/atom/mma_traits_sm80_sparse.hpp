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
#include <cute/arch/mma_sm80_sparse.hpp>    // SM80_SPARSE_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_traits.hpp>         // MMA_Traits

#include <cute/numeric/numeric_types.hpp>   // cute::int4b_t
#include <cute/pointer_sparse.hpp>          // cute::sparse_elem

#include <cute/layout.hpp>                  // Layout


////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION NOTES
////////////////////////////////////////////////////////////////////////////////
//
// * we use the full indices for the sparse matrix A ->
//   double of the number of elements per thread
//

namespace cute
{

namespace {

// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

////////////////////////////////////////////////////////////////////////////////
//////////////////////// fp16 = fp16 * fp16 + fp16 /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_SPARSE_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD  = half_t;
  using ValTypeA  = sparse_elem<2, half_t>;
  using ValTypeE  = sparse_elem<16, uint16_t>;
  using ValTypeB  = half_t;
  using ValTypeC  = half_t;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID     = Layout<_32>;
  // (T32,V8) -> (M16,K16)
  using ALayout   = Layout<Shape <Shape < _4, _8>, Shape < _4, _2>>,
                           Stride<Stride<_64, _1>, Stride<_16, _8>>>;
  // (T8,V32) -> (M16,K16)
  using ELayout   = Layout<Shape <Shape <_4, _8>, Shape<_16, _2>>,
                           Stride<Stride<_0, _1>, Stride<_0, _8>>>;
  // B, C, D same as mma.m16n8k16
  using BLayout   = Layout<Shape <Shape < _4, _8>, Shape <_2,  _2>>,
                           Stride<Stride<_16, _1>, Stride<_8, _64>>>;
  using CLayout   = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x32_F16F16F16F16_TN>
{
  using ValTypeD  = half_t;
  using ValTypeA  = sparse_elem<2, half_t>;
  using ValTypeE  = sparse_elem<16, uint16_t>;
  using ValTypeB  = half_t;
  using ValTypeC  = half_t;

  using Shape_MNK = Shape<_16, _8, _32>;
  using ThrID     = Layout<_32>;
  // (T32,V8) -> (M16,K32)
  using ALayout   = Layout<Shape <Shape < _4, _8>, Shape < _4, _2,   _2>>,
                           Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;
  // (T16,V32) -> (M16,K32)
  using ELayout   = Layout<Shape <Shape <  _2, _2, _8>, Shape<_16, _2>>,
                           Stride<Stride<_256, _0, _1>, Stride<_0, _8>>>;
  // (T32,V8) -> (N8,K32)
  using BLayout   = Layout<Shape <Shape < _4, _8>, Shape <_2,  _4>>,
                           Stride<Stride<_16, _1>, Stride<_8, _64>>>;
  // C, D same as mma.m16n8k16
  using CLayout   = SM80_16x8_Row;
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = fp16 * fp16 + fp32 /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_SPARSE_16x8x16_F32F16F16F32_TN>
     : MMA_Traits<SM80_SPARSE_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x32_F32F16F16F32_TN>
     : MMA_Traits<SM80_SPARSE_16x8x32_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeC = float;
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = bf16 * bf16 + fp32 /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_SPARSE_16x8x16_F32BF16BF16F32_TN>
     : MMA_Traits<SM80_SPARSE_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<16, uint16_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x32_F32BF16BF16F32_TN>
     : MMA_Traits<SM80_SPARSE_16x8x32_F16F16F16F16_TN>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<16, uint16_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

////////////////////////////////////////////////////////////////////////////////
//////////////////////// fp32 = tf32 * tf32 + fp32 /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_SPARSE_16x8x8_F32TF32TF32F32_TN>
{
  using ValTypeD  = float;
  using ValTypeA  = sparse_elem<2, cutlass::tfloat32_t>;
  using ValTypeE  = sparse_elem<8, uint16_t>;
  using ValTypeB  = cutlass::tfloat32_t;
  using ValTypeC  = float;

  using Shape_MNK = Shape<_16, _8, _8>;
  using ThrID     = Layout<_32>;
  // (T32,V4) -> (M16,K8)
  using ALayout   = Layout<Shape <Shape < _4, _8>, Shape < _2, _2>>,
                           Stride<Stride<_32, _1>, Stride<_16, _8>>>;
  // (T8,V32) -> (M16,K8)
  using ELayout   = Layout<Shape <Shape <_4, _8>, Shape< _8, _2>>,
                           Stride<Stride<_0, _1>, Stride<_0, _8>>>;
  // B, C, D same as mma.m16n8k8
  using BLayout   = Layout<Shape<Shape <_4, _8>,  _2>,
                          Stride<Stride<_8, _1>, _32>>;
  using CLayout   = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x16_F32TF32TF32F32_TN>
{
  using ValTypeD  = float;
  using ValTypeA  = sparse_elem<2, cutlass::tfloat32_t>;
  using ValTypeE  = sparse_elem<8, uint16_t>;
  using ValTypeB  = cutlass::tfloat32_t;
  using ValTypeC  = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID     = Layout<_32>;
  // (T32,V8) -> (M16,K16)
  using ALayout   = Layout<Shape <Shape < _4, _8>, Shape < _2, _2,   _2>>,
                           Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
  // (T16,V32) -> (M16,K16)
  using ELayout   = Layout<Shape <Shape <  _2, _2, _8>, Shape< _8, _2>>,
                            Stride<Stride<_128, _0, _1>, Stride<_0, _8>>>;
  // (T32,V8) -> (N8,K16)
  using BLayout   = Layout<Shape <Shape <_4, _8>,  _4>,
                           Stride<Stride<_8, _1>, _32>>;
  // C, D same as mma.m16n8k16
  using CLayout   = SM80_16x8_Row;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s8 * s8 + s32 ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<SM80_SPARSE_16x8x32_S32S8S8S32_TN>
{
  using ValTypeD  = int32_t;
  using ValTypeA  = sparse_elem<2, int8_t>;
  using ValTypeE  = sparse_elem<32, uint32_t>;
  using ValTypeB  = int8_t;
  using ValTypeC  = int32_t;

  using Shape_MNK = Shape<_16, _8, _32>;
  using ThrID     = Layout<_32>;
  // (T32,V16) -> (M16,K32)
  using ALayout   = Layout<Shape <Shape <  _4, _8>, Shape < _8, _2>>,
                           Stride<Stride<_128, _1>, Stride<_16, _8>>>;
  // (T8,V32) -> (M16,K32)
  using ELayout   = Layout<Shape <Shape< _2, _2, _8>, _32>,
                           Stride<Stride<_8, _0, _1>, _0>>;
  // B, C, D same as mma.m16n8k32
  using BLayout   = Layout<Shape <Shape < _4, _8>, Shape <_4,   _2>>,
                           Stride<Stride<_32, _1>, Stride<_8, _128>>>;
  using CLayout   = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x32_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_SPARSE_16x8x32_S32S8S8S32_TN> {};

//------------------------------------------------------------------------------

template <>
struct MMA_Traits<SM80_SPARSE_16x8x64_S32S8S8S32_TN>
{
  using ValTypeD  = int32_t;
  using ValTypeA  = sparse_elem<2, int8_t>;
  using ValTypeE  = sparse_elem<32, uint32_t>;
  using ValTypeB  = int8_t;
  using ValTypeC  = int32_t;

  using Shape_MNK = Shape<_16, _8, _64>;
  using ThrID     = Layout<_32>;
  // (T32,V16) -> (M16,K64)
  using ALayout   = Layout<Shape <Shape <  _4, _8>, Shape < _8, _2,   _2>>,
                           Stride<Stride<_128, _1>, Stride<_16, _8, _512>>>;
  // (T32,V32) -> (M16,K64)
  using ELayout   = Layout<Shape <Shape< _2,   _2, _8>, _32>,
                           Stride<Stride<_8, _512, _1>,  _0>>;
  // (T32,V16) -> (N8,K64)
  using BLayout   = Layout<Shape <Shape < _4, _8>, Shape <_4,   _4>>,
                           Stride<Stride<_32, _1>, Stride<_8, _128>>>;
  // C, D same as mma.m16n8k16
  using CLayout   = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x64_S32S8S8S32_TN_SATURATE>
     : MMA_Traits<SM80_SPARSE_16x8x64_S32S8S8S32_TN> {};


////////////////////////////////////////////////////////////////////////////////
/////////////////////////// s32 = s4 * s4 + s32 ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// NOT TESTED (waiting for sparse_elm + subbyte)
//

template <>
struct MMA_Traits<SM80_SPARSE_16x8x64_S32S4S4S32_TN>
{
  using ValTypeD  = int32_t;
  using ValTypeA  = sparse_elem<2, cute::int4b_t>;
  using ValTypeE  = sparse_elem<64, uint32_t>;
  using ValTypeB  = cute::int4b_t;
  using ValTypeC  = int32_t;

  using Shape_MNK = Shape<_16, _8, _64>;
  using ThrID     = Layout<_32>;
  // (T32,V32) -> (M16,K64)
  using ALayout   = Layout<Shape <Shape <  _4, _8>, Shape <_16, _2>>,
                           Stride<Stride<_256, _1>, Stride<_16, _8>>>;
  // (T16,V64) -> (M16,K64)
  using ELayout   = Layout<Shape <Shape< _2, _2, _8>, _64>,
                           Stride<Stride<_8, _0, _1>,  _0>>;
  // B, C, D same as mma.m16n8k64
  using BLayout   = Layout<Shape <Shape < _4, _8>, Shape <_8,   _2>>,
                           Stride<Stride<_64, _1>, Stride<_8, _256>>>;
  using CLayout   = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x64_S32S4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_SPARSE_16x8x64_S32S4S4S32_TN> {};

//------------------------------------------------------------------------------

template <>
struct MMA_Traits<SM80_SPARSE_16x8x128_S32S4S4S32_TN>
{
  using ValTypeD  = int32_t;
  using ValTypeA  = sparse_elem<2, int4_t>;
  using ValTypeE  = sparse_elem<64, uint32_t>;
  using ValTypeB  = int4_t;
  using ValTypeC  = int32_t;

  using Shape_MNK = Shape<_16, _8, _128>;
  using ThrID     = Layout<_32>;
  // (T32,V64) -> (M16,K128)
  using ALayout   = Layout<Shape <Shape <  _4, _8>, Shape <_16, _2,    _2>>,
                           Stride<Stride<_256, _1>, Stride<_16, _8, _1024>>>;
  // (T32,V64) -> (M16,K128)  (note: PTX doc is wrong)
  using ELayout   = Layout<Shape <Shape <_2,    _2, _8>, _64>,
                           Stride<Stride<_8, _1024, _1>,  _0>>;
  // (T32,V64) -> (N8,K128)
  using BLayout   = Layout<Shape <Shape < _4, _8>, Shape <_8,   _4>>,
                           Stride<Stride<_64, _1>, Stride<_8, _256>>>;
  // C, D same as mma.m16n8k64
  using CLayout   = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM80_SPARSE_16x8x128_S32S4S4S32_TN_SATURATE>
     : MMA_Traits<SM80_SPARSE_16x8x128_S32S4S4S32_TN> {};

} // end namespace cute
