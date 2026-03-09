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
  \brief Fusion callbacks specializations for the sm100 TMA warp-specialized (ws) epilogue
*/


#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

#include "cutlass/epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp"  
#include "cutlass/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp" 

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Sm100 Tma warp specialized callbacks just alias to their sm90 counterpart
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};

// Sm100 direct store callbacks alias to sm100 tma callbacks with 0 stages
// Additional copy atom args will be ignored in the 0-stage specializations of aux load/store nodes
template <
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100NoSmemWarpSpecialized,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm100TmaWarpSpecialized<0, 0, 0, false, false>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm100TmaWarpSpecialized<0, 0, 0, false, false>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};

// {$nv-internal-release begin}
// Strided dgrad
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100TmaWsStridedDgrad<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};
// {$nv-internal-release end}

// Sm100 Ptr array tma warp specialized callbacks just alias to their sm90 counterpart
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, 1>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, 1>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C
// With Row BlockScaleFactor Generation.
// {$nv-internal-release begin}
// 1. Find max of 32 F32 elements
// 2. Convert the max to UE8 (or UE4M3) and store the result.
// 3. Convert the UE8 (or UE4M3) back to F32 scale.
// 4. Reciprocal of F32 scale with MUFU.
// 5. Multiply each F32 element with the above reciprocal, then convert to ElementD
// {$nv-internal-release end}
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombRowBlockScaleFactor =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>, // gen scalefactor
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };
  
  // Ctor inheritance
  using Impl::Impl;
};

// D = alpha * acc + beta * C
// With Col BlockScaleFactor Generation.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombColBlockScaleFactor =
  Sm90EVT<Sm100BlockScaleFactorColStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>, // gen scalefactor
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombColBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombColBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor,  ElementSource, ElementScalar, RoundStyle>;  

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// For Ptr-Array and Grouped GEMM
// D = alpha * acc + beta * C, where alpha and beta can be vectors for each batch/group
// With Row BlockScaleFactor Generation, separate tensors per batch/group.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombRowBlockScaleFactorPtrArray =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor *, RoundStyle>, // gen scalefactor
    Sm90LinearCombinationPtrArray<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;
    ElementBlockScaleFactor ** block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    // NormConst is a single device-side constant value, its not per-batch or per-group
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };
  
  // Ctor inheritance
  using Impl::Impl;
};

// {$nv-internal-release begin}
// For Ptr-Array and Grouped GEMM
// D = alpha * acc + beta * C, where alpha and beta can be vectors for each batch/group
// With Col BlockScaleFactor Generation, separate tensors per batch/group.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombColBlockScaleFactorPtrArray =
  Sm90EVT<Sm100BlockScaleFactorColStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor *, RoundStyle>, // gen scalefactor
    Sm90LinearCombinationPtrArray<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombColBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombColBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor,  ElementSource, ElementScalar, RoundStyle>;  

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;
    ElementBlockScaleFactor ** block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};
// {$nv-internal-release end}

/////////////////////////////////////////////////////////////////////////////////////////////////

// For Ptr-Array and Grouped GEMM
// D = activation(alpha * acc + beta * C), where alpha and beta can be vectors for each batch/group
// With Row BlockScaleFactor Generation, separate tensors per batch/group.
template<
  int SFVecsize,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombEltActRowBlockScaleFactorPtrArray =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor *, RoundStyle>, // gen scalefactor
    Sm90LinCombEltActPtrArray<ActivationFn, ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // activation(beta * C + (alpha * acc))
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombEltActBlockScaleFactor<ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombEltActRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinCombEltActRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombEltActBlockScaleFactor<ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;
    ElementBlockScaleFactor ** block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op: activation(beta * C + (alpha * acc))
            {    // ternary op : beta * C + (alpha * acc)
              {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // binary op : alpha * acc
                {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {}                  // binary args : multiplies
              },                    // end binary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// {$nv-internal-release begin}
// For Ptr-Array and Grouped GEMM
// D = activation(alpha * acc + beta * C), where alpha and beta can be vectors for each batch/group
// With Col BlockScaleFactor Generation, separate tensors per batch/group.
template<
  int SFVecsize,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombEltActColBlockScaleFactorPtrArray =
  Sm90EVT<Sm100BlockScaleFactorColStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor *, RoundStyle>, // gen scalefactor
    Sm90LinCombEltActPtrArray<ActivationFn, ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombEltActBlockScaleFactor<ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombEltActColBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinCombEltActColBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombEltActBlockScaleFactor<ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor,  ElementSource, ElementScalar, RoundStyle>;  

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;
    ElementBlockScaleFactor ** block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op: activation(beta * C + (alpha * acc))
            {    // ternary op : beta * C + (alpha * acc)
              {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // binary op : alpha * acc
                {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {}                  // binary args : multiplies
              },                    // end binary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// The following definition and subsequent specialization on
// IsBatchNormStatDeterministic is required to make Clang
// happy. Originally, a using statement that used
// cute::conditional_template to select between cutlass::plus and
// cutlass::atomic_add was used, but clang cannot seem to figure out
// that the return value from conditional_template is in fact a
// template template type. So instead we specialize on
// IsBatchNormStatDeterministic and have this extra copy of code.

template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBatchNormStat,
  bool IsBatchNormStatDeterministic,
  bool IsBatchNormStatFinal,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm100BatchNormStatFprop {

    static_assert(IsBatchNormStatDeterministic); // Specialization for false below

    using type = Sm90EVT<
        Sm90Compute<cutlass::epilogue::thread::Identity,
	    ElementOutput, ElementCompute, RoundStyle>,
        Sm90EVT<
          Sm90RowReduction<cutlass::square_and_plus, cutlass::plus, cutlass::plus,
              0, CtaTileShapeMNK, ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0,_1,_0>,
              128 / sizeof_bits_v<ElementOutput>, false, IsBatchNormStatFinal, false>,
          Sm90EVT<
            Sm90RowReduction<cutlass::plus, cutlass::plus, cutlass::plus,
                0, CtaTileShapeMNK, ElementBatchNormStat, ElementCompute,
                RoundStyle, Stride<_0,_1,_0>,
                128 / sizeof_bits_v<ElementOutput>, false, IsBatchNormStatFinal, false>,
            Sm90AccFetch
          >
        >
      >;

};
  
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBatchNormStat,
  bool IsBatchNormStatFinal,
  FloatRoundStyle RoundStyle
>
struct Sm100BatchNormStatFprop
<
  CtaTileShapeMNK,
  ElementOutput,
  ElementCompute,
  ElementBatchNormStat,
  /* IsBatchNormStatDeterministic */ false,
  IsBatchNormStatFinal,
  RoundStyle
> {

    using type = Sm90EVT<
      Sm90Compute<cutlass::epilogue::thread::Identity,
    	    ElementOutput, ElementCompute, RoundStyle>,
      Sm90EVT<
        Sm90RowReduction<cutlass::square_and_plus, cutlass::plus, cutlass::atomic_add,
            0, CtaTileShapeMNK, ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0,_1,_0>,
            128 / sizeof_bits_v<ElementOutput>, false, IsBatchNormStatFinal, false>,
        Sm90EVT<
          Sm90RowReduction<cutlass::plus, cutlass::plus, cutlass::atomic_add,
              0, CtaTileShapeMNK, ElementBatchNormStat, ElementCompute,
              RoundStyle, Stride<_0,_1,_0>,
              128 / sizeof_bits_v<ElementOutput>, false, IsBatchNormStatFinal, false>,
          Sm90AccFetch
        >
      >
    >;
};

// SUM = reduce_sum(D) in m dimension
// SUM_OF_SQUARE = reduce_sum(elementwise_mul(D, D)) in m dimension
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBatchNormStat,
  bool IsBatchNormStatDeterministic,
  bool IsBatchNormStatFinal,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::BatchNormStatFprop<ElementOutput, ElementCompute, ElementBatchNormStat, IsBatchNormStatDeterministic, IsBatchNormStatFinal, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100BatchNormStatFprop<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBatchNormStat, IsBatchNormStatDeterministic, IsBatchNormStatFinal, RoundStyle>::type {
  using Impl = typename Sm100BatchNormStatFprop<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBatchNormStat, IsBatchNormStatDeterministic, IsBatchNormStatFinal, RoundStyle>::type;
  using Operation = fusion::BatchNormStatFprop<ElementOutput, ElementCompute, ElementBatchNormStat, IsBatchNormStatDeterministic, IsBatchNormStatFinal, RoundStyle>;

  struct Arguments {
    ElementBatchNormStat* sum_ptr = nullptr;
    ElementBatchNormStat* sum_of_square_ptr = nullptr;
    operator typename Impl::Arguments() const {
      return
        {
          {
            {
              {}, // acc
              {sum_ptr} // sum
            },
            {sum_of_square_ptr} // sum of square
          },
          {} // cast from ElementCompute to ElementOutput
        };
    }
  };

  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesC,
  template <class> class ActivationFn,
  class GmemLayoutTagAux,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpS2R,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementScalar,
  class ElementBatchNormStat,
  bool IsBatchNormStatDeterministic,
  bool IsBatchNormStatFinal,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm100BatchNormStatDgrad {

  static_assert(IsBatchNormStatDeterministic); // Specialization for false below.
  
  // step1: redo the batchnorm_apply in dgrad epilogue
  // x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
  // step2: calculate the d_activation
  // DX = d_activation(acc, x_norm)
  // step3: calculate batchnorm_stat
  // SUM = reduce_sum(DX)
  // SUM_OF_SQUARE = reduce_sum(fprop_act * DX)
  using type = Sm90EVT<
    Sm90Compute<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, RoundStyle>,
    Sm90TopologicalVisitor<
      ElementCompute,
      cute::tuple<
        cute::seq<>,         // 0. load fprop_act
        cute::seq<>,         // 1. load acc
        cute::seq<0>,        // 2. batchnorm_apply i.e. x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
        cute::seq<1, 2>,     // 3. dx = d_activation(acc, x_norm)
        cute::seq<0, 3>,     // 4. bn_sum_of_square = SUM(fprop_act * dx)
        cute::seq<3>         // 5. bn_sum = SUM(dx)
      >,
      Sm90AuxLoad<StagesC, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux>, // fprop_act
      Sm90AccFetch, // acc
      Sm100BatchNormApply<
          detail::compute_row_broadcast_stages<StagesC, CtaTileShapeMNK, EpilogueTile>(),
          CtaTileShapeMNK, ElementScalar, ElementCompute, ElementOutput, Stride<_0,_1,_0>, AlignmentScalar, RoundStyle>, // batchnorm_apply
      Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // dx = d_activation(acc, x_norm)
      Sm90RowReduction<cutlass::homogeneous_multiply_add, cutlass::plus, 
          cutlass::plus, 
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false, cute::seq<1, 2, 0>>, // bn_sum_of_square = SUM(fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::plus, 
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false> // bn_sum = SUM(dx)
    >
  >;
};

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesC,
  template <class> class ActivationFn,
  class GmemLayoutTagAux,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpS2R,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementScalar,
  class ElementBatchNormStat,
  bool IsBatchNormStatFinal,
  int AlignmentAux,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle
>
struct Sm100BatchNormStatDgrad
<
  CtaTileShapeMNK,
  EpilogueTile,
  StagesC,
  ActivationFn,
  GmemLayoutTagAux,
  StrideAux,
  SmemLayoutAtom,
  CopyOpS2R,
  ElementOutput,
  ElementCompute,
  ElementAux,
  ElementScalar,
  ElementBatchNormStat,
  /* IsBatchNormStatDeterministic */ false,
  IsBatchNormStatFinal,
  AlignmentAux,
  AlignmentScalar,
  RoundStyle
> {

  // step1: redo the batchnorm_apply in dgrad epilogue
  // x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
  // step2: calculate the d_activation
  // DX = d_activation(acc, x_norm)
  // step3: calculate batchnorm_stat
  // SUM = reduce_sum(DX)
  // SUM_OF_SQUARE = reduce_sum(fprop_act * DX)
  using type = Sm90EVT<
    Sm90Compute<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, RoundStyle>,
    Sm90TopologicalVisitor<
      ElementCompute,
      cute::tuple<
        cute::seq<>,         // 0. load fprop_act
        cute::seq<>,         // 1. load acc
        cute::seq<0>,        // 2. batchnorm_apply i.e. x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
        cute::seq<1, 2>,     // 3. dx = d_activation(acc, x_norm)
        cute::seq<0, 3>,     // 4. bn_sum_of_square = SUM(fprop_act * dx)
        cute::seq<3>         // 5. bn_sum = SUM(dx)
      >,
      Sm90AuxLoad<StagesC, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux>, // fprop_act
      Sm90AccFetch, // acc
      Sm100BatchNormApply<
          detail::compute_row_broadcast_stages<StagesC, CtaTileShapeMNK, EpilogueTile>(),
          CtaTileShapeMNK, ElementScalar, ElementCompute, ElementOutput, Stride<_0,_1,_0>, AlignmentScalar, RoundStyle>, // batchnorm_apply
      Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // dx = d_activation(acc, x_norm)
      Sm90RowReduction<cutlass::homogeneous_multiply_add, cutlass::plus, 
          cutlass::atomic_add,
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false, cute::seq<1, 2, 0>>, // bn_sum_of_square = SUM(fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::atomic_add,
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false> // bn_sum = SUM(dx)
    >
  >;
};
  
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementScalar,
  class ElementBatchNormStat,
  bool IsBatchNormStatDeterministic,
  bool IsBatchNormStatFinal,
  int AlignmentAux,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpS2R
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::BatchNormStatDgrad<GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
                               ElementBatchNormStat, ElementAux, ElementScalar, IsBatchNormStatDeterministic,
                               IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpS2R
>: Sm100BatchNormStatDgrad<
      CtaTileShapeMNK, EpilogueTile, StagesC, ActivationFn, GmemLayoutTagAux, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpS2R, ElementOutput, ElementCompute, ElementAux, ElementScalar, ElementBatchNormStat,
      IsBatchNormStatDeterministic, IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle
  >::type {

  using Impl =
    typename Sm100BatchNormStatDgrad<
      CtaTileShapeMNK, EpilogueTile, StagesC, ActivationFn, GmemLayoutTagAux, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpS2R, ElementOutput, ElementCompute, ElementAux, ElementScalar, ElementBatchNormStat,
      IsBatchNormStatDeterministic, IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle
    >::type;

  using Operation =
    fusion::BatchNormStatDgrad<GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute, ElementBatchNormStat,
                               ElementAux, ElementScalar, IsBatchNormStatDeterministic, IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle>;

  using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;

  struct Arguments {
    ElementBatchNormStat* sum_ptr = nullptr;
    ElementBatchNormStat* sum_of_square_ptr = nullptr;
    ElementAux const* fprop_act_ptr = nullptr;
    ElementScalar const* fprop_alpha_ptr = nullptr;
    ElementScalar const* fprop_bias_ptr = nullptr;
    ElementScalar const* fprop_mean_ptr = nullptr;
    ElementScalar const* fprop_inv_stddev_ptr = nullptr;

    StrideAux fprop_act_stride = {};

    operator typename Impl::Arguments() const {
      typename Impl::Arguments args = {};
      auto& [topo_args, cast_args] = args;
      auto& [fprop_act_args, acc_args, batchnorm_apply_args,
        compute_args, sum_of_square_args, sum_args] = topo_args;
      fprop_act_args = {fprop_act_ptr, ElementAux(0), fprop_act_stride};
      batchnorm_apply_args = {fprop_alpha_ptr, fprop_bias_ptr, fprop_mean_ptr, fprop_inv_stddev_ptr};
      sum_of_square_args = {sum_of_square_ptr};
      sum_args = {sum_ptr};
      return args;
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesC,
  template <class> class ActivationFn,
  class GmemLayoutTagAux,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpS2R,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementScalar,
  class ElementBatchNormStat,
  bool IsBatchNormStatDeterministic,
  bool IsBatchNormStatFinal,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm100DualBatchNormStatDgrad {

  static_assert(IsBatchNormStatDeterministic); // Specialization for false below
  
  // step1: redo the first batchnorm_apply in dgrad epilogue
  // x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
  // step2: redo the second batchnorm_apply in dgrad epilogue
  // x_norm += (second_fprop_act - second_fprop_mean) * second_fprop_inv_stddev * second_fprop_alpha + second_fprop_bias
  // step3: calculate the d_activation
  // DX = d_activation(acc, x_norm)
  // step4: calculate batchnorm_stat
  // SUM = second SUM = reduce_sum(DX)
  // SUM_OF_SQUARE = reduce_sum(fprop_act * DX)
  // second_SUM_OF_SQUARE = reduce_sum(second_fprop_act * DX)
  using type = Sm90EVT<
    Sm90Compute<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, RoundStyle>,
    Sm90TopologicalVisitor<
      ElementCompute,
      cute::tuple<
        cute::seq<>,         // 0.  load fprop_act
        cute::seq<>,         // 1.  load second_fprop_act
        cute::seq<>,         // 2.  load acc
        cute::seq<0>,        // 3.  the first batchnorm_apply i.e. x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
        cute::seq<1>,        // 4.  the second batchnorm_apply i.e. second_x_norm = (second_fprop_act - second_fprop_mean) * second_fprop_inv_stddev * second_fprop_alpha + second_fprop_bias
        cute::seq<3, 4>,     // 5.  x_norm += second_x_norm
        cute::seq<2, 5>,     // 6.  dx = d_activation(acc, x_norm)
        cute::seq<0, 6>,     // 7.  (fprop_act * dx)
        cute::seq<1, 6>,     // 8.  (second_fprop_act * dx)
        cute::seq<7>,        // 9.  bn_sum_of_square = SUM(fprop_act * dx)
        cute::seq<8>,        // 10. second_bn_sum_of_sqaure = SUM(second_fprop_act * dx)
        cute::seq<6>         // 11. bn_sum = SUM(dx)
      >,
      Sm90AuxLoad<StagesC, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux>, // fprop_act
      Sm90AuxLoad<StagesC, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux>, // second_fprop_act
      Sm90AccFetch, // acc
      Sm100BatchNormApply<
          detail::compute_row_broadcast_stages<StagesC, CtaTileShapeMNK, EpilogueTile>(),
          CtaTileShapeMNK, ElementScalar, ElementCompute, ElementCompute, Stride<_0,_1,_0>, AlignmentScalar, RoundStyle>, // the first batchnorm_apply
      Sm100BatchNormApply<
          detail::compute_row_broadcast_stages<StagesC, CtaTileShapeMNK, EpilogueTile>(),
          CtaTileShapeMNK, ElementScalar, ElementCompute, ElementCompute, Stride<_0,_1,_0>, AlignmentScalar, RoundStyle>, // the second batchnorm_apply
      Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>, // x_norm += second_x_norm
      Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // dx = d_activation(acc, x_norm)
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // (fprop_act * dx)
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // (second_fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::plus, 
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false>, // bn_sum_of_square = SUM(fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::plus, 
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false>, // second_bn_sum_of_square = SUM(second_fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::plus, 
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false> // bn_sum = SUM(dx)
    >
  >;
};

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesC,
  template <class> class ActivationFn,
  class GmemLayoutTagAux,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpS2R,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementScalar,
  class ElementBatchNormStat,
  bool IsBatchNormStatFinal,
  int AlignmentAux,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle
>
struct Sm100DualBatchNormStatDgrad
<
  CtaTileShapeMNK,
  EpilogueTile,
  StagesC,
  ActivationFn,
  GmemLayoutTagAux,
  StrideAux,
  SmemLayoutAtom,
  CopyOpS2R,
  ElementOutput,
  ElementCompute,
  ElementAux,
  ElementScalar,
  ElementBatchNormStat,
  /* IsBatchNormStatDeterministic */ false,
  IsBatchNormStatFinal,
  AlignmentAux,
  AlignmentScalar,
  RoundStyle
> {
  // step1: redo the first batchnorm_apply in dgrad epilogue
  // x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
  // step2: redo the second batchnorm_apply in dgrad epilogue
  // x_norm += (second_fprop_act - second_fprop_mean) * second_fprop_inv_stddev * second_fprop_alpha + second_fprop_bias
  // step3: calculate the d_activation
  // DX = d_activation(acc, x_norm)
  // step4: calculate batchnorm_stat
  // SUM = second SUM = reduce_sum(DX)
  // SUM_OF_SQUARE = reduce_sum(fprop_act * DX)
  // second_SUM_OF_SQUARE = reduce_sum(second_fprop_act * DX)
  using type = Sm90EVT<
    Sm90Compute<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, RoundStyle>,
    Sm90TopologicalVisitor<
      ElementCompute,
      cute::tuple<
        cute::seq<>,         // 0.  load fprop_act
        cute::seq<>,         // 1.  load second_fprop_act
        cute::seq<>,         // 2.  load acc
        cute::seq<0>,        // 3.  the first batchnorm_apply i.e. x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
        cute::seq<1>,        // 4.  the second batchnorm_apply i.e. second_x_norm = (second_fprop_act - second_fprop_mean) * second_fprop_inv_stddev * second_fprop_alpha + second_fprop_bias
        cute::seq<3, 4>,     // 5.  x_norm += second_x_norm
        cute::seq<2, 5>,     // 6.  dx = d_activation(acc, x_norm)
        cute::seq<0, 6>,     // 7.  (fprop_act * dx)
        cute::seq<1, 6>,     // 8.  (second_fprop_act * dx)
        cute::seq<7>,        // 9.  bn_sum_of_square = SUM(fprop_act * dx)
        cute::seq<8>,        // 10. second_bn_sum_of_sqaure = SUM(second_fprop_act * dx)
        cute::seq<6>         // 11. bn_sum = SUM(dx)
      >,
      Sm90AuxLoad<StagesC, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux>, // fprop_act
      Sm90AuxLoad<StagesC, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux>, // second_fprop_act
      Sm90AccFetch, // acc
      Sm100BatchNormApply<
          detail::compute_row_broadcast_stages<StagesC, CtaTileShapeMNK, EpilogueTile>(),
          CtaTileShapeMNK, ElementScalar, ElementCompute, ElementCompute, Stride<_0,_1,_0>, AlignmentScalar, RoundStyle>, // the first batchnorm_apply
      Sm100BatchNormApply<
          detail::compute_row_broadcast_stages<StagesC, CtaTileShapeMNK, EpilogueTile>(),
          CtaTileShapeMNK, ElementScalar, ElementCompute, ElementCompute, Stride<_0,_1,_0>, AlignmentScalar, RoundStyle>, // the second batchnorm_apply
      Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>, // x_norm += second_x_norm
      Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // dx = d_activation(acc, x_norm)
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // (fprop_act * dx)
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // (second_fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::atomic_add, 
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false>, // bn_sum_of_square = SUM(fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::atomic_add,
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false>, // second_bn_sum_of_square = SUM(second_fprop_act * dx)
      Sm90RowReduction<cutlass::plus, cutlass::plus, 
          cutlass::atomic_add,
          0, CtaTileShapeMNK,
          ElementBatchNormStat, ElementCompute, RoundStyle, Stride<_0, _1, _0>, 128 / sizeof_bits_v<ElementBatchNormStat>, false, IsBatchNormStatFinal, false> // bn_sum = SUM(dx)
    >
  >;
};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementScalar,
  class ElementBatchNormStat,
  bool IsBatchNormStatDeterministic,
  bool IsBatchNormStatFinal,
  int AlignmentAux,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpS2R
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::DualBatchNormStatDgrad<GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
                                   ElementBatchNormStat, ElementAux, ElementScalar, IsBatchNormStatDeterministic,
                                   IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpS2R
>: Sm100DualBatchNormStatDgrad<
      CtaTileShapeMNK, EpilogueTile, StagesC, ActivationFn, GmemLayoutTagAux, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpS2R, ElementOutput, ElementCompute, ElementAux, ElementScalar, ElementBatchNormStat,
      IsBatchNormStatDeterministic, IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle
  >::type {

  using Impl =
    typename Sm100DualBatchNormStatDgrad<
      CtaTileShapeMNK, EpilogueTile, StagesC, ActivationFn, GmemLayoutTagAux, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpS2R, ElementOutput, ElementCompute, ElementAux, ElementScalar, ElementBatchNormStat,
      IsBatchNormStatDeterministic, IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle
    >::type;

  using Operation =
    fusion::DualBatchNormStatDgrad<GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute, ElementBatchNormStat,
                                   ElementAux, ElementScalar, IsBatchNormStatDeterministic, IsBatchNormStatFinal, AlignmentAux, AlignmentScalar, RoundStyle>;

  using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;

  struct Arguments {
    ElementBatchNormStat* sum_ptr = nullptr;
    ElementBatchNormStat* sum_of_square_ptr = nullptr;
    ElementAux const* fprop_act_ptr = nullptr;
    ElementScalar const* fprop_alpha_ptr = nullptr;
    ElementScalar const* fprop_bias_ptr = nullptr;
    ElementScalar const* fprop_mean_ptr = nullptr;
    ElementScalar const* fprop_inv_stddev_ptr = nullptr;

    ElementBatchNormStat* second_sum_of_square_ptr = nullptr;
    ElementAux const* second_fprop_act_ptr = nullptr;
    ElementScalar const* second_fprop_alpha_ptr = nullptr;
    ElementScalar const* second_fprop_bias_ptr = nullptr;
    ElementScalar const* second_fprop_mean_ptr = nullptr;
    ElementScalar const* second_fprop_inv_stddev_ptr = nullptr;

    StrideAux fprop_act_stride = {};
    StrideAux second_fprop_act_stride = {};

    operator typename Impl::Arguments() const {
      typename Impl::Arguments args = {};
      auto& [topo_args, cast_args] = args;
      auto& [fprop_act_args, second_fprop_act_args, acc_args,
        batchnorm_apply_args, second_batchnorm_apply_args,
        compute_1_args, compute_2_args, compute_3_args, compute_4_args,
        sum_of_square_args, second_sum_of_square_args, sum_args] = topo_args;
      fprop_act_args = {fprop_act_ptr, ElementAux(0), fprop_act_stride};
      second_fprop_act_args = {second_fprop_act_ptr, ElementAux(0), second_fprop_act_stride};
      batchnorm_apply_args = {fprop_alpha_ptr, fprop_bias_ptr, fprop_mean_ptr, fprop_inv_stddev_ptr};
      second_batchnorm_apply_args = {second_fprop_alpha_ptr, second_fprop_bias_ptr, second_fprop_mean_ptr, second_fprop_inv_stddev_ptr};
      sum_of_square_args = {sum_of_square_ptr};
      second_sum_of_square_args = {second_sum_of_square_ptr};
      sum_args = {sum_ptr};
      return args;
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};
// {$nv-internal-release end}

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per-row bias
//   with row blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, ElementOutput, 
      ElementCompute, ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBias<
      CtaTileShapeMNK, ElementCompute, ElementCompute, 
      ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, 
      ElementScalar, 
      AlignmentBias,
       RoundStyle
    > 
{

  using Impl = 
    Sm100LinCombPerRowBiasRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {
          {  // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },  // end ternary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// D = alpha * acc + beta * C + per-row bias
//   with col blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasColBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorColStore<
      SFVecsize, EpilogueTile, ElementOutput, 
      ElementCompute, ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBias<
      CtaTileShapeMNK, ElementCompute, ElementCompute, 
      ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor,
      ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > 
{

  using Impl = 
    Sm100LinCombPerRowBiasColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {
          {  // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },  // end ternary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per_col bias
//   with row blockScaled generation
template<
  int StagesC,
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerColBiasRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, ElementOutput, 
      ElementCompute, ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerColBias<
      StagesC, CtaTileShapeMNK, EpilogueTile, ElementCompute, ElementCompute, 
      ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerColBiasRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > 
{

  using Impl = 
    Sm100LinCombPerColBiasRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerColBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};


    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {
          {  // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },  // end ternary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias) 
//   with row blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasEltActRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasEltActRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerRowBiasEltActRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias) 
//   with col blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasEltActColBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorColStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasEltActColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerRowBiasEltActColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};


    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per_col bias) 
//   with row blockScaled generation
template<
  int StagesC,
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerColBiasEltActRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerColBiasEltAct<
      StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerColBiasEltActRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerColBiasEltActRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerColBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};


// {$nv-internal-release begin}
/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-Row bias)
// D'       = D with Row blockScaled generation
// Extra D" = D with Col blockScaled generation
template<
  int StagesC,
  int StagesD,
  class StrideExtraD,
  class SmemLayoutAtom,
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias   = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentExtraD = 128 / sizeof_bits_v<ElementOutput>,
  int AlignmentBias   = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasEltAct2xOutput1d2xBlockSF =
  Sm90SplitTreeVisitor<
    // Z = alpha * acc +  beta * c + pre-Row bias
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, 
      ElementCompute, ElementCompute, 
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >,

    // D'= with Row blockScaled generation
    Sm90EVT<
      Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
      Sm90SplitTreeFetch // Z
    >,

    // Extra D" with Col blockScaled generation
    Sm90EVT<Sm90AuxStore<StagesD, EpilogueTile, ElementOutput, RoundStyle, StrideExtraD, SmemLayoutAtom, CopyOpR2S, AlignmentExtraD>,
      Sm90EVT<
        Sm100BlockScaleFactorColStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
        Sm90SplitTreeFetch // Z
      >
    >
  >;


template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagExtraD,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentExtraD,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActSfLayout1dOutD2xSfD2xBlockScaleFactor<
      GmemLayoutTagExtraD, ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor,
      ElementBias, ElementSource, ElementScalar, 
      AlignmentExtraD, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm100LinCombPerRowBiasEltAct2xOutput1d2xBlockSF<
      StagesC, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagExtraD>,
      SmemLayoutAtom, SFVecSize, CtaTileShapeMNK, EpilogueTile, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource,
      ElementCompute, AlignmentExtraD, AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerRowBiasEltAct2xOutput1d2xBlockSF<
      StagesC, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagExtraD>,
      SmemLayoutAtom, SFVecSize, CtaTileShapeMNK, EpilogueTile, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource,
      ElementCompute, AlignmentExtraD, AlignmentBias, RoundStyle
    >;

  using Operation =
    fusion::LinCombPerRowBiasEltActSfLayout1dOutD2xSfD2xBlockScaleFactor<
      GmemLayoutTagExtraD, ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor,
      ElementBias, ElementSource, ElementScalar, 
      AlignmentExtraD, AlignmentBias, RoundStyle
    >;


  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr  = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr       = nullptr;
    ElementBlockScaleFactor * extra_block_scale_factor_ptr = nullptr;
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};


    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using StrideExtraD = cutlass::gemm::TagToStrideC_t<GmemLayoutTagExtraD>;
    ElementOutput* extra_d_ptr = nullptr;
    StrideExtraD dExtraD = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      typename Impl::Arguments args;
      auto& [Z_args, Extra_D_args, D_args] = args;
      Z_args =
        { // unary op : activation(beta * C + (alpha * acc + bias))
          {    // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args : activation
        };     // end unary op
      
      D_args = 
        {
          {},
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst},         // BlockScaleFactor args
        };

      Extra_D_args = 
        { 
          { // unary op : store(D)
            {},
            {extra_block_scale_factor_ptr, norm_constant_ptr, dNormConst}, // BlockScaleFactor args
          },
          {extra_d_ptr, dExtraD} // unary args : store (Args to Aux for D")
        };  // end unary op
      
      return args;
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// D = activation(alpha * acc + beta * C + per-Row bias) 
//   with 2d blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasEltActSfLayout2dOutD1xSfD2xBlockScaleFactor =
  Sm90EVT<
    Sm100Block2DScaleFactorStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActSfLayout2dOutD1xSfD2xBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasEltActSfLayout2dOutD1xSfD2xBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerRowBiasEltActSfLayout2dOutD1xSfD2xBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasEltActSfLayout2dOutD1xSfD2xBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    ElementBlockScaleFactor * extra_block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename FusionCallbacks::Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, extra_block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};
// {$nv-internal-release end}

// --------------------------------------------------------------------
//  Sm100PtrArrayNoSmemWarpSpecialized  (direct-store, grouped GEMM)
// --------------------------------------------------------------------
template <
    class Operation,
    class CtaTile_MNK,
    class EpilogueTile_MN,
    class... Args
>
struct FusionCallbacks<
        epilogue::Sm100PtrArrayNoSmemWarpSpecialized,
        Operation,
        CtaTile_MNK,
        EpilogueTile_MN,
        Args...>
  : FusionCallbacks<
        // reuse the ptr-array *TMA* callbacks with 0 stages
        epilogue::Sm100PtrArrayTmaWarpSpecialized<0,0,0,false,false>,
        Operation,
        CtaTile_MNK,
        EpilogueTile_MN,
        Args...> {

  using Base = FusionCallbacks<
      epilogue::Sm100PtrArrayTmaWarpSpecialized<0,0,0,false,false>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>;

  // bring ctors into scope
  using Base::Base;
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
