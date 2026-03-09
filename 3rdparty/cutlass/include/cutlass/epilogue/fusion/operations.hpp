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

#include <cutlass/numeric_conversion.h>
#include <cutlass/layout/matrix.h>
#include <cute/numeric/numeric_types.hpp>
#include <cute/numeric/integral_constant.hpp> // cute::false_type

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fusion Operations
// Template args must not be implementation dependent
//
/////////////////////////////////////////////////////////////////////////////////////////////////

struct FusionOperation {
  // metadata types/queries that can be overrided
  using ElementOutput = void;
  using ElementCompute = void;
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_indeterminate;

  using ElementSource = void;
  static constexpr bool IsSourceSupported = false;
  static constexpr bool IsResidualSupported = false; // Source is added after activation

  using ElementScalar = void;
  static constexpr int AlignmentScalar = 0;
  static constexpr bool IsScaleFactorSupported = false;
  static constexpr bool IsPerRowScaleSupported = false;
  static constexpr bool IsPerColScaleSupported = false;

  using ElementBias = void;
  static constexpr int AlignmentBias = 0;
  static constexpr bool IsPerRowBiasSupported = false;
  static constexpr bool IsPerColBiasSupported = false;
  static constexpr bool IsDePerRowBiasSupported = false;

  using ActivationFn = void;
  static constexpr bool IsEltActSupported = false;
  static constexpr bool IsEltDualActSupported = false; // {$nv-release-never}
  static constexpr bool IsDeEltActSupported = false;

  using ElementAux = void;
  using GmemLayoutTagAux = void;
  static constexpr int AlignmentAux = 0;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAuxInSupported = false;

  using ElementAmax = void;
  static constexpr bool IsAbsMaxSupported = false;

  // {$nv-internal-release begin}
  using ElementBatchNormStat = void;
  static constexpr bool IsBatchNormStatDeterministic = true;
  static constexpr bool IsBatchNormStatFinal = false;
  static constexpr bool IsBatchNormStatSupported = false;
  static constexpr bool IsBatchNormApplySupported = false;
  static constexpr bool IsDualBatchNormSupported = false;
  // {$nv-internal-release end}
  
  using ElementBlockScaleFactor = void;
  static constexpr int SFVecSize = 0;
  static constexpr bool IsBlockScaleSupported = false;               // Umbrella variable to check BlockScaling support in the epilogues
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = false; // 1D Layout scale factors generated and only one copy of D and SFD (M- or N-major) are stored // {$nv-internal-release}
  
  // {$nv-internal-release begin}
  static constexpr bool IsSfLayout1d_OutD2x_SfD2x_Supported = false; // 1D Layout scale factors generated and 2 copies of D and SFD (both M- and N-major) are stored
  static constexpr bool IsSfLayout2d_OutD1x_SfD2x_Supported = false; // 2D Layout scale factors generated and 1 copy of D is stored and 2 copies SFD (both M- and N-major) are stored.
                                                                     // Note that D is same for M- or N-major scale factors because SF generation is symmetric.
  // {$nv-internal-release end}
  using GmemLayoutTagScalefactor = void;
  // {$nv-internal-release begin}
  using GmemLayoutTagExtraD = void;
  using GmemLayoutTagExtraScalefactor = void;
  static constexpr int AlignmentExtraD = 0;
  // {$nv-internal-release end}
};

// D = alpha * acc
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledAcc : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentScalar = 1;
  static constexpr auto RoundStyle = RoundStyle_;
};

// D = alpha * acc + beta * C
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinearCombination
    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_> {
  using ElementSource = ElementSource_;
  static constexpr bool IsSourceSupported = true;
};

// D = activation(alpha * acc + beta * C)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombEltAct
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
};

// D = softmax(top_k(alpha * acc + beta * C))
template<
  int TopK,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombTopKSoftmaxCol
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
};


// D = alpha * acc + beta * C + per-row bias
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBias
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerRowBiasSupported = true;
};

// D = alpha * acc + beta * C + per-column bias
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBias
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerColBiasSupported = true;
};

// D = activation(alpha * acc + beta * C + per-row bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltAct
    : LinCombPerRowBias<ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
};

// Grouped Wgrad's D = alpha * acc + beta * C with special AccFetch.
template<
  class GroupsPerTile_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinearCombinationGroupedWgrad
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using GroupsPerTile = GroupsPerTile_;
};

// {$nv-release-never begin}
// D = pred(activation0(alpha * acc + beta * C + per-row bias),
//          activation1(alpha * acc + beta * C + per-row bias))
template<
  template <class> class ActivationFn0_,
  template <class> class ActivationFn1_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltDualAct
    : LinCombPerRowBias<ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ActivationFn0 = ActivationFn0_<ElementCompute_>;
  using ActivationFn1 = ActivationFn1_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
  static constexpr bool IsEltDualActSupported = true;
};
// {$nv-release-never end}

// D = activation(alpha * acc + beta * C + per-column bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasEltAct
    : LinCombPerColBias<ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
};

// {$nv-release-never begin}
// D = pred(activation0(alpha * acc + beta * C + per-col bias),
//          activation1(alpha * acc + beta * C + per-col bias))
template<
  template <class> class ActivationFn0_,
  template <class> class ActivationFn1_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasEltDualAct
    : LinCombPerColBias<ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ActivationFn0 = ActivationFn0_<ElementCompute_>;
  using ActivationFn1 = ActivationFn1_<ElementCompute_>;
  static constexpr bool IsEltActSupported = true;
  static constexpr bool IsEltDualActSupported = true;
};
// {$nv-release-never end}

// D = activation(alpha * acc + beta * C + per-row bias)
// aux = alpha * acc + beta * C + per-row bias
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltActAux
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
};

// D = activation(alpha * acc + beta * C + per-col bias)
// aux = alpha * acc + beta * C + per-col bias
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasEltActAux
    : LinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
};

// D = activation(per-row alpha * acc + per-row beta * C + per-row bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_, // per-row alpha/beta
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct PerRowLinCombPerRowBiasEltAct
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerRowScaleSupported = true;
};

// {$nv-release-never begin}
// D = pred(activation0(per-row alpha * acc + per-row beta * C + per-row bias),
//          activation1(per-row alpha * acc + per-row beta * C + per-row bias))
template<
  template <class> class ActivationFn0_,
  template <class> class ActivationFn1_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_, // per-row alpha/beta
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct PerRowLinCombPerRowBiasEltDualAct
    : LinCombPerRowBiasEltDualAct<ActivationFn0_, ActivationFn1_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerRowScaleSupported = true;
};
// {$nv-release-never end}

// D = activation(per-col alpha * acc + per-col beta * C + per-column bias)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_, // per-row alpha/beta
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct PerColLinCombPerColBiasEltAct
    : LinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerColScaleSupported = true;
};

// {$nv-release-never begin}
// D = pred(activation0(per-col alpha * acc + per-col beta * C + per-col bias),
//          activation1(per-col alpha * acc + per-col beta * C + per-col bias))
template<
  template <class> class ActivationFn0_,
  template <class> class ActivationFn1_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_, // per-col alpha/beta
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct PerColLinCombPerColBiasEltDualAct
    : LinCombPerColBiasEltDualAct<ActivationFn0_, ActivationFn1_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr bool IsPerColScaleSupported = true;
};
// {$nv-release-never end}

// D = activation(per-col alpha * acc + per-column bias) + per-col beta * C
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_, // per-row alpha/beta
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct PerColResAddPerColBiasEltAct
    : PerColLinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, AlignmentScalar_, RoundStyle_> {
  static constexpr bool IsResidualSupported = true;
};

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
// if D is fp8 
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerRowBiasEltAct
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr bool IsScaleFactorSupported = true;
};

// {$nv-release-never begin}
// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
// if D is fp8 
//   D = scale_d * pred(activation0(Z), activation1(Z))
// else
//   D = pred(activation0(Z), activation1(Z))
template<
  template <class> class ActivationFn0_,
  template <class> class ActivationFn1_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerRowBiasEltDualAct
    : LinCombPerRowBiasEltDualAct<ActivationFn0_, ActivationFn1_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr bool IsScaleFactorSupported = true;
};
// {$nv-release-never end}

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-col bias
// if D is fp8 
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerColBiasEltAct
    : LinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  static constexpr bool IsScaleFactorSupported = true;
};

// Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
// if D is fp8 
//   amax_d = max(abs(elements in activation(Z)))
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
// if Aux is fp8 
//   amax_aux = max(abs(elements in Z))
//   Aux = scale_aux * Z
// else
//   Aux = Z
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementAmax_ = ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerRowBiasEltActAmaxAux
    : ScaledLinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAmax = ElementAmax_;
  static constexpr bool IsAbsMaxSupported = true;

  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
};

// Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-col bias
// if D is fp8 
//   amax_d = max(abs(elements in activation(Z)))
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
// if Aux is fp8 
//   amax_aux = max(abs(elements in Z))
//   Aux = scale_aux * Z
// else
//   Aux = Z
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementAmax_ = ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledLinCombPerColBiasEltActAmaxAux
    : ScaledLinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_,
        ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementAmax = ElementAmax_;
  static constexpr bool IsAbsMaxSupported = true;

  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
};

// Z = Aux
// dY = alpha * acc + beta * C
// D = d_activation(dY, Z)
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombDeEltAct
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsDeEltActSupported = true;

  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxInSupported = true;
};

// {$nv-internal-release begin}
// SUM = reduce_sum(D) in m dimension
// SUM_OF_SQUARE = reduce_sum(elementwise_mul(D, D)) in m dimension
template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBatchNormStat_,
  bool IsBatchNormStatDeterministic_,
  bool IsBatchNormStatFinal_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct BatchNormStatFprop : FusionOperation {
  using ElementBatchNormStat = ElementBatchNormStat_;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  static constexpr bool IsBatchNormStatDeterministic = IsBatchNormStatDeterministic_;
  static constexpr bool IsBatchNormStatFinal = IsBatchNormStatFinal_;
  static constexpr int AlignmentScalar = 1;
  static constexpr auto RoundStyle = RoundStyle_;
  static constexpr bool IsBatchNormStatSupported = true;
};

// step1: redo the batchnorm_apply in dgrad epilogue
// x_norm = (fprop_act - fprop_mean) * fprop_inv_stddev * fprop_alpha + fprop_bias
// step2: calculate the d_activation
// DX = d_activation(acc, x_norm)
// step3: calculate batchnorm_stat
// SUM = reduce_sum(DX)
// SUM_OF_SQUARE = reduce_sum(fprop_act * DX)
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBatchNormStat_,
  class ElementAux_,
  class ElementScalar_,
  bool IsBatchNormStatDeterministic_,
  bool IsBatchNormStatFinal_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct BatchNormStatDgrad : FusionOperation {
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  using ActivationFn = ActivationFn_<ElementCompute_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementAux = ElementAux_;
  using ElementScalar = ElementScalar_;
  using ElementBatchNormStat = ElementBatchNormStat_;
  static constexpr bool IsBatchNormStatDeterministic = IsBatchNormStatDeterministic_;
  static constexpr bool IsBatchNormStatFinal = IsBatchNormStatFinal_;
  static constexpr bool IsAuxInSupported = true;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr auto RoundStyle = RoundStyle_;
  static constexpr bool IsBatchNormStatSupported = true;
  static constexpr bool IsBatchNormApplySupported = true;
  static constexpr bool IsDeEltActSupported = true;
};

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
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBatchNormStat_,
  class ElementAux_,
  class ElementScalar_,
  bool IsBatchNormStatDeterministic_,
  bool IsBatchNormStatFinal_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentScalar_ = 128 / cute::sizeof_bits_v<ElementScalar_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct DualBatchNormStatDgrad : FusionOperation {
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  using ActivationFn = ActivationFn_<ElementCompute_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementAux = ElementAux_;
  using ElementScalar = ElementScalar_;
  using ElementBatchNormStat = ElementBatchNormStat_;
  static constexpr bool IsBatchNormStatDeterministic = IsBatchNormStatDeterministic_;
  static constexpr bool IsBatchNormStatFinal = IsBatchNormStatFinal_;
  static constexpr bool IsAuxInSupported = true;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr int AlignmentScalar = AlignmentScalar_;
  static constexpr auto RoundStyle = RoundStyle_;
  static constexpr bool IsBatchNormStatSupported = true;
  static constexpr bool IsBatchNormApplySupported = true;
  static constexpr bool IsDualBatchNormSupported = true;
  static constexpr bool IsDeEltActSupported = true;
};
// {$nv-internal-release end}

// Z = Aux
// dY = alpha * acc + beta * C
// D = d_activation(dY, Z)
// dBias = sum of columns of D
template<
  class GmemLayoutTagAux_,
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementBias_ = ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombDeEltActDePerRowBias
    : LinCombDeEltAct<GmemLayoutTagAux_, ActivationFn_, ElementOutput_, ElementCompute_,
        ElementAux_, ElementSource_, ElementScalar_, AlignmentAux_, RoundStyle_> {
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsDePerRowBiasSupported = true;
};

// {$nv-internal-release begin}
// D = alpha * acc + beta * C
// With BlockScaleFactor generation.
// 1. Find max of SFVecSize F32 elements
// 2. Convert the max to UE8 (or UE4M3) and store the result.
// 3. Convert the UE8 (or UE4M3) back to F32 scale.
// 4. Reciprocal of F32 scale with MUFU.
// 5. Multiply each F32 element with the above reciprocal, then convert to ElementD
// ScaleFactor is interleaved 4B aligned format and supports holes.
//    callback supports 1/2/4 elements store in 4B (i.e. 4 elements) depends on auto-vectorize store.
// {$nv-internal-release end}

template<
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombBlockScaleFactor
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = true; // {$nv-internal-release}
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};

// D = activation(alpha * acc + beta * C)
// With BlockScaleFactor generation (same recipe as LinCombBlockScaleFactor).
template<
  template <class> class ActivationFn_,
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombEltActBlockScaleFactor
    : LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = true; // {$nv-internal-release}
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};

// D = alpha * acc + beta * C + per-row bias
// With BlockScaleFactor generation
// {$nv-internal-release begin}
// 1. Find max of SFVecSize_ F32 elements
// 2. Convert the max to UE8 (or UE4M3) and store the result.
// 3. Convert the UE8 (or UE4M3) back to F32 scale.
// 4. Reciprocal of F32 scale with MUFU.
// 5. Multiply each F32 element with the above reciprocal, then convert to ElementD
// ScaleFactor is interleaved 4B aligned format and supports holes.
//    callback supports 1/2/4 elements store in 4B (i.e. 4 elements) depends on auto-vectorize store.
// {$nv-internal-release end}
template<
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementBias_   = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasBlockScaleFactor
    : LinCombPerRowBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = true; // {$nv-internal-release}
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};


// D = alpha * acc + beta * C + per-col bias
// With BlockScaleFactor generation.
// {$nv-internal-release begin}
// 1. Find max of SFVecSize_ F32 elements
// 2. Convert the max to UE8 (or UE4M3) and store the result.
// 3. Convert the UE8 (or UE4M3) back to F32 scale.
// 4. Reciprocal of F32 scale with MUFU.
// 5. Multiply each F32 element with the above reciprocal, then convert to ElementD
// ScaleFactor is interleaved 4B aligned format and supports holes.
//    callback supports 1/2/4 elements store in 4B (i.e. 4 elements) depends on auto-vectorize store.
// {$nv-internal-release end}
template<
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementBias_   = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasBlockScaleFactor
    : LinCombPerColBias<ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = true; // {$nv-internal-release}
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};


// D = activation(alpha * acc + beta * C + per-row bias)
// With BlockScaleFactor generation.
// {$nv-internal-release begin}
// 1. Find max of SFVecSize_ F32 elements
// 2. Convert the max to UE8 (or UE4M3) and store the result.
// 3. Convert the UE8 (or UE4M3) back to F32 scale.
// 4. Reciprocal of F32 scale with MUFU.
// 5. Multiply each F32 element with the above reciprocal, then convert to ElementD
// ScaleFactor is interleaved 4B aligned format and supports holes.
//    callback supports 1/2/4 elements store in 4B (i.e. 4 elements) depends on auto-vectorize store.
// {$nv-internal-release end}
template<
  template <class> class ActivationFn_,
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementBias_   = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltActBlockScaleFactor
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = true; // {$nv-internal-release}
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};


// D = activation(alpha * acc + beta * C + per-col bias)
// With BlockScaleFactor generation.
// {$nv-internal-release begin}
// 1. Find max of SFVecSize_ F32 elements
// 2. Convert the max to UE8 (or UE4M3) and store the result.
// 3. Convert the UE8 (or UE4M3) back to F32 scale.
// 4. Reciprocal of F32 scale with MUFU.
// 5. Multiply each F32 element with the above reciprocal, then convert to ElementD
// ScaleFactor is interleaved 4B aligned format and supports holes.
//    callback supports 1/2/4 elements store in 4B (i.e. 4 elements) depends on auto-vectorize store.
// {$nv-internal-release end}
template<
  template <class> class ActivationFn_,
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  class ElementBias_   = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerColBiasEltActBlockScaleFactor
    : LinCombPerColBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD1x_SfD1x_Supported = true; // {$nv-internal-release}
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
};


// {$nv-internal-release begin}
template<
  class GmemLayoutTagExtraD_,
  template <class> class ActivationFn_,
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class ElementBias_   = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentExtraD_ = 128 / cute::sizeof_bits_v<ElementOutput_>,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltActSfLayout1dOutD2xSfD2xBlockScaleFactor
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout1d_OutD2x_SfD2x_Supported = true;

  using ElementOutput = ElementOutput_; 
  using GmemLayoutTagExtraD = GmemLayoutTagExtraD_;
  static constexpr int AlignmentExtraD = AlignmentExtraD_;
  // SF layouts are pre-determined.
  using GmemLayoutTagScalefactor = cutlass::layout::RowMajor;
  using GmemLayoutTagExtraScalefactor = cutlass::layout::ColumnMajor; 
};


template<
  template <class> class ActivationFn_,
  int SFVecSize_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementBlockScaleFactor_,
  class ElementBias_   = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombPerRowBiasEltActSfLayout2dOutD1xSfD2xBlockScaleFactor
    : LinCombPerRowBiasEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementBias_, ElementSource_, ElementScalar_, AlignmentBias_, RoundStyle_> {
  using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static_assert(SFVecSize == 32, "only vector size of 32 is supported");
  static constexpr bool IsBlockScaleSupported = true;
  static constexpr bool IsSfLayout2d_OutD1x_SfD2x_Supported = true;
  // SF layouts are pre-determined.
  using GmemLayoutTagScalefactor = cutlass::layout::RowMajor;
  using GmemLayoutTagExtraScalefactor = cutlass::layout::ColumnMajor; 
};

// {$nv-internal-release end}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
