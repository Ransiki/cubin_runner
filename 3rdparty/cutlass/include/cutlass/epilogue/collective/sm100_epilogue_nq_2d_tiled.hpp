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
  \brief Functor performing elementwise operations used by epilogues.
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"

#include "cutlass/arch/memory.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

namespace detail {
template<class T, class = void>
struct ElementwiseOpDispatcherNQ2DTiled {
  using Arguments = EmptyArguments;

  T op;

  CUTLASS_HOST_DEVICE
  ElementwiseOpDispatcherNQ2DTiled(Arguments) {}

  template <typename ValueType>
  CUTLASS_HOST_DEVICE
  ValueType operator()(ValueType value) {
    return op(value);
  }
};

template<class T>
struct ElementwiseOpDispatcherNQ2DTiled<T, std::void_t<typename T::Arguments>> {
  using Arguments = typename T::Arguments;

  Arguments args;
  T op;

  CUTLASS_HOST_DEVICE
  ElementwiseOpDispatcherNQ2DTiled(Arguments args_):args(args_) {}

  template <typename ValueType>
  CUTLASS_HOST_DEVICE
  ValueType operator()(ValueType value) {
    if constexpr (cute::is_same_v<T, Clamp<ValueType>>) {
        constexpr bool PropagateNaN = true;
        maximum<T, PropagateNaN> mx;
        minimum<T, PropagateNaN> mn;
        auto tmp = mx(value, args.lower_bound);
        jetfire::warp_switch(); // {$nv-internal-release}
        tmp = mn(tmp, args.upper_bound);
        jetfire::warp_switch(); // {$nv-internal-release}
        return tmp;
    }
    return op(value, args);
  }
};

}

/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
/// Specific opt for nq_2d_tiled kernel
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  typename ElementT_,
  int ElementsPerAccess,
  typename ElementwiseOp_ = Identity<ElementCompute_>,
  typename BinaryOp_ = plus<ElementCompute_>,
  bool StoreT_ = true,
  typename ElementVector_ = ElementC_
>
class LinearCombinationPerChannelScalingBiasElementwiseNQ2DTiled {
public:

  using ElementOutput = ElementC_;
  using ElementD = ElementOutput;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementZ = ElementZ_;
  using ElementT = ElementT_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  /// Follow cutlass3x EVT aliases
  static bool const IsEltActSupported = true;
  static bool const IsPerChannelScalingSupported = true;

  using ElementwiseOp = ElementwiseOp_;
  using BinaryOp = BinaryOp_;

  using ElementwiseOpDispatcher = detail::ElementwiseOpDispatcherNQ2DTiled<ElementwiseOp>;
  using ElementwiseArguments = typename ElementwiseOpDispatcher::Arguments;

  // Indicates that this epilogue applies only one binary operation
  static bool const kIsSingleSource = true;


  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementC, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  // Definitions needed for collective epilogue
  using FragmentSource = FragmentC;
  using FragmentOutput = FragmentZ;
  using ElementBias = ElementVector;
  using FragmentBias = Array<ElementBias, kElementsPerAccess>;
  using ActivationFn = ElementwiseOp;
  static const ScaleType::Kind kScale = ScaleType::PerChannelScaling;

  static bool const kIsHeavy = kIsHeavy_member_or_false<ElementwiseOp>::value;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = true;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = StoreT_;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementCompute beta;                   ///< scales source tensor
    ElementwiseArguments  elementwise;     ///< Arguments for elementwise operation

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      alpha_ptr(nullptr),
      beta_ptr(nullptr),
      beta(ElementCompute(0)) { }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementwiseArguments  elementwise_ = ElementwiseArguments{}
    ): beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), elementwise(elementwise_) {

    }

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr) {

    }
  };

private:

  //
  // Data members
  //

  ElementCompute const* beta_ptr_ = nullptr;
  ElementCompute beta_ = 0;
  ElementwiseArguments const &elementwise_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  LinearCombinationPerChannelScalingBiasElementwiseNQ2DTiled(Params const &params): elementwise_(params.elementwise) {
    if (params.beta_ptr) {
      beta_ptr_ = params.beta_ptr;
    }
    else {
      beta_ = params.beta;
    }
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ptr_ != nullptr || beta_ != ElementCompute(0);
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is true
  /// D = elementwise_op(vector_alpha * accumulator + vector_beta * source + bias)
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const & valpha,
    FragmentCompute const & vbeta,
    FragmentCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    jetfire::warp_switch(); // {$nv-internal-release}

    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    jetfire::warp_switch(); // {$nv-internal-release}

    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i] + vbeta[i] * tmp_C[i], vbias[i]);
      jetfire::warp_switch(); // {$nv-internal-release}
      result_T[i] = z;
      result_Z[i] = elementwise_op(z, elementwise_args);
      jetfire::warp_switch(); // {$nv-internal-release}
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);
    jetfire::warp_switch(); // {$nv-internal-release}

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when elementwise_op require arguments and is_source_needed() is false
  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const & valpha,
    FragmentCompute const & vbias,
    ElementwiseArgs const &elementwise_args) const {

    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    jetfire::warp_switch(); // {$nv-internal-release}
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i], vbias[i]);
      jetfire::warp_switch(); // {$nv-internal-release}
      result_T[i] = z;
      result_Z[i] = elementwise_op(z, elementwise_args);
      jetfire::warp_switch(); // {$nv-internal-release}
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);
    jetfire::warp_switch(); // {$nv-internal-release}

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const & valpha,
    FragmentCompute const & vbeta,
    FragmentCompute const & vbias) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    jetfire::warp_switch(); // {$nv-internal-release}
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    jetfire::warp_switch(); // {$nv-internal-release}
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i] + vbeta[i] * tmp_C[i], vbias[i]);
      jetfire::warp_switch(); // {$nv-internal-release}
      result_T[i] = z;
      result_Z[i] = elementwise_op(z);
      jetfire::warp_switch(); // {$nv-internal-release}
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const & valpha,
    FragmentCompute const & vbias) const {

    ElementwiseOpDispatcher elementwise_op(elementwise_);
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    jetfire::warp_switch(); // {$nv-internal-release}
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(valpha[i] * tmp_Accum[i], vbias[i]);
      jetfire::warp_switch(); // {$nv-internal-release}
      result_T[i] = z;
      result_Z[i] = elementwise_op(z);
      jetfire::warp_switch(); // {$nv-internal-release}
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

};

} // namespace collective
} // namespace epilogue
} // namespace thread


namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// For Nq 2d tiled sm100 kernels
template <
  class EpilogueTile_,
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ElementAccumulator_,
  class ElementCompute_,
  class SmemLayoutAtomC_,
  class SmemLayoutAtomD_,
  class ElementR2T_,
  class FusionOp_,
  class CopyOpT2R_,
  class CopyOpR2S_,
  class CopyOpG2S_,
  class CopyOpS2R_
>
class CollectiveEpilogue<
    Sm100Nq2dTiledWarpSpecialized,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ElementAccumulator_,
    ElementCompute_,
    SmemLayoutAtomC_,
    SmemLayoutAtomD_,
    ElementR2T_,
    FusionOp_,
    CopyOpT2R_,
    CopyOpR2S_,
    CopyOpG2S_,
    CopyOpS2R_>
{
public:
  //
  // Type Aliases
  //
  // Derived types of output thread level operator
  using DispatchPolicy = Sm100Nq2dTiledWarpSpecialized;
  using EpilogueTile = EpilogueTile_;
  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD_, ElementC_>;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using ElementR2T = ElementR2T_;
  using CopyOpT2R = CopyOpT2R_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyOpG2S = CopyOpG2S_;
  using CopyOpS2R = CopyOpS2R_;

  using ThreadEpilogueOp = FusionOp_;
  static constexpr auto ScaleType = DisableSource ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;
  static constexpr auto PerChannelScaleType = thread::ScaleType::PerChannelScaling;
  static constexpr auto RoundStyle = FloatRoundStyle::round_to_nearest;
  // Revisit FragmentSize setting in perf tuning
  static constexpr int FragmentSize = sizeof(ElementD) == 4 ? 16 : 32;

  static_assert(!FusionOp_::IsPerColScaleSupported, "Per-column scaling is not supported.");
  static_assert(!FusionOp_::IsPerColBiasSupported, "Per-column bias is not supported.");
  // Select ThreadEpiOp based on FusionOp
  using ThreadEpiOp = cute::conditional_t<
    FusionOp_::IsPerRowScaleSupported,
    cute::conditional_t<
      (!FusionOp_::IsPerRowBiasSupported || (cute::is_void_v<typename FusionOp_::ElementBias>)) && cute::is_same_v<typename FusionOp_::ActivationFn, thread::Identity<ElementCompute>>,
      // Per-channel scaling
      thread::LinearCombination<
          ElementD, FragmentSize, ElementAccumulator, ElementCompute,
          PerChannelScaleType, RoundStyle, ElementC
      >,
      // Per-channel scaling / bias / activation
      thread::LinearCombinationPerChannelScalingBiasElementwiseNQ2DTiled<
          ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, FragmentSize,
          typename FusionOp_::ActivationFn, cutlass::plus<ElementCompute>,
          false, typename FusionOp_::ElementBias
      >
    >,
    cute::conditional_t<
      FusionOp_::IsEltActSupported,
      // Scaling / bias / activation
      thread::LinearCombinationBiasElementwise<
          ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, FragmentSize,
          typename FusionOp_::ActivationFn, cutlass::plus<ElementCompute>,
          false, typename FusionOp_::ElementBias>,
      cute::conditional_t<
        FusionOp_::IsPerRowBiasSupported,
        // Scaling / bias
        thread::LinearCombinationBiasElementwise<
            ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, FragmentSize,
            thread::Identity<ElementCompute>, cutlass::plus<ElementCompute>,
            false, typename FusionOp_::ElementBias>,
        // Scaling
        thread::LinearCombination<
            ElementD, FragmentSize, ElementAccumulator, ElementCompute,
            ScaleType, RoundStyle, ElementC>
      >
    >
  >;

  static constexpr bool IsActHasArgs = detail::sm100_act_has_arguments<typename FusionOp_::ActivationFn>::value;

  using ElementOutput = typename ThreadEpiOp::ElementOutput;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpiOp>::type;
  using ElementAlpha = ElementCompute;
  using ElementBeta = ElementCompute;
  using LoadPipeline = cutlass::PipelineAsync<1>;
  using LoadPipelineState = cutlass::PipelineState<1>;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  using ElementS2G = uint128_t;
  using ElementG2S = uint128_t;

  constexpr static int NumTensorDimensions = 5;

  constexpr static int ThreadCount = 128;
  constexpr static bool IsSourceSupported = !DisableSource;
  constexpr static bool IsEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpiOp>::value;
  constexpr static bool IsPerChannelScalingSupported = detail::IsThreadEpilogueOpWithPerChannelScaling<ThreadEpiOp>::value;

  using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{}, shape(EpilogueTile{})));    // (EPI_TILE_M,EPI_TILE_N)
  using SmemLayoutD = decltype(tile_to_shape(SmemLayoutAtomD{}, shape(EpilogueTile{})));    // (EPI_TILE_M,EPI_TILE_N)

  constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});

  struct TensorStorageWithC {
    alignas(SmemAlignmentC) array_aligned<ElementC, cute::cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) array_aligned<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };

  struct TensorStorageWithoutC {
    alignas(SmemAlignmentD) array_aligned<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };

  struct SharedStorage {
    using TensorStorage = cute::conditional_t<IsSourceSupported, TensorStorageWithC, TensorStorageWithoutC>;
    TensorStorage tensors;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Args without activation arguments
  template <class ActivationFn, class = void>
  struct EpilogueOpArgs {
    ElementAlpha alpha{0};
    ElementBeta beta{0};
    ElementAlpha const* alpha_ptr = nullptr;
    ElementBeta const* beta_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
  };

  // Args with activation arguments
  template <class ActivationFn>
  struct EpilogueOpArgs<ActivationFn, cute::enable_if_t<detail::sm100_act_has_arguments<ActivationFn>::value>> {
    ElementAlpha alpha{0};
    ElementBeta beta{0};
    ElementAlpha const* alpha_ptr = nullptr;
    ElementBeta const* beta_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    typename ActivationFn::Arguments activation{};
  };

  // Host side epilogue arguments
  struct Arguments {
    EpilogueOpArgs<typename FusionOp_::ActivationFn> thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Params without activation arguments
  template <class ActivationFn, class = void>
  struct ParamsType {
    typename ThreadEpiOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementBias const* ptr_Bias = nullptr;
  };

  // Params with activation arguments
  template <class ActivationFn>
  struct ParamsType<ActivationFn, cute::enable_if_t<detail::sm100_act_has_arguments<ActivationFn>::value>> {
    typename ThreadEpiOp::Params thread{};
    typename ActivationFn::Arguments activation{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementBias const* ptr_Bias = nullptr;
  };

  // Device side epilogue params
  using Params = ParamsType<typename FusionOp_::ActivationFn>;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {

    auto thread_op_args = [=]() {
      if constexpr (IsPerChannelScalingSupported) {
        return typename ThreadEpiOp::Params {args.thread.alpha_ptr, args.thread.beta_ptr};
      } else {
        return typename ThreadEpiOp::Params {args.thread.alpha, args.thread.beta};
      }
    }();

    if constexpr (IsActHasArgs) {
      return {
        thread_op_args,
        args.thread.activation,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        args.thread.bias_ptr
      };
    }
    else {
      return {
        thread_op_args,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        args.thread.bias_ptr
      };
    }
  }

  template<class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    // Alias
    auto const& filter_shape = problem_shape.shape_B; // [K,T,R,S,C]
    auto const& output_channel = filter_shape[0];

    bool implementable = true;

    // Alignment requirement for vectorized G2S copy of residual load
    implementable &= ((output_channel * sizeof(ElementC)) % sizeof(ElementG2S)) == 0;
    // Alignment requirement for vectorized S2G copy of output store
    implementable &= ((output_channel * sizeof(ElementD)) % sizeof(ElementS2G)) == 0;

    // Max offset under uint32_t
    auto problemMNKL = cutlass::conv::detail::get_transformed_problem_shape_MNK_nq_2d_tiled(problem_shape);

    auto M = get<0>(problemMNKL);               // K
    auto N = shape<1>(problemMNKL);             // (Q,N)
    auto P = get<1>(shape<2>(problemMNKL));     // P
    auto Z = get<3>(problemMNKL);               // Z

    // Unpack problem shape and compute stride by ourselves
    auto [stride_tile_q,
          stride_tile_p,
          stride_tile_z,
          stride_tile_n] = get<0>(args.dD);          // (q, p, z, n)

    // Construct real layout D
    auto layoutD = make_layout(make_shape(M, N, Z, P),
      make_stride(_1{}, make_stride(stride_tile_q, stride_tile_n), stride_tile_z, stride_tile_p));         // (M,N,Z,P)
    implementable &= cute::cosize(layoutD) <= static_cast<uint32_t>(0xffffffffu);

    return implementable;
  }

  // template<class TensorStorage>
  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params,
                     [[maybe_unused]] TensorStorage& shared_tensors)
                     : params(params), epilogue_op(params.thread) {
  };

  CUTLASS_DEVICE bool
  is_source_needed() {
    return epilogue_op.is_source_needed() && params.ptr_C != nullptr;
  }

  template <
    class ProblemShape,
    class TileShape,
    class Params
  >
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape const& problem_shape,
      TileShape const& tile_shape,
      Params const& params) {
    auto [gC_mnl, cC_mnl] = construct_C_D_tensors(params.ptr_C, params.dC, problem_shape, tile_shape);

    return cute::make_tuple(gC_mnl, cC_mnl);
  }

  template<
    class EpiLoadPipeline,
    class EpiLoadPipelineState,
    class GTensorC,
    class CTensorC,
    class TileCoordMNKL
  >
  CUTLASS_DEVICE auto
  load(
      EpiLoadPipeline epi_load_pipeline,
      EpiLoadPipelineState epi_load_pipe_producer_state,
      cute::tuple<GTensorC,CTensorC> const& load_inputs,
      TileCoordMNKL tile_coord_mnkl,
      int p_pixels_start, int p_pixels_end,
      TensorStorage& shared_tensors) {
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    const int thread_idx = canonical_lane_idx() ;

    // matrix C
    GTensorC mC     = get<0>(load_inputs);                                                   // (M,N,Z,P)
    // Get cC coordinate tensor for current CTA
    CTensorC cC_mnl = get<1>(load_inputs);                                                   // (TILE_M,TILE_N,m,n,l,p)
    Tensor   cC     = cC_mnl(_,_,m_coord,n_coord,l_coord,_);                                 // (TILE_M,TILE_N,      p)

    // Get SMEM tensor
    ElementC* ptr_sC = nullptr;
    if constexpr (IsSourceSupported) { ptr_sC = shared_tensors.smem_C.data(); }
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(                            // (EPI_TILE_M,EPI_TILE_N)
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));

    // (t)hread-partition for (g)mem to (s)mem copy
    constexpr int VectorizedG2SCopyElements = sizeof(ElementG2S) / sizeof(ElementC);
    auto atom_tv_layout = make_layout(make_shape(Int<NumThreadsPerWarp>{}, Int<VectorizedG2SCopyElements>{}), LayoutRight{});
    auto tiled_g2s = make_cotiled_copy(
      CopyOpG2S{},
      atom_tv_layout,
      sC_epi.layout().layout_b());
    ThrCopy thread_g2s  = tiled_g2s.get_slice(thread_idx);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int p = p_pixels_start; p < p_pixels_end; ++p) {
      // Epilogue silcing
      auto cC_slice = cC(_,_,p);                                                                     // (TILE_M,TILE_N)

      // Epilogue subtiling
      constexpr auto epi_tiler = EpilogueTile{};                                          // (EPI_TILE_M : EPI_TILE_N)
      Tensor cC_epi = zipped_divide(cC_slice, epi_tiler)(make_coord(_,_),_);   // (EPI_TILE_M,EPI_TILE_N,(EPI_M,EPI_N))

      static_assert(size<2,0>(cC_epi) == 1, "Multiple EPI_M is not supported");
      for (int epi_n = 0; epi_n < size<2,1>(cC_epi); ++epi_n) {
        // Get epilogue subtile
        Tensor cC_epi_slice = cC_epi(_,_,make_coord(_0{}/* epi_m */,epi_n));                 // (EPI_TILE_M,EPI_TILE_N)

        // (g)mem to (s)mem copy (tGS_)
        Tensor tGS_cC = thread_g2s.partition_S(cC_epi_slice);                                      // (G2R,G2R_M,G2R_N)
        Tensor tGS_sC = thread_g2s.partition_D(sC_epi);                                            // (R2S,R2S_M,R2S_N)

        // Acquire an empty buffer
        epi_load_pipeline.producer_acquire(epi_load_pipe_producer_state);

        // CuTe cannot recast a linearized layout to perform LDGSTS.128 on elements like half_t, so break tiled_g2s to multiple manual copys
        // copy_if(tiled_g2s, tGS_pC, tGS_gC, tGS_sC);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<1>(tGS_cC); mi++) {
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < size<2>(tGS_cC); ni++) {
            auto coord_mnzp = tGS_cC(_0{},mi,ni);
            Tensor dst = tGS_sC(_,mi,ni);
            Tensor src = make_tensor(make_gmem_ptr(&mC(coord_mnzp)), size<0>(tGS_sC));
            auto  pred = elem_less(coord_mnzp, tGS_cC.layout().layout_a().shape());
            cute::copy(tiled_g2s.with(pred), src, dst);
          }
        }

        // {$nv-internal-release begin}
        // XXX: WAR for https://nvbugspro.nvidia.com/bug/4705547 to bypass the racecheck failure
        // compute-sanitizer can only correctly handle dependency for LDGSTS writes
        // that use scoreboard counting with {LDGDEPBAR + DEPBAR}.
        // {$nv-internal-release end}
        // Commit buffer full
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        epi_load_pipeline.producer_commit(epi_load_pipe_producer_state);
        ++epi_load_pipe_producer_state;
      }
    }

    return cute::make_tuple(epi_load_pipe_producer_state);
  }

  template <
    class ProblemShape,
    class TileShape,
    class Params
  >
  CUTLASS_DEVICE auto
  store_init(
      ProblemShape const& problem_shape,
      TileShape const& tile_shape,
      Params const& params) {
    auto [gD_mnl, cD_mnl] = construct_C_D_tensors(params.ptr_D, params.dD, problem_shape, tile_shape);

    auto [ptr_alpha, ptr_beta] = [=]() {
      if constexpr (IsPerChannelScalingSupported) {
        return cute::make_tuple(params.thread.alpha_ptr, params.thread.beta_ptr);
      }
      else {
        return cute::make_tuple(&params.thread.alpha, &params.thread.beta);
      }}();
    auto [gAlpha_mnl, cAlpha_mnl] = construct_bias_alpha_beta_tensors(ptr_alpha,       problem_shape, tile_shape);
    auto [gBeta_mnl,  cBeta_mnl ] = construct_bias_alpha_beta_tensors(ptr_beta,        problem_shape, tile_shape);
    auto [gBias_mnl,  cBias_mnl ] = construct_bias_alpha_beta_tensors(params.ptr_Bias, problem_shape, tile_shape);

    return cute::make_tuple(gD_mnl, cD_mnl, gBias_mnl, cBias_mnl, gAlpha_mnl, cAlpha_mnl, gBeta_mnl, cBeta_mnl);
  }

  template<
    class Params,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class EpiLoadPipeline,
    class EpiLoadPipelineState,
    class GTensorD,
    class CTensorD,
    class GTensorBias,
    class CTensorBias,
    class GTensorAlpha,
    class CTensorAlpha,
    class GTensorBeta,
    class CTensorBeta,
    class TileCoordMNKL,
    class AccEngine, class AccLayout
  >
  CUTLASS_DEVICE auto
  store(
      Params const& params,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState accumulator_pipe_consumer_state,
      EpiLoadPipeline epi_load_pipeline,
      EpiLoadPipelineState epi_load_pipe_consumer_state,
      cute::tuple<GTensorD,CTensorD,GTensorBias,CTensorBias,GTensorAlpha,CTensorAlpha,GTensorBeta,CTensorBeta> const& store_inputs,
      TileCoordMNKL tile_coord_mnkl,
      int p_pixels_start, int p_pixels_end,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      [[maybe_unused]] TensorStorage& shared_tensors) {
    using namespace cute;
    using X = Underscore;

    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(AccLayout{}) == 4, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N,STAGE)");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    const int epi_thread_idx = threadIdx.x % ThreadCount;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    // Get accumulator (TMEM) tensor for current CTA
    Tensor   tAcc   = accumulators(_,_0{},_0{},_);                                         // ((TILE_M,TILE_N),ACC_PIPE)

    // Get output tensor (GMEM) without slicing
    GTensorD mD     = get<0>(store_inputs);                                                //  (M,N,Z,P)
    // Get output coordinate tensor for current CTA
    CTensorD cD_mnl = get<1>(store_inputs);                                                //  (TILE_M,TILE_N ,m,n,l,p)
    Tensor   cD     = cD_mnl(_,_,m_coord,n_coord,l_coord,_);                               //  (TILE_M,TILE_N ,      p)

    // Get bias tensor (GMEM) without slicing
    GTensorBias mBias     = get<2>(store_inputs);                                          //  (M,N)
    // Get bias coordinate tensor for current CTA
    CTensorBias cBias_mnl = get<3>(store_inputs);                                          //  (TILE_M,TILE_N ,m,n)
    Tensor      cBias     = cBias_mnl(_,_,m_coord,n_coord);                                //  (TILE_M,TILE_N ,   )

    // Get Alpha tensor (GMEM) for current CTA
    GTensorAlpha mAlpha     = get<4>(store_inputs);                                        //  (M,N)
    // Get Alpha coordinate tensor for current CTA
    CTensorAlpha cAlpha_mnl = get<5>(store_inputs);                                        //  (TILE_M,TILE_N ,m,n)
    Tensor       cAlpha     = cAlpha_mnl(_,_,m_coord,n_coord);                             //  (TILE_M,TILE_N ,   )

    // Get Beta tensor (GMEM) for current CTA
    GTensorBeta mBeta     = get<6>(store_inputs);                                          //  (M,N)
    // Get Beta coordinate tensor for current CTA
    CTensorBeta cBeta_mnl = get<7>(store_inputs);                                          //  (TILE_M,TILE_N ,m,n)
    Tensor      cBeta     = cBeta_mnl(_,_,m_coord,n_coord);                                //  (TILE_M,TILE_N ,   )

    // Get SMEM tensor
    ElementC * ptr_sC = nullptr;
    if constexpr (IsSourceSupported) { ptr_sC = shared_tensors.smem_C.data(); }
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(                            // (EPI_TILE_M,EPI_TILE_N)
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));
    ElementD* ptr_sD = shared_tensors.smem_D.data();
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(                            // (EPI_TILE_M,EPI_TILE_N)
                      make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));

    // Use epi_tiler to tile Acc in epilogue steps
    constexpr auto epi_tiler = EpilogueTile{}; // (EPI_TILE_M : EPI_TILE_N)

    // Accumulator for entire CTA tile
    auto tC_slice_sample = tAcc(_,_0{}/* ACC_PIPE */)(make_coord(make_coord(_,_)));          // (ACC_TILE_M,ACC_TILE_N)
    // Accumulator for one epilogue tile
    //   Slicing on EPI_M/EPI_N because tiled_r2s only copy one epilogue tile each time
    auto tAcc_epi_sample = zipped_divide(
      tC_slice_sample, epi_tiler)(make_coord(_,_),_0{}/* (EPI_M,EPI_N) */);                  // (EPI_TILE_M,EPI_TILE_N)

    // (t)hread-partition for (t)mem to (r)egister copy
    TiledCopy tiled_t2r  = make_tmem_copy(CopyOpT2R{}, tAcc_epi_sample);
    ThrCopy  thread_t2r  = tiled_t2r.get_slice(epi_thread_idx);

    // (t)hread-partition for (r)egister to (s)mem copy
    TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<CopyOpR2S, ElementD>{}, tiled_t2r);
    ThrCopy thread_r2s  = tiled_r2s.get_slice(epi_thread_idx);

    // (t)hread-partition for (s)mem to (g)mem copy
    constexpr int VectorizedS2GCopyElements = sizeof(ElementS2G) / sizeof(ElementD);
    auto atom_tv_layout = make_layout(make_shape(Int<ThreadCount>{}, Int<VectorizedS2GCopyElements>{}), LayoutRight{});
    auto tiled_s2g = make_cotiled_copy(
      Copy_Atom<UniversalCopy<ElementS2G>, ElementD>{},
      atom_tv_layout,
      sD_epi.layout().layout_b());
    ThrCopy thread_s2g  = tiled_s2g.get_slice(epi_thread_idx);

    // (t)hread-partition for (s)mem to (r)egister copy
    TiledCopy tiled_s2r = make_tiled_copy_D(Copy_Atom<CopyOpS2R, ElementC>{}, tiled_t2r);
    ThrCopy thread_s2r  = tiled_s2r.get_slice(epi_thread_idx);

    // Epilogue slicing outside loop p
    Tensor cD_slice = cD(_,_,_0{});                                                           // (    TILE_M,    TILE_N)
    Tensor cBias_slice = cBias(_,_);                                                          // (    TILE_M,    TILE_N)
    Tensor cAlpha_slice = cAlpha(_,_);                                                        // (    TILE_M,    TILE_N)
    Tensor cBeta_slice = cBeta(_,_);                                                          // (    TILE_M,    TILE_N)

    // Epilogue subtiling
    Tensor cD_epi   = zipped_divide(cD_slice, epi_tiler)(make_coord(_,_),_); // (EPI_TILE_M,EPI_TILE_N,(EPI_M,EPI_N))
    Tensor cBias_epi = zipped_divide(cBias_slice, epi_tiler)(make_coord(_,_),_); // (EPI_TILE_M,EPI_TILE_N,(EPI_M,EPI_N))
    Tensor cAlpha_epi = zipped_divide(cAlpha_slice, epi_tiler)(make_coord(_,_),_); // (EPI_TILE_M,EPI_TILE_N,(EPI_M,EPI_N))
    Tensor cBeta_epi = zipped_divide(cBeta_slice, epi_tiler)(make_coord(_,_),_); // (EPI_TILE_M,EPI_TILE_N,(EPI_M,EPI_N))

    // (t)mem to (r)egister copy (tTR_)
    Tensor tTR_cD     = thread_t2r.partition_D(  cD_epi);                           // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))
    Tensor tTR_cBias  = thread_t2r.partition_D(cBias_epi);                          // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))
    Tensor tTR_cAlpha = thread_t2r.partition_D(cAlpha_epi);                         // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))
    Tensor tTR_cBeta  = thread_t2r.partition_D(cBeta_epi);                          // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))

    // Allocate register tensors
    // rAcc covers entire epilogue block
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_cD));              // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))
    // rC covers a epilogue tile
    Tensor tTR_rC   = make_tensor<ElementC>(shape(tTR_cD(_,_,_,make_coord(_0{},_0{}))));         // (T2R,T2R_M,T2R_N)
    // rD covers a epilogue tile
    Tensor tTR_rD   = make_tensor<ElementD>(shape(tTR_cD(_,_,_,make_coord(_0{},_0{}))));         // (T2R,T2R_M,T2R_N)

    // rBias covers a epilogue tile
    Tensor tTR_rBias  = make_tensor<ElementBias>(shape(tTR_cD));    // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))
    // rAlpha covers a epilogue tile
    Tensor tTR_rAlpha = make_tensor<ElementAlpha>(shape(tTR_cD));   // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))
    // rBeta covers a epilogue tile
    Tensor tTR_rBeta  = make_tensor<ElementBeta>(shape(tTR_cD));    // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))

    // Vectorized fragment view
    Tensor tTR_rAcc_frg = recast<typename ThreadEpiOp::FragmentAccumulator>(coalesce(tTR_rAcc)); // (EPI_V x EPI_M x EPI_N)
    Tensor tTR_rC_frg   = recast<typename ThreadEpiOp::FragmentSource>     (coalesce(tTR_rC)  ); // (EPI_V)
    Tensor tTR_rD_frg   = recast<typename ThreadEpiOp::FragmentOutput>     (coalesce(tTR_rD)  ); // (EPI_V)
    Tensor tTR_rAlpha_frg = recast<typename ThreadEpiOp::FragmentCompute>  (coalesce(tTR_rAlpha)); // (EPI_V x EPI_M x EPI_N)
    Tensor tTR_rBeta_frg  = recast<typename ThreadEpiOp::FragmentCompute>  (coalesce(tTR_rBeta) ); // (EPI_V x EPI_M x EPI_N)

    CUTE_STATIC_ASSERT(size(tTR_rC_frg) == size(tTR_rD_frg), "Fragment sizes of tTR_rC_frg and tTR_rD_frg do not match.");
    CUTE_STATIC_ASSERT(size(tTR_rAlpha_frg) == size(tTR_rAcc_frg), "Fragment sizes of tTR_rAlpha_frg and tTR_rAcc_frg do not match.");
    CUTE_STATIC_ASSERT(size(tTR_rBeta_frg) == size(tTR_rAcc_frg), "Fragment sizes of tTR_rBeta_frg and tTR_rAcc_frg do not match.");

    CUTE_STATIC_ASSERT(size(tTR_rAcc  ) % ThreadEpiOp::kCount == 0, "Fragment size does not vectorize properly");
    CUTE_STATIC_ASSERT(size(tTR_rC    ) % ThreadEpiOp::kCount == 0, "Fragment size does not vectorize properly");
    CUTE_STATIC_ASSERT(size(tTR_rD    ) % ThreadEpiOp::kCount == 0, "Fragment size does not vectorize properly");
    CUTE_STATIC_ASSERT(size(tTR_rBias ) % ThreadEpiOp::kCount == 0, "Fragment size does not vectorize properly");
    CUTE_STATIC_ASSERT(size(tTR_rAlpha) % ThreadEpiOp::kCount == 0, "Fragment size does not vectorize properly");
    CUTE_STATIC_ASSERT(size(tTR_rBeta ) % ThreadEpiOp::kCount == 0, "Fragment size does not vectorize properly");

    // (s)mem to (r)egister copy (tSR_)
    Tensor tSR_rC = thread_s2r.retile_D(tTR_rC);                                                 // (S2R,S2R_M,S2R_N)
    Tensor tSR_sC = thread_s2r.partition_S(sC_epi);                                              // (S2R,S2R_M,S2R_N)

    // (r)egister to (s)mem copy (tRS_)
    Tensor tRS_rD = thread_r2s.retile_S(tTR_rD);                                                 // (R2S,R2S_M,R2S_N)
    Tensor tRS_sD = thread_r2s.partition_D(sD_epi);                                              // (R2S,R2S_M,R2S_N)

    // To get EPI_N size.
    static_assert(size<2,0>(cD_epi) == 1, "Multiple EPI_M is not supported");
    constexpr int EPI_N = size<2,1>(cD_epi);

    // Copy SMEM -> RMEM -> GMEM
    Tensor sD_epi_slice = sD_epi;                                                                // (EPI_TILE_M,EPI_TILE_N)
    // (s)mem to (g)mem copy (tSG_)
    Tensor tSG_sD = thread_s2g.partition_S(sD_epi_slice);                                        // (S2R,S2R_M,S2R_N)
    // To get MI NI in gD STG iteration.
    constexpr int MI = size<1>(tSG_sD);
    constexpr int NI = size<2>(tSG_sD);

    // smem ptr as STG source.
    uint4 *src_ptr[MI][NI];
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < MI; mi++) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < NI; ni++) {
        src_ptr[mi][ni] = recast_ptr<uint4>(&tSG_sD(_0{},mi,ni));
      }
    }

    // gmem ptr without P dim as STG dst.
    uint32_t dst_offset[EPI_N][MI][NI];
    // Predicates to prevent OOB copy
    int dst_pred[EPI_N][MI][NI];

    ElementD *dst_base_ptr = recast_ptr<ElementD>(&mD(_0{}));
    auto mOffset = make_tensor(counting_iterator<uint32_t>(0), mD.layout());

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < EPI_N; epi_n++) {
      Tensor cD_epi_slice = cD_epi(_,_,make_coord(_0{},epi_n));                                  // (EPI_TILE_M,EPI_TILE_N)
      Tensor tSG_cD = thread_s2g.partition_D(cD_epi_slice);                                      // (R2G,R2G_M,R2G_N)

      static_assert((size<1>(tSG_cD) == MI) && (size<2>(tSG_cD) == NI), "shape of partitioned gD and sD does not match.");

      // Since we iterate on dim P, dim P will not Hit OOB copy.
      // So we only focus on dim NQ and dim K's OOB.
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < MI; mi++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < NI; ni++) {
          auto coord_mnzp = tSG_cD(_0{},_,_)(mi,ni);
          dst_offset[epi_n][mi][ni] = mOffset(coord_mnzp);
          dst_pred[epi_n][mi][ni] = elem_less(coord_mnzp,tSG_cD.layout().layout_a().shape());
        }
      }
    }

    if constexpr (IsEpilogueBiasSupported) {
      if (params.ptr_Bias) {
        load_beta_alpha_bias_tensors<ElementBias>(tTR_cBias, mBias, tTR_rBias);
      }
      else {
        if constexpr (IsPerChannelScalingSupported) {
          // Pass bias by default; set zero for params.ptr_Bias == nullptr
          fill(tTR_rBias, ElementBias(0));
        }
      }
    }

    if constexpr (IsPerChannelScalingSupported) {
      load_beta_alpha_bias_tensors<ElementAlpha>(tTR_cAlpha, mAlpha, tTR_rAlpha);
      if (params.thread.beta_ptr) {
        load_beta_alpha_bias_tensors<ElementBeta>(tTR_cBeta, mBeta, tTR_rBeta);
      }
    }

    auto stride_p = stride<3>(mD);

    // Use uint32_t instead of `int` as the index type, to avoid extra instructions
    // due to conversion from int32_t to uint64_t (for address).
    CUTLASS_PRAGMA_NO_UNROLL
    for (uint32_t p = static_cast<uint32_t>(p_pixels_start); p < static_cast<uint32_t>(p_pixels_end); ++p) {

      uint32_t offset = p * stride_p;

      // Wait for mma warp to fill tmem buffer with accumulator results
      acc_pipeline.consumer_wait(accumulator_pipe_consumer_state);

      // Epilogue silcing
      Tensor tC_slice = tAcc(_,accumulator_pipe_consumer_state.index())(
        make_coord(make_coord(_,_)));                                                        // (ACC_TILE_M,ACC_TILE_N)

      // Epilogue subtiling
      Tensor tAcc_epi = zipped_divide(tC_slice, epi_tiler)(make_coord(_,_),_); // (EPI_TILE_M,EPI_TILE_N,(EPI_M,EPI_N))

      // (t)mem to (r)egister copy (tTR_)
      // Full tAcc_epi is needed to be partitioned because t2r copy entire epilogue block at once
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc_epi);                            // (T2R,T2R_M,T2R_N,(EPI_M,EPI_N))

      // Copy accumulator from TMEM to RMEM with TMEM_LOAD
      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

      cutlass::arch::fence_view_async_tmem_load();

      // After flushing acc can be released
      acc_pipeline.consumer_release(accumulator_pipe_consumer_state);
      ++accumulator_pipe_consumer_state;

      // Loop over epilogue tiles
      CUTLASS_PRAGMA_UNROLL
      for (int epi_n = 0; epi_n < EPI_N; ++epi_n) {

        // Static branch for residual C
        if (IsSourceSupported) {
          // Runtime branch for residual C
          if (is_source_needed()) {
            // Wait for C tensor being loaded to SMEM
            epi_load_pipeline.consumer_wait(epi_load_pipe_consumer_state);

            // Copy residual from SMEM to RMEM with LDSM
            copy(tiled_s2r, tSR_sC, tSR_rC);

            // After loading C to RMEM, SMEM can be released
            epi_load_pipeline.consumer_release(epi_load_pipe_consumer_state);
            ++epi_load_pipe_consumer_state;
          }
        }

        // Perform epilogue_op: LinearCombinationPerChannelScalingBiasElementwise/LinearCombinationBiasElementwise
        if constexpr (IsEpilogueBiasSupported) {
          cutlass::NumericArrayConverter<typename ThreadEpiOp::ElementCompute, ElementBias,
            ThreadEpiOp::FragmentBias::kElements> bias_converter;
          Tensor tTR_rBias_frg = recast<typename ThreadEpiOp::FragmentBias>  (coalesce(tTR_rBias)); // (EPI_V x EPI_M x EPI_N)
          CUTE_STATIC_ASSERT(size(tTR_rBias_frg) == size(tTR_rAcc_frg), "Fragment sizes of tTR_rBias_frg and tTR_rAcc_frg do not match.");
          CUTLASS_PRAGMA_UNROLL
          for (int ii = 0; ii < size(tTR_rD_frg); ++ii) {
            typename ThreadEpiOp::FragmentCompute converted_bias = bias_converter(tTR_rBias_frg(epi_n * size(tTR_rD_frg) + ii));
            if constexpr (IsActHasArgs) {
              if constexpr (IsPerChannelScalingSupported) {
                if constexpr (!IsSourceSupported) {
                  // D = activation(valpha * accumulator + bias, activation_param)
                  epilogue_op(tTR_rD_frg(ii), tTR_rD_frg(ii), tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii),
                      tTR_rAlpha_frg(epi_n * size(tTR_rD_frg) + ii), converted_bias, params.activation);
                }
                else {
                  // D = activation(valpha * accumulator + vbeta * C + bias, activation_param)
                  epilogue_op(tTR_rD_frg(ii), tTR_rD_frg(ii), tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rC_frg(ii),
                      tTR_rAlpha_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rBeta_frg(epi_n * size(tTR_rD_frg) + ii), converted_bias, params.activation);
                }
              }
              else {
                // D = activation(alpha * accumulator + beta * C + bias, activation_param)
                epilogue_op(tTR_rD_frg(ii), tTR_rD_frg(ii), tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rC_frg(ii), converted_bias, params.activation);
              }
            }
            else {
              if constexpr (IsPerChannelScalingSupported) {
                if constexpr (!IsSourceSupported) {
                  // D = activation(valpha * accumulator + bias)
                  epilogue_op(tTR_rD_frg(ii), tTR_rD_frg(ii), tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rC_frg(ii),
                      tTR_rAlpha_frg(epi_n * size(tTR_rD_frg) + ii), converted_bias);
                }
                else {
                  // D = activation(valpha * accumulator + vbeta * C + bias)
                  epilogue_op(tTR_rD_frg(ii), tTR_rD_frg(ii), tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rC_frg(ii),
                      tTR_rAlpha_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rBeta_frg(epi_n * size(tTR_rD_frg) + ii), converted_bias);
                }
              }
              else {
                // D = activation(alpha * accumulator + beta * C + bias)
                epilogue_op(tTR_rD_frg(ii), tTR_rD_frg(ii), tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rC_frg(ii), converted_bias);
              }
            }
            jetfire::warp_switch(); // {$nv-internal-release}
          }
        // Perform epilogue_op: LinearCombination
        }
        else {
          if constexpr (IsPerChannelScalingSupported) {
            if constexpr (!IsSourceSupported) {
              // D = valpha * accumulator
              CUTLASS_PRAGMA_UNROLL
              for (int ii = 0; ii < size(tTR_rD_frg); ++ii) {
                tTR_rD_frg(ii) = epilogue_op(tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii),
                                             tTR_rAlpha_frg(epi_n * size(tTR_rD_frg) + ii));
                jetfire::warp_switch(); // {$nv-internal-release}
              }
            }
            else {
              // D = valpha * accumulator + vbeta * C
              CUTLASS_PRAGMA_UNROLL
              for (int ii = 0; ii < size(tTR_rD_frg); ++ii) {
                tTR_rD_frg(ii) = epilogue_op(tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii),
                                             tTR_rC_frg(ii),
                                             tTR_rAlpha_frg(epi_n * size(tTR_rD_frg) + ii),
                                             tTR_rBeta_frg(epi_n * size(tTR_rD_frg) + ii));
                jetfire::warp_switch(); // {$nv-internal-release}
              }
            }
          }
          else {
            // D = alpha * accumulator + beta * C
            CUTLASS_PRAGMA_UNROLL
            for (int ii = 0; ii < size(tTR_rD_frg); ++ii) {
              tTR_rD_frg(ii) = epilogue_op(tTR_rAcc_frg(epi_n * size(tTR_rD_frg) + ii), tTR_rC_frg(ii));
              jetfire::warp_switch(); // {$nv-internal-release}
            }
          }
        }

        // Copy RMEM -> SMEM
        copy(tiled_r2s, tRS_rD, tRS_sD);

        // Wait for data to be copied to smem
        synchronize();

        // CuTe cannot recast a linearized layout to perform LDS128/STG128 on elements like half_t, so break tiled_s2g to multiple manual copys
        // copy_if() may hit perf issue too.
        // copy_if(tiled_s2g, tSG_pD, tSG_sD, tSG_gD);

        // insert sync between LDS and STG explicitly for compiler to optimize more easily.
        uint4 src[MI][NI];
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < MI; mi++) {
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < NI; ni++) {
            src[mi][ni] = *(src_ptr[mi][ni]);
            jetfire::warp_switch(); // {$nv-internal-release}
          }
        }

        synchronize();

        // add loop P offset finally on gmem_ptr to avoid too much gmem_ptr calculation instruction.
        // use global_store to generate correct predicated STG instead of branch.
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < MI; mi++) {
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < NI; ni++) {
            ElementD* dst = dst_base_ptr + dst_offset[epi_n][mi][ni] + offset;
            cutlass::arch::global_store<uint4, 16>(src[mi][ni], (void*)dst, dst_pred[epi_n][mi][ni]);
          }
        }
      } // loop epi_n
    } // loop p

    return cute::make_tuple(accumulator_pipe_consumer_state, epi_load_pipe_consumer_state);
  }

private:

  template <
    class Element,
    class CTensorType,
    class MTensorType,
    class RTensorType
  >
  CUTLASS_DEVICE void
  load_beta_alpha_bias_tensors(
      CTensorType& cTensor,
      MTensorType& mTensor,
      RTensorType& rTensor) {
    using X = Underscore;
    // remove all static 0s
    Tensor cTensor_nz = coalesce(cTensor);
    Tensor rTensor_nz = make_tensor<Element>(shape(cTensor_nz));
    // force stride = 0 if mode != 0 since we will broadcast the mode-0 data
    Tensor rTensor_broadcast = make_tensor(rTensor_nz.data(),
                                           make_layout(shape(cTensor_nz),
                                           cute::transform_leaf(stride(cTensor_nz),
                                                                stride(rTensor_nz),
                                                                [](auto const& s, auto const& t) {
                                                                  return conditional_return(s.mode() == Int<0>{}, t, Int<0>{}*t);
                                                                })));
    auto coord_mode0 = cute::transform_leaf(stride(cTensor_nz),
                                            [](auto const&s) {
                                              return conditional_return(s.mode() == Int<0>{}, _, Int<0>{});
                                            });
    // the tensor will become the real object if there is no Underscore
    // add an extra dimension to avoid slicing completely
    auto cTensor_mode0 = make_tensor(cTensor_nz.data(), make_layout(cTensor_nz.layout(), make_layout(make_shape(_1{}))))(coord_mode0,_);
    auto rTensor_mode0 = make_tensor(rTensor_nz.data(), make_layout(rTensor_nz.layout(), make_layout(make_shape(_1{}))))(coord_mode0,_);

    // gmem to rmem
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < size(cTensor_mode0); ++ii) {
      auto coord_mnzp = cTensor_mode0(ii);
      if (elem_less(coord_mnzp, shape(mTensor))) {
        rTensor_mode0(ii) = mTensor(coord_mnzp);
      }
    }
    // broadcast
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < size(rTensor); ++ii) {
      rTensor(ii) = rTensor_broadcast(ii);
    }
  }

  template <
    class Element,
    class Stride,
    class ProblemShape,
    class TileShape
  >
  CUTLASS_DEVICE auto
  construct_C_D_tensors(
      Element *ptr,
      Stride stride,
      ProblemShape const& problem_shape,
      TileShape const& tile_shape) {
    using namespace cute;
    using X = Underscore;

    auto M = get<0>(problem_shape);               // K
    auto N = shape<1>(problem_shape);             // (Q,N)
    auto P = get<1>(shape<2>(problem_shape));     // P
    auto Z = get<3>(problem_shape);               // Z

    // Unpack problem shape and compute stride by ourselves
    auto [stride_tile_q,
          stride_tile_p,
          stride_tile_z,
          stride_tile_n] = get<0>(stride);          // (q, p, z, n)

    // Construct linearized mTensor then tile to gTensor_mnl
    Tensor mTensor = make_tensor(make_gmem_ptr(ptr), make_shape(M, N, Z, P),
      make_stride(_1{}, make_stride(stride_tile_q, stride_tile_n), stride_tile_z, stride_tile_p));         // (M,N,Z,P)
    Tensor mTensor_linear = make_identity_tensor(make_shape(M, size(N), Z, P));                            // (M,N,Z,P)
    Tensor mTensor_linearized = make_tensor(mTensor.data(),
      composition(mTensor.layout(), mTensor_linear(_0{}), mTensor_linear.layout()));                       // (M,N,Z,P)
    Tensor gTensor_mnl = local_tile(
      mTensor_linearized, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                  // (TILE_M,TILE_N,m,n,z,p)

    // Construct linearized cTensor (coordinate tensor for S2G copy predication) then tile to cTensor_mnl
    auto mTensor_linearized_layout = mTensor_linearized.layout();                                          // (M,N,Z,P)
    auto cTensor_layout = make_composed_layout(
      make_layout(
        mTensor_linearized_layout.layout_a().shape(),
        make_basis_like(mTensor_linearized_layout.layout_a().shape())),
      mTensor_linearized_layout.offset(),
      mTensor_linearized_layout.layout_b());                                                               // (M,N,Z,P)
    Tensor cTensor = make_tensor(make_inttuple_iter(coprofile(cTensor_layout.layout_a())), cTensor_layout);
    auto cTensor_mnl = local_tile(
      cTensor, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});                             // (TILE_M,TILE_N,m,n,z,p)

    return cute::make_tuple(mTensor, cTensor_mnl);
  }

  template <
    class Element,
    class ProblemShape,
    class TileShape
  >
  CUTLASS_DEVICE auto
  construct_bias_alpha_beta_tensors(
      Element *ptr,
      ProblemShape const& problem_shape,
      TileShape const& tile_shape) {
    using namespace cute;
    using X = Underscore;

    auto M = get<0>(problem_shape);               // CONV-K
    auto N = size<1>(problem_shape);              // CONV-Q x CONV-N
    Tensor mTensor = make_tensor(make_gmem_ptr(ptr), make_shape(M, N), make_stride(_1{}, _0{}));
    static_assert(is_integral<decltype(M)>::value, "M should be single mode.");
    Tensor gTensor_mnl = local_tile(mTensor, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});

    Tensor cTensor = make_identity_tensor(shape(mTensor));
    Tensor cTensor_mnl = local_tile(cTensor, tile_shape, make_coord(_,_,_), Step<_1,_1, X>{});

    return cute::make_tuple(mTensor, cTensor_mnl);
  }

protected:
  Params const & params;
  ThreadEpiOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
