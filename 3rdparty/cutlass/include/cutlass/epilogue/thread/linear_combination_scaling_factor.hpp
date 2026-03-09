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
  \brief Functor performing linear combination operations (including alpha, beta,
         scale_a/scale_b/scale_c/scale_d, abs_maximum) used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * scale_a * scale_b * accumulator + beta * scale_c * source + Bias
/// Activation(D)
/// if D is fp8 type:
///    abs_max_output = max( abs(d) | (for every d in D))
///    D = scale_d * D
/// endif
///
///

template <
  typename ElementOutput_,                                        ///< Data type used to load and store tensors
  int Count,                                                      ///< Number of elements computed per operation.
  typename ElementAccumulator_ = ElementOutput_,                  ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,                      ///< Data type used to compute linear combination
  typename ElementBias_ = ElementCompute_,                        ///< Data type of Bias elements.
  template<typename T> class ActivationFunctor_ = Identity,        ///< Fused Activation
  ScaleType::Kind Scale = ScaleType::Default,                     ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename ElementSource_ = ElementOutput_,
  bool IsScalingAndAmaxOutputNeeded_ = (cute::is_same_v<ElementOutput_, cutlass::float_e4m3_t> ||
                                        cute::is_same_v<ElementOutput_, cutlass::float_e5m2_t>)
>
class LinearCombinationScalingFactor {
public:

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementBias = ElementBias_;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;
  using ElementScalingFactor = ElementAccumulator_;
  using ElementActivationFunctor = ActivationFunctor_<ElementCompute>;

  static constexpr bool IsScalingAndAmaxOutputNeeded = IsScalingAndAmaxOutputNeeded_;
  static int const kCount = Count;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentBias = Array<ElementBias, kCount>;
  using FragmentSource = Array<ElementC, kCount>;
  using FragmentActivationFunctor = ActivationFunctor_<FragmentCompute>;

  using BinaryOp = plus<FragmentCompute>;
  static const ScaleType::Kind kScale = Scale;
  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha{};                  ///< scales accumulators
    ElementCompute beta{};                   ///< scales source tensor
    ElementCompute const* alpha_ptr = nullptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const* beta_ptr = nullptr;        ///< pointer to source scalar - if not null, loads it from memory

    ElementScalingFactor const* scale_a_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_b_ptr = nullptr; ///< pointer to source scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_c_ptr = nullptr; ///< pointer to source scalar - if not null, loads it from memory
    ElementScalingFactor const* scale_d_ptr = nullptr; ///< pointer to source scalar - if not null, loads it from memory

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }
  
    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
        : alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr){}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr)
        : alpha_ptr(alpha_ptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha,
           ElementCompute beta,
           ElementScalingFactor const* scale_a_ptr = nullptr,
           ElementScalingFactor const* scale_b_ptr = nullptr,
           ElementScalingFactor const* scale_c_ptr = nullptr,
           ElementScalingFactor const* scale_d_ptr = nullptr)
        : alpha(alpha),
          beta(beta),
          scale_a_ptr(scale_a_ptr),
          scale_b_ptr(scale_b_ptr),
          scale_c_ptr(scale_c_ptr),
          scale_d_ptr(scale_d_ptr){}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr,
           ElementCompute const* beta_ptr,
           ElementScalingFactor const* scale_a_ptr = nullptr,
           ElementScalingFactor const* scale_b_ptr = nullptr,
           ElementScalingFactor const* scale_c_ptr = nullptr,
           ElementScalingFactor const* scale_d_ptr = nullptr)
        : alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr),
          scale_a_ptr(scale_a_ptr),
          scale_b_ptr(scale_b_ptr),
          scale_c_ptr(scale_c_ptr),
          scale_d_ptr(scale_d_ptr){}
  };

private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

  ElementCompute scale_d_;
  ElementCompute abs_max_output_local_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationScalingFactor(Params const& params)
      : alpha_(params.alpha_ptr ? *params.alpha_ptr : params.alpha),
        beta_(params.beta_ptr ? *params.beta_ptr : params.beta),
        abs_max_output_local_(ElementCompute(0)),
        scale_d_(ElementCompute(params.scale_d_ptr ? *(params.scale_d_ptr) : ElementScalingFactor(1))) {

    auto scale_a =
        ElementCompute(params.scale_a_ptr ? *(params.scale_a_ptr) : ElementScalingFactor(1));
    auto scale_b =
        ElementCompute(params.scale_b_ptr ? *(params.scale_b_ptr) : ElementScalingFactor(1));
    auto scale_c =
        ElementCompute(params.scale_c_ptr ? *(params.scale_c_ptr) : ElementScalingFactor(1));

    multiplies<ElementCompute> multiply;
    alpha_ = multiply(alpha_, multiply(scale_a, scale_b));
    beta_ = multiply(beta_, scale_c);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if constexpr (Scale == ScaleType::NoBetaScaling) return true;

    if constexpr (Scale == ScaleType::OnlyAlphaScaling) return false;

    if constexpr (Scale == ScaleType::Nothing) return false;

    return beta_ != ElementCompute(0);
  }

  /// Computes linear scaling with source: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentSource const &source) {

    NumericArrayConverter<ElementCompute, ElementC, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    multiplies<FragmentCompute> multiply;
    multiply_add<FragmentCompute> mul_add_accumulator;

    if constexpr (Scale == ScaleType::Nothing) {
      if constexpr (IsScalingAndAmaxOutputNeeded) {
        // abs_max_output = max( abs(d) | (for every d in D))
        abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
            abs_max_output_local_,
            absolute_value_op<FragmentCompute>{}(converted_accumulator));
        // D = scale_d * D
        converted_accumulator = multiply(scale_d_, converted_accumulator);
      }
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    FragmentCompute intermediate;

    if constexpr (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
    }
    else {
      // X =  beta * C + uniform
      intermediate = multiply(beta_, converted_source);
    }

    // D = alpha * Accum + X
    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);

    if constexpr (IsScalingAndAmaxOutputNeeded) {
      // abs_max_output = max( abs(d) | (for every d in D))
      abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
        abs_max_output_local_,
        absolute_value_op<FragmentCompute>{}(intermediate));
      // D = scale_d * D
      intermediate = multiply(scale_d_, intermediate);
    }

    return destination_converter(intermediate);
  }

  /// Computes linear scaling with source: D = alpha * accumulator + beta * source + bias
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentSource const &source, FragmentBias const &bias) {
    NumericArrayConverter<ElementCompute, ElementC, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
    NumericArrayConverter<ElementCompute, ElementBias, kCount, Round> bias_converter;

    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    BinaryOp binary_op;

    multiplies<FragmentCompute> multiply;
    multiply_add<FragmentCompute> mul_add_accumulator;
    FragmentActivationFunctor activation;

    if constexpr (Scale == ScaleType::Nothing) {
      FragmentCompute converted_bias = bias_converter(bias);
      converted_accumulator = binary_op(converted_accumulator, converted_bias);
      converted_accumulator = activation(converted_accumulator);

      if constexpr (IsScalingAndAmaxOutputNeeded) {
        // abs_max_output = max( abs(d) | (for every d in D))
        abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
            abs_max_output_local_,
            absolute_value_op<FragmentCompute>{}(converted_accumulator));
        // D = scale_d * D
        converted_accumulator = multiply(scale_d_, converted_accumulator);                           
      }
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    FragmentCompute intermediate;

    if constexpr (Scale == ScaleType::NoBetaScaling) {
      intermediate = converted_source;
    }
    else {
      // X =  beta * C + uniform
      intermediate = multiply(beta_, converted_source);                             
    }

    // D = alpha * Accum + X
    intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);
    FragmentCompute converted_bias = bias_converter(bias);
    //intermediate += converted_bias;
    intermediate = binary_op(intermediate, converted_bias);
    // fused activation function.
    intermediate = activation(intermediate);

    if constexpr (IsScalingAndAmaxOutputNeeded) {
      // abs_max_output = max( abs(d) | (for every d in D))
      abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
        abs_max_output_local_,
        absolute_value_op<FragmentCompute>{}(intermediate));
      // D = scale_d * D
      intermediate = multiply(scale_d_, intermediate);
    }

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    multiplies<FragmentCompute> multiply;

    if constexpr (Scale == ScaleType::Nothing) {
      if constexpr (IsScalingAndAmaxOutputNeeded) {
        // abs_max_output = max( abs(d) | (for every d in D))
        abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
            abs_max_output_local_,
            absolute_value_op<FragmentCompute>{}(converted_accumulator));
        // D = scale_d * D
        converted_accumulator = multiplies<FragmentCompute>{}(scale_d_, converted_accumulator);
      }
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    FragmentCompute intermediate;

    intermediate = multiply(alpha_, converted_accumulator);    // D = alpha * Accum

    if constexpr (IsScalingAndAmaxOutputNeeded) {
      // abs_max_output = max( abs(d) | (for every d in D))
      abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
        abs_max_output_local_,
        absolute_value_op<FragmentCompute>{}(intermediate));
      // D = scale_d * D
      intermediate = multiply(scale_d_, intermediate);
    }

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator + bias
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentBias const &bias, const bool &is_bias = true) {

    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
    NumericArrayConverter<ElementCompute, ElementBias, kCount, Round> bias_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    multiplies<FragmentCompute> multiply;
    FragmentActivationFunctor activation;
    BinaryOp binary_op;

    if constexpr (Scale == ScaleType::Nothing) {
      FragmentCompute converted_bias = bias_converter(bias);
      //converted_accumulator += converted_bias;
      converted_accumulator = binary_op(converted_accumulator, converted_bias);
      // fused activation function.
      converted_accumulator = activation(converted_accumulator);
      if constexpr (IsScalingAndAmaxOutputNeeded) {
        // abs_max_output = max( abs(d) | (for every d in D))
        abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
            abs_max_output_local_,
            absolute_value_op<FragmentCompute>{}(converted_accumulator));
        // D = scale_d * D
        converted_accumulator = multiplies<FragmentCompute>{}(scale_d_, converted_accumulator);
      }
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    FragmentCompute intermediate;

    intermediate = multiply(alpha_, converted_accumulator);    // D = alpha * Accum
    FragmentCompute converted_bias = bias_converter(bias);
    //intermediate += converted_bias;
    intermediate = binary_op(intermediate, converted_bias);
    // fused activation function.
    intermediate = activation(intermediate);

    if constexpr (IsScalingAndAmaxOutputNeeded) {
      // abs_max_output = max( abs(d) | (for every d in D))
      abs_max_output_local_ = maximum_with_nan_propogation<FragmentCompute>{}(
        abs_max_output_local_,
        absolute_value_op<FragmentCompute>{}(intermediate));
      // D = scale_d * D
      intermediate = multiply(scale_d_, intermediate);                                               
    }

    return destination_converter(intermediate);
  }

  //
  // Specializations for scalar
  //
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementC const source, ElementBias const bias) {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementCompute, ElementBias, Round> bias_converter;
    [[maybe_unused]] NumericConverter<ElementCompute, ElementC, Round> source_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;

    // Convert to destination numeric type

    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    multiplies<ElementCompute> multiply;
    ElementActivationFunctor activation;

    if constexpr (Scale == ScaleType::Nothing) {
      ElementCompute converted_bias = bias_converter(bias);
      converted_accumulator += converted_bias;
      // fused activation function.
      converted_accumulator = activation(converted_accumulator);

      if constexpr (IsScalingAndAmaxOutputNeeded) {
        // abs_max_output = max( abs(d) | (for every d in D))
        abs_max_output_local_ = maximum_with_nan_propogation<ElementCompute>{}(
            abs_max_output_local_,
            absolute_value_op<ElementCompute>{}(converted_accumulator)); 

        converted_accumulator = multiply(scale_d_, converted_accumulator);                           // D = scale_d * D
      }
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;
    multiply_add<ElementCompute> madd;

    if constexpr (Scale == ScaleType::NoBetaScaling) {
      intermediate = source_converter(source);
    }
    else {
      intermediate = multiply(beta_, source);                                                // X =  beta * C + uniform
    }

    intermediate = madd(alpha_, converted_accumulator, intermediate);                          // D = alpha * Accum + X
    ElementCompute converted_bias = bias_converter(bias);
    intermediate += converted_bias;
    // fused activation function.
    intermediate = activation(intermediate);

    if constexpr (IsScalingAndAmaxOutputNeeded) {
      // abs_max_output = max( abs(d) | (for every d in D))
      abs_max_output_local_ = maximum_with_nan_propogation<ElementCompute>{}(
          abs_max_output_local_,
          absolute_value_op<ElementCompute>{}(intermediate));

      intermediate = multiply(scale_d_, intermediate);                                               // D = scale_d * D
    }

    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementBias const bias) {
    // Convert everything to Compute type, do compute, and then store to output type
    NumericConverter<ElementCompute, ElementAccumulator, Round> accumulator_converter;
    NumericConverter<ElementCompute, ElementBias, Round> bias_converter;
    NumericConverter<ElementD, ElementCompute, Round> destination_converter;
    ElementCompute converted_accumulator = accumulator_converter(accumulator);
    [[maybe_unused]] multiplies<ElementCompute> multiply;
    ElementActivationFunctor activation;

    // Convert to destination numeric type
    if constexpr (Scale == ScaleType::Nothing) {
      ElementCompute converted_bias = bias_converter(bias);
      converted_accumulator += converted_bias;
      // fused activation function.
      converted_accumulator = activation(converted_accumulator);

      if constexpr (IsScalingAndAmaxOutputNeeded) {
        // abs_max_output = max( abs(d) | (for every d in D))
        abs_max_output_local_ = maximum_with_nan_propogation<ElementCompute>{}(
            abs_max_output_local_,
            absolute_value_op<ElementCompute>{}(converted_accumulator));

        converted_accumulator = multiply(scale_d_, converted_accumulator);                           // D = scale_d * D
      }
      return destination_converter(converted_accumulator);
    }

    // Perform binary operations
    ElementCompute intermediate;

    ElementCompute converted_bias = bias_converter(bias);
    intermediate = multiply_add<ElementCompute>{}(alpha_, converted_accumulator, converted_bias);
    // fused activation function.
    intermediate = activation(intermediate);

    if constexpr (IsScalingAndAmaxOutputNeeded) {
      
      abs_max_output_local_ = maximum_with_nan_propogation<ElementCompute>{}(
          abs_max_output_local_,
          absolute_value_op<ElementCompute>{}(intermediate));

      intermediate = multiply(scale_d_, intermediate);                                               // D = scale_d * D
    }

    return destination_converter(intermediate);
  }

  CUTLASS_HOST_DEVICE
  ElementAccumulator get_output_abs_max() {
    return NumericConverter<ElementAccumulator, ElementCompute, Round>{}(abs_max_output_local_);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
