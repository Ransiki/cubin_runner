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
//
// {$nv-internal-release file}
//
/*! \file
    \brief Reference implementation for GETT with input reduction.
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"

#include "cute/tensor.hpp"
#include "gett.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementAccumulator_,
  class TensorA_,                                                                                         // (M, K, L)
  class TensorB_                                                                                          // (N, K, L)
>
struct GettInputReductionMainloopParams {
  using ElementAccumulator = ElementAccumulator_;
  using TensorA = TensorA_;
  using TensorB = TensorB_;
  using EngineA = typename TensorA::engine_type;
  using LayoutA = typename TensorA::layout_type;
  using EngineB = typename TensorB::engine_type;
  using LayoutB = typename TensorB::layout_type;

  TensorA A{};
  TensorB B{};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementScalar_,
  class ElementAccumulator_,
  class ElementCompute_,
  class TensorC_,                                                                                          // (M, N, L)
  class TensorD_,                                                                                          // (M, N, L)
  class VectorReductionA_ = TensorD_
>
struct GettInputReductionEpilogueParams {
  using ElementScalar = ElementScalar_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using TensorC = TensorC_;
  using TensorD = TensorD_;
  using VectorReductionA = VectorReductionA_;

  using EngineC = typename TensorC::engine_type;
  using LayoutC = typename TensorC::layout_type;
  using EngineD =  typename TensorD::engine_type;
  using LayoutD = typename TensorD::layout_type;

  ElementScalar alpha = ElementScalar(1);
  ElementScalar beta = ElementScalar(0);

  TensorC C{};
  TensorD D{};
  VectorReductionA VReductionA{};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Mainloop with extra input reduction
template <class MainloopParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_input_reduction_mainloop(
    MainloopParams const& mainloop_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN],
    ElementAccumulator (&acc_redA)[kBlockM][1])
{

  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == 3, "M, K, B");
  static_assert(cute::rank(typename MainloopParams::LayoutB{}) == 3, "N, K, B");

  using ElementA = typename ElementTraits<typename MainloopParams::EngineA::value_type>::type;
  using ElementB = typename ElementTraits<typename MainloopParams::EngineB::value_type>::type;

  using RingOp = multiply_add<ElementAccumulator, ElementAccumulator, ElementAccumulator>;
  RingOp fma_op;

  // Zero out accumulators
  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    acc_redA[m_b][0] = ElementAccumulator(0); // For BgradA acc
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      acc[m_b][n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity 
    }
  }

  // Compute on this k-block
  for (int64_t k = 0; k < cute::size<1>(mainloop_params.A.layout()); ++k) {
    // Load A
    ElementAccumulator a_frag[kBlockM];
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      if (m + m_b < cute::size<0>(mainloop_params.A.layout())) {
        // Perform reference GEMM calculations at the accumulator's precision. Cast A value to accumulator type.
        a_frag[m_b] = static_cast<ElementAccumulator>(ElementA(mainloop_params.A(m + m_b, k, l)));
      } else {
        a_frag[m_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // Load B
    ElementAccumulator b_frag[kBlockN];
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (n + n_b < cute::size<0>(mainloop_params.B.layout())) {
        // Perform reference GEMM calculations at the accumulator's precision. Cast A value to accumulator type.
        b_frag[n_b] = static_cast<ElementAccumulator>(ElementB(mainloop_params.B(n + n_b, k, l)));
      } else {
        b_frag[n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // do compute
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      for (int n_b = 0; n_b < kBlockN; ++n_b) {
        acc[m_b][n_b] = fma_op(a_frag[m_b], b_frag[n_b], acc[m_b][n_b]);
        if(n_b == 0){
          acc_redA[m_b][0] = fma_op(a_frag[m_b], ElementAccumulator(1), acc_redA[m_b][0]);
        }
        
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Epilogue with extra input reduction
template <class EpilogueParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_input_reduction_epilogue(
    EpilogueParams const& epilogue_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN],
    ElementAccumulator (&acc_redA)[kBlockM][1])
{
  static_assert(cute::rank(typename EpilogueParams::LayoutC{}) == 3, "M, K, B");
  static_assert(cute::rank(typename EpilogueParams::LayoutD{}) == 3, "N, K, B");

  using ElementCompute = typename EpilogueParams::ElementCompute;
  using ElementC = typename EpilogueParams::TensorC::value_type;
  using ElementD = typename EpilogueParams::TensorD::value_type;
  using ElementReductionA = typename EpilogueParams::VectorReductionA::value_type;
  using ElementScalar = typename EpilogueParams::ElementScalar;
  // Input related converter
  NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
  NumericConverter<ElementCompute, ElementC> source_converter;

  // Scale related converter
  NumericConverter<ElementCompute, ElementScalar> scale_converter;

  // Output related converter
  NumericConverter<ElementD, ElementCompute> destination_converter;
  NumericConverter<ElementReductionA, ElementAccumulator> ReductionA_converter;

  // Epilogue operations
  multiply_add<ElementCompute, ElementCompute, ElementCompute> epilogue_fma;
  multiplies<ElementCompute> mul;

  // Do conversion
  ElementCompute converted_alpha = scale_converter(epilogue_params.alpha);
  ElementCompute converted_beta = scale_converter(epilogue_params.beta);

  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n + n_b < cute::size<1>(epilogue_params.D.layout())) {
        // Convert every type to ElementCompute first, do compute, convert to output type, write it out
        ElementCompute converted_acc = accumulator_converter(acc[m_b][n_b]);
        ElementCompute output = mul(converted_alpha, converted_acc);

        // using cute::raw_pointer_cast;
        //if (raw_pointer_cast(epilogue_params.C.data())) {
        //  ElementCompute converted_src = source_converter(epilogue_params.C(m + m_b, n + n_b, l));
        //  output = epilogue_fma(converted_beta, converted_src, output);
        //}

        epilogue_params.D(m + m_b, n + n_b, l) = destination_converter(output);
      }
    } // n_b

    if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n < cute::size<1>(epilogue_params.D.layout())) {
      ElementReductionA converted_reductionA = ReductionA_converter(acc_redA[m_b][0]);
      epilogue_params.VReductionA(m + m_b) = converted_reductionA;
    }
  } // m_b
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - General Tensor-Tensor contraction reference kernel
template <
  class MainloopParams,
  class EpilogueParams
>
void GemmWithInputReduction3x(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{

  static_assert(rank(typename MainloopParams::LayoutA{}) == 3, "M, K, B");
  static int constexpr kBlockM = 64;
  static int constexpr kBlockN = 64;

#if defined(_OPENMP)
  #pragma omp parallel for collapse(3)
#endif
  for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
    for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
      for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
        // General acc group to store the A*B
        typename MainloopParams::ElementAccumulator acc[kBlockM][kBlockN];
        // We add extra acc group to store A * constant1, which model the reduction on A
        typename MainloopParams::ElementAccumulator acc_redA[kBlockM][1];
        gett_input_reduction_mainloop(mainloop_params, m, n, l, acc, acc_redA);
        gett_input_reduction_epilogue(epilogue_params, m, n, l, acc, acc_redA);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
