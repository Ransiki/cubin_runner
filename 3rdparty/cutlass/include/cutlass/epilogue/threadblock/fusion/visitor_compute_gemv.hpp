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

/*! \file
  \brief Visitor tree compute operations for the CUTLASS 2x epilogue Gemv
*/

#pragma once

#include "cutlass/epilogue/threadblock/fusion/visitor_2x_gemv.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// N-nary Elementwise Compute Operation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  template <class> class ComputeFn_,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest,
  class = void
>
struct VisitorComputeGemv : VisitorImpl2xGemv<>  {

  using VisitorImpl2xGemv<>::VisitorImpl2xGemv;

  using ComputeFn = ComputeFn_<ElementCompute>;

  struct Callbacks : EmptyCallbacksGemv {

    template <class ElementAccumulator, typename... ElementInputs>
    CUTLASS_DEVICE ElementOutput
    visit(int idx_batch, int idx_row_m, ElementAccumulator accum, ElementInputs const& ... inputs) {
      return transform_apply(cute::make_tuple(inputs...),
        [&] (auto&& input) {
          using ElementInput = typename cute::remove_cvref_t<decltype(input)>;
          using ConvertInput = NumericConverter<ElementCompute, ElementInput, RoundStyle>;
          ConvertInput convert_input{};
          return convert_input(input);
        },
        [&] (auto&&... cvt_inputs) {
          using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
          ComputeFn compute_output{};
          ConvertOutput convert_output{};

          return convert_output(compute_output(cvt_inputs...));
        }
      );
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    return Callbacks{};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////