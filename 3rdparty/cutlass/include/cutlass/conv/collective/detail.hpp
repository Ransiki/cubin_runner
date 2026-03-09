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

#include "cutlass/conv/convnd_problem_shape.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}
// TODO: instead of taking the whole dispatch policy as inputs here, take Major:: enum class instead
// TODO: can simplify this even more and have the same type for flt/act if we just make them fully dynamic strides 
// {$nv-internal-release end}

// Construct the stride types for conv collectives based on the dispatch policy, strides 64b by default
template <class DispatchPolicy>
constexpr auto
sm90_dispatch_policy_to_stride_A() {
  if constexpr (DispatchPolicy::ConvOp == conv::Operator::kFprop) {
    // Maps to modes ((w,n), C)
    if constexpr (DispatchPolicy::NumSpatialDimensions == 1) {
      return cute::Stride<cute::Stride<int64_t, int64_t>,
                          cute::Int<1>>{};
    }
    // Maps to modes ((w,h,n), C)
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 2) {
      return cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,
                          cute::Int<1>>{};
    }
    // Maps to modes ((w,h,d,n), C)
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 3) {
      return cute::Stride<cute::Stride<int64_t, int64_t, int64_t, int64_t>,
                          cute::Int<1>>{};
    }
    // error dims assert
    else {
      static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported spatial dim count.");
    }
  }
  else if constexpr (DispatchPolicy::ConvOp == conv::Operator::kWgrad) {
    // Maps to modes (k, nq/npq/nzpq)
    if constexpr (DispatchPolicy::NumSpatialDimensions == 1 ||
                  DispatchPolicy::NumSpatialDimensions == 2 ||
                  DispatchPolicy::NumSpatialDimensions == 3) {
      return cute::Stride<cute::Int<1>, int64_t>{};
    }
    // error dims assert
    else {
      static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported spatial dim count.");
    }
  }
  else if constexpr (DispatchPolicy::ConvOp == conv::Operator::kDgrad) {
    // Maps to modes ((q,n), K)
    if constexpr (DispatchPolicy::NumSpatialDimensions == 1) {
      return cute::Stride<cute::Stride<int64_t, int64_t>,
                          cute::Int<1>>{};
    }
    // Maps to modes ((q,p,n), K)
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 2) {
      return cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,
                          cute::Int<1>>{};
    }
    // Maps to modes ((q,p,z,n), K)
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 3) {
      return cute::Stride<cute::Stride<int64_t, int64_t, int64_t, int64_t>,
                          cute::Int<1>>{};
    }
    // error dims assert
    else {
      static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported spatial dim count.");
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported ConvOp.");
  }
}

// Construct the stirde types for conv collectives based on the dispatch policy, strides 64b by default
template <class DispatchPolicy>
constexpr auto
sm90_dispatch_policy_to_stride_B() {
  if constexpr (DispatchPolicy::ConvOp == conv::Operator::kFprop) {
    // Maps to modes (k, (C,s))
    if constexpr      (DispatchPolicy::NumSpatialDimensions == 1) {
      return cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t>>{};
    }
    // Maps to modes (k, (C,s,r))
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 2) {
      return cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t>>{};
    }
    // Maps to modes (k, (C,s,r,t))
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 3) {
      return cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t, int64_t>>{};
    }
    // error dims assert
    else {
      static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported spatial dim count.");
    }
  }
  else if constexpr (DispatchPolicy::ConvOp == conv::Operator::kWgrad) {
    // Maps to modes (C, (w,n))
    if constexpr (DispatchPolicy::NumSpatialDimensions == 1) {
      return cute::Stride<cute::Int<1>,
                          cute::Stride<int64_t, int64_t>>{};
    }
    // Maps to modes (C, (w,h,n))
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 2) {
      return cute::Stride<cute::Int<1>,
                          cute::Stride<int64_t, int64_t, int64_t>>{};
    }
    // Maps to modes (C, (w,h,d,n))
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 3) {
      return cute::Stride<cute::Int<1>,
                          cute::Stride<int64_t, int64_t, int64_t, int64_t>>{};
    }
    // error dims assert
    else {
      static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported spatial dim count.");
    }
  }
  else if constexpr (DispatchPolicy::ConvOp == conv::Operator::kDgrad) {
    // Maps to modes (C, (k,s))
    if constexpr      (DispatchPolicy::NumSpatialDimensions == 1) {
      return cute::Stride<cute::Int<1>, cute::Stride<int64_t, int64_t>>{};
    }
    // Maps to modes (C, (k,s,r))
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 2) {
      return cute::Stride<cute::Int<1>, cute::Stride<int64_t, int64_t, int64_t>>{};
    }
    // Maps to modes (C, (k,s,r,t))
    else if constexpr (DispatchPolicy::NumSpatialDimensions == 3) {
      return cute::Stride<cute::Int<1>, cute::Stride<int64_t, int64_t, int64_t, int64_t>>{};
    }
    // error dims assert
    else {
      static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported spatial dim count.");
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Unsupported ConvOp.");
  }
}


template <class DispatchPolicy>
constexpr auto
sm100_dispatch_policy_to_stride_A() {
  return sm90_dispatch_policy_to_stride_A<DispatchPolicy>();
}

template <class DispatchPolicy>
constexpr auto
sm100_dispatch_policy_to_stride_B() {
  return sm90_dispatch_policy_to_stride_B<DispatchPolicy>();
}


/////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the lower/near corner, returning it as a cute::array in [W,H,D] order
template <conv::Operator ConvOp, int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_lower_corner_whd(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  using cute::for_each;
  using cute::make_seq;

  cute::array<int, NumSpatialDimensions> lower{};
  if constexpr (ConvOp == conv::Operator::kFprop ||
                ConvOp == conv::Operator::kWgrad) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions-1-i] = -1 * problem_shape.lower_padding[i];
    });
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions-1-i] = problem_shape.lower_padding[i] -
        (problem_shape.shape_B[i+1] - 1) * problem_shape.dilation[i];
    });
  }
  return lower;
}

// Computes the upper/far corner, returning it as a cute::array in [W,H,D] order
template <conv::Operator ConvOp, int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_upper_corner_whd(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  using cute::for_each;
  using cute::make_seq;

  cute::array<int, NumSpatialDimensions> upper{};
  if constexpr (ConvOp == conv::Operator::kFprop) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      upper[NumSpatialDimensions-1-i] = problem_shape.upper_padding[i] -
        (problem_shape.shape_B[i+1] - 1) * problem_shape.dilation[i];
    });
  }
  else if constexpr (ConvOp == conv::Operator::kWgrad) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      upper[NumSpatialDimensions-1-i] = problem_shape.upper_padding[i] -
        (problem_shape.shape_C[i+1] - 1) * problem_shape.dilation[i];
    });
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      upper[NumSpatialDimensions-1-i] = problem_shape.lower_padding[i] -
        (problem_shape.shape_B[i+1] - 1) * problem_shape.dilation[i] + problem_shape.shape_C[i+1] - problem_shape.shape_A[i+1];
    });
  }
  return upper;
}

// Compute the lower/near corner of (t,r,s), returning it as a cute::array in [S,R,T] order
template <conv::Operator ConvOp, int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_lower_srt(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  using cute::for_each;
  using cute::make_seq;

  cute::array<int, NumSpatialDimensions> lower{};
  if constexpr (ConvOp == conv::Operator::kFprop ||
                ConvOp == conv::Operator::kWgrad) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions-1-i] = 0;
    });
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    for_each(make_seq<NumSpatialDimensions>{}, [&](auto i) {
      lower[NumSpatialDimensions-1-i] = (problem_shape.shape_B[i+1] - 1) * problem_shape.dilation[i];
    });
  }
  return lower;
}

// {$nv-internal-release begin}

// Compute the base/far corner for W mode, returning it as a cute::tuple in [BaseW, FarW] order
template <conv::Operator ConvOp, int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr auto
compute_corner_w(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  using cute::get;
  using cute::make_shape;
  using cute::reverse;
  using cute::take;

  // The relative position of the left/right pixel
  // Pixel Range: [0, Padding[0] + W + Padding[1])
  int32_t left = 0;
  int32_t right = 0;
  int32_t upper_bound = 0;
  // skip pixels as we don't need to load them.
  int32_t skip_left = 0;
  int32_t skip_right = 0;
  // placeholder (w_Offset)
  int32_t offset = 0;
  cute::array<int32_t, 1> lower_w{};
  cute::array<int32_t, 1> upper_w{};

  const int32_t stride_w = problem_shape.traversal_stride[2];

  cute::array<int32_t, 2> padding{};

  // For nq tiling, dilation is always 1 for now.
  padding[0] = -1 * problem_shape.lower_padding[NumSpatialDimensions-1] +
    (problem_shape.shape_B[NumSpatialDimensions] - 1);                                         // lower padding w
  padding[1] = problem_shape.lower_padding[NumSpatialDimensions-1] +
    problem_shape.shape_C[NumSpatialDimensions] - problem_shape.shape_A[NumSpatialDimensions]; // upper padding w

  auto shape_Act_orig = make_shape(reverse(take<0,NumSpatialDimensions+1>(problem_shape.shape_A)),
                                           problem_shape.shape_A[NumSpatialDimensions+1]);
  const int32_t shape_Act_w = get<0>(get<0>(shape_Act_orig));

  if ((stride_w == 1) || (stride_w == 2)) {
    // Skip loading 1 pixels when stride2x2
    //              2 pixels when stride1x1
    const int32_t skip_pixel =  stride_w == 1 ? 2 :
                                stride_w == 2 ? 1 :
                                0;
    skip_left = cute::min(padding[0], skip_pixel);
    skip_right = skip_pixel - skip_left;
    // Considering "general" batch case rather than 0th batch
    // offset = 1 if pixels in buffer1 is in front of the pixels in buffer0
    //            | p0b1 | p0b0 | p1b1 | p1b1 | ..
    // offset = 0 if pixels in buffer1 is behind the pixels in buffer0
    //            | p0b0 | p0b1 | p1b0 | p1b1 | ..
    offset = (skip_left == 1) && (stride_w != 1);
    // Relative position
    // Data Range | --------- | --------- | ---------- |
    //            | skip_left | load_data | skip_right |
    //            0         left        right       upper_bound
    upper_bound = (padding[0] + shape_Act_w + padding[1] - 1) / stride_w * stride_w;
    left = skip_left * stride_w;
    right = upper_bound - skip_right * stride_w;
    // Relative position => Absolute position
    lower_w[0] = (left - padding[0]) - offset;
    upper_w[0] = (right - padding[0] - shape_Act_w + 1) - offset;
    // make sure all original pixels are loaded
    assert((upper_w[0] + offset >= (stride_w == 2 ? -1 : 0)) && "Padding shape not support");
  }
  else {
    assert("Stride shape not supported");
  }
  return cute::make_tuple(lower_w, upper_w);
}

// nq tiled 2d kernel's padding limit due to czm.
template <conv::Operator ConvOp, int NumSpatialDimensions>
CUTLASS_HOST_DEVICE
constexpr bool
corner_w_can_implement(ConvProblemShape<ConvOp, NumSpatialDimensions> const& problem_shape) {
  using cute::get;
  using cute::make_shape;
  using cute::reverse;
  using cute::take;

  const int32_t stride_w = problem_shape.traversal_stride[2];
  cute::array<int32_t, 2> padding{};

  // For nq tiling, dilation is always 1 for now.
  padding[0] = -1 * problem_shape.lower_padding[NumSpatialDimensions-1] +
    (problem_shape.shape_B[NumSpatialDimensions] - 1);                                         // lower padding w
  padding[1] = problem_shape.lower_padding[NumSpatialDimensions-1] +
    problem_shape.shape_C[NumSpatialDimensions] - problem_shape.shape_A[NumSpatialDimensions]; // upper padding w

  auto shape_Act_orig = make_shape(reverse(take<0,NumSpatialDimensions+1>(problem_shape.shape_A)),
                                           problem_shape.shape_A[NumSpatialDimensions+1]);
  const int32_t shape_Act_w = get<0>(get<0>(shape_Act_orig));

  if ((stride_w == 1) || (stride_w == 2)) {
    // Skip loading 1 pixels when stride2x2
    //              2 pixels when stride1x1
    const int32_t skip_pixel =  stride_w == 1 ? 2 :
                                stride_w == 2 ? 1 :
                                0;
    int32_t skip_left = cute::min(padding[0], skip_pixel);
    int32_t skip_right = skip_pixel - skip_left;
    // Considering "general" batch case rather than 0th batch
    // offset = 1 if pixels in buffer1 is in front of the pixels in buffer0
    //            | p0b1 | p0b0 | p1b1 | p1b1 | ..
    // offset = 0 if pixels in buffer1 is behind the pixels in buffer0
    //            | p0b0 | p0b1 | p1b0 | p1b1 | ..
    int32_t offset = (skip_left == 1) && (stride_w != 1);
    // Relative position
    // Data Range | --------- | --------- | ---------- |
    //            | skip_left | load_data | skip_right |
    //            0         left        right       upper_bound
    int32_t upper_bound = (padding[0] + shape_Act_w + padding[1] - 1) / stride_w * stride_w;
    int32_t right = upper_bound - skip_right * stride_w;
    // Relative position => Absolute position
    int32_t upper_w = (right - padding[0] - shape_Act_w + 1) - offset;
    // make sure all original pixels are loaded
    return (upper_w + offset >= (stride_w == 2 ? -1 : 0));
  }
  else {
    return false;
  }
}
// {$nv-internal-release end}

template <class CopyOp> struct is_im2col_load { static constexpr bool value = false; };
template <> struct is_im2col_load<cute::SM90_TMA_LOAD_IM2COL          > { static constexpr bool value = true; };
template <> struct is_im2col_load<cute::SM90_TMA_LOAD_IM2COL_MULTICAST> { static constexpr bool value = true; };
template <> struct is_im2col_load<cute::SM100_TMA_2SM_LOAD_IM2COL          > { static constexpr bool value = true; }; 
template <> struct is_im2col_load<cute::SM100_TMA_2SM_LOAD_IM2COL_MULTICAST> { static constexpr bool value = true; }; 

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective::detail
