/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Template for a Convolution Small Channel Fprop kernel.
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/mma.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/jetfire.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/conv/small_channel_convolution_utils.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {
namespace small_channel_fprop {

// Return WarpsPerCTA
template<typename Conv2dFilterShape>
constexpr int
get_warps_per_CTA() {
  return (Conv2dFilterShape::kH + 1) / 2;
}

// Return FltKPerCTA
template<typename Conv2dFilterShape>
constexpr int
get_fltK_per_CTA() {
  constexpr int kFltK = Conv2dFilterShape::kN;
  constexpr int kFltR = Conv2dFilterShape::kH;
  constexpr int kFltS = Conv2dFilterShape::kW;
  constexpr int kFltC = Conv2dFilterShape::kC;
  constexpr int kFltKPerCTA = Conv2dFilterShape::kN;
  if constexpr (kFltC == 8) {
    if constexpr (kFltR == 7 && kFltS == 7) {
      if constexpr (kFltK == 32) {
        return kFltKPerCTA / 2;
      } else if constexpr (kFltK == 64) {
        return kFltKPerCTA / 4;
      }
    } else if constexpr (kFltR == 5 && kFltS == 5) {
      if constexpr (kFltK == 64) {
        return kFltKPerCTA / 2;
      }
    }
  }
  return kFltKPerCTA;
}

// Return MinBlocksPerMultiprocessor
template<typename Conv2dFilterShape>
constexpr int
get_min_blocks_per_multiprocessor() {
  constexpr int kFltK = Conv2dFilterShape::kN;
  constexpr int kFltR = Conv2dFilterShape::kH;
  constexpr int kFltS = Conv2dFilterShape::kW;
  constexpr int kFltC = Conv2dFilterShape::kC;
  constexpr int MinBlocksPerMultiprocessor = 1;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 750)) // Sm80/Sm90/Sm100
  if constexpr (kFltK == 32 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
    return 4;
  }
  if constexpr (kFltK == 64 && kFltR == 7 && kFltS == 7 && kFltC == 4) {
    return 2;
  }
  if constexpr (kFltK == 64 && kFltR == 5 && kFltS == 5 && kFltC == 4) {
    return 3;
  }
  if constexpr (kFltK == 64 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
    return 6;
  }
#else // Sm75
  if constexpr (kFltK == 16 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
    return 13;
  }
  if constexpr (kFltK == 64 && kFltR == 5 && kFltS == 5 && kFltC == 4) {
    return 2;
  }
#endif
  return MinBlocksPerMultiprocessor;
}

// Return buf size of smem_flt
template<typename Conv2dFilterShape>
constexpr int
get_smem_flt_size() {
  constexpr int kFltKPerCTA = get_fltK_per_CTA<Conv2dFilterShape>();
  constexpr int kFltR = Conv2dFilterShape::kH;
  constexpr int kFltS = Conv2dFilterShape::kW;
  constexpr int kFltC = Conv2dFilterShape::kC;
  constexpr int kEvenFltS = (kFltS % 2 == 0) ? kFltS : kFltS + 1;
  constexpr int kEvenFltRS = kFltR * kEvenFltS;
  constexpr int kEvenFltKRS = kFltKPerCTA * kEvenFltRS;
  return kEvenFltKRS * kFltC;
}

// Return buf size of smem_img
template<typename Conv2dFilterShape>
constexpr int
get_smem_img_size() {
  constexpr int kFltS = Conv2dFilterShape::kW;
  constexpr int kFltC = Conv2dFilterShape::kC;
  constexpr int kPadW = (kFltS - 1) / 2;
  constexpr int kOutP = get_warps_per_CTA<Conv2dFilterShape>();
  constexpr int kOutQ = 64;
  constexpr int kImgH = 2 * kOutP;
  constexpr int kImgW = 2 * kOutQ + 2 * kPadW;
  constexpr int kImgHW = kImgH * kImgW;
  return kImgHW * kFltC;
}

} // end of namespace small_channel_fprop
} // end of namespace detail

template <
  typename ArchTag_,
  typename Conv2dFilterShape_,
  bool Swish_ = false
>
struct SmallChannelConv2dFprop {

  using ArchTag = ArchTag_;
  using Conv2dFilterShape = Conv2dFilterShape_;

  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = cutlass::half_t;

  using LayoutA = cutlass::layout::TensorNHWC;
  using LayoutB = cutlass::layout::TensorNHWC;
  using LayoutC = cutlass::layout::TensorNHWC;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Conv2dStride = cutlass::conv::Stride2D<2, 2>;

  using SmemLayoutA = cutlass::layout::RowMajor;
  using SmemLayoutB = cutlass::layout::ColumnMajor;
  using SmemLayoutC = cutlass::layout::RowMajor;

  // Matrix multiply operator (concept: arch::Mma)
  using Mma = cutlass::arch::Mma<InstructionShape,
                                 32,
                                 ElementA,
                                 SmemLayoutA,
                                 ElementB,
                                 SmemLayoutB,
                                 ElementAccumulator,
                                 SmemLayoutC,
                                 cutlass::arch::OpMultiplyAdd>;

  using AccessType = int2;
  using LdsmOperandA = Array<unsigned, 2>;
  using LdsmOperandB = Array<unsigned, 1>;

  using MmaOperandA = typename Mma::FragmentA;
  using MmaOperandB = typename Mma::FragmentB;
  using MmaOperandC = typename Mma::FragmentC;

  static constexpr Operator kConvolutionalOperator = conv::Operator::kFprop;

  static constexpr uint32_t WarpsPerCTA = detail::small_channel_fprop::get_warps_per_CTA<Conv2dFilterShape>();
  static constexpr uint32_t FltKPerCTA = detail::small_channel_fprop::get_fltK_per_CTA<Conv2dFilterShape>();
  static constexpr uint32_t MaxThreadsPerBlock = WarpsPerCTA * 32;
  static constexpr uint32_t MinBlocksPerMultiprocessor = detail::small_channel_fprop::get_min_blocks_per_multiprocessor<Conv2dFilterShape>();
  static constexpr bool Swish = Swish_;

  struct Arguments {

    //
    // Data members
    //

    cutlass::Tensor4DCoord input_tensor_size;
    cutlass::Tensor4DCoord conv_filter_size;
    cutlass::Tensor4DCoord output_tensor_size;
    cutlass::Tensor4DCoord bias_tensor_size;
    TensorRef<half_t, layout::TensorNHWC> ref_A;
    TensorRef<half_t, layout::TensorNHWC> ref_B;
    TensorRef<half_t, layout::TensorNHWC> ref_C;
    TensorRef<half_t, layout::TensorNHWC> ref_bias;
    half_t kAlpha = half_t(1.0);
    half_t kLowerBound = cutlass::platform::numeric_limits<half_t>::lowest();
    half_t kUpperBound = cutlass::platform::numeric_limits<half_t>::max();

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      cutlass::Tensor4DCoord input_tensor_size,
      cutlass::Tensor4DCoord conv_filter_size,
      cutlass::Tensor4DCoord output_tensor_size,
      cutlass::Tensor4DCoord bias_tensor_size
    ):
      input_tensor_size(input_tensor_size),
      conv_filter_size(conv_filter_size),
      output_tensor_size(output_tensor_size),
      bias_tensor_size(bias_tensor_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::Tensor4DCoord input_tensor_size,
      cutlass::Tensor4DCoord conv_filter_size,
      cutlass::Tensor4DCoord output_tensor_size,
      TensorRef<half_t, layout::TensorNHWC> ref_A,
      TensorRef<half_t, layout::TensorNHWC> ref_B,
      TensorRef<half_t, layout::TensorNHWC> ref_C,
      half_t kAlpha = half_t(1.0),
      half_t kLowerBound = cutlass::platform::numeric_limits<half_t>::lowest(),
      half_t kUpperBound = cutlass::platform::numeric_limits<half_t>::max()
    ):
      input_tensor_size(input_tensor_size),
      conv_filter_size(conv_filter_size),
      output_tensor_size(output_tensor_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      kAlpha(kAlpha),
      kLowerBound(kLowerBound),
      kUpperBound(kUpperBound) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::Tensor4DCoord input_tensor_size,
      cutlass::Tensor4DCoord conv_filter_size,
      cutlass::Tensor4DCoord output_tensor_size,
      cutlass::Tensor4DCoord bias_tensor_size,
      TensorRef<half_t, layout::TensorNHWC> ref_A,
      TensorRef<half_t, layout::TensorNHWC> ref_B,
      TensorRef<half_t, layout::TensorNHWC> ref_C,
      TensorRef<half_t, layout::TensorNHWC> ref_bias,
      half_t kAlpha = half_t(1.0),
      half_t kLowerBound = cutlass::platform::numeric_limits<half_t>::lowest(),
      half_t kUpperBound = cutlass::platform::numeric_limits<half_t>::max()
    ):
      input_tensor_size(input_tensor_size),
      conv_filter_size(conv_filter_size),
      output_tensor_size(output_tensor_size),
      bias_tensor_size(bias_tensor_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_bias(ref_bias),
      kAlpha(kAlpha),
      kLowerBound(kLowerBound),
      kUpperBound(kUpperBound) { }
  };

  struct Params {
    cutlass::Tensor4DCoord input_tensor_size;
    cutlass::Tensor4DCoord conv_filter_size;
    cutlass::Tensor4DCoord output_tensor_size;
    cutlass::Tensor4DCoord bias_tensor_size;
    TensorRef<half_t, layout::TensorNHWC> ref_A;
    TensorRef<half_t, layout::TensorNHWC> ref_B;
    TensorRef<half_t, layout::TensorNHWC> ref_C;
    TensorRef<half_t, layout::TensorNHWC> ref_bias;
    half_t kAlpha;
    int kPadTop;
    int kPadLeft;
    half_t kLowerBound;
    half_t kUpperBound;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params(): kAlpha(half_t(1.0)) {
      kPadTop = (Conv2dFilterShape::kH - 1) / 2;
      kPadLeft = (Conv2dFilterShape::kW - 1) / 2;
      kLowerBound = cutlass::platform::numeric_limits<half_t>::lowest();
      kUpperBound = cutlass::platform::numeric_limits<half_t>::max();
    }

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *workspace = nullptr
    ):
      input_tensor_size(args.input_tensor_size),
      conv_filter_size(args.conv_filter_size),
      output_tensor_size(args.output_tensor_size),
      bias_tensor_size(args.bias_tensor_size),
      ref_A(args.ref_A),
      ref_B(args.ref_B),
      ref_C(args.ref_C),
      ref_bias(args.ref_bias),
      kAlpha(args.kAlpha),
      kLowerBound(args.kLowerBound),
      kUpperBound(args.kUpperBound)
    {
      kPadTop = (Conv2dFilterShape::kH - 1) / 2;
      kPadLeft = (Conv2dFilterShape::kW - 1) / 2;
    }
  };

  // grid planning
  static dim3 get_grid_shape(Params const& params) {
    constexpr int kFltK = Conv2dFilterShape::kN;
    constexpr int kFltR = Conv2dFilterShape::kH;
    constexpr int kFltS = Conv2dFilterShape::kW;
    constexpr int kFltC = Conv2dFilterShape::kC;

    int ctas_in_row = (params.output_tensor_size.h() + 15) / 16;
    int ctas_in_col = (params.output_tensor_size.w() + 63) / 64;

    if constexpr (kFltC == 8) {
      if constexpr (kFltR == 7 && kFltS == 7) {
        if constexpr (kFltK == 32) {
          ctas_in_col *= 2;
        } else if constexpr (kFltK == 64) {
          ctas_in_col *= 4;
        }
      } else if constexpr (kFltR == 5 && kFltS == 5) {
        if constexpr (kFltK == 64) {
          ctas_in_col *= 2;
        }
      }
    }
    return dim3(ctas_in_col, ctas_in_row, params.output_tensor_size.n());
  }

  /// Shared memory storage structure
  struct SharedStorage {
    uint16_t smem_flt_[detail::small_channel_fprop::get_smem_flt_size<Conv2dFilterShape>()];
    uint16_t smem_img_[detail::small_channel_fprop::get_smem_img_size<Conv2dFilterShape>()];
    uint16_t smem_out_[WarpsPerCTA][8][FltKPerCTA * 2 + 4 * 4];
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  SmallChannelConv2dFprop() { }

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {

// {$nv-internal-release begin}
// Knobs required by https://nvbugs/4851721 // {$nv-internal-release}
    asm volatile(".pragma \"global knob DisableStageAndFence=1\";\n" : : : "memory");
// {$nv-internal-release end}

    //
    // Define
    //
    constexpr int kFltK = Conv2dFilterShape::kN;
    constexpr int kFltR = Conv2dFilterShape::kH;
    constexpr int kFltS = Conv2dFilterShape::kW;
    constexpr int kFltC = Conv2dFilterShape::kC;

    auto input_tensor_size = params.input_tensor_size;
    auto conv_filter_size = params.conv_filter_size;
    auto output_tensor_size = params.output_tensor_size;
    auto bias_tensor_size = params.bias_tensor_size;
    auto ref_A = params.ref_A;
    auto ref_B = params.ref_B;
    auto ref_C = params.ref_C;
    auto ref_bias = params.ref_bias;
    int kPadTop = params.kPadTop;
    int kPadLeft = params.kPadLeft;
    half_t kAlpha = params.kAlpha;
    half_t kLowerBound = params.kLowerBound;
    half_t kUpperBound = params.kUpperBound;

    bool kWithBias = ref_bias.good();

    Mma mma;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // The number of threads per CTA
    constexpr int kThreadsPerWarp = 32;
    // The number of threads per CTA
    constexpr int kThreadsPerCTA = kThreadsPerWarp * WarpsPerCTA;

    // The Padding
    constexpr int kPadW = (kFltS - 1) / 2;

    // Each warp computes a tile of PIXELSxFILTERS where PIXELS == 64 and FILTERS is in
    // {16,32,64}
    constexpr int kOutP = WarpsPerCTA;
    constexpr int kOutQ = 64;

    // The size of the tile we keep in shared memory
    constexpr int kImgH = 2 * kOutP;
    constexpr int kImgW = 2 * kOutQ + 2 * kPadW;

    // Decompose the CTA into warps
    int kWarp = threadIdx.x / kThreadsPerWarp;
    int kLane = threadIdx.x % kThreadsPerWarp;

    //
    // Filter
    //

    // Shared memory to store the weights
    constexpr int kEvenFltS = (kFltS % 2 == 0) ? kFltS : kFltS + 1;
    constexpr int kEvenFltRS = kFltR * kEvenFltS;
    constexpr int kEvenFltKRS = FltKPerCTA * kEvenFltRS;
    auto& smem_flt_ = shared_storage.smem_flt_;

    // The number of K coefficients loaded per step
    constexpr int kFltKPerStep = kThreadsPerCTA / kEvenFltRS; // k_tile_of_CTA
    // The number of steps
    constexpr int kFltSteps = (FltKPerCTA + kFltKPerStep - 1) / kFltKPerStep; // k_tile_num_per_CTA
    // The K coefficient loaded by that thread
    int kLoadK = threadIdx.x / kEvenFltRS;
    // The R*S coefficient loaded by that thread
    int kLoadR = threadIdx.x % kEvenFltRS / kEvenFltS;
    int kLoadS = threadIdx.x % kEvenFltRS % kEvenFltS;

    // Each CTA covers FltKPerCTA filters
    // To compute the whole filters, we need kFltK/FltKPerCTA CTAs
    // CTAs in X dim : |Q0(kColsPerCTA)| Q1       | Q2       | ... | QN       |
    //                 |K0|K1|K2|...|KN| kKSlices | kKSlices | ... | kKSlices |
    constexpr int kKSlices = (kFltK + FltKPerCTA - 1) / FltKPerCTA;
    // Which part of K that the CTA covers
    int kKCTA = blockIdx.x % kKSlices;
    // The base K offset of current tile
    int kBaseK = kKCTA * FltKPerCTA;
    // The offset in global memory for the filter
    LongIndex kGmemFltOffset = (kLoadK + kBaseK) * ref_B.stride(2) +
                                    kLoadR * ref_B.stride(1) + kLoadS * ref_B.stride(0);

    // Each thread can load one pixel (4x fp16s) with a single LDG.64
    const AccessType *gmem_flt = reinterpret_cast<const AccessType *>(ref_B.data() + kGmemFltOffset);

    // We handle 4 channels each time
    constexpr int kCGroups = kFltC / 4;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int c = 0; c < kCGroups; ++c) {
      // Load he filter from global memory
      AccessType fetch_flt[kFltSteps];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kFltSteps; ++i) {
        fetch_flt[i].x = 0;
        fetch_flt[i].y = 0;
      }

      // fetch as usual
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kFltSteps; ++i) {
        if (kLoadK < kFltKPerStep && kLoadS < kFltS && kLoadK + kFltKPerStep * i < FltKPerCTA && kLoadK + kFltKPerStep * i < kFltK - kBaseK) {
          fetch_flt[i] = gmem_flt[i * kFltKPerStep * kFltR * kFltS * kCGroups + c];
        }
      }
      AccessType *smem_flt = reinterpret_cast<AccessType *>(
          &smem_flt_[(kLoadK * kEvenFltRS + kLoadR * kEvenFltS + kLoadS + c * kEvenFltKRS) * 4]);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kFltSteps; ++i) {
        if (kLoadK < kFltKPerStep && kLoadK + kFltKPerStep * i < FltKPerCTA) {
          smem_flt[i * kFltKPerStep * kEvenFltRS] = fetch_flt[i];
        }
      }
    }

    //
    // Activation
    //

    constexpr int kImgHW = kImgH * kImgW;

    auto& smem_img_ = shared_storage.smem_img_;
  
    // The n index
    int kN = blockIdx.z;

    // The P,Q coordinates
    // The beginning of the chunk
    constexpr int kRowsPerCta = 16;
    int kPBeg = blockIdx.y * kRowsPerCta;
    int kQBeg = blockIdx.x / kKSlices * kOutQ;

    // The end of the chunk
    int kPEnd = min(output_tensor_size.h(), kPBeg + kRowsPerCta);
    int kQEnd = min(output_tensor_size.w(), kQBeg + kOutQ);

    // The H,W coordinates
    int kHBeg = Conv2dStride::kU * kPBeg + kWarp - kPadTop;
    int kWBeg = Conv2dStride::kV * kQBeg + kLane - kPadLeft;
    int kHEnd =
        min(input_tensor_size.h(), Conv2dStride::kU * kPBeg + Conv2dStride::kU * kRowsPerCta + kFltR - 1 - kPadTop);
    int kWEnd =
        min(input_tensor_size.w(), Conv2dStride::kV * kQBeg + Conv2dStride::kV * kOutQ + kFltS - 1 - kPadLeft);
    // The offset where the thread starts loading
    LongIndex kImgInitOffset =
        kN * ref_A.stride(2) + kHBeg * ref_A.stride(1) + kWBeg * ref_A.stride(0);
    // Each thread can read one pixel (4x fp16s) with a single LDG.64 and the a warp works on a
    // row of data first then moves to the "next row" if needed.
    const ElementA *kGmemImgInit =
        reinterpret_cast<const ElementA *>(&ref_A.data()[kImgInitOffset]);

    // The number of steps to fetch a row and the different rows in the tile, resp.
    constexpr int kImgStepsInitH = (kImgH + WarpsPerCTA - 1) / WarpsPerCTA;
    constexpr int kImgStepsInitW = (kImgW + kThreadsPerWarp - 1) / kThreadsPerWarp;

    AccessType *smem_img_write;

    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kCGroups; ++c) {
      // In shared memory we maintain the structure of the tiles.
      smem_img_write =
          reinterpret_cast<AccessType *>(&smem_img_[(kWarp * kImgW + kLane + c * kImgHW) * 4]);

      // Allocate registers to fetch data.
      AccessType fetch_img_init[kImgStepsInitH][kImgStepsInitW];
      for (int hi = 0; hi < kImgStepsInitH; ++hi) {
        for (int wi = 0; wi < kImgStepsInitW; ++wi) {
          fetch_img_init[hi][wi].x = 0;
          fetch_img_init[hi][wi].y = 0;
        }
      }

      // Load the data.
      CUTLASS_PRAGMA_UNROLL
      for (int hi = 0; hi < kImgStepsInitH; ++hi) {
        int H = kHBeg + hi * WarpsPerCTA;
        CUTLASS_PRAGMA_UNROLL
        for (int wi = 0; wi < kImgStepsInitW; ++wi) {
          int W = kWBeg + wi * kThreadsPerWarp;
          if (H >= 0 && H < kHEnd && W >= 0 && W < kWEnd) {
            int offset = hi * WarpsPerCTA * ref_A.stride(1) + wi * kThreadsPerWarp * ref_A.stride(0);
            fetch_img_init[hi][wi] =
                *(reinterpret_cast<const AccessType *>(&kGmemImgInit[offset]) + c);
          }
        }
      }

      // Store the data into shared memory.
      for (int hi = 0; hi < kImgStepsInitH; ++hi) {
        for (int wi = 0; wi < kImgStepsInitW; ++wi) {
          if (kWarp + hi * WarpsPerCTA < kImgH && kLane + wi * kThreadsPerWarp < kImgW) {
            smem_img_write[hi * WarpsPerCTA * kImgW + wi * kThreadsPerWarp] = fetch_img_init[hi][wi];
          }
        }
      }
    }
    // Make sure the data is in shared memory
    __syncthreads();

    // Compute the offset in SMEM for threads to read the image
    // Different warps read the data in the exact same way
    // For LDSM.16.M88.2, only the first 16 threads provide the address
    // We distribute the work as (each column is 2 pixels, i.e. 8 x fp16):
    //
    // lane  0 -> column  0
    // lane  1 -> column  1
    // lane  2 -> column  2
    // lane  3 -> column  3
    // lane  4 -> column  4
    // lane  5 -> column  5
    // lane  6 -> column  6
    // lane  7 -> column  7
    // lane  8 -> column  8
    // lane  9 -> column  9
    // lane 10 -> column 10
    // lane 11 -> column 11
    // lane 12 -> column 12
    // lane 13 -> column 13
    // lane 14 -> column 14
    // lane 15 -> column 15
    //
    // the offset is lane*8
    int kSmemImgCol = kLane * 8;
    // Keep the pointer live across iterations of the main loop
    const int4 *smem_img_read = reinterpret_cast<const int4 *>(&smem_img_[kSmemImgCol]);

    // Compute the offset in SMEM for threads to read the filter
    // Different warps read the data in the exact same way
    // For LDSM.16.M88.1, only the first 8 threads provide the address
    // We distribute the work as (each column is 1 filter):
    //
    // lane  0 -> column  0
    // lane  1 -> column  1
    // lane  2 -> column  2
    // lane  3 -> column  3
    // lane  4 -> column  4
    // lane  5 -> column  5
    // lane  6 -> column  6
    // lane  7 -> column  7
    // lane  8 -> column  8
    //
    // the offset is lane
    int kSmemFltOffset = kLane * kEvenFltRS * 4;
    // The address in filter memory to read the filter from
    const AccessType *smem_flt_read =
        reinterpret_cast<const AccessType *>(&smem_flt_[kSmemFltOffset]);

    //
    // Calculate the index for the next LDG 2 rows
    //

    // In main loop we load 2 rows into SMEM per iteration
    constexpr int kThreadsPerRow = kThreadsPerCTA / 2;

    int kTidxDivThreadsPerRow = threadIdx.x / kThreadsPerRow;
    int kTidxModThreadsPerRow = threadIdx.x % kThreadsPerRow;

    // H,W coordinates
    int h = Conv2dStride::kU * kPBeg + kImgH + kTidxDivThreadsPerRow - kPadTop;
    int w = Conv2dStride::kV * kQBeg + kTidxModThreadsPerRow - kPadLeft;

    // The offset where the thread starts loading
    LongIndex kGmemImgOffset = kN * ref_A.stride(2) + h * ref_A.stride(1) + w * ref_A.stride(0);

    // Reinitialize the gmem_img pointer
    const ElementA *gmem_img = reinterpret_cast<const ElementA *>(&ref_A.data()[kGmemImgOffset]);

    // Where to write tha data
    int smem_img_write_row = kTidxDivThreadsPerRow;
    // Reinitialize the shared memory pointer
    smem_img_write = reinterpret_cast<AccessType *>(&smem_img_[kTidxModThreadsPerRow * 4]);

    // The number of iterations for R/S.
    constexpr int kStepsR = (kFltR + 1) / 2;
    constexpr int kStepsS = kEvenFltS / 2;

    int kGmemImgOffsetPerRStep = 2 * ref_A.stride(1);

    constexpr int kHMMARows = kOutQ / 16;
    constexpr int kHMMACols = FltKPerCTA / 8;

    // Outer-loop computes 4 different output rows
    CUTLASS_PRAGMA_NO_UNROLL
    for (int p = kPBeg; p < kPEnd; p += WarpsPerCTA) {
      Array<float, 4> acc[kHMMARows][kHMMACols];

      for (int i = 0; i < kHMMARows; ++i) {
        for (int j = 0; j < kHMMACols; ++j) {
          acc[i][j].clear();
        }
      }
      jetfire::ifence(); // {$nv-internal-release}
      // The position where we start reading from in SMEM.
      // We have one warp per row and a stride of 2 between two rows
      int smem_img_row = kWarp * Conv2dStride::kU;

      // The filter coefficient that we are currently reading. Rollback for every iteration.
      int smem_flt_idx = 0;

      // Perform the convolution, we decompose the kFltR rows into 2+2+2+1
      CUTLASS_PRAGMA_NO_UNROLL
      for (int r = 0; r < kStepsR; ++r) {
        // Make sure data is in SMEM
        __syncthreads();

        // Issue the loading of the rows
        constexpr int kImgStepsW = (kImgW + kThreadsPerRow - 1) / kThreadsPerRow;
        AccessType fetch_img[kImgStepsW];

        int tmp_smem_img_row = smem_img_row;
        int tmp_smem_flt_idx = smem_flt_idx;

        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kCGroups; ++c) {
          for (int wi = 0; wi < kImgStepsW; ++wi) {
            fetch_img[wi].x = 0;
            fetch_img[wi].y = 0;
          }

          for (int wi = 0; wi < kImgStepsW; ++wi) {
            int W = w + wi * kThreadsPerRow;
            if (h >= 0 && h < kHEnd && W >= 0 && W < kWEnd) {
              int offset = wi * kThreadsPerRow * ref_A.stride(0);
              fetch_img[wi] = *(reinterpret_cast<const AccessType *>(&gmem_img[offset]) + c);
            }
          }

          jetfire::ifence(); // {$nv-internal-release}

          smem_img_row = tmp_smem_img_row;
          smem_flt_idx = tmp_smem_flt_idx;

          int kReadOffsetC = c * (kImgHW / 2);
          int kWriteOffsetC = c * kImgHW;

          // Update the convolution using 2 rows unless we are in the last iteration
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            // For odd filters, skip the last iteration as we have already visited all rows.
            if ((kFltR % 2 == 1) && r == kFltR / 2 && i == 1) {
              break;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < kStepsS; s++) {
              LdsmOperandA A[kHMMARows];
              CUTLASS_PRAGMA_UNROLL
              for (int ri = 0; ri < kHMMARows; ++ri) {
                cutlass::arch::ldsm<SmemLayoutA, 2>(
                    A[ri], &smem_img_read[smem_img_row * (kImgW / 2) + s + ri * 16 + kReadOffsetC]);
              }

              for (int ki = 0; ki < kHMMACols; ki++) {
                LdsmOperandB B;
                cutlass::arch::ldsm<cutlass::layout::RowMajor, 1>(
                    B, &smem_flt_read[smem_flt_idx + 8 * ki * kEvenFltRS + c * kEvenFltKRS]);
                jetfire::ifence(); // {$nv-internal-release}
                const MmaOperandB *ptr_B = reinterpret_cast<const MmaOperandB *>(&B);
                CUTLASS_PRAGMA_UNROLL
                for (int ri = 0; ri < kHMMARows; ri++) {
                  const MmaOperandA *ptr_A = reinterpret_cast<const MmaOperandA *>(&A[ri]);
                  MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&acc[ri][ki]);
                  mma(ptr_D[0], ptr_A[0], ptr_B[0], ptr_D[0]);
                }
              }
              smem_flt_idx += 2;
            }  // for (s)

            // Move to the next row in SMEM
            smem_img_row = (smem_img_row == kImgH - 1) ? 0 : (smem_img_row + 1);

          }  // for (i)

          // Make sure we read the data from SMEM
          __syncthreads();

          // Store the rows to SMEM
          for (int wi = 0; wi < kImgStepsW; ++wi) {
            if (kTidxModThreadsPerRow + wi * kThreadsPerRow < kImgW) {
              smem_img_write[kImgW * smem_img_write_row + wi * kThreadsPerRow + kWriteOffsetC] =
                  fetch_img[wi];
            }
          }
          jetfire::ifence(); // {$nv-internal-release}
        }  // for (c)
        // Update the row
        smem_img_write_row =
            smem_img_write_row >= kImgH - 2 ? kTidxDivThreadsPerRow : smem_img_write_row + 2;

        h += 2;
        gmem_img += kGmemImgOffsetPerRStep;
      }  // for (r)

      //
      // OUTPUT
      //

      // The buffer to reorder the data
      // Pack two pixels per row
      auto& smem_out_ = shared_storage.smem_out_;
      // For a given warp, we have the following output q
      //
      // lane 0, 1, 2, 3 -> q = 0, 8, 16, 24, 32, 40, 48, 56
      // lane 4, 5, 6, 7 -> q = 1, 9, 17, 25, 33, 41, 49, 57
      // lane 8, 9, 10, 11 -> q = 2, 10, 18, 26, 34, 42, 50, 58
      // lane 12, 13, 14, 15 -> q = 3, 11, 19, 27, 35, 43, 51, 59
      // ...
      // lane 28, 29, 30, 31 -> q = 7, 15, 23, 31, 39, 47, 55, 63
      //
      // The offset is: lane/4
      int kSmemQ = kLane / 4;

      // For a given warp, we have the following output in K dimension
      // We express those offsets in terms of int2 numbers
      //
      // lane 0, 4, 8, 12, 16, 20, 24, 28 -> k = {0, 1}, {8, 9}, {16, 17}, {24, 25}, ...
      // lane 1, 5, 9, 13, 17, 21, 25, 29 -> k = {2, 3}, {10, 11}, {18, 19}, {26, 27}, ...
      // lane 2, 6, 10, 14, 18, 22, 26, 30 -> k = {4, 5}, {12, 13}, {20, 21}, {28, 29}, ...
      // lane 3, 7, 11, 15, 19, 23, 27, 31 -> k = {6, 7}, {14, 15}, {22, 23}, {30, 31}, ...
      //
      // So, the offset in the K dimension is lane&0x3*2
      int kSmemK = (kLane & 0x3) * 4;  // Pack 4 pixels ([q0k0 q0k1 q1k0 q1k1])

      // The number of threads per pixel
      constexpr int kThreadsPerPixel = FltKPerCTA <= 32 ? 16 : 32;

      // Decompose the warp
      int kLaneDivTpp = kLane / kThreadsPerPixel;
      int kLaneModTpp = kLane % kThreadsPerPixel;

      // The destination coordinates
      int kDstP = p + kWarp;
      int kDstQ = kQBeg + kLaneDivTpp;
      int kDstK = kLaneModTpp;

      // The destination offset
      LongIndex kDstOffset = kN * ref_C.stride(2) + kDstP * ref_C.stride(1) +
                                  kDstQ * ref_C.stride(0) + kDstK * 2 + kKCTA * FltKPerCTA;

      // The destination pointer
      int *gmem_out = reinterpret_cast<int *>(&ref_C.data()[kDstOffset]);

      uint32_t bias = 0;

      // Valid check
      bool kPValid = kDstP < kPEnd;
      bool kKValid = kDstK * 2 < FltKPerCTA && kDstK * 2 < kFltK - kBaseK;

      // Load bias
      if (kWithBias && kKValid) {
          // The bias pointer
          uint32_t *bias_ptr = reinterpret_cast<uint32_t *>(&ref_bias.data()[kKCTA * FltKPerCTA]);
          bias = bias_ptr[kLaneModTpp];
      }

      // Output the 4x8 different blocks
      CUTLASS_PRAGMA_UNROLL
      for (int qi = 0; qi < kHMMARows; ++qi) {
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < kHMMACols; ++ki) {
          uint32_t q0_k0k1 = float2_to_half2(acc[qi][ki][0], acc[qi][ki][1]);
          uint32_t q1_k0k1 = float2_to_half2(acc[qi][ki][2], acc[qi][ki][3]);
          Array<uint32_t, 2> q0q1;
          q0q1[0] = q0_k0k1;
          q0q1[1] = q1_k0k1;

          // Store to shared memory
          reinterpret_cast<AccessType *>(&smem_out_[kWarp][kSmemQ][16 * ki + kSmemK + 0])[0] =
              reinterpret_cast<AccessType *>(&q0q1)[0];
        }

        // Make sure data is in smem
        __syncwarp();

        uint32_t kAlphax2 = half_to_half2(kAlpha);
        uint32_t kReluLBx2 = half_to_half2(kLowerBound);
        uint32_t kReluUBx2 = half_to_half2(kUpperBound);

        // Print the 16 pixels per warp
        CUTLASS_PRAGMA_UNROLL
        for (int qii = 0; qii < 2; ++qii) {
          CUTLASS_PRAGMA_UNROLL
          for (int qiii = 0; qiii < 4; ++qiii) {
            if (FltKPerCTA > 32 * (qiii % 2)) {
              // Valid check
              bool kQValid0 = kDstQ + qi * 16 + qii * 4 + qiii + 0 < kQEnd;
              bool kQValid1 = kDstQ + qi * 16 + qii * 4 + qiii + 8 < kQEnd;

              // Load from shared memory
              int2 data;
              if (kKValid) {
                data = reinterpret_cast<const int2 *>(
                    &smem_out_[kWarp][qii * 4 + qiii + kLaneDivTpp][4 * kLaneModTpp])[0];
              }

              data.x = fma_fp16x2(kAlphax2, data.x, bias);
              data.y = fma_fp16x2(kAlphax2, data.y, bias);

              if (Swish == false) {
                data.x = clippedRelu(data.x, kReluLBx2, kReluUBx2);
                data.y = clippedRelu(data.y, kReluLBx2, kReluUBx2);
              } else {
                data.x = swish(data.x);
                data.y = swish(data.y);
              }

              // Output to memory
              int kQIdx = 16 * qi + qii * 4 + qiii;
              int kHalfStrideK = ref_C.stride(0) / 2;
              if (kPValid && kQValid0 && kKValid) {
                gmem_out[kQIdx * kHalfStrideK] = data.x;
              }

              if (kPValid && kQValid1 && kKValid) {
                gmem_out[(kQIdx + 8) * kHalfStrideK] = data.y;
              }
            }
          }
        }  // for (qii)
      // Make sure the data was consumed.
      __syncwarp();
      }    // for (qi)
    }      // for (p)
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
