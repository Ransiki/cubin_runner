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
    \brief Template for a Convolution Small Channel Wgrad kernel.
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
namespace small_channel_wgrad {

// Return MinBlocksPerMultiprocessor
template<typename Conv2dFilterShape>
constexpr int
get_min_blocks_per_multiprocessor() {
  constexpr int kFltK = Conv2dFilterShape::kN;
  constexpr int kFltR = Conv2dFilterShape::kH;
  constexpr int kFltS = Conv2dFilterShape::kW;
  constexpr int kFltC = Conv2dFilterShape::kC;
  constexpr int MinBlocksPerMultiprocessor = 1;

  if constexpr (kFltK == 16 && kFltR == 5 && kFltS == 5 && kFltC == 4) {
    return 6;
  }
  if constexpr (kFltK == 32 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
    return 9;
  }
  if constexpr (kFltK == 64 && kFltR == 7 && kFltS == 7 && kFltC == 4) {
    return 2;
  }
  if constexpr (kFltK == 64 && kFltR == 5 && kFltS == 5 && kFltC == 4) {
    return 3;
  }
  if constexpr (kFltK == 64 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
    return 5;
  }

  return MinBlocksPerMultiprocessor;
}

// Return buf size of smem_flt
template<typename Conv2dFilterShape>
constexpr int
get_smem_err_size() {
  constexpr int kFltKPerCTA =  Conv2dFilterShape::kN;
  // Compute the number of threads per pixel. We load 8 elements per LDG.128.
  constexpr int kThreadsPerErrPixel = kFltKPerCTA / 8;
  // Compute the skew as described above. This skew is in number of fp16.
  constexpr int kSmemErrSkew = 8 / kThreadsPerErrPixel * 8;
  // Compute the number of slices.
  constexpr int kSmemErrSlices = kFltKPerCTA / 8;
  // The main loop operates on 32 different error pixels taken from a single row.
  constexpr int kErrQ = 32;
  // The number of fp16 elements per slice.
  constexpr int kSmemErrFp16sPerSlice = kErrQ * 8 + kSmemErrSkew;
  return kSmemErrSlices * kSmemErrFp16sPerSlice;
}

// Return buf size of smem_img
template<typename Conv2dStride, typename Conv2dFilterShape>
constexpr int
get_smem_img_size() {
  constexpr int kFltR = Conv2dFilterShape::kH;
  constexpr int kFltS = Conv2dFilterShape::kW;
  constexpr int kFltC = Conv2dFilterShape::kC;
  constexpr int kPadW = (kFltS - 1) / 2;
  // The main loop operates on 32 different error pixels taken from a single row.
  constexpr int kErrQ = 32;
  constexpr int kImgH = Conv2dStride::kU * kFltR;
  constexpr int kImgW = Conv2dStride::kV * kErrQ + 2 * kPadW;
  constexpr int kImgHW = kImgH * kImgW;
  return kImgHW * kFltC;
}

} // end of namespace small_channel_wgrad
} // end of namespace detail

template <
  typename ArchTag_,
  typename Conv2dFilterShape_,
  int NumLocks_ = 64
>
struct SmallChannelConv2dWgrad {

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
  using LdsmOperandA = typename LdsmA<Conv2dFilterShape::kH, Conv2dFilterShape::kW>::LdsmOperand;
  using LdsmOperandB = typename LdsmB<Conv2dFilterShape::kN>::LdsmOperand;

  using MmaOperandA = typename Mma::FragmentA;
  using MmaOperandB = typename Mma::FragmentB;
  using MmaOperandC = typename Mma::FragmentC;

  static constexpr Operator kConvolutionalOperator = conv::Operator::kWgrad;

  static constexpr uint32_t WarpsPerCTA = Conv2dFilterShape::kH;
  static constexpr uint32_t FltKPerCTA = Conv2dFilterShape::kN;
  static constexpr uint32_t MaxThreadsPerBlock = WarpsPerCTA * 32;
  static constexpr uint32_t MinBlocksPerMultiprocessor = detail::small_channel_wgrad::get_min_blocks_per_multiprocessor<Conv2dFilterShape>();
  static constexpr int NumLocks = NumLocks_;

  struct Arguments {

    //
    // Data members
    //

    cutlass::Tensor4DCoord input_tensor_size;
    cutlass::Tensor4DCoord output_tensor_size;
    cutlass::Tensor4DCoord conv_filter_size;
    TensorRef<half_t, layout::TensorNHWC> ref_A;
    TensorRef<half_t, layout::TensorNHWC> ref_B;
    TensorRef<half_t, layout::TensorNHWC> ref_C;

    struct {
      int slices = 1;
      int buffers = 64;
    } split_k;
    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      cutlass::Tensor4DCoord input_tensor_size,
      cutlass::Tensor4DCoord output_tensor_size,
      cutlass::Tensor4DCoord conv_filter_size
    ):
      input_tensor_size(input_tensor_size),
      output_tensor_size(output_tensor_size),
      conv_filter_size(conv_filter_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::Tensor4DCoord input_tensor_size,
      cutlass::Tensor4DCoord output_tensor_size,
      cutlass::Tensor4DCoord conv_filter_size,
      TensorRef<half_t, layout::TensorNHWC> ref_A,
      TensorRef<half_t, layout::TensorNHWC> ref_B,
      TensorRef<half_t, layout::TensorNHWC> ref_C
    ):
      input_tensor_size(input_tensor_size),
      output_tensor_size(output_tensor_size),
      conv_filter_size(conv_filter_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::Tensor4DCoord input_tensor_size,
      cutlass::Tensor4DCoord output_tensor_size,
      cutlass::Tensor4DCoord conv_filter_size,
      TensorRef<half_t, layout::TensorNHWC> ref_A,
      TensorRef<half_t, layout::TensorNHWC> ref_B,
      TensorRef<half_t, layout::TensorNHWC> ref_C,
      int slices, int buffers
    ):
      input_tensor_size(input_tensor_size),
      output_tensor_size(output_tensor_size),
      conv_filter_size(conv_filter_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      split_k({slices, buffers}) { }

  };

  struct Params {
    cutlass::Tensor4DCoord input_tensor_size;
    cutlass::Tensor4DCoord output_tensor_size;
    cutlass::Tensor4DCoord conv_filter_size;
    TensorRef<half_t, layout::TensorNHWC> ref_A;
    TensorRef<half_t, layout::TensorNHWC> ref_B;
    TensorRef<half_t, layout::TensorNHWC> ref_C;
    int kPadTop;
    int kPadLeft;
    int kRowsPerCta;
    int kSlices;
    int kNumLocks;
    int *gmem_locks;
    int *gmem_retired_ctas;
    uint16_t *gmem_red;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params(): kNumLocks(64/*default_values*/), gmem_locks(nullptr), gmem_retired_ctas(nullptr), gmem_red(nullptr) {
      kPadTop = (Conv2dFilterShape::kH - 1) / 2;
      kPadLeft = (Conv2dFilterShape::kW - 1) / 2;
    }

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *workspace = nullptr
    ):
      input_tensor_size(args.input_tensor_size),
      output_tensor_size(args.output_tensor_size),
      conv_filter_size(args.conv_filter_size),
      ref_A(args.ref_A),
      ref_B(args.ref_B),
      ref_C(args.ref_C)
    {
      kPadTop = (Conv2dFilterShape::kH - 1) / 2;
      kPadLeft = (Conv2dFilterShape::kW - 1) / 2;

      constexpr int kFltK = Conv2dFilterShape::kN;
      constexpr int kFltR = Conv2dFilterShape::kH;
      // 16 bytes alignment for reduction buffer stg
      constexpr int alignment = 16;
      constexpr int kFltKPerCTA = kFltK;
      constexpr int kCtasPerK = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;

      int kSplitsInP = cutlass::platform::max(args.split_k.slices, 1);
      kSlices = kSplitsInP;
      int kSplitsInQ = (args.output_tensor_size.w() + 31) / 32 * kCtasPerK;
      int kSplitsInN = args.output_tensor_size.n();
      int kTotalSplits = kSplitsInP * kSplitsInQ * kSplitsInN;
      kRowsPerCta = (args.output_tensor_size.h() + kSplitsInP - 1) / kSplitsInP;
      auto find_log2 = [](int x) {
        int clz = 32;
        for (int i = 31; i >= 0; --i) {
          if ((1 << i) & x) {
            clz = 31 - i;
            break;
          }
        }
        int result = 31- clz;
        result += (x & (x - 1)) != 0; // Roundup, add 1 if not a power of 2.
        return result;
      };
      kNumLocks = args.split_k.buffers > 0 ? args.split_k.buffers
          : (1 << ((find_log2(kTotalSplits) + 1) >> 1));
      size_t kLocksSize = ((kNumLocks + 1) * kCtasPerK * kFltR * sizeof(int) + alignment - 1)
        / alignment * alignment;
      gmem_locks = workspace;
      gmem_retired_ctas = &workspace[kNumLocks * kCtasPerK * kFltR];
      gmem_red = reinterpret_cast<uint16_t *>((char *)workspace + kLocksSize);
    }
  };

  // grid planning
  static dim3 get_grid_shape(Params const& params) {
    constexpr int kFltK = Conv2dFilterShape::kN;

    // int kRowsPerCTA = params.output_tensor_size.h();
    constexpr int kFltKPerCTA = kFltK; 
    constexpr int kCtasPerK = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;
    int kCtasInRow = cutlass::platform::max(params.kSlices, 1);
    int kCtasInCol = (params.output_tensor_size.w() + 31) / 32 * kCtasPerK;
    return dim3(kCtasInCol, kCtasInRow, params.output_tensor_size.n());
  }

  /// Shared memory storage structure
  struct SharedStorage {
    uint16_t smem_err_[2][detail::small_channel_wgrad::get_smem_err_size<Conv2dFilterShape>()];
    uint16_t smem_img_[detail::small_channel_wgrad::get_smem_img_size<Conv2dStride, Conv2dFilterShape>()];
    uint16_t smem_flt_[Conv2dFilterShape::kH][Conv2dFilterShape::kW][4 * FltKPerCTA + 16];
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  SmallChannelConv2dWgrad() { }

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
    auto output_tensor_size = params.output_tensor_size;
    auto conv_filter_size = params.conv_filter_size;
    auto ref_A = params.ref_A;
    auto ref_B = params.ref_B;
    auto ref_C = params.ref_C;
    int kPadTop = params.kPadTop;
    int kPadLeft = params.kPadLeft;
    int kNumLocks = params.kNumLocks;
    int *gmem_locks = params.gmem_locks;
    int *gmem_retired_ctas = params.gmem_retired_ctas;
    uint16_t *gmem_red = params.gmem_red;

    int kRowsPerCta = params.kRowsPerCta;

    Mma mma;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // The number of ctas to compute all K
    constexpr int kCtasPerK = (kFltK + FltKPerCTA - 1) / FltKPerCTA;
    // The number of ctas in group of all K
    int kNumCtas = gridDim.x * gridDim.y * gridDim.z / kCtasPerK;
    // The number of threads per warp.
    constexpr int kThreadsPerWarp = 32;
    // The number of threads per CTA.
    constexpr int kThreadsPerCTA = WarpsPerCTA * kThreadsPerWarp;

    // The padding.
    constexpr int kPadW = (kFltS - 1) / 2;

    // Each warp works on C x K coefficients for an entire row of the filter. So, for example, if
    // the filter is 7 x 7, we have 7 warps and each warp computes a 7 x C x K filter values. In
    // this kernel we load 32 pixels of the error tensor per loop iteration.

    // The main loop operates on 32 different error pixels taken from a single row.
    constexpr int kErrQ = 32;

    // The size of the tile that we keep in shared memory.
    constexpr int kImgH = Conv2dStride::kU * kFltR;
    constexpr int kImgW = Conv2dStride::kV * kErrQ + 2 * kPadW;

    // The number of filter coefficients computed per thread.
    // We compute 4 x s per HMMA.1688
    constexpr int kFltSPerThread = (kFltS + 3) / 4;

    // Decompose the CTA into warps.
    int kWarp = threadIdx.x / kThreadsPerWarp;
    int kLane = threadIdx.x % kThreadsPerWarp;

    //
    // ERRORS
    //

    // Shared memory to store the error pixels. We decompose the channels of the pixels into groups
    // of 8, e.g. for 64 channels, we have 8 groups per pixel. For a given group of 8 channels, we
    // store the pixels contiguously in shared memory. To be concrete, consider a tile of size
    // 1x32 pixels (32 pixels from one row) and each pixel has 64 channels. We organize the shared
    // memory as 8 x 32 x 8 elements with 8 slices of 32 pixels each with 8 channels.
    //
    // At the end of each slice, we add a skew to avoid bank conflicts when we store the pixels to
    // memory. When we load the pixels from global memory, we use LDG.128 such that we need
    // FLT_K / 8 threads to load a single pixel. We store the data "as-is" using STS.128. We
    // support FLT_K == 16, 32 or 64 and we need a different skew for those 3 cases:
    //
    // 1/ If FLT_K == 64, we use 8 threads per pixel and each thread in a group of 8 threads (the
    // granularity to determine bank conflicts for STS.128) writes to a different slice. In that
    // case it is sufficient to use a skew of 8 fp16 values (16B).
    //
    // 2/ If FLT_K == 32, we use 4 threads per pixel and two threads in a group of 8 write to two
    // consecutive pixels in a slice. In that case, we need a skew of 32B.
    //
    // 3/ If FLT_K == 16, we use 2 threads per pixel and four threads in a group of 8 write to 4
    // consecutive pixels in a slice. In that case, we need a skew of 64B.

    // Compute the number of threads per pixel. We load 8 elements per LDG.128.
    constexpr int kThreadsPerErrPixel = FltKPerCTA / 8;
    // Compute the skew as described above. This skew is in number of fp16.
    constexpr int kSmemErrSkew = 8 / kThreadsPerErrPixel * 8;
    // The number of fp16 elements per slice.
    constexpr int kSmemErrFp16sPerSlice = kErrQ * 8 + kSmemErrSkew;

    // Declare the shared memory buffer.
    auto& smem_err_ = shared_storage.smem_err_;

    // The n index
    int kN = blockIdx.z;

    // The qk indices.
    int kCtaQ = blockIdx.x / (kFltK / FltKPerCTA);
    int kCtaK = blockIdx.x % (kFltK / FltKPerCTA);

    // The p,q coordinate. It is the beginning of the chunk.
    int kPBeg = blockIdx.y * kRowsPerCta;
    int kQBeg = kCtaQ * kErrQ;

    // The end of the chunk for p,q
    int kPEnd = cutlass::platform::min(output_tensor_size.h(), kPBeg + kRowsPerCta);
    int kQEnd = cutlass::platform::min(output_tensor_size.w(), kQBeg + kErrQ);

    // Track which row of errs to load from global memory
    int ldg_p = kPBeg;

    // The start offset of current err row in global memory for the error tensor.
    LongIndex kGmemErrOffset =
        kN * ref_B.stride(2) + ldg_p * ref_B.stride(1);

    // Load the memory by groups of 8 channels.
    const ElementB *gmem_err = &ref_B.data()[kGmemErrOffset];
    constexpr int kErrLDG32Num = (kErrQ * kFltK + kThreadsPerCTA * 2 - 1) / (kThreadsPerCTA * 2);
    int fetch_err[kErrLDG32Num];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kErrLDG32Num; ++i) {
      fetch_err[i] = 0;
    }

    wgrad_ldg_b<kFltK, kFltR, kFltS>(threadIdx.x, ldg_p, kPEnd, kQBeg, kQEnd, kCtaK, FltKPerCTA, ref_B.stride(0), reinterpret_cast<uint16_t const*>(gmem_err), fetch_err);

    constexpr int kKPerSlice = 8;

    // Move to the next row of err
    gmem_err += ref_B.stride(1);
    ldg_p += 1;

    wgrad_sts_b<kFltK, kFltR, kFltS>(threadIdx.x, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err_[0], fetch_err);

    //
    // ACTIVATIONS
    //

    // Shared memory to store the activations. This buffer is used as a circular buffer with
    // 2*FLT_R different rows containing up to 64+2*PAD_W pixels each. In the main loop, we circle
    // over the rows to store the data needed to compute the convolutions.
    auto& smem_img_ = shared_storage.smem_img_;

    // The position in the H and W dimensions. Each warp works on a different row in the prologue
    // and we take the padding into account.
    int kHBeg = Conv2dStride::kU * kPBeg + kWarp - kPadTop;
    int kWBeg = Conv2dStride::kV * kQBeg + kLane - kPadLeft;

    // The end of the chunkf for h/w.
    int kHEnd =
        cutlass::platform::max(cutlass::platform::min(input_tensor_size.h(), Conv2dStride::kU * (kPBeg + kRowsPerCta - 1) + kFltR - kPadTop), 0);
    int kWEnd =
        cutlass::platform::max(cutlass::platform::min(input_tensor_size.w(), Conv2dStride::kV * (kQBeg + kErrQ - 1) + kFltS - kPadLeft), 0);

    // The offset where the thread starts loading.
    LongIndex kGmemImgOffsetInit =
        kN * ref_A.stride(2) + kHBeg * ref_A.stride(1) + kWBeg * ref_A.stride(0);

    // Each thread can read one pixel (4x fp16s) with a single LDG.64 and a warp works on
    // a row of data first then moves to the "next row" if needed.
    const ElementA *kGmemImgInit = &ref_A.data()[kGmemImgOffsetInit];

    // In shared memory we maintain the structure of tiles.
    ElementA *smem_img_write =
        reinterpret_cast<ElementA *>(&smem_img_[kWarp * kImgW * 4 + kLane * 4]);

    // The number of steps to fetch a row and the different rows in the tile, resp.
    constexpr int kImgStepsInitH = (kFltR + WarpsPerCTA - 1) / WarpsPerCTA;
    constexpr int kImgStepsInitW = (kImgW + kThreadsPerWarp - 1) / kThreadsPerWarp;

    // Allocate registers to fetch data.
    AccessType fetch_img_init[kImgStepsInitH][kImgStepsInitW];
    CUTLASS_PRAGMA_UNROLL
    for (int hi = 0; hi < kImgStepsInitH; ++hi) {
      CUTLASS_PRAGMA_UNROLL
      for (int wi = 0; wi < kImgStepsInitW; ++wi) {
          fetch_img_init[hi][wi].x = 0;
          fetch_img_init[hi][wi].y = 0;
      }
    }

    // Load the data.
    CUTLASS_PRAGMA_UNROLL
    for (int hi = 0; hi < kImgStepsInitH; ++hi) {
      bool kValidH = (unsigned)(kHBeg + hi * WarpsPerCTA) < kHEnd;
      CUTLASS_PRAGMA_UNROLL
      for (int wi = 0; wi < kImgStepsInitW; ++wi) {
        bool kValidW = (unsigned)(kWBeg + wi * kThreadsPerWarp) < kWEnd;
        if (kValidH && kValidW) {
          LongIndex kOffset =
              hi * WarpsPerCTA * ref_A.stride(1) + wi * kThreadsPerWarp * ref_A.stride(0);
          fetch_img_init[hi][wi] = reinterpret_cast<const AccessType *>(&kGmemImgInit[kOffset])[0];
        }
      }
    }

    // Store the data into shared memory.
    CUTLASS_PRAGMA_UNROLL
    for (int hi = 0; hi < kImgStepsInitH; ++hi) {
      CUTLASS_PRAGMA_UNROLL
      for (int wi = 0; wi < kImgStepsInitW; ++wi) {
        if (kWarp + hi * WarpsPerCTA < kImgH && kLane + wi * kThreadsPerWarp < kImgW) {
          LongIndex kOffset = hi * WarpsPerCTA * kImgW * 4 + wi * kThreadsPerWarp * 4;
          reinterpret_cast<AccessType *>(&smem_img_write[kOffset])[0] = fetch_img_init[hi][wi];
        }
      }
    }

    // Make sure the data is in shared memory
    __syncthreads();

    // Prefetch

    // Issue the loading of the error row.
    wgrad_ldg_b<kFltK, kFltR, kFltS>(threadIdx.x, ldg_p, kPEnd, kQBeg, kQEnd, kCtaK, FltKPerCTA, ref_B.stride(0), reinterpret_cast<uint16_t const*>(gmem_err), fetch_err);

    jetfire::ifence(); // {$nv-internal-release}

    // We can now determine how many threads to assign per row to load the activations in the main
    // loop.
    constexpr int kThreadsPerRow = kThreadsPerCTA / Conv2dStride::kU;
    constexpr int kImgLDG32Num = (kImgW * 4 + kThreadsPerRow * 2 - 1) / (kThreadsPerRow * 2);

    // Decompose the thread index to fetch rows per iteration.
    int kTidxDivThreadsPerRow = threadIdx.x / kThreadsPerRow;
    int kTidxModThreadsPerRow = threadIdx.x % kThreadsPerRow;

    // In the main loop we fetch two rows per iteration so we redistribute the threads to do so.
    int h = Conv2dStride::kU * kPBeg + kTidxDivThreadsPerRow - kPadTop + kFltR;
    int w = Conv2dStride::kV * kQBeg - kPadLeft; // The first w in this row

    // The offset where the threads starts loading.
    LongIndex kGmemImgOffset = kN * ref_A.stride(2) + h * ref_A.stride(1);
    // Reinitialize the gmem_img pointer
    const ElementA *gmem_img = &ref_A.data()[kGmemImgOffset];

    int fetch_img[kImgLDG32Num];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kImgLDG32Num; ++i) {
      fetch_img[i] = 0;
    }

    // Issue the loading of the image rows.
    wgrad_ldg_a<kFltR, kFltS>(kThreadsPerRow, kTidxModThreadsPerRow, h, w, kHEnd, kWEnd, ref_A.stride(0),
      reinterpret_cast<uint16_t const*>(gmem_img), fetch_img);

    // Move the pointers.
    gmem_err += ref_B.stride(1);
    ldg_p += 1;
    gmem_img += Conv2dStride::kU * ref_A.stride(1);
    h += Conv2dStride::kU;

    // Compute the offset in SMEM for each thread to load the image. The different warps read the
    // data in the exact same way but from different rows. Using LDSM.16.MT88.4. 
    // Each column is 1 pixel (4x fp16s). We distribute the work as:
    //
    // lane  0 -> column 0
    // lane  1 -> column 2
    // lane  2 -> column 4
    // lane  3 -> column 6
    // lane  4 -> column 8
    // lane  5 -> column 10
    // lane  6 -> column 12
    // lane  7 -> column 14
    //
    // lane  8 -> column 2
    // lane  9 -> column 4
    // lane 10 -> column 6
    // lane 11 -> column 8
    // lane 12 -> column 10
    // lane 13 -> column 12
    // lane 14 -> column 14
    // lane 15 -> column 16
    //
    // lane 16 -> column 4
    // lane 17 -> column 6
    // lane 18 -> column 8
    // lane 19 -> column 10
    // lane 20 -> column 12
    // lane 21 -> column 14
    // lane 22 -> column 16
    // lane 23 -> column 18
    //
    // lane 24 -> column 6
    // lane 25 -> column 8
    // lane 26 -> column 10
    // lane 27 -> column 12
    // lane 28 -> column 14
    // lane 29 -> column 16
    // lane 30 -> column 18
    // lane 31 -> column 20
    //
    // So, the offset is: lane/8*2 + (lane&0x7)*2.
    int kSmemImgCol = kLane / 8 * 2 + (kLane & 0x7) * Conv2dStride::kV;

    // Compute the offset in SMEM for each thread to read the error. The different warps read the
    // data in the exact same way. Using LDSM.16.MT88.4.
    // Each column is 4x fp16s. We distribute the work as:
    //
    // lane 0 -> slice 0, column 0
    // lane 1 -> slice 0, column 2
    // lane 2 -> slice 0, column 4
    // lane 3 -> slice 0, column 6
    // lane 4 -> slice 0, column 8
    // lane 5 -> slice 0, column 10
    // lane 6 -> slice 0, column 12
    // lane 7 -> slice 0, column 14
    //
    // So, the offset is: lane*2.
    int smem_err_col = kLane % 8 * 2;
    int smem_err_slice = kLane / 8;

    // Where to store the data into shared memory.
    int smem_img_write_row = kTidxDivThreadsPerRow + kFltR;
    // Reinitialize the shared memory pointer.
    smem_img_write = reinterpret_cast<ElementA *>(&smem_img_[kTidxModThreadsPerRow * 4]);
    // The row position where to start reading from in SMEM.
    int smem_img_row = kWarp;

    // Clear the accumulators.
    Array<float, 4> acc[kFltSPerThread][FltKPerCTA / 8];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kFltSPerThread; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < FltKPerCTA / 8; ++j) {
        acc[i][j].clear();
      }
    }

    // Preload
    // The offset in the image buffer.
    const LongIndex kImgOffset =
        smem_img_row * kImgW * kFltC;

    LdsmOperandA A[2];
    LdsmA<kFltR, kFltS> ldsm_a;
    ldsm_a(&A[0], &smem_img_[kImgOffset], kSmemImgCol);

    LdsmOperandB B[2];
    LdsmB<kFltK> ldsm_b;
    ldsm_b(&B[0], &smem_err_[0][smem_err_slice * kSmemErrFp16sPerSlice + smem_err_col * 4], kSmemErrFp16sPerSlice);

    // Outer-loop computes different output rows.
    CUTLASS_GEMM_LOOP
    for (int p = kPBeg; p < kPEnd; p += 1, ldg_p += 1, h += Conv2dStride::kU) {
      JETFIRE_MAC_LOOP_HEADER // {$nv-internal-release}

      int kDeltaP = p - kPBeg;

      // Do the math.
      CUTLASS_PRAGMA_UNROLL
      for (int qi = 0; qi < kErrQ / 8; ++qi) {

        jetfire::ifence(); // {$nv-internal-release}

        if (qi == kErrQ / 8 - 2) {
          // STS

          // Make sure the data in SMEM is read.
          __syncthreads();

          // Store the error row to SMEM.
          wgrad_sts_b<kFltK, kFltR, kFltS>(threadIdx.x, kErrQ, kSmemErrFp16sPerSlice, kKPerSlice, 0, smem_err_[(kDeltaP + 1) & 1], fetch_err);

          jetfire::ifence(); // {$nv-internal-release}

          // Store the image rows to SMEM.
          wgrad_sts_a<kFltR, kFltS>(kThreadsPerRow, kTidxModThreadsPerRow, kImgW, &smem_img_[smem_img_write_row * kImgW * 4], fetch_img);

          jetfire::ifence(); // {$nv-internal-release}

          // Make sure we write the data into SMEM.
          __syncthreads();

          wgrad_ldg_b<kFltK, kFltR, kFltS>(threadIdx.x, ldg_p, kPEnd, kQBeg, kQEnd, kCtaK, FltKPerCTA, ref_B.stride(0), reinterpret_cast<uint16_t const*>(gmem_err), fetch_err);

          jetfire::ifence(); // {$nv-internal-release}

          for (int i = 0; i < kImgLDG32Num; ++i) {
            fetch_img[i] = 0;
          }

          wgrad_ldg_a<kFltR, kFltS>(kThreadsPerRow, kTidxModThreadsPerRow, h, w, kHEnd, kWEnd, ref_A.stride(0), reinterpret_cast<uint16_t const*>(gmem_img), fetch_img);
        }
        if (qi == kErrQ / 8 - 1) {
          // Move to the next row in SMEM for reads.
          smem_img_row = (smem_img_row >= kImgH - Conv2dStride::kU) ? smem_img_row + Conv2dStride::kU - kImgH
                                                                  : (smem_img_row + Conv2dStride::kU);

          // Update the image row.
          smem_img_write_row = smem_img_write_row >= kImgH - Conv2dStride::kU
                                  ? smem_img_write_row + Conv2dStride::kU - kImgH
                                  : smem_img_write_row + Conv2dStride::kU;

          // Move the pointers.
          gmem_err += ref_B.stride(1);
          gmem_img += Conv2dStride::kU * ref_A.stride(1);
        }

        jetfire::ifence(); // {$nv-internal-release}

        // The offset in the image buffer.
        LongIndex kImgOffset =
            smem_img_row * kImgW * kFltC + ((qi + 1) & (kErrQ / 8 - 1)) * Conv2dStride::kV * 8 * kFltC;
        // The offset in the error buffer
        LongIndex kErrOffset = ((qi + 1) & (kErrQ / 8 - 1)) * 8 * kKPerSlice;

        int kSmemErrIdx = (qi == kErrQ / 8 - 1) ? ((kDeltaP + 1) & 1) : (kDeltaP & 1);

        ldsm_a(&A[(qi + 1) & 1], &smem_img_[kImgOffset], kSmemImgCol);
        ldsm_b(&B[(qi + 1) & 1], &smem_err_[kSmemErrIdx][smem_err_slice * kSmemErrFp16sPerSlice + smem_err_col * 4 + kErrOffset], kSmemErrFp16sPerSlice);

        jetfire::ifence(); // {$nv-internal-release}

  #if (__CUDA_ARCH__ >= 900) && (CUDACC_VERSION == 118)
        // WAR the compiler bug https://nvbugs/3805670 in CUDA 11.8 for k16c4r3s3 first layer wgrad
        if (kFltK == 16 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
            __syncwarp();
        }
  #endif

        // Do the HMMA.
        const MmaOperandA *ptr_A = reinterpret_cast<const MmaOperandA*>(&A[qi & 1]);
        const MmaOperandB *ptr_B = reinterpret_cast<const MmaOperandB*>(&B[qi & 1]);

        CUTLASS_PRAGMA_UNROLL
          for (int si = 0; si < kFltSPerThread; ++si) {
            CUTLASS_PRAGMA_UNROLL
              for (int ki = 0; ki < FltKPerCTA / 8; ++ki) {
                MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&acc[si][ki]);
                mma(ptr_D[0], ptr_A[si], ptr_B[ki], ptr_D[0]);
              }
          }

        jetfire::warp_switch(); // {$nv-internal-release}
      } // (q)
    }  // (p)

    __syncthreads();

    //
    // OUTPUT
    //

    // The buffer to reorder the data.
    auto& smem_flt_ = shared_storage.smem_flt_;

    // Determine the SMEM destination for each thread. On the 1st row of shared memory we store
    // the data as:
    //
    // Flt: [k=0,c=0] [k=1,c=0] [k=0,c=1] [k=1,c=1] [k=0,c=2] [k=1,c=2] [k=0,c=3] [k=1,c=3] [k=2,c=0] [k=3,c=0] ...
    //
    // Each column is 2xfp16s
    //
    // lane  0 -> row = 0, col = 0  - lane  8 -> row = 0, col = 2  - lane 16 -> row = 1, col = 0 ...
    // lane  1 -> row = 0, col = 4  - lane  9 -> row = 0, col = 6  - lane 17 -> row = 1
    // lane  2 -> row = 0, col = 8  - lane 10 -> row = 0, col = 10 - lane 18 -> row = 1
    // lane  3 -> row = 0, col = 12 - lane 11 -> row = 0, col = 14 - lane 19 -> row = 1
    // lane  4 -> row = 0, col = 1  - lane 12 -> row = 0, col = 3  - lane 20 -> row = 1
    // lane  5 -> row = 0, col = 5  - lane 13 -> row = 0, col = 7  - lane 21 -> row = 1
    // lane  6 -> row = 0, col = 9  - lane 14 -> row = 0, col = 11 - lane 22 -> row = 1
    // lane  7 -> row = 0, col = 13 - lane 15 -> row = 0, col = 15 - lane 23 -> row = 1

    // Compute the row/col for a warp.
    int kSmemFltRow = (kLane & 0x10) / 16;
    int kSmemFltCol = (kLane & 0x03) * 4 + (kLane & 0x0f) / 4;

    // Store the data to shared memory.
    CUTLASS_PRAGMA_UNROLL
    for (int si = 0; si < kFltSPerThread; ++si) {
      CUTLASS_PRAGMA_UNROLL
      for (int ki = 0; ki < FltKPerCTA / 8; ++ki) {
        // Convert the coefficients of a same row.
        uint32_t s0 = float2_to_half2(acc[si][ki][0], acc[si][ki][1]);
        uint32_t s1 = float2_to_half2(acc[si][ki][2], acc[si][ki][3]);

        // Coefficients for s=0,s=1.
        uint16_t *addr_0 = nullptr, *addr_1 = nullptr;

        addr_0 = &smem_flt_[kWarp][kSmemFltRow + 4 * si][32 * ki + 2 * kSmemFltCol];
        addr_1 = &smem_flt_[kWarp][kSmemFltRow + 4 * si + 2][32 * ki + 2 * kSmemFltCol];

        if (kSmemFltRow + 4 * si < kFltS) {
            reinterpret_cast<uint32_t *>(addr_0)[0] = s0;
        }
        if (kSmemFltRow + 4 * si + 2 < kFltS) {
            reinterpret_cast<uint32_t *>(addr_1)[0] = s1;
        }
      }
    }

    // We work ad the warp level so sync inside the warp.
    __syncwarp();

    // The number of CTAs in the X dimension working on the same K values.
    int kGridDimX = gridDim.x / kCtasPerK;
    // The global cta id.
    int kCtaId = blockIdx.z * gridDim.y * kGridDimX + blockIdx.y * kGridDimX + kCtaQ;
    // Is it the last CTA?
    int kIsLastCta = kCtaId == kNumCtas - 1;
    // The corresponding lock (when we have multiple locks).
    int lock_id = kCtaK * kNumLocks + kCtaId % kNumLocks;
    // Acquire the lock. One warp at a time.
    int *gmem_lock = &gmem_locks[lock_id * WarpsPerCTA + kWarp];
    // The number of retired CTAs for a given warp.
    int *gmem_retired_ctas_ptr = &gmem_retired_ctas[kCtaK * WarpsPerCTA + kWarp];

    // Make sure the lock is ready.
    int kExpected = kCtaId / kNumLocks;
    int found = 0;
    while (__shfl_sync(0xffffffff, found, 0) != kExpected) {
      if (kLane == 0) {
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(found) : "l"(gmem_lock));
      }
    }

    // The number of coefficients loaded per step.
    constexpr int kFltRowsPerStep = kThreadsPerWarp / kThreadsPerWarp;
    // The number of steps
    constexpr int kFltSteps = (kFltS + kFltRowsPerStep - 1) / kFltRowsPerStep;

    // Read the different elements from SMEM. Each thread owns 4x2 values.
    Array<ElementC, 8> curr[kFltSteps];
    for (int k = 0; k < kFltSteps; ++k) {
      int row = k * kFltRowsPerStep;
      if (row < kFltS && kLane < FltKPerCTA / 2) {
        curr[k] =
            reinterpret_cast<const Array<ElementC, 8> *>(&smem_flt_[kWarp][row][8 * kLane])[0];
      }
    }

    // The last CTA performs the very final reduction. The 1st CTA simply writes. Others do a
    // step of the reduction.
    int num_steps = 0;
    if (kNumLocks > 1 && kIsLastCta) {
      found = 0;
      while (__shfl_sync(0xffffffff, found, 0) != kNumCtas - 1) {
        if (kLane == 0) {
          asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(found) : "l"(gmem_retired_ctas_ptr));
        }
      }
      num_steps = kNumLocks;
    } else if (kExpected > 0) {
      num_steps = 1;
    }

    // The very last CTA outputs to the main location.
    if (kIsLastCta) {
      lock_id = kCtaK * kNumLocks;
    }

    // The reduction buffer.
    int kRedId = lock_id * WarpsPerCTA + kWarp;
    // Assemble the pointer in GMEM.
    uint16_t *gmem_red_ptr = &gmem_red[kRedId * kFltS * FltKPerCTA * kFltC + kLane * 8];

    // Perform the reduction steps. Each thread owns 4x2 values.
    for (int i = 0; i < num_steps; ++i) {
      // Load the old values.
      Array<ElementC, 8> old[kFltSteps];
      for (int k = 0; k < kFltSteps; ++k) {
        int kRow = k * kFltRowsPerStep;
        int kOffset = i * kFltR * kFltS * FltKPerCTA * kFltC + kRow * FltKPerCTA * kFltC;
        int *old_ptr = reinterpret_cast<int *>(&old[k]);
        if (kRow < kFltS && kLane * 2 < kFltK) {
          asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                      : "=r"(old_ptr[0]), "=r"(old_ptr[1]), "=r"(old_ptr[2]), "=r"(old_ptr[3])
                      : "l"(&gmem_red_ptr[kOffset]));
        }
      }

      // Update the current sum.
      plus<Array<ElementC, 8>> op;
      for (int j = 0; j < kFltSteps; ++j) {
        curr[j] = op(curr[j], old[j]);
      }
    }

    // The last CTA must reorganize the data to write the final result. USE PRMT and write 2x int4.
    if (kIsLastCta) {
      // Is it a valid K?
      bool kIsValidK = kCtaK * FltKPerCTA + kLane * 2 < kFltK;
      // The location to write to.
      ElementC *gmem_flt_0 = &ref_C.data()[(kCtaK * FltKPerCTA + kLane * 2) * kFltR * kFltS * kFltC +
                                          kWarp * kFltS * kFltC];
      ElementC *gmem_flt_1 =
          &ref_C.data()[(kCtaK * FltKPerCTA + kLane * 2 + 1) * kFltR * kFltS * kFltC +
                        kWarp * kFltS * kFltC];

      // Dump the outputs.
      for (int k = 0; k < kFltSteps; ++k) {
        int4 flt;
        int *curr_ptr = reinterpret_cast<int *>(&curr[k]);
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;"
                    : "=r"(flt.x)
                    : "r"(curr_ptr[0]), "r"(curr_ptr[1]));
        asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(flt.y) : "r"(curr_ptr[2]), "r"(curr_ptr[3]));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;"
                    : "=r"(flt.z)
                    : "r"(curr_ptr[0]), "r"(curr_ptr[1]));
        asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(flt.w) : "r"(curr_ptr[2]), "r"(curr_ptr[3]));

        if (k * kFltRowsPerStep < kFltS && kIsValidK) {
          Array<half_t, 4> *flt_ptr = reinterpret_cast<Array<half_t, 4> *>(&flt);
          cutlass::arch::global_store<Array<half_t, 4>, sizeof(uint2)>(flt_ptr[0], &gmem_flt_0[k * kFltRowsPerStep * kFltC], true);
          cutlass::arch::global_store<Array<half_t, 4>, sizeof(uint2)>(flt_ptr[1], &gmem_flt_1[k * kFltRowsPerStep * kFltC], true);
        }
      }
    } else {
      for (int k = 0; k < kFltSteps; ++k) {
        if (k * kFltRowsPerStep < kFltS && kLane * 2 < FltKPerCTA) {
          cutlass::arch::global_store<Array<ElementC, 8>, sizeof(uint4)>(curr[k],
              &gmem_red_ptr[k * kFltRowsPerStep * FltKPerCTA * kFltC], true);
        }
      }
    }

    // Make sure all threads are done issueing.
    __syncwarp();

    // Update the lock.
    if (kLane == 0) {
      // Before we update the lock, we want to guarantee that all writes are issued and visible.
      __threadfence();

      // That's the case so we can update the lock and quit.
      atomicAdd(gmem_lock, 1);
      // That's the sum of CTAs that are done.
      atomicAdd(gmem_retired_ctas_ptr, 1);
    }
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
