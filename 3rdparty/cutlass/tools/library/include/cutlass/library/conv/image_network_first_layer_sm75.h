// {$nv-internal-release file}

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/functional.h"
#include "cutlass/jetfire.h"
#include "image_network_first_layer_utils.h"

#include "cutlass_common.h"
#include "first_layer.h"

namespace cutlass_build_env {
namespace library {

struct First_layer_fprop_params {
  cutlass::Tensor4DCoord input_tensor_size;
  cutlass::Tensor4DCoord conv_filter_size;
  cutlass::Tensor4DCoord output_tensor_size;
  cutlass::Tensor4DCoord bias_tensor_size;
  TensorRef<half_t, layout::TensorNHWC> ref_A;
  TensorRef<half_t, layout::TensorNHWC> ref_B;
  TensorRef<half_t, layout::TensorNHWC> ref_C;
  TensorRef<half_t, layout::TensorNHWC> ref_bias;
  int const kParamsPadTop;
  int const kParamsPadLeft;
  half_t const kParamsAlpha;
  int const kParamsWithBias;
  int const kParamsWithRelu;
  half_t const kParamsRelu;
  half_t const kParamsUpperBound;
};

struct First_layer_wgrad_params {
  cutlass::Tensor4DCoord input_tensor_size;
  cutlass::Tensor4DCoord output_tensor_size;
  cutlass::Tensor4DCoord conv_filter_size;
  TensorRef<half_t, layout::TensorNHWC> ref_A;
  TensorRef<half_t, layout::TensorNHWC> ref_B;
  TensorRef<half_t, layout::TensorNHWC> ref_C;
  int const kParamsPadTop;
  int const kParamsPadLeft;
  int const kRowsPerCta;
  int const kNumLocks;
  int *gmem_locks;
  int *gmem_retired_ctas;
  uint16_t *gmem_red;
};

template <int kWarpsPerCTA,
          int kFltC,
          int kFltR,
          int kFltS,
          int kFltK,
          int kFltKPerCTA,
          bool kSwish = false,
          typename ConvStride = cutlass::conv::Stride<2, 2>,
          typename ConvDilation = cutlass::conv::Dilation<1, 1>,
          typename InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>>
__device__ void image_network_first_layer_hmma_fprop_impl(First_layer_fprop_params param) {

  //
  // Define
  //

  auto input_tensor_size = param.input_tensor_size;
  auto output_tensor_size = param.output_tensor_size;
  auto ref_A = param.ref_A;
  auto ref_B = param.ref_B;
  auto ref_C = param.ref_C;
  auto ref_bias = param.ref_bias;
  int const kParamsPadTop = param.kParamsPadTop;
  int const kParamsPadLeft = param.kParamsPadLeft;
  half_t const kParamsAlpha = param.kParamsAlpha;
  int const kParamsWithBias = param.kParamsWithBias;
  half_t const kParamsRelu = param.kParamsRelu;
  half_t const kParamsUpperBound = param.kParamsUpperBound;

  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementACC = float;

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
                                 ElementACC,
                                 SmemLayoutC,
                                 cutlass::arch::OpMultiplyAdd>;

  using AccessType = int2;
  using LdsmOperandA = Array<unsigned, 2>;
  using LdsmOperandB = Array<unsigned, 1>;

  using MmaOperandA = typename Mma::FragmentA;
  using MmaOperandB = typename Mma::FragmentB;
  using MmaOperandC = typename Mma::FragmentC;

  Mma mma;

  // The number of threads per CTA
  int const kThreadsPerWarp = 32;
  // The number of threads per CTA
  int const kThreadsPerCTA = kThreadsPerWarp * kWarpsPerCTA;

  // The Padding
  int const kPadW = (kFltS - 1) / 2;

  // Each warp computes a tile of PIXELSxFILTERS where PIXELS == 64 and FILTERS is in
  // {16,32,64}
  int const kOutP = kWarpsPerCTA;
  int const kOutQ = 64;

  // The size of the tile we keep in shared memory
  int const kImgH = 2 * kOutP;
  int const kImgW = 2 * kOutQ + 2 * kPadW;

  // Decompose the CTA into warps
  int const kWarp = threadIdx.x / kThreadsPerWarp;
  int const kLane = threadIdx.x % kThreadsPerWarp;

  //
  // Filter
  //

  // Shared memory to store the weights
  int const kEvenFltS = (kFltS % 2 == 0) ? kFltS : kFltS + 1;
  int const kEvenFltRS = kFltR * kEvenFltS;
  int const kEvenFltKRS = kFltKPerCTA * kEvenFltRS;
  __shared__ uint16_t smem_flt_[kEvenFltKRS * kFltC];

  // The number of K coefficients loaded per step
  int const kFltKPerStep = kThreadsPerCTA / kEvenFltRS; // k_tile_of_CTA
  // The number of steps
  int const kFltSteps = (kFltKPerCTA + kFltKPerStep - 1) / kFltKPerStep; // k_tile_num_per_CTA
  // The K coefficient loaded by that thread
  int const kLoadK = threadIdx.x / kEvenFltRS;
  // The R*S coefficient loaded by that thread
  int const kLoadR = threadIdx.x % kEvenFltRS / kEvenFltS;
  int const kLoadS = threadIdx.x % kEvenFltRS % kEvenFltS;

  // Each CTA covers kFltKPerCTA filters
  // To compute the whole filters, we need kFltK/kFltKPerCTA CTAs
  // CTAs in X dim : |Q0(kColsPerCTA)| Q1       | Q2       | ... | QN       |
  //                 |K0|K1|K2|...|KN| kKSlices | kKSlices | ... | kKSlices |
  int const kKSlices = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;
  // Which part of K that the CTA covers
  int const kKCTA = blockIdx.x % kKSlices;
  // The base K offset of current tile
  int const kBaseK = kKCTA * kFltKPerCTA;
  // The offset in global memory for the filter
  LongIndex const kGmemFltOffset = (kLoadK + kBaseK) * ref_B.stride(2) +
                                   kLoadR * ref_B.stride(1) + kLoadS * ref_B.stride(0);

  // Each thread can load one pixel (4x fp16s) with a single LDG.64
  AccessType const *gmem_flt = reinterpret_cast<AccessType const *>(ref_B.data() + kGmemFltOffset);

  // We handle 4 channels each time
  int const kCGroups = kFltC / 4;

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
      if (kLoadK < kFltKPerStep && kLoadS < kFltS && kLoadK + kFltKPerStep * i < kFltKPerCTA && kLoadK + kFltKPerStep * i < kFltK - kBaseK) {
        fetch_flt[i] = gmem_flt[i * kFltKPerStep * kFltR * kFltS * kCGroups + c];
      }
    }
    AccessType *smem_flt = reinterpret_cast<AccessType *>(
        &smem_flt_[(kLoadK * kEvenFltRS + kLoadR * kEvenFltS + kLoadS + c * kEvenFltKRS) * 4]);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kFltSteps; ++i) {
      if (kLoadK < kFltKPerStep && kLoadK + kFltKPerStep * i < kFltKPerCTA) {
        smem_flt[i * kFltKPerStep * kEvenFltRS] = fetch_flt[i];
      }
    }
  }

  //
  // Activation
  //

  int const kImgHW = kImgH * kImgW;

  __shared__ uint16_t smem_img_[kImgHW * kFltC];

  // The n index
  int const kN = blockIdx.z;

  // The P,Q coordinates
  // The beginning of the chunk
  int const kRowsPerCta = 16;
  int const kPBeg = blockIdx.y * kRowsPerCta;
  int const kQBeg = blockIdx.x / kKSlices * kOutQ;

  // The end of the chunk
  int const kPEnd = min(output_tensor_size.h(), kPBeg + kRowsPerCta);
  int const kQEnd = min(output_tensor_size.w(), kQBeg + kOutQ);

  // The H,W coordinates
  int const kHBeg = ConvStride::kU * kPBeg + kWarp - kParamsPadTop;
  int const kWBeg = ConvStride::kV * kQBeg + kLane - kParamsPadLeft;
  int const kHEnd =
      min(input_tensor_size.h(), ConvStride::kU * kPBeg + ConvStride::kU * kRowsPerCta + kFltR - 1 - kParamsPadTop);
  int const kWEnd =
      min(input_tensor_size.w(), ConvStride::kV * kQBeg + ConvStride::kV * kOutQ + kFltS - 1 - kParamsPadLeft);
  // The offset where the thread starts loading
  LongIndex const kImgInitOffset =
      kN * ref_A.stride(2) + kHBeg * ref_A.stride(1) + kWBeg * ref_A.stride(0);
  // Each thread can read one pixel (4x fp16s) with a single LDG.64 and the a warp works on a
  // row of data first then moves to the "next row" if needed.
  ElementA const *kGmemImgInit =
      reinterpret_cast<ElementA const *>(&ref_A.data()[kImgInitOffset]);

  // The number of steps to fetch a row and the different rows in the tile, resp.
  int const kImgStepsInitH = (kImgH + kWarpsPerCTA - 1) / kWarpsPerCTA;
  int const kImgStepsInitW = (kImgW + kThreadsPerWarp - 1) / kThreadsPerWarp;

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
      int const H = kHBeg + hi * kWarpsPerCTA;
      CUTLASS_PRAGMA_UNROLL
      for (int wi = 0; wi < kImgStepsInitW; ++wi) {
        int const W = kWBeg + wi * kThreadsPerWarp;
        if (H >= 0 && H < kHEnd && W >= 0 && W < kWEnd) {
          int offset = hi * kWarpsPerCTA * ref_A.stride(1) + wi * kThreadsPerWarp * ref_A.stride(0);
          fetch_img_init[hi][wi] =
              *(reinterpret_cast<AccessType const *>(&kGmemImgInit[offset]) + c);
        }
      }
    }

    // Store the data into shared memory.
    for (int hi = 0; hi < kImgStepsInitH; ++hi) {
      for (int wi = 0; wi < kImgStepsInitW; ++wi) {
        if (kWarp + hi * kWarpsPerCTA < kImgH && kLane + wi * kThreadsPerWarp < kImgW) {
          smem_img_write[hi * kWarpsPerCTA * kImgW + wi * kThreadsPerWarp] = fetch_img_init[hi][wi];
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
  int const kSmemImgCol = kLane * 8;
  // Keep the pointer live across iterations of the main loop
  int4 const *smem_img_read = reinterpret_cast<int4 const *>(&smem_img_[kSmemImgCol]);

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
  int const kSmemFltOffset = kLane * kEvenFltRS * 4;
  // The address in filter memory to read the filter from
  AccessType const *smem_flt_read =
      reinterpret_cast<AccessType const *>(&smem_flt_[kSmemFltOffset]);

  //
  // Calculate the index for the next LDG 2 rows
  //

  // In main loop we load 2 rows into SMEM per iteration
  int const kThreadsPerRow = kThreadsPerCTA / 2;

  int const kTidxDivThreadsPerRow = threadIdx.x / kThreadsPerRow;
  int const kTidxModThreadsPerRow = threadIdx.x % kThreadsPerRow;

  // H,W coordinates
  int h = ConvStride::kU * kPBeg + kImgH + kTidxDivThreadsPerRow - kParamsPadTop;
  int const w = ConvStride::kV * kQBeg + kTidxModThreadsPerRow - kParamsPadLeft;

  // The offset where the thread starts loading
  LongIndex const kGmemImgOffset = kN * ref_A.stride(2) + h * ref_A.stride(1) + w * ref_A.stride(0);

  // Reinitialize the gmem_img pointer
  ElementA const *gmem_img = reinterpret_cast<ElementA const *>(&ref_A.data()[kGmemImgOffset]);

  // Where to write tha data
  int smem_img_write_row = kTidxDivThreadsPerRow;
  // Reinitialize the shared memory pointer
  smem_img_write = reinterpret_cast<AccessType *>(&smem_img_[kTidxModThreadsPerRow * 4]);

  // The number of iterations for R/S.
  int const kStepsR = (kFltR + 1) / 2;
  int const kStepsS = kEvenFltS / 2;

  int const kGmemImgOffsetPerRStep = 2 * ref_A.stride(1);

  int const kHMMARows = kOutQ / 16;
  int const kHMMACols = kFltKPerCTA / 8;

  // Outer-loop computes 4 different output rows
  CUTLASS_PRAGMA_NO_UNROLL
  for (int p = kPBeg; p < kPEnd; p += kWarpsPerCTA) {
    Array<float, 4> acc[kHMMARows][kHMMACols];

    for (int i = 0; i < kHMMARows; ++i) {
      for (int j = 0; j < kHMMACols; ++j) {
        acc[i][j].clear();
      }
    }
    jetfire::ifence(); // {$nv-internal-release}
    // The position where we start reading from in SMEM.
    // We have one warp per row and a stride of 2 between two rows
    int smem_img_row = kWarp * ConvStride::kU;

    // The filter coefficient that we are currently reading. Rollback for every iteration.
    int smem_flt_idx = 0;

    // Perform the convolution, we decompose the kFltR rows into 2+2+2+1
    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = 0; r < kStepsR; ++r) {
      // Make sure data is in SMEM
      __syncthreads();

      // Issue the loading of the rows
      const int kImgStepsW = (kImgW + kThreadsPerRow - 1) / kThreadsPerRow;
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
          int const W = w + wi * kThreadsPerRow;
          if (h >= 0 && h < kHEnd && W >= 0 && W < kWEnd) {
            int offset = wi * kThreadsPerRow * ref_A.stride(0);
            fetch_img[wi] = *(reinterpret_cast<AccessType const *>(&gmem_img[offset]) + c);
          }
        }

        jetfire::ifence(); // {$nv-internal-release}

        smem_img_row = tmp_smem_img_row;
        smem_flt_idx = tmp_smem_flt_idx;

        int const kReadOffsetC = c * (kImgHW / 2);
        int const kWriteOffsetC = c * kImgHW;

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
              MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
              CUTLASS_PRAGMA_UNROLL
              for (int ri = 0; ri < kHMMARows; ri++) {
                MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A[ri]);
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
    __shared__ uint16_t smem_out_[kWarpsPerCTA][8][kFltKPerCTA * 2 + 4 * 4];

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
    int const kSmemQ = kLane / 4;

    // For a given warp, we have the following output in K dimension
    // We express those offsets in terms of int2 numbers
    //
    // lane 0, 4, 8, 12, 16, 20, 24, 28 -> k = {0, 1}, {8, 9}, {16, 17}, {24, 25}, ...
    // lane 1, 5, 9, 13, 17, 21, 25, 29 -> k = {2, 3}, {10, 11}, {18, 19}, {26, 27}, ...
    // lane 2, 6, 10, 14, 18, 22, 26, 30 -> k = {4, 5}, {12, 13}, {20, 21}, {28, 29}, ...
    // lane 3, 7, 11, 15, 19, 23, 27, 31 -> k = {6, 7}, {14, 15}, {22, 23}, {30, 31}, ...
    //
    // So, the offset in the K dimension is lane&0x3*2
    int const kSmemK = (kLane & 0x3) * 4;  // Pack 4 pixels ([q0k0 q0k1 q1k0 q1k1])

    // The number of threads per pixel
    int const kThreadsPerPixel = kFltKPerCTA <= 32 ? 16 : 32;

    // Decompose the warp
    int const kLaneDivTpp = kLane / kThreadsPerPixel;
    int const kLaneModTpp = kLane % kThreadsPerPixel;

    // The destination coordinates
    int const kDstP = p + kWarp;
    int const kDstQ = kQBeg + kLaneDivTpp;
    int const kDstK = kLaneModTpp;

    // The destination offset
    LongIndex const kDstOffset = kN * ref_C.stride(2) + kDstP * ref_C.stride(1) +
                                 kDstQ * ref_C.stride(0) + kDstK * 2 + kKCTA * kFltKPerCTA;

    // The destination pointer
    int *gmem_out = reinterpret_cast<int *>(&ref_C.data()[kDstOffset]);

    // The bias pointer
    uint32_t *bias_ptr = reinterpret_cast<uint32_t *>(&ref_bias.data()[kKCTA * kFltKPerCTA]);
    uint32_t bias = 0;

    // Valid check
    int const kPValid = kDstP < kPEnd;
    int const kKValid = kDstK * 2 < kFltKPerCTA && kDstK * 2 < kFltK - kBaseK;

    // Load bias
    if (kParamsWithBias && kKValid) {
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

      const uint32_t kAlpha = half_to_half2(kParamsAlpha);
      const uint32_t kReluLB = half_to_half2(kParamsRelu);
      const uint32_t kReluUB = half_to_half2(kParamsUpperBound);

      // Print the 16 pixels per warp
      for (int qii = 0; qii < 2; ++qii) {
        for (int qiii = 0; qiii < 4; ++qiii) {
          if (kFltKPerCTA > 32 * (qiii % 2)) {
            // Valid check
            int const kQValid0 = kDstQ + qi * 16 + qii * 4 + qiii + 0 < kQEnd;
            int const kQValid1 = kDstQ + qi * 16 + qii * 4 + qiii + 8 < kQEnd;

            // Load from shared memory
            int2 data;
            if (kKValid) {
              data = reinterpret_cast<int2 const *>(
                  &smem_out_[kWarp][qii * 4 + qiii + kLaneDivTpp][4 * kLaneModTpp])[0];
            }

            data.x = fma_fp16x2(kAlpha, data.x, bias);
            data.y = fma_fp16x2(kAlpha, data.y, bias);

            if (kSwish == false) {
              data.x = clippedRelu(data.x, kReluLB, kReluUB);
              data.y = clippedRelu(data.y, kReluLB, kReluUB);
            } else {
              data.x = swish(data.x);
              data.y = swish(data.y);
            }

            // Output to memory
            int const kQIdx = 16 * qi + qii * 4 + qiii;
            int const kHalfStrideK = ref_C.stride(0) / 2;
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

template <int kWarpsPerCTA,
          int kFltC,
          int kFltR,
          int kFltS,
          int kFltK,
          int kFltKPerCTA,
          bool kSwish = false,
          typename ConvStride = cutlass::conv::Stride<2, 2>,
          typename ConvDilation = cutlass::conv::Dilation<1, 1>,
          typename InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>>
__global__ __launch_bounds__(256) void image_network_first_layer_hmma_fprop_kernel(
    cutlass::Tensor4DCoord input_tensor_size,
    cutlass::Tensor4DCoord conv_filter_size,
    cutlass::Tensor4DCoord output_tensor_size,
    cutlass::Tensor4DCoord bias_tensor_size,
    TensorRef<half_t, layout::TensorNHWC> ref_A,
    TensorRef<half_t, layout::TensorNHWC> ref_B,
    TensorRef<half_t, layout::TensorNHWC> ref_C,
    TensorRef<half_t, layout::TensorNHWC> ref_bias,
    int const kParamsPadTop,
    int const kParamsPadLeft,
    half_t const kParamsAlpha,
    int const kParamsWithBias,
    int const kParamsWithRelu,
    half_t const kParamsRelu,
    half_t const kParamsUpperBound) {
  image_network_first_layer_hmma_fprop_impl<kWarpsPerCTA, kFltC, kFltR, kFltS, kFltK, kFltKPerCTA,
                                            kSwish, ConvStride, ConvDilation, InstructionShape>({
      input_tensor_size,
      conv_filter_size,
      output_tensor_size,
      bias_tensor_size,
      ref_A,
      ref_B,
      ref_C,
      ref_bias,
      kParamsPadTop,
      kParamsPadLeft,
      kParamsAlpha,
      kParamsWithBias,
      kParamsWithRelu,
      kParamsRelu,
      kParamsUpperBound
  });
}


template <int kWarpsPerCTA,
          int kFltC,
          int kFltR,
          int kFltS,
          int kFltK,
          int kFltKPerCTA,
          typename ConvStride = cutlass::conv::Stride<2, 2>,
          typename ConvDilation = cutlass::conv::Dilation<1, 1>,
          typename InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>>
__device__ void image_network_first_layer_hmma_wgrad_impl(
    First_layer_wgrad_params param) {
  //
  // Define
  //

  auto input_tensor_size = param.input_tensor_size;
  auto output_tensor_size = param.output_tensor_size;
  auto ref_A = param.ref_A;
  auto ref_B = param.ref_B;
  auto ref_C = param.ref_C;
  int const kParamsPadTop = param.kParamsPadTop;
  int const kParamsPadLeft = param.kParamsPadLeft;
  int const kRowsPerCta = param.kRowsPerCta;
  int const kNumLocks = param.kNumLocks;
  int *gmem_locks = param.gmem_locks;
  int *gmem_retired_ctas = param.gmem_retired_ctas;
  uint16_t *gmem_red = param.gmem_red;

  using LongIndex = cutlass::layout::TensorNHWC::LongIndex;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementACC = float;

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
                                 ElementACC,
                                 SmemLayoutC,
                                 cutlass::arch::OpMultiplyAdd>;

  using AccessType = int2;
  using LdsmOperandA = typename LdsmA<kFltR, kFltS>::LdsmOperand;
  using LdsmOperandB = typename LdsmB<kFltK>::LdsmOperand;

  using MmaOperandA = typename Mma::FragmentA;
  using MmaOperandB = typename Mma::FragmentB;
  using MmaOperandC = typename Mma::FragmentC;

  Mma mma;

  // The number of ctas to compute all K
  int const kCtasPerK = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;
  // The number of ctas in group of all K
  int const kNumCtas = gridDim.x * gridDim.y * gridDim.z / kCtasPerK;
  // The number of threads per warp.
  int const kThreadsPerWarp = 32;
  // The number of threads per CTA.
  int const kThreadsPerCTA = kWarpsPerCTA * kThreadsPerWarp;

  // The padding.
  int const kPadW = (kFltS - 1) / 2;

  // Each warp works on C x K coefficients for an entire row of the filter. So, for example, if
  // the filter is 7 x 7, we have 7 warps and each warp computes a 7 x C x K filter values. In
  // this kernel we load 32 pixels of the error tensor per loop iteration.

  // The main loop operates on 32 different error pixels taken from a single row.
  int const kErrQ = 32;

  // The size of the tile that we keep in shared memory.
  int const kImgH = ConvStride::kU * kFltR;
  int const kImgW = ConvStride::kV * kErrQ + 2 * kPadW;

  // The number of filter coefficients computed per thread.
  // We compute 4 x s per HMMA.1688
  int const kFltSPerThread = (kFltS + 3) / 4;

  // Decompose the CTA into warps.
  int const kWarp = threadIdx.x / kThreadsPerWarp;
  int const kLane = threadIdx.x % kThreadsPerWarp;

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
  int const kThreadsPerErrPixel = kFltKPerCTA / 8;
  // Compute the skew as described above. This skew is in number of fp16.
  int const kSmemErrSkew = 8 / kThreadsPerErrPixel * 8;
  // Compute the number of slices.
  int const kSmemErrSlices = kFltKPerCTA / 8;
  // The number of fp16 elements per slice.
  int const kSmemErrFp16sPerSlice = kErrQ * 8 + kSmemErrSkew;

  // Declare the shared memory buffer.
  __shared__ uint16_t smem_err_[2][kSmemErrSlices * kSmemErrFp16sPerSlice];

  // The n index
  int const kN = blockIdx.z;

  // The qk indices.
  int const kCtaQ = blockIdx.x / (kFltK / kFltKPerCTA);
  int const kCtaK = blockIdx.x % (kFltK / kFltKPerCTA);

  // The p,q coordinate. It is the beginning of the chunk.
  int const kPBeg = blockIdx.y * kRowsPerCta;
  int const kQBeg = kCtaQ * kErrQ;

  // The end of the chunk for p,q
  int const kPEnd = min(output_tensor_size.h(), kPBeg + kRowsPerCta);
  int const kQEnd = min(output_tensor_size.w(), kQBeg + kErrQ);

  // Track which row of errs to load from global memory
  int ldg_p = kPBeg;

  // The start offset of current err row in global memory for the error tensor.
  LongIndex const kGmemErrOffset =
      kN * ref_B.stride(2) + ldg_p * ref_B.stride(1);

  // Load the memory by groups of 8 channels.
  ElementB const *gmem_err = &ref_B.data()[kGmemErrOffset];
  int const kErrLDG32Num = (kErrQ*kFltK + kThreadsPerCTA*2 - 1)/(kThreadsPerCTA*2);
  int fetch_err[kErrLDG32Num];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kErrLDG32Num; ++i) {
    fetch_err[i] = 0;
  }

  wgrad_ldg_b<kFltK,kFltR,kFltS>(threadIdx.x,ldg_p,kPEnd,kQBeg,kQEnd,kCtaK,kFltKPerCTA,reinterpret_cast<uint16_t const*>(gmem_err),fetch_err);

  int const kKPerSlice = 8;

  // Move to the next row of err
  gmem_err += output_tensor_size.w() * output_tensor_size.c();
  ldg_p += 1;

  wgrad_sts_b<kFltK,kFltR,kFltS>(threadIdx.x,kErrQ,kSmemErrFp16sPerSlice,kKPerSlice,0,smem_err_[0],fetch_err);

  //
  // ACTIVATIONS
  //

  // Shared memory to store the activations. This buffer is used as a circular buffer with
  // 2*FLT_R different rows containing up to 64+2*PAD_W pixels each. In the main loop, we circle
  // over the rows to store the data needed to compute the convolutions.
  __shared__ uint16_t smem_img_[kImgH * kImgW * kFltC];

  // The position in the H and W dimensions. Each warp works on a different row in the prologue
  // and we take the padding into account.
  int const kHBeg = ConvStride::kU * kPBeg + kWarp - kParamsPadTop;
  int const kWBeg = ConvStride::kV * kQBeg + kLane - kParamsPadLeft;

  // The end of the chunkf for h/w.
  int const kHEnd =
      max(min(input_tensor_size.h(), ConvStride::kU * (kPBeg + kRowsPerCta - 1) + kFltR - kParamsPadTop), 0);
  int const kWEnd =
      max(min(input_tensor_size.w(), ConvStride::kV * (kQBeg + kErrQ - 1) + kFltS - kParamsPadLeft), 0);

  // The offset where the thread starts loading.
  LongIndex const kGmemImgOffsetInit =
      kN * ref_A.stride(2) + kHBeg * ref_A.stride(1) + kWBeg * ref_A.stride(0);

  // Each thread can read one pixel (4x fp16s) with a single LDG.64 and a warp works on
  // a row of data first then moves to the "next row" if needed.
  ElementA const *kGmemImgInit = &ref_A.data()[kGmemImgOffsetInit];

  // In shared memory we maintain the structure of tiles.
  ElementA *smem_img_write =
      reinterpret_cast<ElementA *>(&smem_img_[kWarp * kImgW * 4 + kLane * 4]);

  // The number of steps to fetch a row and the different rows in the tile, resp.
  int const kImgStepsInitH = (kFltR + kWarpsPerCTA - 1) / kWarpsPerCTA;
  int const kImgStepsInitW = (kImgW + kThreadsPerWarp - 1) / kThreadsPerWarp;

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
    int const kValidH = (unsigned)(kHBeg + hi * kWarpsPerCTA) < kHEnd;
    CUTLASS_PRAGMA_UNROLL
    for (int wi = 0; wi < kImgStepsInitW; ++wi) {
      int const kValidW = (unsigned)(kWBeg + wi * kThreadsPerWarp) < kWEnd;
      if (kValidH && kValidW) {
        LongIndex const kOffset =
            hi * kWarpsPerCTA * ref_A.stride(1) + wi * kThreadsPerWarp * ref_A.stride(0);
        fetch_img_init[hi][wi] = reinterpret_cast<AccessType const *>(&kGmemImgInit[kOffset])[0];
      }
    }
  }

  // Store the data into shared memory.
  for (int hi = 0; hi < kImgStepsInitH; ++hi) {
    for (int wi = 0; wi < kImgStepsInitW; ++wi) {
      if (kWarp + hi * kWarpsPerCTA < kImgH && kLane + wi * kThreadsPerWarp < kImgW) {
        LongIndex const kOffset = hi * kWarpsPerCTA * kImgW * 4 + wi * kThreadsPerWarp * 4;
        reinterpret_cast<AccessType *>(&smem_img_write[kOffset])[0] = fetch_img_init[hi][wi];
      }
    }
  }

  // Make sure the data is in shared memory
  __syncthreads();

  // Prefetch

  // Issue the loading of the error row.
  wgrad_ldg_b<kFltK,kFltR,kFltS>(threadIdx.x,ldg_p,kPEnd,kQBeg,kQEnd,kCtaK,kFltKPerCTA,reinterpret_cast<uint16_t const*>(gmem_err),fetch_err);

  jetfire::ifence(); // {$nv-internal-release}

  // We can now determine how many threads to assign per row to load the activations in the main
  // loop.
  int const kThreadsPerRow = kThreadsPerCTA / ConvStride::kU;
  int const kImgLDG32Num = (kImgW * 4 + kThreadsPerRow * 2 - 1) / (kThreadsPerRow * 2);

  // Decompose the thread index to fetch rows per iteration.
  int const kTidxDivThreadsPerRow = threadIdx.x / kThreadsPerRow;
  int const kTidxModThreadsPerRow = threadIdx.x % kThreadsPerRow;

  // In the main loop we fetch two rows per iteration so we redistribute the threads to do so.
  int h = ConvStride::kU * kPBeg + kTidxDivThreadsPerRow - kParamsPadTop + kFltR;
  int const w = ConvStride::kV * kQBeg - kParamsPadLeft; // The first w in this row

  // The offset where the threads starts loading.
  LongIndex const kGmemImgOffset = kN * ref_A.stride(2) + h * ref_A.stride(1);
  // Reinitialize the gmem_img pointer
  ElementA const *gmem_img = &ref_A.data()[kGmemImgOffset];

  int fetch_img[kImgLDG32Num];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kImgLDG32Num; ++i) {
    fetch_img[i] = 0;
  }

  // Issue the loading of the image rows.
  wgrad_ldg_a<kFltR,kFltS>(kThreadsPerRow,kTidxModThreadsPerRow,h,w,kHEnd,kWEnd,reinterpret_cast<uint16_t const*>(gmem_img),fetch_img);

  // Move the pointers.
  gmem_err += output_tensor_size.w() * output_tensor_size.c();
  ldg_p += 1;
  gmem_img += ConvStride::kU * input_tensor_size.w() * input_tensor_size.c();
  h += ConvStride::kU;

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
  int const kSmemImgCol = kLane / 8 * 2 + (kLane & 0x7) * ConvStride::kV;

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
  int const smem_err_col = kLane % 8 * 2;
  int const smem_err_slice = kLane / 8;

  // Where to store the data into shared memory.
  int smem_img_write_row = kTidxDivThreadsPerRow + kFltR;
  // Reinitialize the shared memory pointer.
  smem_img_write = reinterpret_cast<ElementA *>(&smem_img_[kTidxModThreadsPerRow * 4]);
  // The row position where to start reading from in SMEM.
  int smem_img_row = kWarp;

  // Clear the accumulators.
  Array<float, 4> acc[kFltSPerThread][kFltKPerCTA / 8];
  for (int i = 0; i < kFltSPerThread; ++i) {
    for (int j = 0; j < kFltKPerCTA / 8; ++j) {
      acc[i][j].clear();
    }
  }

  // Preload
  // The offset in the image buffer.
  LongIndex const kImgOffset =
      smem_img_row * kImgW * kFltC;

  LdsmOperandA A[2];
  LdsmA<kFltR, kFltS> ldsm_a;
  ldsm_a(&A[0], &smem_img_[kImgOffset], kSmemImgCol);

  LdsmOperandB B[2];
  LdsmB<kFltK> ldsm_b;
  ldsm_b(&B[0], &smem_err_[0][smem_err_slice * kSmemErrFp16sPerSlice + smem_err_col * 4], kSmemErrFp16sPerSlice);

  // Outer-loop computes different output rows.
  CUTLASS_GEMM_LOOP
  for (int p = kPBeg; p < kPEnd; p += 1, ldg_p += 1, h += ConvStride::kU) {
    JETFIRE_MAC_LOOP_HEADER // {$nv-internal-release}

    int const kDeltaP = p - kPBeg;

    // Do the math.
    CUTLASS_PRAGMA_UNROLL
    for (int qi = 0; qi < kErrQ / 8; ++qi) {

      jetfire::ifence(); // {$nv-internal-release}

      if (qi == kErrQ / 8 - 2) {
        // STS

        // Make sure the data in SMEM is read.
        __syncthreads();

        // Store the error row to SMEM.
        wgrad_sts_b<kFltK,kFltR,kFltS>(threadIdx.x,kErrQ,kSmemErrFp16sPerSlice,kKPerSlice,0,smem_err_[(kDeltaP+1)&1],fetch_err);

        jetfire::ifence(); // {$nv-internal-release}

        // Store the image rows to SMEM.
        wgrad_sts_a<kFltR,kFltS>(kThreadsPerRow,kTidxModThreadsPerRow,kImgW,&smem_img_[smem_img_write_row * kImgW * 4],fetch_img);

        jetfire::ifence(); // {$nv-internal-release}

        // Make sure we write the data into SMEM.
        __syncthreads();

        wgrad_ldg_b<kFltK,kFltR,kFltS>(threadIdx.x,ldg_p,kPEnd,kQBeg,kQEnd,kCtaK,kFltKPerCTA,reinterpret_cast<uint16_t const*>(gmem_err),fetch_err);

        jetfire::ifence(); // {$nv-internal-release}

        for (int i = 0; i < kImgLDG32Num; ++i) {
          fetch_img[i] = 0;
        }

        wgrad_ldg_a<kFltR,kFltS>(kThreadsPerRow,kTidxModThreadsPerRow,h,w,kHEnd,kWEnd,reinterpret_cast<uint16_t const*>(gmem_img),fetch_img);


      }
      if (qi == kErrQ / 8 - 1) {
        // Move to the next row in SMEM for reads.
        smem_img_row = (smem_img_row >= kImgH - ConvStride::kU) ? smem_img_row + ConvStride::kU - kImgH
                                                                : (smem_img_row + ConvStride::kU);

        // Update the image row.
        smem_img_write_row = smem_img_write_row >= kImgH - ConvStride::kU
                                 ? smem_img_write_row + ConvStride::kU - kImgH
                                 : smem_img_write_row + ConvStride::kU;

        // Move the pointers.
        gmem_err += output_tensor_size.w() * output_tensor_size.c();
        gmem_img += ConvStride::kU * input_tensor_size.w() * input_tensor_size.c();
      }

      jetfire::ifence(); // {$nv-internal-release}

      // The offset in the image buffer.
      LongIndex const kImgOffset =
          smem_img_row * kImgW * kFltC + ((qi+1) & (kErrQ / 8 - 1)) * ConvStride::kV * 8 * kFltC;
      // The offset in the error buffer
      LongIndex const kErrOffset = ((qi+1) & (kErrQ / 8 - 1)) * 8 * kKPerSlice;

      int const kSmemErrIdx = (qi == kErrQ / 8 - 1) ? ((kDeltaP + 1) & 1) : (kDeltaP & 1);

      ldsm_a(&A[(qi+1)&1], &smem_img_[kImgOffset], kSmemImgCol);
      ldsm_b(&B[(qi+1)&1], &smem_err_[kSmemErrIdx][smem_err_slice * kSmemErrFp16sPerSlice + smem_err_col * 4 + kErrOffset], kSmemErrFp16sPerSlice);

      jetfire::ifence(); // {$nv-internal-release}

#if (__CUDA_ARCH__ >= 900) && (CUDACC_VERSION == 118)
      // WAR the compiler bug https://nvbugs/3805670 in CUDA 11.8 for k16c4r3s3 first layer wgrad
      if (kFltK == 16 && kFltR == 3 && kFltS == 3 && kFltC == 4) {
          __syncwarp();
      }
#endif

      // Do the HMMA.
      MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A[qi&1]);
      MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B[qi&1]);

      CUTLASS_PRAGMA_UNROLL
        for (int si = 0; si < kFltSPerThread; ++si) {
          CUTLASS_PRAGMA_UNROLL
            for (int ki = 0; ki < kFltKPerCTA / 8; ++ki) {
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
  __shared__ uint16_t smem_flt_[kFltR][kFltS][4 * kFltKPerCTA + 16];

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
  int const kSmemFltRow = (kLane & 0x10) / 16;
  int const kSmemFltCol = (kLane & 0x03) * 4 + (kLane & 0x0f) / 4;

  // Store the data to shared memory.
  for (int si = 0; si < kFltSPerThread; ++si) {
    for (int ki = 0; ki < kFltKPerCTA / 8; ++ki) {
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
  int const kGridDimX = gridDim.x / kCtasPerK;
  // The global cta id.
  int const kCtaId = blockIdx.z * gridDim.y * kGridDimX + blockIdx.y * kGridDimX + kCtaQ;
  // Is it the last CTA?
  int const kIsLastCta = kCtaId == kNumCtas - 1;
  // The corresponding lock (when we have multiple locks).
  int lock_id = kCtaK * kNumLocks + kCtaId % kNumLocks;
  // Acquire the lock. One warp at a time.
  int *gmem_lock = &gmem_locks[lock_id * kWarpsPerCTA + kWarp];
  // The number of retired CTAs for a given warp.
  int *gmem_retired_ctas_ptr = &gmem_retired_ctas[kCtaK * kWarpsPerCTA + kWarp];

  // Make sure the lock is ready.
  int const kExpected = kCtaId / kNumLocks;
  int found = 0;
  while (__shfl_sync(0xffffffff, found, 0) != kExpected) {
    if (kLane == 0) {
      asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(found) : "l"(gmem_lock));
    }
  }

  // The number of coefficients loaded per step.
  int const kFltRowsPerStep = kThreadsPerWarp / kThreadsPerWarp;
  // The number of steps
  int const kFltSteps = (kFltS + kFltRowsPerStep - 1) / kFltRowsPerStep;

  // Read the different elements from SMEM. Each thread owns 4x2 values.
  Array<ElementC, 8> curr[kFltSteps];
  for (int k = 0; k < kFltSteps; ++k) {
    const int row = k * kFltRowsPerStep;
    if (row < kFltS && kLane < kFltKPerCTA / 2) {
      curr[k] =
          reinterpret_cast<Array<ElementC, 8> const *>(&smem_flt_[kWarp][row][8 * kLane])[0];
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
  int const kRedId = lock_id * kWarpsPerCTA + kWarp;
  // Assemble the pointer in GMEM.
  uint16_t *gmem_red_ptr = &gmem_red[kRedId * kFltS * kFltKPerCTA * kFltC + kLane * 8];

  // Perform the reduction steps. Each thread owns 4x2 values.
  for (int i = 0; i < num_steps; ++i) {
    // Load the old values.
    Array<ElementC, 8> old[kFltSteps];
    for (int k = 0; k < kFltSteps; ++k) {
      int const kRow = k * kFltRowsPerStep;
      int const kOffset = i * kFltR * kFltS * kFltKPerCTA * kFltC + kRow * kFltKPerCTA * kFltC;
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
    int const kIsValidK = kCtaK * kFltKPerCTA + kLane * 2 < kFltK;
    // The location to write to.
    ElementC *gmem_flt_0 = &ref_C.data()[(kCtaK * kFltKPerCTA + kLane * 2) * kFltR * kFltS * kFltC +
                                         kWarp * kFltS * kFltC];
    ElementC *gmem_flt_1 =
        &ref_C.data()[(kCtaK * kFltKPerCTA + kLane * 2 + 1) * kFltR * kFltS * kFltC +
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
      if (k * kFltRowsPerStep < kFltS && kLane * 2 < kFltKPerCTA) {
        cutlass::arch::global_store<Array<ElementC, 8>, sizeof(uint4)>(curr[k],
            &gmem_red_ptr[k * kFltRowsPerStep * kFltKPerCTA * kFltC], true);
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

template <int kWarpsPerCTA,
          int kFltC,
          int kFltR,
          int kFltS,
          int kFltK,
          int kFltKPerCTA,
          typename ConvStride = cutlass::conv::Stride<2, 2>,
          typename ConvDilation = cutlass::conv::Dilation<1, 1>,
          typename InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>>
__global__ __launch_bounds__(256, 2) void image_network_first_layer_hmma_wgrad_kernel(
    cutlass::Tensor4DCoord input_tensor_size,
    cutlass::Tensor4DCoord output_tensor_size,
    cutlass::Tensor4DCoord conv_filter_size,
    TensorRef<half_t, layout::TensorNHWC> ref_A,
    TensorRef<half_t, layout::TensorNHWC> ref_B,
    TensorRef<half_t, layout::TensorNHWC> ref_C,
    int const kParamsPadTop,
    int const kParamsPadLeft,
    int const kRowsPerCta,
    int const kNumLocks,
    int *gmem_locks,
    int *gmem_retired_ctas,
    uint16_t *gmem_red) {
  image_network_first_layer_hmma_wgrad_impl<kWarpsPerCTA, kFltC, kFltR, kFltS, kFltK, kFltKPerCTA,
                                            ConvStride, ConvDilation, InstructionShape>({
    input_tensor_size,
    output_tensor_size,
    conv_filter_size,
    ref_A,
    ref_B,
    ref_C,
    kParamsPadTop,
    kParamsPadLeft,
    kRowsPerCta,
    kNumLocks,
    gmem_locks,
    gmem_retired_ctas,
    gmem_red
  });
}

void image_network_first_layer_hmma_fprop(cutlass::Tensor4DCoord input_tensor_size,
                       cutlass::Tensor4DCoord conv_filter_size,
                       cutlass::Tensor4DCoord output_tensor_size,
                       cutlass::Tensor4DCoord bias_tensor_size,
                       TensorRef<half_t, layout::TensorNHWC> ref_A,
                       TensorRef<half_t, layout::TensorNHWC> ref_B,
                       TensorRef<half_t, layout::TensorNHWC> ref_C,
                       TensorRef<half_t, layout::TensorNHWC> ref_bias);

cudaError_t image_network_first_layer_hmma_wgrad(cutlass::Tensor4DCoord input_tensor_size,
                              cutlass::Tensor4DCoord output_tensor_size,
                              cutlass::Tensor4DCoord conv_filter_size,
                              TensorRef<half_t, layout::TensorNHWC> ref_A,
                              TensorRef<half_t, layout::TensorNHWC> ref_B,
                              TensorRef<half_t, layout::TensorNHWC> ref_C);

}  // End namespace library
}  // End namespace cutlass
