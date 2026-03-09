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
    \brief Emulation of matrix multiply-accumulate operators for SM89
*/


#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ >= 900)
#    if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
#      define CUDA_PTX_FP8_CVT_ENABLED 1
#    endif
#  elif (__CUDA_ARCH__ == 890)
#    if (__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1))
#      define CUDA_PTX_FP8_CVT_ENABLED 1
#    endif
#  endif
#endif


namespace cutlass {
namespace arch {
namespace emu {
namespace device {
namespace detail {

// Perform the underlying HMMAs for the decomposed QMMA
__device__
void compute_hmmas(uint32_t* D, uint32_t const (&A_f16)[2][4], uint32_t const (&B_f16)[2][2], uint32_t const* C) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    D[i] = C[i];
  }

  // Perform each HMMA
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A_f16[i][0]), "r"(A_f16[i][1]), "r"(A_f16[i][2]), "r"(A_f16[i][3]),
        "r"(B_f16[i][0]), "r"(B_f16[i][1]),
        "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]));
  }
}

} // namespace detail

__device__
void mma_16832_e4m3_e4m3_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)

  // Convert A
  // In QMMA, each consecutive set of 4 elements is in the same row.
  // However, in HMMA only each consecutive set of 2 elements is in the same row -- the next
  // set of 2 elements is in a different row. Thus, to ensure that the correct values are
  // being accumulated with one another, we place subsets of 2 elements from the original
  // consecutive set of 4 elements in the QMMA in the same position in each of the two
  // constituent HMMAs that will be performed.
  // For example, if we have the following elements in FP8:
  //        E0 E1 E2 E3
  // They will be placed as follows:
  //    HMMA0       HMMA1
  //    E0 E1       E2 E3
  uint32_t A_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(A_f16[0][i]), "=r"(A_f16[1][i]) : "r"(A[i]));
  }

  // Convert B
  uint32_t B_f16[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(B_f16[0][i]), "=r"(B_f16[1][i]) : "r"(B[i]));
  }

  detail::compute_hmmas(D, A_f16, B_f16, C);

#endif
}

__device__
void mma_16832_e4m3_e5m2_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)

  // Convert A
  uint32_t A_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(A_f16[0][i]), "=r"(A_f16[1][i]) : "r"(A[i]));
  }

  // Convert B
  uint32_t B_f16[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(B_f16[0][i]), "=r"(B_f16[1][i]) : "r"(B[i]));
  }

  detail::compute_hmmas(D, A_f16, B_f16, C);

#endif
}

__device__
void mma_16832_e5m2_e4m3_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)

  // Convert A
  uint32_t A_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(A_f16[0][i]), "=r"(A_f16[1][i]) : "r"(A[i]));
  }

  // Convert B
  uint32_t B_f16[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(B_f16[0][i]), "=r"(B_f16[1][i]) : "r"(B[i]));
  }

  detail::compute_hmmas(D, A_f16, B_f16, C);

#endif
}

__device__
void mma_16832_e5m2_e5m2_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C) {
#if defined(CUDA_PTX_FP8_CVT_ENABLED)

  // Convert A
  uint32_t A_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(A_f16[0][i]), "=r"(A_f16[1][i]) : "r"(A[i]));
  }

  // Convert B
  uint32_t B_f16[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(B_f16[0][i]), "=r"(B_f16[1][i]) : "r"(B[i]));
  }

  detail::compute_hmmas(D, A_f16, B_f16, C);

#endif
}

} // namespace device
} // namespace emu
} // namespace arch
} // namespace cutlass
