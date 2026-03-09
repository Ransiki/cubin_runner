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
    \brief Emulation of sparse matrix multiply-accumulate operators for SM89
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

// Convert the sparse metadata used for the sparse QMMA into that needed
// by the sparse HMMAs.
//
// Elements in operand A held by each thread will be partitioned into
// the two underlying HMMAs in groups of two. For example, suppose T0 holds
//      00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
// Then, the elements held by T0 for each of the two HMMAs will be:
//      HMMA 0:  00  01  04  05  08  09  12  13
//      HMMA 1:  02  03  06  07  10  11  14  15
// Each group of two elements represents two nonzero elements among four
// consecutive elements in operand A, with the positions of the nonzero
// entries indicated by a group of two two-bit metadata values in E.
// Thus, we must also partition E among the two underlying HMMAs.
//
// Representation of E in QMMA
// ---------------------------
// For sparse 16864 MMAs: "All threads within a group of four consecutive
// threads contribute the sparsity metadata." Each thread holds 32 bits
// of sparse metadata with every 2 bits representing the location of a
// nonzero value.
//
// The following shows the E metadata held by each of the 4 threads in a quad.
// Each element of metadata is annotated as TX:Y to indicate that the metadata corresponds
// to thread X's Yth element of operand A.
//
// E[T0] = T0:00 T0:01 T0:02 T0:03 T1:00 T1:01 T1:02 T1:03 T2:00 T2:01 T2:02 T2:03 T3:00 T3:01 T3:02 T3:03
// E[T1] = T0:04 T0:05 T0:06 T0:07 T1:04 T1:05 T1:06 T1:07 T2:04 T2:05 T2:06 T2:07 T3:04 T3:05 T3:06 T3:07
// E[T2] = T0:08 T0:09 T0:10 T0:11 T1:08 T1:09 T1:10 T1:11 T2:08 T2:09 T2:10 T2:11 T3:08 T3:09 T3:10 T3:11
// E[T3] = T0:12 T0:13 T0:14 T0:15 T1:12 T1:13 T1:14 T1:15 T2:12 T2:13 T2:14 T2:15 T3:12 T3:13 T3:14 T3:15
//
// Representation of E in HMMA
// ---------------------------
// For sparse 16832 MMAs: "A thread-pair within a group of four consecutive threads contributes the
// sparsity metadata." Each thread holds 32 bits of sparse metadata with every 2 bits representing
// the location of a nonzero value. However, since the total number of nonzero elements in operand A
// for a quad of threads is only 32, we do not need all four threads in the quad to contribute metadata
// values in each call to the HMMA. Thus, a "sparsity selector" field is used. If the sparsity selector
// is 0, then metadata is sourced from the E values of threads 0 and 1 in the quad. If the sparsity selector
// is 1, then metadata is sourced from the E values of threads 1 and 2 in the quad.
//
// The following shows the E metadata held by the two participating threads in the quad.
// Each element of metadata is annotated as TX:Y to indicate that the metadata corresponds
// to thread X's Yth element of operand A.
//
// E[TA]  = T0:00 T0:01 T1:00 T1:01 T2:00 T2:01 T3:00 T3:01 T0:02 T0:03 T1:02 T1:03 T2:02 T2:03 T3:02 T3:03
// E[TB]  = T0:04 T0:05 T1:04 T1:05 T2:04 T2:05 T3:04 T3:05 T0:06 T0:07 T1:06 T1:07 T2:06 T2:07 T3:06 T3:07
// where:
//   TA = T0 if sparsity selector is 0 else T2
//   TB = T1 if sparsity selector is 0 else T3
//
// Converting QMMA E to Es needed for underlying HMMAs
// ---------------------------------------------------
// We decompose the QMMA E metadata into two sets that will be used with the two different sparsity selector
// values. Thus, threads 0 and 1 in a given quad will contain metadata for HMMA 0 and threads 2 and 3 will
// contain metadata for HMMA 1.
//
// We perform this conversion in two steps:
//   1. Each thread reorganizes its metadata to be of the form {E_HMMA0, E_HMMA1}.
//      Even-numbered groups of 4 bits belong to E_HMMA0, while odd-numbered groups of 4 bits
//      belong to E_HMMA1.
//   2. Intra-warp shuffles are performed to populate T0 and T1 in a quad with all metadata
//      for HMMA 0 and T2 and T3 in a quad with all metadata for HMMA1.
__device__
uint32_t convert_E(uint32_t E) {
  // Number of bits per element of metadata
  static constexpr uint16_t bits_per_element = 2u;

  // Number of elements of metadata per uint32_t
  static constexpr uint16_t num_elements = 32 / bits_per_element;

  // Number of elements of metadata grouped sequentially along the K dimension
  // in the sparse HMMA
  static constexpr uint16_t elements_per_group = 2u;

  // Number of HMMAs into which we decompose the QMMA
  static constexpr uint16_t num_hmmas = 2u;

  uint32_t E_new = 0u;

  // New metadata bitvectors that will be used in each of the HMMAs
  uint16_t* E_hmma = reinterpret_cast<uint16_t*>(&E_new);

  // Mask for selecting the metadata bits for a single element
  static constexpr uint16_t mask = (1u << bits_per_element) - 1u;

  // Iterate through the groups of elements, assigning pairs of consecutive
  // groups to each of the different HMMAs with metadata in corresponding places.
  #pragma unroll
  for (uint16_t i = 0u; i < num_elements / elements_per_group; ++i) {
    #pragma unroll
    for (uint16_t j = 0u; j < elements_per_group; ++j) {
      uint16_t shift_out = (i * bits_per_element * elements_per_group) + (j * bits_per_element);
      uint16_t val = static_cast<uint16_t>(E >> shift_out) & mask;
      uint16_t shift_in = ((i / num_hmmas) * bits_per_element * elements_per_group) + (j * bits_per_element);
      E_hmma[i % num_hmmas] |= (val << shift_in);
    }
  }

  // At this point, each thread has the portion of its metadata that will be used
  // in HMMA 0 in E_hmma[0], and that which will be used in HMMA 1 in E_hmma[1].
  // However, a few adjustments must be made:
  //  1. Since T0 and T1 contribute metadata with sparsity selector 0 and
  //     T2 and T3 contribute with sparsity selector 1, we must shuffle metadata
  //     so that T0 and T1 contain only metadata from E_hmma[0] and T2 and T3
  //     contain only metadata from E_hmma[1].
  //  2. Because all of T0's metadata will be used before any of T1's (similarly for
  //     T2 and T3 with sparsity selector 1), we must shuffle values between threads
  //     so that the correct order of metadata is preserved.
  //
  // The series of operations to be performed is easiest to visualzie pictorially.
  // At this point, each thread has the 16-bit values of E0 and E1 on the left.
  // We would like to transform this to the version on the right
  //
  //      T0 = E0:0 E1:0                       T0 = E0:0 E0:1
  //      T1 = E0:1 E1:1         ---->         T1 = E0:2 E0:3
  //      T2 = E0:2 E1:2                       T2 = E1:0 E1:1
  //      T3 = E0:3 E1:3                       T3 = E1:2 E1:3
  // We achieve this transformation through a series of intra-warp shuffles
  // within each quad

  uint32_t id_in_quad = threadIdx.x % 4u;

  // Set up values of E0 and E1 (the metadata for HMMA 0 and HMMA 1, respectively).
  // At the end, threads 0 and 1 in a quad will only use E0, while threads 2 and 3
  // in a quad will only use E1.
  uint32_t E0 = 0u;
  uint32_t E1 = 0u;

  // Reinterpret E0 and E1 into a pair of uint16_ts for ease of access
  uint16_t* E0_half = reinterpret_cast<uint16_t*>(&E0);
  uint16_t* E1_half = reinterpret_cast<uint16_t*>(&E1);

  // T0 and T2 in the quad have correct values for the lower 16 bits of each E,
  // while T1 and T3 have correct values for the upper 16 bits of each E.
  //
  //           E0           E1
  //       ---------    ---------
  //  T0 = E0:0   -     E1:0   -
  //  T1 =   -  E0:1      -  E1:1
  //  T2 = E0:2   -     E1:2   -
  //  T3 =   -  E0:3      -  E1:3
  E0_half[id_in_quad % 2u] = E_hmma[0];
  E1_half[id_in_quad % 2u] = E_hmma[1];

  // Send our calculated E values down by one thread so that T0 gets T1's values
  // and T2 gets T3's values.
  uint32_t E_other = __shfl_down_sync(0xffffffff, E_new, 1);
  if (id_in_quad % 2u == 0) {
    uint16_t* E_other_half = reinterpret_cast<uint16_t*>(&E_other);

    E0_half[1] = E_other_half[0];
    E1_half[1] = E_other_half[1];

    // At this point:
    //   * T0 has E0 correctly populated and will keep this value.
    //   * T0 has E1 correctly populated and will eventually transfer this value to T2
    //   * T2 has E0 correctly populated and will eventually transfer this value to T1
    //   * T2 has E1 correctly populated and will eventually transfer this value to T3
    //
    //           E0           E1
    //       ---------    ---------
    //  T0 = E0:0 E0:1    E1:0 E1:1
    //  T1 =   -  E0:1      -  E1:1
    //  T2 = E0:2 E0:3    E1:2 E1:3
    //  T3 =   -  E0:3      -  E1:3
  }

  // Transfer T2's E0 to T1
  E_other = __shfl_down_sync(0xffffffff, E0, 1);
  if (id_in_quad == 1) {
    E0 = E_other;
    //           E0           E1
    //       ---------    ---------
    //  T0 = E0:0 E0:1    E1:0 E1:1
    //  T1 = E0:2 E0:3      -  E1:1
    //  T2 = E0:2 E0:3    E1:2 E1:3
    //  T3 =   -  E0:3      -  E1:3
  }

  // Transfer T2's E1 to T3
  E_other = __shfl_up_sync(0xffffffff, E1, 1);
  if (id_in_quad == 3) {
    E1 = E_other;
    //           E0           E1
    //       ---------    ---------
    //  T0 = E0:0 E0:1    E1:0 E1:1
    //  T1 = E0:2 E0:3      -  E1:1
    //  T2 = E0:2 E0:3    E1:2 E1:3
    //  T3 =   -  E0:3    E1:2 E1:3
  }

  // Transfer T0's E1 to T2
  E_other = __shfl_up_sync(0xffffffff, E1, 2);
  if (id_in_quad == 2) {
    E1 = E_other;
    //           E0           E1
    //       ---------    ---------
    //  T0 = E0:0 E0:1    E1:0 E1:1
    //  T1 = E0:2 E0:3      -  E1:1
    //  T2 = E0:2 E0:3    E1:0 E1:1
    //  T3 =   -  E0:3    E1:2 E1:3
  }

  // At this point, T0 and T1 have the correct values for E0
  // and T2 and T3 have the correct values for E1.
  // Thus, we are complete, as T0 and T1 will contribute metadata only
  // for HMMA 0 and T2 and T3 will contribute metadata only for HMMA 1.
  return id_in_quad < 2 ? E0 : E1;
}

// Reorganize values of B among threads in a quad.
// For a sparse 16864 QMMA, a quad of threads holds the following row values
// for a given column:
//   T0 = 00 01 02 03 16 17 18 19 32 33 34 35 48 49 50 51
//   T1 = 04 05 06 07 20 21 22 23 36 37 38 39 52 53 54 55
//   T2 = 08 09 10 11 24 25 26 27 40 41 42 43 56 57 58 59
//   T3 = 12 13 14 15 28 29 30 31 44 45 46 47 60 61 62 63
// For a sparse 16832 HMMA, a quad of threads holds the following row values
// for a given column:
//   T0 = 00 01 08 09 16 17 24 25
//   T1 = 02 03 10 11 18 19 26 27
//   T2 = 04 05 12 13 20 21 28 29
//   T3 = 06 07 14 15 22 23 30 31
// Based on our alternating mapping of groups of 4 consecutive elements in operand A
// to HMMAs 0 and 1 (e.g., cols 0--3 to HMMA 0, 4--7 to HMMA 1), we must map the
// corresponding rows of operand A to these same HMMAs (e.g., rows 0--3 to HMMA 0,
// 4--7 to HMMA 1). Thus, we need each thread to hold the following values for each HMMA:
//             HMMA 0                               HMMA 1
//             ------                               ------
//   T0 = 00 01 16 17 32 33 48 49         T0 = 04 05 20 21 36 37 52 53
//   T1 = 02 03 18 19 34 35 50 51         T1 = 06 07 22 23 38 39 54 55
//   T2 = 08 09 24 25 40 41 56 57         T2 = 12 13 28 29 44 45 60 61
//   T3 = 10 11 26 27 42 43 58 59         T3 = 14 15 30 31 46 47 62 63
// Achieving this mapping requires:
//   1. Shuffling the first two elements per group of four (b01) from T1 -> T0 and T3 -> T2.
//      This allows T0 and T2 to hold the correct values for HMMA 1 (e.g., T0 having {04, 05}
//      and {20, 21}, which come from T1 in the QMMA view)
//   2. Shuffling the second two elements per group of four (b23) from T0 -> T1 and T2 -> T3.
//      This allows T1 and T3 to hold the correct values for HMMA 0 (e.g., T1 having {02, 03}
//      and {18, 19}, which come from T0 in the QMMA view)
__device__
void shuffle_B(int i, uint32_t b01, uint32_t b23, uint32_t (&B_f16)[2][4]) {
  uint32_t o_b01 = __shfl_down_sync(0xffffffff, b01, 1);
  uint32_t o_b23 = __shfl_up_sync(0xffffffff, b23, 1);
  if (threadIdx.x % 2 == 0) {
    B_f16[0][i] = b01;
    B_f16[1][i] = o_b01;
  }
  else {
    B_f16[0][i] = o_b23;
    B_f16[1][i] = b23;
  }
}

// Perform the underlying HMMAs for the decomposed QMMA
__device__
void compute_hmmas(uint32_t* D, uint32_t const (&A_f16)[2][4], uint32_t const (&B_f16)[2][4], uint32_t const* C, uint32_t E) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    D[i] = C[i];
  }

  // Perform the first HMMA with sparsity selector 0x0
  asm volatile(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A_f16[0][0]), "r"(A_f16[0][1]), "r"(A_f16[0][2]), "r"(A_f16[0][3]),
        "r"(B_f16[0][0]), "r"(B_f16[0][1]), "r"(B_f16[0][2]), "r"(B_f16[0][3]),
        "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
        "r"(E));

  // Perform the second HMMA with sparsity selector 0x1
  asm volatile(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A_f16[1][0]), "r"(A_f16[1][1]), "r"(A_f16[1][2]), "r"(A_f16[1][3]),
        "r"(B_f16[1][0]), "r"(B_f16[1][1]), "r"(B_f16[1][2]), "r"(B_f16[1][3]),
        "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
        "r"(E));
}
} // namespace detail

__device__
void mma_sparse_16864_e4m3_e4m3_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C, uint32_t E) {
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
  uint32_t B_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t b01, b23;
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(b01), "=r"(b23) : "r"(B[i]));

    detail::shuffle_B(i, b01, b23, B_f16);
  }

  E = detail::convert_E(E);

  detail::compute_hmmas(D, A_f16, B_f16, C, E);

#endif
}

__device__
void mma_sparse_16864_e4m3_e5m2_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C, uint32_t E) {
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
  uint32_t B_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t b01, b23;
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(b01), "=r"(b23) : "r"(B[i]));

    detail::shuffle_B(i, b01, b23, B_f16);
  }

  E = detail::convert_E(E);

  detail::compute_hmmas(D, A_f16, B_f16, C, E);

#endif
}

__device__
void mma_sparse_16864_e5m2_e4m3_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C, uint32_t E) {
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
  uint32_t B_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t b01, b23;
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(b01), "=r"(b23) : "r"(B[i]));

    detail::shuffle_B(i, b01, b23, B_f16);
  }

  E = detail::convert_E(E);

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    D[i] = C[i];
  }

  detail::compute_hmmas(D, A_f16, B_f16, C, E);

#endif
}

__device__
void mma_sparse_16864_e5m2_e5m2_f32(uint32_t* D, uint32_t const* A, uint32_t const* B, uint32_t const* C, uint32_t E) {
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
  uint32_t B_f16[2][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t b01, b23;
    asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(b01), "=r"(b23) : "r"(B[i]));

    detail::shuffle_B(i, b01, b23, B_f16);
  }

  E = detail::convert_E(E);

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    D[i] = C[i];
  }

  detail::compute_hmmas(D, A_f16, B_f16, C, E);

#endif
}

} // namespace device
} // namespace emu
} // namespace arch
} // namespace cutlass