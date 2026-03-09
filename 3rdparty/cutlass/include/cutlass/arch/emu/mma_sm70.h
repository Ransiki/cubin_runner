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
    \brief Emulation of matrix multiply-accumulate operators for SM70
    \brief Emulation of matrix multiply-accumulate operators added for SM70 on device
 */
 
#pragma once

#include <cuda_fp16.h>

namespace cutlass {
namespace arch {
namespace emu {

constexpr int kWarp = 32;
constexpr unsigned kMask = 0xffffffff;

namespace detail {

/**
 * \brief transposes 4 halves in a thread with other in same quad
 */
__device__ void transpose_4xhalf_quad(__half2 a[2]) {
  int laneId = threadIdx.x % 32;
  const int kLaneMask01 = 0x1, kLaneMask02 = 0x2;

  __half2 tmp1_a[2], tmp2_a[2];

  int srcLane = kLaneMask01 ^ laneId;
  tmp1_a[0].x = __shfl_sync(kMask, a[0].x, srcLane, kWarp);
  tmp1_a[0].y = __shfl_sync(kMask, a[0].y, srcLane, kWarp);
  tmp1_a[1].x = __shfl_sync(kMask, a[1].x, srcLane, kWarp);
  tmp1_a[1].y = __shfl_sync(kMask, a[1].y, srcLane, kWarp);

  tmp2_a[0].x = (laneId & 0x1) == 0x1 ? tmp1_a[0].y : a[0].x;
  tmp2_a[0].y = (laneId & 0x1) == 0x1 ? a[0].y : tmp1_a[0].x;
  tmp2_a[1].x = (laneId & 0x1) == 0x1 ? tmp1_a[1].y : a[1].x;
  tmp2_a[1].y = (laneId & 0x1) == 0x1 ? a[1].y : tmp1_a[1].x;

  srcLane = kLaneMask02 ^ laneId;
  tmp1_a[0].x = __shfl_sync(kMask, tmp2_a[0].x, srcLane, kWarp);
  tmp1_a[0].y = __shfl_sync(kMask, tmp2_a[0].y, srcLane, kWarp);
  tmp1_a[1].x = __shfl_sync(kMask, tmp2_a[1].x, srcLane, kWarp);
  tmp1_a[1].y = __shfl_sync(kMask, tmp2_a[1].y, srcLane, kWarp);

  a[0].x = ((laneId >> 0x1) & 0x1) == 0x1 ? tmp1_a[1].x : tmp2_a[0].x;
  a[0].y = ((laneId >> 0x1) & 0x1) == 0x1 ? tmp1_a[1].y : tmp2_a[0].y;
  a[1].x = ((laneId >> 0x1) & 0x1) == 0x1 ? tmp2_a[1].x : tmp1_a[0].x;
  a[1].y = ((laneId >> 0x1) & 0x1) == 0x1 ? tmp2_a[1].y : tmp1_a[0].y;
}

__device__ void transpose_fragment_c(__half2 c[4]) {
  int laneId = threadIdx.x % kWarp;
  __half2 tmp1_c[4];

  int srcLane = laneId ^ 2;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    tmp1_c[i] = __shfl_sync(kMask, c[i], srcLane, kWarp);
  }

  c[0] = (laneId / 2) % 2 == 1 ? tmp1_c[1] : c[0];
  c[1] = (laneId / 2) % 2 == 1 ? c[1] : tmp1_c[0];
  c[2] = (laneId / 2) % 2 == 1 ? tmp1_c[3] : c[2];
  c[3] = (laneId / 2) % 2 == 1 ? c[3] : tmp1_c[2];
}

}  // namespace detail

namespace device {

__device__ void mma884_row_col_fp16_fp16_fp16_fp16(__half2 d[4],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  int tx = threadIdx.x;
  int srcLane = ((tx % 16) / 4) * 4;

  __half2 shfl_b;
  float r;

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      shfl_b = __shfl_sync(kMask, b[0], srcLane + i * 16 + j, kWarp);
      r = __half2float(a[0].x) * __half2float(shfl_b.x) +
          __half2float(a[0].y) * __half2float(shfl_b.y) + __half2float(c[(j / 2) + i * 2].x);
      shfl_b = __shfl_sync(kMask, b[1], srcLane + i * 16 + j, kWarp);
      r = __half2float(a[1].x) * __half2float(shfl_b.x) +
          __half2float(a[1].y) * __half2float(shfl_b.y) + r;
      d[(j / 2) + i * 2].x = __float2half(r);

      shfl_b = __shfl_sync(kMask, b[0], srcLane + i * 16 + j + 1, kWarp);
      r = __half2float(a[0].x) * __half2float(shfl_b.x) +
          __half2float(a[0].y) * __half2float(shfl_b.y) + __half2float(c[(j / 2) + i * 2].y);
      shfl_b = __shfl_sync(kMask, b[1], srcLane + i * 16 + j + 1, kWarp);
      r = __half2float(a[1].x) * __half2float(shfl_b.x) +
          __half2float(a[1].y) * __half2float(shfl_b.y) + r;
      d[(j / 2) + i * 2].y = __float2half(r);
    }
  }
}

__device__ void mma884_row_row_fp16_fp16_fp16_fp16(__half2 d[4],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  __half2 tmp_b[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_b[i] = b[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_b);

  mma884_row_col_fp16_fp16_fp16_fp16(d, a, tmp_b, c);
}

__device__ void mma884_col_row_fp16_fp16_fp16_fp16(__half2 d[4],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  __half2 tmp_a[2], tmp_b[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_a[i] = a[i];
    tmp_b[i] = b[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_a);
  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_b);

  mma884_row_col_fp16_fp16_fp16_fp16(d, tmp_a, tmp_b, c);
}

__device__ void mma884_col_col_fp16_fp16_fp16_fp16(__half2 d[4],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  __half2 tmp_a[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_a[i] = a[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_a);

  mma884_row_col_fp16_fp16_fp16_fp16(d, tmp_a, b, c);
}

__device__ void mma884_row_col_fp32_fp16_fp16_fp16(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  int laneId = threadIdx.x % kWarp;
  int srcLaneA = (laneId / 4) * 4 + (laneId % 2);
  int srcLaneB = ((laneId / 2) * 2) % 16;

  float ra[2][4], rb[4][4], rc[8];

  __half2 tmp;

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      tmp = __shfl_sync(kMask, a[j], srcLaneA + i * 2, kWarp);
      ra[i][2 * j] = __half2float(tmp.x);
      ra[i][2 * j + 1] = __half2float(tmp.y);
    }
  }

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        tmp = __shfl_sync(kMask, b[k], srcLaneB + j + i * 16, kWarp);
        rb[j + i * 2][2 * k] = __half2float(tmp.x);
        rb[j + i * 2][2 * k + 1] = __half2float(tmp.y);
      }
    }
  }

  __half2 tmp_c[4];

  #pragma unroll
  for(int i = 0; i < 4; i++) {
    tmp_c[i] = c[i];
  }

  cutlass::arch::emu::detail::transpose_fragment_c(tmp_c);

#pragma unroll
  for (int i = 0; i < 4; i++) {
    rc[2 * i] = __half2float(tmp_c[i].x);
    rc[2 * i + 1] = __half2float(tmp_c[i].y);
  }

// second-half step through B
#pragma unroll
  for (int i = 0; i < 2; i++) {
// step through A
#pragma unroll
    for (int j = 0; j < 2; j++) {
// first-half step through B
#pragma unroll
      for (int k = 0; k < 2; k++) {
        int idxC = k + j * 2 + i * 2 * 2;
        int idxB = k + i * 2;
        int idxA = j;
        float acc = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
          acc += ra[idxA][l] * rb[idxB][l];
        }
        d[idxC] = acc + rc[idxC];
      }
    }
  }
}

__device__ void mma884_row_row_fp32_fp16_fp16_fp16(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  __half2 tmp_b[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_b[i] = b[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_b);

  mma884_row_col_fp32_fp16_fp16_fp16(d, a, tmp_b, c);
}

__device__ void mma884_col_row_fp32_fp16_fp16_fp16(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  __half2 tmp_a[2], tmp_b[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_a[i] = a[i];
    tmp_b[i] = b[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_a);
  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_b);

  mma884_row_col_fp32_fp16_fp16_fp16(d, tmp_a, tmp_b, c);
}

__device__ void mma884_col_col_fp32_fp16_fp16_fp16(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   __half2 c[4]) {
  __half2 tmp_a[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_a[i] = a[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_a);

  mma884_row_col_fp32_fp16_fp16_fp16(d, tmp_a, b, c);
}

__device__ void mma884_row_col_fp32_fp16_fp16_fp32(float d[8],
                                                  __half2 a[2],
                                                  __half2 b[2],
                                                  float c[8]) {
  int laneId = threadIdx.x % kWarp;
  int srcLaneA = (laneId / 4) * 4 + (laneId % 2);
  int srcLaneB = ((laneId / 2) * 2) % 16;
  float ra[2][4], rb[4][4];

  __half2 tmp;

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      tmp = __shfl_sync(kMask, a[j], srcLaneA + i * 2, kWarp);
      ra[i][2 * j] = __half2float(tmp.x);
      ra[i][2 * j + 1] = __half2float(tmp.y);
    }
  }

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        tmp = __shfl_sync(kMask, b[k], srcLaneB + j + i * 16, kWarp);
        rb[j + i * 2][2 * k] = __half2float(tmp.x);
        rb[j + i * 2][2 * k + 1] = __half2float(tmp.y);
      }
    }
  }

// second-half step through B
#pragma unroll
  for (int i = 0; i < 2; i++) {
// step through A
#pragma unroll
    for (int j = 0; j < 2; j++) {
// first-half step through B
#pragma unroll
      for (int k = 0; k < 2; k++) {
        int idxC = k + j * 2 + i * 2 * 2;
        int idxB = k + i * 2;
        int idxA = j;
        float acc = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
          acc += ra[idxA][l] * rb[idxB][l];
        }
        d[idxC] = acc + c[idxC];
      }
    }
  }
}

__device__ void mma884_row_row_fp32_fp16_fp16_fp32(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   float c[8]) {
  __half2 tmp_b[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_b[i] = b[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_b);

  mma884_row_col_fp32_fp16_fp16_fp32(d, a, tmp_b, c);
}

__device__ void mma884_col_row_fp32_fp16_fp16_fp32(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   float c[8]) {
  __half2 tmp_a[2], tmp_b[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_a[i] = a[i];
    tmp_b[i] = b[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_a);
  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_b);

  mma884_row_col_fp32_fp16_fp16_fp32(d, tmp_a, tmp_b, c);
}

__device__ void mma884_col_col_fp32_fp16_fp16_fp32(float d[8],
                                                   __half2 a[2],
                                                   __half2 b[2],
                                                   float c[8]) {
  __half2 tmp_a[2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
    tmp_a[i] = a[i];
  }

  cutlass::arch::emu::detail::transpose_4xhalf_quad(tmp_a);

  mma884_row_col_fp32_fp16_fp16_fp32(d, tmp_a, b, c);
}

}  // namespace device

}  // namespace emu
}  // namespace arch
}  // namespace cutlass