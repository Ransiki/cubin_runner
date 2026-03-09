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
    \brief Emulation of matrix multiply-accumulate operators added for SM70 on host
*/
#pragma once

#include "cutlass/layout/tensor.h"

namespace util {
namespace reference {
namespace host {
namespace emu {
namespace detail {

template <typename Element>
cutlass::HostTensor<Element, cutlass::layout::ColumnMajor> transpose(
    cutlass::HostTensor<Element, cutlass::layout::RowMajor>& input_tensor) {
  cutlass::HostTensor<Element, cutlass::layout::ColumnMajor> output_tensor(
      {1, 32 * 4/* how to get dimensions of input_tensor here? */});
  Element* input_matrix = input_tensor.host_data();
  Element* output_matrix = output_tensor.host_data();
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = j; k < 4; k++) {
        output_matrix[j + k * 4 + i * 16] = input_matrix[k + j * 4 + i * 16];
        output_matrix[k + j * 4 + i * 16] = input_matrix[j + k * 4 + i * 16];
      }
    }
  }
  return output_tensor;
}

template <typename Element>
cutlass::HostTensor<Element, cutlass::layout::RowMajor> transpose(
    cutlass::HostTensor<Element, cutlass::layout::ColumnMajor>& input_tensor) {
  cutlass::HostTensor<Element, cutlass::layout::RowMajor> output_tensor(
      {1, 32 * 4/* how to get dimensions of input_tensor here? */});
  Element* input_matrix = input_tensor.host_data();
  Element* output_matrix = output_tensor.host_data();
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = j; k < 4; k++) {
        output_matrix[j + k * 4 + i * 16] = input_matrix[k + j * 4 + i * 16];
        output_matrix[k + j * 4 + i * 16] = input_matrix[j + k * 4 + i * 16];
      }
    }
  }
  return output_tensor;
}

}  // namespace detail

template<typename Element>
void transpose_c(cutlass::HostTensor<Element, cutlass::layout::ColumnMajor>& tensor) {
  Element* matrix = tensor.host_data();

  for (int h = 0; h < 8 * 32; h += 4 * 8) {
    for (int k = 0; k < 8; k += 4) {
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          int load_index = j + 2 + i * 8 + k + h;
          int store_index = 2 * 8 + j + i * 8 + k + h;
          Element tmp = matrix[load_index];
          matrix[load_index] = matrix[store_index];
          matrix[store_index] = tmp;
        }
      }
    }
  }

}

void mma884_row_col_fp16_fp16_fp16_fp16(
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& tensor_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_b,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_c) {

  cutlass::half_t* matrix_d = tensor_d.host_data();
  cutlass::half_t* matrix_a = tensor_a.host_data();
  cutlass::half_t* matrix_b = tensor_b.host_data();
  cutlass::half_t* matrix_c = tensor_c.host_data();
  for (int g = 0; g < 4; g++) {      // 4 quad pairs
    for (int h = 0; h < 2; h++) {    // quad pair
      for (int i = 0; i < 4; i++) {  // a quad
        int index_a = i + h * 16 + g * 4;
        for (int j = 0; j < 2; j++) {    // 2 x 4 elements in a thread
          for (int k = 0; k < 4; k++) {  // first 4 elements in a thread
            int index_b = k + j * 16 + g * 4;
            int index_c = k + j * 4 + i * 8 + h * 8 * 16 + g * 8 * 4;
            cutlass::half_t c = matrix_c[index_c];
            for (int l = 0; l < 4; l++) {  // inner product dimension
              c += matrix_a[l + index_a * 4] * matrix_b[index_b * 4 + l];
            }
            matrix_d[index_c] = c;
          }
        }
      }
    }
  }
}


void mma884_row_row_fp16_fp16_fp16_fp16(
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_d,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& tensor_a,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& tensor_b,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_c) {

      cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tmp_tensor_b =
      util::reference::host::emu::detail::transpose(tensor_b);
  mma884_row_col_fp16_fp16_fp16_fp16(tensor_d, tensor_a, tmp_tensor_b, tensor_c);
}

void mma884_col_row_fp16_fp16_fp16_fp16(
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_d,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_a,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& tensor_b,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_c) {

      cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tmp_tensor_a =
      util::reference::host::emu::detail::transpose(tensor_a);
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tmp_tensor_b =
      util::reference::host::emu::detail::transpose(tensor_b);
  mma884_row_col_fp16_fp16_fp16_fp16(tensor_d, tmp_tensor_a, tmp_tensor_b, tensor_c);
}


void mma884_col_col_fp16_fp16_fp16_fp16(
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_d,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_a,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_b,
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_c) {

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tmp_tensor_a =
      util::reference::host::emu::detail::transpose(tensor_a);
      mma884_row_col_fp16_fp16_fp16_fp16(tensor_d, tmp_tensor_a, tensor_b, tensor_c);
}


void mma884_row_col_fp32_fp16_fp16_fp16(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& tensor_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& tensor_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_b,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_c) {
  float* matrix_d = tensor_d.host_data();
  cutlass::half_t* matrix_a = tensor_a.host_data();
  cutlass::half_t* matrix_b = tensor_b.host_data();
  cutlass::half_t* matrix_c = tensor_c.host_data();
  for (int g = 0; g < 4; g++) {              // 4 mma per warp
    for (int h = 0; h < 2; h++) {            // t16->t19
      for (int i = 0; i < 2; i++) {          // t0->t2
        for (int j = 0; j < 2; j++) {        // t0->t1
          for (int k = 0; k < 2; k++) {      // half quad lanes right
            for (int l = 0; l < 2; l++) {    // 2 lanes down
              for (int m = 0; m < 2; m++) {  // neighbor
                int index_d = m + l * 2 + k * 2 * 2 + j * 2 * 2 * 2 +
                              i * 2 * 2 * 2 * 2 + h * 16 * 8 + g * 4 * 8;
                int index_c = m + l * 2 + k * 2 * 2 + j * 2 * 2 * 2 +
                              i * 2 * 2 * 2 * 2 + h * 8 * 16 + g * 8 * 4;
                float c = float(matrix_c[index_c]);
                for (int n = 0; n < 4; n++) {
                  int index_a = l * 2 + j + g * 4 + h * 16;
                  int index_b = m + k * 16 + i * 2 + g * 4;
                  c += float(matrix_a[index_a * 4 + n]) * float(matrix_b[index_b * 4 + n]);
                }
                matrix_d[index_d] = c;
              }
            }
          }
        }
      }
    }
  }
}

void mma884_row_row_fp32_fp16_fp16_fp16(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& matrix_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& matrix_b,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_c) {
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tmp_matrix_b =
      util::reference::host::emu::detail::transpose(matrix_b);
  mma884_row_col_fp32_fp16_fp16_fp16(matrix_d, matrix_a, tmp_matrix_b, matrix_c);
}

void mma884_col_row_fp32_fp16_fp16_fp16(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& matrix_b,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_c) {
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tmp_matrix_a =
      util::reference::host::emu::detail::transpose(matrix_a);
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tmp_matrix_b =
      util::reference::host::emu::detail::transpose(matrix_b);
  mma884_row_col_fp32_fp16_fp16_fp16(matrix_d, tmp_matrix_a, tmp_matrix_b, matrix_c);
}

void mma884_col_col_fp32_fp16_fp16_fp16(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_b,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_c) {
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tmp_matrix_a =
      util::reference::host::emu::detail::transpose(matrix_a);

  mma884_row_col_fp32_fp16_fp16_fp16(matrix_d, tmp_matrix_a, matrix_b, matrix_c);
}

void mma884_row_col_fp32_fp16_fp16_fp32(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& tensor_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& tensor_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& tensor_b,
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& tensor_c) {
  float* matrix_d = tensor_d.host_data();
  cutlass::half_t* matrix_a = tensor_a.host_data();
  cutlass::half_t* matrix_b = tensor_b.host_data();
  float* matrix_c = tensor_c.host_data();
  for (int g = 0; g < 4; g++) {              // 4 mma per warp
    for (int h = 0; h < 2; h++) {            // t16->t19
      for (int i = 0; i < 2; i++) {          // t0->t2
        for (int j = 0; j < 2; j++) {        // t0->t1
          for (int k = 0; k < 2; k++) {      // half quad lanes right
            for (int l = 0; l < 2; l++) {    // 2 lanes down
              for (int m = 0; m < 2; m++) {  // neighbor
                int index_c = m + l * 2 + k * 2 * 2 + j * 2 * 2 * 2 +
                              i * 2 * 2 * 2 * 2 + h * 16 * 8 + g * 4 * 8;
                float c = matrix_c[index_c];
                for (int n = 0; n < 4; n++) {
                  int index_a = l * 2 + j + g * 4 + h * 16;
                  int index_b = m + k * 16 + i * 2 + g * 4;
                  c += float(matrix_a[index_a * 4 + n]) * float(matrix_b[index_b * 4 + n]);
                }
                matrix_d[index_c] = c;
              }
            }
          }
        }
      }
    }
  }
}

void mma884_row_row_fp32_fp16_fp16_fp32(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& matrix_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& matrix_b,
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_c) {
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tmp_matrix_b =
      util::reference::host::emu::detail::transpose(matrix_b);
  mma884_row_col_fp32_fp16_fp16_fp32(matrix_d, matrix_a, tmp_matrix_b, matrix_c);
}

void mma884_col_row_fp32_fp16_fp16_fp32(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor>& matrix_b,
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_c) {
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tmp_matrix_a =
      util::reference::host::emu::detail::transpose(matrix_a);
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tmp_matrix_b =
      util::reference::host::emu::detail::transpose(matrix_b);
  mma884_row_col_fp32_fp16_fp16_fp32(matrix_d, tmp_matrix_a, tmp_matrix_b, matrix_c);
}

void mma884_col_col_fp32_fp16_fp16_fp32(
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_d,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_a,
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>& matrix_b,
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor>& matrix_c) {
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tmp_matrix_a =
      util::reference::host::emu::detail::transpose(matrix_a);

  mma884_row_col_fp32_fp16_fp16_fp32(matrix_d, tmp_matrix_a, matrix_b, matrix_c);
}

}  // namespace emu
}  // namespace host
}  // namespace reference
}  // namespace util
