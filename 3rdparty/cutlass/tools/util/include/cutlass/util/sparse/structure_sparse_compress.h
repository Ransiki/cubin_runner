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

/*! \file
    \brief Compress utils for structure sparse kernels
*/

// {$nv-internal-release file}

#pragma once

#include "cute/config.hpp"

namespace cutlass {

namespace detail {

/*
Index     Bin   HEX
0, 1    0b0100   4
1, 2    0b1001   9
2, 3    0b1110   E
0, 2    0b1000   8
1, 3    0b1101   D
0, 3    0b1100   C
2, 1    0b0110   6  (Not used)
-----------------------------------
TF32
0       0b0100   4
1       0b1110   E
*/

template<typename T>
CUTLASS_HOST_DEVICE
uint8_t encode_in_chunk_idx(int in_chunk_idx){
  if (sizeof(T) == 4){
    return in_chunk_idx == 0 ? 0b0100 : 0b1110;
  }
  else {
    uint8_t res = 0;
    if (in_chunk_idx == 0){
      res = 0b00;
    }
    else if (in_chunk_idx == 1){
      res = 0b01;
    }
    else if (in_chunk_idx == 2){
      res = 0b10;
    }
    else {
      res = 0b11;
    }
    return res;
  }
}

template<typename T, typename Metadata_t>
CUTLASS_HOST_DEVICE
int decode_metadata_one_elem(Metadata_t metadata_val){
  if (sizeof(T) == 4){
    if (metadata_val == 0b0100){
      return 0;
    }
    else if (metadata_val == 0b1110){
      return 1;
    }
    else {
      assert(false && "illegal metadata value!");
    }
  }
  else {
    if (metadata_val == 0b00){
      return 0;
    }
    else if (metadata_val == 0b01){
      return 1;
    }
    else if (metadata_val == 0b10){
      return 2;
    }
    else if (metadata_val == 0b11){
      return 3;
    }
    else{
      assert(false && "illegal metadata value!");
    }
  }
  return 0;
}
} // End namespace detail
} // End namesparse cutlass
