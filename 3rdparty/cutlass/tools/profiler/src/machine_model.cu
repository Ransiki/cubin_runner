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
/* \file
   \brief Machine model computing peak flops and utilization.

   {$nv-profiling file}
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/mma.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/library/util.h"

#include "cutlass/profiler/machine_model.h"
#include <cuda_runtime.h>

namespace cutlass {
namespace profiler {

///////////////////////////////////////////////////////////////////////////////////////////////////

static double const kBytesPerGiB = double(1ull << 30);

///////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream & operator<<(std::ostream &out, FmaOp const &op) {
  return out << "(" << op.shape[0] << ", " << op.shape[1] << ", " << op.shape[2] << "), class: " << library::to_string(op.opcode_class)
    << "(" << library::to_string(op.A) << ", " << library::to_string(op.B) << ", " << library::to_string(op.C) << "}";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void MachineModel::init_fma_inst_sm5x() {

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {1, 128}    // 1 issue per clock per datapath; 128 data paths
  });
}

void MachineModel::init_fma_inst_sm60() {

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {1, 64}
  });
  
  // DFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {2, 64}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,2,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 64}
  });
}

void MachineModel::init_fma_inst_sm61() {

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {1, 128}
  });
  
  // DFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {2, 1}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,2,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {2, 1}
  });

  // IDP4A
  fma_inst_.insert({
    {{1,1,4}, library::OpcodeClassID::kSimt, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {1, 128}
  });
}

void MachineModel::init_fma_inst_sm70() {
  // FFMA Complex 
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32},
    {2, 128}
  });

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {2, 128}
  });
  
  // DFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {4, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // IDP4A
  fma_inst_.insert({
    {{1,1,4}, library::OpcodeClassID::kSimt, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {2, 128}
  });

  // HMMA.884.F16
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 16}
  });

  // HMMA.884.F16
  fma_inst_.insert({
    {{16,16,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.884.F32
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 16}
  });

  // HMMA.884.F32
  fma_inst_.insert({
    {{16,16,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // WMMA.161616.F16
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.161616.F32
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 4}
  });

  // WMMA.83216.F16
  fma_inst_.insert({
    {{8,32,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.83216.F32
  fma_inst_.insert({
    {{8,32,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 4}
  });

  // WMMA.32816.F16
  fma_inst_.insert({
    {{32,8,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.32816.F32
  fma_inst_.insert({
    {{32,8,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 4}
  });
}

void MachineModel::init_fma_inst_sm72() {

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {2, 128}
  });
  
  // DFMA2
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {4, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // IDP4A
  fma_inst_.insert({
    {{1,1,4}, library::OpcodeClassID::kSimt, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {2, 128}
  });

  // HMMA.884.F16
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 16}
  });

  // HMMA.884.F16
  fma_inst_.insert({
    {{16,16,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.884.F32
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 16}
  });

  // HMMA.884.F32
  fma_inst_.insert({
    {{16,16,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {8, 4}
  });

  // WMMA.161616.F16
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.161616.F32
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 4}
  });

  // WMMA.83216.F16
  fma_inst_.insert({
    {{8,32,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // WMMA.83216.F32
  fma_inst_.insert({
    {{8,32,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // WMMA.32816.F16
  fma_inst_.insert({
    {{32,8,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.32816.F32
  fma_inst_.insert({
    {{32,8,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 4}
  });
}

void MachineModel::init_fma_inst_sm75() {
  
  // FFMA Complex
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32},
    {2, 128}
  });

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {2, 128}
  });
  
  // DFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {16, 32}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,2,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // IDP4A
  fma_inst_.insert({
    {{1,1,4}, library::OpcodeClassID::kSimt, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {2, 128}
  });

  // HMMA.884.F16
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {16, 16}
  });

  // HMMA.884.F16
  fma_inst_.insert({
    {{16,16,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {16, 4}
  });

  // HMMA.884.F32
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 16, 2}
  });

  // HMMA.884.F32
  fma_inst_.insert({
    {{16,16,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {32, 4, 2}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU8, library::NumericTypeID::kU8, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU4, library::NumericTypeID::kU4, library::NumericTypeID::kS32},
    {4, 4}
  });

  // BMMA.88128
  fma_inst_.insert({
    {{8,8,128}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kB1, library::NumericTypeID::kB1, library::NumericTypeID::kS32},
    {4, 4}
  });

  // HMMA.1688.F16
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {16, 4, 2}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {16, 4, 2}
  });

  // HMMA.1684.F32
  fma_inst_.insert({
    {{16,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {16, 4, 2}
  });

  // HMMA.1684.F32
  fma_inst_.insert({
    {{16,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {16, 4, 2}
  });

  // WMMA.161616.F16
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.161616.F32
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {64, 4, 2}
  });

  // WMMA.83216.F16
  fma_inst_.insert({
    {{8,32,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.83216.F32
  fma_inst_.insert({
    {{8,32,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {64, 4, 2}
  });

  // WMMA.32816.F16
  fma_inst_.insert({
    {{32,8,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {32, 4}
  });

  // WMMA.32816.F32
  fma_inst_.insert({
    {{32,8,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {64, 4, 2}
  });
}

void MachineModel::init_fma_inst_sm80() {

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {2, 128}
  });
 
  // FFMA Complex
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32},
    {2, 128}
  });
  
  // DFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {4, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,2,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // IDP4A
  fma_inst_.insert({
    {{1,1,4}, library::OpcodeClassID::kSimt, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {2, 128}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU8, library::NumericTypeID::kU8, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU4, library::NumericTypeID::kU4, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.16832
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.16832
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU8, library::NumericTypeID::kU8, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.16864
  fma_inst_.insert({
    {{16,8,64}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.16864
  fma_inst_.insert({
    {{16,8,64}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU4, library::NumericTypeID::kU4, library::NumericTypeID::kS32},
    {8, 4}
  });

  // BMMA.168256
  fma_inst_.insert({
    {{16,8,256}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kB1, library::NumericTypeID::kB1, library::NumericTypeID::kS32},
    {8, 4}
  });

  // HMMA.1688.F16
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {4, 4}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {4, 4}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {4, 4}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {4, 4}
  });

  // HMMA.16816.F16
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.16816.F32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // HMMA.16816.F32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // HMMA.1688.F32.TF32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {8, 4}
  });
    
  // HMMA.1688.F32.TF32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {8, 4}
  });
  
  // DMMA.884
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {16, 4}
  });

  // HMMA.SP.16832.F16
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.SP.16832.F32
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // HMMA.SP.16832.F32
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // HMMA.SP.16816.F32.TF32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {8, 4}
  });

  // HMMA.SP.16816.F32.TF32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {8, 4}
  });

  // IMMA.SP.16864
  fma_inst_.insert({
    {{16,8,64}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.SP.168128
  fma_inst_.insert({
    {{16,8,128}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {8, 4}
  });

  // WMMA.161616.F16
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {16, 4}
  });

  // WMMA.161616.F32
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {16, 4}
  });
}

void MachineModel::init_fma_inst_sm86()  {

  // FFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {1, 128}
  });
  
  // FFMA Complex
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32, library::NumericTypeID::kCF32},
    {1, 128}
  }); 

  // DFMA
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {4, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,1,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // HFMA2
  fma_inst_.insert({
    {{1,2,1}, library::OpcodeClassID::kSimt, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {1, 128}
  });

  // IDP4A
  fma_inst_.insert({
    {{1,1,4}, library::OpcodeClassID::kSimt, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {2, 128}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8816
  fma_inst_.insert({
    {{8,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU8, library::NumericTypeID::kU8, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.8832
  fma_inst_.insert({
    {{8,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU4, library::NumericTypeID::kU4, library::NumericTypeID::kS32},
    {4, 4}
  });

  // IMMA.16832
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.16832
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU8, library::NumericTypeID::kU8, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.16864
  fma_inst_.insert({
    {{16,8,64}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {8, 4}
  });

  // IMMA.16864
  fma_inst_.insert({
    {{16,8,64}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kU4, library::NumericTypeID::kU4, library::NumericTypeID::kS32},
    {8, 4}
  });

  // BMMA.168256
  fma_inst_.insert({
    {{16,8,256}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kB1, library::NumericTypeID::kB1, library::NumericTypeID::kS32},
    {8, 4}
  });

  // HMMA.1688.F16
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {4, 4}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4, 2}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {8, 4, 2}
  });

  // HMMA.1688.F32
  fma_inst_.insert({
    {{16,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {8, 4, 2}
  });

  // HMMA.16816.F16
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.16816.F32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {16, 4}
  });

  // HMMA.16816.F32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {16, 4}
  });

  // HMMA.16816.F32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {16, 4}
  });

  // HMMA.1688.F32.TF32
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {16, 4}
  });
   
  // DMMA.884
  fma_inst_.insert({
    {{8,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {16, 4}
  });

  // HMMA.SP.16832.F16
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {8, 4}
  });

  // HMMA.SP.16832.F32
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {8, 4}
  });

  // HMMA.SP.16832.F32
  fma_inst_.insert({
    {{16,8,32}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kBF16, library::NumericTypeID::kBF16, library::NumericTypeID::kF32},
    {16, 4}
  });

  // HMMA.SP.16816.F32.TF32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kTF32, library::NumericTypeID::kTF32, library::NumericTypeID::kF32},
    {16, 4}
  });

  // HMMA.1688.F32.TF32
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF32, library::NumericTypeID::kF32, library::NumericTypeID::kF32},
    {16, 4}
  });

  // IMMA.SP.16864
  fma_inst_.insert({
    {{16,8,64}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS8, library::NumericTypeID::kS8, library::NumericTypeID::kS32},
    {16, 4}
  });

  // IMMA.SP.168128
  fma_inst_.insert({
    {{16,8,128}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kS4, library::NumericTypeID::kS4, library::NumericTypeID::kS32},
    {16, 4}
  });

  // WMMA.161616.F16
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF16},
    {16, 4}
  });

  // WMMA.161616.F32
  fma_inst_.insert({
    {{16,16,16}, library::OpcodeClassID::kWmmaTensorOp, library::NumericTypeID::kF16, library::NumericTypeID::kF16, library::NumericTypeID::kF32},
    {16, 4}
  });
}


void MachineModel::init_fma_inst_sm90() {
  // Multiples of 8, up to 256
  std::vector<int> wgmma_n_slices;
  for (int i = 1; i <= 32; ++i) {
    wgmma_n_slices.push_back(i*8);
  }

  // WGMMA 64xNx16 F16|F32 += F16 * F16
  std::vector<library::NumericTypeID> hwgmma_16b_ab_types = {library::NumericTypeID::kF16, library::NumericTypeID::kBF16};
  std::vector<library::NumericTypeID> hwgmma_16b_c_types  = {library::NumericTypeID::kF16, library::NumericTypeID::kF32};

  for (auto wgmma_n_slice : wgmma_n_slices) {
    for (auto a_type : hwgmma_16b_ab_types) {
      for (auto b_type : hwgmma_16b_ab_types) {
        for (auto acc_type : hwgmma_16b_c_types) {
          fma_inst_.insert({
            {{64,wgmma_n_slice,16}, library::OpcodeClassID::kTensorOp, a_type, b_type, acc_type},
            {wgmma_n_slice * 2, 4}
          });
        }
      }
    }
  }

  // WGMMA 64xNx8 F32 += TF32 * TF32
  std::vector<library::NumericTypeID> hwgmma_32b_ab_types = {library::NumericTypeID::kTF32, library::NumericTypeID::kF32};
  std::vector<library::NumericTypeID> hwgmma_32b_c_types  = {library::NumericTypeID::kF32};

  for (auto wgmma_n_slice : wgmma_n_slices) {
    for (auto a_type : hwgmma_32b_ab_types) {
      for (auto b_type : hwgmma_32b_ab_types) {
        for (auto acc_type : hwgmma_32b_c_types) {
          fma_inst_.insert({
            {{64,wgmma_n_slice,8}, library::OpcodeClassID::kTensorOp, a_type, b_type, acc_type},
            {wgmma_n_slice * 2, 4}
          });
        }
      }
    }
  }

  // WGMMA 64xNx32 S32 += S8 * S8
  std::vector<library::NumericTypeID> iwgmma_8b_ab_types = {library::NumericTypeID::kS8, library::NumericTypeID::kU8};
  std::vector<library::NumericTypeID> iwgmma_32b_c_types  = {library::NumericTypeID::kS32};

  for (auto wgmma_n_slice : wgmma_n_slices) {
    for (auto a_type : iwgmma_8b_ab_types) {
      for (auto b_type : iwgmma_8b_ab_types) {
        for (auto acc_type : iwgmma_32b_c_types) {
          fma_inst_.insert({
            {{64,wgmma_n_slice,32}, library::OpcodeClassID::kTensorOp, a_type, b_type, acc_type},
            {wgmma_n_slice * 2, 4}
          });
        }
      }
    }
  }

  // WGMMA 64xNx32 F16|F32 += E4M3/E5M2 * E4M3/E5M2
  std::vector<library::NumericTypeID> qwgmma_8b_ab_types = {library::NumericTypeID::kFE4M3,library::NumericTypeID::kFE5M2};
  std::vector<library::NumericTypeID> qwgmma_16b_c_types  = {library::NumericTypeID::kF16, library::NumericTypeID::kF32};

  for (auto wgmma_n_slice : wgmma_n_slices) {
    for (auto a_type : qwgmma_8b_ab_types) {
      for (auto b_type : qwgmma_8b_ab_types) {
        for (auto acc_type : qwgmma_16b_c_types) {
          fma_inst_.insert({
            {{64,wgmma_n_slice,32}, library::OpcodeClassID::kTensorOp, a_type, b_type, acc_type},
            {wgmma_n_slice * 2, 4}
          });
        }
      }
    }
  }

  // TODO: Add sparse variants

  // FP64 instructions
  fma_inst_.insert({
    {{16,8,4}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {16, 4}
  });
  fma_inst_.insert({
    {{16,8,8}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {32, 4}
  });
  fma_inst_.insert({
    {{16,8,16}, library::OpcodeClassID::kTensorOp, library::NumericTypeID::kF64, library::NumericTypeID::kF64, library::NumericTypeID::kF64},
    {64, 4}
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Accumulator>
static __global__ void k_detect_full_speed_accumulation(
  long long *clock_diff, 
  int iterations,
  Accumulator *D = nullptr, 
  half_t const *A = nullptr, 
  half_t const *B = nullptr, 
  Accumulator const *C = nullptr,
  bool guard = false) {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)

  using Mma = arch::Mma<
    gemm::GemmShape<16, 8, 8>,
    32,
    half_t, layout::RowMajor,
    half_t, layout::ColumnMajor,
    Accumulator, layout::RowMajor,
    arch::OpMultiplyAdd
  >;

  Mma mma_op;

  int const kM = 4;
  int const kN = 4;
  int const kK = 4;

  typename Mma::FragmentA frag_A[kM];
  typename Mma::FragmentB frag_B[kN];
  typename Mma::FragmentC accum[kM * kN];

  if (guard) {

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kM; ++m) {
      frag_A[m].clear();

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < int(Mma::FragmentA::kElements); ++i) {
        frag_A[m][i] = A[Mma::FragmentA::kElements * threadIdx.x + i];
      }
    }
    
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kN; ++n) {
      frag_B[n].clear();

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < int(Mma::FragmentB::kElements); ++i) {
        frag_B[n][i] = B[Mma::FragmentB::kElements * threadIdx.x + i];
      }
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < kM; ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < kN; ++n) {
      accum[m + n * kM].clear();
    }
  }

  long long int clock_start = clock64();

  __syncthreads();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < iterations; ++iter) {

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kK; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < kM; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < kN; ++n) {

          mma_op(accum[m + n * kM], frag_A[m], frag_B[n], accum[m + n * kM]);
        }
      }
    }
  }

  __syncthreads();

  long long int clock_end = clock64();

  if (guard) {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kN; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (size_t i = 0; i < accum[m + n * kM].size(); ++i) {
          D[(m + n * kM) * accum[m + n * kM].size() * threadIdx.x + i] = accum[m + n * kM][i];
        }
      }
    }
  }

  clock_diff[threadIdx.x] = clock_end - clock_start;
#endif
}

/// Simple heuristic for detecting whether F32 accumulation is full speed.
static bool detect_full_speed_f32_accumulation(double threshold = 0.25) {

  int const kThreads = 32;

  dim3 grid(1,1);
  dim3 block(kThreads, 1, 1);

  int const kIterations = 100;

  HostTensor<long long, layout::RowMajor> clock_diff_f16({kThreads, 1});

  k_detect_full_speed_accumulation<half_t><<< grid, block >>>(
    clock_diff_f16.device_data(), 
    kIterations);

  clock_diff_f16.sync_host();

  HostTensor<long long, layout::RowMajor> clock_diff_f32({kThreads, 1});

  k_detect_full_speed_accumulation<float><<< grid, block >>>(
    clock_diff_f32.device_data(), 
    kIterations);

  clock_diff_f32.sync_host();

  // If average f32 runtime is approximately the same as f16 runtime, then it's probably full-speed.
  long long avg_f16 = 0;
  long long avg_f32 = 0;

  for (int i = 0; i < kThreads; ++i) {
    avg_f16 += clock_diff_f16.host_data()[i];
    avg_f32 += clock_diff_f32.host_data()[i];
  }

  if (double(avg_f32) < (1.0 + threshold) * double(avg_f16)) {
    return true;
  }

  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void MachineModel::update_fma_inst_sm75(bool f32_full_speed) {

  if (f32_full_speed) {
    for (auto & fma : fma_inst_) {
      if (fma.first.C == library::NumericTypeID::kF32 && 
        fma.first.opcode_class == library::OpcodeClassID::kTensorOp) {
        
        fma.second.issue_interval /= fma.second.throttle;
      }
    } 
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

MachineModel::MachineModel() {

}

MachineModel::MachineModel(cudaDeviceProp const &prop) {
  update(prop);
}

/// Compute capability
int MachineModel::compute_capability() const {
  return device_.major * 10 + device_.minor;
}

/// Updates the machine model
void MachineModel::update(cudaDeviceProp const &prop) {
  device_ = prop;

  // lookup 
  switch (compute_capability()) {
    case 50: // fallthrough
    case 52: // fallthrough
    case 53:
      init_fma_inst_sm5x();
      break;
    case 60:
      init_fma_inst_sm60();
      break;
    case 61:
      init_fma_inst_sm61();
      break;
    case 70:
      init_fma_inst_sm70();
      break;
    case 72:
      init_fma_inst_sm72();
      break;
    case 75:
      init_fma_inst_sm75();
      break;
    case 80:
      init_fma_inst_sm80();
      break;
    case 86:
      init_fma_inst_sm86();
      break;
    case 90:
      init_fma_inst_sm90();
      break;
    default:
      break;
  }
}

/// Measures some properties via a kernel launch. 
bool MachineModel::measure() {

  bool model_changed = false;

  switch (compute_capability()) {
  
  case 75:
    // Detect whether FP32 accumulation is full speed
    model_changed = detect_full_speed_f32_accumulation();
    update_fma_inst_sm75(model_changed);
    break;

  default:
    break;
  }

  return model_changed;
}

/// Returns true if the instruction is supported
bool MachineModel::supported(FmaOp const &op) const {
  return fma_inst_.find(op) != fma_inst_.end();
}

/// Gets the instruction model
InstructionModel const &MachineModel::fma_model(FmaOp const &op) const {
  return fma_inst_.at(op);
}

/// Returns peak memory bandwidth in (GiB/s)
double MachineModel::memory_bandwidth() const {
  int memoryClockRate;
  cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
  double clock_MHz = double(2 * memoryClockRate) / 1000.0;
  int memoryBusWidth;
  cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth , 0);
  double busWidth_B = double(memoryBusWidth / 8);

  double bandwidth = busWidth_B * (double(clock_MHz * 1.0e6) / double(kBytesPerGiB));

  return bandwidth;
}

/// Determines the duration (in ms) for transferring the
/// given number of bytes.
double MachineModel::memory_duration(int64_t bytes) const {

  return double(bytes * 1000) / kBytesPerGiB / memory_bandwidth();
}

/// Determines peak math throughput in flops per clock across the chip
int MachineModel::flops_per_clock(FmaOp const &op) const {

  if (supported(op)) {
    InstructionModel const &perf = fma_model(op);

    int flops_per_clock_per_SM = (op.flops() * perf.datapaths_per_SM) / perf.issue_interval;

    return flops_per_clock_per_SM * device_.multiProcessorCount;
  }

  return 0;
}

/// Determines peak math throughput in GFLOP/s
double MachineModel::math_throughput(FmaOp const &op) const {

  int flops_per_clock_ = flops_per_clock(op);

  if (flops_per_clock_) {
    int clock_KHz;
    cudaDeviceGetAttribute(&clock_KHz, cudaDevAttrClockRate, 0);
    return double(flops_per_clock_) * double(clock_KHz) / 1.0e6;
  }

  return 0;
}

/// Determines the duration (in ms) for performing the
/// given number of flops.
double MachineModel::math_duration(FmaOp const &op, int64_t flops) const {

  if (supported(op)) {
    double gflops = math_throughput(op);    
    return double(flops) / gflops * 1000.0; 
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
