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

#pragma once

#include <unordered_map>

#include "cutlass/library/library.h"
#include "cutlass/gemm/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fused multiply-add operator (dense)
struct FmaOp {

  gemm::GemmCoord shape;
  library::OpcodeClassID opcode_class;
  library::NumericTypeID A;
  library::NumericTypeID B;
  library::NumericTypeID C;

  //
  // Methods
  //

  FmaOp(): 
    shape(0, 0, 0), 
    opcode_class(library::OpcodeClassID::kSimt),
    A(library::NumericTypeID::kInvalid), 
    B(library::NumericTypeID::kInvalid), 
    C(library::NumericTypeID::kInvalid) { }

  FmaOp(
    gemm::GemmCoord shape, 
    library::OpcodeClassID opcode_class,
    library::NumericTypeID A, 
    library::NumericTypeID B, 
    library::NumericTypeID C
  ):
    shape(shape), opcode_class(opcode_class), A(A), B(B), C(C) { }

  /// Number of flops performed per instruction (1 fma = 2 flops)
  int flops() const {
    return 2 * shape.m() * shape.n() * shape.k();
  }

  /// Equality operator
  inline bool operator==(FmaOp const &rhs) const {
    return shape == rhs.shape &&
      opcode_class == rhs.opcode_class && 
      A == rhs.A && B == rhs.B && C == rhs.C;
  }

  /// Inequality operator
  inline bool operator!=(FmaOp const &rhs) const {
    return !(*this == rhs);
  }
};

/// Hash function for FmaOp
struct HashFmaOp {
  size_t operator()(FmaOp const &op) const {
    
    std::hash<int> hash_int;

    size_t res = 0;

    res = res * 7 + hash_int(op.shape.m());
    res = res * 7 + hash_int(op.shape.n());
    res = res * 7 + hash_int(op.shape.k());
    res = res * 7 + hash_int(int(op.opcode_class));
    res = res * 7 + hash_int(int(op.A));
    res = res * 7 + hash_int(int(op.B));
    res = res * 7 + hash_int(int(op.C));

    return res;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Performance model of an instruction
struct InstructionModel {

  int issue_interval;       ///< number of cycles between issues of this instruction to the same datapath
  int datapaths_per_SM;     ///< number of virtual datapaths that may simultaneously issue per SM
  int throttle;

  //
  // Methods
  //

  InstructionModel(): issue_interval(0), datapaths_per_SM(0), throttle(1) { }

  InstructionModel(
    int issue_interval, 
    int datapaths_per_SM,
    int throttle = 1
  ): 
    issue_interval(issue_interval), 
    datapaths_per_SM(datapaths_per_SM),
    throttle(throttle) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Maps FmaOp to instruction models
using FmaInstructionModel = std::unordered_map<FmaOp, InstructionModel, HashFmaOp>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Machine model 
class MachineModel {

  cudaDeviceProp device_;           ///< CUDA device properties
  FmaInstructionModel fma_inst_;    ///< dense fused multiply-add instructions

public:

  MachineModel();
  MachineModel(cudaDeviceProp const &prop);

  /// Updates the machine model
  void update(cudaDeviceProp const &prop);

  /// Compute capability
  int compute_capability() const;

  /// Measures some properties via a CUDA kernel launch. Returns true if the model has changed
  /// due to what was measured.
  bool measure();

  /// Returns true if the instruction is supported
  bool supported(FmaOp const &op) const;

  /// Gets the instruction model
  InstructionModel const &fma_model(FmaOp const &op) const;

  /// Returns peak memory bandwidth in (GiB/s)
  double memory_bandwidth() const;

  /// Determines the duration (in ms) for transferring the
  /// given number of bytes.
  double memory_duration(int64_t bytes) const;

  /// Determines math throughput across the chip in flops per clock (1 fma = 2 flops)
  int flops_per_clock(FmaOp const &op) const;

  /// Determines peak math throughput in GFLOP/s
  double math_throughput(FmaOp const &op) const;

  /// Determines the duration (in ms) for performing the
  /// given number of flops.
  double math_duration(FmaOp const &op, int64_t flops) const;

private:

  void init_fma_inst_sm5x();
  void init_fma_inst_sm60();
  void init_fma_inst_sm61();
  void init_fma_inst_sm70();
  void init_fma_inst_sm72();
  void init_fma_inst_sm75();
  void init_fma_inst_sm80();
  void init_fma_inst_sm86();
  void init_fma_inst_sm90();

  void update_fma_inst_sm75(bool f32_full_speed);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
