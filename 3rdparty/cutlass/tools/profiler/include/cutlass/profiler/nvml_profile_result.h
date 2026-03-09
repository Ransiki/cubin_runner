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
/* \file
   \brief Defines a math function
   {$nv-internal-release file}
*/

#pragma once

#include <vector>

#include "cutlass/cutlass.h"

// CUTLASS Profiler includes
#include "enumerated_types.h"

// CUTLASS Library includes
#include "cutlass/library/library.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile result object
struct NvmlProfileResult {

  /// Index of problem
  size_t problem_index;

  /// library::Provider
  library::Provider provider;

  /// Operation kind
  library::OperationKind op_kind;

  /// Operation name
  std::string operation_name;

  /// Stringified vector of argument values
  std::vector<std::pair<std::string, std::string> > arguments;

  struct GPUMetrics {
    unsigned gpcclk;
    unsigned temperature;
    unsigned power;
  };

  std::vector<GPUMetrics> warmup_metrics;

  std::vector<GPUMetrics> profiling_metrics;

  /// Ctor
  NvmlProfileResult(): 
    problem_index(0),
    op_kind(library::OperationKind::kInvalid),
    provider(library::Provider::kInvalid) 
  { }
  
};

using NvmlProfileResultVector = std::vector<NvmlProfileResult>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

