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
   \brief Class performing output during profiling
   {$nv-internal-release file}
*/

#include "cutlass/profiler/nvml_profile_report.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

NvmlProfileReport::NvmlProfileReport(Options const &options, std::vector<std::string> const& argument_names, library::OperationKind const &op_kind) : options_(options), op_kind_(op_kind) {
  if (op_kind == library::OperationKind::kGemm) {
    output_file_ = std::ofstream(options.report.nvml_results_output_path);
    output_file_ << "Operation,Warmup,gpcClck(MHz),temperature(C),power(mW)";
    for (auto name : argument_names) {
      output_file_ << ',' << name;
    }
    output_file_ << std::endl;
  }
}

NvmlProfileReport::~NvmlProfileReport() {
  if (op_kind_ == library::OperationKind::kGemm) {
    output_file_.close();
  }
}

void NvmlProfileReport::append_result(NvmlProfileResult result) {
  for (auto metrics : result.warmup_metrics) {
    output_file_ << result.operation_name 
      << ",true"
      << ',' << metrics.gpcclk
      << ',' << metrics.temperature
      << ',' << metrics.power;
    for (auto arg : result.arguments) {
      output_file_ << "," << arg.second;
    }
    output_file_ << std::endl;
  }
  for (auto metrics : result.profiling_metrics) {
    output_file_ << result.operation_name 
      << ",false"
      << ',' << metrics.gpcclk
      << ',' << metrics.temperature
      << ',' << metrics.power;
    for (auto arg : result.arguments) {
      output_file_ << "," << arg.second;
    }
    output_file_ << std::endl;
  }
}

void NvmlProfileReport::append_results(NvmlProfileResultVector &&results) {
  for (auto result : results) {
    append_result(result);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

