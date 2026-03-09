/******************************************************************************
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
 ******************************************************************************/

/**
 * \file
 * \brief NVML monitor GPU metrics 
 * {$nv-internal-release file}
 */

#include "cutlass/profiler/nvml_monitor.h"

namespace cutlass {
namespace profiler {

nvml_monitor::nvml_monitor(std::shared_ptr<nvml_handle> handle) 
    : handle_(handle)
    , endWarmup(false)
    , endProfiling(false) 
{ }

nvml_monitor::~nvml_monitor() {}

void nvml_monitor::start_warm_up() {
  if (handle_) {
    monitor = std::thread([&]() {
      do {
        result.warmup_metrics.push_back({
          handle_->gpc_clock(),
          handle_->gpu_temperature(),
          handle_->gpu_power() 
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } while(!endWarmup);
      
      do {
        result.profiling_metrics.push_back({
          handle_->gpc_clock(),
          handle_->gpu_temperature(),
          handle_->gpu_power() 
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } while(!endProfiling);
    });
  }
}

void nvml_monitor::start_profiling() {
  if (handle_) {
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw cuda_exception("Unable to synchronize GPU", err);
    }
    endWarmup.store(true);
  }
}

NvmlProfileResult nvml_monitor::end_profiling() {
  if (handle_) {
    endProfiling.store(true);
    if (monitor.joinable()) {
      monitor.join();
    }
  }
  return result;
}

}  // namespace profiling
}  // namespace cutlass
