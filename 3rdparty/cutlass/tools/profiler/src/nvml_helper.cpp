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
 * \brief NVML helper for monitoring GPU metrics 
 * {$nv-internal-release file}
 */

#include <iosfwd>
#include <stdexcept>
#include <nvml.h>

#include "cutlass/profiler/nvml_helper.h"

namespace cutlass {

nvml_handle::nvml_handle(int device) {
  nvmlReturn_t nvml_status = nvmlInit_v2();
  if (nvml_status != NVML_SUCCESS) {
    throw nvml_exception("Error initializing nvml", nvml_status); 
  }
  nvml_status = nvmlDeviceGetHandleByIndex_v2(device, &nvml_gpu);
  if (nvml_status != NVML_SUCCESS) {
    try {
      release();
    } 
    catch(nvml_exception) {
      throw nvml_exception("Error releasing after error getting device handle", nvml_status); 
    }
    throw nvml_exception("Error getting device handle", nvml_status);
  }
}

nvml_handle::~nvml_handle() {}

void nvml_handle::release() {
  nvmlReturn_t nvml_status = nvmlShutdown();
  if (nvml_status != NVML_SUCCESS) {
    throw nvml_exception("Error shutting down nvml", nvml_status);
  }
}

/// Returns gpc clock in MHz
unsigned nvml_handle::gpc_clock() const {
  unsigned clock;
  nvmlReturn_t nvml_status = nvmlDeviceGetClock(nvml_gpu, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &clock);
  if (nvml_status != NVML_SUCCESS) {
    throw nvml_exception("Error getting clock", nvml_status);
  }
  return clock;
}

/// Returns gpu power in milliwatts
unsigned nvml_handle::gpu_power() const {
  unsigned power;
  nvmlReturn_t nvml_status = nvmlDeviceGetPowerUsage(nvml_gpu, &power);
  if (nvml_status != NVML_SUCCESS) {
    throw nvml_exception("Error getting power usage", nvml_status); 
  }
  return power;
}

/// Returns gpu power in degrees C
unsigned nvml_handle::gpu_temperature() const {
  unsigned degrees;
  nvmlReturn_t nvml_status = nvmlDeviceGetTemperature(nvml_gpu, NVML_TEMPERATURE_GPU, &degrees);
  if (nvml_status != NVML_SUCCESS) {
    throw nvml_exception("Error getting temperature", nvml_status); 
  }
  return degrees;
}

}  // namespace cutlass
