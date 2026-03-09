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

#pragma once

#include <iosfwd>
#include <ostream>
#include <stdexcept>
#include <memory>
#include <nvml.h>
#include <atomic>
#include <thread>

/**
 * \file
 * \brief NVML helper for monitoring GPU metrics 
 * {$nv-internal-release file}
 */

namespace cutlass {

/// C++ exception wrapper for NVML \p nvmlReturn_t 
class nvml_exception : public std::exception {
 public:
  /// Constructor
  nvml_exception(const char* msg = "", nvmlReturn_t err = NVML_ERROR_UNKNOWN) : msg(msg), err(err) {}

  /// Returns the underlying CUDA \p nvmlReturn_t
  nvmlReturn_t nvmlError() const { return err; }

 protected:
  /// Explanatory string
  const char* msg;

  /// Underlying NVML \p nvmlReturn_t
  nvmlReturn_t err;
};

/// Writes a cuda_exception instance to an output stream
inline std::ostream& operator<<(std::ostream& out, nvml_exception const& e) {
  return out << e.what() << ": " << nvmlErrorString(e.nvmlError());
}

class nvml_handle {
public: 

  explicit nvml_handle(int device = 0);

  ~nvml_handle();

  void release();

  /// Returns gpc clock in MHz
  unsigned gpc_clock() const;

  /// Returns gpu power in milliwatts
  unsigned gpu_power() const;

  /// Returns gpu power in degrees C
  unsigned gpu_temperature() const;

private:
  nvmlDevice_t nvml_gpu;
};

}  // namespace cutlass
