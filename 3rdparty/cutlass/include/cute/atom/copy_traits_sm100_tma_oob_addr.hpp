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

// {$nv-release-never file}

#pragma once

/// @file copy_traits_sm100_tma_oob_addr.hpp
/// @brief Functions for get tma oob addr mode from tensor layout. Only applies to >= SM100 TMA.

#include "cute/numeric/numeric_types.hpp" // cute::bits_to_bytes
#include "cute/tensor.hpp" // cute::Tensor

namespace cute::detail {

static constexpr size_t TMAOobModeBase128kBSizeInBytes = 128 * 1024;

template <class Engine, class Layout>
TMA::OobAddrGenMode
get_tma_oob_addr_mode(cute::Tensor<Engine,Layout> const& tensor)
{
  size_t tensorSize = cute::bits_to_bytes(
                        cute::cosize(tensor.layout()) *
                        cute::sizeof_bits<typename Engine::value_type>::value);

  if (tensorSize >= TMAOobModeBase128kBSizeInBytes) {
    return TMA::OobAddrGenMode::OOB_ADDR_GEN_MODE_BASE_128kB;
  }
  return TMA::OobAddrGenMode::OOB_ADDR_GEN_MODE_LIB_4kB;
}

} // namespace cute::detail
