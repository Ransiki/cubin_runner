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
#pragma once

#include <cutlass/detail/dependent_false.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class DispatchPolicy,
  class... Args
>
class CollectiveEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Could not find an epilogue specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "detail.hpp"

//
// Gemm
//
#include "default_epilogue.hpp"
#include "default_epilogue_array.hpp"
#include "epilogue_fused_amax.hpp" // {$nv-internal-release}
#include "epilogue_fused_amax_with_aux_tensor.hpp" // {$nv-internal-release}
#include "epilogue_tensor_broadcast.hpp"
#include "sm70_epilogue_vectorized.hpp"
#include "sm70_epilogue_vectorized_array.hpp"
#include "sm90_epilogue_tma_warpspecialized.hpp"
#include "sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp"
#include "sm90_epilogue_array_tma_warpspecialized.hpp"
#include "sm100_epilogue_nosmem.hpp"  
#include "sm100_epilogue_array_nosmem.hpp"  
#include "sm100_epilogue_array_planar_complex_nosmem.hpp"  // {$nv-internal-release}
#include "sm100_epilogue_tma_warpspecialized.hpp" 
#include "sm100_epilogue_planar_complex_tma_warpspecialized.hpp" // {$nv-internal-release}
#include "sm100_epilogue_array_tma_warpspecialized.hpp" 
#include "sm100_epilogue_array_planar_complex_tma_warpspecialized.hpp" // {$nv-internal-release}
#include "sm100_epilogue_interleave_complex_tma_warpspecialized.hpp" // {$nv-internal-release}
//
// Conv
//
#include "sm100_epilogue_nq_2d_tiled.hpp"  // {$nv-internal-release}
#include "sm100_epilogue_nq_2d_tiled_nosmem.hpp"  // {$nv-internal-release}
#include "sm100_epilogue_nq_2d_tiled_tmem_shuffle.hpp"  // {$nv-internal-release}
#include "sm100_epilogue_tma_warpspecialized_strided_dgrad.hpp" // {$nv-internal-release}

/////////////////////////////////////////////////////////////////////////////////////////////////
