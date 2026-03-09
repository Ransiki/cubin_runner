/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Grid dependent control (GDC) helpers for programmatic dependent launches (PDL).
*/

#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#ifndef CUTLASS_GDC_ENABLED
  #if (CUDA_BARRIER_ENABLED && \
    defined(CUTLASS_ENABLE_GDC_FOR_SM90) && \
     __CUDACC_VER_MAJOR__ >= 12 && \
     defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL))
    #define CUTLASS_GDC_ENABLED
  #endif
  #if (defined(CUTLASS_ENABLE_GDC_FOR_SM100) && \
     __CUDACC_VER_MAJOR__ >= 12 && \
     defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL))
    #define CUTLASS_GDC_ENABLED
  #endif
#endif

#ifndef CUTLASS_GDC_ENABLED
  #if(CUDA_BARRIER_ENABLED && \
    defined(CUTLASS_ENABLE_GDC_FOR_SM100) && \
    defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 1000 &&\
        (defined(__CUDA_ARCH_FEAT_SM100_ALL) || CUDA_ARCH_FAMILY(1000))) || \
     (__CUDA_ARCH__ == 1010 &&\
        (defined(__CUDA_ARCH_FEAT_SM101_ALL) || CUDA_ARCH_FAMILY(1010))) || \
     (__CUDA_ARCH__ == 1030 &&\
        (defined(__CUDA_ARCH_FEAT_SM103_ALL) || CUDA_ARCH_FAMILY(1030))) || \
     (__CUDA_ARCH__ == 1200 &&\
        (defined(__CUDA_ARCH_FEAT_SM120_ALL) || CUDA_ARCH_FAMILY(1200))) || \
     (__CUDA_ARCH__ == 1210 &&\
        (defined(__CUDA_ARCH_FEAT_SM121_ALL) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210)))))
    #define CUTLASS_GDC_ENABLED
  #endif
#endif

// {$nv-internal-release begin}

// Separate check for unreleased archs as unable to add #nv-internal-release inline comments with multiline MACROs
// Merge with the above checks when exposing an arch.
#ifndef CUTLASS_GDC_ENABLED
  #if(CUDA_BARRIER_ENABLED && \
    defined(CUTLASS_ENABLE_GDC_FOR_SM100) && \
    defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 1020 &&\
        (defined(__CUDA_ARCH_FEAT_SM102_ALL) || CUDA_ARCH_FAMILY(1020))) || \
     (__CUDA_ARCH__ == 1070 &&\
        (defined(__CUDA_ARCH_FEAT_SM107_ALL) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1070)))))
    #define CUTLASS_GDC_ENABLED
  #endif
#endif
// {$nv-internal-release end}

namespace cutlass {
namespace arch {

// Issuing the launch_dependents instruction hints a dependent kernel to launch earlier
// launch_dependents doesn't impact the functionality but the performance:
// Launching a dependent kernel too early can compete with current kernels,
// while launching too late can lead to a long latency.
CUTLASS_DEVICE
void launch_dependent_grids() {
#if (defined(CUTLASS_GDC_ENABLED))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Issuing the griddepcontrol.wait instruction enforces no global memory access
// prior to this istruction. This ensures the correctness of global memory access
// when launching a dependent kernel earlier.
CUTLASS_DEVICE
void wait_on_dependent_grids() {
#if (defined(CUTLASS_GDC_ENABLED))
  asm volatile("griddepcontrol.wait;");
#endif
}

// Enable kernel-level query regarding whether the GDC feature is turned on
#if (defined(CUTLASS_GDC_ENABLED))
static constexpr bool IsGdcGloballyEnabled = true;
#else
static constexpr bool IsGdcGloballyEnabled = false;
#endif

// {$nv-internal-release begin}
///////////////////////////////////////////////////////////////////////////////
/// Early smem release manager
///////////////////////////////////////////////////////////////////////////////

template <
  class KernelSharedStorage_
>
class SmemEarlyReleaseManager {

public:
  using BarrierType = typename cutlass::arch::NamedBarrier;

  // GDC resizes the shared memory space.
  //
  // The original layout (SM100):
  // CUDA reserved | PipelineStorage | CLCResponse | ... | EpilogueTensorStorage | MainloopTensorStorage
  //
  // The resized layout (SM100):
  // CUDA reserved | PipelineStorage | CLCResponse | ... | EpilogueTensorStorage
  //
  // The original layout (SM90):
  // CUDA reserved | PipelineStorage | EpilogueTensorStorage | MainloopTensorStorage
  //
  // The resized layout (SM90):
  // CUDA reserved | PipelineStorage | EpilogueTensorStorage
  //
  // By doing this, MainloopTensorStorage is released for a dependent kernel to launch early.
  //
  // To launch a dependent kernel sooner, as we always put MainloopTensorStorage in the end of SharedStorage,
  // so the unrounded resized shared memory bytes should be the bytes of (1024 /* CUDA reserved */ + ToTal Shared Storage - Shared Mainloop Tensor Storage).
   using KernelSharedStorage = KernelSharedStorage_;
   using KernelSharedMainloopTensorStorage = typename KernelSharedStorage::TensorStorage::MainloopTensorStorage;
   static constexpr int ReservedAndPaddingSmemSize = 1024 /* CUDA reserved */;
   static constexpr int UnRoundedResizedSmemSize = ReservedAndPaddingSmemSize + (sizeof(KernelSharedStorage) - sizeof(KernelSharedMainloopTensorStorage));

  // Shared memory resizing requires a target size divisible by 128 bytes, so conduct rounding here.
  static constexpr int GdcSmemAlignment = 128;
  static constexpr int ResizedSmemSize = (UnRoundedResizedSmemSize + GdcSmemAlignment - 1) / GdcSmemAlignment * GdcSmemAlignment;

  // NamedBarrier constructs by defining thread_cnt_ and reserved_named_barriers_,
  // referring to cutlass/arch/barrier.h
  CUTLASS_DEVICE explicit 
  SmemEarlyReleaseManager() : 
    smem_release_barrier_( /*thread_cnt_*/ blockDim.x * blockDim.y * blockDim.z, 
                           /*reserved_named_barriers_*/ arch::ReservedNamedBarriers::GdcSmemEarlyReleaseBarrier) {}

  // Resize shared memory and barrier arrive
  CUTLASS_DEVICE
  void resize_for_self_non_blocking() {
#if (defined(CUTLASS_GDC_ENABLED)) && defined(CUTLASS_ENABLE_EXTENDED_PTX) && (CUTLASS_ENABLE_EXTENDED_PTX)
    asm volatile("setsmemsize.sync.u32 %0;\n" : : "n"(ResizedSmemSize));
    smem_release_barrier_.arrive_unaligned();
#endif
  }

  // Resize shared memory, wait on the barrier, and commit resizing.
  CUTLASS_DEVICE
  void resize_for_all_blocking() {
#if (defined(CUTLASS_GDC_ENABLED)) && defined(CUTLASS_ENABLE_EXTENDED_PTX) && (CUTLASS_ENABLE_EXTENDED_PTX)
    smem_release_barrier_.arrive_and_wait();
    asm volatile("setsmemsize.sync.u32 %0;\n" : : "n"(ResizedSmemSize));

    // Elect one thread of a warp to commit shared memory resizing
    if (cute::elect_one_sync()) {
      asm volatile("setsmemsize.flush.u32; \n");
    }
#endif
  }

private:
  BarrierType smem_release_barrier_;

};
// {$nv-internal-release end}

} // namespace arch
} // namespace cutlass
