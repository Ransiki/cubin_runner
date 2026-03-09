/******************************************************************************
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
 ******************************************************************************/

// {$nv-internal-release file}

/*! \file
    \brief Wrappers around PTX knobs, often enabled by the CUTLASS CMake system's detection
    of whether or not the knob is available in a given toolkit.

    NOTE: If you add a new knob to this list, you may also need to add a corresponding macro
    that controls whether or not it is enabled, and add logic for detecting the feature
    in the following files used by the CUTLASS CMake system:
      1. $CUTLASS_HOME/cmake/ptx_knobs.cu
      2. $CUTLASS_HOME/feature_detect.cmake
    Setting knobs introduced in new toolkits may cause PTXAS errors in older toolkits, and,
    thus, may require explicit protection via a macro.
*/

#pragma once

#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This knob avoids the use of L2 Descriptor.
static CUTLASS_DEVICE void global_knob_disable_implicit_mem_desc() {
#if CUDA_PTX_KNOB_DISABLE_IMPLICIT_MEM_DESC_ENABLED
  asm volatile(".pragma \"global knob DisableImplicitMemDesc\";\n" : : : "memory");
#endif
}

/// This knob enforces sector promotion if ptx support is not available.
static CUTLASS_DEVICE void global_knob_l2_prefetch() {
#if CUTLASS_AMPERE_256B_INTERNAL_L2_PREFETCH
  asm volatile(".pragma \"global knob sectorpromotion=256\";\n" : : : "memory");
#elif !defined(__clang__)
  asm volatile(".pragma \"global knob sectorpromotion=128\";\n" : : : "memory");
#endif
}

/// This knob avoids inserting extra LDS instructions between LDS{M} and LDGSTS instructions.
///  See GA100 hardware bug http://nvbugs/2549067
static CUTLASS_DEVICE void global_knob_disable_war_sw2549067() {
#if CUDA_PTX_KNOB_DISABLE_WAR_ENABLED
  asm volatile(".pragma \"global knob DisableWar_SW2549067\";\n": : : "memory");
#endif
}

/// Set the base load-store instruction-issue cadence to be every 12 cycles (avoids
/// backpressure/throttling)
static CUTLASS_DEVICE void global_knob_sched_res_busy_lsu_12cycles() {
#if CUDA_PTX_KNOB_SCHED_LSU_RES_BUSY_ENABLED
  asm volatile(".pragma \"global knob SchedResBusyLSU=12\";\n" : : : "memory");
#endif
}

/// Allow load-store instruction-issue candence to be influenced by access-size
static CUTLASS_DEVICE void global_knob_lsu_res_busy_size() {
#if CUDA_PTX_KNOB_LSU_RES_BUSY_SIZE_ENABLED
  asm volatile(".pragma \"global knob LsuResBusySize\";\n" : : : "memory");
#endif
}

// This knob avoids inserting extra LDS instructions between LDS{M} and LDGSTS instructions.
//  See GA100 hardware bug http://nvbugs/2549067
static CUTLASS_DEVICE void global_knob_sched_mem_no_alias() {
#if CUDA_PTX_KNOB_SCHED_MEM_NO_ALIAS_ENABLED
  asm volatile(".pragma \"global knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" : : : "memory");
#endif
}

// This knob avoids uniform predicate spill for GETT kernels.
// https://nvbugspro.nvidia.com/bug/5339770
static CUTLASS_DEVICE void global_knob_urf_promote_cond() {
#if CUDA_PTX_KNOB_URF_PROMOTE_JT_COND_ENABLED
  asm volatile(".pragma \"global knob URFPromoteJtCond=0\";\n" : : : "memory");
#endif
}

/// This block-begin knob signals the beginning of a basic-block region in which
/// subsequent IFENCEs are to additionally demarcate "dependence stages" in which
/// all asynchronous operations live-into that stage are to be protected by
/// the same scoreboard
static CUTLASS_DEVICE void block_begin_knob_advanced_sb_depstage_reuse() {
  asm volatile (".pragma \"set knob AdvancedSBDepStageReuse\";\n" : : : "memory");
}

/// This block-end knob signals the end of a basic-block region in which
/// subsequent IFENCEs are to additionally demarcate "dependence stages" in which
/// all asynchronous operations live-into that stage are to be protected by
/// the same scoreboard
static CUTLASS_DEVICE void block_end_knob_advanced_sb_depstage_reuse() {
  asm volatile (".pragma \"reset knob AdvancedSBDepStageReuse\";\n" : : : "memory");
}

/// This block-begin knob signals the beginning of a sass code block region in which
/// sass code in this region will be put at the end of all sass code file.
static CUTLASS_DEVICE void block_begin_knob_cold_block() {
    asm volatile(".pragma \"set knob ColdBlock\";\n" : : : "memory");
}

/// This block-begin knob signals the end of a sass code block region in which
/// sass code in this region will be put at the end of all sass code file.
static CUTLASS_DEVICE void block_end_knob_cold_block() {
    asm volatile(".pragma \"reset knob ColdBlock\";\n" : : : "memory");
}

/// RAII wrapper for cold block knob
struct ColdBlockKnobScopeGuard {
  CUTLASS_DEVICE
  ColdBlockKnobScopeGuard(){
    block_begin_knob_cold_block();
  }

  CUTLASS_DEVICE
  ~ColdBlockKnobScopeGuard(){
    block_end_knob_cold_block();
  }
};

/// Knobs required by https://nvbugs/4110466
static CUTLASS_DEVICE void global_knob_elect_one_r2ur_placement() {
  asm volatile(".pragma \"global knob ForceLateCommoning=1\";\n" : : : "memory");
  asm volatile(".pragma \"global knob HoistLate=3\";\n" : : : "memory");
}

/// Needed for TRYWAIT/SEL WR SB dependency elimination and RD SB dependency on BRA
/// https://nvbugs/4444998 (umbrella bug)
static CUTLASS_DEVICE void global_knob_trywait_sel_sb_dep() {
  asm volatile(".pragma \"global knob SchedSyncsPhasechkLatency=90\";\n" : : : "memory");
}

/// opt-in some ocg knobs to boost perf on solid path, remove the block when ocg make the knobs as default
static CUTLASS_DEVICE void global_knob_demote_to_pred_blockidx_limit() {
  asm volatile(".pragma \"global knob DemoteToPredBlockIdxLimit=30\";\n" : : : "memory");
}

/// Knob to hoist LDC/LDCU for prologue perf (https://nvbugs/4816139)
static CUTLASS_DEVICE void global_knob_ldc_ldcu_hoisting() {
  asm volatile(".pragma \"global knob HoistCBOMode=3\";\n" : : : "memory");
}

/// WAR for https://nvbugs/4950853, delete after OCG fix
static CUTLASS_DEVICE void global_knob_mbarrier_init_mapping() {
#if __CUDA_ARCH__ >= 1000 && CUTLASS_FORCE_VECTOR_MBARRIER_INIT
  // Forces mbarrier.init to lower to STS
  asm volatile(".pragma \"global knob MbarrierInitRegMapping=2\";\n" : : : "memory");
#elif __CUDA_ARCH__ >= 1000
  // Forces mbarrier.init to lower to SYNCS.EXCH
  asm volatile(".pragma \"global knob MbarrierInitRegMapping=0\";\n" : : : "memory");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
