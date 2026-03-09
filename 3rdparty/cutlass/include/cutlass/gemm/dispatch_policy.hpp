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

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp" // cute::false_type
#include "cute/atom/copy_traits_sm100.hpp"
#include "cutlass/detail/collective/sm103_kernel_type.hpp"
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

template <class T, template <int...> class U>
struct is_kernel_tag_of : cute::false_type {};

template <template <int...> class U, int... Args>
struct is_kernel_tag_of<U<Args...>, U> : cute::true_type {};

template <class T, template <int...> class U>
constexpr bool is_kernel_tag_of_v = is_kernel_tag_of<T, U>::value;

template <class T, template <int,bool> class U>
struct is_asymmetric_dma_kernel_tag_of : cute::false_type {};

template <template <int, bool> class U, int I0, bool B0>
struct is_asymmetric_dma_kernel_tag_of<U<I0, B0>, U> : cute::true_type {};

template <class T, template <int, bool> class U>
constexpr bool is_asymmetric_dma_kernel_tag_of_v = \
                              is_asymmetric_dma_kernel_tag_of<T, U>::value;

}

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

//////////////////////////////////////////////////////////////////////////////

namespace detail {

// TODO https://jirasw.nvidia.com/browse/CFK-13241 Remove it when we unify several apis in input transform kernel layer. // {$nv-release-never}
enum class KernelInputTransformType {
    FastF32,
    InterleavedComplexTF32,
    MixedInput
};

} // namespace detail

//////////////////////////////////////////////////////////////////////////////

namespace kernel::detail {

// Has_SwapAB<T>::value will be true only if:
//   class T has member SwapAB and T::SwapAB is true
template <typename T, typename = void>
struct Has_SwapAB { static constexpr bool value = false; };

template <typename T>
struct Has_SwapAB <T, CUTE_STL_NAMESPACE::void_t<decltype(T::SwapAB)>>
{ static constexpr bool value = T::SwapAB; };

template <typename T>
static constexpr bool Has_SwapAB_v = Has_SwapAB<T>::value;

// additional producer warp role check for block scaling mainloop
template<typename T>
struct HasAuxiliaryLoad : cute::false_type{};

template <typename T>
static constexpr bool HasAuxiliaryLoad_v = HasAuxiliaryLoad<T>::value;

} // namespace kernel::detail

//////////////////////////////////////////////////////////////////////////////

//
// Kernel schedule policies (the base class tags, one for each kernel layer file)
//
struct KernelMultistage { };
struct KernelPtrArrayMultistage { };
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { 
  static constexpr int SchedulerPipelineStageCount = 0;
};
struct KernelTmaWarpSpecializedCooperative { 
  static constexpr int SchedulerPipelineStageCount = 0;
};

struct KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpong { };

// FP8 related policies (including Blocked Scaled Accumulation)
struct KernelTmaWarpSpecializedCooperativeFP8Blockwise: KernelTmaWarpSpecializedCooperative { };
struct KernelTmaWarpSpecializedPingpongFP8Blockwise: KernelTmaWarpSpecializedPingpong { };
struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise: KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise: KernelPtrArrayTmaWarpSpecializedPingpong { };

using KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum = KernelTmaWarpSpecializedCooperativeFP8Blockwise;
using KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum = KernelTmaWarpSpecializedPingpongFP8Blockwise;
using KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum = KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
using KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum = KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;

// Policies to opt into mixed type GEMMs
struct KernelTmaWarpSpecializedMixedInput : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongMixedInput : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeMixedInput: KernelTmaWarpSpecializedCooperative { };

//////////////////////////////////////////////////////////////////////////////

//
// Builder dispatch policies (not a part of the main CUTLASS layers, simply used to opt into
// specific collective builder dispatches)
//

// FP8 related policies (including Fast Accumulation)
struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum : KernelPtrArrayTmaWarpSpecializedPingpong { };

//////////////////////////////////////////////////////////////////////////////

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

// 2 stage pipeline through 1 stage in smem, 1 in rmem, WITHOUT predicated gmem loads
struct MainloopSm70TwoStageUnpredicated {
  constexpr static int Stages = 2;
  using ArchTag = arch::Sm70;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// 2 stage pipeline through 1 stage in smem, 1 in rmem, with predicated gmem loads
struct MainloopSm70TwoStage {
  constexpr static int Stages = 2;
  using ArchTag = arch::Sm70;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, WITHOUT predicated gmem loads
template<int Stages_>
struct MainloopSm80CpAsyncUnpredicated {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::Sm80;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm80CpAsync {
  constexpr static int Stages = Stages_;
  using ArchTag = cute::conditional_t<(size(ClusterShape_{}) > 1), arch::Sm90, arch::Sm80>;
  using Schedule = KernelMultistage;
  using ClusterShape = ClusterShape_;
};

// n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads for SM100 Simt Ptr-Array
template<int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm80ArrayCpAsync {
  constexpr static int Stages = Stages_;
  using ArchTag = cute::conditional_t<(size(ClusterShape_{}) > 1), arch::Sm90, arch::Sm80>;
  using Schedule = KernelPtrArrayMultistage;
  using ClusterShape = ClusterShape_;
};

// n-buffer in smem (cp.async), pipelined with Hopper GMMA, with predicated gmem loads, warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelCpAsyncWarpSpecialized
>
struct MainloopSm90CpAsyncGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// n-buffer in smem (cp.async), pipelined with Hopper GMMA, with predicated gmem loads, warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelCpAsyncWarpSpecialized
>
struct MainloopSm90CpAsyncGmmaRmemAWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, static schedule between TMA and GMMA
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  int PipelineAsyncMmaStages_ = 1
>
struct MainloopSm90TmaGmma {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  constexpr static int PipelineAsyncMmaStages = PipelineAsyncMmaStages_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelTma;
};

// {$nv-internal-release begin}
// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, static schedule between TMA and GMMA
// A operand for GMMA consumed from registers rather than streaming from shared memory.
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>
  // TODO? int PipelineAsyncMmaStages_ = 1
>
struct MainloopSm90TmaGmmaRmemA {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm90;
  using Schedule = KernelTma;
};
// {$nv-internal-release end}

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// With GMMA's A data from registers.
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaRmemAWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};


template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// For FP8 kernels
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaWarpSpecializedFP8
  : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};


// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// For FP8 kernels with Blockwise (Software) Scaling
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperativeFP8Blockwise
>
struct MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8
  : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedCooperativeFP8Blockwise> ||
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedPingpongFP8Blockwise>,
    "KernelSchedule must be one of the warp specialized FP8 block scale policies");
};

// {$nv-internal-release begin}
// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// Inputs are transformed into multiple FP16 compute matrices ahead-of-time before this kernel was called
/// NOTE: On-the-fly transformation only implemented for Blackwell for now as Hopper's GMMA is harder to
///       feed than Blackwell UMMA due to RMEM being "too close" to the SIMT units to create MMA buffers
template<
  int Stages_,
  // Number of FP16 compute matrices to use
  int NumComputeMtxs_,
  // Whether to use staged accumulators (periodically promote a separate accumulator to fix the RZ vs RN issue)
  bool StagedAccum_ = false,
  // Whether to compute less-significant bands (a_small*b_small) in addition to major bands (a_big*b_big + a_small*b_big + a_big*b_small)
  bool AllBands_ = false,
  // Need a separate marker for complex kernels since inputs may have alternative layouts that make deduction from input type not viable
  bool Complex_ = false,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecializedFastF32PreTransformed {
  constexpr static int Stages = Stages_;
  constexpr static int NumComputeMtxs = NumComputeMtxs_;
  constexpr static bool StagedAccum = StagedAccum_;
  constexpr static bool AllBands = AllBands_;
  constexpr static bool Complex = Complex_;
  static_assert(Stages % NumComputeMtxs == 0, "Number of stages must be a multiple of NumComputeMtxs.");
  static_assert(NumComputeMtxs > 1 && NumComputeMtxs <= 3, "Only 2 or 3 Mtxs are supported.");
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule;
  using ArchTag = arch::Sm90;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// B is transformed ahead-of-time into FP16 while A is being transformed on-the-fly and fed from the register side
template<
  int Stages_,
  // Number of FP16 compute matrices to use
  int NumComputeMtxs_,
  // Whether to use staged accumulators (periodically promote a separate accumulator to fix the RZ vs RN issue)
  bool StagedAccum_ = false,
  // Whether to compute less-significant bands (a_small*b_small) in addition to major bands (a_big*b_big + a_small*b_big + a_big*b_small)
  bool AllBands_ = false,
  // Need a separate marker for complex kernels since inputs may have alternative layouts that make deduction from input type not viable
  bool Complex_ = false,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative,
  class ElementScalar_ = float
>
struct MainloopSm90TmaGmmaRmemAWarpSpecializedFastF32PreTransformed {
  constexpr static int Stages = Stages_;
  constexpr static int NumComputeMtxs = NumComputeMtxs_;
  constexpr static bool StagedAccum = StagedAccum_;
  constexpr static bool AllBands = AllBands_;
  constexpr static bool Complex = Complex_;
  static_assert(!StagedAccum, "StagedAccum not implemented for RS kernels.");
  static_assert(NumComputeMtxs > 1 && NumComputeMtxs <= 3, "Only 2 or 3 Mtxs are supported.");
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule;
  using ElementScalar = ElementScalar_;
  using ArchTag = arch::Sm90;
};
// {$nv-internal-release end}

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  constexpr static int PipelineAsyncMmaStages = 1;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelSchedule> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, KernelSchedule>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
// For FP8 kernels
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedFP8
  : MainloopSm90ArrayTmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelSchedule> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, KernelSchedule>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper sparse GMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecializedSparse {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// For slow-accumulation sparse FP8 kernels
template<
  int Stages,
  class ClusterShape = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecializedSparseFP8 
  : MainloopSm90TmaGmmaWarpSpecializedSparse<Stages, ClusterShape, KernelSchedule> {
};

// Mixed precision version n-buffer in rmem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedCooperative> ||
    cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedPingpong>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// For FP8 kernels with Block Scaling
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedBlockwise
  : MainloopSm90ArrayTmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_any_of_v<
      KernelSchedule,
      KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise,
      KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise
    >,
    "KernelSchedule must be one of the warp specialized FP8 block scale policies");
};

//////////////////////////////////////////////////////////////////////////////

//
// Kernel Scheduler Tag
//

// Dense GEMM: SM100 tensor op policy that applies to both 1SM and 2SM MMA atoms
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelMixedTmaCpAsyncWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Gemm with block scaling factors
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedBlockScaledSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedMmaTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedMmaTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedBlockScaledSm103 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedBlockScaledSm103 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Sparse Gemm
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelSparseTmaWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Sparse Gemm with block scaling factors
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelSparseTmaWarpSpecializedBlockScaledSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// InputTransform GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedInputTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Mixed Input Transform GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedMixedInputTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Ptr-Array Dense GEMM: SM100 tensor op policy that applies to both 1SM and 2SM MMA atoms
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Ptr-Array Block Scaled GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedBlockScaledSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Ptr-Array InputTransform GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedInputTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};


// SM120 kernel schedules
template<int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedCooperativeSm120 : KernelTmaWarpSpecializedCooperative { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template<int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedPingpongSm120 : KernelTmaWarpSpecializedPingpong { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};


template<int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedCooperativeBlockScaledSm120 : KernelTmaWarpSpecializedCooperative { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template<int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedPingpongBlockScaledSm120 : KernelTmaWarpSpecializedPingpong { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

// SM120 dense Ptr-array kernel schedules
template<int SchedulerPipelineStageCount_>
struct KernelPtrArrayTmaWarpSpecializedCooperativeSm120 : KernelPtrArrayTmaWarpSpecializedCooperative { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template<int SchedulerPipelineStageCount_>
struct KernelPtrArrayTmaWarpSpecializedPingpongSm120 : KernelPtrArrayTmaWarpSpecializedPingpong { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template<int SchedulerPipelineStageCount_>
struct KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120 : KernelPtrArrayTmaWarpSpecializedCooperative { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template<int SchedulerPipelineStageCount_>
struct KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120 : KernelPtrArrayTmaWarpSpecializedPingpong { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

// SM120 sparse kernel schedules
template<int SchedulerPipelineStageCount_, bool isAsymmetric_>
struct KernelTmaWarpSpecializedCooperativeSparseSm120 {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr bool isAsymmetric = isAsymmetric_;
};

template<int SchedulerPipelineStageCount_, bool isAsymmetric_>
struct KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120 {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr bool isAsymmetric = isAsymmetric_;
};

// SM120 blockwise kernel schedules
template <int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedCooperativeBlockwiseScalingSm120 : KernelTmaWarpSpecializedCooperative {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template <int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedPingpongBlockwiseScalingSm120 : KernelTmaWarpSpecializedPingpong {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template <int SchedulerPipelineStageCount_>
struct KernelPtrArrayTmaWarpSpecializedCooperativeBlockwiseScalingSm120 : KernelPtrArrayTmaWarpSpecializedCooperative {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template <int SchedulerPipelineStageCount_>
struct KernelPtrArrayTmaWarpSpecializedPingpongBlockwiseScalingSm120 : KernelPtrArrayTmaWarpSpecializedPingpong {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

// Auxiliary Load Tag.

namespace kernel::detail {

template<
  int Stages,
  class ClusterShape,
  class KernelSchedule
>
struct HasAuxiliaryLoad<
  MainloopSm90ArrayTmaGmmaWarpSpecializedBlockwise<
    Stages,
    ClusterShape,
    KernelSchedule
  >
> : cute::true_type{};

template<
  int Stages,
  class ClusterShape,
  class KernelSchedule
>
struct HasAuxiliaryLoad<
  MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8<
    Stages,
    ClusterShape,
    KernelSchedule
  >
> : cute::true_type{};

} // namespace kernel::detail

//////////////////////////////////////////////////////////////////////////////

//
// Collective Builder Tag Property
//

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
//          SM100 Dispatch Policies
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Builder Tag Base Dispatch Policies
struct KernelSchedule1Sm {};
struct KernelSchedule2Sm {};
struct KernelScheduleSm100 {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100DenseGemm : KernelScheduleSm100 {};   // Base policy
// Dense GEMM: Specialize for 1SM vs 2SM
struct KernelTmaWarpSpecialized1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100DenseGemm {};  // Use for 1SM Dense GEMM Kernels for Collective Mainloop Builder
struct KernelTmaWarpSpecialized2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100DenseGemm {};  // Use for 2SM Dense GEMM Kernels for Collective Mainloop Builder
struct KernelWarpSpecialized1SmSm100    final : KernelSchedule1Sm, KernelScheduleSm100DenseGemm {};  // Use for 1SM Dense GEMM Kernels for Collective Mainloop Builder Without TMA
struct KernelMixedTmaCpAsyncWarpSpecialized1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100DenseGemm {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Ptr-Array Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Dense GEMM + (Ptr Array or Group GEMM)
struct KernelScheduleSm100PtrArrayDenseGemm : KernelScheduleSm100DenseGemm {};
// Ptr-Array Dense GEMM: Specialize for 1SM vs 2SM
struct KernelPtrArrayTmaWarpSpecialized1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayDenseGemm {};
struct KernelPtrArrayTmaWarpSpecialized2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayDenseGemm {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Blockwise GEMM + Ptr-Array GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100Blockwise  : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedBlockwise1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100Blockwise {};
struct KernelTmaWarpSpecializedBlockwise2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100Blockwise {};

struct KernelScheduleSm100PtrArrayBlockwise  : KernelScheduleSm100Blockwise {};
struct KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayBlockwise {};
struct KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayBlockwise {};

// {$nv-internal-release begin}
///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 FastF32 (4xFP16) GEMM via Blockwise SF for pre-transformed inputs
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100BlockwiseFastFP32Gemm  : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedBlockwiseFastFP32Gemm1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100BlockwiseFastFP32Gemm {};
struct KernelTmaWarpSpecializedBlockwiseFastFP32Gemm2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100BlockwiseFastFP32Gemm {};

struct KernelScheduleSm100PtrArrayBlockwiseFastFP32Gemm  : KernelScheduleSm100BlockwiseFastFP32Gemm {};
struct KernelPtrArrayTmaWarpSpecializedBlockwiseFastFP32Gemm1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayBlockwiseFastFP32Gemm {};
struct KernelPtrArrayTmaWarpSpecializedBlockwiseFastFP32Gemm2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayBlockwiseFastFP32Gemm {};
// {$nv-internal-release end}

// {$nv-internal-release begin}
///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 BgradA GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100BgradAGemm : KernelScheduleSm100{};  // Base policy
// BgradA Dense GEMM
struct KernelTmaWarpSpecialized1SmBgradASm100 final : KernelSchedule1Sm, KernelScheduleSm100BgradAGemm { }; // Use for 1SM BgradA dense GEMM Kernels for collective mainloop builder
struct KernelTmaWarpSpecialized2SmBgradASm100 final : KernelSchedule2Sm, KernelScheduleSm100BgradAGemm { }; // Use for 2SM BgradA dense GEMM Kernels for collective mainloop builder
// {$nv-internal-release end}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Planar Complex GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100PlanarComplexGemm : KernelScheduleSm100{};
// Planar Complex GEMM: Specialize for 1SM vs 2SM
struct KernelTmaWarpSpecialized1SmPlanarComplexSm100 final : KernelSchedule1Sm, KernelScheduleSm100PlanarComplexGemm { };
struct KernelTmaWarpSpecialized2SmPlanarComplexSm100 final : KernelSchedule2Sm, KernelScheduleSm100PlanarComplexGemm { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Ptr-Array Planar Complex GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Planar Complex GEMM + (Ptr Array or Group GEMM)
struct KernelScheduleSm100PtrArrayPlanarComplexGemm : KernelScheduleSm100PlanarComplexGemm {};

struct KernelPtrArrayTmaWarpSpecialized1SmPlanarComplexSm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayPlanarComplexGemm {};
struct KernelPtrArrayTmaWarpSpecialized2SmPlanarComplexSm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayPlanarComplexGemm {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 FastF32 (9xBF16) GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100FastFP32Gemm           : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedFastFP32SmemSm100 : KernelScheduleSm100FastFP32Gemm { };
// Dispatch policies without smem load the A operand from tmem
struct KernelTmaWarpSpecialized1SmFastFP32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100FastFP32Gemm { };
struct KernelTmaWarpSpecialized2SmFastFP32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100FastFP32Gemm { };
// Dispatch policies with smem load the A operand from smem
struct KernelTmaWarpSpecialized1SmFastFP32SmemSm100 final : KernelSchedule1Sm, KernelTmaWarpSpecializedFastFP32SmemSm100 { };
struct KernelTmaWarpSpecialized2SmFastFP32SmemSm100 final : KernelSchedule2Sm, KernelTmaWarpSpecializedFastFP32SmemSm100 { };

// {$nv-internal-release begin}
///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 FastF32 (4xFP16) GEMM with fused Blockwise scaling: SF layout fixed within the mainloop
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100FusedBlockwiseFastFP32Gemm : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedFusedBlockwiseFastFP32SmemSm100 : KernelScheduleSm100FusedBlockwiseFastFP32Gemm {};
// Dispatch policies without smem: load the A operand from tmem
struct KernelTmaWarpSpecialized1SmFusedBlockwiseFastFP32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100FusedBlockwiseFastFP32Gemm { };
struct KernelTmaWarpSpecialized2SmFusedBlockwiseFastFP32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100FusedBlockwiseFastFP32Gemm { };
// Dispatch policies with smem: load the A operand from smem
struct KernelTmaWarpSpecialized1SmFusedBlockwiseFastFP32SmemSm100 final : KernelSchedule1Sm, KernelTmaWarpSpecializedFusedBlockwiseFastFP32SmemSm100 { };
struct KernelTmaWarpSpecialized2SmFusedBlockwiseFastFP32SmemSm100 final : KernelSchedule2Sm, KernelTmaWarpSpecializedFusedBlockwiseFastFP32SmemSm100 { };

struct KernelScheduleSm100PtrArrayFusedBlockwiseFastFP32Gemm : KernelScheduleSm100FusedBlockwiseFastFP32Gemm {};
struct KernelPtrArrayTmaWarpSpecializedFusedBlockwiseFastFP32SmemSm100 : KernelScheduleSm100PtrArrayFusedBlockwiseFastFP32Gemm {};
// Dispatch policies without smem: load the A operand from tmem
struct KernelPtrArrayTmaWarpSpecialized1SmFusedBlockwiseFastFP32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayFusedBlockwiseFastFP32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized2SmFusedBlockwiseFastFP32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayFusedBlockwiseFastFP32Gemm { };
// Dispatch policies with smem: load the A operand from smem
struct KernelPtrArrayTmaWarpSpecialized1SmFusedBlockwiseFastFP32SmemSm100 final : KernelSchedule1Sm, KernelPtrArrayTmaWarpSpecializedFusedBlockwiseFastFP32SmemSm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmFusedBlockwiseFastFP32SmemSm100 final : KernelSchedule2Sm, KernelPtrArrayTmaWarpSpecializedFusedBlockwiseFastFP32SmemSm100 { };
// {$nv-internal-release end}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Mixed Precision Input GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100MixedInputGemm           : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedMixedInputSmemSm100 : KernelScheduleSm100MixedInputGemm { };
struct KernelTmaWarpSpecialized1SmMixedInputSm100 final : KernelSchedule1Sm, KernelScheduleSm100MixedInputGemm { };
struct KernelTmaWarpSpecialized1SmMixedInputSmemSm100 final : KernelSchedule1Sm, KernelTmaWarpSpecializedMixedInputSmemSm100 { };
struct KernelTmaWarpSpecialized2SmMixedInputSm100 final : KernelSchedule2Sm, KernelScheduleSm100MixedInputGemm { };
struct KernelTmaWarpSpecialized2SmMixedInputSmemSm100 final : KernelSchedule2Sm, KernelTmaWarpSpecializedMixedInputSmemSm100 { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Ptr-Array FastF32 (9xBF16) GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Ptr-Array Transform GEMM: Specialize for 1SM vs 2SM FastF32 GEMM
struct KernelScheduleSm100PtrArrayFastFP32Gemm           : KernelScheduleSm100FastFP32Gemm {};
struct KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100 : KernelScheduleSm100PtrArrayFastFP32Gemm { };

struct KernelPtrArrayTmaWarpSpecialized1SmFastFP32Sm100     final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayFastFP32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized2SmFastFP32Sm100     final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayFastFP32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized1SmFastFP32SmemSm100 final : KernelSchedule1Sm, KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmFastFP32SmemSm100 final : KernelSchedule2Sm, KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100 { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Interleaved Complex GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100InterleavedComplexTF32Gemm : KernelScheduleSm100 {};
// Transform GEMM: Specialize for Interleaved Complex GEMMs
struct KernelTmaWarpSpecialized1SmInterleavedComplexTF32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100InterleavedComplexTF32Gemm { };
struct KernelTmaWarpSpecialized2SmInterleavedComplexTF32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100InterleavedComplexTF32Gemm { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Ptr-Array Interleaved Complex GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Interleaved Complex GEMM + (Ptr array or Group GEMM)
struct KernelScheduleSm100PtrArrayInterleavedComplexTF32Gemm : KernelScheduleSm100InterleavedComplexTF32Gemm {};
// Ptr-Array Transform GEMM: Specialize for 1SM vs 2SM Complex GEMM
// Transform GEMM: Specialize for Interleaved Complex GEMMs
struct KernelPtrArrayTmaWarpSpecialized1SmInterleavedComplexTF32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayInterleavedComplexTF32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized2SmInterleavedComplexTF32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayInterleavedComplexTF32Gemm { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Sparse GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSparseGemmSm100 : KernelScheduleSm100 {};
// Sparse GEMM: Specialize for 1SM vs 2SM
struct KernelSparseTmaWarpSpecialized1SmSm100 final : KernelSchedule1Sm, KernelScheduleSparseGemmSm100 { };
struct KernelSparseTmaWarpSpecialized2SmSm100 final : KernelSchedule2Sm, KernelScheduleSparseGemmSm100 { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 BlockScaled Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleBlockScaledGemmSm100   : KernelScheduleSm100 {};                  
struct KernelScheduleMxNvf4Sm100            : KernelScheduleBlockScaledGemmSm100 {};
struct KernelScheduleMxf8f6f4Sm100          : KernelScheduleBlockScaledGemmSm100 {};
struct KernelScheduleMxQmmaSm100            : KernelScheduleBlockScaledGemmSm100 {}; // {$nv-release-never}
// Block Scaled Dense GEMM: Specialize for instruction type, scale factor vector size, and 1SM vs. 2SM
struct KernelTmaWarpSpecialized1SmBlockScaledSm100       final : KernelSchedule1Sm, KernelScheduleBlockScaledGemmSm100 { };
struct KernelTmaWarpSpecialized2SmBlockScaledSm100       final : KernelSchedule2Sm, KernelScheduleBlockScaledGemmSm100 { };
struct KernelTmaWarpSpecialized1SmNvf4Sm100              final : KernelSchedule1Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized2SmNvf4Sm100              final : KernelSchedule2Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized1SmMxf4Sm100              final : KernelSchedule1Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized2SmMxf4Sm100              final : KernelSchedule2Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized1SmMxf8f6f4Sm100          final : KernelSchedule1Sm, KernelScheduleMxf8f6f4Sm100 { };
struct KernelTmaWarpSpecialized2SmMxf8f6f4Sm100          final : KernelSchedule2Sm, KernelScheduleMxf8f6f4Sm100 { };
struct KernelTmaWarpSpecialized1SmMxQmmaSm100            final : KernelSchedule1Sm, KernelScheduleMxQmmaSm100 { }; // {$nv-release-never}
struct KernelTmaWarpSpecialized2SmMxQmmaSm100            final : KernelSchedule2Sm, KernelScheduleMxQmmaSm100 { }; // {$nv-release-never}
struct KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100 final : KernelSchedule1Sm, KernelScheduleBlockScaledGemmSm100 {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 BlockScaled Ptr Array Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
// BlockScaled Dense GEMM + (Ptr Array or Group GEMM)
struct KernelSchedulePtrArrayBlockScaledGemmSm100   : KernelScheduleBlockScaledGemmSm100 {};
struct KernelSchedulePtrArrayMxNvf4Sm100            : KernelSchedulePtrArrayBlockScaledGemmSm100 {};
struct KernelSchedulePtrArrayMxf8f6f4Sm100          : KernelSchedulePtrArrayBlockScaledGemmSm100 {};
struct KernelSchedulePtrArrayMxQmmaSm100            : KernelSchedulePtrArrayBlockScaledGemmSm100 {}; // {$nv-release-never}
// Ptr-Array Block Scaled Dense GEMM: Specialize for instruction type, scale factor vector size, and 1SM vs. 2SM
struct KernelPtrArrayTmaWarpSpecialized1SmBlockScaledSm100       final : KernelSchedule1Sm, KernelSchedulePtrArrayBlockScaledGemmSm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmBlockScaledSm100       final : KernelSchedule2Sm, KernelSchedulePtrArrayBlockScaledGemmSm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100              final : KernelSchedule1Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100              final : KernelSchedule2Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmMxf4Sm100              final : KernelSchedule1Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmMxf4Sm100              final : KernelSchedule2Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100          final : KernelSchedule1Sm, KernelSchedulePtrArrayMxf8f6f4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100          final : KernelSchedule2Sm, KernelSchedulePtrArrayMxf8f6f4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmMxQmmaSm100            final : KernelSchedule1Sm, KernelSchedulePtrArrayMxQmmaSm100 { }; // {$nv-release-never}
struct KernelPtrArrayTmaWarpSpecialized2SmMxQmmaSm100            final : KernelSchedule2Sm, KernelSchedulePtrArrayMxQmmaSm100 { }; // {$nv-release-never}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 BlockScaled Sparse GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleBlockScaledSparseGemmSm100 : KernelScheduleSm100 {};
struct KernelScheduleSparseMxNvf4Sm100          : KernelScheduleBlockScaledSparseGemmSm100 {};
struct KernelScheduleSparseMxf8f6f4Sm100        : KernelScheduleBlockScaledSparseGemmSm100 {};
// Block Scaled Sparse GEMM: Specialize for instruction type, scale factor vector size, and 1SM vs. 2SM
struct KernelSparseTmaWarpSpecialized1SmBlockScaledSm100 final : KernelSchedule1Sm, KernelScheduleBlockScaledSparseGemmSm100 {};
struct KernelSparseTmaWarpSpecialized2SmBlockScaledSm100 final : KernelSchedule2Sm, KernelScheduleBlockScaledSparseGemmSm100 {};
struct KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100    final : KernelSchedule1Sm, KernelScheduleSparseMxf8f6f4Sm100 { };
struct KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100    final : KernelSchedule2Sm, KernelScheduleSparseMxf8f6f4Sm100 { };
struct KernelSparseTmaWarpSpecialized1SmNvf4Sm100        final : KernelSchedule1Sm, KernelScheduleSparseMxNvf4Sm100 { };
struct KernelSparseTmaWarpSpecialized2SmNvf4Sm100        final : KernelSchedule2Sm, KernelScheduleSparseMxNvf4Sm100 { };
struct KernelSparseTmaWarpSpecialized1SmMxf4Sm100        final : KernelSchedule1Sm, KernelScheduleSparseMxNvf4Sm100 { };
struct KernelSparseTmaWarpSpecialized2SmMxf4Sm100        final : KernelSchedule2Sm, KernelScheduleSparseMxNvf4Sm100 { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
//          SM103 Dispatch Policies
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelScheduleSm103 {};
struct KernelScheduleSm103BlockScaledGemm                  : KernelScheduleSm103 {};
struct KernelScheduleSm103BlockScaledMxNvf4UltraTmaPrefetch     : KernelScheduleSm103BlockScaledGemm {};
struct KernelScheduleSm103BlockScaledMxNvf4UltraTmaLoadPrefetch : KernelScheduleSm103BlockScaledGemm {}; // {$nv-internal-release}
struct KernelScheduleSm103BlockScaledMxNvf4UltraDisablePrefetch : KernelScheduleSm103BlockScaledGemm {};

// Blockscaled Gemm: Specialized for instruction type, scale factor vector size, and 1SM vs. 2SM
// These are the public dispatch policy name
// TMAPF Prefetch // {$nv-release-never}
struct KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch final : KernelSchedule1Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaPrefetch { };
struct KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch final : KernelSchedule2Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaPrefetch { };
struct KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103TmaPrefetch final : KernelSchedule1Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaPrefetch { };
struct KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103TmaPrefetch final : KernelSchedule2Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaPrefetch { };

// {$nv-internal-release begin}
// TMALDG Prefetch // {$nv-release-never}
struct KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103TmaLoadPrefetch final : KernelSchedule1Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaLoadPrefetch { };
struct KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103TmaLoadPrefetch final : KernelSchedule2Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaLoadPrefetch { };
struct KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103TmaLoadPrefetch final : KernelSchedule1Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaLoadPrefetch { };
struct KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103TmaLoadPrefetch final : KernelSchedule2Sm, KernelScheduleSm103BlockScaledMxNvf4UltraTmaLoadPrefetch { };
// {$nv-internal-release end}
// Disable Prefetch // {$nv-release-never}
struct KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch final : KernelSchedule1Sm, KernelScheduleSm103BlockScaledMxNvf4UltraDisablePrefetch { };
struct KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch final : KernelSchedule2Sm, KernelScheduleSm103BlockScaledMxNvf4UltraDisablePrefetch { };
struct KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103DisablePrefetch final : KernelSchedule1Sm, KernelScheduleSm103BlockScaledMxNvf4UltraDisablePrefetch { };
struct KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103DisablePrefetch final : KernelSchedule2Sm, KernelScheduleSm103BlockScaledMxNvf4UltraDisablePrefetch { };

// Make TMAPF Prefetch as default// {$nv-release-never}
using KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103 = KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch;
using KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103 = KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch;
using KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103 = KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103TmaPrefetch;
using KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103 = KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103TmaPrefetch;


struct KernelSchedulePtrArraySm103BlockScaledGemm                  : KernelScheduleSm103 {};
struct KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraTmaPrefetch     : KernelSchedulePtrArraySm103BlockScaledGemm {};
struct KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraDisablePrefetch : KernelSchedulePtrArraySm103BlockScaledGemm {};

struct KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch final : KernelSchedule1Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraTmaPrefetch { };
struct KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch final : KernelSchedule2Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraTmaPrefetch { };
struct KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103TmaPrefetch final : KernelSchedule1Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraTmaPrefetch { };
struct KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103TmaPrefetch final : KernelSchedule2Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraTmaPrefetch { };

struct KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch final : KernelSchedule1Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraDisablePrefetch { };
struct KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch final : KernelSchedule2Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraDisablePrefetch { };
struct KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103DisablePrefetch final : KernelSchedule1Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraDisablePrefetch { };
struct KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103DisablePrefetch final : KernelSchedule2Sm, KernelSchedulePtrArraySm103BlockScaledMxNvf4UltraDisablePrefetch { };

using KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103 = KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch;
using KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103 = KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch;
using KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103 = KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103DisablePrefetch;
using KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103 = KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103DisablePrefetch;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
//          SM120 Dispatch Policies
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Builder Tag Base Dispatch Policies
struct KernelScheduleSm120 {};
struct KernelScheduleAcc2x4Sm120 {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm120DenseGemm : KernelScheduleSm120 {};
// Dense GEMM: Specialize for instruction type
struct KernelScheduleF8f6f4Sm120 final : KernelScheduleSm120DenseGemm {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 BlockScaled GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleBlockScaledGemmSm120 : KernelScheduleSm120 {};
struct KernelScheduleMxf8f6f4Sm120        : KernelScheduleBlockScaledGemmSm120 {};
struct KernelScheduleMxNvf4Sm120          : KernelScheduleBlockScaledGemmSm120 {};
struct KernelScheduleMxQmmaSm120          : KernelScheduleBlockScaledGemmSm120 {}; // {$nv-release-never}
// Block Scaled GEMM: Specialize for instruction type, scale factor vector size.
struct KernelTmaWarpSpecializedNvf4Sm120             final : KernelScheduleMxNvf4Sm120, KernelTmaWarpSpecializedCooperative { };
struct KernelTmaWarpSpecializedPingpongNvf4Sm120     final : KernelScheduleMxNvf4Sm120, KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedMxf4Sm120             final : KernelScheduleMxNvf4Sm120, KernelTmaWarpSpecializedCooperative { };
struct KernelTmaWarpSpecializedPingpongMxf4Sm120     final : KernelScheduleMxNvf4Sm120, KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedMxf8f6f4Sm120         final : KernelScheduleMxf8f6f4Sm120, KernelTmaWarpSpecializedCooperative { };
struct KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120 final : KernelScheduleMxf8f6f4Sm120, KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedMxQmmaSm120           final : KernelScheduleMxQmmaSm120, KernelTmaWarpSpecializedCooperative { }; // {$nv-release-never}
struct KernelTmaWarpSpecializedPingpongMxQmmaSm120   final : KernelScheduleMxQmmaSm120, KernelTmaWarpSpecializedPingpong { };    // {$nv-release-never}

// Blockwise Scaled GEMM
struct KernelScheduleSm120Blockwise: KernelScheduleSm120 { };
struct KernelTmaWarpSpecializedBlockwiseCooperativeSm120 final : KernelScheduleSm120Blockwise, KernelTmaWarpSpecializedCooperative { };
struct KernelTmaWarpSpecializedBlockwisePingpongSm120 final : KernelScheduleSm120Blockwise, KernelTmaWarpSpecializedPingpong { };


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 Sparse GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSparseGemmSm120 : KernelScheduleSm120 {};
// Sparse GEMM: Specialize for instruction type
struct KernelScheduleSparseF8f6f4Sm120 final : KernelScheduleSparseGemmSm120 {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 BlockScaled Sparse GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleBlockScaledSparseGemmSm120 : KernelScheduleSm120 {};
struct KernelScheduleSparseMxNvf4Sm120          : KernelScheduleBlockScaledSparseGemmSm120 {};
struct KernelScheduleSparseMxf8f6f4Sm120        : KernelScheduleBlockScaledSparseGemmSm120 {};
// Block Scaled Sparse GEMM: Specialize for instruction type, scale factor vector size, Acc2x4
struct KernelSparseTmaWarpSpecializedNvf4Sm120           final : KernelScheduleSparseMxNvf4Sm120 { };
struct KernelSparseTmaWarpSpecializedMxf4Sm120           final : KernelScheduleSparseMxNvf4Sm120 { };
struct KernelSparseTmaWarpSpecializedMxf8f6f4Sm120       final : KernelScheduleSparseMxf8f6f4Sm120 { };
struct KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120 final : KernelScheduleSparseMxf8f6f4Sm120, KernelScheduleAcc2x4Sm120 { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Dispatch Policies
//

// n-buffer in smem, pipelined with Blackwell UMMA and CPASYNC, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100UmmaCpAsyncWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100UmmaMixedTmaCpAsyncWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelMixedTmaCpAsyncWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100UmmaMixedTmaCpAsyncWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelMixedTmaCpAsyncWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedMmaTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelPtrArrayTmaWarpSpecializedMmaTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

// {$nv-internal-release begin}
// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  // FP32 -> 2x FP16
  int NumComputeMtxs_ = 2,
  // 4xFP16 in 3 bands
  int NumBandsToCompute_ = 3,
  // 11 effective bits from FP16 -- Each compute mtx have 2^11 relative shift
  int ScalingFactor_ = 11
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockwiseFastF32 {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedMmaTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
  constexpr static int NumComputeMtxs = NumComputeMtxs_;
  constexpr static int NumBandsToCompute = NumBandsToCompute_;
  constexpr static int ScalingFactor = ScalingFactor_;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  // FP32 -> 2x FP16
  int NumComputeMtxs_ = 2,
  // 4xFP16 in 3 bands
  int NumBandsToCompute_ = 3,
  // 11 effective bits from FP16 -- Each compute mtx have 2^11 relative shift
  int ScalingFactor_ = 11
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseFastF32 {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelPtrArrayTmaWarpSpecializedMmaTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
  constexpr static int NumComputeMtxs = NumComputeMtxs_;
  constexpr static int NumBandsToCompute = NumBandsToCompute_;
  constexpr static int ScalingFactor = ScalingFactor_;
};
// {$nv-internal-release end}

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedSparse {
  constexpr static int Stages = Stages_;
  constexpr static int MetadataS2TStages = 4;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelSparseTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockScaledSparse {
  constexpr static int Stages = Stages_;
  constexpr static int MetadataS2TStages = 4;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelSparseTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

// n-buffer in smem, pipelined with Blackwell Fast FP32 kernel with UMMA (HwScaled) and TMA,
// Warp specialized dynamic schedule
template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Load2TransformPipelineStageCount_,
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Transform2MmaPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // Number of MMA Bands to be computed in a single FastF32 MMA operation.
  // For BF16 emulation, we have 3 compute matrices, with 9 MMAs forming 5 bands.
  //    We can eliminate bands 4 and/or 5 (up to last 3 MMA operations).
  //    Valid values are 3, 4, 5
  int NumBandsToCompute_,
  // Scaling factor for decomposed matrices (2^ScalingFactor)
  // 8 for BF16, 11 for TF32
  int ScalingFactor_,
  // Number of UMMA instructions emulated a single stage
  // Ex: Staged16 has 1 FastF32 MMA per stage
  // Should be smaller than K-mode of a single ClusterTile
  int AccPromotionInterval_,
  // {$nv-internal-release begin}
  // Selects between 9xBF16 (dynamic-range-aligned) and 4xFP16 (blockwise scaling)
  typename ElementMmaRaw_ = bfloat16_t,
  // {$nv-internal-release end}
  // ClusterShape for the kernel
  class ClusterShape_ = Shape<_1,_1,_1>,
  // The TMEM_LOAD atom to be used for loading local accumulator
  // from TMEM to registers
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_32dp32b32x
>
struct MainloopSm100TmaUmmaWarpSpecializedFastF32 {
  constexpr static int Load2TransformPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Transform2MmaPipelineStageCount = Transform2MmaPipelineStageCount_;
  constexpr static int NumBandsToCompute = NumBandsToCompute_;
  constexpr static int ScalingFactor = ScalingFactor_;
  constexpr static int AccPromotionInterval = AccPromotionInterval_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::FastF32;
  using ElementMmaRaw = ElementMmaRaw_; // {$nv-internal-release}
  using ClusterShape = ClusterShape_;
  using AccumulatorCopyAtom = AccumulatorCopyAtom_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = Load2TransformPipelineStageCount;
};


template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Transformation
  int ComputationPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // Number of Pipeline stages for
  // Transformation <-> MMA
  int TransformationPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_16dp256b1x
>
struct MainloopSm100TmaUmmaWarpSpecializedInterleavedComplexTF32 {
  constexpr static int ComputationPipelineStageCount = ComputationPipelineStageCount_;
  constexpr static int TransformationPipelineStageCount = TransformationPipelineStageCount_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::InterleavedComplexTF32;
  using ClusterShape = ClusterShape_;
  using AccumulatorCopyAtom = AccumulatorCopyAtom_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = ComputationPipelineStageCount;
};

// n-buffer in smem, pipelined with Blackwell Mixed Input kernel with UMMA (HwScaled) and TMA,
template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Load2TransformPipelineStageCount_,
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Transform2MmaPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // ClusterShape for the kernel
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedMixedInput {
  constexpr static int Load2TransformPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Load2MmaPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Transform2MmaPipelineStageCount = Transform2MmaPipelineStageCount_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::MixedInput;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedMixedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = Load2TransformPipelineStageCount;
};


// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedPlanarComplex {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};
// {$nv-internal-release begin}
// n-buffer in smem for general dense GEMM, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
// 1-buffer in smem for the extra output vector from the reduction based on input A tensor
// The general GEMM may be either the 1cta or 2cta version,
// the extra second MMA for reduction always same as the general GEMM used
template<
  int Stages_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaWarpSpecializedBgradA {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};
// {$nv-internal-release end}


// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100ArrayTmaUmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = false;
  using Schedule = KernelPtrArrayTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100RCGroupGemmTmaUmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = false;
  using Schedule = KernelPtrArrayTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100RCGroupGemmTmaUmmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};



// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedPlanarComplex {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = false;
  using Schedule = KernelPtrArrayTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};


// n-buffer in smem, pipelined with Blackwell Fast FP32 kernel with UMMA (HwScaled) and TMA,
// Warp specialized dynamic schedule
template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Load2TransformPipelineStageCount_,
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Transform2MmaPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // Number of MMA Bands to be computed in a single FastF32 MMA operation.
  // For BF16 emulation, we have 3 compute matrices, with 9 MMAs forming 5 bands.
  //    We can eliminate bands 4 and/or 5 (up to last 3 MMA operations).
  //    Valid values are 3, 4, 5
  int NumBandsToCompute_,
  // Scaling factor for decomposed matrices (2^ScalingFactor)
  // 8 for BF16, 11 for TF32
  int ScalingFactor_,
  // Number of UMMA instructions emulated a single stage
  // Ex: Staged16 has 1 FastF32 MMA per stage
  // Should be smaller than K-mode of a single ClusterTile
  int AccPromotionInterval_,
  // {$nv-internal-release begin}
  // Selects between 9xBF16 (dynamic-range-aligned) and 4xFP16 (blockwise scaling)
  typename ElementMmaRaw_ = bfloat16_t,
  // {$nv-internal-release end}
  // ClusterShape for the kernel
  class ClusterShape_ = Shape<_1,_1,_1>,
  // The TMEM_LOAD atom to be used for loading local accumulator
  // from TMEM to registers
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_32dp32b32x
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedFastF32 {
  constexpr static int Load2TransformPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Transform2MmaPipelineStageCount = Transform2MmaPipelineStageCount_;
  constexpr static int NumBandsToCompute = NumBandsToCompute_;
  constexpr static int ScalingFactor = ScalingFactor_;
  constexpr static int AccPromotionInterval = AccPromotionInterval_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::FastF32;
  using ElementMmaRaw = ElementMmaRaw_; // {$nv-internal-release}
  using ClusterShape = ClusterShape_;
  using AccumulatorCopyAtom = AccumulatorCopyAtom_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelPtrArrayTmaWarpSpecializedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = Load2TransformPipelineStageCount;
};


template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Transformation
  int ComputationPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // Number of Pipeline stages for
  // Transformation <-> MMA
  int TransformationPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_16dp256b1x
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedInterleavedComplexTF32 {
  constexpr static int ComputationPipelineStageCount = ComputationPipelineStageCount_;
  constexpr static int TransformationPipelineStageCount = TransformationPipelineStageCount_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::InterleavedComplexTF32;
  using ClusterShape = ClusterShape_;
  using AccumulatorCopyAtom = AccumulatorCopyAtom_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelPtrArrayTmaWarpSpecializedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = ComputationPipelineStageCount;
};


// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int LoadABPipelineStageCount_,
  int LoadSFPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  cutlass::sm103::detail::KernelPrefetchType PrefetchType_ = cutlass::sm103::detail::KernelPrefetchType::TmaPrefetch
>
struct MainloopSm103TmaUmmaWarpSpecializedBlockScaled {
  constexpr static int LoadABPipelineStageCount = LoadABPipelineStageCount_;
  constexpr static int LoadSFPipelineStageCount = LoadSFPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm103;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelTmaWarpSpecializedBlockScaledSm103<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = LoadABPipelineStageCount;
  constexpr static cutlass::sm103::detail::KernelPrefetchType PrefetchType = PrefetchType_;
};

// Mainloop schedule for array-based TMA

template<
  int LoadABPipelineStageCount_,
  int LoadSFPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  cutlass::sm103::detail::KernelPrefetchType PrefetchType_ = cutlass::sm103::detail::KernelPrefetchType::TmaPrefetch
>
struct MainloopSm103ArrayTmaUmmaWarpSpecializedBlockScaled {
  constexpr static int LoadABPipelineStageCount = LoadABPipelineStageCount_;
  constexpr static int LoadSFPipelineStageCount = LoadSFPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm103;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelPtrArrayTmaWarpSpecializedBlockScaledSm103<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = LoadABPipelineStageCount;
  constexpr static cutlass::sm103::detail::KernelPrefetchType PrefetchType = PrefetchType_;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120TmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120ArrayTmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
  static_assert(
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, Schedule> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, Schedule>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies");                                     
};


template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120TmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120ArrayTmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using Schedule = KernelSchedule_;
  using ArchTag = arch::Sm120;

  static_assert(cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, Schedule> ||
                cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, Schedule>, 
                "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies.");
};


template<
  int StagesA_,
  int StagesB_,
  int StagesE_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm120TmaWarpSpecializedSparse {
  constexpr static int StagesA = StagesA_;
  constexpr static int StagesB = StagesB_;
  constexpr static int StagesE = StagesE_;
  constexpr static bool isAsymmetric = (StagesA != StagesB);
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm120;
  using Schedule = KernelTmaWarpSpecializedCooperativeSparseSm120<SchedulerPipelineStageCount_, isAsymmetric>;
};

template<
  int StagesA_,
  int StagesB_,
  int StagesE_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm120TmaWarpSpecializedSparseBlockScaled {
  constexpr static int StagesA = StagesA_;
  constexpr static int StagesB = StagesB_;
  constexpr static int StagesE = StagesE_;
  constexpr static bool isAsymmetric = (StagesA != StagesB);
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm120;
  using Schedule = KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<SchedulerPipelineStageCount_, isAsymmetric>;
};

template <
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120TmaWarpSpecializedBlockwiseScaling {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;

  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

template <
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_
>
struct MainloopSm120ArrayTmaWarpSpecializedBlockwiseScaling {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;

  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;

  static_assert(cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, Schedule> ||
                cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, Schedule>, 
                "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies.");
};



//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm
