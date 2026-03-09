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

#include "cutlass/conv/convolution.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/arch/arch.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"

//////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}
namespace cutlass::detail {

template <class T, template <class, class, template<class> class, int, int> class U>
struct is_batch_norm_builder_tag_of : cute::false_type {};

template <template <class, class, template<class> class, int, int> class U, class T1, class T2, template<class> class C1, int I1, int I2>
struct is_batch_norm_builder_tag_of<U<T1, T2, C1, I1, I2>, U> : cute::true_type {};

template <class T, template <class, class, template<class> class, int, int> class U>
constexpr bool is_batch_norm_builder_tag_of_v = is_batch_norm_builder_tag_of<T, U>::value;

template <class T, template <int, int, class, class, template<class> class, int, int> class U>
struct is_batch_norm_kernel_tag_of : cute::false_type {};

template <template <int, int, class, class, template<class> class, int, int> class U, int I1, int I2, class T1, class T2, template<class> class C1, int I3, int I4>
struct is_batch_norm_kernel_tag_of<U<I1, I2, T1, T2, C1, I3, I4>, U> : cute::true_type {};

template <class T, template <int, int, class, class, template<class> class, int, int> class U>
constexpr bool is_batch_norm_kernel_tag_of_v = is_batch_norm_kernel_tag_of<T, U>::value;
}

// {$nv-internal-release end}

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv {

//////////////////////////////////////////////////////////////////////////////

//
// Policies for categorical dispatch of mainloop against kernel grid schedules
//
struct KernelImplicitTmaWarpSpecializedSm90 : cutlass::gemm::KernelTmaWarpSpecialized { };
struct KernelImplicitTmaWarpSpecializedSm90Cooperative { };
struct KernelImplicitTmaWarpSpecializedSm90Pingpong { };

//
// Collective Mainloop Policies
//

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, static schedule between TMA and GMMA
// for fprop
template<
  conv::Operator ConvOp_,
  int Stages_,
  int NumSpatialDimensions_,
  class ClusterShape_ = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>,
  class KernelSchedule = KernelImplicitTmaWarpSpecializedSm90,
  int PipelineAsyncMmaStages_ = 1
>
struct MainloopSm90TmaGmmaWarpSpecializedImplicitGemm {
  static constexpr int Stages = Stages_;
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  static constexpr Operator ConvOp = ConvOp_;
  static constexpr int PipelineAsyncMmaStages = PipelineAsyncMmaStages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;

  static_assert(NumSpatialDimensions >= 1);
  static_assert(! (cute::is_same_v<KernelSchedule,KernelImplicitTmaWarpSpecializedSm90Cooperative> ||
                   cute::is_same_v<KernelSchedule,KernelImplicitTmaWarpSpecializedSm90Pingpong>),
    "Persistent schedules not support for conv yet.");
};



// SM100 tensor op kernel schedule
struct KernelImplicitTmaWarpSpecializedSm100 {
  static constexpr int SchedulerPipelineStageCount = 0;
  static constexpr int AccumulatorPipelineStageCount = 0;
};

// Pseudo-policies for builder auto override that dispatches to the KernelImplicitTmaWarpSpecializedSm100
// but for opting into 1 or 2 SM atoms
struct KernelImplicitTmaWarpSpecialized1SmSm100 : KernelImplicitTmaWarpSpecializedSm100 { };
struct KernelImplicitTmaWarpSpecialized2SmSm100 : KernelImplicitTmaWarpSpecializedSm100 { };

struct KernelStridedDgradTmaWs1SmSm100 { };
struct KernelStridedDgradTmaWs2SmSm100 { };

// Policy for implicit gemm kernel
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelScheduleImplicitTmaWarpSpecializedSm100 : KernelImplicitTmaWarpSpecializedSm100 {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// n-buffer in smem (Blackwell TMA), pipelined with Blackwell UMMA and TMA, fprop
template<
  conv::Operator ConvOp_,
  int Stages_,
  int NumSpatialDimensions_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>
>
struct MainloopSm100TmaUmmaWarpSpecializedImplicitGemm {
  static constexpr int Stages = Stages_;
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  static constexpr Operator ConvOp = ConvOp_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelScheduleImplicitTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  static_assert(NumSpatialDimensions >= 1);
}; 

// {$nv-internal-release begin}

// Policy for nq 2d-tiled kernel
// Stride 1x1 is different from Stride 2x2
template<class FilterShapeTRS_ = cute::Shape<cute::C<1>,cute::C<3>,cute::C<3>>>
struct KernelNq2dTiledTmaWarpSpecializedStride1x1x1Sm100 {
  using TraveralStrideDHW = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>;
  using FilterShapeTRS = FilterShapeTRS_;
};

template<
  conv::Operator ConvOp_,
  int AStages_,
  int BStages_,
  int NumSpatialDimensions_,
  class ClusterShape_ = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>,
  class KernelSchedule = KernelNq2dTiledTmaWarpSpecializedStride1x1x1Sm100<cute::Shape<cute::C<1>,cute::C<3>,cute::C<3>>>
>
struct MainloopSm100TmaWeightStationaryUmmaWarpSpecializedNq2dTiled {
  static constexpr int AStages = AStages_;
  static constexpr int BStages = BStages_;
  static constexpr int Stages = AStages + BStages; // Needed for Conv Universal Adapter sanity 
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  static constexpr Operator ConvOp = ConvOp_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelSchedule;

  static_assert(NumSpatialDimensions >= 2);
};

// Policy for sparse nq 2d-tiled kernel
template<class FilterShapeTRS_ = cute::Shape<cute::C<1>,cute::C<3>,cute::C<3>>>
struct KernelSparseNq2dTiledTmaWarpSpecializedStride1x1x1Sm100 {
  using TraveralStrideDHW = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>;
  using FilterShapeTRS = FilterShapeTRS_;
};

template<
  conv::Operator ConvOp_,
  int AStages_,
  int BStages_,
  int NumSpatialDimensions_,
  class ClusterShape_ = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>,
  class KernelSchedule = KernelSparseNq2dTiledTmaWarpSpecializedStride1x1x1Sm100<cute::Shape<cute::C<1>,cute::C<3>,cute::C<3>>>
>
struct MainloopSm100TmaSparseWeightStationaryUmmaWarpSpecializedNq2dTiled {
  static constexpr int AStages = AStages_;
  static constexpr int BStages = BStages_;
  static constexpr int Stages = AStages + BStages; // Needed for Conv Universal Adapter sanity 
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  static constexpr Operator ConvOp = ConvOp_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelSchedule;

  static_assert(NumSpatialDimensions >= 2);
};

//
// Builder dispatch policies (not a part of the main CUTLASS layers, simply used to opt into
// specific collective builder dispatches)
//

template<
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::ReLu,
  int AlignmentAAlpha_ = ceil_div(16, sizeof(ElementAAlpha_)),
  int AlignmentABias_ = ceil_div(16, sizeof(ElementABias_))
>
struct KernelBatchNormTmaWarpSpecializedSm100 {
  using ElementAAlpha = ElementAAlpha_;
  using ElementABias = ElementABias_;

  template<class T>
  using ActivationFunctor = ActivationFunctor_<T>;

  static constexpr bool IsBatchNormApplySupported = true;
  static constexpr int AlignmentAAlpha = AlignmentAAlpha_;
  static constexpr int AlignmentABias = AlignmentABias_;
};

template<
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::ReLu,
  int AlignmentAAlpha_ = ceil_div(16, sizeof(ElementAAlpha_)),
  int AlignmentABias_ = ceil_div(16, sizeof(ElementABias_))
>
struct KernelBatchNormTmaWarpSpecialized1SmSm100 final : KernelBatchNormTmaWarpSpecializedSm100<ElementAAlpha_, ElementABias_, ActivationFunctor_, AlignmentAAlpha_, AlignmentABias_> {
  using ElementAAlpha = ElementAAlpha_;
  using ElementABias = ElementABias_;

  template<class T>
  using ActivationFunctor = ActivationFunctor_<T>;
};

template<
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::ReLu,
  int AlignmentAAlpha_ = ceil_div(16, sizeof(ElementAAlpha_)),
  int AlignmentABias_ = ceil_div(16, sizeof(ElementABias_))
>
struct KernelBatchNormTmaWarpSpecialized2SmSm100 final : KernelBatchNormTmaWarpSpecializedSm100<ElementAAlpha_, ElementABias_, ActivationFunctor_, AlignmentAAlpha_, AlignmentABias_> {
  using ElementAAlpha = ElementAAlpha_;
  using ElementABias = ElementABias_;

  template<class T>
  using ActivationFunctor = ActivationFunctor_<T>;
};

template<
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::ReLu,
  int AlignmentAAlpha_ = ceil_div(16, sizeof(ElementAAlpha_)),
  int AlignmentABias_ = ceil_div(16, sizeof(ElementABias_))
>
struct KernelBatchNormScheduleAuto final : KernelBatchNormTmaWarpSpecializedSm100<ElementAAlpha_, ElementABias_, ActivationFunctor_, AlignmentAAlpha_, AlignmentABias_> {
  using ElementAAlpha = ElementAAlpha_;
  using ElementABias = ElementABias_;

  template<class T>
  using ActivationFunctor = ActivationFunctor_<T>;
};

//
// Kernel Scheduler Tag
//

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::ReLu,
  int AlignmentAAlpha_ = ceil_div(16, sizeof(ElementAAlpha_)),
  int AlignmentABias_ = ceil_div(16, sizeof(ElementABias_))
>
struct KernelScheduleBatchNormTmaWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;

  using ElementAAlpha = ElementAAlpha_;
  using ElementABias = ElementABias_;

  template<class T>
  using ActivationFunctor = ActivationFunctor_<T>;

  static constexpr bool IsBatchNormApplySupported = true;
  static constexpr int AlignmentAAlpha = AlignmentAAlpha_;
  static constexpr int AlignmentABias = AlignmentABias_;
};

//
// Collective Mainloop Dispatch Policies
//

// Policy for batch_norm kernel
template<
  conv::Operator ConvOp_,
  int Stages_,
  // Number of Pipeline stages for
  // BatchNormApply <-> MMA
  int BatchNormApplyStages_,
  int SchedulerPipelineStages_,
  int AccumulatorStages_,
  int NumSpatialDimensions_,
  class CopyAtomR2T_,
  class CoptAtomS2R_,
  class ElementAAlpha_,
  class ElementABias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::ReLu,
  class ClusterShape_ = cute::Shape<cute::C<1>,cute::C<1>,cute::C<1>>,
  int AlignmentAAlpha_ = ceil_div(16, sizeof(ElementAAlpha_)),
  int AlignmentABias_ = ceil_div(16, sizeof(ElementABias_))
>
struct MainloopSm100TmaBatchNormUmmaWarpSpecializedImplicitGemm {
  static constexpr int Stages = Stages_;
  static constexpr int BatchNormApplyStages = BatchNormApplyStages_;
  static constexpr int NumSpatialDimensions = NumSpatialDimensions_;
  static constexpr Operator ConvOp = ConvOp_;
  using CopyAtomR2T = CopyAtomR2T_;
  using CoptAtomS2R = CoptAtomS2R_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelScheduleBatchNormTmaWarpSpecializedSm100<SchedulerPipelineStages_, AccumulatorStages_,
      ElementAAlpha_, ElementABias_, ActivationFunctor_, AlignmentAAlpha_, AlignmentABias_>;
}; // {$nv-internal-release end}

//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv 

//////////////////////////////////////////////////////////////////////////////
