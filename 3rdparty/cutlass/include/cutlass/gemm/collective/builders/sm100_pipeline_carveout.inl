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


#pragma once

namespace cutlass::gemm::collective::detail {

template<
  class ClusterShape_MNK,
  int AccumulatorPipelineStageCount,
  int SchedulerPipelineStageCount,
  int CLCResponseSize,
  bool IsArrayOfPointersGemm,
  int NumTensorMaps=2
>
struct Sm100DenseGemmTmaUmmaCarveout {
  // {$nv-release-never begin}
  // GemmUniversal::SharedStorage (Kernel Smem Usage)
  //   sizeof(GemmUniversal::SharedStorage) <= detail::sm100_smem_capacity_bytes
  // 1. CollectiveMainloop::SharedStorage (include stage dependent and stage independent part)
  //    a. CollectiveMainloop::TensorStorage
  //    b. CollectiveMainloop::PipelineStorage
  // 2. CollectiveEpilogue::SharedStorage
  //    a. CollectiveEpilogue::TensorStorage
  //    b. CollectiveEpilogue::PipelineStorage
  //    sizeof(CollectiveEpilogue::SharedStorage) = carveout_bytes in StageCountType<carveout_bytes>
  // 3. Kernel layer only storage (everything in GemmUniversal::SharedStorage without CollectiveMainloop:: / CollectiveEpilogue::)
  //    a. Barriers
  //    b. Extra tmem related storage
  //
  // KernelSmemCarveout includes
  // * All parts of (3)
  // * Stage independent part of (1) (e.g. fix size smem for Bgrad)
  //
  // detail::sm100_compute_stage_count_or_override computes
  // * Stage dependent part of (1) (e.g. smem for a/b buffer)
  // {$nv-release-never end}

  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLCPipeline = PipelineCLCFetchAsync
  // This is a WAR : Grouped Gemm uses CLCPipeline = PipelineAsync, while they have same smem storage size as PipelineCLCFetchAsync. // {$nv-release-never}
  // For pointer-array and grouped GEMM, we have two CLC responses, one for TMA updater, one for the TMA/MMA/Epilogue warps.
  static constexpr int NumCLCResponses = (IsArrayOfPointersGemm ? 2 : 1);
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage) * NumCLCResponses;
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLC (scheduler) response
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize * NumCLCResponses;
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = SchedulerPipelineStageCount * sizeof(uint32_t);
  // Tensormap Storage
  static constexpr auto TensorMapStorage = 
    IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * NumTensorMaps * 5 /* We have five tensormaps smem */ :
    0;

  // TensorMapReady pipeline storage (specific to grouped/array kernels)
  static constexpr auto TensorMapReadyPipelineStorage = 
    IsArrayOfPointersGemm ? sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage) :
    0;

  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               LoadOrderBarrierStorage +
                                                               TmemDeallocStorage +
                                                               CLCThrottlePipelineStorage +
                                                               CLCResponseStorage +
                                                               TmemBasePtrsStorage +
                                                               TensorMapStorage +
                                                               TensorMapReadyPipelineStorage
                                                              );
};

template<class ClusterShape_MNK, int AccumulatorPipelineStageCount, int SchedulerPipelineStageCount, int CLCResponseSize>
struct Sm100SparseGemmTmaUmmaCarveout {

  // {$nv-release-never begin}
  // * Note on GemmUniversal::SharedStorage (Kernel Smem Usage)
  //   sizeof(GemmUniversal::SharedStorage) <= detail::sm100_smem_capacity_bytes
  // 1. CollectiveMainloop::SharedStorage (include stage dependent and stage independent part)
  //    a. CollectiveMainloop::TensorStorage
  //    b. CollectiveMainloop::PipelineStorage
  // 2. CollectiveEpilogue::SharedStorage
  //    a. CollectiveEpilogue::TensorStorage
  //    b. CollectiveEpilogue::PipelineStorage
  // 3. Kernel layer only storage (everything in GemmUniversal::SharedStorage without CollectiveMainloop:: / CollectiveEpilogue::)
  //    a. Barriers
  //    b. Extra tmem related storage
  // {$nv-release-never end}

  // * GemmUniversal::SharedStorage::PipelineStorage
  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);
  // CLCPipelineStorage = PipelineCLCFetchAsync
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);
  // AccumulatorPipeline = PipelineUmmaAsync
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);
  // CLC Throttle pipeline storage
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  // Tmem dealloc
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);

  static constexpr auto PipelineStorage = static_cast<int>(cutlass::round_up(
                                                      cutlass::round_up(LoadOrderBarrierStorage, 16) +
                                                      cutlass::round_up(CLCPipelineStorage, 16) +
                                                      cutlass::round_up(AccumulatorPipelineStorage, 16) +
                                                      cutlass::round_up(CLCThrottlePipelineStorage, 16) +
                                                      cutlass::round_up(TmemDeallocStorage, 16),
                                                    16));

  // * GemmUniversal::SharedStorage::Others
  // CLC (scheduler) response
  static constexpr auto CLCQueryResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;
  // Tmem ptr storage
  static constexpr auto TmemBasePtrsStorage = sizeof(uint32_t);

  static constexpr auto OtherStorage = static_cast<int>(cutlass::round_up(
                                                   cutlass::round_up(CLCQueryResponseStorage, 16) +
                                                   cutlass::round_up(TmemBasePtrsStorage, 16),
                                                 16));
  
  // Smem usage that's not part of CollectiveEpilogue::SharedStorage & CollectiveMainloop::SharedStorage
  static constexpr auto KernelSmemCarveout = static_cast<int>( PipelineStorage +
                                                               OtherStorage);
};
} // namespace cutlass::gemm::collective::detail