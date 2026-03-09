/***************************************************************************************************
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
 **************************************************************************************************/
/*! \file
    \brief Tensor contraction params with minimal template arguments.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/contraction/threadblock/fused_tensor_ndim_iterator_params.h"
#include "cutlass/contraction/threadblock/free_access_layout.h"
#include "cutlass/contraction/threadblock/fused_tensor_ndim_iterator_params.h"
#include "cutlass/contraction/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_affine_layout_params.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

namespace cutlass {
namespace contraction {
namespace kernel {

template < 
  int kBlockedModesM,
  int kBlockedModesN,
  int kMaxRank = 28 /* CUTENSOR_NAMESPACE::kMaxNumModes */
>
struct GettParams {
  using FreeLayoutM = typename cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesM>;
  using FreeLayoutN = typename cutlass::contraction::threadblock::FreeAccessLayout<kBlockedModesN>;

  using Index = int32_t; 
  using LongIndex = int64_t;

  static const int kEpilogueAffineRank = FreeLayoutM::kRank + FreeLayoutN::kRank;
  static const uint32_t kBitIndexStKFullWaves = 10U;

  using TensorCoord = Coord<kMaxRank, Index>;
  using TensorStrideCoord = Coord<kMaxRank, int64_t, int64_t>;

  // Choose max capacity for host side for detemplatization work purpose,
  // device logic will not be affected. 
  using MmaIteratorAParams = typename cutlass::contraction::threadblock::FusedTensorNdimIteratorParams<kMaxRank, FreeLayoutM::kRank>;  
  using MmaIteratorBParams = typename cutlass::contraction::threadblock::FusedTensorNdimIteratorParams<kMaxRank, FreeLayoutN::kRank>; 
  using MmaIteratorParamsContracted = typename cutlass::contraction::threadblock::FusedTensorNdimIteratorParamsContracted<kMaxRank>;
  
  using EpilogueFastIteratorParams = cutlass::epilogue::threadblock::PredicatedTileIteratorParams;
  using ThreadblockSwizzleBase = cutlass::contraction::threadblock::Gemm1DRowThreadblockSwizzleBase;
  using SwizzleTileShapeInfo = typename ThreadblockSwizzleBase::TiledShapeInfoSemiFlat<false>;
  using EpiligueAffineParams = typename cutlass::epilogue::threadblock::PredicatedTileIteratorAffineLayoutRankNParams<kEpilogueAffineRank>;
  using OutputOpParams = cutlass::epilogue::thread::LinearCombinationParams;

  cutlass::gemm::GemmCoord problem_size;

  MmaIteratorAParams params_A;
  MmaIteratorBParams params_B;  
  EpilogueFastIteratorParams fast_params_C;
  EpilogueFastIteratorParams fast_params_D;

  void *dataA;
  void *dataB;
  void *dataC;
  void *dataD;

  MmaIteratorParamsContracted params_contracted;
  EpiligueAffineParams  affine_params;
  OutputOpParams output_op;

  const TensorStrideCoord strideA;
  const TensorStrideCoord strideB;
  const TensorStrideCoord strideC;

  const int numModesM; ///< number of free modes of A
  const int numModesN; ///< number of free modes of B
  const int numModesK; ///< number of contracted modes
  const int numModesL; ///< number of batched modes
  const int32_t startM, startN, startL;
  const int64_t strideParallelK; ///< strideParallelK > 0 denotes that parallel-k ought to be used

  FreeLayoutM freeLayoutM;
  FreeLayoutN freeLayoutN;

  const SwizzleTileShapeInfo grid_tiled_shape;
  const bool useAffineEpilogue;
  const int gemm_k_iterations;
  const int output_tile_num; ///< tiling in C&D, as described above
  int gridSize;

  int tiles_streamk; ///< Number of StK tiles. Used also as indicator that this kernel is a StK instance.
  int combo_loop_streamk; ///< For StK: number of DP-combo loops per SM.
  // Splitted k for StK or SpK.
  // For SpK/Pure-DP: k-iterations per DP-or-SpK-threadblock.
  // For StK+DP-combo: number of k-iterations per threadblock in StK part.
  int gemm_k_slice;
  int *semaphore;

  const FastDivmod k_divmod;
  FastDivmod divmod[kMaxRank];

  CUTLASS_HOST_DEVICE
  GettParams(
    const cutlass::gemm::GemmCoord &problem_size_,
    const int32_t* blockingContracted_,
    const int32_t kNumBlockedModesContracted_,
    const MmaIteratorAParams &params_A_,
    const MmaIteratorBParams &params_B_,
    const EpilogueFastIteratorParams &params_C_,
    const EpilogueFastIteratorParams &params_D_,
    void *dataA_,
    void *dataB_,
    void *dataC_,
    void *dataD_,
    FreeLayoutN freeLayoutN_,
    const MmaIteratorParamsContracted &params_contracted_,
    const EpiligueAffineParams &affine_params_,
    const OutputOpParams &output_op_,
    const TensorStrideCoord &strideA_,
    const TensorStrideCoord &strideB_,
    const TensorStrideCoord &strideC_,
    const TensorCoord &extent_,
    const int numModesM_,
    const int numModesN_,
    const int numModesK_,
    const int numModesL_,
    const int64_t strideParallelK_,
    const int threadblockShapeM_,
    const int threadblockShapeN_,
    const bool useAffineEpilogue_,
    const int numSMs,
    const int nThreadblocks,
    const int partitions_,
    const int ccTarget,
    bool SplitK,
    bool StreamK
  ):
    problem_size(problem_size_),
    params_A(params_A_),
    params_B(params_B_),
    fast_params_C(params_C_),
    fast_params_D(params_D_),
    dataA(dataA_),
    dataB(dataB_),
    dataC(dataC_),
    dataD(dataD_),
    freeLayoutN(freeLayoutN_),
    params_contracted(params_contracted_),
    affine_params(affine_params_),
    output_op(output_op_),
    strideA(strideA_),
    strideB(strideB_),
    strideC(strideC_),
    numModesM(numModesM_),
    numModesN(numModesN_),
    numModesK(numModesK_),
    numModesL(numModesL_),
    strideParallelK(strideParallelK_),
    startM(numModesK_),
    startN(numModesK_ + numModesM_),
    startL(numModesK_ + numModesM_ + numModesN_),
    freeLayoutM(numModesM_, numModesK_, threadblockShapeM_, extent_),
    // freeLayoutN(extent_.template slice<kBlockedModesN>(startN), extent_.template slice<kBlockedModesN>(startN), threadblockShapeN_),
    useAffineEpilogue(useAffineEpilogue_),
    gemm_k_iterations(getLoopCount(extent_, blockingContracted_, kNumBlockedModesContracted_, numModesK_)),
    grid_tiled_shape(ThreadblockSwizzleBase::get_tiled_shape(extent_, freeLayoutM, freeLayoutN, startM, startN, startL, 1)),
    output_tile_num(grid_tiled_shape.product()),
    gridSize(output_tile_num),
    tiles_streamk(0),
    combo_loop_streamk(0),
    gemm_k_slice(gemm_k_iterations),
    k_divmod(gemm_k_iterations),
    semaphore(nullptr)
  { 
    for (int i = 0; i < TensorCoord::kRank; ++i)
    {
      divmod[i] = FastDivmod(extent_[i]);
    }

    int nBlocksPerSM = 0;
    if (numSMs)
      nBlocksPerSM = nThreadblocks / numSMs;
    auto partitions = partitions_ % (1 << kBitIndexStKFullWaves);
    auto full_waves = partitions_ / (1 << kBitIndexStKFullWaves);
    if (ccTarget >= 90) {
      // Exploit Hopper's scheduling advantage to use StK w/ more DP combos.
      full_waves = 1;
      // if (partitions <= 1)
      //   // Override the heuristics to fully exploit StK.
      //   partitions = 0;
    }

    if (StreamK && nThreadblocks > 0 && numSMs > 0 && nBlocksPerSM <= 2 && !partitions)
      updateGridSizeStreamK(numSMs, nBlocksPerSM, full_waves, ccTarget);
    else if (SplitK && partitions > 1)
      updateGridSizeSplitK(partitions);
  }

  CUTLASS_HOST_DEVICE
  void updateGridSizeStreamK(const int numSMs, const int nBlocksPerSM, const int full_waves, const int ccTarget)
  {
    combo_loop_streamk = fast_max(0, output_tile_num / numSMs - full_waves);
    if (combo_loop_streamk) {
      if (combo_loop_streamk * numSMs >= output_tile_num ||
          // In Pre-Ampere architectures, we have no explicit control over SM affinity.
          // Hence, we prefer direct load balance & hyper-blocking via DP.
          (ccTarget < 90 && combo_loop_streamk > full_waves + 1))
      { // Revert to DP.
        updateGridSizeDP();
        return;
      }
      // StK w/ DP combo.
      //
      tiles_streamk = output_tile_num - combo_loop_streamk * numSMs;
      int total_k_size_streamk = tiles_streamk * gemm_k_iterations;
      gemm_k_slice = ceil_div(total_k_size_streamk, numSMs);
      // This case the kernel is launched with twice the number of SMs for 2-occupancy scheduling.
      if (ccTarget < 90)
        gridSize = numSMs * 2;
      else
        gridSize = numSMs + combo_loop_streamk * numSMs;
    } else {
      // Pure StK.
      //
      tiles_streamk = output_tile_num;
      int total_k_size_streamk = tiles_streamk * gemm_k_iterations;
      gemm_k_slice = ceil_div(total_k_size_streamk, numSMs * nBlocksPerSM);
      // Schedule with maximum possible CTAs.
      gridSize = numSMs * nBlocksPerSM;
    }
  }

  CUTLASS_HOST_DEVICE
  void updateGridSizeDP()
  {
    tiles_streamk = 0;
    combo_loop_streamk = 0;
    gemm_k_slice = gemm_k_iterations;
    gridSize = output_tile_num;
  }

  CUTLASS_HOST_DEVICE
  void resetPartitions()
  {
    updateGridSizeDP();
  }

  CUTLASS_HOST_DEVICE
  void updateGridSizeSplitK(int tb_per_output_tile)
  {
    // StK off.
    tiles_streamk = 0;
    combo_loop_streamk = 0;
    // Split K dimension.
    gemm_k_slice = ceil_div(gemm_k_iterations, tb_per_output_tile);
    gridSize = tb_per_output_tile * output_tile_num;
  }

  CUTLASS_HOST_DEVICE
  int getLoopCount(const TensorCoord &extentK, const int32_t* blockingContracted, const int32_t kNumBlockedModesContracted, int32_t numModesK)
  {
      int loopCount = 1;
      for(int i=0; i < numModesK; ++i){
          const int32_t blocking = i < kNumBlockedModesContracted ? blockingContracted[i] : 1;
          loopCount *= (extentK[i] + blocking - 1) / blocking;
      }
      return loopCount;
  }

  CUTLASS_HOST_DEVICE
  bool splittingActive() const
  {
    bool useParallelK = strideParallelK > 0;
    return !useParallelK && // ParallelK does not induce a CUTLASS-level splitting.
      (tiles_streamk /* StreamK */ || gemm_k_slice < gemm_k_iterations /* SplitK */);
  }
};

} // namespace kernel
} // namespace contraction
} // namespace cutlass
