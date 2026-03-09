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
    \brief Template for a Universal GETT kernel with Split-K & Stream-K support.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/block_striped.h"
#include "cutlass/fast_math.h"
#include "cutlass/contraction/int_tuple.h"
#include "cutlass/contraction/kernel/gett_params.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

#include "cutlass/cuda_ptx_global_knobs.h" // {$nv-internal-release}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contraction {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
static const uint32_t kBitIndexStKFullWaves = 10U;

template<int rank, typename T>
CUTLASS_HOST_DEVICE
cutlass::Array<typename T::Storage, rank> make_array(const T &from) {
  cutlass::Array<typename T::Storage, rank> arr;
  arr.clear();

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < T::kElements; i++) {
    if (i < rank) 
      arr[i] = from[i];
  }
  return arr;
}

template <typename OutputOp>
CUTLASS_HOST_DEVICE
cutlass::epilogue::thread::LinearCombinationParams make_LinearCombinationParams(
  const typename OutputOp::Params &params
) {
  return cutlass::epilogue::thread::LinearCombinationParams(
    params.alpha_ptr ? *params.alpha_ptr : params.alpha,
    params.beta_ptr ? *params.beta_ptr : params.beta
  );
}

/// The total number of SM on the device
CUTLASS_DEVICE
int num_sm() {
  int ret;
  asm ("mov.u32 %0, %%nsmid;" : "=r"(ret) : );
  return ret;
}

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  int      ccTarget_,             ///! Target compute capability that this kernel will be compiled for
  bool     SplitKSerial = true,   ///! Ignored. SpK is always compiled with peer-sharing support.
  bool     StreamK = true         ///! Ignored. StK compilation is determined upon CTA count & k-access.
>
struct Gett {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using OutputOp = typename Epilogue::OutputOp;

  using TensorCoord = typename ThreadblockSwizzle::TensorCoord;
  using TensorCoordK = typename ThreadblockSwizzle::TensorCoordK;
  using TensorStrideCoord = Coord<TensorCoord::kRank, int64_t, int64_t>;
  using TensorStrideCoordK = Coord<TensorCoordK::kRank, int64_t, int64_t>;

  static const int ccTarget = ccTarget_;

  static const int kMaxRank = TensorCoord::kRank;
  static const int kNumBlockedModesContracted = Mma::IteratorA::kNumBlockedModesContracted;
  static const int kWarp = Mma::Base::WarpCount::kCount;
  static const int kThreadCount = 32 * kWarp;

  using StripedReduction = BlockStripedReduce<kThreadCount, typename Mma::FragmentC>;
  using ParamsBase = cutlass::contraction::kernel::GettParams<
                          Mma::IteratorA::FreeLayout::kRank,
                          Mma::IteratorB::FreeLayout::kRank,
                          kMaxRank >;
  using ShapeK = typename Mma::IteratorA::ShapeK;

  static const bool kSplitK = true;
  // StK compilation is disabled when:
  // 1. We don't expect it to be used.
  //    StK is heavy on memory access. We want each SM to run one StK per kernel, followed by (optional) DP combos.
  //    This requies CTAs we launch to have SM-affinities, with CUDA only supports up to 2 CTAs per SM.
  //    To estimate maxCTAs/SM at compile time, consider the most limiting resource: # of registers.
  //    maxCTAs/SM = (nRegs/SM) / (nRegs/CTA) = 65536 / (nThread/CTA * nRegs/Thread)
  //              >= 65536 / (nThread/CTA * 256) = 256 / (nThread/CTA) = 256 / (32 * kWarp) = 8 / kWarp.
  //    Hence, 8 / kWarp is the lower limit for "mCTAs/SM's upper limit" via register occupation.
  //    Future Work: Have to consider another upper limit casted by smem size.
  //    The 32 CTA/SM limit is never reached. Ignore it.
  // 2. Compiling it slows the kernel down.
  //    This part is done upon heuristics: Complicated k-layout seems causing kernel-side slowdown.
  //    Future Work: This effect seems only on < 64bit access types.
  static const bool kStreamK = (8 / kWarp) <= 2 &&
      (At<0, ShapeK>::value == At<Min<1, ShapeK::kRank-1>::kValue, ShapeK>::value ||
        At<Min<1, ShapeK::kRank-1>::kValue, ShapeK>::value == At<Min<2, ShapeK::kRank-1>::kValue, ShapeK>::value);

  // Parameters structure
  struct Params : ParamsBase {
    // [NOTE] Terminology used in this kernel.
    // As Stream-k somehow ``breaks down'' the k-dimension, making it more important
    // to distinguish concepts like ``tiles'' against ``blocks'' or ``CTAs'',
    // I'm writing here basically naming conventions of this kernel template:
    //  [tile] here refers only to the output, namely C&D-tiling.
    //    Tiles here are never connected to any k range whenever they're processed by
    //    Stream-k (StK), Data-Parallelism (DP) or Split-k (SpK).
    //  [block] here refers to some threadblock, as is the same as other parts of the CUTLASS library.
    //  [grid] is the collection of all blocks launched in a single ``<<< >>>'' call.
    //    Number of blocks can also be referred as the grid's size.
    //  [CTA] is generally AVOIDED from using since sometimes it's used to denote tiling size,
    //    others take it equivalent to a threadblock and still others use it as the number of tiles
    //    in some direction.

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      const cutlass::gemm::GemmCoord &problem_size_,
      const int32_t* blockingContracted_,
      const typename Mma::IteratorA::Params &params_A_,
      typename Mma::IteratorA::Element *dataA_,
      const typename Mma::IteratorB::Params &params_B_,
      typename Mma::IteratorB::Element *dataB_,
      typename Mma::IteratorB::FreeLayout freeLayoutN_,
      const typename Mma::IteratorB::ParamsContracted &params_contracted_,
      typename Epilogue::Affine::OutputTileIterator::TensorCoord affine_extent,
      typename Epilogue::Affine::OutputTileIterator::Layout      affine_layout,
      typename Epilogue::Fast::OutputTileIterator::TensorRef     linear_ref_C,
      typename Epilogue::Fast::OutputTileIterator::TensorRef     linear_ref_D,
      const typename OutputOp::Params &output_op,
      const TensorStrideCoord &strideA_,
      const TensorStrideCoord &strideB_,
      const TensorStrideCoord &strideC_,
      const TensorCoord &extent_,
      const int numModesM_,
      const int numModesN_,
      const int numModesK_,
      const int numModesL_,
      const int threadblockShapeM_,
      const int threadblockShapeN_,
      const bool useAffineEpilogue_,
      const int numSMs,
      const int nThreadblocks,
      const int partitions_,
      const int64_t strideParallelK_ = 0 ///< strideParallelK_ > 0 denotes that parallel-k ought to be used
    ): 
      ParamsBase(
        problem_size_,
        blockingContracted_,
        kNumBlockedModesContracted,
        params_A_,
        params_B_,
        typename Epilogue::Fast::OutputTileIterator::Params(linear_ref_C.layout()),
        typename Epilogue::Fast::OutputTileIterator::Params(linear_ref_D.layout()),
        dataA_,
        dataB_,
        linear_ref_C.data(),
        linear_ref_D.data(),
        freeLayoutN_,
        params_contracted_,
        typename Epilogue::Affine::OutputTileIterator::Params(affine_extent, affine_layout),
        make_LinearCombinationParams<OutputOp>(output_op),
        strideA_,
        strideB_,
        strideC_,
        extent_, 
        numModesM_,
        numModesN_,
        numModesK_,
        numModesL_,
        strideParallelK_,
        threadblockShapeM_,
        threadblockShapeN_,
        useAffineEpilogue_,
        numSMs,
        nThreadblocks,
        partitions_,
        ccTarget,
        kSplitK,
        kStreamK
      ) { }
  };

  // Extended semaphore for Stream-k
  struct SemaphoreStreamK : public Semaphore {
    CUTLASS_HOST_DEVICE
    SemaphoreStreamK(int *lock_, int thread_id):
      Semaphore(lock_, thread_id) { }

    /// Waits until the semaphore is greater or equal to the given value
    CUTLASS_DEVICE
    void wait_ge(int status = 0) {
      while( __syncthreads_and(state < status) ) {
        fetch();
      }

      __syncthreads();
    }


    /// Release increments
    CUTLASS_DEVICE
    void release_inc(int increment = 0)
    {
      __syncthreads();

      if (wait_thread) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        /// SM70 and newer use memory consistency qualifiers

        // Release pattern using acq_rel fence + relaxed modifier.  (The fence also releases data
        // that was weakly-written by other threads prior to the last syncthreads)
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(increment));

        #else
        __threadfence();
        atomicAdd(lock, increment);
        #endif
      }
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_DEVICE
  Gett()
  {
    // {$nv-internal-release begin}
    // Supermooch. TODO: Ask Dr. Duane Merrill what it does.
    global_knob_sched_mem_no_alias();
    global_knob_disable_war_sw2549067();
    global_knob_sched_res_busy_lsu_12cycles();
    global_knob_lsu_res_busy_size();
    global_knob_urf_promote_cond();
    // {$nv-internal-release end}
  }

  /// Executes one GETT, dispatching to consume_tile
  CUTLASS_DEVICE
  void operator()(
    const Params &params, SharedStorage &shared_storage)
  {
    int iloop = -1;

    do
    {
      // C-coordinates.
      int tile_idx;
      // K-coordinates.
      int k_iter_num, k_iter_begin;

      if (kStreamK && params.tiles_streamk)
      {
        // StK instance.
        // A 2D grid shall be used to launch this kernel case.

        // (Pre-Ampere) StK w/ DP combo launches 2 * numSMs blocks.
        // The first half is for StK and the last half is for DP combo.
        if (!params.combo_loop_streamk || blockIdx.x < num_sm()) {
          // StK block.

          int remainder;
          params.k_divmod(tile_idx, remainder, blockIdx.x * params.gemm_k_slice);
          if (iloop < 0) {
            int nloop;
            int total_k_size_streamk = params.tiles_streamk * params.gemm_k_iterations;
            params.k_divmod(nloop, remainder,
                fast_min<unsigned>((blockIdx.x + 1) * params.gemm_k_slice, total_k_size_streamk) - 1);
            nloop -= tile_idx;
            iloop = nloop;
          }
          tile_idx += iloop;

          // Location of this output tile in global k-iteration indexing.
          const int k_iter_by_tile = tile_idx * params.gemm_k_iterations;
          // Now compute the range of K to compute.
          // NOTE: Out-of-bound StK CTAs will have k_iter_begin == params.gemm_k_iterations thus gemm_k_size == 0,
          //       while k_iter_num is not guaranteed to be zero.
          k_iter_begin = fast_max<int>(0, blockIdx.x * params.gemm_k_slice - k_iter_by_tile);
          k_iter_num = (blockIdx.x + 1) * params.gemm_k_slice - k_iter_by_tile - k_iter_begin;
        } else if (ccTarget < 90) {
          // For Ampere or earlier where SM affinity cannot be fixed.
          // DP block

          const int numSMs = num_sm();
          // Offset StK tiles.
          tile_idx = params.tiles_streamk;
          tile_idx += blockIdx.x - numSMs; ///< One tile for each block.

          int nloop = params.combo_loop_streamk - 1; ///< Number of DP tiles per SM - 1.
          if (iloop < 0)
            iloop = nloop;

          // Tile increment: numSMs.
          tile_idx += (nloop - iloop) * numSMs;

          k_iter_begin = 0;
          // DP combo should contain full waves only.
          k_iter_num = params.gemm_k_iterations;
        } else {
          // Post-hopper architectures where block clusters can be utilized to attain SM affinity.
          // DP block

          tile_idx = (blockIdx.x - num_sm()) + params.tiles_streamk;
          iloop = 0;

          k_iter_begin = 0;
          k_iter_num = params.gemm_k_iterations;
        }
      }
      else
      {
        // DP / SpK instance.

        tile_idx = blockIdx.x;
        iloop = 0;

        k_iter_begin = 0;
        k_iter_num = params.gemm_k_slice; ///< @DP == gemm_k_iterations.
      }

      // Important: tile_idx must be passed in to override blockIdx.x.
      const auto &tiled_offset = ThreadblockSwizzle(params.freeLayoutM, params.freeLayoutN).get_tile_offset(params.grid_tiled_shape, tile_idx);

      uint64_t uoffsetA, uoffsetB, uoffsetC;
      compute_unblocked_offset(params,
        tiled_offset.unblockedM(),
        tiled_offset.unblockedN(),
        tiled_offset.batched(),
        uoffsetA, uoffsetB, uoffsetC);

      if (kSplitK) ///! For StK/DP, tile_idx < output_tile_num hence tiled_offset.k() is always zero.
      {
        // Adjust tile index to output coord.
        tile_idx -= tiled_offset.k() * params.output_tile_num;
        // K-coordinate of SpK case. Increment is zero otherwise.
        k_iter_begin += tiled_offset.k() * params.gemm_k_slice;
      }

      consume_tile(
        params,
        shared_storage,
        tile_idx,
        tiled_offset,
        k_iter_begin,
        k_iter_num,
        uoffsetA, uoffsetB, uoffsetC);

      if (kStreamK) {
        if (iloop-- <= 0)
          return;
        // If epilogue is executed, we have to wait for all threads to converge before we can proceed to the next StK tile.
        // Otherwise, fetching of A & B into SMem will cause accumulator to be overwritten.
        __syncthreads();
      }
    } while (kStreamK);
  }

  CUTLASS_DEVICE
  void compute_unblocked_offset(
    const Params &params,
    int idxM, int idxN, int idxL,
    uint64_t &uoffsetA,
    uint64_t &uoffsetB,
    uint64_t &uoffsetC)
  {
    int offset;
    uoffsetA = 0;
    uoffsetB = 0;
    uoffsetC = 0;

    CUTLASS_PRAGMA_UNROLL
    for  (int i = 0; i < TensorCoord::kRank; i++)
    {
        if (i < params.startM + ThreadblockSwizzle::kBlockedModesM_) continue;
        if (i >= params.startM + params.numModesM) break;
        params.divmod[i](idxM, offset, idxM);
        uoffsetA += params.strideA[i] * offset; // A
        uoffsetC += params.strideC[i] * offset;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TensorCoord::kRank; ++i)
    {
        if (i < params.startN + ThreadblockSwizzle::kBlockedModesN_) continue;
        if (i >= params.startN + params.numModesN) break;
        params.divmod[i](idxN, offset, idxN);
        uoffsetB += params.strideB[i] * offset; // B
        uoffsetC += params.strideC[i] * offset;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TensorCoord::kRank; ++i)
    {
        if (i < params.startL) continue;
        if (i >= params.startL + params.numModesL) break;
        params.divmod[i](idxL, offset, idxL);
        uoffsetA += params.strideA[i] * offset; // A
        uoffsetB += params.strideB[i] * offset; // B
        uoffsetC += params.strideC[i] * offset;
    }
  }

  // Process a range of tile-relative iterations for the given tile
  CUTLASS_DEVICE
  void consume_tile(
    const Params &params,
    SharedStorage &shared_storage,
    const int tile_idx,            ///< linear tile index
    const GettCoord &tiled_offset, ///< location of this tile in threadblock-tile coordinates
    const int k_iter_begin,        ///< first tile-relative iteration
    const int k_iter_num,          ///< number of tile-relative iterations to run
    const uint64_t uoffsetA,
    const uint64_t uoffsetB,
    const uint64_t uoffsetC)
  {
    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.blockedM()   <= tiled_offset.blockedM()   ||
        params.grid_tiled_shape.blockedN()   <= tiled_offset.blockedN()   ||
        params.grid_tiled_shape.unblockedM() <= tiled_offset.unblockedM() ||
        params.grid_tiled_shape.unblockedN() <= tiled_offset.unblockedN())
    { return; }

    // Restore tile index from computed offset.
    MatrixCoord::Index blockedM_idx(Mma::Shape::kM * tiled_offset.blockedM());
    MatrixCoord::Index blockedN_idx(Mma::Shape::kN * tiled_offset.blockedN());
    MatrixCoord::Index k_idx       (Mma::Shape::kK * k_iter_begin);

    //
    // Prologue
    //

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.params_contracted,
      params.freeLayoutM,
      static_cast<typename Mma::IteratorA::Element const*>(params.dataA),
      threadIdx.x,
      MatrixCoord(
        MatrixCoord::Index(k_idx),       // offset in K (i.e., extentK[0])
        MatrixCoord::Index(blockedM_idx) // offset in C (i.e., extentM[0])
      )
    );

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.params_contracted,
      params.freeLayoutN,
      static_cast<typename Mma::IteratorB::Element const*>(params.dataB),
      threadIdx.x,
      MatrixCoord(
        MatrixCoord::Index(k_idx),       // offset in K (i.e., extentK[0])
        MatrixCoord::Index(blockedN_idx) // offset in C (i.e., extentN[0])
      )
    );

    //
    // Offset A, B, C, D at the threadblock-level
    //
    typename Epilogue::Fast::OutputTileIterator::TensorRef ref_C {
        static_cast<typename Epilogue::Fast::ElementOutput*>(params.dataC), 
        typename Epilogue::Fast::OutputTileIterator::TensorRef::Layout()
    };
    typename Epilogue::Fast::OutputTileIterator::TensorRef ref_D {
        static_cast<typename Epilogue::Fast::ElementOutput*>(params.dataD), 
        typename Epilogue::Fast::OutputTileIterator::TensorRef::Layout()
    };

    // Offset pointers from computed values.
    iterator_A.add_pointer_offset(uoffsetA); // A
    iterator_B.add_pointer_offset(uoffsetB); // B
    ref_C.add_pointer_offset(uoffsetC);
    ref_D.add_pointer_offset(uoffsetC + tiled_offset.k() * params.strideParallelK);

    //
    // Main loop
    //
    const int gemm_k_size = fast_min(k_iter_num, params.gemm_k_iterations - k_iter_begin);

    const int warp_id = canonical_warp_idx_sync();
    const int lane_id =                         threadIdx.x % 32;
    const bool useParallelK = params.strideParallelK > 0;

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, threadIdx.x, warp_id, lane_id);

    typename Mma::FragmentC accumulators;

    // Clear accumulators
    accumulators.clear();

    if (gemm_k_size > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_size, accumulators, iterator_A, iterator_B, accumulators);
    } else if (params.gemm_k_iterations && ! useParallelK)  // if parallel K is used, make sure to execute the epilogue to initialize the reduction buffer
      // This is not a scaling-only problem. => This CTA is empty-spinning.
      return;

    //
    // Epilogue
    //    
    typename OutputOp::Params output_op_params = typename OutputOp::Params(
        reinterpret_cast<typename OutputOp::ElementCompute const &>(params.output_op.alpha_data),
        reinterpret_cast<typename OutputOp::ElementCompute const &>(params.output_op.beta_data)
      );
    OutputOp output_op(output_op_params);

    SemaphoreStreamK semaphore(params.semaphore + tile_idx, threadIdx.x);

    const int k_iter_end_ = k_iter_begin + gemm_k_size;
    const bool tile_finished_ = k_iter_end_ == params.gemm_k_iterations;
    const bool tile_started = k_iter_begin == 0;
    if (kStreamK || kSplitK) { // Stream-k or parallel split-k.
      if (!useParallelK) { // Peer reduction.
        // round_up(params.output_tile_num * sizeof(int), 64) // Alignment length=64. See below [Notes on Padding]
        int partials_shift = ((params.output_tile_num + ((1 << 4) - 1)) >> 4) << 4; // Round-up.
        int *partials_workspace = params.semaphore + partials_shift;
        typename Mma::FragmentC *accum_tile_workspace = reinterpret_cast<typename Mma::FragmentC *>(partials_workspace);
        int accum_tile_offset = tile_idx * kThreadCount;

        if (!tile_finished_) {
          // Share accumulators.
          if (tile_started) {
            StripedReduction::store(accum_tile_workspace + accum_tile_offset, accumulators, threadIdx.x);
          } else {
            // semaphore.wait_ge(1); ///< Wait until the tile starter has modified lock to a non-zero value.
            // Above behaviors is non-deterministic: Wait until previous CTA has finished contributing to this tile.
            semaphore.wait_ge(k_iter_begin);
            StripedReduction::reduce(accum_tile_workspace + accum_tile_offset, accumulators, threadIdx.x);
          }
          semaphore.release_inc(gemm_k_size); ///< With wait_ge below, using k_iter_num here should be legitimate also.
          return; // Skip epilogue.
        } else {
          if (!tile_started) {
            // Acquire accumulators.
            semaphore.fetch();
            semaphore.wait_ge(k_iter_begin);
            StripedReduction::load_add(accumulators, accum_tile_workspace + accum_tile_offset, threadIdx.x);
            semaphore.lock[0] = 0; // Async reset.
          }
          // Otherwise, it's equivalent to a DP. Just write.
        }
      }
    }

    MatrixCoord tb_offset_C(blockedM_idx, blockedN_idx);

    if (params.useAffineEpilogue) {

      // Tile iterator writing to destination tensor.
      typename Epilogue::Affine::OutputTileIterator iterator_D(
              params.affine_params, params.freeLayoutM, params.freeLayoutN,
              // #ifdef CUTLASS_TRANSPOSE
              // { make_Coord(8, 8), make_Coord(8, 1) },
              // #else
              // { make_Coord(8, 8), make_Coord(1, 8) },
              // #endif
              ref_D.data(),
              params.problem_size.mn(),
              threadIdx.x,
              tb_offset_C
              );

      typename Epilogue::Affine epilogue(shared_storage.epilogue.fast, threadIdx.x, warp_id, lane_id);

      // Execute the epilogue operator to update the destination tensor.
      if (!output_op.is_source_needed()) {
        epilogue(output_op, iterator_D, accumulators, typename Epilogue::Affine::SourceAspectNotNeeded());
      } else {
        // Tile iterator loading from source tensor.
        typename Epilogue::Affine::OutputTileIterator iterator_C(
                params.affine_params, params.freeLayoutM, params.freeLayoutN,
                // #ifdef CUTLASS_TRANSPOSE
                // { make_Coord(8, 8), make_Coord(8, 1) },
                // #else
                // { make_Coord(8, 8), make_Coord(1, 8) },
                // #endif
                ref_C.data(),
                params.problem_size.mn(),
                threadIdx.x,
                tb_offset_C
                );
        epilogue(output_op, iterator_D, accumulators, typename Epilogue::Affine::SourceAspectNeeded(iterator_C));
      }

    } else {
      // Tile iterator writing to destination tensor.
      typename Epilogue::Fast::OutputTileIterator iterator_D(
              params.fast_params_D,
              ref_D.data(),
              params.problem_size.mn(),
              threadIdx.x,
              tb_offset_C
              );
      typename Epilogue::Fast epilogue(shared_storage.epilogue.fast, threadIdx.x, warp_id, lane_id);

      // Execute the epilogue operator to update the destination tensor.
      if (!output_op.is_source_needed()) {
        epilogue(output_op, iterator_D, accumulators, typename Epilogue::Fast::SourceAspectNotNeeded());
      } else {
        // Tile iterator loading from source tensor.
        typename Epilogue::Fast::OutputTileIterator iterator_C(
                params.fast_params_C,
                ref_C.data(),
                params.problem_size.mn(),
                threadIdx.x,
                tb_offset_C
                );
        epilogue(output_op, iterator_D, accumulators, typename Epilogue::Fast::SourceAspectNeeded(iterator_C));
      }
    }
  }

  /// Gets the workspace size
  static size_t get_workspace_size_kernel(Params const &params) {
    size_t bytes = 0;

    auto tiled_shape = params.grid_tiled_shape;
    if (params.splittingActive()) {

      bytes += sizeof(int) * size_t(tiled_shape.blockedM()) * size_t(tiled_shape.blockedN()) * size_t(tiled_shape.unblockedM()) * size_t(tiled_shape.unblockedN()) * size_t(tiled_shape.batched());
      if (kStreamK || kSplitK) {
        // Padding to align starting address of scratchpads.
        // [Notes on Padding] Theoretically only needs to align addresses to StripedAccessType<typename Mma::FragmentC>,
        // but we cannot afford computing `( + sizeof(...) - 1) / sizeof(...) * sizeof(...)` in a CUDA kernel.
        // Hence, here we force aligning against 64 bytes (16 integers).
        // This should support up to 512bit vector registers (Currently registers are scalar hence <= 128bit).
        bytes += sizeof(int) * (((params.output_tile_num + ((1 << 4) - 1)) >> 4) << 4);

        // Additional scratchpad, excluding StK's DP-combo blocks.
        bytes += sizeof(typename Mma::FragmentC) * size_t(params.tiles_streamk ? params.tiles_streamk : params.output_tile_num) * kThreadCount;
      }
    }

    return bytes;
  }
};

} // namespace kernel
} // namespace contraction
} // namespace cutlass
