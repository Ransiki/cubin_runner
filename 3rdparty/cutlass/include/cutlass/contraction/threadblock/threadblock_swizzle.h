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
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to 
      GEMM problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "cutlass/contraction/contraction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contraction {
namespace threadblock {

struct Gemm1DRowThreadblockSwizzleBase {
  //////////////////////////////////////////////////////////////////////////////
  // Semi-opaque shape descriptors for FastDivmod-based tile shape descriptors.
  //////////////////////////////////////////////////////////////////////////////

  template<bool EnsureKInRange>
  struct TiledShapeInfoSemiFlat {
    const FastDivmod bM_divmod;
    const FastDivmod bN_divmod;
    const FastDivmod uM_divmod;
    const FastDivmod uN_divmod;
    const FastDivmod l_divmod;
    int K;

    static const int kEnsureKInRange = EnsureKInRange;

    CUTLASS_HOST_DEVICE int const&   blockedM() const { return bM_divmod.divisor; }
    CUTLASS_HOST_DEVICE int const&   blockedN() const { return bN_divmod.divisor; }
    CUTLASS_HOST_DEVICE int const& unblockedM() const { return uM_divmod.divisor; }
    CUTLASS_HOST_DEVICE int const& unblockedN() const { return uN_divmod.divisor; }
    CUTLASS_HOST_DEVICE int const&    batched() const { return l_divmod.divisor; }
    CUTLASS_HOST_DEVICE int const&          k() const { return K; }
    CUTLASS_HOST_DEVICE int      &          k()       { return K; }
    CUTLASS_HOST_DEVICE int product() const
    { return blockedM() * unblockedM() * blockedN() * unblockedN() * batched() * k(); }

    CUTLASS_HOST_DEVICE TiledShapeInfoSemiFlat(const GettCoord &shape):
      bM_divmod(shape.blockedM()), bN_divmod(shape.blockedN()),
      uM_divmod(shape.unblockedM()), uN_divmod(shape.unblockedN()),
      l_divmod(shape.batched()), K(shape.k()) { }
  };

  template< 
    typename FreeLayoutM,
    typename FreeLayoutN,
    typename TensorCoord
  >
  CUTLASS_HOST_DEVICE
  static GettCoord get_tiled_shape( 
    const TensorCoord &extent,
    FreeLayoutM freeLayoutM_,
    FreeLayoutN freeLayoutN_,
    const int32_t start_m,
    const int32_t start_n,
    const int32_t start_l,
    const int partitions = 1 ) 
  {
    using Index = typename TensorCoord::Index;
    
    const Index blockedM = freeLayoutM_.total_blocks;

    Index nCTAs = 1;
    // unblocked modes
    CUTLASS_PRAGMA_UNROLL
    for ( int i = start_m + FreeLayoutM::kRank; i < start_n; i ++ )
    {
        nCTAs *= extent[i];
    }

    const Index unblockedM = nCTAs;

    // blocked modes N
    const Index blockedN = freeLayoutN_.total_blocks;

    nCTAs = 1;
    // unblocked modes
    CUTLASS_PRAGMA_UNROLL
    for ( int i = start_n + FreeLayoutN::kRank; i < start_l; i ++ )
    {
        nCTAs *= extent[i];
    }
    const Index unblockedN = nCTAs;
    const Index k = partitions;

    Index batched = 1;
    CUTLASS_PRAGMA_UNROLL
    for ( int i = start_l; i < TensorCoord::kRank; ++ i )
    {
        batched *= extent[i];
    }

    return GettCoord(blockedM, unblockedM, blockedN, unblockedN, k, batched);
  }
};

/// Threadblock swizzling function for GEMMs
template<int kBlockRows,
         int kMaxRank,
         int kMaxContractedDim,
         typename FreeLayoutM,
         typename FreeLayoutN,
         bool kThreadblockLoopInM = true
        >
struct Gemm1DRowThreadblockSwizzle : Gemm1DRowThreadblockSwizzleBase {

  static const int kMaxRank_ = kMaxRank;
  static const int kMaxContracedDim_ = kMaxContractedDim;
  static const int kBlockedModesM_ = FreeLayoutM::kRank;
  static const int kBlockedModesN_ = FreeLayoutN::kRank;
  using Index = int32_t;
  using TensorCoord = Coord<kMaxRank,Index>;
  using TensorCoordK = Coord<kMaxContractedDim,Index>;

  FreeLayoutN freeLayoutN_;
  FreeLayoutM freeLayoutM_;

  // Plain GettCoord is the most straightforward proxy to tiled shape information:
  //   using TiledShapeInfo = GettCoord;
  // But computing integer division (/) and modulo (%) on the device is taking extra time.
  // Hence here we only keep the implementation using FastDivmod.
  using TiledShapeInfo = TiledShapeInfoSemiFlat<false>;

  //////////////////////////////////////////////////////////////////////////////
  // Swizzling functors.
  //////////////////////////////////////////////////////////////////////////////

  CUTLASS_HOST_DEVICE
  Gemm1DRowThreadblockSwizzle() {}

  CUTLASS_HOST_DEVICE
  Gemm1DRowThreadblockSwizzle(const FreeLayoutM& freeLayoutM, const FreeLayoutN& freeLayoutN)
      : freeLayoutM_(freeLayoutM), freeLayoutN_(freeLayoutN) {}

  CUTLASS_HOST_DEVICE
  GettCoord get_tiled_shape( const TensorCoord &extent,
                             const int32_t start_m,
                             const int32_t start_n,
                             const int32_t start_l,
                             const int partitions = 1) const
  {
      return Gemm1DRowThreadblockSwizzleBase::get_tiled_shape(extent, freeLayoutM_, freeLayoutN_, start_m, start_n, start_l, partitions);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  uint32_t get_grid_shape( const TensorCoord &extent,
                           const int32_t start_m,
                           const int32_t start_n,
                           const int32_t start_l,
                           const int partitions = 1) const
  {
      const auto gettCoord = get_tiled_shape(extent,
                                             start_m, start_n, start_l,
                                             partitions);
      return gettCoord.product();
  }

  CUTLASS_HOST_DEVICE
  GettCoord get_tile_offset(const FastDivmod &bM_divmod,
                            const FastDivmod &bN_divmod,
                            const FastDivmod &uM_divmod,
                            const FastDivmod &uN_divmod,
                            const FastDivmod &l_divmod,
                            int blockIdx) const
  {
    int blockedM, unblockedM;
    int blockedN, unblockedN;
    if (kThreadblockLoopInM) {
      // Default: N after M.
      bM_divmod(blockIdx, blockedM, blockIdx);
      bN_divmod(blockIdx, blockedN, blockIdx);
      uN_divmod(blockIdx, unblockedN, blockIdx);
      uM_divmod(blockIdx, unblockedM, blockIdx);
    } else {
      // Transposed: M after N.
      bN_divmod(blockIdx, blockedN, blockIdx);
      bM_divmod(blockIdx, blockedM, blockIdx);
      uM_divmod(blockIdx, unblockedM, blockIdx);
      uN_divmod(blockIdx, unblockedN, blockIdx);
    }

    int batched, k;
    l_divmod(blockIdx, batched, blockIdx);
    k = blockIdx;

    return GettCoord{
      uint32_t(blockedM), uint32_t(unblockedM),
      uint32_t(blockedN), uint32_t(unblockedN),
      uint32_t(k), uint32_t(batched)
    };
  }

  CUTLASS_HOST_DEVICE
  GettCoord get_tile_offset(const TiledShapeInfo &info, int blockIdx) const
  {
    auto tile_offset = get_tile_offset(
      info.bM_divmod, info.bN_divmod,
      info.uM_divmod, info.uN_divmod,
      info.l_divmod, blockIdx);

    // Additional safe guard around K dimension.
    if (TiledShapeInfo::kEnsureKInRange) {
      tile_offset.k() %= info.k();
    }
    return tile_offset;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace contraction
} // namespace cutlass

