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
    \brief Tile access iterator to load filters for Conv2d Fprop. This uses straightforward,
    analytic functions which can be seen to be correct by inspection.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/memory.h"
#include "cutlass/contraction/int_tuple.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contraction {
namespace threadblock {

using namespace cutlass::contraction; // TODO remove (used for int_tuple)

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int kRank>
class FusedTensorNdimIteratorParamsContracted
{
    public:
        using CoordExtentsContracted = Coord<kRank, int32_t>;

        CUTLASS_HOST_DEVICE
        FusedTensorNdimIteratorParamsContracted() { }

        template<typename ArrayShapeK> 
        CUTLASS_HOST_DEVICE
        FusedTensorNdimIteratorParamsContracted(
                CoordExtentsContracted const &extentsContracted,
                ArrayShapeK const& blockingContracted,
                int kNumModesContracted,
                int kNumBlockedModesContracted ) :
            extentsContracted_(extentsContracted)
        {
            CUTLASS_PRAGMA_UNROLL
            for(int i=0; i < kNumBlockedModesContracted; ++i)
            {
                divmodContracted_[i] = FastDivmod((extentsContracted_[i] + blockingContracted[i] - 1) / blockingContracted[i]);
            }
            
            CUTLASS_PRAGMA_UNROLL
            for(int i=kNumBlockedModesContracted; i < kNumModesContracted; ++i)
            {
                divmodContracted_[i] = FastDivmod(extentsContracted_[i]);
            }
        }

        template<int kRankOther_>
        CUTLASS_HOST_DEVICE
        FusedTensorNdimIteratorParamsContracted(
                FusedTensorNdimIteratorParamsContracted<kRankOther_> const& other )
        : extentsContracted_(other.extentsContracted_.template slice<kRank>())
        {
            CUTLASS_PRAGMA_UNROLL
            for(int i=0; i < const_min(kRankOther_, kRank); ++i)
            {
                divmodContracted_[i] = other.divmodContracted_[i];
            }
        }

        const CoordExtentsContracted extentsContracted_;
        FastDivmod divmodContracted_[kRank];
};


template< int kRank, int kNumModesFree >
struct FusedTensorNdimIteratorParams
{
    using CoordStridesContracted = Coord<kRank, int64_t, int64_t>;
    using CoordExtentsContracted = Coord<kRank, int32_t>;
    using CoordStridesFree = Coord<kNumModesFree, int64_t, int64_t>;
    using CoordExtentsFree = Coord<kNumModesFree, int32_t>;
 
    const CoordStridesFree stridesFree_;
    const CoordStridesContracted stridesContracted_;
    CoordStridesContracted incContracted_;

    CUTLASS_HOST_DEVICE
    FusedTensorNdimIteratorParams() { }
    
    template<typename ArrayShapeK> 
    CUTLASS_HOST_DEVICE
    FusedTensorNdimIteratorParams(
            CoordExtentsFree const &extentsFree,
            CoordStridesFree const &stridesFree,
            CoordExtentsContracted const &extentsContracted,
            CoordStridesContracted const &stridesContracted,
            ArrayShapeK const& blockingContracted,
            int kNumModesContracted,
            int kNumBlockedModesContracted ) 
    :   stridesFree_(stridesFree), stridesContracted_(stridesContracted)
    {
        incContracted_[0] = stridesContracted_[0] * blockingContracted[0];
        CUTLASS_PRAGMA_UNROLL
        for(int i=1; i < kNumModesContracted; ++i)
        {
            const int32_t blockingPrev = ((i-1) < kNumBlockedModesContracted) ? blockingContracted[i-1] : 1;
            const int32_t blocking = (i < kNumBlockedModesContracted) ? blockingContracted[i] : 1;
            int32_t lastPosPrev = ((extentsContracted[i-1] + blockingPrev - 1)/blockingPrev) * blockingPrev; // roundup to next multiple of blockingContracted[i-1]
            incContracted_[i] = /* (1) */- (stridesContracted_[i-1] * lastPosPrev)
                                /* (2) */+ stridesContracted_[i] * blocking;
        }
    }

    template<int kRankOther_>
    CUTLASS_HOST_DEVICE
    FusedTensorNdimIteratorParams(
            FusedTensorNdimIteratorParams<kRankOther_, kNumModesFree> const& other )
    : stridesFree_(other.stridesFree_)
    , stridesContracted_(other.stridesContracted_.template slice<kRank>())
    , incContracted_(other.incContracted_.template slice<kRank>())
    { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
