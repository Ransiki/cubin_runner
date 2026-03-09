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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

namespace cutlass {
namespace contraction {

namespace threadblock {

// The FreeAccessLayout encodes how blocked free modes (M or N) are accessed.
// It is used in the  iterators for A and B, as well as in the epilogue.
//
// The FreeAccessLayout supports two modes of operations:
// 1. A fused mode (which is how we used to handle this), in which we have
//    an iteration space of extent[0] * ... * extent[num_blocked_extents-1] and
//    block that iteration space using the threadblock blocking.
// 2. A blocked mode, where each mode is blocked individually. This is primarily
//    useful to increase reuse if the stride-1 mode in A and C are not identical.
//
// For option (1), the math works out best if we set a ficticious blocking that
// is equal to the extent. That way, all the required calculations are performed
// at the "element" level, and the block-level reduces to no-ops.

template<
    int kRank_  /// Rank of the free access layout, i.e. how many modes can either be fused or blocked
>
struct FreeAccessLayout {

    static const int kRank = kRank_;
    using Index = int32_t;
    using TensorCoord = Coord<kRank, Index>;

    TensorCoord extent;
    TensorCoord blocking;
    Index total_blocks;
    FastDivmod divmod_block[kRank];
    FastDivmod divmod_num_blocks[kRank];

    CUTLASS_HOST_DEVICE
    FreeAccessLayout() {}

    CUTLASS_HOST_DEVICE
    FreeAccessLayout(TensorCoord extent, TensorCoord blocking, int kBlock)
      : blocking(blocking), extent(extent) 
    {
        Index padded_elements = 1;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRank; i++) {
            Index num_blocks = ceil_div(extent[i], blocking[i]);
            divmod_block[i] = FastDivmod(blocking[i]);
            divmod_num_blocks[i] = FastDivmod(num_blocks);
            padded_elements *= blocking[i] * num_blocks;
        }
        total_blocks = ceil_div(padded_elements, kBlock);
    }

    CUTLASS_HOST_DEVICE
    FreeAccessLayout(TensorCoord extent, int kBlock) : extent(extent) {
        Index padded_elements = 1;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRank; i++) {
            blocking[i] = extent[i];
            divmod_block[i] = FastDivmod(blocking[i]);
            divmod_num_blocks[i] = FastDivmod(1);
            padded_elements *= extent[i];
        }
        total_blocks = ceil_div(padded_elements, kBlock);
    }

    template<typename OtherTensorCoord>
    CUTLASS_HOST_DEVICE
    FreeAccessLayout(int num, int start, int kBlock, const OtherTensorCoord &otherExtent) {
        Index padded_elements = 1;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRank; i++) {
            extent[i] = i < num ? otherExtent[start + i] : 1;
            blocking[i] = extent[i];
            divmod_block[i] = FastDivmod(blocking[i]);
            divmod_num_blocks[i] = FastDivmod(1);
            padded_elements *= blocking[i];
        }
        total_blocks = ceil_div(padded_elements, kBlock);
    }

    CUTLASS_HOST_DEVICE
    TensorCoord get_coord(Index element, bool& mask) const {
        TensorCoord result = get_element_reduce(element);
        result += get_block_reduce(element);
        mask = (element == 0) && (result < extent);
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorCoord get_block_reduce(Index& blockIndex) const {
        TensorCoord result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRank; i++) {
            Index pos = 0;
            divmod_num_blocks[i](blockIndex, pos, blockIndex);
            result[i] = pos * blocking[i];
        }
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorCoord get_element_reduce(Index& element) const {
        TensorCoord result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kRank; i++) {
            Index pos = 0;
            divmod_block[i](element, pos, element);
            result[i] = pos;
        }
        return result;
    }
};

}  // namespace threadblock

}  // namespace contraction

}  // namespace cutlass
