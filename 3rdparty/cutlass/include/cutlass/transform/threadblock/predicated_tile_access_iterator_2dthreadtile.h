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
    \brief Templates calculating the address and predicates to the load of tiles
   from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last
   "residue" tile first, with the objective of minimizing predicate mask updates
   during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator_params.h"

////////////////////////////////////////////////////////////////////////////////

// {$nv-internal-release begin}
#if ! defined(CUTLASS_CUDA_RP2RP_ENABLED)
#define CUTLASS_CUDA_RP2RP_ENABLED 0
#endif
// {$nv-internal-release end}

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileAccessIterator2dThreadTile
///
template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType>
class PredicatedTileAccessIterator2dThreadTile;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator2dThreadTile for pitch-linear data.
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator2dThreadTile<Shape_, Element_, layout::PitchLinear,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  static int const kPredicatesPerByte = 4;
  static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

  /// Number of 32b words containing predicates
  static int const kPredicateByteCount = (ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kStrided + kPredicatesPerByte - 1) / kPredicatesPerByte;
  static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

  static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

  static_assert(kPredicateWordCount <= 4, "Too many predicates.");

  /// Predicate vector stores mask to guard accesses
  using Mask = Array<uint32_t, kPredicateWordCount>;

  /// Uses a non-template class
  struct Params : PredicatedTileAccessIteratorParams {

   public:
    friend PredicatedTileAccessIterator2dThreadTile;

    using Base = PredicatedTileAccessIteratorParams;

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : 
      Base(layout.stride(0),
            MakePredicatedTileAccessIteratorDesc<Shape, Element, Layout, kAdvanceRank, ThreadMap>()()
        ) { }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) : 
      Base(base) { }
  };


 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Parameters object with precomputed internal state
  Params const &params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;

  /// Guard predicates
  uint32_t predicates_[kPredicateWordCount];

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))

  /// Cache of predicates filled from general purpose register
  bool predicate_cache_[kPredicatesPerByte];

  /// Tracks current predicate word
  int word_idx_;

  /// Tracks current predicate byte
  int byte_idx_;

  /// Tracks predicate bit within byte
  int pred_idx_;

#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

  /// Size of tensor
  TensorCoord extent_;

  /// Initial offset for each thread
  TensorCoord thread_offset_;

  /// Index of residue tile
  int residue_tile_idx_;

  /// Used for out-of-order visitation
  bool is_residue_tile_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

  /// Tracks iterations within the thread loop
  int iteration_thread_;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_HOST_DEVICE
  void compute_predicates_(
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0u;
    }

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))

    int word_idx = 0;
    int byte_idx = 0;
    int pred_idx = 0;

    bool preds[kPredicatesPerByte];

    __nv_r2p(0, preds, kPredicateMask, 0);

#endif  // if CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++) {

          TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous,
                                      ts + s * ThreadMap::Delta::kStrided);

          TensorCoord coord = thread_offset_ + iteration_coord;

          bool guard;

          if (is_steady_state) {
            if (kAdvanceRank == 0) {
              guard = (coord.strided() < extent_.strided());
            } else {
              guard = (coord.contiguous() < extent_.contiguous());
            }
          } else {
            guard = (coord.strided() < extent_.strided() &&
                     coord.contiguous() < extent_.contiguous());
          }

#if (!(CUTLASS_CUDA_RP2RP_ENABLED) || !defined(__CUDA_ARCH__)) // {$nv-internal-release}

          int pred_idx = ts + c *  ThreadMap::ThreadAccessShape::kStrided + s * ThreadMap::Iterations::kContiguous *  ThreadMap::ThreadAccessShape::kStrided;
          int word_idx = pred_idx / kPredicatesPerWord;
          int residual = pred_idx % kPredicatesPerWord;
          int byte_idx = residual / kPredicatesPerByte;
          int bit_idx = residual % kPredicatesPerByte;
          
          predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));

  // {$nv-internal-release begin}
#else
          if (pred_idx == 0) {
            preds[0] = guard;
          }
          else if (pred_idx == 1) {
            preds[1] = guard;
          }
          else if (pred_idx == 2) {
            preds[2] = guard;
          }
          else if (pred_idx == 3) {
            preds[3] = guard;
          }

          ++pred_idx;

          if (pred_idx == kPredicatesPerByte) {
            uint32_t gpr;

            if (word_idx == 0) {
              gpr = predicates_[0];
            }
            else if (word_idx == 1) {
              gpr = predicates_[1];
            }
            else if (word_idx == 2) {
              gpr = predicates_[2];
            }
            else if (word_idx == 3) {
              gpr = predicates_[3];
            }

            if (byte_idx == 0) {
              __nv_p2r(0, preds, kPredicateMask, &gpr);
            }
            else if (byte_idx == 1) {
              __nv_p2r(1, preds, kPredicateMask, &gpr);
            }
            else if (byte_idx == 2) {
              __nv_p2r(2, preds, kPredicateMask, &gpr);
            }
            else if (byte_idx == 3) {
              __nv_p2r(3, preds, kPredicateMask, &gpr);
            }

            if (word_idx == 0) {
              predicates_[0] = gpr;
            }
            else if (word_idx == 1) {
              predicates_[1] = gpr;
            }
            else if (word_idx == 2) {
              predicates_[2] = gpr;
            }
            else if (word_idx == 3) {
              predicates_[3] = gpr;
            }

            pred_idx = 0;
            ++byte_idx;
            if (byte_idx == 4) {
              byte_idx = 0;
              ++word_idx;
            }
          }
#endif  // if CUTLASS_CUDA_RP2RP_ENABLED
  // {$nv-internal-release end}
        }
      }
    }

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))
    if (pred_idx) {
      
      uint32_t gpr;

      if (word_idx == 0) {
        gpr = predicates_[0];
      }
      else if (word_idx == 1) {
        gpr = predicates_[1];
      }
      else if (word_idx == 2) {
        gpr = predicates_[2];
      }
      else if (word_idx == 3) {
        gpr = predicates_[3];
      }

      if (byte_idx == 0) {
        __nv_p2r(0, preds, kPredicateMask, &gpr);
      }
      else if (byte_idx == 1) {
        __nv_p2r(1, preds, kPredicateMask, &gpr);
      }
      else if (byte_idx == 2) {
        __nv_p2r(2, preds, kPredicateMask, &gpr);
      }
      else if (byte_idx == 3) {
        __nv_p2r(3, preds, kPredicateMask, &gpr);
      }

      if (word_idx == 0) {
        predicates_[0] = gpr;
      }
      else if (word_idx == 1) {
        predicates_[1] = gpr;
      }
      else if (word_idx == 2) {
        predicates_[2] = gpr;
      }
      else if (word_idx == 3) {
        predicates_[3] = gpr;
      }
    }
#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}
  }

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))

  /// Fills predicate cache from the appropriate predicate_ word
  CUTLASS_DEVICE
  void fill_predicates_() {
    uint32_t gpr;

    if (word_idx_ == 0) {
      gpr = predicates_[0];
    }
    else if (word_idx_ == 1) {
      gpr = predicates_[1];
    }
    else if (word_idx_ == 2) {
      gpr = predicates_[2];
    }
    else if (word_idx_ == 3) {
      gpr = predicates_[3];
    }

    if (byte_idx_ == 0) {
      __nv_r2p(0, predicate_cache_, kPredicateMask, gpr);
    }
    else if (byte_idx_ == 1) {
      __nv_r2p(1, predicate_cache_, kPredicateMask, gpr);
    }
    else if (byte_idx_ == 2) {
      __nv_r2p(2, predicate_cache_, kPredicateMask, gpr);
    }
    else if (byte_idx_ == 3) {
      __nv_r2p(3, predicate_cache_, kPredicateMask, gpr);
    }
  }
#endif // if CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        extent_(extent),
        is_residue_tile_(true) {
          

    TensorCoord residue_offset;
    if (kAdvanceRank) {
      residue_tile_idx_ =
          (extent_[kAdvanceRank] - threadblock_offset[kAdvanceRank] - 1) /
          Shape::kStrided;
      residue_offset = make_Coord(0, residue_tile_idx_ * Shape::kStrided);
    } else {
      residue_tile_idx_ =
          (extent_[kAdvanceRank] - threadblock_offset[kAdvanceRank] - 1) /
          Shape::kContiguous;
      residue_offset = make_Coord(residue_tile_idx_ * Shape::kContiguous, 0);
    }

    // Per-thread offset in logical coordinates of tensor
    thread_offset_ = threadblock_offset + residue_offset +
                     ThreadMap::initial_offset(thread_id);

    // update internal pointers
    Layout layout(params_.stride_);
    add_pointer_offset(layout(thread_offset_));

    compute_predicates_(false);

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))

    word_idx_ = -1;
    byte_idx_ = -1;
    pred_idx_ = 0;

#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

    set_iteration_index(0);
  }

  /// Construct a PredicatedTileAccessIterator2dThreadTile with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id)
      : PredicatedTileAccessIterator2dThreadTile(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {

    int residual = index % (ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided);
    iteration_strided_ = index / (ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided);
    
    iteration_contiguous_ = residual / ThreadMap::ThreadAccessShape::kStrided;
    iteration_thread_ = residual % ThreadMap::ThreadAccessShape::kStrided;

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))

    word_idx_ = index / (kPredicatesPerByte * 4);
    int predicate_residual = index % (kPredicatesPerByte * 4);

    byte_idx_ = predicate_residual / kPredicatesPerByte;
    pred_idx_ = predicate_residual % kPredicatesPerByte;

    fill_predicates_();

#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += int(sizeof(Element)) * pointer_offset;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {
    if (is_residue_tile_) {
      TensorCoord residue_offset;
      if (kAdvanceRank) {
        residue_offset = TensorCoord(0, residue_tile_idx_ * Shape::kStrided);
      } else {
        residue_offset = TensorCoord(residue_tile_idx_ * Shape::kContiguous, 0);
      }

      thread_offset_ -= residue_offset;

      Layout layout(params_.stride_);
      add_pointer_offset(-layout(residue_offset));

      compute_predicates_(true);

      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * (tile_offset.strided() - 1);
        pointer_ += Shape::kContiguous * tile_offset.contiguous();
      } else {
        pointer_ += params_.inc_advance_ * (tile_offset.contiguous() - 1);
        pointer_ += Shape::kStrided * tile_offset.strided();
      }
    } else {
      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * tile_offset.strided();
        pointer_ += Shape::kContiguous * tile_offset.contiguous();
      } else {
        pointer_ += params_.inc_advance_ * tile_offset.contiguous();
        pointer_ += Shape::kStrided * tile_offset.strided();
      }
    }
    is_residue_tile_ = false;
  }

  CUTLASS_HOST_DEVICE
  AccessType *get() const {

    AccessType *ret_val = reinterpret_cast<AccessType *>(
                pointer_ + (iteration_thread_ * params_.stride_  + iteration_contiguous_ * ThreadMap::Delta::kContiguous) * int(sizeof(Element)));

    return ret_val;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile &operator++() {

    iteration_thread_++;

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))
    ++pred_idx_;
    if (pred_idx_ == kPredicatesPerByte) {
      pred_idx_ = 0;
      ++byte_idx_;
 
      if (byte_idx_ == kPredicateByteCount) {
        byte_idx_ = 0;
        ++word_idx_;

        if (word_idx_ == kPredicateWordCount) {
          word_idx_ = 0;
        }
      }

      fill_predicates_();
    }
#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

    if (iteration_thread_ < ThreadMap::ThreadAccessShape::kStrided)
      return *this;

    iteration_thread_ = 0;

    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_ += params_.inc_strided_;
      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    // advance to next tile
    pointer_ += params_.inc_next_;

    // now return to start tile - if the iterator is subsequently advanced, this
    // subtraction as well as the subsequent integer addition are both elided by
    // the compiler.
    pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile operator++(int) {
    PredicatedTileAccessIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = enable ? 0u : predicates_[i];
    }

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))
    if (enable) {
      __nv_r2p(0, predicate_cache_, kPredicateMask, 0);
    }
#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}

  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0xffffffff;
    }
// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))
    __nv_r2p(0, predicate_cache_, kPredicateMask, kPredicateMask);
#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { 
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = mask[i];
    }

// {$nv-internal-release begin}
#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__))
    fill_predicates_();
#endif // CUTLASS_CUDA_RP2RP_ENABLED
// {$nv-internal-release end}
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
     CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      mask[i] = predicates_[i];
    }
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {

#if (CUTLASS_CUDA_RP2RP_ENABLED && defined(__CUDA_ARCH__)) // {$nv-internal-release begin}

    bool guard;

    if (pred_idx_ == 0) {
      guard = predicate_cache_[0];
    }
    else if (pred_idx_ == 1) {
      guard = predicate_cache_[1];
    }
    else if (pred_idx_ == 2) {
      guard = predicate_cache_[2];
    }
    else if (pred_idx_ == 3) {
      guard = predicate_cache_[3];
    }

    return guard;

#else                                                      // {$nv-internal-release end}

    int pred_idx = 
      iteration_thread_ + 
      iteration_contiguous_ * ThreadMap::ThreadAccessShape::kStrided + 
      iteration_strided_ * ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided;

    int word_idx = pred_idx / kPredicatesPerWord;
    int residual = pred_idx % kPredicatesPerWord;
    int byte_idx = residual / kPredicatesPerByte;
    int bit_idx = residual % kPredicatesPerByte;
    
    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    
    return pred;
#endif                                                     // {$nv-internal-release}
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator2dThreadTile<Shape_, Element_, layout::ColumnMajor,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileAccessIterator2dThreadTile<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap, AccessType>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIterator2dThreadTile;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))){}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base) 
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row(), extent.column()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.row(),
                                           threadblock_offset.column())) {}

  /// Construct a PredicatedTileAccessIterator2dThreadTile with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileAccessIterator2dThreadTile(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile operator++(int) {
    PredicatedTileAccessIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator2dThreadTile<Shape_, Element_, layout::RowMajor,
                                   AdvanceRank, ThreadMap_, AccessType_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileAccessIterator2dThreadTile<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, AccessType>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileAccessIterator2dThreadTile;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))){}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base) 
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  /// Construct a PredicatedTileAccessIterator2dThreadTile with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileAccessIterator2dThreadTile(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator2dThreadTile operator++(int) {
    PredicatedTileAccessIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
