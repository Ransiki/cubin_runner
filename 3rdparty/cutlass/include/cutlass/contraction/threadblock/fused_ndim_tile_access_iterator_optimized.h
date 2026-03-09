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
#include "cutlass/contraction/threadblock/fused_tensor_ndim_iterator_params.h"
#include "cutlass/contraction/threadblock/free_access_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contraction {
namespace threadblock {

using namespace cutlass::contraction; // TODO remove (used for int_tuple)

/////////////////////////////////////////////////////////////////////////////////////////////////

//template<int... tilings> struct Rank;
//template<int i, int... tilings> struct Rank<i, tilings...>
//{
//    static const int value = 1 + Rank<tilings...>::value;
//};
//template<> struct Rank<>
//{
//    static const int value = 0;
//};


template <
  typename Element_,
  int kElementsPerAccess,
  typename ThreadMap_,
  bool isContractedContiguous_,
  typename ShapeK_, // concept: IntTuple
  typename FreeLayout_,
  int kMaxRank
>
class MyIterator
{

    public:

    static const int numModesFree = FreeLayout_::kRank;
    using ShapeK = ShapeK_;
    static const bool isContractedContiguous = isContractedContiguous_;
    static const int kNumModesContracted = ShapeK::kRank;
    static const int kNumModesFree = numModesFree;
    static const int kNumBlockedModesContracted = Count<ShapeK>::value;
    static_assert(kNumModesContracted == ShapeK::kRank, "ranks don't match");
    using CoordStridesContracted = Coord<kNumModesContracted, int64_t, int64_t>;
    using CoordExtentsContracted = Coord<kNumModesContracted, int32_t>;
    using CoordStridesFree = Coord<kNumModesFree, int64_t, int64_t>;
    using CoordExtentsFree = Coord<kNumModesFree, int32_t>;
    using ArrayShapeK = Array<int32_t, kNumBlockedModesContracted>;
    const Tuple<ShapeK> blockingContracted;
    using FreeLayout = FreeLayout_;

    using ParamsBase = FusedTensorNdimIteratorParams<kNumModesContracted, kNumModesFree>;
    using ParamsContracted = FusedTensorNdimIteratorParamsContracted<kNumModesContracted>;

    struct Params : ParamsBase
    {
        CUTLASS_HOST_DEVICE
        Params(
                CoordExtentsFree const &extentsFree,CoordStridesFree const &stridesFree,
                CoordExtentsContracted const &extentsContracted,
                CoordStridesContracted const &stridesContracted,
                ArrayShapeK const& blockingContracted ) 
        : ParamsBase(
                extentsFree,
                stridesFree, 
                extentsContracted,
                stridesContracted,
                blockingContracted,
                kNumModesContracted,
                kNumBlockedModesContracted )
        { }

        CUTLASS_HOST_DEVICE
        Params(ParamsBase const& base)
        : ParamsBase(base) 
        { }

    };

    using Index = int32_t;
    using LongIndex = int64_t;
    using ThreadMap = ThreadMap_;
    using Element = Element_;
    static const int kElementsPerAccess_ = kElementsPerAccess;
    using AccessType = AlignedArray<Element, kElementsPerAccess_>;

    static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
    static int const kAccessesPerVectorFree = isContractedContiguous ? 1: kAccessesPerVector;
    static int const kAccessesPerVectorContracted = (!isContractedContiguous) ? 1: kAccessesPerVector;

    /// Fragment object to be loaded or stored
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kCount * kElementsPerAccess_ * kAccessesPerVector >;

    static int const kContiguousElementsPerThread_ = kAccessesPerVector * kElementsPerAccess_;
  
    static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements), 
        "Vectors implied by the thread map must be divisible by the access type.");

    static const int kElementsPerAccessContracted_ =  isContractedContiguous ? kElementsPerAccess_ * kAccessesPerVector : 1;
    static const int kElementsPerAccessFree_       =  !isContractedContiguous ? kElementsPerAccess_ * kAccessesPerVector : 1;

    // num iterations along the free modes
    static const int kNumIterationsFree_       =  isContractedContiguous ? ThreadMap::Iterations::kStrided :
                                                                           ThreadMap::Iterations::kContiguous;
    static const int kNumIterationsContracted_ = !isContractedContiguous ? ThreadMap::Iterations::kStrided :
                                                                           ThreadMap::Iterations::kContiguous;
    // (linear) offset between two elements along the free modes
    static const int kDeltaFree       =  isContractedContiguous ? ThreadMap::Delta::kStrided : ThreadMap::Delta::kContiguous;
    static const int kDeltaContracted_ = !isContractedContiguous ? ThreadMap::Delta::kStrided : ThreadMap::Delta::kContiguous;

    const Element *pointer_[kNumIterationsFree_ * kAccessesPerVectorFree]; // one pointer for each element along the free dimension (since those stay constant)

    int32_t coordContracted_[kNumModesContracted]; // keeps track of the current coord w.r.t. the contracted modes (i.e. all indices)
    int32_t resetCoordContracted_[kNumBlockedModesContracted]; // keeps track of the initial coord w.r.t. the contracted modes (i.e. all indices)

    const Params params_;
    const ParamsContracted paramsContracted_;
    const FreeLayout freeLayout_;

    template<int a, int b>
    struct Min
    {
        static const int value = a < b ? a : b;
    };

    static const int kNumThreadsK = isContractedContiguous ? 
        Min<ThreadMap::kThreads,
        ThreadMap::Shape::kContiguous / (kElementsPerAccessContracted_*kNumIterationsContracted_)>::value :
        ThreadMap::kThreads / (Min<ThreadMap::kThreads,
        ThreadMap::Shape::kContiguous / (kElementsPerAccessFree_*kNumIterationsFree_)>::value);
    static const int kNumThreadsM = ThreadMap::kThreads / kNumThreadsK;

#if defined(_MSC_VER) && _MSC_VER < 1930
#pragma warning(push)
#pragma warning(disable:4348)
#endif

    template<int kDeltaContracted, int kNumIterationsContracted, int kNumThreads, typename Shape, int level, int end, typename enable = void> struct IterationSpace;

#if defined(_MSC_VER) && _MSC_VER < 1930
#pragma warning(pop)
#endif

//    // this assert ensures that the delta for all blocked k-dim --except for the first one-- is equal to one.
//    static_assert(kDeltaContracted <= At<0, ShapeK>::value && At<0, ShapeK>::value % kDeltaContracted == 0, "K-blocking is invalid to due constraints w.r.t. Delta");


//def f(delta, numIterations, numThreadsK, blocking, l):
//    if (l >= len(blocking)): return []
//    b = blocking[l]
//    numThreadsThis = min(b,delta) if (numIterations * delta >= b) else int(b/(numIterations))
//    numThreadsNext = numThreadsK / numThreadsThis
//    deltaThis = delta
//    deltaNext = int(ceil(delta/b))
//    numIterThis = 1 if (b < numThreadsThis) else b / numThreadsThis
//    if( b % numIterThis != 0):
//        print("error:", b, numThreadsThis)
//        exit(-1)
//    numIterNext = numIterations / numIterThis
//    print("%d: %d %d"%(l, deltaThis, numIterThis))
//    return [(deltaThis, numIterThis)] + f(deltaNext, numIterNext, numThreadsNext, blocking, l+1)


    template<int kDeltaContracted, int kNumIterationsContracted, int kNumThreads, typename Shape, int level, int end>
    struct IterationSpace<kDeltaContracted, kNumIterationsContracted, kNumThreads, Shape, level, end, typename platform::enable_if<level != Shape::kRank>::type>
    {
        static const int kBlockingThis = At<level, Shape>::value;

        static_assert(kBlockingThis % kNumThreads == 0 || 
                      kNumThreads % kBlockingThis  == 0, "Invalid blocking");

        static const int kElementsPerThread = (level == 0 && isContractedContiguous) ? kContiguousElementsPerThread_ : 1;
//        static const int kDelta = (kNumThreads * kElementsPerThread >= kBlockingThis) ? kBlockingThis : ((level == 0) ? kDeltaContracted : 1);//kNumThreads * kElementsPerThread;
//        static const int kIterations = (kNumThreads * kElementsPerThread >= kBlockingThis) ? 1 : kBlockingThis / (kNumThreads * kElementsPerThread);
//
//        static const int kThreadsRemaining = (kNumThreads * kElementsPerThread >= kBlockingThis) ? (kNumThreads * kElementsPerThread) / kBlockingThis : 1;
        static const int kThreadsThis = kElementsPerThread * kNumIterationsContracted * kDeltaContracted >= kBlockingThis ? Min<kBlockingThis/kElementsPerThread, kDeltaContracted>::value : kBlockingThis / (kNumIterationsContracted*kElementsPerThread);
        static_assert(kNumThreads % kThreadsThis == 0, "threads invalid");
        static const int kThreadsNext = kNumThreads/ kThreadsThis;
        static const int kDelta = kDeltaContracted;
        static const int kDeltaNext = (kDeltaContracted + kBlockingThis - 1) / kBlockingThis;
        static const int kIterations = (kBlockingThis < kThreadsThis) ? 1 : kBlockingThis / (kThreadsThis * kElementsPerThread);
        static_assert(kNumIterationsContracted % kIterations == 0, "iterations invalid");
        static const int kIterationsNext = kNumIterationsContracted / kIterations;

        using Next = IterationSpace<kDeltaNext, kIterationsNext, kThreadsNext, Shape, level + 1, end>;
        using Iterations = typename Prepend<Int<kIterations>, typename Next::Iterations>::Type; // in units of AccessType
        using Delta = typename Prepend<Int<kDelta>, typename Next::Delta>::Type; // in units of Element
    };

    template<int kDeltaContracted, int kNumIterationsContracted, int kNumThreads, typename Shape, int level>
    struct IterationSpace<kDeltaContracted, kNumIterationsContracted, kNumThreads, Shape, level, level, typename platform::enable_if<level == Shape::kRank>::type>
    {
        using Iterations = IntTuple<>;
        using Delta = IntTuple<>;
    };

    using IterationSpaceContracted = IterationSpace<kDeltaContracted_, kNumIterationsContracted_, kNumThreadsK, ShapeK, 0, ShapeK::kRank>;
    static_assert(Product<typename IterationSpaceContracted::Iterations>::value == (!isContractedContiguous ?  // TODO is this correct?
                ThreadMap::Iterations::kStrided : ThreadMap::Iterations::kContiguous), "Number of iterations does not match");

    // notice: we don't need predicates for contracted dimension that are not blocked (they'd be true all the time, except for the last iteration)
    uint32_t predicates_ = 0; // bits 0 till (kNumIterationsFree_-1) represent the predicates in m-direction,
                              // bits kNumIterationsFree_ till (kNumIterationsFree_ + At<0, typename IterationSpaceContracted::Iterations>::value -1) represent the predicates in the first blocked contracted dimension
                              // bits (kNumIterationsFree_ + At<0, typename IterationSpaceContracted::Iterations>::value) till (kNumIterationsFree_ + At<0, typename IterationSpaceContracted::Iterations>::value + At<1, typename IterationSpaceContracted::Iterations>::value) represent the predicates in the first blocked contracted dimension
    static_assert(sizeof(predicates_) * 8 >= kNumIterationsFree_ * kAccessesPerVectorFree + kNumIterationsContracted_ * kAccessesPerVectorContracted, "This kernel uses too many predicates.");

    CUTLASS_DEVICE
    MyIterator(
            ParamsBase const &params,
            ParamsContracted const &paramsContracted,
            FreeLayout const &freeLayout,
            Element const *ptr,
            int thread_idx,
            MatrixCoord const &threadblock_offset = MatrixCoord()
            ) :
        params_(params),
        paramsContracted_(paramsContracted),
        freeLayout_(freeLayout)
    {
        // linearized offset along the free modes for the active thread
        const int32_t threadOffsetFree = threadblock_offset.column() +
            (isContractedContiguous ? ThreadMap::initial_offset(thread_idx).strided() :
                                      ThreadMap::initial_offset(thread_idx).contiguous() );

        // logical offset into contracted-dimension
        auto threadOffsetContracted = threadblock_offset.row() +
            (isContractedContiguous ? ThreadMap::initial_offset(thread_idx).contiguous() :
                                      ThreadMap::initial_offset(thread_idx).strided());

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < kNumBlockedModesContracted; ++i) // TODO change to compile-time loop
        {
            coordContracted_[i] = threadOffsetContracted % blockingContracted[i];
            resetCoordContracted_[i] = coordContracted_[i];
            threadOffsetContracted /= blockingContracted[i];
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i=kNumBlockedModesContracted; i < kNumModesContracted; ++i)
        {
            coordContracted_[i] = 0;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kNumModesContracted; ++i)
        {
            int32_t offset;
            paramsContracted_.divmodContracted_[i](threadOffsetContracted, offset, threadOffsetContracted);
            coordContracted_[i] += offset * blockingContracted[i];
        }

        /*
         * Offset pointers
         */
        int64_t offsetK = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < kNumModesContracted; ++i)
        {
            offsetK += coordContracted_[i] * params_.stridesContracted_[i];
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < kNumIterationsFree_; ++i)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kAccessesPerVectorFree; ++v)
            {
                auto idxFree = threadOffsetFree + i * kDeltaFree + v * kElementsPerAccess_;
                bool validM = false;
                auto coord = freeLayout_.get_coord(idxFree, validM);
                int64_t offsetFree = dot(coord, params_.stridesFree_);
                predicates_ |= (((validM) ? 1UL : 0UL) << (i * kAccessesPerVectorFree + v));
                pointer_[i * kAccessesPerVectorFree + v] = ptr + (offsetFree + offsetK);
            }
        }

        /*
         * Compute predicates along the contracted modes
         */
        updatePredicatesContracted<0>();
    }

  /// Advances to the next tile in memory.
  CUTLASS_DEVICE
  MyIterator &operator++() {
    this->advance();
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_DEVICE
  MyIterator operator++(int) {
    MyIterator self(*this);
    operator++();
    return self;
  }


  template<int level>
  CUTLASS_HOST_DEVICE
  void updatePredicatesContracted()
  {
      updatePredicatesContracted<level>(platform::integral_constant<bool, level != kNumBlockedModesContracted>{});
  }

  template<int level>
  CUTLASS_HOST_DEVICE
  void updatePredicatesContracted(platform::true_type)
  {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < At<level, typename IterationSpaceContracted::Iterations>::value; ++k)
      {
          if (level == 0 && isContractedContiguous)
          {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < kAccessesPerVector; ++v) // belongs to k-idx
              {
                  bool pred = ((coordContracted_[level] + v * kElementsPerAccess_ + k * At<level, typename IterationSpaceContracted::Delta>::value) < paramsContracted_.extentsContracted_[level]);
                  if(pred)
                      predicates_ = predicates_ | ((1U) << (v + k * kAccessesPerVector + getPredicateOffset<level>()));
//                  predicates_ |= ((pred ? 1U : 0U) << (v + k * kAccessesPerVector + getPredicateOffset<level>()));
              }
          } else {
              bool pred = ((coordContracted_[level] + k * At<level, typename IterationSpaceContracted::Delta>::value) < paramsContracted_.extentsContracted_[level]);
              if(pred)
                  predicates_ = predicates_ | ((1U) << (k + getPredicateOffset<level>()));
//              predicates_ |= (pred ? 1U : 0U) << (k + getPredicateOffset<level>());
          }
      }

      updatePredicatesContracted<level + 1>();
  }

  template<int level>
  CUTLASS_HOST_DEVICE
  void updatePredicatesContracted(platform::false_type) {}

  /// \param[in] i ith k-mode (i == 0 denotes the first (i.e., fastest-varying) k-mode)
  template<int i>
  CUTLASS_HOST_DEVICE
  int getPredicateOffset()
  {
      return getPredicateOffset<i>(platform::integral_constant<bool, i!=0>{});
  }

  template<int i>
  CUTLASS_HOST_DEVICE
  int getPredicateOffset(platform::true_type)
  {
      if (isContractedContiguous)
          return ProductRange<i-1, typename IterationSpaceContracted::Iterations, kAccessesPerVector>::value + getPredicateOffset<i-1>();
      else
          return ProductRange<i-1, typename IterationSpaceContracted::Iterations>::value + getPredicateOffset<i-1>();
  }

  template<int i>
  CUTLASS_HOST_DEVICE
  int getPredicateOffset(platform::false_type)
  {
      if (isContractedContiguous)
      {
          return kNumIterationsFree_;
      } else {
          return kNumIterationsFree_ * kAccessesPerVector;
      }
  }

  /**
   * Loop order for the case where isContractedContiguous == false
   */
  template<int i, int end>
  CUTLASS_DEVICE
  void loopKstrided(AccessType* frag_ptr, const int32_t idxK, const int64_t offset, const bool pred)
  {
      loopKstrided<i, end>(frag_ptr, idxK, offset, pred, platform::integral_constant<bool, i != end>{});
  }

  /**
   * idxK linearized offset in k-direction
   */
  template<int i, int end>
  CUTLASS_DEVICE
  void loopKstrided(AccessType* frag_ptr, const int32_t idxK, const int64_t offset, const bool pred, platform::true_type)
  {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < At<i, typename IterationSpaceContracted::Iterations>::value; ++k)
      {
          const int64_t offsetK = k * At<i, typename IterationSpaceContracted::Delta>::value * params_.stridesContracted_[i];
          int32_t stride = ProductRange<i-1, typename IterationSpaceContracted::Iterations>::value;
          const bool predK = i < kNumBlockedModesContracted ? (predicates_ & (1u << (k + getPredicateOffset<i>()))) : true;
          loopKstrided<i-1, end>(frag_ptr, idxK + k * stride,
                         offset + offsetK, pred && predK);
      }
  }

  template<int i, int end>
  CUTLASS_DEVICE
  void loopKstrided(AccessType* frag_ptr, const int32_t idxK, const int64_t offsetK, const bool pred, platform::false_type)
  {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < kNumIterationsFree_; ++m)
      {
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kAccessesPerVector; ++v) // belongs to m-idx
          {
              cutlass::arch::global_load< AccessType,
                                          sizeof(AccessType)
                                        >(
                          frag_ptr[v + (m + idxK * kNumIterationsFree_) * kAccessesPerVector ],
                          pointer_[m * kAccessesPerVector + v] + offsetK,
                          pred && (predicates_ & (1u << (v + m * kAccessesPerVector))));
          }
      }
  }

  /**
   * Loop order for the case where isContractedContiguous == false
   */
  template<int i, int end>
  CUTLASS_DEVICE
  void loopKcontiguous(AccessType *frag_ptr, const Element *ptr, const int32_t idxK, const int64_t offset, const bool pred)
  {
      loopKcontiguous<i, end>(frag_ptr, ptr, idxK, offset, pred, platform::integral_constant<bool, i != end>{});
  }

  template<int i, int end>
  CUTLASS_DEVICE
  void loopKcontiguous(AccessType *frag_ptr, const Element *ptr, const int32_t idxK, const int64_t offset, const bool pred, platform::true_type)
  {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < At<i, typename IterationSpaceContracted::Iterations>::value; ++k)
      {
          if (isContractedContiguous && i == 0)
          {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < kAccessesPerVector; ++v) // belongs to k-idx
              {
                  const bool predK = i < kNumBlockedModesContracted ? (predicates_ & (1u << (v + k * kAccessesPerVector + getPredicateOffset<i>()))) : true;
                  const int64_t offsetK = (k * At<i, typename IterationSpaceContracted::Delta>::value + v * kElementsPerAccess_) * params_.stridesContracted_[i];
                  int32_t stride = ProductRange<i-1, typename IterationSpaceContracted::Iterations>::value * kAccessesPerVector;
                  loopKcontiguous<i-1, end>(frag_ptr, ptr, idxK + k * stride + v,
                                 offset + offsetK, pred && predK);
              }
          }
          else
          {
              const bool predK = i < kNumBlockedModesContracted ? (predicates_ & (1u << (k + getPredicateOffset<i>()))) : true;
              const int64_t offsetK = k * At<i, typename IterationSpaceContracted::Delta>::value * params_.stridesContracted_[i];
              int32_t stride = ProductRange<i-1, typename IterationSpaceContracted::Iterations>::value;
              loopKcontiguous<i-1, end>(frag_ptr, ptr, idxK + k * stride,
                             offset + offsetK, pred && predK);
          }
      }
  }

  template<int i, int end>
  CUTLASS_DEVICE
  void loopKcontiguous(AccessType *frag_ptr, const Element *ptr, const int32_t idxK, const int64_t offset, const bool pred, platform::false_type)
  {
      cutlass::arch::global_load< AccessType,
                                  sizeof(AccessType)
                                >(
                  frag_ptr[idxK],
                  ptr + offset,
                  pred);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    frag.clear();

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    if (!isContractedContiguous)
    {
        loopKstrided<kNumModesContracted-1, -1>(frag_ptr, 0, 0, true);
    }
    else
    {
        constexpr int32_t kStride = Product<typename IterationSpaceContracted::Iterations>::value * kAccessesPerVector;
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kNumIterationsFree_; ++m)
        {
            loopKcontiguous<kNumModesContracted-1, -1>(frag_ptr + m * kStride, pointer_[m], 0, 0, (predicates_ & (1u << m)));
        }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
//    tile_access_iterator_.set_iteration_index(0);
    load_with_pointer_offset(frag, 0);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i < kNumIterationsFree_; ++i){
          CUTLASS_PRAGMA_UNROLL
          for(int v=0; v < kAccessesPerVectorFree; ++v){
              pointer_[i * kAccessesPerVectorFree + v] += pointer_offset;
          }
      }
  }

  /**
   * Advances the iterator to the next position along the contracted dimensions (i.e.,
   * advance along logical k-dimension by a total of blockingContracted[0] * blockingContracted[1] * ... )
   */
  CUTLASS_DEVICE
  void advance() {
      advanceMode<0, kNumModesContracted>();
  }

  template<int32_t modeIdx, int32_t numModes>
  CUTLASS_DEVICE
  void advanceMode()
  {
      advanceMode<modeIdx, numModes>(platform::integral_constant<bool, modeIdx != numModes>{});
  }

  template<int32_t modeIdx, int32_t numModes>
  CUTLASS_DEVICE
  void advanceMode(platform::true_type)
  {
      // update Coordinate
      coordContracted_[modeIdx] += blockingContracted[modeIdx];

      // update pointers
      CUTLASS_PRAGMA_UNROLL
      for(int i=0; i < kNumIterationsFree_; ++i)
      {
          CUTLASS_PRAGMA_UNROLL
          for(int v=0; v < kAccessesPerVectorFree; ++v)
          {
              pointer_[i * kAccessesPerVectorFree + v] += params_.incContracted_[modeIdx];
          }
      }

      const int32_t reset = (modeIdx < kNumBlockedModesContracted) ? resetCoordContracted_[modeIdx] : 0;

      if (coordContracted_[modeIdx] >= paramsContracted_.extentsContracted_[modeIdx] + reset)
      {
          // reset coordinate for this mode
          coordContracted_[modeIdx] = reset;
          
          // we've reached the end of this mode => increment next
          advanceMode<modeIdx + 1, numModes>();
      }

      // update predicates (in k-dim only)
      // we only have to update the predicates if we reached the last block along this mode
//          if (coordContracted[modeIdx] + blockingContracted[modeIdx] >= extentsContracted[modeIdx]) // or we just reset
//          {
      if (modeIdx < kNumBlockedModesContracted)
      {
          CUTLASS_PRAGMA_UNROLL
          for(int i=0; i < At<modeIdx, typename IterationSpaceContracted::Iterations>::value; ++i)
          {
              if (isContractedContiguous && modeIdx == 0)
              {
                  CUTLASS_PRAGMA_UNROLL
                  for (int v = 0; v < kAccessesPerVector; ++v) // belongs to k-idx
                  {
                      const int predicateOffset = i * kAccessesPerVector + getPredicateOffset<modeIdx>() + v;
                      predicates_ &= (~((1UL < paramsContracted_.extentsContracted_[modeIdx]) << (predicateOffset))); // reset bit
                      predicates_ |= ((coordContracted_[modeIdx] + v * kElementsPerAccess_ + i * At<modeIdx, typename IterationSpaceContracted::Delta>::value) < paramsContracted_.extentsContracted_[modeIdx]) << (predicateOffset);
                  }
              }
              else
              {
                  const int predicateOffset = i + getPredicateOffset<modeIdx>();
                  predicates_ &= (~((1UL < paramsContracted_.extentsContracted_[modeIdx]) << (predicateOffset))); // reset bit
                  predicates_ |= ((coordContracted_[modeIdx] + i * At<modeIdx, typename IterationSpaceContracted::Delta>::value) < paramsContracted_.extentsContracted_[modeIdx]) << (predicateOffset);
              }
          }
      }
  }

  template<int32_t modeIdx, int32_t numModes>
  CUTLASS_DEVICE
  void advanceMode(platform::false_type)
  {
      predicates_ = 0; // we've reached the end of the contracted dim
  }


};





/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


