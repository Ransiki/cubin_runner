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

// {$nv-internal-release file}

/*! \file
  \brief Visitor tree load operations for the gemv epilogue
*/


#pragma once
#include "cutlass/epilogue/threadblock/fusion/visitor_2x_gemv.hpp"
#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;

using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Accumulator Reorder Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// returns accumulator
template<
  class ThreadShape_,          // The CTA threadshape (concept: GemmShape)
  class ElementAccumulator_    // Accumulator data type
>
struct VisitorAccReorderGemv : public VisitorNodeBaseGemv<ThreadShape_>{
public:
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using ThreadShape = ThreadShape_;
  using ElementAccumulator = ElementAccumulator_;

  using CtaTileMN = decltype(make_tile(Layout<Int<ThreadShape::kM>>{}, Layout<_1>{}));

  // coord -> tid
  using ThreadMapR2S = Layout<
    Shape<Int<ThreadShape::kM>, Int<ThreadShape::kN>>,
    Stride<Int<ThreadShape::kN>, _1>
  >;

  // coord -> tid
  using ThreadMapS2R = typename Base::ThreadMap;

  // The shared storage buffer used to reorder accumulator between threads
  using SharedStorageShape = Shape<Int<ThreadShape::kM>, _1>;
  

  struct SharedStorage {
    AlignedArray<ElementAccumulator, size(SharedStorageShape{}), 16> reorder_buffer;
  };

  struct Arguments {};
  using Params = Arguments;

  // Constructor
  CUTLASS_HOST_DEVICE
  VisitorAccReorderGemv(Params const& params, SharedStorage const& shared_storage)
    : smem_buffer_ptr(const_cast<ElementAccumulator*>(shared_storage.reorder_buffer.data())) { }

private:
  ElementAccumulator* smem_buffer_ptr;

public:
  template<
    class STensorR2S, class CTensorR2S,
    class STensorS2R, class CTensorS2R>
  struct Callbacks : EmptyCallbacksGemv {
  
  public:
    CUTLASS_DEVICE
    Callbacks(
      STensorR2S&& tRS_sB,
      CTensorR2S&& tRS_cB,
      STensorS2R&& tSR_sB,
      CTensorS2R&& tSR_cB,
      int thread_idx
    ):
      tRS_sB_(cute::forward<STensorR2S>(tRS_sB)),
      tRS_cB_(cute::forward<CTensorR2S>(tRS_cB)),
      tSR_sB_(cute::forward<STensorS2R>(tSR_sB)),
      tSR_cB_(cute::forward<CTensorS2R>(tSR_cB)),
      thread_idx_(thread_idx) {}

    CUTLASS_DEVICE ElementAccumulator
    visit(int idx_batch, int idx_row_m, ElementAccumulator accum) {

      if (elem_less(tRS_cB_(thread_idx_), SharedStorageShape{})) {
        tRS_sB_(thread_idx_) = accum;
      }

      __syncthreads();

      if (elem_less(tSR_cB_(thread_idx_), SharedStorageShape{})) {
        return tSR_sB_(thread_idx_);
      }
      return ElementAccumulator(0);
    }

    // Check whether the thread is active disregard of the problem size
    CUTLASS_DEVICE
    bool is_active() {
      return thread_idx_ < size<0>(CtaTileMN{});
    }
    
  private:
    STensorR2S tRS_sB_;
    CTensorR2S tRS_cB_;
    STensorS2R tSR_sB_;
    CTensorS2R tSR_cB_;
    int thread_idx_;
  };

  template<class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    Tensor mB = make_tensor(
      make_smem_ptr(smem_buffer_ptr), SharedStorageShape{}
    );
    Tensor cB = make_identity_tensor(mB.shape());

    // R->S
    Tensor tRS_sB = mB.compose(left_inverse(ThreadMapR2S{}));    // (TID)
    Tensor tRS_cB = cB.compose(left_inverse(ThreadMapR2S{}));    // (TID)
    // S->R
    Tensor tSR_sB = mB.compose(left_inverse(ThreadMapS2R{}));    // (TID)
    Tensor tSR_cB = cB.compose(left_inverse(ThreadMapS2R{}));    // (TID)

    return Callbacks<
      decltype(tRS_sB), decltype(tRS_cB), 
      decltype(tSR_sB), decltype(tSR_cB)
    >{cute::move(tRS_sB), cute::move(tRS_cB), cute::move(tSR_sB), cute::move(tSR_cB), thread_idx};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Accumulator Fetch Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

struct VisitorAccFetchGemv {
public:
  struct SharedStorage {};
  struct Arguments {};
  using Params = Arguments;

  // Constructor
  CUTLASS_HOST_DEVICE
  VisitorAccFetchGemv(){ }
  CUTLASS_HOST_DEVICE
  VisitorAccFetchGemv(Params const& params, SharedStorage const& shared_storage){ }

  struct Callbacks : EmptyCallbacksGemv {
  
  public:
    CUTLASS_DEVICE
    Callbacks() {}

    template <class ElementAccumulator>
    CUTLASS_DEVICE ElementAccumulator
    visit(int idx_batch, int idx_row_m, ElementAccumulator accum) {
      return accum;
    }
  };

  template<class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    return Callbacks{};
  }

private:
  // Size of the visitor node cannot be 0
  int8_t _x;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadShape_,
  class Element_,
  class StrideMNL_=Stride<_1,_0,int64_t>,
  FloatRoundStyle RoundStyle=FloatRoundStyle::round_to_nearest>
struct VisitorAuxLoadGemv : public VisitorNodeBaseGemv<ThreadShape_> {
public:
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using ThreadShape = ThreadShape_;
  using Element = Element_;
  using StrideMNL = StrideMNL_;

  // coord -> tid
  using ThreadMapG2R = typename Base::ThreadMap;
  // The tile accessed by the same CTA
  using CtaTileMN = typename Base::CtaTileMN;

  struct SharedStorage{};

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  using Params = Arguments;

  CUTLASS_HOST_DEVICE
  VisitorAuxLoadGemv(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }
  
private:
  Params const* params_ptr;

public:
  template<
    class GTensorG2R,
    class ProblemShape>
  struct Callbacks : public CallbacksImplGemv<ProblemShape, CtaTileMN> {
    using BaseCallbacks = CallbacksImplGemv<ProblemShape, CtaTileMN>;
    CUTLASS_DEVICE
    Callbacks(
      GTensorG2R&& tC_gAux,
      ProblemShape problem_shape,
      int thread_idx
    ):
      tC_gAux_(cute::forward<GTensorG2R>(tC_gAux)),
      BaseCallbacks(problem_shape, thread_idx)
    {}

    GTensorG2R tC_gAux_;

    template <class ElementAccumulator>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m, ElementAccumulator const& accum) {      
      if (!this->predicate(idx_row_m, idx_batch)) return Element(0);
      // Load aux
      return tC_gAux_(idx_row_m, idx_batch);
    };
  };

  template<class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    Tensor mAux = make_tensor(
      make_gmem_ptr(this->make_iterator(params_ptr->ptr_aux)),
      problem_shape,
      params_ptr->dAux
    );

    Tensor tC_gAux = mAux.tile(CtaTileMN{}).compose(
      left_inverse(ThreadMapG2R{}),_,_,_)(thread_idx,_,_0{},_);  // (STEP_M, BATCH)

    return Callbacks<
      decltype(tC_gAux),
      ProblemShape
      >{cute::move(tC_gAux), problem_shape, thread_idx};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Scaling Factor Load Operations 
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadShape_,
  class ElementSF_,
  class LayoutTagSF_>
struct VisitorAuxLoadSFGemv : public VisitorNodeBaseGemv<ThreadShape_> {
public:
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using ThreadShape = ThreadShape_;
  using ElementSF = ElementSF_;
  using LayoutTagSF = LayoutTagSF_;

  // coord -> tid
  using ThreadMapG2R = typename Base::ThreadMap;
  // The tile accessed by the same CTA
  using CtaTileMN = typename Base::CtaTileMN;

  // Get the Layout of the SFD
  static constexpr int kVectorSize = ThreadShape::kM;
  static constexpr bool kIsKMajorSFD = is_same_v<LayoutTagSF, cutlass::layout::RowMajor>;
  using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<
    kVectorSize
    , kIsKMajorSFD ? cute::UMMA::Major::K : cute::UMMA::Major::MN // {$nv-internal-release}
  >;

  struct SharedStorage{};

  struct Arguments {
    ElementSF* ptr_sf = nullptr;
  };

  using Params = Arguments;


  CUTLASS_HOST_DEVICE
  VisitorAuxLoadSFGemv(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }
  
private:
  Params const* params_ptr;

public:
  template<class GTensorSFG2R, class ProblemShape>
  struct Callbacks : public CallbacksImplGemv<ProblemShape, CtaTileMN> {
    using BaseCallbacks = CallbacksImplGemv<ProblemShape, CtaTileMN>;

    CUTLASS_DEVICE
    Callbacks(
      GTensorSFG2R&& tC_gSF,
      ProblemShape problem_shape,
      int thread_idx
    ):
      tC_gSF_(cute::forward<GTensorSFG2R>(tC_gSF)),
      BaseCallbacks(problem_shape, thread_idx) { }

    GTensorSFG2R tC_gSF_;

    template <class ElementAccumulator>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m, ElementAccumulator const& accum) {
      if (!this->predicate(idx_row_m, idx_batch)) return ElementSF(0);

      return tC_gSF_(idx_row_m, idx_batch);
    };
  };

  template<class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    // SF Tensor
    auto sf_layout = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(
      insert<2>(problem_shape, _1{})
    );

    Tensor mSF = make_tensor(
      make_gmem_ptr(params_ptr->ptr_sf),
      sf_layout
    );

    auto tile_m = make_tile(get<0>(CtaTileMN{}));

    Tensor tC_gSF = mSF.tile(tile_m)(_0{},_,_0{},_);

    return Callbacks<
      decltype(tC_gSF) , ProblemShape
      >{cute::move(tC_gSF), problem_shape, thread_idx};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Broadcast Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar broadcast
template<
  class ThreadShape_,
  class Element_,
  class StrideMNL_ = Stride<_0,_0,_0>
>
struct VisitorScalarBroadcastGemv : public VisitorNodeBaseGemv<ThreadShape_> {
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using Element = Element_;
  using StrideMNL = StrideMNL_;
  using CtaTileMN = typename Base::CtaTileMN;

  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_0,_0>>) || // scalar broadcast, e.g. alpha
    (cute::is_same_v<StrideMNL, Stride<_0,_0,_1>>) ||
    (cute::is_same_v<StrideMNL, Stride<_0,_0,int>>));  // batched scalar broadcast, e.g. per-batch alpha
  
  struct SharedStorage { };

  struct Arguments {
    Element scalars = {};
    Element const* scalar_ptrs = {};
    StrideMNL dScalar = {};
  };

  using Params = Arguments;
  
  CUTLASS_HOST_DEVICE
  VisitorScalarBroadcastGemv(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Element scalar;
  Params const* params_ptr;

  struct Callbacks {
  public:
    // Constructors
    CUTLASS_DEVICE
    Callbacks(){}

    CUTLASS_DEVICE
    Callbacks(int thread_idx, const Params* params_ptr)
      : _thread_idx(thread_idx), _params_ptr(params_ptr) {
      if (_thread_idx < size<0>(CtaTileMN{})) {
        if constexpr (cute::is_same_v<StrideMNL, Stride<_0,_0,_0>>) {
          update_scalar();
        }
      }
    }
    
    int _thread_idx;
    Element _scalar;
    const Params* _params_ptr;

    CUTLASS_DEVICE void
    begin_epilogue(int idx_batch) {
      if (_thread_idx < size<0>(CtaTileMN{})) {
        if constexpr (
          cute::is_same_v<StrideMNL, Stride<_0,_0,_1>> ||
          cute::is_same_v<StrideMNL, Stride<_0,_0,int>>) {
          update_scalar(idx_batch);
        }
      }
    }

    template <typename ElementAccumulator>
    CUTLASS_DEVICE Element
    visit(int idx_batch, int idx_row_m, ElementAccumulator accum) {
      return _scalar;
    }

    CUTLASS_DEVICE void
    end_epilogue(int idx_batch) {}

  private:
    CUTLASS_DEVICE void
    update_scalar(int l_coord = 0) {
      int l_offset = l_coord * size<2>(_params_ptr->dScalar);

      if (_params_ptr->scalar_ptrs != nullptr) {
        _scalar = _params_ptr->scalar_ptrs[l_offset];
      } else {
        // batch stride is ignored for nullptr fallback
        _scalar = _params_ptr->scalars;
      }
    }
  }; 

  template<class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    return Callbacks{thread_idx, params_ptr};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////