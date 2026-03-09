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
  \brief Visitor tree store operations for the CUTLASS 2x epilogue gemv
*/

#pragma once

#include "cutlass/epilogue/threadblock/fusion/visitor_2x_gemv.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadShape_,
  class Element_,
  class StrideMNL_=Stride<_1,_0,int64_t>,
  FloatRoundStyle RoundStyle=FloatRoundStyle::round_to_nearest>
struct VisitorAuxStoreGemv : public VisitorNodeBaseGemv<ThreadShape_>{
public:
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using ThreadShape = ThreadShape_;
  using Element = Element_;
  using StrideMNL = StrideMNL_;

  static_assert(!cutlass::is_subbyte<Element>::value, "AuxStoreGemv does not support subbyte types.");

  // coord -> tid
  using ThreadMapR2G = typename Base::ThreadMap;
  // The tile accessed by the same CTA
  using CtaTileMN = typename Base::CtaTileMN;

  // No shared storage is required
  struct SharedStorage{};

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  using Params = Arguments;

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreGemv(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

private:
  Params const* params_ptr;

public:
  template<class GTensorG2R, class ProblemShape>
  struct Callbacks : public CallbacksImplGemv<ProblemShape, CtaTileMN> {
    using BaseCallbacks = CallbacksImplGemv<ProblemShape, CtaTileMN>;

    CUTLASS_DEVICE
    Callbacks(
      GTensorG2R&& tC_gAux,
      ProblemShape problem_shape,
      int thread_idx
    ):
      tC_gAux_(cute::forward<GTensorG2R>(tC_gAux)),
      BaseCallbacks(problem_shape, thread_idx){ }

    GTensorG2R tC_gAux_;

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m, ElementAccumulator const& frg_acc, ElementInput const& frg_input) {
      Element data = NumericConverter<Element, ElementInput, RoundStyle>{}(frg_input);
      if (this->predicate(idx_row_m, idx_batch)) {
        tC_gAux_(idx_row_m, idx_batch) = data;
      }
      return frg_input;
    }
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
      left_inverse(ThreadMapR2G{}),_,_,_)(thread_idx,_,_0{},_);  // (STEP_M, BATCH)

    return Callbacks<decltype(tC_gAux), ProblemShape>{
      cute::move(tC_gAux), problem_shape, thread_idx};
  }
};

template<
  class ThreadShape_,
  class Element_,
  class ElementSF_,
  class LayoutTagSF_,
  class StrideMNL_=Stride<_1,_0,int64_t>,
  FloatRoundStyle RoundStyle=FloatRoundStyle::round_to_nearest>
struct VisitorAuxStoreBlockScaledGemv : public VisitorNodeBaseGemv<ThreadShape_>{
public:
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using ThreadShape = ThreadShape_;
  using Element = Element_;
  using ElementSF = ElementSF_;
  using LayoutTagSF = LayoutTagSF_;
  using StrideMNL = StrideMNL_;

  // Static assert to ensure that intra-warp reduction is available
  static_assert(ThreadShape::kM <= 32, "ThreadShape::kM should be <= 32");

  // coord -> tid
  using ThreadMapR2G = typename Base::ThreadMap;
  // The tile accessed by the same CTA
  using CtaTileMN = typename Base::CtaTileMN;

  // Get the Layout of the SFD
  static constexpr int kVectorSize = ThreadShape::kM;
  static constexpr bool kIsKMajorSFD = is_same_v<LayoutTagSF, cutlass::layout::RowMajor>;
  using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<
    kVectorSize
    , kIsKMajorSFD ? cute::UMMA::Major::K : cute::UMMA::Major::MN  // {$nv-internal-release}
  >;

  // No shared storage is required
  struct SharedStorage{};

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};
    ElementSF* ptr_sf = nullptr;
    float st{0};
  };

  using Params = Arguments;

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreBlockScaledGemv(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

private:
  Params const* params_ptr;

public:
  template<class GTensorR2G, class GTensorSFR2G, class ProblemShape>
  struct Callbacks : public CallbacksImplGemv<ProblemShape, CtaTileMN> {
    using BaseCallbacks = CallbacksImplGemv<ProblemShape, CtaTileMN>;

    CUTLASS_DEVICE
    Callbacks(
      float st,
      GTensorR2G&& tC_gAux,
      GTensorSFR2G&& tC_gSF,
      ProblemShape problem_shape,
      int thread_idx
    ):
      st(st),
      tC_gAux_(cute::forward<GTensorR2G>(tC_gAux)),
      tC_gSF_(cute::forward<GTensorSFR2G>(tC_gSF)),
      BaseCallbacks(problem_shape, thread_idx)
    {
      float fp_subtype_max = static_cast<float>(cutlass::platform::numeric_limits<cutlass::float_e2m1_t>::max());
      st_scale_down = st / fp_subtype_max;
    }

    float st;
    float st_scale_down;
    GTensorR2G tC_gAux_;
    GTensorSFR2G tC_gSF_;

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m, ElementAccumulator const& frg_acc, ElementInput const& frg_input) {

      ElementAccumulator data = NumericConverter<ElementAccumulator, ElementInput, RoundStyle>{}(frg_input);
      if (!this->predicate(idx_row_m, idx_batch)) {
        // Set the data to min(ElementAccumulator)
        data = ElementAccumulator(0);
      }
      ElementAccumulator max_data = this->reduce<ThreadShape::kM>(data);

      // FP32 pvscale = normalize_scale(accum_max, St')
      const float pvscale = max_data * st_scale_down;

      // e4m3
      // FP8 Qpvscale = FP32_to_Scale_Format(pvscale)
      const ElementSF qpvscale = static_cast<ElementSF>(pvscale);

      // FP32 Qpvscale_Up = Scale_Format_to_FP32(QPvscale)
      const float qpvscale_up = static_cast<float>(qpvscale);

      // FP32 Qpvscale_RCP = 1/Qpvscale_Up
      const float qpvscale_rcp = __frcp_rn(qpvscale_up);

      // FP32 val[i] = accum[i] * St * Qpvscale_RCP
      const float output_fp32 = data * st * qpvscale_rcp; // * qpvscale_rcp;

      // Write output D
      // Source of bug: there is a race condition when using subbyte_iterator
      // CuTe's version is unsafe if multiple threads write to elements in the same byte
      if constexpr (cutlass::is_subbyte<Element>::value) {
        const float output_fp32_2 = __shfl_down_sync(__activemask(), output_fp32, 1);
        if (this->_thread_idx % 2 == 0) {

          Tensor tC_gAux_pack = filter(tC_gAux_(_,idx_row_m, idx_batch)).tile(make_tile(Layout<_2>{}))(_, this->_thread_idx / 2);
          Tensor tC_rAux_pack = make_tensor_like(tC_gAux_pack);
        
          tC_rAux_pack(0) = static_cast<Element>(output_fp32);
          tC_rAux_pack(1) = static_cast<Element>(output_fp32_2);

          if (this->predicate(idx_row_m, idx_batch, 1)) {
            copy(AutoVectorizingCopyWithAssumedAlignment<8>{}, tC_rAux_pack, tC_gAux_pack);
          } else if (this->predicate(idx_row_m, idx_batch)) {
            tC_gAux_pack[0] = tC_rAux_pack[0];
          }
        }
      } else {
        Element qval_i = static_cast<Element>(output_fp32);
        if (this->predicate(idx_row_m, idx_batch)) {
          tC_gAux_(this->_thread_idx+1, idx_row_m, idx_batch) = qval_i;
        }
      }
      

      if (this->predicate(idx_row_m, idx_batch)) {
        // Only the first thread writes the SF
        if (this->_thread_idx == 0) {
          tC_gSF_(idx_row_m, idx_batch) = qpvscale;
        }
      }
      return frg_input;
    }

  private:
    template<int kThreads, class ElementReduce>
    CUTLASS_DEVICE ElementReduce
    reduce(ElementReduce x) {
      ElementReduce max_x = fabsf(x);
      for (int mask = 1; mask < kThreads; mask <<=1) {
        max_x = cutlass::fast_max(max_x, __shfl_xor_sync(__activemask(), max_x, mask));
      }
      return max_x;
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
      left_inverse(ThreadMapR2G{}),_,_,_)(_,_,_0{},_);  // (TID, STEP_M, BATCH)

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

    return Callbacks<decltype(tC_gAux), decltype(tC_gSF) , ProblemShape>{
      params_ptr->st, cute::move(tC_gAux), cute::move(tC_gSF), problem_shape, thread_idx};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Reduction
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadShape_,
  template <class> class RegReduceFn_,
  template <class> class AtomicReduceFn_,
  class ElementOutput_,
  class StrideMNL_ = Stride<_0,_0,_0>,
  class ElementCompute_=float,
  FloatRoundStyle RoundStyle=FloatRoundStyle::round_to_nearest
>
struct VisitorScalarReductionGemv : public VisitorNodeBaseGemv<ThreadShape_> {
public:
  using Base = VisitorNodeBaseGemv<ThreadShape_>;
  using Base::Base;
  using ThreadShape = ThreadShape_;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using StrideMNL = StrideMNL_;

  using RegReduceFn = RegReduceFn_<ElementCompute>;
  using AtomicReduceFn = AtomicReduceFn_<ElementCompute>;

  // The tile accessed by the same CTA
  using CtaTileMN = typename Base::CtaTileMN;

  // Static assert to ensure that intra-warp reduction is available
  static_assert(ThreadShape::kM <= 32, "ThreadShape::kM should be <= 32");

  struct SharedStorage{};

  struct Arguments {
    ElementOutput* ptr_scalar = nullptr;
    StrideMNL dScalar = {};
    ElementCompute reduction_identity = 0;
  };

  using Params = Arguments;
  
  CUTLASS_HOST_DEVICE
  VisitorScalarReductionGemv(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

private:
  Params const* params_ptr;

public:
  template<
    class GTensorR2G, class ProblemShape>
  struct Callbacks : public CallbacksImplGemv<ProblemShape, CtaTileMN> {

    using BaseCallbacks = CallbacksImplGemv<ProblemShape, CtaTileMN>;
    
    CUTLASS_DEVICE
    Callbacks(
      GTensorR2G&& tC_gScalar,
      ProblemShape problem_shape,
      int thread_idx,
      ElementCompute reduction_identity
    ):
      tC_gScalar_(tC_gScalar),
      reduction_identity(reduction_identity),
      BaseCallbacks(problem_shape, thread_idx) { }

    GTensorR2G tC_gScalar_;
    ElementCompute reduction_identity;


    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m, ElementAccumulator const& frg_acc, ElementInput const& frg_input) {
      // In-warp reduction
      ElementCompute data = NumericConverter<ElementCompute, ElementInput, RoundStyle>{}(frg_input);
      if (!this->predicate(idx_row_m, idx_batch)) {
        data = reduction_identity;
      }
      ElementCompute reduce_data  = this->reduce<ThreadShape::kM>(data);

      // Atomic reduction
      AtomicReduceFn atomic_reduce{};
      NumericConverter<ElementOutput, ElementCompute, RoundStyle> convert_output{};

      if (this->_thread_idx == 0) {
        atomic_reduce(&tC_gScalar_(idx_batch), convert_output(reduce_data));
      }

      return frg_input;
    }

  private:
    template<int kThreads>
    CUTLASS_DEVICE ElementCompute
    reduce(ElementCompute x) {
      RegReduceFn reduce_fn{};
      ElementCompute reduce_x = x;
      for (int mask = 1; mask < kThreads; mask <<=1) {
        reduce_x = reduce_fn(reduce_x, __shfl_xor_sync(__activemask(), reduce_x, mask));
      }
      return reduce_x;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(ProblemShape problem_shape, int thread_idx) {
    Tensor mScalar = make_tensor(
      make_gmem_ptr(params_ptr->ptr_scalar),
      problem_shape,
      params_ptr->dScalar
    );

    Tensor tC_gScalar = mScalar(_0{},_0{},_);

    // Generate the predicate tensor
    ElementCompute reduction_identity = params_ptr->reduction_identity;

    return Callbacks<
      decltype(tC_gScalar), ProblemShape> {
        cute::move(tC_gScalar), problem_shape, thread_idx, reduction_identity};
  }
};

}  // namespace cutlass::epilogue::threadblock