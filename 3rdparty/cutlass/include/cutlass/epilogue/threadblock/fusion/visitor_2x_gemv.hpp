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
  \brief Visitor tree operation base implementation to enable composable fusions
         for the CUTLASS 2x epilogue of gemv
*/

#pragma once

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using cute::tuple;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <class... Ops>
struct VisitorImpl2xGemv: fusion::detail::Sm90VisitorImplBase<Ops...> {
  using fusion::detail::Sm90VisitorImplBase<Ops...>::Sm90VisitorImplBase;
  using fusion::detail::Sm90VisitorImplBase<Ops...>::ops;

  template <class CallbacksTuple>
  struct Callbacks {
  public:
    // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
    CallbacksTuple callbacks_tuple;

    /// Called at the start of the epilogue just before iterating over accumulator slices
    CUTLASS_DEVICE void
    begin_epilogue(int idx_batch) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.begin_epilogue(idx_batch);
        }
      );
    }

    /// Called after accumulators have been exchanged for each accumulator
    template <typename ElementAccumulator, typename... ElementInputs>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m,
          ElementAccumulator const& frg_acc,
          ElementInputs const&... frg_inputs)
      = delete; // Must be implemented for each operation

    /// Called after all steps have been completed
    CUTLASS_DEVICE void
    end_epilogue(int idx_batch) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.end_epilogue(idx_batch);
        }
      );
    }
  };

  // Callbacks factory
  // All operations must redefine this
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    ProblemShape problem_shape,
    int thread_idx
  ) {
    return transform_apply(ops,
      [&] (auto& op) {
        return op.get_callbacks(problem_shape, thread_idx);
      },
      [] (auto&&... callbacks) {
        auto callbacks_tuple = cute::make_tuple(callbacks...);
        return Callbacks<decltype(callbacks_tuple)>{callbacks_tuple};
      }
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Convenience aliases
using EmptyCallbacksGemv = VisitorImpl2xGemv<>::Callbacks<cute::tuple<>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tree visitor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
struct TreeVisitor2xGemv : VisitorImpl2xGemv<ChildOps..., NodeOp> {

  using VisitorImpl2xGemv<ChildOps..., NodeOp>::VisitorImpl2xGemv;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m,
          ElementAccumulator const& frg_acc) {
      constexpr int Rm1 = sizeof...(ChildOps);
      return cute::detail::tapply(callbacks_tuple,
        [&] (auto& child_callbacks) {
          return child_callbacks.visit(idx_batch, idx_row_m, frg_acc);
        },
        [&] (auto&&... frg_inputs) {
          return get<Rm1>(callbacks_tuple).visit(idx_batch, idx_row_m, frg_acc, frg_inputs...);
        },
        make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    ProblemShape problem_shape,
    int thread_idx
  ) {
    return Callbacks<
    decltype(VisitorImpl2xGemv<ChildOps..., NodeOp>::
      get_callbacks(
        problem_shape,
        thread_idx
      ))>(
      VisitorImpl2xGemv<ChildOps..., NodeOp>::
      get_callbacks(
        problem_shape,
        thread_idx
      )
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Topological visitor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementCompute,
  class EdgeTuple_,
  class... Ops
>
struct TopologicalVisitor2xGemv : VisitorImpl2xGemv<Ops...> {
  using EdgeTuple = EdgeTuple_;
  static_assert(is_static_v<EdgeTuple>);
  static_assert(cute::rank(EdgeTuple{}) == sizeof...(Ops));
  static_assert(sizeof...(Ops) > 1);

  using VisitorImpl2xGemv<Ops...>::VisitorImpl2xGemv;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator>
    CUTLASS_DEVICE auto
    visit(int idx_batch, int idx_row_m,
          ElementAccumulator const& frg_acc) {
      constexpr int Rm1 = sizeof...(Ops) - 1;
      auto frg_compute_tuple = cute::repeat<Rm1>(ElementCompute{});

      return cute::detail::tapply(EdgeTuple{}, callbacks_tuple, frg_compute_tuple,
        // Visit the first R-1 ops in topological order
        [&] (auto&& edge_seq, auto& callbacks, auto& frg_compute) {
          frg_compute = cute::detail::apply(frg_compute_tuple,
          // Compute the current op with children inputs
          [&] (auto const&... frg_inputs) {
            auto frg_output = callbacks.visit(idx_batch, idx_row_m, frg_acc, frg_inputs...);
            using ElementOutput = decltype(frg_output);
            using ConvertOutput = NumericConverter<ElementCompute, ElementOutput>;
            ConvertOutput convert_output{};

            return convert_output(frg_output);
          },
          // Get inputs in the sequence given by the children indices of the current op
          edge_seq
        );
        return frg_compute;
      },
      // Visit the last op
      [&] (auto const&...ops) {
        return cute::detail::apply(frg_compute_tuple,
          // Compute the last op with children inputs
          [&] (auto const&... frg_inputs) {
            return get<Rm1>(callbacks_tuple).visit(idx_batch, idx_row_m, frg_acc, frg_inputs...);
          },
          // Get inputs in the sequence given by the children indices of the last op
          get<Rm1>(EdgeTuple{})
        );
      },
      // Transform to visit R-1 ops, apply to visit last op
      make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    ProblemShape problem_shape,
    int thread_idx
  ) {
    return Callbacks<decltype(
      VisitorImpl2xGemv<Ops...>::
      get_callbacks(
        problem_shape,
        thread_idx
      ))>(
      VisitorImpl2xGemv<Ops...>::
      get_callbacks(
        problem_shape,
        thread_idx
      )
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Base class for GEMV EVT Nodes
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class ThreadShape>
struct VisitorNodeBaseGemv {
public:
  // The threadmap used throughout the GEMV EVT. The threads are assigned to consecutive
  // elements in the column
  using ThreadMap = Layout<
    Shape<Int<ThreadShape::kM * ThreadShape::kN>, _1>>;

  // The tile size of each CTA
  using CtaTileMN = decltype(make_tile(Layout<Int<ThreadShape::kM>>{}, Layout<_1>{}));

  // Constructor
  CUTLASS_HOST_DEVICE
  VisitorNodeBaseGemv(){}

  template <typename T>
  CUTLASS_DEVICE
  static auto make_iterator(T* ptr) {
    return cute::recast_ptr<T>(ptr);
  }
};

template<class ProblemShape, class CtaTileMN>
struct CallbacksImplGemv : public VisitorImpl2xGemv<>::Callbacks<cute::tuple<>> {
public:

  using CTensor = decltype(
    make_identity_tensor(ProblemShape{}).tile(CtaTileMN{})(_,_,_0{},_)
  );

  CTensor _ctensor;
  ProblemShape _problem_shape;
  int _thread_idx;

  // Constructor
  CUTLASS_DEVICE
  CallbacksImplGemv(){}

  CUTLASS_DEVICE
  CallbacksImplGemv(ProblemShape problem_shape, int thread_idx)
    : _problem_shape(problem_shape), _thread_idx(thread_idx) {
    _ctensor = make_identity_tensor(problem_shape).tile(CtaTileMN{})(_,_,_0{},_);
  }

  // Check whether the coord is out of the problem size
  CUTLASS_DEVICE
  bool predicate(int idx_row_m, int idx_batch, int tid_offset=0) {
    auto coord = _ctensor(_thread_idx + tid_offset, idx_row_m, idx_batch);
    return elem_less(coord, _problem_shape);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
