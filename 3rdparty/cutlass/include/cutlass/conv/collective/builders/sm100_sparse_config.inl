/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Sparse configs specific for SM100 structure sparse kernels
*/


#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace cutlass {

using namespace cute;

template<
  // Must be sparse_elem<S, T> to tell the sparsity and fundamental datatype information.
  class ElementAMma, class ElementEMma, class TileShape_
>
struct Sm100ConvSparseConfig {
  using TensorAType     = typename ElementAMma::raw_type;
  static constexpr int TensorASparsity = ElementAMma::sparsity;
  using TensorEType     = typename ElementEMma::raw_type;
  static constexpr int TensorESparsity = ElementEMma::sparsity;

  using ElementA = TensorAType;
  using TileShape = TileShape_;

  static constexpr int TileM = size<0>(TileShape{});
  static constexpr int TileN = size<1>(TileShape{});

  using UnreorderedAtom32b = Layout<Shape <_16, _32>,
                                    Stride<_32,  _1>>;
  using UnreorderedAtom16b = Layout<Shape <_16, _64>,
                                    Stride<_64,  _1>>;
  using UnreorderedAtom8b  = Layout<Shape < _16, _128>,
                                    Stride<_128,   _1>>;
  using UnreorderedAtom = conditional_t<sizeof_bits_v<ElementA> ==  8, UnreorderedAtom8b, 
                          conditional_t<sizeof_bits_v<ElementA> == 16, UnreorderedAtom16b,
                                                                       UnreorderedAtom32b>>;

  // https://confluence.nvidia.com/pages/viewpage.action?spaceKey=BLACKWELLSMARCH&title=UTCHMMA+TID+M%3D128+or+M%3D256 // {$nv-release-never}
  static constexpr int AlignmentM = size<0>(shape(UnreorderedAtom{}));
  static constexpr int AlignmentK = size<1>(shape(UnreorderedAtom{}));
  static constexpr int RepeatM = TileM / AlignmentM;

  using GmemReorderedAtom32b = Layout<Shape <Shape <  _8, _2>, Shape < _8,   _2, _4>>,
                                      Stride<Stride< _64, _8>, Stride< _1, _512,_16>>>;
  using GmemReorderedAtom16b = Layout<Shape <Shape <  _8, _2>, Shape <_16,   _2, _4>>,
                                      Stride<Stride<_128,_16>, Stride< _1,_1024,_32>>>;
  using GmemReorderedAtom8b  = Layout<Shape<_16,_128>, Stride<_128,_1>>;
  using GmemReorderedAtom    = conditional_t<sizeof_bits_v<ElementA> ==  8, GmemReorderedAtom8b, 
                               conditional_t<sizeof_bits_v<ElementA> == 16, GmemReorderedAtom16b,
                                                                            GmemReorderedAtom32b>>;

  using SmemReorderedAtom32b = Layout<Shape <Shape <  _8, _2,Int<RepeatM>>, Shape < _8,   _2, _4>>,
                                      Stride<Stride< _64, _8,       _1024>, Stride< _1, _512,_16>>>;
  using SmemReorderedAtom16b = Layout<Shape <Shape <  _8, _2,Int<RepeatM>>, Shape <_16,   _2, _4>>,
                                      Stride<Stride<_128,_16,       _2048>, Stride< _1,_1024,_32>>>;
  using SmemReorderedAtom8b  = Layout<Shape <Shape < _16,Int<RepeatM>>, _128>,
                                      Stride<Stride<_128,       _2048>,   _1>>;
  using SmemReorderedAtom    = conditional_t<sizeof_bits_v<ElementA> ==  8, SmemReorderedAtom8b, 
                               conditional_t<sizeof_bits_v<ElementA> == 16, SmemReorderedAtom16b,
                                                                            SmemReorderedAtom32b>>;      

  using SmemLayoutAtomE    = ComposedLayout<Swizzle<0,4,3>,
                                            smem_sparse_ptr_flag_bits<TensorESparsity, sizeof_bits_v<TensorEType>>,
                                            SmemReorderedAtom>;
  using TileShapeE  = decltype(replace<2>(TileShape{}, make_shape(Int<size<1>(SmemReorderedAtom{})>{})));

  // Note: this differs from GmemReorderedAtom which considers padding in shape.
  // Used for metadata copy which does not consider padding in shape.
  using ReorderedAtom32b = Layout<Shape <Shape <  _8, _2>, Shape < _8,   _2, _2>>,
                                  Stride<Stride< _64, _8>, Stride< _1, _512,_32>>>;
  using ReorderedAtom16b = Layout<Shape <Shape <  _8, _2>, Shape <_16,   _2, _2>>,
                                  Stride<Stride<_128,_16>, Stride< _1,_1024,_64>>>;
  using ReorderedAtom8b  = Layout<Shape<_16,_128>, Stride<_128,_1>>;
  using ReorderedAtom = conditional_t<sizeof_bits_v<ElementA> ==  8, ReorderedAtom8b, 
                        conditional_t<sizeof_bits_v<ElementA> == 16, ReorderedAtom16b, 
                                                                     ReorderedAtom32b>>;
  
  using ExpandFactor = conditional_t<sizeof_bits_v<ElementA> == 32, _2,
                       conditional_t<sizeof_bits_v<ElementA> == 16, _2,
                       _1>>;

  // Fundamental datatype.
  using LogicalElemsAPerChunk = conditional_t<sizeof_bits_v<ElementA> == 32, _2, 
                                conditional_t<sizeof_bits_v<ElementA> == 16, _4,
                                conditional_t<sizeof_bits_v<ElementA> ==  8, _4,
                                _8>>>;
  using PhysicalElemsAPerChunk = Int<LogicalElemsAPerChunk{} / TensorASparsity>;
  // At present, only FP4 with OMMA is pair-wise.
  using ElemsARawPerElementAMmaRaw = conditional_t<sizeof_bits_v<ElementA> == 4, _2, _1>;
  // The number of bits used to represent a subchunk.
  using ElementEBitsPerElementAMma = conditional_t<sizeof_bits_v<ElementA> == 32, _4, 
                                     conditional_t<sizeof_bits_v<ElementA> == 16, _2,
                                     conditional_t<sizeof_bits_v<ElementA> ==  8, _2,
                                     _2>>>;

  // (K, (C, S, R, T))
  using LayoutA = Layout< Shape<int32_t,  Shape< Shape<Int<TensorASparsity>,              int32_t>, int32_t, int32_t, int32_t>>,
                         Stride<int64_t, Stride<Stride<                  _1, Int<TensorASparsity>>, int64_t, int64_t, int64_t>>>;
  // ((atomK, repK), (atomC, repC), S, R, T)
  using LayoutE32b = Layout< Shape< Shape<  _8, _2, int32_t>,  Shape< Shape< _8,   _2, _4, int32_t>, int32_t, int32_t, int32_t>>,
                            Stride<Stride< _64, _8,   _1024>, Stride<Stride< _1, _512,_16, int64_t>, int64_t, int64_t, int64_t>>>;
  using LayoutE16b = Layout< Shape< Shape<  _8, _2, int32_t>,  Shape< Shape<_16,   _2, _4, int32_t>, int32_t, int32_t, int32_t>>,
                            Stride<Stride<_128,_16,   _2048>, Stride<Stride< _1,_1024,_32, int64_t>, int64_t, int64_t, int64_t>>>;
  using LayoutE8b  = Layout< Shape< Shape< _16, int32_t>,  Shape< Shape<_128, int32_t>, int32_t, int32_t, int32_t>>,
                            Stride<Stride<_128,   _2048>, Stride<Stride<  _1, int64_t>, int64_t, int64_t, int64_t>>>;
  using LayoutE = conditional_t<sizeof_bits_v<ElementA> ==  8, LayoutE8b, 
                  conditional_t<sizeof_bits_v<ElementA> == 16, LayoutE16b, 
                                                               LayoutE32b>>;

  // The following two functions are provided for users to fill dynamic problem size to the layout_a/e.
  template <class Problem_t, class ShapeIdxType, class StrideIdxType>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layoutFlt(Problem_t problem, ShapeIdxType a, StrideIdxType b) {
    // filter params
    ShapeIdxType k = cute::get<0>(problem.shape_B);
    ShapeIdxType t = cute::get<1>(problem.shape_B);
    ShapeIdxType r = cute::get<2>(problem.shape_B);
    ShapeIdxType s = cute::get<3>(problem.shape_B);
    ShapeIdxType c = cute::get<4>(problem.shape_B);

    StrideIdxType stride_k = cute::get<0>(problem.stride_B);
    StrideIdxType stride_t = cute::get<1>(problem.stride_B);
    StrideIdxType stride_r = cute::get<2>(problem.stride_B);
    StrideIdxType stride_s = cute::get<3>(problem.stride_B);
    StrideIdxType stride_c = cute::get<4>(problem.stride_B);

    return make_layout(
              make_shape(
                k,
                make_shape(
                  make_shape(Int<TensorASparsity>{}, c / Int<TensorASparsity>{}),  // Note: The filter C has to be rounded
                s, r, t)),
              make_stride(
                stride_k,                                                          // Note: The stride K might need to be padded
                make_stride(
                  make_stride(_1{}, Int<TensorASparsity>{}),
                stride_s, stride_r, stride_t)));
  }

  template <class Problem_t, class ShapeIdxType, class StrideIdxType>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layoutMeta(Problem_t problem, ShapeIdxType a, StrideIdxType b) {
    // filter params
    ShapeIdxType k = cute::get<0>(problem.shape_B);
    ShapeIdxType t = cute::get<1>(problem.shape_B);
    ShapeIdxType r = cute::get<2>(problem.shape_B);
    ShapeIdxType s = cute::get<3>(problem.shape_B);
    ShapeIdxType c = cute::get<4>(problem.shape_B);

    auto expandE = ExpandFactor{};
    ShapeIdxType k_rounded = (k + AlignmentM - 1) / AlignmentM * AlignmentM;
    ShapeIdxType c_rounded = (c + AlignmentK - 1) / AlignmentK * AlignmentK;
    ShapeIdxType k_atom_repeat = k_rounded / size<0>(GmemReorderedAtom{});
    ShapeIdxType c_atom_repeat = c_rounded * expandE / size<1>(GmemReorderedAtom{});

    StrideIdxType stride_s = k_rounded * c_rounded * expandE;
    StrideIdxType stride_r = k_rounded * c_rounded * expandE * s;
    StrideIdxType stride_t = k_rounded * c_rounded * expandE * s * r;
    // it is static integer
    auto stride_k_atom = cosize(GmemReorderedAtom{});
    StrideIdxType stride_c_atom = cosize(GmemReorderedAtom{}) * k_rounded / size<0>(GmemReorderedAtom{});

    return make_layout(
      make_shape(
        append(shape<0>(GmemReorderedAtom{}), k_atom_repeat),
        make_shape(
          append(shape<1>(GmemReorderedAtom{}), c_atom_repeat),
          s, r, t)),
      make_stride(
        append(stride<0>(GmemReorderedAtom{}), stride_k_atom),
        make_stride(
          append(stride<1>(GmemReorderedAtom{}), stride_c_atom),
          stride_s, stride_r, stride_t
        )
      )
    );
  }

  template <class Problem_t, class ShapeIdxType, class StrideIdxType>
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_reorder_layoutMeta(Problem_t problem, ShapeIdxType a, StrideIdxType b) {

    // filter params
    ShapeIdxType k = cute::get<0>(problem.shape_B);
    ShapeIdxType t = cute::get<1>(problem.shape_B);
    ShapeIdxType r = cute::get<2>(problem.shape_B);
    ShapeIdxType s = cute::get<3>(problem.shape_B);
    ShapeIdxType c = cute::get<4>(problem.shape_B);

    auto expandE = ExpandFactor{};
    ShapeIdxType k_rounded = (k + AlignmentM - 1) / AlignmentM * AlignmentM;
    ShapeIdxType c_rounded = (c + AlignmentK - 1) / AlignmentK * AlignmentK;
    ShapeIdxType k_atom_repeat = k_rounded / size<0>(ReorderedAtom{});
    ShapeIdxType c_atom_repeat = c_rounded / size<1>(ReorderedAtom{});

    StrideIdxType stride_s = k_rounded * c_rounded * expandE;
    StrideIdxType stride_r = k_rounded * c_rounded * expandE * s;
    StrideIdxType stride_t = k_rounded * c_rounded * expandE * s * r;
    
    return make_layout(
      make_shape(
        shape(ReorderedAtom{}),
        make_shape(
          k_rounded / size<0>(ReorderedAtom{}),
          c_rounded / size<1>(ReorderedAtom{}),
          make_shape(s, r, t)
        )
      ),
      make_stride(
        stride(ReorderedAtom{}),
        make_stride(
          // can't use cosize as there is padding in the tensor
          size<0>(ReorderedAtom{}) * size<1>(ReorderedAtom{}) * expandE,
          size<1>(ReorderedAtom{}) * k_rounded * expandE,
          make_stride(stride_s, stride_r, stride_t)
        )
      )
    );
  }

};

} // End namespace cutlass