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

#pragma once

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/pointer_base.hpp>               // cute::iter_adaptor
#include <cute/numeric/integral_constant.hpp>  // cute::false_type, cute::true_type
#include <cute/numeric/integral_ratio.hpp>     // cute::ratio

namespace cute
{

// A data type that holds one physical element meant to represent Sparsity number of logical elements
// This class is purposely not compatible with anything -- know what you're doing if you attempt to use it
template <int Sparsity, class T>
struct sparse_elem
{
  static constexpr int sparsity = Sparsity;
  using raw_type = T;
  T elem_;

  CUTE_HOST_DEVICE constexpr
  explicit sparse_elem(T const& elem = {}) : elem_(elem) {}

  CUTE_HOST_DEVICE constexpr friend bool operator==(sparse_elem const& a, sparse_elem const& b) { return a.elem_ == b.elem_; }
  CUTE_HOST_DEVICE constexpr friend bool operator!=(sparse_elem const& a, sparse_elem const& b) { return a.elem_ != b.elem_; }
  CUTE_HOST_DEVICE constexpr friend bool operator< (sparse_elem const& a, sparse_elem const& b) { return a.elem_ <  b.elem_; }
  CUTE_HOST_DEVICE constexpr friend bool operator<=(sparse_elem const& a, sparse_elem const& b) { return a.elem_ <= b.elem_; }
  CUTE_HOST_DEVICE constexpr friend bool operator> (sparse_elem const& a, sparse_elem const& b) { return a.elem_ >  b.elem_; }
  CUTE_HOST_DEVICE constexpr friend bool operator>=(sparse_elem const& a, sparse_elem const& b) { return a.elem_ >= b.elem_; }
};

template <class T>
struct is_sparse : false_type {};
template <class T>
struct is_sparse<T const> : is_sparse<T> {};
template <int S, class T>
struct is_sparse<sparse_elem<S,T>> : true_type {};
template<class T>
static constexpr auto is_sparse_v = is_sparse<T>::value;

// Overload sizeof_bits for sparse_elem.
//   Much like subbyte element types, this is the effective number of bits in a sparse_elem
//   rather than actual physical bits that may be used in storing one. Also like subbyte element
//   types, modified iterators are required to properly index and access sparse_elems.
//
//   Defining sizeof_bits like this makes reasonable expressions like N * sizeof_bits_v<E> meaningful
//   even when E is subbyte or sparse. However, this also means that sparse_elem can rather easily be
//   confused with subbyte elements and special care should be taken with each.
template <int S, class T>
struct sizeof_bits<sparse_elem<S,T>> {
  // Simple implementation that conforms to sizeof_bits
  //static constexpr auto value = sizeof_bits<T>::value / S;
  //static_assert(value != 0, "sizeof_bits=0 detected. Sparsity is larger than width.");
  //static_assert((sizeof_bits<T>::value % S) == 0, "Width needs to be a multiple of sparsity.")

  // Interesting experiment that allows any sparsity level to be used by potentially presenting
  // an integral_ratio rather than size_t. This is valid in most integer expressions as well.
  static constexpr auto value = cute::ratio(cute::Int<cute::sizeof_bits_v<T>>{}, cute::Int<S>{});
};

//
// sparse_ptr
//

template <class T, class = void>
struct is_sparse_ptr : false_type {};
template <class T>
struct is_sparse_ptr<T, void_t<typename T::iterator>> : is_sparse_ptr<typename T::iterator> {};

template <int Sparsity, class Iterator>
struct sparse_ptr : iter_adaptor<Iterator, sparse_ptr<Sparsity, Iterator>>
{
  using reference    = typename iterator_traits<Iterator>::reference;
  using element_type = typename iterator_traits<Iterator>::element_type;
  using value_type   = typename iterator_traits<Iterator>::value_type;

  // Sanity, for now
  static_assert(is_sparse<value_type>::value, "Enforce sparse value-type");
  static_assert(Sparsity == iter_value_t<Iterator>::sparsity, "Enforce sparsity S");
  static_assert(not is_sparse_ptr<Iterator>::value, "Enforce sparse singleton");

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  sparse_ptr operator+(Index const& i) const {
    // Only allow offset by multiples of the sparsity factor,
    // else the misalignments become a bug. E.g. (sparse_ptr<8,I>{} + 7) + 7
    // Motivation for subsparse_iterator or generalization of subbyte_iterator?
    assert(i % Sparsity == 0);
    return {this->get() + i / Sparsity};
  }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const {
    // Allow offset by any value and dereference.
    // Not implemented in terms of sparse_ptr::op+()
    return *(this->get() + i / Sparsity);
  }
};

template <int S, class I>
struct is_sparse_ptr<sparse_ptr<S,I>> : true_type {};

template <int Sparsity, class Iter>
CUTE_HOST_DEVICE constexpr
auto
make_sparse_ptr(Iter const& iter) {
  if constexpr (Sparsity == 1) {
    return iter;
  } else {
    return sparse_ptr<Sparsity, Iter>{iter};
  }
  CUTE_GCC_UNREACHABLE;
}

template <class NewT, int S, class Iter>
CUTE_HOST_DEVICE constexpr
auto
recast_ptr(sparse_ptr<S,Iter> const& ptr) {
  static_assert(not is_sparse<NewT>::value);
  return recast_ptr<NewT>(ptr.get());
}

//
// Display utilities
//

template <int S, class Iter>
CUTE_HOST_DEVICE void print(sparse_ptr<S,Iter> ptr)
{
  printf("sparse<%d>_", S); print(ptr.get());
}

#if !defined(__CUDACC_RTC__)
template <int S, class Iter>
CUTE_HOST std::ostream& operator<<(std::ostream& os, sparse_ptr<S,Iter> ptr)
{
  return os << "sparse<" << S << ">_" << ptr.get();
}
#endif

// {$nv-internal-release begin}
#if 0
// Reference implementation of the SM100 structured-sparsity representation
// When we have a pointer of sparse data and a corresponding pointer of metadata,
// we can create a fully functional sparse tensor as follows
//   Tensor sptensor = make_tensor(sparse_ptr_sm100(dataptr, metaptr), logical_layout);
// This can be used in reference implementations, passed to gemm, etc.
//
// This is NOT used in the SpGEMM kernels and is kept here only as a reference.
// SpGEMM kernels rely on the data tensor and metadata tensors having different layouts, pipelines, etc
// and should not be as tightly coupled as this.
template <class DataPtr, class MetaPtr>
struct sparse_ptr_sm100
{
  using value_type = iter_value_t<DataPtr>;

  // Representation only for 8-bit meta value_types
  static_assert(sizeof(iter_value_t<MetaPtr>) == 1);

  // Number of elements per uint8_t meta data
  static constexpr uint8_t ElementsPerStoredItem = 8;
  static constexpr uint8_t ElementsSparsity      = 2;

  DataPtr data_ptr_;
  MetaPtr meta_ptr_;  // An unpacked array_subbyte::iterator representation
  uint8_t idx_;       // RI: 0 <= idx_ < ElementsPerStoredItem

  CUTE_HOST_DEVICE constexpr
  sparse_ptr_sm100(DataPtr const& dptr, MetaPtr const& mptr, uint8_t idx = 0)
    : data_ptr_(dptr), meta_ptr_(mptr), idx_(idx)
  {}

  /*
  Legal metadata chunk in SM100
  FP16
  Index     Bin   HEX
  0, 1    0b0100   4
  1, 2    0b1001   9
  2, 3    0b1110   E
  0, 2    0b1000   8
  1, 3    0b1101   D
  0, 3    0b1100   C
  2, 1    0b0110   6  (Not used)
  XXX TODO    -----------------------------------
  TF32
  0       0b0100   4
  1       0b1110   E
  XXX TODO    -----------------------------------
  */
  CUTE_HOST_DEVICE constexpr static
  value_type decode_to_value_or_zero(DataPtr data, uint8_t meta, uint8_t idx)
  {
    // There are 4 abstract elements that data points to, we need the physical idx [0,2) now
    if (idx == ((meta >> 0) & 0b0011)) {
      return data[0];       // Real data value idx 0
    } else
    if (idx == ((meta >> 2) & 0b0011)) {
      return data[1];       // Real data value idx 1
    } else {
      return value_type{};  // Sparse implicit zero element
    }
  }

  // Dereference
  // GOAL: For efficient sparse kernels, this should never actually be used.
  //       Still nice to have the representation-to-abstract-value mapping recorded here
  CUTE_HOST_DEVICE constexpr
  value_type operator*() const
  {
    uint8_t meta = *meta_ptr_;
    uint8_t key = idx_ % 4;
    if (key == idx_) {
      return decode_to_value_or_zero(data_ptr_ + 0, meta >> 0, key);
    } else {
      return decode_to_value_or_zero(data_ptr_ + 2, meta >> 4, key);
    }
  }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  sparse_ptr_sm100 operator+(Index const& i) const
  {
    auto idx = idx_ + i;
    return {data_ptr_ + (idx / ElementsPerStoredItem) * (ElementsPerStoredItem / ElementsSparsity),
            meta_ptr_ + (idx / ElementsPerStoredItem),
            uint8_t(idx % ElementsPerStoredItem)};
  }

  template <class Index>
  CUTE_HOST_DEVICE constexpr
  value_type operator[](Index const& i) const { return *(*this + i); }
};
#endif
// {$nv-internal-release end}

} // end namespace cute