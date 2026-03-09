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

/**
 * \file
 * \brief C++ interface to use custom abi.
  {$nv-internal-release file}
 */

namespace cutlass {

template <size_t... Seq>
struct index_sequence;

template <size_t N, size_t... Next>
struct index_sequence_helper : index_sequence_helper<N - 1, N - 1, Next...> {};

template <size_t... Next>
struct index_sequence_helper<0, 0, Next...> {
  using type = index_sequence<0, Next...>;
};

template <size_t N>
using make_index_sequence = typename index_sequence_helper<N>::type;

namespace custom_abi_helper {
template <size_t N, size_t Regcount, typename Tarr, class Functor, typename... Tarrps>
__device__ __noinline__ static void callee(Functor f, Tarrps... arrps) {
  asm volatile(" .pragma \"abi_param_reg %0\";" ::"n"(Regcount));
  uint32_t arr[] = {arrps...};
  return f(reinterpret_cast<const Tarr(&)[N]>(arr));
}

template <size_t Regcount, class Functor, typename... Tarrps>
__device__ __noinline__ static void callee(Tarrps... arrps) {
  asm volatile(" .pragma \"abi_param_reg %0\";" ::"n"(Regcount));
  uint32_t arr[] = {arrps...};
  return reinterpret_cast<const Functor&>(arr[0])();
}

template <class Seq>
struct caller;

template <size_t... Seq>
struct caller<index_sequence<Seq...>> {
  template <size_t N, size_t Regcount, class Functor, typename Tarr>
  __device__ __inline__ static void f(Functor f, Tarr arr[]) {
    asm volatile(".pragma \"call_abi_param_reg %0\";" ::"n"(Regcount));
    return callee<N, Regcount, Tarr>(f, reinterpret_cast<const uint32_t *>(arr)[Seq]...);
  }
  template <size_t Regcount, class Functor>
  __device__ __inline__ static void f(Functor const& f) {
    asm volatile(".pragma \"call_abi_param_reg %0\";" ::"n"(Regcount));
    return callee<Regcount, Functor>((&reinterpret_cast<const uint32_t&>(f))[Seq]...);
  }
};
}  // namespace custom_abi_helper

template <size_t N, class Functor, typename Tarr>
__device__ __inline__ void custom_abi_call(Functor f, Tarr arr[]) {
  static constexpr size_t Regcount = (N * sizeof(Tarr) + sizeof(uint32_t) - 1) / sizeof(uint32_t) +
                                     (sizeof(Functor) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  return custom_abi_helper::caller<
      make_index_sequence<(N * sizeof(Tarr) + sizeof(uint32_t) - 1) / sizeof(uint32_t)>>::template f<N, Regcount>(f,
                                                                                                                  arr);
}
template <class Functor>
__device__ __inline__ void custom_abi_call(Functor const& f) {
  static constexpr size_t Regcount = (sizeof(Functor) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  return custom_abi_helper::caller<make_index_sequence<Regcount>>::template f<Regcount>(f);
}

////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
