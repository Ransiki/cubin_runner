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

/**
 * \file
 * \brief C++ interface to use custom abi.
 */

//  {$nv-internal-release file}

#pragma once

#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

namespace custom_abi_helper {
template <size_t N, size_t Regcount, typename Tarr, class Functor, typename... Tarrps>
__device__ __noinline__ static void callee(Functor f, Tarrps... arrps) {
  asm volatile(" .pragma \"abi_param_reg %0\";" ::"n"(Regcount));
  uint32_t arr[] = {arrps...};
  return f(reinterpret_cast<const Tarr(&)[N]>(arr));
}
template <size_t Regcount, class Functor, typename Targ, typename... Tarrps>
__device__ __noinline__ static auto callee2(Functor f, Tarrps... arrps) -> decltype(std::declval<Functor>()(std::declval<Targ>())) {
  asm volatile(" .pragma \"abi_param_reg %0\";" ::"n"(Regcount));
  uint32_t arr[] = {arrps...};
  return f(reinterpret_cast<Targ&>(arr[0]));
}
template <size_t Regcount, class Functor, typename... Tarrps>
__device__ __noinline__ static decltype(std::declval<Functor>()()) callee(Tarrps... arrps) {
  asm volatile(" .pragma \"abi_param_reg %0\";" ::"n"(Regcount+4));
  uint32_t arr[] = {arrps...};
  return reinterpret_cast<Functor&>(arr[0])();
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
  template <size_t Regcount, class Functor, typename Targ>
  __device__ __inline__ static auto f(Functor const& f, Targ arg) -> decltype(f(arg)) {
    asm volatile(".pragma \"call_abi_param_reg %0\";" ::"n"(Regcount));
    return callee2<Regcount, Functor, Targ>(f, (&reinterpret_cast<const uint32_t&>(arg))[Seq]...);
  }
  template <size_t Regcount, class Functor>
  __device__ __inline__ static auto f(Functor f) -> decltype(f()) {
    asm volatile(".pragma \"call_abi_param_reg %0\";" ::"n"(Regcount+4));
    // static_assert(false);
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
template <class Functor, typename Targ>
__device__ __inline__ auto custom_abi_call(Functor f, Targ arg) -> decltype(f(arg)) {
  static constexpr size_t Regcount = (sizeof(Targ) + sizeof(uint32_t) - 1) / sizeof(uint32_t) +
                                     (sizeof(Functor) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  return custom_abi_helper::caller<make_index_sequence<
      (sizeof(Targ) + sizeof(uint32_t) - 1) / sizeof(uint32_t)>>::template f<Regcount>(f, arg);
}
template <class Functor>
__device__ __inline__ auto custom_abi_call(Functor f) -> decltype(f()) {
  static constexpr size_t Regcount = (sizeof(Functor) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  return custom_abi_helper::caller<make_index_sequence<Regcount>>::template f<Regcount>(f);
}

////////////////////////////////////////////////////////////////////////////////

template <class Functor>
__device__ __noinline__ auto noinline_call(Functor f) -> decltype(f()) {
  return f();
}

}  // namespace cutlass
