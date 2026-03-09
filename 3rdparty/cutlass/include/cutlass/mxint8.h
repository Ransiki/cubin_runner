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

/*!
  \file
  \brief MX8 format
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_size.h"
#include "cutlass/platform/platform.h"
// S2_6 types are available starting CUDA 12+
#if (__CUDACC_VER_MAJOR__ >= 12) && defined(CUDA_UTCMXQMMA_MXINT8_S2E6_ENABLED) && (CUDA_UTCMXQMMA_MXINT8_S2E6_ENABLED)
#define CUDA_MX8_ENABLED 1
#endif

#if defined(__CUDA_ARCH__) && defined(CUDA_MX8_ENABLED) && (CUDA_MX8_ENABLED)
#  if (__CUDACC_VER_MAJOR__ >= 12) && (__CUDA_ARCH__ >= 1000) && defined(__CUDA_ARCH_FEAT_SM100_ALL)
#    define CUDA_PTX_MX8_CVT_ENABLED 1
#  endif
#endif

// #define CUTLASS_DEBUG_TRACE_LEVEL 2
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

struct mxint8_t
{

  using Storage = int8_t;

  //
  // Data members
  //
  static constexpr int8_t MAX_STORAGE = 127;
  /// Data container
  Storage storage = {};

  /// Ctors.
  mxint8_t() = default;

  CUTLASS_HOST_DEVICE
  explicit mxint8_t(Storage s) : storage(s) {
  }

  /// Is finite implementation
  CUTLASS_HOST_DEVICE
  static bool isfinite(mxint8_t /*flt*/) {
    return true;
  }

  /// Is NaN implementation
  CUTLASS_HOST_DEVICE
  static bool isnan(mxint8_t /*flt*/) {
    return false;
  }

  /// Is infinite implementation
  CUTLASS_HOST_DEVICE
  static bool isinf(mxint8_t /*flt*/) {
    return false;
  }

  /// Is infinite implementation
  CUTLASS_HOST_DEVICE
  static bool isnormal(mxint8_t /*flt*/) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static mxint8_t bitcast(Storage x) {
    mxint8_t f;
    f.storage = x;
    return f;
  }

  CUTLASS_HOST_DEVICE
  static float fractional_bits() {
    return float(pow(float(2),float(-6)));
  }

  CUTLASS_HOST_DEVICE
  static mxint8_t max() {
    return bitcast(MAX_STORAGE);
  }

  /// Floating point conversion
  CUTLASS_HOST_DEVICE
  explicit mxint8_t(float x) {
    const float max_float_val = float(max());

    #if defined(__CUDA_ARCH__)
    bool is_nan = ::isnan(x);
    bool is_inf = ::isinf(x);
    #else 
    bool is_nan = std::isnan(x);
    bool is_inf = std::isinf(x);
    #endif 

    if (is_nan) {
      storage = MAX_STORAGE;
    }
    else if (is_inf) {
      storage = (x == -cutlass::platform::numeric_limits<float>::infinity() ? -MAX_STORAGE : MAX_STORAGE);
    }
    else {
      if (x > max_float_val) {
        storage = MAX_STORAGE;
      }
      else if (x < -max_float_val) {
        storage = -MAX_STORAGE; // SATNARROW
      }
      else {
        float val = x * pow(float(2),float(6));
        #if defined(__CUDA_ARCH__)
          storage = static_cast<int8_t>(rintf(val));     // halfway cases are rounded to the nearest even
        #else
          storage = static_cast<int8_t>(std::rint(val)); // halfway cases are rounded to the nearest even
        #endif  
      }
    }
  }

  CUTLASS_HOST_DEVICE
  explicit mxint8_t(double x) : mxint8_t(static_cast<float>(x)) { 
  }

  // Integer conversion
  CUTLASS_HOST_DEVICE
  explicit mxint8_t(int x) : mxint8_t(static_cast<float>(x)) { 
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator float() const {
    return static_cast<float>(storage * fractional_bits());
  }

  /// Converts to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return static_cast<int>(static_cast<float>(storage * fractional_bits()));
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  Storage &raw() {
    return storage;
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  Storage raw() const {
    return storage;
  }

  /// Returns the sign bit
  CUTLASS_HOST_DEVICE
  bool signbit() const {
    return (storage < 0);
  }


  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Arithmetic operators
  //
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // Note: Almost all data types cast to float then do the arithmetic operations
  // Types inheriting from this class can overload them if specialized instructions are available
  // in HW (e.g. half_t)


  CUTLASS_HOST_DEVICE
  friend bool operator==(mxint8_t const &lhs, mxint8_t const &rhs) {
    return lhs.storage == rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator!=(mxint8_t const &lhs, mxint8_t const &rhs) {
    return lhs.storage != rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator<(mxint8_t const &lhs, mxint8_t const &rhs) {
    return lhs.storage < rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator<=(mxint8_t const &lhs, mxint8_t const &rhs) {
    return lhs.storage <= rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator>(mxint8_t const &lhs, mxint8_t const &rhs) {
    return lhs.storage > rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  friend bool operator>=(mxint8_t const &lhs, mxint8_t const &rhs) {
    return lhs.storage >= rhs.storage;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator+(mxint8_t const &lhs, mxint8_t const &rhs) {
    return mxint8_t(float(lhs) + float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator-(mxint8_t const &lhs) {
    return mxint8_t(-float(lhs));
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator-(mxint8_t const &lhs, mxint8_t const &rhs) {
    return mxint8_t(float(lhs) - float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator*(mxint8_t const &lhs, mxint8_t const &rhs) {
    return mxint8_t(float(lhs) * float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator/(mxint8_t const &lhs, mxint8_t const &rhs) {
    return mxint8_t(float(lhs) / float(rhs));
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t &operator+=(mxint8_t &lhs, mxint8_t const &rhs) {
    lhs = mxint8_t(float(lhs) + float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t &operator-=(mxint8_t &lhs, mxint8_t const &rhs) {
    lhs = mxint8_t(float(lhs) - float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t &operator*=(mxint8_t &lhs, mxint8_t const &rhs) {
    lhs = mxint8_t(float(lhs) * float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t &operator/=(mxint8_t &lhs, mxint8_t const &rhs) {
    lhs = mxint8_t(float(lhs) / float(rhs));
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t &operator++(mxint8_t &lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = mxint8_t(tmp);
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t &operator--(mxint8_t &lhs) {
    float tmp(lhs);
    --tmp;
    lhs = mxint8_t(tmp);
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator++(mxint8_t &lhs, int) {
    mxint8_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = mxint8_t(tmp);
    return ret;
  }

  CUTLASS_HOST_DEVICE
  friend mxint8_t operator--(mxint8_t &lhs, int) {
    mxint8_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = mxint8_t(tmp);
    return ret;
  }

};

CUTLASS_HOST_DEVICE
mxint8_t abs(mxint8_t const& h) {
  mxint8_t abs_val;
  using Storage = typename mxint8_t::Storage;
  abs_val.storage = (h.storage < 0) ? Storage(-h.storage) : h.storage;
  return abs_val;
}

/// Defines the size of an element in bits - specialized for mxint8_t
template <>
struct sizeof_bits<mxint8_t> {
  static constexpr int value = 8;
};
} // namespace cutlass

//
// User-defined literals
//

CUTLASS_HOST_DEVICE
cutlass::mxint8_t operator "" _mint8(long double x) {
  return cutlass::mxint8_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::mxint8_t operator "" _mint8(unsigned long long int x) {
  return cutlass::mxint8_t(int(x));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__)
namespace std {

/// Numeric limits
template <>
struct numeric_limits<cutlass::mxint8_t> {
  using type = cutlass::mxint8_t;
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = false;
  static bool const has_signaling_NaN = false;
  static bool const has_denorm_loss = false;
  static std::float_denorm_style const has_denorm = std::denorm_absent;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 8;
  static bool const has_infinity = false;
  /// Least positive value
  static type min() { return type::bitcast(1); }
  /// Maximum finite value
  static type max() { return type::bitcast(127); }
  /// Least finite value
  static type lowest() { return type::bitcast(-128); }
};
}  // namespace std
#endif

namespace cutlass {
namespace platform {

// TODO https://jirasw.nvidia.com/browse/CFK-15931 avoid duplicate numeric_limits for device and host side // {$nv-release-never}

/// Forward Declaration
template <class T>
struct numeric_limits;

/// Numeric limits
template <>
struct numeric_limits<cutlass::mxint8_t> {
  using type = cutlass::mxint8_t;
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = false;
  static bool const has_signaling_NaN = false;
  static bool const has_denorm_loss = false;
  static cutlass::platform::float_denorm_style const has_denorm = cutlass::platform::denorm_absent;
  static cutlass::platform::float_round_style const round_style = cutlass::platform::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 8;
  static bool const has_infinity = false;
  /// Least positive value
  static type min() { return type::bitcast(1); }
  /// Maximum finite value
  CUTLASS_HOST_DEVICE static type max() { return type::bitcast(127); }
  /// Least finite value
  static type lowest() { return type::bitcast(-128); }
};

}  // namespace platform 
}  // namespace cutlass