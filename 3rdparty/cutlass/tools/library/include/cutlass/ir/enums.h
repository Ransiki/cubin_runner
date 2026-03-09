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
/* \file
   \brief Declares a enums definitions used in CUTLASS IR.
*/

#pragma once

#include <string>

#include "cutlass/cutlass.h"

namespace cutlass {
namespace ir {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Enums used by CUTLASS Library IR
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T from_string(std::string const &);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// To string method for cutlass::Status
char const *to_string(Status status, bool pretty = false);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Top-level enumeration indicating type ID
// Type can only be Scalar Type. Thus, enums kScalar. Unlike ArgumentType where kScalar and kNumeric
enum class TypeID {
  kPointer,
  kScalar,
  kArray,
  kStructure,
  kLayout,
  kTensor,
  kFunction,
  kEnum,
  kInvalid
}; 

/// Converts a TypeID enumerant to a string
char const *to_string(TypeID type, bool pretty = false);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Type qualifiers
enum class QualifierID {
  kNonConst,
  kConst,
  kInvalid
};

/// Converts a QualifierID enumerant to a string
char const *to_string(QualifierID type, bool pretty = false);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Indicates whether kernel is safety qualified
enum class SafetyQualificationID {
  kNotQualified,
  kSafetyQualified,
  kInvalid
};

/// Converts a QualifierID enumerant to a string
char const *to_string(SafetyQualificationID type, bool pretty = false);

/// Converts a SafetyQualificationID enumerant from a string
template <>
SafetyQualificationID from_string<SafetyQualificationID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration Type
enum class EnumeratedTypeID {
  kNumericType,
  kOpcodeClass,
  kConvMode,
  kSafetyQualification,
  kInvalid
};

/// Converts a EnumeratedTypeID enumerant to a string
char const *to_string(EnumeratedTypeID type, bool pretty = false);

/// Converts a EnumeratedTypeID enumerant from a string
template <>
EnumeratedTypeID from_string<EnumeratedTypeID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Operation Type
enum class OpcodeClassID {
  kSimt,
  kTensorOp,
  kInvalid
};

/// Converts a OpcodeClassID enumerant to a string
char const *to_string(OpcodeClassID type, bool pretty = false);

/// Converts a OpcodeClassID enumerant from a string
template <>
OpcodeClassID from_string<OpcodeClassID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Operation Type
enum class ConvModeID {
  kCrossCorrelation,
  kConvolution,
  kInvalid
};

/// Converts a ConvModeID enumerant to a string
char const *to_string(ConvModeID type, bool pretty = false);

/// Converts a ConvModeID enumerant from a string
template <>
ConvModeID from_string<ConvModeID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Indicates the type of kernel argument
// ArgumentType can be both ScalarType or NumericType. Thus, enums kScalar and kNumeric
// 1) kScalar: e.g. of a Scalar ArgumentType is u32 is a Scalar type. 
// Its c++ equivalent as "type name = initializer" is "u32 m = 32"
// 2) kNumeric: e.g. of a Numeric ArgumentType is NumericTypeID is a Numeric type. 
// Its c++ equivalent as "type name = initializer" is "NumericTypeID numeric_type = u32"
enum class ArgumentTypeID {
  kScalar,
  kInteger,
  kTensor,
  kBatchedTensor,
  kStructure,
  kEnumerated,
  kInvalid
};

/// Converts a ArgumentTypeID enumerant to a string
char const *to_string(ArgumentTypeID type, bool pretty = false);

/// Parses a ArgumentTypeID enumerant from a string
template <>
ArgumentTypeID from_string<ArgumentTypeID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Numeric data type
enum class NumericTypeID {
  kUnknown,
  kVoid,
  kF16,
  kFE8M7,     // {$nv-internal-release}
  kFE8M10,    // {$nv-internal-release}
  kF32,
  kF64,
  kS2,
  kS4,
  kS8,
  kS16,
  kS32,
  kS64,
  kU2,
  kU4,
  kU8,
  kU16,
  kU32,
  kU64,
  kB1,
  kInvalid
};

/// Converts a NumericType enumerant to a string
char const *to_string(NumericTypeID type, bool pretty = false);

/// Parses a NumericType enumerant from a string
template <>
NumericTypeID from_string<NumericTypeID>(std::string const &str);

/// Returns the size of a data type in bits
int sizeof_bits(NumericTypeID type);

/// Returns true if numeric type is integer
bool is_integer_type(NumericTypeID type);

/// Returns true if numeric type is signed
bool is_signed_type(NumericTypeID type);

/// Returns true if numeric type is a signed integer
bool is_signed_integer(NumericTypeID type);

/// returns true if numeric type is an unsigned integer
bool is_unsigned_integer(NumericTypeID type);

/// Returns true if numeric type is floating-point type
bool is_float_type(NumericTypeID type);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Layout type identifier - must correspond to built-in CUTLASS Layouts.
enum class LayoutTypeID {
  kUnknown,
  kColumnMajor,
  kRowMajor,
  kBlockScalingTensor,          // {$nv-internal-release}
  kColumnMajorInterleavedK2,
  kRowMajorInterleavedK2,
  kColumnMajorInterleavedK4,
  kRowMajorInterleavedK4,
  kColumnMajorInterleavedK16,
  kRowMajorInterleavedK16,
  kColumnMajorInterleavedK32,
  kRowMajorInterleavedK32,
  kColumnMajorInterleavedK64,
  kRowMajorInterleavedK64,
  kTensorNCHW,
  kTensorNHWC,
  kTensorNC32HW32,
  kTensorC32RSK32,
  kTensorNC64HW64,
  kTensorC64RSK64,
  kInvalid
};

/// Converts a LayoutTypeID enumerant to a string
char const *to_string(LayoutTypeID layout, bool pretty = false);

/// Parses a LayoutType enumerant from a string
template <>
LayoutTypeID from_string<LayoutTypeID>(std::string const &str);

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ir
} // namespace cutlass
