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
   \brief Declares Constants used in CUTLASS IR
*/

#pragma once
#include "cutlass/ir/value.h"

namespace cutlass {
namespace ir {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Constant Value Classes

// Compile-time ConstantScalar 
class ConstantScalar : public Value {
public:

  // Union to hold the consatnt signed/unsigned integers, float, and double
  union {
    double floating_point;
    uint64_t unsigned_integer;
    int64_t signed_integer;
  } value;

  //
  // Methods
  //
  // Basic ctor sets constant value to default
  ConstantScalar(
    Type *type_ = nullptr,
    std::string const &name_ = std::string(),
    Value *parent_ = nullptr
  ) : Value(type_, name_,  QualifierID::kConst, parent_) {}

  // Specialized ctor to sets the constant value at object creation
  template <typename T>
  ConstantScalar(
    T val_,
    Type *type_ = nullptr,
    std::string const &name_ = std::string(),
    Value *parent_ = nullptr
  ) : Value(type_, name_, QualifierID::kConst, parent_) { set(val_); }


  /// Copy constructor
  ConstantScalar(ConstantScalar const &constant_scalar);

  ~ConstantScalar() {}

  bool is_signed_integer() const;
  bool is_unsigned_integer() const;
  bool is_floating_point() const;


  /// Sets the internal storage to the unsigned 32-bit integer value
  void set(uint32_t val);

  /// Sets the internal storage to the unsigned 64-bit integer value
  void set(uint64_t val);

  /// Sets the internal storage to the signed 32-bit integer value
  void set(int32_t val);
  
  /// Sets the internal storage to the signed 64-bit integer value
  void set(int64_t val);

  /// Sets the internal storage to the floating point value
  void set(double val);
  

  /// Get compile-time ConstantScalar value
  uint32_t as_uint32() const;
  int32_t as_int32() const;
  int64_t as_int64() const;
  uint64_t as_uint64() const;
  double as_double() const;

  //Print ConstantScalar Value
  virtual std::ostream &print(std::ostream &) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Compile-time Constant Enum Value
class ConstantEnum : public Value {
public:
  // Union to hold constant enum value
  union {
    NumericTypeID numeric_type_id;
    OpcodeClassID opcode_class_id;
    ConvModeID  conv_mode_id;
    SafetyQualificationID safety_qualification_id;
  } value;

  // Methods
  //
  // Basic ctor sets constant value to default
  ConstantEnum(
    Type *type_ = nullptr,
    std::string const &name_ = std::string(),
    Value *parent_ = nullptr
    ) : Value(type_, name_, QualifierID::kConst, parent_) {}

  // Specialized ctor to sets the constant value at object creation
  template <typename T>
    ConstantEnum(
    T val_,
    Type *type_ = nullptr,
    std::string const &name_ = std::string(),
    Value *parent_ = nullptr
    ) : Value(type_, name_, QualifierID::kConst, parent_) { set(val_); }

  ~ConstantEnum() {}

  
  /// Sets the internal storage to NumericTypeID val
  void set(NumericTypeID val);
  void set(OpcodeClassID val);
  void set(ConvModeID val);
  void set(SafetyQualificationID val);

  /// Get compile-time ConstantEnum value
  NumericTypeID as_numeric_type() const;
  OpcodeClassID as_opcode_class() const;
  ConvModeID as_conv_mode() const;
  SafetyQualificationID as_safetey_qualification() const;

  //Print ConstantScalar Value
  virtual std::ostream &print(std::ostream &) const;
  
};

} // namespace ir
} // namespace cutlass