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
   \brief Defines a enums definitions used in cutlass::ir and users of cutlass::ir
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include "cutlass/ir/enums.h"

namespace cutlass {
namespace ir {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Base class for the type system 
/// Used to derive ScalarType, PointerType, ArrayType, StructureType, LayoutType, TensorType, EnumeratedType
class Type {
public:
  TypeID type_id;
  //
  // Methods
  //

  Type(TypeID type_id_ = TypeID::kInvalid) : type_id(type_id_) {}
  virtual ~Type() = default;

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const =0;

  /// Gets the size of a type
  virtual size_t size() const =0;

  /// Computes the alignment required for objects of this type
  virtual size_t alignment() const;

  /// Computes the padding needed to align the type at the given offset
  virtual size_t padding(size_t offset = 0) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Data type representing scalars
class ScalarType : public Type {
public:
  NumericTypeID element_type;

  //
  // Methods
  //

  ScalarType(
    NumericTypeID element_type_ = NumericTypeID::kInvalid
  ): 
    element_type (element_type_), Type(TypeID::kScalar) {}

  virtual ~ScalarType() {}

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Address space enumeration
enum class AddressSpaceID {
  kDevice,
  kHost,
  kInvalid
};

/// Pointer type
class PointerType : public Type {
public:
  Type *element_type;
  AddressSpaceID address_space;

  //
  // Methods
  //
  PointerType(
    Type *element_type_ = nullptr,
    AddressSpaceID address_space_ = AddressSpaceID::kDevice) : 
      element_type(element_type_), 
      address_space(address_space_), 
      Type(TypeID::kPointer) {}

  virtual ~PointerType();

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;

  /// Computes the alignment required for objects of this type
  virtual size_t alignment() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Data type representing contant-length arrays
class ArrayType : public Type {
public:
  Type *element_type;
  size_t capacity;

  //
  // Methods
  //

  ArrayType(
    Type *element_type_ = nullptr,
    size_t capacity_ = 0) : element_type(element_type_), capacity(capacity_), Type(TypeID::kArray) {}

  virtual ~ArrayType();

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;

  /// Computes the alignment required for objects of this type
  virtual size_t alignment() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure type
class StructureType : public Type {
public:
  //
  // Data members
  //

  std::vector<Type *> members;
  //
  // Methods
  //
  StructureType(
    std::vector<Type *> const &members_ = std::vector<Type *>()
  ): 
    members(members_), Type(TypeID::kStructure) {}

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;

  /// Computes the alignment required for objects of this type
  virtual size_t alignment() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Layout type
class LayoutType : public StructureType {
public:

  /// ID of built-in layout type
  LayoutTypeID layout;

  //
  // Methods
  //

  /// Layout type - members are defined by the constructor based on the LayoutTypeID
  LayoutType(LayoutTypeID layout_ = LayoutTypeID::kColumnMajor) : 
    layout(layout_), 
    StructureType() {

    type_id = TypeID::kLayout;
  }

  /// Prints the type 
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;

  /// Computes the alignment required for objects of this type
  virtual size_t alignment() const;

  /// Get stride rank for the layout_id
  static int get_stride_rank(LayoutTypeID layout_id);
  
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tensor type
class TensorType : public StructureType {
public:

  /// Element type of tensor
  ScalarType *element_type;

  /// Layout type of tensor
  LayoutType *layout_type;

  //
  // Methods
  //

  TensorType(
    ScalarType *element_type_ = nullptr, LayoutType *layout_type_ = nullptr
  ):
    element_type(element_type_),
    layout_type(layout_type_),
    StructureType(std::vector<Type*>{element_type_, layout_type_}) {

    type_id = TypeID::kTensor;
  }

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;

  /// Computes the alignment required for objects of this type
  virtual size_t alignment() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerated type
class EnumeratedType : public Type {
public:
  // Element type of enumeration
  EnumeratedTypeID element_type;

  //
  // Methods
  //
  EnumeratedType(
    EnumeratedTypeID element_type_ = EnumeratedTypeID::kInvalid
  ) : element_type(element_type_), Type(TypeID::kEnum) {}


  virtual ~EnumeratedType() {}

  /// Prints the type
  virtual std::ostream &print(std::ostream &) const;

  /// Gets the size of a type
  virtual size_t size() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Global type registry - this owns all type instances with a data structure to expedite lookup
/// all types are created only once in an Context instance. Type pointers can be retrevied by the 
/// clients of cutlass::ir for say creating new FunctionSignature
class Context {
public:
  //
  // Data members
  //

  // placeholder for all basic ScalarType/NumericType
  ScalarType void_ty, f16_ty, fe8m7_ty, fe8m10_ty, f32_ty, f64_ty; 
  ScalarType s2_ty, s4_ty, s8_ty, s16_ty, s32_ty, s64_ty;
  ScalarType u2_ty, u4_ty, u8_ty, u16_ty, u32_ty, u64_ty, b1_ty;

  // placeholder for all basic LayoutType
  LayoutType row_major_ty, col_major_ty, nchw_ty, nhwc_packed_c_ty;

  // map to lookup all unique StructureType
  using StructTypeMapT = std::map<std::vector<Type*>, std::unique_ptr<StructureType>>;
  StructTypeMapT struct_type_map;

  // map to lookup all unique TensorType
  using TensorTypeMapT = std::map<std::pair<ScalarType*, LayoutType*>, std::unique_ptr<TensorType>>;
  TensorTypeMapT tensor_type_map;

  // placeholder for enumerated types
  EnumeratedType numeric_ty, opcode_class_ty, conv_mode_ty;

  //
  // Methods
  //
  // ctor create all basic types 
  Context();
  
  // Gets scalar types specified by the numeric_type_id in the function name
  ScalarType* getVoidTy() {return &void_ty;}
  ScalarType* getF16Ty() {return &f16_ty;}
  ScalarType* getFE8M7Ty() {return &fe8m7_ty;}      // {$nv-internal-release}
  ScalarType* getFE8M10Ty() {return &fe8m10_ty;}    // {$nv-internal-release}
  ScalarType* getF32Ty() {return &f32_ty;}
  ScalarType* getF64Ty() {return &f64_ty;}

  ScalarType* getS2Ty() {return &s2_ty;}
  ScalarType* getS4Ty() {return &s4_ty;}
  ScalarType* getS8Ty() {return &s8_ty;}
  ScalarType* getS16Ty(){return &s16_ty;}
  ScalarType* getS32Ty(){return &s32_ty;}
  ScalarType* getS64Ty(){return &s64_ty;}

  ScalarType* getU2Ty() {return &u2_ty;}
  ScalarType* getU4Ty() {return &u4_ty;}
  ScalarType* getU8Ty() {return &u8_ty;}
  ScalarType* getU16Ty(){return &u16_ty;}
  ScalarType* getU32Ty(){return &u32_ty;}
  ScalarType* getU64Ty(){return &u64_ty;}

  ScalarType* getB1Ty(){return &b1_ty;}

  // Gets enumerated types specified by enum_type_id
  EnumeratedType* getEnumeratedType(EnumeratedTypeID enum_type_id);

  // Gets scalar types specified by numeric_type_id
  ScalarType *getScalarType(NumericTypeID numeric_type_id);

  // Gets LayoutType for LayoutTypeID id
  LayoutType* getLayoutType(LayoutTypeID layout_type_id);

  // Gets StructureType declared with std::vector<Type*>& members
  StructureType* getStructureType(std::vector<Type*> const & members);

  // Gets TensorType declared with element type and layout type
  TensorType* getTensorType(ScalarType *element_type, LayoutType * layout_type);
 
};

/////////////////////////////////////////////////////////////////////////////////////////////////
  
} // namespace ir
} // namespace cutlass
