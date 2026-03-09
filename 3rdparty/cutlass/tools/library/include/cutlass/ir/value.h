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
   \brief Declares Value class. An object of Value class holds the following information for 
   a function argument {name, type, qualifier, and metadata vector}
*/

#pragma once
#include "cutlass/ir/type.h"
#include "cutlass/ir/metadata.h"

namespace cutlass {
namespace ir {

/////////////////////////////////////////////////////////////////////////////////////////////////

using OwningMetadataVector = std::vector<std::unique_ptr<MDNode>>;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Values
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Value base class
class Value {
public:

  /// Data type of value
  Type *type;

  /// Name of value
  std::string name;

  /// Qualifier indicating const or non-const
  QualifierID qualifier;

  /// Parent value
  Value *parent;

  /// Metadata attached to value
  OwningMetadataVector metadata;

  //
  // Methods
  //

  Value(
    Type *type_ = nullptr,
    std::string const &name_ = std::string(),
    QualifierID qualifier_ = QualifierID::kNonConst,
    Value *parent_ = nullptr
  ): 
    type(type_), name(name_), qualifier(qualifier_), parent (parent_) {}

  virtual ~Value() {}

  // virtual Value *clone() const; TODO - implement clone Value

  /// Gets the fully qualified name of a variable
  std::string qualified_name() const;

  /// Prints the Value
  virtual std::ostream &print(std::ostream &) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

using OwningValueVector = std::vector<std::unique_ptr<Value>>; 

/////////////////////////////////////////////////////////////////////////////////////////////////

class StructureValue : public Value {
  public:

  /// Member values whose types must correspond to the positional type in this type's SturctureType.
  OwningValueVector members;
  
  //
  // Methods
  //
  // ctor
  StructureValue( 
    Type *type_ = nullptr, 
    std::string const &name_ = std::string(),
    QualifierID qualifier_ = QualifierID::kNonConst,
    Value *parent_ = nullptr
  ): Value(type_, name_, qualifier_, parent_) {

    if (type->type_id != TypeID::kStructure && 
      type->type_id != TypeID::kLayout && 
      type->type_id != TypeID::kTensor) {

      throw std::runtime_error("Structure Value can only be created for Structure Type");
    }
  }

  /// Add owning structure member values
  void add_members(OwningValueVector && members_);

  /// Prints the StructureValue
  virtual std::ostream &print(std::ostream & out) const; 

  ~StructureValue(){}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ir
} // namespace cutlass