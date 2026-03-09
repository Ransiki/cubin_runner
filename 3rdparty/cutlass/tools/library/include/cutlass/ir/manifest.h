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
   \brief Defines classes owning the Deliverables Library's manifest
*/

#pragma once

#include "cutlass/ir/enums.h"
#include "cutlass/ir/type.h"
#include "cutlass/ir/function.h"

namespace cutlass {
namespace ir {

/////////////////////////////////////////////////////////////////////////////////////////////////

using OwningFunctionSchemaVector = std::vector<std::unique_ptr<FunctionSchema>>;

/// Wrapper for a range of function schemas
class FunctionSchemaRange {
private:
  OwningFunctionSchemaVector::iterator start_;
  OwningFunctionSchemaVector::iterator end_;
public:

  FunctionSchemaRange();
  ~FunctionSchemaRange();

  FunctionSchemaRange(
    OwningFunctionSchemaVector::iterator start,
    OwningFunctionSchemaVector::iterator end
  );

  OwningFunctionSchemaVector::iterator begin();
  OwningFunctionSchemaVector::iterator end();
};

/////////////////////////////////////////////////////////////////////////////////////////////////
using FunctionTypeMapT = std::map<std::vector<Type*>, FunctionType*>;
using OwningFunctionTypeVector = std::vector<std::unique_ptr<FunctionType>>;

/// Wrapper for a range of function schemas
class FunctionTypeRange {
private:
  OwningFunctionTypeVector::iterator start_;
  OwningFunctionTypeVector::iterator end_;
public:

  FunctionTypeRange();
  ~FunctionTypeRange();

  FunctionTypeRange(
    OwningFunctionTypeVector::iterator start,
    OwningFunctionTypeVector::iterator end
  );

  OwningFunctionTypeVector::iterator begin();
  OwningFunctionTypeVector::iterator end();
};

/////////////////////////////////////////////////////////////////////////////////////////////////

using OwningFunctionVector = std::vector<std::unique_ptr<Function>>;

/// Wrapper for a range of function schemas
class FunctionRange {
private:
  OwningFunctionVector::iterator start_;
  OwningFunctionVector::iterator end_;
public:

  FunctionRange();
  ~FunctionRange();

  FunctionRange(
    OwningFunctionVector::iterator start,
    OwningFunctionVector::iterator end
  );

  OwningFunctionVector::iterator begin();
  OwningFunctionVector::iterator end();
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Singleton object
class Manifest {
public:

  /// Context object owning all data types
  Context context_;

  /// Function schemas - owning
  OwningFunctionSchemaVector function_schemas_;

  /// Function types - owning
  OwningFunctionTypeVector function_types_;

  /// Function instances - owning
  OwningFunctionVector functions_;

  /// Map to lookup all unique FunctionType
  FunctionTypeMapT function_type_map_;


public:

  //
  // Methods
  //

  Manifest();
  ~Manifest();

  /// Gets the singleton manifest object
  static Manifest &get();

  /// Gets the singleton context
  Context &context();

  /// Create an instance of FunctionType
  void create_function_type(TypeVector argument, std::string name = std::string());

  /// Adds a function schema and takes ownership
  FunctionSchema * add_function_schema(std::unique_ptr<FunctionSchema> &&schema);

  /// Adds a funciton type and takes ownership
  FunctionType * add_function_type(std::unique_ptr<FunctionType> &&function_type);

  /// Adds a function and takes ownership.
  Function * add_function(std::unique_ptr<Function> &&function);

  /// Returns a range of funciton schemas
  FunctionSchemaRange function_schemas();

  /// Returns an iterator at the start of a range of function schemas
  OwningFunctionSchemaVector::iterator begin_function_schemas();

  /// Returns an iterator at the end of a range of function schemas
  OwningFunctionSchemaVector::iterator end_function_schemas();

  /// Returns a range of funciton schemas
  FunctionTypeRange function_types();

  /// Returns an iterator at the start of a range of function types
  OwningFunctionTypeVector::iterator begin_function_types();

  /// Returns an iterator at the end of a range of function types
  OwningFunctionTypeVector::iterator end_function_types();

  /// Returns a range of function schemas
  FunctionRange functions();

  /// Returns an iterator at the start of a range of functions
  OwningFunctionVector::iterator begin_functions();

  /// Returns an iterator at the end of a range of functions
  OwningFunctionVector::iterator end_functions();

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ir
} // namespace cutlass
