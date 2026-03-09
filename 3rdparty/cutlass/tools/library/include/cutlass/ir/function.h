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
   \brief Declares FunctionScehma, FunctionType, Function, and DeviceFunction classes
*/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <climits>

#include "cutlass/cutlass.h"
#include "cutlass/ir/constants.h"
#include "cutlass/ir/type.h"
#include "cutlass/ir/value.h"
#include "cutlass/ir/metadata.h"

namespace cutlass {
namespace ir {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct FunctionSchema;
struct FunctionType;
struct Function;
struct DeviceFunction;

/// Vector of FunctionSchema objects - non-owning
using FunctionSchemaVector = std::vector<FunctionSchema *>;

/// Vector of FunctionSchema objects  - non-owning
using FunctionTypeVector = std::vector<FunctionType *>;

/// Vector of FunctionSchema objects - non-owning
using FunctionVector = std::vector<Function *>;

/// Vector of types - non-owning
using TypeVector = std::vector<Type *>;

/// Vector of values - owning
using OwningValueVector = std::vector<std::unique_ptr<Value>>;


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Schema indicates the sequence of arguments needed.
struct FunctionSchema {

  //
  // Type definitions
  //

  /// Argument structure
  struct Argument {

    /// Indicates the type of argument
    ArgumentTypeID type;

    /// Argument name
    std::string name;

    /// Description of argument
    std::string description;

    /// If type is ArgumentTypeID::kStructure, these are its members
    std::vector<Argument> members;

    //
    // Methods
    //

    Argument(): type(ArgumentTypeID::kInvalid) { }

    Argument(
      ArgumentTypeID type_, 
      std::string const &name_,
      std::string const &description_ = "",      
      std::vector<Argument> const &members_ = std::vector<Argument>()
    );

  };
  
  /// Sequence of arguments
  using ArgumentSchemaVector = std::vector<Argument>;

  //
  // Data members
  //

  /// Name of math function schema
  std::string name;

  /// Descruption of math function
  std::string description;

  /// List of arguments schema vector
  ArgumentSchemaVector arguments;

  /// Collection of function types which are in scope of this function schema
  FunctionTypeVector function_types;

  //
  // Methods
  //

  FunctionSchema(
    std::string const &name_,
    std::string const &description_ = "",
    ArgumentSchemaVector const &arguments_ = ArgumentSchemaVector()
  ): 
    name(name_), description(description_), arguments(arguments_) { }

  virtual ~FunctionSchema() {}

  /// Prints the argument schema
  void print_usage(std::ostream &out, int indent = 0) const;

private:

  /// Prints the argument schema
  void print_usage_(std::ostream &out, bool top_level, int indent, Argument const &arg) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// FunctionType
struct FunctionType {

  /// Parent
  FunctionSchema * parent;

   /// Name of math function schema e.g. "hgemm signature"
  std::string name;

  // Type information for call exposed function arguments
  TypeVector arguments;

  /// Collection of functions
  FunctionVector functions;

  //
  // Methods
  //

  // ctor
  FunctionType(
    FunctionSchema *parent_ = nullptr,
    std::string const &name_ = std::string()
  );

  /// Prints function type to stdout
  std::ostream & print(std::ostream &out) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Defining Function outline
struct Function {

  //
  // Data Members
  //

  /// Parent
  FunctionType * parent;

  /// Globally unique identifier
  uint64_t guid;

  /// Minimum CUDA device compute capability needed to run the kernel
  int minimum_compute_capability;

  /// Maximum CUDA device compute capability needed to run the kernel
  int maximum_compute_capability;

  /// Name for math function implementation
  std::string name;

  // Defines/stores Value {type, qualifier, name, metadata vector} for arguments that passed 
  // in from the caller (e.g gemmcood, tensorrefs, epilog, ...)
  OwningValueVector arguments;

  /// Defines/stores Value {type, qualifier, name, metadata vector} for constants that are not passed 
  // in from the caller (e.g. cta-tile-size, warp-tile-size, ...)
  OwningValueVector constants;

  /// Owning pointer to device function.
  std::unique_ptr<DeviceFunction> device_function;

  //
  // Methods
  //

  // Ctor
  Function(
    FunctionType *parent_ = nullptr,
    uint64_t guid_ = 0,
    std::string name_ = std::string(),
    int minimum_compute_capability_ = INT_MIN,
    int maximum_compute_capability_ = INT_MAX
  );

  /// Insert Tensor argument in Function::arguments
  Status insert_tensor_argument(Type* t, FunctionSchema::Argument & arg_schema);

  /// Insert Structure argument in Function::arguments
  Status insert_structure_argument(Type* t, FunctionSchema::Argument & arg_schema);

  /// Insert Integer argument in Function::arguments
  Status insert_integer_argument(Type* t, FunctionSchema::Argument & arg_schema);

  /// Insert ConstantScalars values in Function::constants
  template <typename T> 
  Status insert_constant_scalar(
    Context & context,                // context object
    FunctionSchema::Argument & arg,   // corresponding schema argument
    NumericTypeID numeric_type_id,    // ScalarType's numeric element type
    T const_numeric_value) {          // constant value

    ScalarType *st = context.getScalarType(numeric_type_id);
    constants.push_back(std::unique_ptr<ir::ConstantScalar>(new ir::ConstantScalar(const_numeric_value, st, arg.name, nullptr)));
    return Status::kSuccess;
  }

  /// Insert ConstantEnum values in Function::constants
  template <typename T> 
  Status insert_constant_enum(
    Context & context,                // context object
    FunctionSchema::Argument & arg,   // corresponding schema argument
    EnumeratedTypeID emum_type_id,    // EnumeratedType's element type
    T const_enum_value) {             // constant value

    EnumeratedType *et = context.getEnumeratedType(emum_type_id);
    constants.push_back(std::unique_ptr<ir::ConstantEnum>(new ir::ConstantEnum(const_enum_value, et, arg.name, nullptr)));
    return Status::kSuccess;
  }

 /// Insert a StrcutureValue with all constant values of u32 types (e.g. cta-tile-size, warp-tile-size)
 Status insert_constant_structure(
  Context & context,                    // context object
  FunctionSchema::Argument & arg,       // corresponding schema argument
  std::vector<uint32_t> const &const_values); // vector of constant values

  /// Prints function definition to otstream
  std::ostream & print(std::ostream &out) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Abstract base class for device functions.
struct DeviceFunction {

  /// Function definition
  Function *function;

  //
  // Methods
  //

  /// Default constructor
  DeviceFunction(Function *function_ = nullptr);

  /// Virtual destructor
  virtual ~DeviceFunction();

  /// Returns success if a device function can implement the solution to the given problem.
  virtual Status can_implement(void const *arguments) const = 0;

  /// Gets the host-side workspace size in bytes given a pointer to the argument structure
  virtual Status get_host_workspace_size(size_t *size, void const *arguments) const = 0;

  /// Gets the device-side workspace size in bytes given a pointer to the argument structure
  virtual Status get_device_workspace_size(size_t *size, void const *arguments) const = 0;

  /// Initializes the host-side workspace given the kernel argument structure
  virtual Status initialize_workspace(
    void const *arguments,
    void *host_workspace = nullptr, 
    void *device_workspace = nullptr) const = 0;

  /// Updates the workspace
  virtual Status update_workspace(
    void const *arguments,
    void *host_workspace = nullptr,
    void *device_workspace = nullptr) const;

  /// Runs the kernel
  virtual Status run(
    void const *arguments,
    void *host_workspace = nullptr,
    void *device_workspace = nullptr) const = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ir
} // namespace cutlass
