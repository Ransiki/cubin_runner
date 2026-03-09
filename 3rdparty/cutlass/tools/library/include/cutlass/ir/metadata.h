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
#include <memory>
#include <iostream>


namespace cutlass {
namespace ir {

//===--------------------------------------------------------------------===//
/// Definitions of all of the base types for the Type system.
enum class MDKind {
  // Type definitions in the base class
  kRange,
  kAlign,
  kDiv,
  kConst,
  kExpr,
  kInvalid
};

struct MDNode {
  //
  // Data Members
  //
  MDKind md_kind;          // metadata kind info

  //
  // Methods
  //
  // Constructors
  MDNode(MDKind md_kind_) : md_kind(md_kind_) {}

  // Destructor
  virtual ~MDNode() = default; // make MDNode class polymorphic

};

// Create MDRange metadata to specify range [low, high, inc] restrictions
struct MDRange : public MDNode {
  //
  // Data Members
  //
  int64_t low, high, inc;

  //
  // Methods
  //
  // ctor
  MDRange(int64_t low_, int64_t high_, int64_t inc_ = 1) : low(low_), high(high_), inc(inc_), MDNode(MDKind::kRange) {}
};

// Create MDAlign metadata to specify pointer address' alignment restrictions
struct MDAlignment : public MDNode {
  //
  // Data Members
  //
  int32_t align;

  //
  // Methods
  //
  // ctor
  MDAlignment(int32_t align_) : align(align_), MDNode(MDKind::kAlign) {}
};

// Create MDDiv metadata to specify value's divisibility restrictions
struct MDDivisible : public MDNode {
  //
  // Data Members
  //
  int32_t div;

  //
  // Methods
  //
  // ctor
  MDDivisible(int32_t div_) : div(div_), MDNode(MDKind::kDiv) {}
};

// Create MDConst metadata to specify constant metadata
struct MDConst : public MDNode {
  //
  // Data Members
  //
  bool is_const;

  //
  // Methods
  //
  // ctor
  MDConst(bool is_const_) : is_const(is_const_), MDNode(MDKind::kConst) {}
};

// Create MDExpr metadata to specify expression in between *two* function arguments
struct MDExpr : public MDNode {
  //
  // Data Members
  //
  // TODO: Create expression metadata

  //
  // Methods
  //
  // ctor
  MDExpr() : MDNode(MDKind::kExpr) {throw std::runtime_error("MDExpr metadata is not ready for use!!!");}
};

/// operator overloading function declarations
std::ostream& operator<<(std::ostream& stream, const MDRange& val);
std::ostream& operator<<(std::ostream& stream, const MDAlignment& val);
std::ostream& operator<<(std::ostream& stream, const MDDivisible& val);
std::ostream& operator<<(std::ostream& stream, const MDConst& val);
std::ostream& operator<<(std::ostream& stream, const MDExpr& val);


} // namespace ir
} // namespace cutlass
