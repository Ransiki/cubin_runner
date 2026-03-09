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
  \file Extracts metadata and implementation details of kernels.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <sstream>
#include <fstream>

#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/library/types.h>
#include <cutlass/library/descriptions.h>
#include <cutlass/library/payloads.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/arch/arch.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace emitter {
// All helper functions to help emit types and data structures as strings.

#define STRINGIFY_INDIRECTION(x) #x
#define STRINGIFY(x) STRINGIFY_INDIRECTION(x)

#define ENUMERANT(enumerated_type, enumerant)                   \
  case enumerated_type :: enumerant:                            \
    return STRINGIFY(enumerated_type) "::" STRINGIFY(enumerant)

static const char* to_string(LayoutTypeID const& layout_type) {
  switch(layout_type) {
    ENUMERANT(LayoutTypeID, kUnknown);
    ENUMERANT(LayoutTypeID, kColumnMajor);
    ENUMERANT(LayoutTypeID, kRowMajor);
    ENUMERANT(LayoutTypeID, kBlockScalingTensor);           // {$nv-internal-release}
    ENUMERANT(LayoutTypeID, kColumnMajorInterleavedK2);
    ENUMERANT(LayoutTypeID, kRowMajorInterleavedK2);
    ENUMERANT(LayoutTypeID, kColumnMajorInterleavedK4);
    ENUMERANT(LayoutTypeID, kRowMajorInterleavedK4);
    ENUMERANT(LayoutTypeID, kColumnMajorInterleavedK16);
    ENUMERANT(LayoutTypeID, kRowMajorInterleavedK16);
    ENUMERANT(LayoutTypeID, kColumnMajorInterleavedK32);
    ENUMERANT(LayoutTypeID, kRowMajorInterleavedK32);
    ENUMERANT(LayoutTypeID, kColumnMajorInterleavedK64);
    ENUMERANT(LayoutTypeID, kRowMajorInterleavedK64);
    ENUMERANT(LayoutTypeID, kTensorNCHW);
    ENUMERANT(LayoutTypeID, kTensorNCDHW);
    ENUMERANT(LayoutTypeID, kTensorNHWC);
    ENUMERANT(LayoutTypeID, kTensorNDHWC);
    ENUMERANT(LayoutTypeID, kTensorNC32HW32);
    ENUMERANT(LayoutTypeID, kTensorC32RSK32);
    ENUMERANT(LayoutTypeID, kTensorNC64HW64);
    ENUMERANT(LayoutTypeID, kTensorC64RSK64);
    ENUMERANT(LayoutTypeID, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}


static inline const char* to_string(NumericTypeID const&  numeric_type) {
  switch(numeric_type) {
    ENUMERANT(NumericTypeID, kUnknown);
    ENUMERANT(NumericTypeID, kVoid);
    ENUMERANT(NumericTypeID, kB1);
    ENUMERANT(NumericTypeID, kU2);
    ENUMERANT(NumericTypeID, kU4);
    ENUMERANT(NumericTypeID, kU8);
    ENUMERANT(NumericTypeID, kU16);
    ENUMERANT(NumericTypeID, kU32);
    ENUMERANT(NumericTypeID, kU64);
    ENUMERANT(NumericTypeID, kS2);
    ENUMERANT(NumericTypeID, kS4);
    ENUMERANT(NumericTypeID, kS8);
    ENUMERANT(NumericTypeID, kS16);
    ENUMERANT(NumericTypeID, kS32);
    ENUMERANT(NumericTypeID, kS64);
    ENUMERANT(NumericTypeID, kFE4M3);
    ENUMERANT(NumericTypeID, kFE5M2);
// {$nv-internal-release begin}
    ENUMERANT(NumericTypeID, kFE3M4);
    ENUMERANT(NumericTypeID, kFE2M3);
    ENUMERANT(NumericTypeID, kFE3M2);
    ENUMERANT(NumericTypeID, kFE2M1);
    ENUMERANT(NumericTypeID, kFE0M3);
    ENUMERANT(NumericTypeID, kFUE8M0);
    ENUMERANT(NumericTypeID, kFUE4M3);
    ENUMERANT(NumericTypeID, kMXINT8);
    ENUMERANT(NumericTypeID, kF8);
    ENUMERANT(NumericTypeID, kF6);
    ENUMERANT(NumericTypeID, kF4);
// {$nv-internal-release end}
    ENUMERANT(NumericTypeID, kF16);
    ENUMERANT(NumericTypeID, kBF16);
    ENUMERANT(NumericTypeID, kTF32);
    ENUMERANT(NumericTypeID, kF32);
    ENUMERANT(NumericTypeID, kF64);
    ENUMERANT(NumericTypeID, kCF16);
    ENUMERANT(NumericTypeID, kCBF16);
    ENUMERANT(NumericTypeID, kCF32);
    ENUMERANT(NumericTypeID, kCTF32);
    ENUMERANT(NumericTypeID, kCF64);
    ENUMERANT(NumericTypeID, kCS2);
    ENUMERANT(NumericTypeID, kCS4);
    ENUMERANT(NumericTypeID, kCS8);
    ENUMERANT(NumericTypeID, kCS16);
    ENUMERANT(NumericTypeID, kCS32);
    ENUMERANT(NumericTypeID, kCS64);
    ENUMERANT(NumericTypeID, kCU2);
    ENUMERANT(NumericTypeID, kCU4);
    ENUMERANT(NumericTypeID, kCU8);
    ENUMERANT(NumericTypeID, kCU16);
    ENUMERANT(NumericTypeID, kCU32);
    ENUMERANT(NumericTypeID, kCU64);
    ENUMERANT(NumericTypeID, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(ComplexTransform const&  complex_transform) {
  switch(complex_transform) {
    ENUMERANT(ComplexTransform, kNone);
    ENUMERANT(ComplexTransform, kConjugate);
    ENUMERANT(ComplexTransform, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(Provider const&  provider) {
  switch(provider) {
    ENUMERANT(Provider, kNone);
    ENUMERANT(Provider, kCUTLASS);
    ENUMERANT(Provider, kReferenceHost);
    ENUMERANT(Provider, kReferenceDevice);
    ENUMERANT(Provider, kCUBLAS);
    ENUMERANT(Provider, kCUDNN);
    ENUMERANT(Provider, kCUSPARSE);          // {$nv-internal-release}
    ENUMERANT(Provider, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(OperationKind const&  op_kind) {
  switch(op_kind) {
    ENUMERANT(OperationKind, kGemm);
    ENUMERANT(OperationKind, kBlockScaledGemm); 
    ENUMERANT(OperationKind, kBlockwiseGemm); 
    ENUMERANT(OperationKind, kRankK);
    ENUMERANT(OperationKind, kRank2K);
    ENUMERANT(OperationKind, kTrmm);
    ENUMERANT(OperationKind, kSymm);
    ENUMERANT(OperationKind, kConv2d);
    ENUMERANT(OperationKind, kConv3d);
    ENUMERANT(OperationKind, kEqGemm);
    ENUMERANT(OperationKind, kSparseGemm);
    ENUMERANT(OperationKind, kReduction);
    ENUMERANT(OperationKind, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(ScalarPointerMode const&  scalar_ptr_mode) {
  switch(scalar_ptr_mode) {
    ENUMERANT(ScalarPointerMode, kHost);
    ENUMERANT(ScalarPointerMode, kDevice);
    ENUMERANT(ScalarPointerMode, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(SplitKMode const&  splitk_mode) {
  switch(splitk_mode) {
    ENUMERANT(SplitKMode, kNone);
    ENUMERANT(SplitKMode, kSerial);
    ENUMERANT(SplitKMode, kParallel);
    ENUMERANT(SplitKMode, kParallelSerial);
    ENUMERANT(SplitKMode, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(OpcodeClassID const&  opcode_class) {
  switch(opcode_class) {
    ENUMERANT(OpcodeClassID, kSimt);
    ENUMERANT(OpcodeClassID, kTensorOp);
    ENUMERANT(OpcodeClassID, kWmmaTensorOp);
    ENUMERANT(OpcodeClassID, kSparseTensorOp);
    ENUMERANT(OpcodeClassID, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(MathOperationID const&  math_op) {
  switch(math_op) {
    ENUMERANT(MathOperationID, kAdd);
    ENUMERANT(MathOperationID, kMultiplyAdd);
    ENUMERANT(MathOperationID, kMultiplyAddSaturate);
    ENUMERANT(MathOperationID, kMultiplyAddFastBF16);
    ENUMERANT(MathOperationID, kMultiplyAddFastF16);
    ENUMERANT(MathOperationID, kMultiplyAddFastF32);
    ENUMERANT(MathOperationID, kMultiplyAddComplex);
    ENUMERANT(MathOperationID, kMultiplyAddComplexFastF32);
    ENUMERANT(MathOperationID, kMultiplyAddGaussianComplex);
    ENUMERANT(MathOperationID, kXorPopc);
    ENUMERANT(MathOperationID, kAndPopc);  // {$nv-internal-release}
    ENUMERANT(MathOperationID, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(GemmKind const&  gemm_kind) {
  switch(gemm_kind) {
    ENUMERANT(GemmKind, kGemm);
    ENUMERANT(GemmKind, kSparse);
    ENUMERANT(GemmKind, kUniversal);
    ENUMERANT(GemmKind, kPlanarComplex);
    ENUMERANT(GemmKind, kPlanarComplexArray);
    ENUMERANT(GemmKind, kGrouped);
    ENUMERANT(GemmKind, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
// CUtensorMap related enums are defined since cuda toolkits >= 12
static const char* to_string(CUtensorMapL2promotion const& l2_promotion) {
  switch(l2_promotion) {
    ENUMERANT(CUtensorMapL2promotion, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    ENUMERANT(CUtensorMapL2promotion, CU_TENSOR_MAP_L2_PROMOTION_L2_64B);
    ENUMERANT(CUtensorMapL2promotion, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    ENUMERANT(CUtensorMapL2promotion, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(CUtensorMapFloatOOBfill const& oob_fill) {
  switch(oob_fill) {
    ENUMERANT(CUtensorMapFloatOOBfill, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    ENUMERANT(CUtensorMapFloatOOBfill, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(CUtensorMapSwizzle const& smem_swizzle) {
  switch(smem_swizzle) {
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_NONE);
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_32B);
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_64B);
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_128B);
#if defined(CUDA_BLACKWELL_TMA_SWIZZLE_ENABLED) && CUDA_BLACKWELL_TMA_SWIZZLE_ENABLED
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B);
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B);
    ENUMERANT(CUtensorMapSwizzle, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B);
#endif // CUDA_BLACKWELL_TMA_SWIZZLING_ENABLED
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

#else

static std::string to_string(int32_t const& enum_value) {
  return std::to_string(enum_value);
}

#endif // // CUDA_VERSION >= 12000

static const char* to_string(CopyEngineID const& copy_engine) {
  switch(copy_engine) {
    ENUMERANT(CopyEngineID, kDereference);
    ENUMERANT(CopyEngineID, kCopyAsync);
    ENUMERANT(CopyEngineID, kTma);
    ENUMERANT(CopyEngineID, kInvalid);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(MainloopScheduleType const& mainloop_schedule_type) {
  switch(mainloop_schedule_type) {
      ENUMERANT(MainloopScheduleType, kMainloopScheduleAuto);
      ENUMERANT(MainloopScheduleType, kTmaWarpSpecialized1SmSm100);
      ENUMERANT(MainloopScheduleType, kTmaWarpSpecialized2SmSm100);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(EpilogueScheduleType const& epilogue_schedule_type) {
  switch(epilogue_schedule_type) {
    ENUMERANT(EpilogueScheduleType, kEpilogueScheduleAuto);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}

static const char* to_string(EpilogueTileType const& epilogue_tile_type) {
  switch(epilogue_tile_type) {
    ENUMERANT(EpilogueTileType, kEpilogueTileAuto);
    default:
      assert(false && "An unexpected enumerant encountered.");
      return "<invalid>";
  }
}


#undef STRINGIFY_INDIRECTION
#undef STRINGIFY
#undef ENUMERANT

#define ENUMERANT(enumerated_type, member, str) case enumerated_type :: member : return str

static char const *to_ptx_string(NumericTypeID id) {
  switch (id) {
    ENUMERANT(NumericTypeID, kVoid, "void");
    ENUMERANT(NumericTypeID, kB1, "b1");
    ENUMERANT(NumericTypeID, kU2, "u2");
    ENUMERANT(NumericTypeID, kU4, "u4");
    ENUMERANT(NumericTypeID, kU8, "u8");
    ENUMERANT(NumericTypeID, kU16, "u16");
    ENUMERANT(NumericTypeID, kU32, "u32");
    ENUMERANT(NumericTypeID, kU64, "u64");
    ENUMERANT(NumericTypeID, kS2, "s2");
    ENUMERANT(NumericTypeID, kS4, "s4");
    ENUMERANT(NumericTypeID, kS8, "s8");
    ENUMERANT(NumericTypeID, kS16, "s16");
    ENUMERANT(NumericTypeID, kS32, "s32");
    ENUMERANT(NumericTypeID, kS64, "s64");
    ENUMERANT(NumericTypeID, kFE4M3, "fe4m3");
    ENUMERANT(NumericTypeID, kFE5M2, "fe5m2");
    ENUMERANT(NumericTypeID, kF16, "f16");
    ENUMERANT(NumericTypeID, kBF16, "bf16");
    ENUMERANT(NumericTypeID, kTF32, "tf32");
    ENUMERANT(NumericTypeID, kF32, "f32");
    ENUMERANT(NumericTypeID, kF64, "f64");
    default:
      break;
  }
  return "Invalid";
}

#undef ENUMERANT

/////////////////////////////////////////////////////////////////////////////////////////////////

static std::string to_string(
  cutlass::gemm::GemmCoord const&  coord) {
  std::stringstream ss;
  ss << "{ ";
  for (int i = 0; i < coord.kRank; ++i) {
    ss << coord[i] << (i == coord.kRank - 1 ? "" : ", ");
  }
  ss << " }";
  return ss.str();
}

static std::string indent(int indent_lvl) {
  return std::string(2 * indent_lvl, ' ');
}

static std::string to_string(
  MathInstructionDescription const& math_inst_desc, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "MathInstructionDescription {\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(math_inst_desc.instruction_shape) << ", // instruction_shape\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(math_inst_desc.element_accumulator) << ", // element_accumulator\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(math_inst_desc.opcode_class) << ", // opcode_class\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(math_inst_desc.math_operation) <<  " // math_operation\n";
  ss << indent(indent_lvl) << "}";

  return ss.str();
}

static std::string to_string(TileDescription const& tile_desc, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "TileDescription {\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tile_desc.threadblock_shape) << ", // threadblock_shape\n";
  ss << indent(indent_lvl + 1) <<  tile_desc.threadblock_stages << ", // threadblock_stages\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tile_desc.warp_count) << ", // warp_count\n";
  ss << cutlass::library::emitter::to_string(tile_desc.math_instruction, indent_lvl + 1) << ",\n";
  ss << indent(indent_lvl + 1) <<  tile_desc.minimum_compute_capability << ", // minimum_compute_capability\n";
  ss << indent(indent_lvl + 1) <<  tile_desc.maximum_compute_capability << ", // maximum_compute_capability\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tile_desc.cluster_shape) << ", // cluster_shape\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(OperationDescription const& op_desc, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "OperationDescription {\n";
  ss << indent(indent_lvl + 1) << "\"" << op_desc.name << "\",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(op_desc.provider) << ",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(op_desc.kind) << ",\n";
  ss << cutlass::library::emitter::to_string(op_desc.tile_description, indent_lvl + 1) << "\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(TensorDescription const& tensor_desc, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "TensorDescription {\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(tensor_desc.element) << ",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(tensor_desc.layout) << ",\n";
  ss << indent(indent_lvl + 1) << tensor_desc.alignment << ", // alignment\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(GemmDescription const& gemm_desc, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "GemmDescription {\n";
  ss << cutlass::library::emitter::to_string(*reinterpret_cast<OperationDescription const*>(&gemm_desc), indent_lvl + 1) << ",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_desc.gemm_kind) << ",\n";
  ss << cutlass::library::emitter::to_string(gemm_desc.A, indent_lvl + 1) << ", // A\n";
  ss << cutlass::library::emitter::to_string(gemm_desc.B, indent_lvl + 1) << ", // B\n";
  ss << cutlass::library::emitter::to_string(gemm_desc.C, indent_lvl + 1) << ", // C\n";
  ss << cutlass::library::emitter::to_string(gemm_desc.D, indent_lvl + 1) << ", // D\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_desc.element_epilogue) << ", // element_epilogue\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_desc.split_k_mode) << ",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_desc.transform_A) << ", // transform_A\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_desc.transform_B) << ", // transform_B\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(std::array<int32_t, 5> const& arr) {
  std::stringstream ss;
  ss << " { ";
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (it != arr.begin()) {
      ss << ", ";
    }
    ss << *it;
  }
  ss << " }";
  return ss.str();
}

static std::string to_string(LaunchConfigurationPayload const& launch_config, int indent_lvl = 0) {
  const char* ns = "cutlass::library::";
  std::stringstream ss;
  ss << indent(indent_lvl) << ns << "LaunchConfigurationPayload {\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(launch_config.block_dim) << ", // block_dim\n";
  ss << indent(indent_lvl + 1) << launch_config.dynamic_shared_memory << ", // dynamic_shared_memory\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(TmaOperandPayload const&  tma_op, int indent_lvl = 0) {
  const char* ns = "cutlass::library::";
  std::stringstream ss;
  ss << indent(indent_lvl) << ns << "TmaOperandPayload {\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tma_op.box_size) << ", // box_size\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tma_op.box_stride) << ", // box_stride\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tma_op.basis_permutation) << ", // basis_permutation\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tma_op.l2_promotion) << ",\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tma_op.oob_fill) << ",\n";
  ss << indent(indent_lvl + 1) << cutlass::library::emitter::to_string(tma_op.smem_swizzle) << ",\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(TensorOperandPayload const& tensor_op, int indent_lvl = 0) {
  const char* ns = "cutlass::library::";
  std::stringstream ss;
  ss << indent(indent_lvl) << ns << "TensorOperandPayload {\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(tensor_op.copy_engine) << ",\n";
  if (tensor_op.copy_engine == CopyEngineID::kTma) {
    ss << cutlass::library::emitter::to_string(tensor_op.tma, indent_lvl + 1) << ",\n";
  }
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

static std::string to_string(GemmPayload const& gemm_payload, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "GemmPayload {\n";
  ss << cutlass::library::emitter::to_string(gemm_payload.launch_config, indent_lvl + 1) << ",\n";
  ss << cutlass::library::emitter::to_string(gemm_payload.A, indent_lvl + 1) << ", // A\n";
  ss << cutlass::library::emitter::to_string(gemm_payload.B, indent_lvl + 1) << ", // B\n";
  ss << cutlass::library::emitter::to_string(gemm_payload.C, indent_lvl + 1) << ", // C\n";
  ss << cutlass::library::emitter::to_string(gemm_payload.D, indent_lvl + 1) << ", // D\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_payload.mainloop_schedule_type) << ",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_payload.epilogue_schedule_type) << ",\n";
  ss << indent(indent_lvl + 1) << ns << cutlass::library::emitter::to_string(gemm_payload.epilogue_tile_type) << ",\n";
  ss << indent(indent_lvl) <<  "}";
  return ss.str();
}

static std::string to_string(GemmInstance const& gemm_instance, int indent_lvl = 0) {
  std::stringstream ss;
  const char* ns = "cutlass::library::";
  ss << indent(indent_lvl) << ns << "GemmInstance {\n";
  ss << cutlass::library::emitter::to_string(gemm_instance.description, indent_lvl + 1) << ",\n";
  ss << cutlass::library::emitter::to_string(gemm_instance.payload, indent_lvl + 1) << ",\n";
  ss << indent(indent_lvl) << "}";
  return ss.str();
}

} // namespace emitter

/////////////////////////////////////////////////////////////////////////////////////////////////

class MetadataEmitter {
private:

#if defined (CUDA_VERSION) && CUDA_VERSION >= 12000

  template <typename Bitfield>
  int32_t DecodeTmaExtent(Bitfield bitfield) const {
    return int32_t(static_cast<uint8_t>(bitfield) + 1);
  }

  template <
    typename Bitfield0,
    typename Bitfield1,
    typename Bitfield2,
    typename Bitfield3,
    typename Bitfield4
  >
  std::array<int32_t, 5> DecodeTmaExtents(
    Bitfield0 _0,
    Bitfield1 _1,
    Bitfield2 _2,
    Bitfield3 _3,
    Bitfield4 _4) const {

    return std::array<int32_t, 5>{
      DecodeTmaExtent(_0),
      DecodeTmaExtent(_1),
      DecodeTmaExtent(_2),
      DecodeTmaExtent(_3),
      DecodeTmaExtent(_4)
    };
  }

  CUtensorMapSwizzle DecodeTmaSharedMemorySwizzle(cute::TmaDescriptorInternal const&  tma_desc) const {
    uint8_t swizzle_base = uint8_t(tma_desc.swizzle_base_);
    uint8_t swizzle_bits = uint8_t(tma_desc.swizzle_bits_);
    if (swizzle_base > 0 && swizzle_bits != 3) {
      // A non-zero swizzle base is only defined with 128B swizzle bits
      assert(false && "Unexpected swizzle base and atom encountered.");
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
    switch (swizzle_base + swizzle_bits) {
      case 0: return CU_TENSOR_MAP_SWIZZLE_NONE;
      case 1: return CU_TENSOR_MAP_SWIZZLE_32B;
      case 2: return CU_TENSOR_MAP_SWIZZLE_64B;
      case 3: return CU_TENSOR_MAP_SWIZZLE_128B;
#if defined(CUDA_BLACKWELL_TMA_SWIZZLE_ENABLED) && CUDA_BLACKWELL_TMA_SWIZZLE_ENABLED
      case 4: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
      case 5: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B;
      case 6: return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
#endif // CUDA_BLACKWELL_TMA_SWIZZLING_ENABLED
      default:
        assert(false && "Unexpected swizzle pattern encountered.");
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
  }

  template <typename TensorOperand>
  TensorOperandPayload toTensorOperandPayload(TensorOperand const& tensor_op) const {
    // TMA descriptor
    cute::TmaDescriptorInternal const&  tma_desc = *tensor_op.get_tma_descriptor();

    //
    // Emit the TensorOperandPayload
    //

    CUtensorMapSwizzle smem_swizzle = DecodeTmaSharedMemorySwizzle(tma_desc);

    std::array<int32_t, 5> box_size = DecodeTmaExtents(
      tma_desc.bsize0_,
      tma_desc.bsize1_,
      tma_desc.bsize2_,
      tma_desc.bsize3_,
      tma_desc.bsize4_
    );

    std::array<int32_t, 5> box_stride = DecodeTmaExtents(
      tma_desc.tstride0_,
      tma_desc.tstride1_,
      tma_desc.tstride2_,
      tma_desc.tstride3_,
      tma_desc.tstride4_
    );

    // TODO: Replace the hard-coded values below
    std::array<int32_t, 5> basis_permutation = {0, 1, 2, 3, 4};

    return TensorOperandPayload {
      CopyEngineID::kTma,
      TmaOperandPayload {
        box_size,
        box_stride,
        basis_permutation,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,    // TODO: replace the hard-coded value
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,  // TODO: replace the hard-coded value
        smem_swizzle
      }
    };
  }

  template<class GemmUniversal3xOp>
  MainloopScheduleType toMainloopScheduleType(GemmUniversal3xOp const& op) const {
    if (cute::size<0>(
        typename GemmUniversal3xOp::GemmKernel::CollectiveMainloop::TiledMma::ThrLayoutVMNK{}) == 2) {
        return MainloopScheduleType::kTmaWarpSpecialized2SmSm100;
    } else {
      return MainloopScheduleType::kTmaWarpSpecialized1SmSm100;
    }
  }

  template<class GemmUniversal3xOp>
  int toArchTagNumber(GemmDescription const& gemm_desc, GemmUniversal3xOp const& op) const {
    // TODO: fix this, special case, for handling SM100 SGEMM kernels (which use SM80's DispatchPolicy)
    if (std::is_same<
          typename GemmUniversal3xOp::GemmKernel::ArchTag,
          cutlass::arch::Sm80>::value &&
          GemmUniversal3xOp::InstructionShape::kMNK == 2) {
      return 100;
    }
    return gemm_desc.tile_description.minimum_compute_capability;
  }

  template<class GemmUniversal3xOp>
  GemmInstance extractInstance(GemmDescription const& gemm_desc, GemmUniversal3xOp const& op) const {

    const int32_t thread_count = GemmUniversal3xOp::GemmKernel::MaxThreadsPerBlock;
    const int32_t dynamic_shared_memory =
      static_cast<int32_t>(GemmUniversal3xOp::GemmKernel::SharedStorageSize);

    // CTA launch configurations:
    const auto cta_launch_config = cutlass::library::LaunchConfigurationPayload {
      cutlass::gemm::GemmCoord(thread_count, 1, 1),
      dynamic_shared_memory
    };

  TensorOperandPayload tensor_A_payload(CopyEngineID::kCopyAsync);
  TensorOperandPayload tensor_B_payload(CopyEngineID::kCopyAsync);
  TensorOperandPayload tensor_C_payload(CopyEngineID::kCopyAsync);
  TensorOperandPayload tensor_D_payload(CopyEngineID::kCopyAsync);
  if constexpr (
    cutlass::detail::is_tma_copy_engine<
      typename GemmUniversal3xOp::GemmKernel::CollectiveMainloop::GmemTiledCopyA>()) {
    tensor_A_payload = toTensorOperandPayload(op.params().mainloop.tma_load_a);
  }
  if constexpr (
    cutlass::detail::is_tma_copy_engine<
      typename GemmUniversal3xOp::GemmKernel::CollectiveMainloop::GmemTiledCopyB>()) {
    tensor_B_payload = toTensorOperandPayload(op.params().mainloop.tma_load_b);
  }

  if constexpr (
    cutlass::detail::is_tma_copy_engine<
      typename GemmUniversal3xOp::GemmKernel::CollectiveEpilogue::GmemTiledCopyC>()) {
    tensor_C_payload = toTensorOperandPayload(op.params().epilogue.tma_load_c);
  }
  if constexpr (
    cutlass::detail::is_tma_copy_engine<
      typename GemmUniversal3xOp::GemmKernel::CollectiveEpilogue::GmemTiledCopyD>()) {
    tensor_D_payload = toTensorOperandPayload(op.params().epilogue.tma_store_d);
  }

    const auto mainloop_schedule_type = toMainloopScheduleType(op);

    const auto payload = GemmPayload {
      cta_launch_config,
      tensor_A_payload,
      tensor_B_payload,
      tensor_C_payload,
      tensor_D_payload,
      mainloop_schedule_type,
      EpilogueScheduleType::kEpilogueScheduleAuto,
      EpilogueTileType::kEpilogueTileAuto
    };

    return GemmInstance {
      gemm_desc,
      payload
    };
  }
#else

  template<class GemmUniversal3xOp>
  GemmInstance extractInstance(GemmDescription const& gemm_desc, GemmUniversal3xOp const& op) const {
    assert(false && "Metadata extraction is supposed to be used with CUDA toolkit >= 12.0");
    return GemmInstance {gemm_desc, {}};
  }

#endif // CUDA_VERSION >= 12000
public:

  // Extracts all required metadata information which can be used for
  // definition and instantiation of device kernels and host-side initialization of them.
  // The result is then extracted into some files.
  template<class GemmUniversal3xOp>
  void emit(
    GemmDescription const& gemm_desc,
    GemmUniversal3xOp const& op) const {

    const auto gemm_instance = extractInstance(gemm_desc, op);

    std::string output_dir = CUTLASS_EMIT_KERNEL_METADATA_DIR;

    if (output_dir.empty()) {
      assert(false && "Invalid directory at CUTLASS_EMIT_KERNEL_METADATA_DIR");
      return;
    }

    // Name of file which is going to be used for metadata extraction:
    std::stringstream ss;
    ss  << output_dir
        << "/gemm_"
        << "sm" << toArchTagNumber(gemm_desc, op) << "_"
        << (gemm_desc.tile_description.math_instruction.opcode_class == OpcodeClassID::kSimt ?
            "simt" : "tensorop") << "_"
        << emitter::to_ptx_string(gemm_desc.A.element) << "_"
        << emitter::to_ptx_string(gemm_desc.B.element) << "_"
        << emitter::to_ptx_string(gemm_desc.tile_description.math_instruction.element_accumulator) << "_"
        << emitter::to_ptx_string(gemm_desc.C.element) << "_"
        << emitter::to_ptx_string(gemm_desc.D.element)
        << ".inl";

    // New metadata is appended in the end of each file name
    std::ofstream file(ss.str().c_str(), std::ios_base::app);

    // Appending the extracted information as a string into a file.
    if (file.is_open()) {
      file << emitter::to_string(gemm_instance) << ",\n";
    }
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
