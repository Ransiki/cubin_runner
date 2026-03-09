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
/* \file
   \brief Implementation of an interface onto the kernel timing model

   {$nv-internal-release file}
*/

#ifdef CUTLASS_ENABLE_KERNEL_TIMING_MODEL
#define KTM_NAMESPACE cutlass::profiler::ktm
#include "ktm.h"
#undef KTM_NAMESPACE
#endif

#include "cutlass/profiler/kernel_timing_model.h"
#include "cutlass/profiler/gemm_operation_profiler.h"
#include "cutlass/profiler/conv2d_operation_profiler.h"
#include "cutlass/profiler/conv3d_operation_profiler.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {
namespace ktm {

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_ENABLE_KERNEL_TIMING_MODEL

///////////////////////////////////////////////////////////////////////////////////////////////////

MathType scalar_type(
  library::NumericTypeID numericTypeID) {

  switch (numericTypeID) {
    case library::NumericTypeID::kU8:
    case library::NumericTypeID::kS8:
    case library::NumericTypeID::kCS8:
    case library::NumericTypeID::kCU8:
      return MathType::INT8;
    case library::NumericTypeID::kFE4M3:
      return MathType::FP8;
    case library::NumericTypeID::kFE5M2:
      return MathType::FP8;
    case library::NumericTypeID::kBF16:
    case library::NumericTypeID::kCBF16:
      return MathType::BF16;
    case library::NumericTypeID::kF16:
    case library::NumericTypeID::kCF16:
      return MathType::FP16;
    case library::NumericTypeID::kF32:
    case library::NumericTypeID::kCF32:
      return MathType::FP32;
    case library::NumericTypeID::kTF32:
    case library::NumericTypeID::kCTF32:
      return MathType::TF32;
    case library::NumericTypeID::kCS4:
    case library::NumericTypeID::kCU4:
      return MathType::INT8;
    case library::NumericTypeID::kU16:
    case library::NumericTypeID::kS16:
    case library::NumericTypeID::kCS16:
    case library::NumericTypeID::kCU16:
      return MathType::FP16; // Size correct, math throughput wrong.
    case library::NumericTypeID::kU32:
    case library::NumericTypeID::kS32:
    case library::NumericTypeID::kCS32:
    case library::NumericTypeID::kCU32:
      return MathType::FP32; // Size correct, math throughput wrong.
    case library::NumericTypeID::kU64:
    case library::NumericTypeID::kS64:
    case library::NumericTypeID::kF64:
    case library::NumericTypeID::kCF64:
    case library::NumericTypeID::kCS64:
    case library::NumericTypeID::kCU64:
      return MathType::FP32; // Size wrong, math throughput wrong.
    case library::NumericTypeID::kB1:
    case library::NumericTypeID::kU2:
    case library::NumericTypeID::kU4:
    case library::NumericTypeID::kS2:
    case library::NumericTypeID::kS4:
    case library::NumericTypeID::kCS2:
    case library::NumericTypeID::kCU2:
    case library::NumericTypeID::kUnknown:
    case library::NumericTypeID::kVoid:
    case library::NumericTypeID::kInvalid:
    default:
      return MathType::INT8; // Size wrong, math throughput wrong.
  }
}

int32_t size_in_bytes_of_type(MathType type) {
  switch (type) {
    case MathType::FP8:
    case MathType::INT8:
      return 1;
    case MathType::BF16:
    case MathType::FP16:
      return 2;
    case MathType::FP32:
    case MathType::TF32:
      return 4;
  }
  return 0;
}

bool is_complex(library::NumericTypeID numericTypeID) {

  switch (numericTypeID) {
    case library::NumericTypeID::kCS4:
    case library::NumericTypeID::kCU4:
    case library::NumericTypeID::kCS8:
    case library::NumericTypeID::kCU8:
    case library::NumericTypeID::kCF16:
    case library::NumericTypeID::kCBF16:
    case library::NumericTypeID::kCS16:
    case library::NumericTypeID::kCU16:
    case library::NumericTypeID::kCF32:
    case library::NumericTypeID::kCTF32:
    case library::NumericTypeID::kCS32:
    case library::NumericTypeID::kCU32:
    case library::NumericTypeID::kCF64:
    case library::NumericTypeID::kCS64:
    case library::NumericTypeID::kCU64:
    case library::NumericTypeID::kCS2:
    case library::NumericTypeID::kCU2:
      return true;
  }

  return false;
}

bool is_64_bit(library::NumericTypeID numericTypeID) {

  switch (numericTypeID) {
    case library::NumericTypeID::kU64:
    case library::NumericTypeID::kS64:
    case library::NumericTypeID::kF64:
    case library::NumericTypeID::kCF64:
    case library::NumericTypeID::kCS64:
    case library::NumericTypeID::kCU64:
      return true;
  }

  return false;
}

void finishGemmKernelProperties(
  library::TensorDescription const &tensor_desc,
  GemmKernelProperties &gemmKernelProperties) {

  if (is_64_bit(tensor_desc.element)) {
    if (gemmKernelProperties.mmaShape != MmaShape::SHAPE_111) {
      gemmKernelProperties.mathSpeed /= 4;
    }
    gemmKernelProperties.sizePerElementA *= 2;
    gemmKernelProperties.sizePerElementC *= 2;
    gemmKernelProperties.sizePerElementACC *= 2;
  }

  if (is_complex(tensor_desc.element)) {
    gemmKernelProperties.mathSpeed /= 4;
    gemmKernelProperties.sizePerElementA *= 2;
    gemmKernelProperties.sizePerElementC *= 2;
    gemmKernelProperties.sizePerElementACC *= 2;
  }
}

KernelType kernel_type(
  library::OperationDescription const &operation_desc) {

  std::string name(operation_desc.name);

  if (name.find("wo_smem") != std::string::npos) {
    return KernelType::CONV_WITHOUT_SMEM;
  }

  if (name.find("gemm") != std::string::npos) {
    return KernelType::GEMM;
  }

  return KernelType::CONV_IMPLICIT_GEMM;
}

bool is_warp_specialized(
  library::OperationDescription const &operation_desc) {

  std::string name(operation_desc.name);
  return name.find("warpspecialized") != std::string::npos;
}

bool is_analytic(
  library::OperationDescription const &operation_desc) {

  std::string name(operation_desc.name);
  return name.find("analytic") != std::string::npos;
}

SMArch sm_version(const cudaDeviceProp &options_device_properties) {
  switch (options_device_properties.major) {
  case 10:
    return SMArch::SM_100;
  case 9:
    return SMArch::SM_90;
  case 8:
    switch (options_device_properties.minor) {
    case 9: return SMArch::SM_89;
    case 8: return SMArch::SM_88;
    case 7: return SMArch::SM_87;
    case 6: return SMArch::SM_86;
    default: return SMArch::SM_80;
    }
  case 7:
    switch (options_device_properties.minor) {
    case 5: return SMArch::SM_75;
    case 2: return SMArch::SM_72;
    default: return SMArch::SM_70;
    }
  case 6:
    switch (options_device_properties.minor) {
    case 2: return SMArch::SM_62;
    case 1: return SMArch::SM_61;
    default: return SMArch::SM_60;
    }
  default:
    switch (options_device_properties.minor) {
    case 3: return SMArch::SM_53;
    case 2: return SMArch::SM_52;
    default: return SMArch::SM_50;
    }
  }
}

SMArch sm_arch(library::OperationDescription const &operation_desc) {

  int min = operation_desc.tile_description.minimum_compute_capability;
  if (min >= 100) {
    return SMArch::SM_100;
  } else if (min >= 90) {
    return SMArch::SM_90;
  } else if (min >= 89) {
    return SMArch::SM_89;
  } else if (min >= 88) {
    return SMArch::SM_88;
  } else if (min >= 87) {
    return SMArch::SM_87;
  } else if (min >= 86) {
    return SMArch::SM_86;
  } else if (min >= 80) {
    return SMArch::SM_80;
  } else if (min >= 75) {
    return SMArch::SM_75;
  } else if (min >= 72) {
    return SMArch::SM_72;
  } else if (min >= 70) {
    return SMArch::SM_70;
  } else if (min >= 62) {
    return SMArch::SM_62;
  } else if (min >= 61) {
    return SMArch::SM_61;
  } else if (min >= 60) {
    return SMArch::SM_60;
  } else if (min >= 53) {
    return SMArch::SM_53;
  } else if (min >= 52) {
    return SMArch::SM_52;
  } else {
    return SMArch::SM_50;
  }
}

MmaShape mma_shape(
  library::OperationDescription const &operation_desc) {

  if (operation_desc.tile_description.math_instruction.opcode_class != library::OpcodeClassID::kSimt) {
    switch (operation_desc.tile_description.math_instruction.instruction_shape.m()) {
      case 8:
        switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
          case 4:
            return MmaShape::SHAPE_884;
          case 16:
            return MmaShape::SHAPE_8816;
          case 32:
            return MmaShape::SHAPE_8832;
        }
      case 16:
        switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
          case 4:
            return MmaShape::SHAPE_1684;
          case 8:
            return MmaShape::SHAPE_1688;
          case 16:
            return MmaShape::SHAPE_16816;
          case 32:
            return MmaShape::SHAPE_16832;
          case 64:
            return MmaShape::SHAPE_16864;
        }
      case 64:
        switch (operation_desc.tile_description.math_instruction.instruction_shape.n()) {
          case 8:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
            }
          case 16:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
            }
          case 32:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
            }
          case 64:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
              case 64:
                return MmaShape::SHAPE_64N64;
            }
          case 96:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
            }
          case 128:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
              case 64:
                return MmaShape::SHAPE_64N64;
            }
          case 160:
            return MmaShape::SHAPE_64N32;
          case 192:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
              case 64:
                return MmaShape::SHAPE_64N64;
            }
          case 208:
            return MmaShape::SHAPE_64N64;
          case 256:
            switch (operation_desc.tile_description.math_instruction.instruction_shape.k()) {
              case 8:
                return MmaShape::SHAPE_64N8;
              case 16:
                return MmaShape::SHAPE_64N16;
              case 32:
                return MmaShape::SHAPE_64N32;
              case 64:
                return MmaShape::SHAPE_64N64;
            }
        }
    }
  }

  return MmaShape::SHAPE_111;
}

void tensor_properties(
  library::TensorDescription const &tensorDescription,
  int32_t *sizePerElement, int32_t *alignment) {

  if (sizePerElement != nullptr) {
    *sizePerElement = size_in_bytes_of_type(
      scalar_type(tensorDescription.element));
  }

  if (alignment != nullptr) {
    *alignment = tensorDescription.alignment;
  }
}

bool is_sparse(library::OperationDescription const &operation_desc) {
  return operation_desc.kind == library::OperationKind::kSparseGemm;
}

int32_t split_k_kernels(library::SplitKMode mode) {
  switch (mode) {
  default:
  case library::SplitKMode::kNone:
  case library::SplitKMode::kInvalid:
    return 0;
  case library::SplitKMode::kSerial:
    return 1;
  case library::SplitKMode::kParallel:
  case library::SplitKMode::kParallelSerial:
    return 2;
  }
}

FormatConvert format_converter(int32_t smVersion) {
    if (smVersion >= 86) {
        return FormatConvert::PACKED;
    }
    return FormatConvert::UNPACKED;
}

LayerType conv_layer_type(library::ConvDescription const &conv_desc) {
    switch (conv_desc.conv_kind) {
    case library::ConvKind::kFprop: return LayerType::CONV_FPROP;
    case library::ConvKind::kDgrad: return LayerType::CONV_DGRAD;
    case library::ConvKind::kWgrad: return LayerType::CONV_WGRAD;
    }
    return LayerType::INVALID;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // CUTLASS_ENABLE_KERNEL_TIMING_MODEL

///////////////////////////////////////////////////////////////////////////////////////////////////

void predict(
  [[maybe_unused]] PerformanceResult &result,
  [[maybe_unused]] Options const &options,
  [[maybe_unused]] library::OperationDescription const &operation_desc,
  [[maybe_unused]] OperationProfiler const &operation_profiler) {
#ifdef CUTLASS_ENABLE_KERNEL_TIMING_MODEL

  double cycles = 0;
  double microseconds = 0;
  double millisPerMicro = 1e3;

  HardwareProperties hardwareProperties = HardwareProperties(
    options.device.properties.multiProcessorCount,                                 // smCount
    sm_version(options.device.properties),                                         // smVersion
    static_cast<int32_t>(options.device.properties.sharedMemPerMultiprocessor),    // sharedMemCapacity
    options.device.properties.l2CacheSize,                                         // l2CacheSize
    static_cast<float>(options.device.properties.clockRate / millisPerMicro),      // smClk
    static_cast<float>(options.device.properties.memoryClockRate / millisPerMicro) // dramClk
  );
  hardwareProperties.vsl.push_back(hardwareProperties.smCount / 2);

  switch (operation_desc.kind) {
    case library::OperationKind::kGemm:
      {
        library::GemmDescription const &gemm_desc =
          static_cast<library::GemmDescription const &>(operation_desc);

        GemmOperationProfiler::GemmProblem const &gemmProblem =
          static_cast<GemmOperationProfiler const &>(operation_profiler).problem();

        GemmLayerParams gemmLayerParams;
        gemmLayerParams.typeLayer = LayerType::GEMM;
        gemmLayerParams.bias = false;
        gemmLayerParams.beta = false;
        gemmLayerParams.gemmN = gemmProblem.n;
        gemmLayerParams.gemmK = gemmProblem.k;
        gemmLayerParams.gemmM = gemmProblem.m;
        gemmLayerParams.batch = gemmProblem.batch_count;
        for (auto beta : gemmProblem.beta) {
          if (beta != 0) {
            gemmLayerParams.beta = true;
            break;
          }
        }

        GemmKernelProperties gemmKernelProperties;
        gemmKernelProperties.typeKernel = kernel_type(gemm_desc);
        gemmKernelProperties.kernelBackend = KernelBackend::CUTLASS_9X;
        gemmKernelProperties.ctaM = gemm_desc.tile_description.threadblock_shape.m();
        gemmKernelProperties.ctaN = gemm_desc.tile_description.threadblock_shape.n();
        gemmKernelProperties.ctaK = gemm_desc.tile_description.threadblock_shape.k();
        gemmKernelProperties.warpM = gemm_desc.tile_description.threadblock_shape.m() / gemm_desc.tile_description.warp_count.m();
        gemmKernelProperties.warpN = gemm_desc.tile_description.threadblock_shape.n() / gemm_desc.tile_description.warp_count.n();
        gemmKernelProperties.cgaM = gemm_desc.tile_description.cluster_shape.m();
        gemmKernelProperties.cgaN = gemm_desc.tile_description.cluster_shape.n();
        gemmKernelProperties.occupancy = 1;
        gemmKernelProperties.nbWarps = gemm_desc.tile_description.warp_count.m() * gemm_desc.tile_description.warp_count.n();
        gemmKernelProperties.sharedMemBytes = 0;
        gemmKernelProperties.nbRegisters = 0;
        gemmKernelProperties.smArch = sm_arch(gemm_desc);
        gemmKernelProperties.name = gemm_desc.name;
        gemmKernelProperties.bypassA = false;
        gemmKernelProperties.bypassB = false;
        gemmKernelProperties.stage = gemm_desc.tile_description.threadblock_stages;
        gemmKernelProperties.mmaShape = mma_shape(gemm_desc);
        gemmKernelProperties.mathType = scalar_type(gemm_desc.A.element);
        gemmKernelProperties.mathSpeed = MmaHelper::queryMathSpeed(
          hardwareProperties.smVersion, gemmKernelProperties.mathType, gemmKernelProperties.mmaShape, is_sparse(gemm_desc));
        gemmKernelProperties.kPhase = MmaHelper::queryMMAShapeK(gemmKernelProperties.mmaShape);
        gemmKernelProperties.tileG = 1;
        gemmKernelProperties.doSwap = false;
        gemmKernelProperties.splitKFactor = gemmProblem.split_k_slices;
        gemmKernelProperties.splitKBuffer = 0;
        gemmKernelProperties.splitKKernels = split_k_kernels(gemmProblem.split_k_mode);
        gemmKernelProperties.blockK = gemm_desc.tile_description.threadblock_shape.k();
        gemmKernelProperties.formatConverter = format_converter(hardwareProperties.smVersion);
        gemmKernelProperties.interleavedLayout = false;
        gemmKernelProperties.isWarpSpecialized = is_warp_specialized(gemm_desc);
        gemmKernelProperties.isAnalytic = is_analytic(gemm_desc);
        tensor_properties(gemm_desc.A, &gemmKernelProperties.sizePerElementA, &gemmKernelProperties.alignmentA);
        tensor_properties(gemm_desc.C, &gemmKernelProperties.sizePerElementC, &gemmKernelProperties.alignmentC);
        tensor_properties(gemm_desc.element_epilogue, &gemmKernelProperties.sizePerElementACC, nullptr);
        finishGemmKernelProperties(gemm_desc.A, gemmKernelProperties);

        KTMOutput ktmOutput(OutputMode::TOTAL_CYCLES);
        ktmOutput.featureVersions.push_back({Resource::MATH, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::MIO, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::XU, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::GNIC_READ, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::L2_READ, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::DRAM, gemmKernelProperties.smArch, 0, 0});
        size_t elemsPerBatch = ktmGetOutputShape(gemmLayerParams.typeLayer, ktmOutput);
        std::vector<float> output(elemsPerBatch);
        ktmOutput.data = output.data();

        ktmPredictOne(&gemmLayerParams, &gemmKernelProperties, hardwareProperties, ktmOutput);

        cycles = output.at(0);
        microseconds = output.at(0) / hardwareProperties.smClk;

        break;
      }
    case library::OperationKind::kConv2d:
      {
        library::ConvDescription const &conv_desc =
          static_cast<library::ConvDescription const &>(operation_desc);

        Conv2dOperationProfiler::Conv2dProblem const &conv2dProblem =
          static_cast<Conv2dOperationProfiler const &>(operation_profiler).problem();

        ConvLayerParams convLayerParams;
        convLayerParams.typeLayer = conv_layer_type(conv_desc);
        convLayerParams.nbSpatialDims = 2;
        convLayerParams.g = static_cast<int32_t>(conv2dProblem.groups);
        convLayerParams.inDims.nbDims = 4;
        convLayerParams.inDims.d[0] = static_cast<int32_t>(conv2dProblem.n);
        convLayerParams.inDims.d[1] = static_cast<int32_t>(conv2dProblem.c);
        convLayerParams.inDims.d[2] = static_cast<int32_t>(conv2dProblem.h);
        convLayerParams.inDims.d[3] = static_cast<int32_t>(conv2dProblem.w);
        convLayerParams.outDims.nbDims = 4;
        convLayerParams.outDims.d[0] = static_cast<int32_t>(conv2dProblem.n);
        convLayerParams.outDims.d[1] = static_cast<int32_t>(conv2dProblem.k);
        convLayerParams.outDims.d[2] = static_cast<int32_t>(conv2dProblem.p);
        convLayerParams.outDims.d[3] = static_cast<int32_t>(conv2dProblem.q);
        convLayerParams.kernelDims.nbDims = 2;
        convLayerParams.kernelDims.d[0] = static_cast<int32_t>(conv2dProblem.r);
        convLayerParams.kernelDims.d[1] = static_cast<int32_t>(conv2dProblem.s);
        convLayerParams.strideDims.nbDims = 2;
        convLayerParams.strideDims.d[0] = static_cast<int32_t>(conv2dProblem.stride_h);
        convLayerParams.strideDims.d[1] = static_cast<int32_t>(conv2dProblem.stride_w);
        convLayerParams.dilationDims.nbDims = 2;
        convLayerParams.dilationDims.d[0] = static_cast<int32_t>(conv2dProblem.dilation_h);
        convLayerParams.dilationDims.d[1] = static_cast<int32_t>(conv2dProblem.dilation_w);
        convLayerParams.padDims.nbDims = 4;
        convLayerParams.padDims.d[0] = static_cast<int32_t>(conv2dProblem.pad_h);
        convLayerParams.padDims.d[1] = static_cast<int32_t>(conv2dProblem.pad_w);
        convLayerParams.padDims.d[2] = static_cast<int32_t>(conv2dProblem.pad_h);
        convLayerParams.padDims.d[3] = static_cast<int32_t>(conv2dProblem.pad_w);
        convLayerParams.bias = false;
        convLayerParams.beta = false;
        convLayerParams.updateGemmDims();
        for (auto beta : conv2dProblem.beta) {
          if (beta != 0) {
            convLayerParams.beta = true;
            break;
          }
        }

        GemmKernelProperties gemmKernelProperties;
        gemmKernelProperties.typeKernel = kernel_type(conv_desc);
        gemmKernelProperties.kernelBackend = KernelBackend::CUTLASS_9X;
        gemmKernelProperties.ctaM = conv_desc.tile_description.threadblock_shape.m();
        gemmKernelProperties.ctaN = conv_desc.tile_description.threadblock_shape.n();
        gemmKernelProperties.ctaK = conv_desc.tile_description.threadblock_shape.k();
        gemmKernelProperties.warpM = conv_desc.tile_description.threadblock_shape.m() / conv_desc.tile_description.warp_count.m();
        gemmKernelProperties.warpN = conv_desc.tile_description.threadblock_shape.n() / conv_desc.tile_description.warp_count.n();
        gemmKernelProperties.cgaM = conv_desc.tile_description.cluster_shape.m();
        gemmKernelProperties.cgaN = conv_desc.tile_description.cluster_shape.n();
        gemmKernelProperties.occupancy = 1;
        gemmKernelProperties.nbWarps = conv_desc.tile_description.warp_count.m() * conv_desc.tile_description.warp_count.n();
        gemmKernelProperties.sharedMemBytes = 0;
        gemmKernelProperties.nbRegisters = 0;
        gemmKernelProperties.smArch = sm_arch(conv_desc);
        gemmKernelProperties.name = conv_desc.name;
        gemmKernelProperties.bypassA = false;
        gemmKernelProperties.bypassB = false;
        gemmKernelProperties.stage = conv_desc.tile_description.threadblock_stages;
        gemmKernelProperties.mmaShape = mma_shape(conv_desc);
        gemmKernelProperties.mathType = scalar_type(conv_desc.A.element);
        gemmKernelProperties.mathSpeed = MmaHelper::queryMathSpeed(
          hardwareProperties.smVersion, gemmKernelProperties.mathType, gemmKernelProperties.mmaShape, is_sparse(conv_desc));
        gemmKernelProperties.kPhase = MmaHelper::queryMMAShapeK(gemmKernelProperties.mmaShape);
        gemmKernelProperties.tileG = 1;
        gemmKernelProperties.doSwap = false;
        gemmKernelProperties.splitKFactor = static_cast<int32_t>(conv2dProblem.split_k_slices);
        gemmKernelProperties.splitKBuffer = 0;
        gemmKernelProperties.splitKKernels = split_k_kernels(conv2dProblem.split_k_mode);
        gemmKernelProperties.blockK = conv_desc.tile_description.threadblock_shape.k();
        gemmKernelProperties.formatConverter = format_converter(hardwareProperties.smVersion);
        gemmKernelProperties.interleavedLayout = false;
        gemmKernelProperties.isWarpSpecialized = is_warp_specialized(conv_desc);
        gemmKernelProperties.isAnalytic = is_analytic(conv_desc);
        tensor_properties(conv_desc.A, &gemmKernelProperties.sizePerElementA, &gemmKernelProperties.alignmentA);
        tensor_properties(conv_desc.C, &gemmKernelProperties.sizePerElementC, &gemmKernelProperties.alignmentC);
        tensor_properties(conv_desc.element_epilogue, &gemmKernelProperties.sizePerElementACC, nullptr);
        finishGemmKernelProperties(conv_desc.A, gemmKernelProperties);

        KTMOutput ktmOutput(OutputMode::TOTAL_CYCLES);
        ktmOutput.featureVersions.push_back({Resource::MATH, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::MIO, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::XU, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::GNIC_READ, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::L2_READ, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::DRAM, gemmKernelProperties.smArch, 0, 0});
        size_t elemsPerBatch = ktmGetOutputShape(convLayerParams.typeLayer, ktmOutput);
        std::vector<float> output(elemsPerBatch);
        ktmOutput.data = output.data();

        ktmPredictOne(&convLayerParams, &gemmKernelProperties, hardwareProperties, ktmOutput);

        cycles = output.at(0);
        microseconds = output.at(0) / hardwareProperties.smClk;

        break;
      }
    case library::OperationKind::kConv3d:
      {
        library::ConvDescription const &conv_desc =
          static_cast<library::ConvDescription const &>(operation_desc);

        Conv3dOperationProfiler::Conv3dProblem const &conv3dProblem =
          static_cast<Conv3dOperationProfiler const &>(operation_profiler).problem();

        ConvLayerParams convLayerParams;
        convLayerParams.typeLayer = conv_layer_type(conv_desc);
        convLayerParams.nbSpatialDims = 3;
        convLayerParams.g = 1;
        convLayerParams.inDims.nbDims = 5;
        convLayerParams.inDims.d[0] = static_cast<int32_t>(conv3dProblem.n);
        convLayerParams.inDims.d[1] = static_cast<int32_t>(conv3dProblem.c);
        convLayerParams.inDims.d[2] = static_cast<int32_t>(conv3dProblem.d);
        convLayerParams.inDims.d[3] = static_cast<int32_t>(conv3dProblem.h);
        convLayerParams.inDims.d[4] = static_cast<int32_t>(conv3dProblem.w);
        convLayerParams.outDims.nbDims = 5;
        convLayerParams.outDims.d[0] = static_cast<int32_t>(conv3dProblem.n);
        convLayerParams.outDims.d[1] = static_cast<int32_t>(conv3dProblem.k);
        convLayerParams.outDims.d[2] = static_cast<int32_t>(conv3dProblem.z);
        convLayerParams.outDims.d[3] = static_cast<int32_t>(conv3dProblem.p);
        convLayerParams.outDims.d[4] = static_cast<int32_t>(conv3dProblem.q);
        convLayerParams.kernelDims.nbDims = 3;
        convLayerParams.kernelDims.d[0] = static_cast<int32_t>(conv3dProblem.t);
        convLayerParams.kernelDims.d[1] = static_cast<int32_t>(conv3dProblem.r);
        convLayerParams.kernelDims.d[2] = static_cast<int32_t>(conv3dProblem.s);
        convLayerParams.strideDims.nbDims = 3;
        convLayerParams.strideDims.d[0] = static_cast<int32_t>(conv3dProblem.stride_d);
        convLayerParams.strideDims.d[1] = static_cast<int32_t>(conv3dProblem.stride_h);
        convLayerParams.strideDims.d[2] = static_cast<int32_t>(conv3dProblem.stride_w);
        convLayerParams.dilationDims.nbDims = 3;
        convLayerParams.dilationDims.d[0] = static_cast<int32_t>(conv3dProblem.dilation_d);
        convLayerParams.dilationDims.d[1] = static_cast<int32_t>(conv3dProblem.dilation_h);
        convLayerParams.dilationDims.d[2] = static_cast<int32_t>(conv3dProblem.dilation_w);
        convLayerParams.padDims.nbDims = 6;
        convLayerParams.padDims.d[0] = static_cast<int32_t>(conv3dProblem.pad_d);
        convLayerParams.padDims.d[1] = static_cast<int32_t>(conv3dProblem.pad_h);
        convLayerParams.padDims.d[2] = static_cast<int32_t>(conv3dProblem.pad_w);
        convLayerParams.padDims.d[3] = static_cast<int32_t>(conv3dProblem.pad_d);
        convLayerParams.padDims.d[4] = static_cast<int32_t>(conv3dProblem.pad_h);
        convLayerParams.padDims.d[5] = static_cast<int32_t>(conv3dProblem.pad_w);
        convLayerParams.bias = false;
        convLayerParams.beta = false;
        convLayerParams.updateGemmDims();
        for (auto beta : conv3dProblem.beta) {
          if (beta != 0) {
            convLayerParams.beta = true;
            break;
          }
        }

        GemmKernelProperties gemmKernelProperties;
        gemmKernelProperties.typeKernel = kernel_type(conv_desc);
        gemmKernelProperties.kernelBackend = KernelBackend::CUTLASS_9X;
        gemmKernelProperties.ctaM = conv_desc.tile_description.threadblock_shape.m();
        gemmKernelProperties.ctaN = conv_desc.tile_description.threadblock_shape.n();
        gemmKernelProperties.ctaK = conv_desc.tile_description.threadblock_shape.k();
        gemmKernelProperties.warpM = conv_desc.tile_description.threadblock_shape.m() / conv_desc.tile_description.warp_count.m();
        gemmKernelProperties.warpN = conv_desc.tile_description.threadblock_shape.n() / conv_desc.tile_description.warp_count.n();
        gemmKernelProperties.cgaM = conv_desc.tile_description.cluster_shape.m();
        gemmKernelProperties.cgaN = conv_desc.tile_description.cluster_shape.n();
        gemmKernelProperties.occupancy = 1;
        gemmKernelProperties.nbWarps = conv_desc.tile_description.warp_count.m() * conv_desc.tile_description.warp_count.n();
        gemmKernelProperties.sharedMemBytes = 0;
        gemmKernelProperties.nbRegisters = 0;
        gemmKernelProperties.smArch = sm_arch(conv_desc);
        gemmKernelProperties.name = conv_desc.name;
        gemmKernelProperties.bypassA = false;
        gemmKernelProperties.bypassB = false;
        gemmKernelProperties.stage = conv_desc.tile_description.threadblock_stages;
        gemmKernelProperties.mmaShape = mma_shape(conv_desc);
        gemmKernelProperties.mathType = scalar_type(conv_desc.A.element);
        gemmKernelProperties.mathSpeed = MmaHelper::queryMathSpeed(
          hardwareProperties.smVersion, gemmKernelProperties.mathType, gemmKernelProperties.mmaShape, is_sparse(conv_desc));
        gemmKernelProperties.kPhase = MmaHelper::queryMMAShapeK(gemmKernelProperties.mmaShape);
        gemmKernelProperties.tileG = 1;
        gemmKernelProperties.doSwap = false;
        gemmKernelProperties.splitKFactor = static_cast<int32_t>(conv3dProblem.split_k_slices);
        gemmKernelProperties.splitKBuffer = 0;
        gemmKernelProperties.splitKKernels = split_k_kernels(conv3dProblem.split_k_mode);
        gemmKernelProperties.blockK = conv_desc.tile_description.threadblock_shape.k();
        gemmKernelProperties.formatConverter = format_converter(hardwareProperties.smVersion);
        gemmKernelProperties.interleavedLayout = false;
        gemmKernelProperties.isWarpSpecialized = is_warp_specialized(conv_desc);
        gemmKernelProperties.isAnalytic = is_analytic(conv_desc);
        tensor_properties(conv_desc.A, &gemmKernelProperties.sizePerElementA, &gemmKernelProperties.alignmentA);
        tensor_properties(conv_desc.C, &gemmKernelProperties.sizePerElementC, &gemmKernelProperties.alignmentC);
        tensor_properties(conv_desc.element_epilogue, &gemmKernelProperties.sizePerElementACC, nullptr);
        finishGemmKernelProperties(conv_desc.A, gemmKernelProperties);

        KTMOutput ktmOutput(OutputMode::TOTAL_CYCLES);
        ktmOutput.featureVersions.push_back({Resource::MATH, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::MIO, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::XU, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::GNIC_READ, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::L2_READ, gemmKernelProperties.smArch, 0, 0});
        ktmOutput.featureVersions.push_back({Resource::DRAM, gemmKernelProperties.smArch, 0, 0});
        size_t elemsPerBatch = ktmGetOutputShape(convLayerParams.typeLayer, ktmOutput);
        std::vector<float> output(elemsPerBatch);
        ktmOutput.data = output.data();

        ktmPredictOne(&convLayerParams, &gemmKernelProperties, hardwareProperties, ktmOutput);

        cycles = output.at(0);
        microseconds = output.at(0) / hardwareProperties.smClk;

        break;
      }
    default:
      return;
  }

  result.kernel_timing_model_cycles = static_cast<int64_t>(cycles);
  result.kernel_timing_model_microseconds = static_cast<int64_t>(microseconds);

#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ktm
} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
