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

/*! \file
  \brief Compress utils specific for SM100 structure sparse kernels
*/

// {$nv-internal-release file}

#pragma once

#include <algorithm>
#include <random>

#include "cutlass/coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/conv/collective/builders/sm100_sparse_config.inl"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/util/debug.hpp"
#include "cute/config.hpp"

#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"

namespace cutlass::transform::kernel {

namespace detail {

template<typename T>
CUTLASS_HOST_DEVICE
static uint8_t
encode_in_chunk_idx(int in_chunk_idx){
  if (sizeof(T) == 4) {
    return in_chunk_idx == 0 ? 0b0100 : 0b1110;
  }
  else {
    uint8_t res = 0;
    if (in_chunk_idx == 0) {
      res = 0b00;
    }
    else if (in_chunk_idx == 1) {
      res = 0b01;
    }
    else if (in_chunk_idx == 2) {
      res = 0b10;
    }
    else {
      res = 0b11;
    }
    return res;
  }
}

template <
  class Sm1xxSparseConfig,
  class EngineA,
  class LayoutA,
  class EngineAc,
  class LayoutAc
>
CUTLASS_HOST_DEVICE
static void
compress_two_chunks(
  Tensor<EngineA, LayoutA> tensorA,
  Tensor<EngineAc, LayoutAc> tensorAc,
  uint8_t& meta_two_chunk,
  int effective_elems) {

  using ElementA = typename EngineAc::value_type;

  static constexpr int LogicalElemsAPerChunk  = typename Sm1xxSparseConfig::LogicalElemsAPerChunk{};
  static constexpr int PhysicalElemsAPerChunk  = typename Sm1xxSparseConfig::PhysicalElemsAPerChunk{};
  static constexpr int ElemsARawPerElementAMmaRaw    = typename Sm1xxSparseConfig::ElemsARawPerElementAMmaRaw{};
  static constexpr int ElementEBitsPerElementAMma = typename Sm1xxSparseConfig::ElementEBitsPerElementAMma{};
  static constexpr int LogicalSubChunk     = ceil_div(LogicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);
  static constexpr int PhysicalSubChunk    = ceil_div(PhysicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);

  /*
  Legal metadata chunk in SM100
  Index   Bin   HEX
  0, 1  0b0100   4
  1, 2  0b1001   9
  2, 3  0b1110   E
  0, 2  0b1000   8
  1, 3  0b1101   D
  0, 3  0b1100   C
  2, 1  0b0110   6  (Not used)
  -----------------------------------
  TF32
  0     0b0100   4
  1     0b1110   E
  */

  if (effective_elems <= 0) {
    return;
  }

  // initialize
  // 0 is the initial value for this function while 0x44 is the initial value for hardware.
  meta_two_chunk = 0;

  for (int chunk_idx = 0; chunk_idx < 2; ++chunk_idx) {
    /// init result;
    int non_zero_cnt = 0;
    int32_t nnz_chunk_idx[PhysicalSubChunk] = { 0 };
    ElementA Ac_chunk[PhysicalSubChunk][ElemsARawPerElementAMmaRaw] = { ElementA{0} };

    for (int subchunk_idx = 0; subchunk_idx < LogicalSubChunk; ++subchunk_idx) {
      bool is_nz = true;
      ElementA subchunk_elems[ElemsARawPerElementAMmaRaw] = { ElementA{0} };
      /// Check if subchunk is non-zero
      for(int elem_idx = 0; elem_idx < ElemsARawPerElementAMmaRaw; elem_idx++) {
        int offset = chunk_idx * LogicalElemsAPerChunk + subchunk_idx * ElemsARawPerElementAMmaRaw + elem_idx;
        subchunk_elems[elem_idx] = offset < effective_elems ? tensorA(offset) : ElementA(0);
        
        if (subchunk_elems[elem_idx] != ElementA(0)) {
          if (non_zero_cnt >= PhysicalSubChunk) {
            #ifdef  __CUDA_ARCH__
              asm volatile ("brkpt;\n" ::);
            #else
              throw std::runtime_error("Found extra non-zero elements in a chunk!\n");
            #endif
          }
          is_nz = false;
        }
      }

      /// There is non-zero element in the subchunk
      if(!is_nz) {
        nnz_chunk_idx[non_zero_cnt] = subchunk_idx;
        memcpy(Ac_chunk[non_zero_cnt], subchunk_elems, sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
        non_zero_cnt++;
      }
    }

    /*
    Special cases
    nnz == 1 and non-tf32 and nnz_idx = 3
    */
    ElementA elementA_zeros[ElemsARawPerElementAMmaRaw] = { ElementA{0} };
    if constexpr (sizeof_bits_v<ElementA> < 32) {
      if (non_zero_cnt == 1 && nnz_chunk_idx[0] == 3) {
        memcpy(Ac_chunk[1], Ac_chunk[0], sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
        memcpy(Ac_chunk[0], elementA_zeros, sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
        nnz_chunk_idx[1] = 3;
        nnz_chunk_idx[0] = 0;
      }
      else if (non_zero_cnt == 1) {
        memcpy(Ac_chunk[1], elementA_zeros, sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
        nnz_chunk_idx[1] = 3;
      }
    }

    /// Setup metadata
    uint8_t meta_chunk = 0;
    for (int i = 0; i < PhysicalSubChunk; i++) {
      meta_chunk = static_cast<uint8_t>(meta_chunk | (encode_in_chunk_idx<ElementA>(nnz_chunk_idx[i]) << (i * ElementEBitsPerElementAMma)));
      for(int j = 0; j < ElemsARawPerElementAMmaRaw; j++) {
        tensorAc(chunk_idx * PhysicalElemsAPerChunk + i * ElemsARawPerElementAMmaRaw + j) = Ac_chunk[i][j];
      }
    }
    meta_two_chunk = uint8_t(meta_two_chunk | (meta_chunk << (chunk_idx * _4{})));
  }
}

} // namespace detail

using namespace cute;

template<
  class ElementA_,
  class Sm1xxSparseConfig_
>
class SM100StructuredSparseConvCompressorUtility {
public:
  using Sm1xxSparseConfig = Sm1xxSparseConfig_;
  
  using ElementA = ElementA_;

  using ElementAMmaRaw = typename Sm1xxSparseConfig::TensorAType;
  using ElementEMmaRaw = typename Sm1xxSparseConfig::TensorEType;
  using ElementAMmaSparsity = Int<Sm1xxSparseConfig::TensorASparsity>;
  using ElementEMmaSparsity = Int<Sm1xxSparseConfig::TensorESparsity>;

  static constexpr int TensorEAlignmentK = Sm1xxSparseConfig::AlignmentK;
  static constexpr int TensorEAlignmentM = Sm1xxSparseConfig::AlignmentM;
  static constexpr int ExpandFactor = typename Sm1xxSparseConfig::ExpandFactor{};

  static constexpr int LogicalElemsAPerChunk  = typename Sm1xxSparseConfig::LogicalElemsAPerChunk{};

  SM100StructuredSparseConvCompressorUtility() = default;

  template <class Problem_t>
  SM100StructuredSparseConvCompressorUtility(Problem_t const& problem_shape) {
    set_problem_size(problem_shape);
  }

  template <class Problem_t>
  void set_problem_size(Problem_t const& problem_shape) {
    // filter params
    k = cute::get<0>(problem_shape.shape_B);
    t = cute::get<1>(problem_shape.shape_B);
    r = cute::get<2>(problem_shape.shape_B);
    s = cute::get<3>(problem_shape.shape_B);
    c = cute::get<4>(problem_shape.shape_B);

    stride_k = cute::get<0>(problem_shape.stride_B);
    stride_t = cute::get<1>(problem_shape.stride_B);
    stride_r = cute::get<2>(problem_shape.stride_B);
    stride_s = cute::get<3>(problem_shape.stride_B);
    stride_c = cute::get<4>(problem_shape.stride_B);
  }

  void structure_sparse_zero_mask_fill(void* host_a_ptr, int seed) {

    Tensor raw_tensor_a = make_tensor(
      reinterpret_cast<ElementA*>(host_a_ptr),
      make_layout(
        make_shape(
          k,
          make_shape(c, s, r, t)),
        make_stride(
          stride_k,
          make_stride(_1{}, stride_s, stride_r, stride_t))));

    int chunk_count = ceil_div(c, LogicalElemsAPerChunk);
    constexpr int elems_per_subchunk = typename Sm1xxSparseConfig::ElemsARawPerElementAMmaRaw{};
    int nnzb_indicator[4] = {0};
    for (int i = 0; i < 2; i++) {
      nnzb_indicator[i] = 1;
    }
    constexpr bool is_tf32 = LogicalElemsAPerChunk == 2;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      for (int t_idx = 0; t_idx < t; ++t_idx) {
        for (int r_idx = 0; r_idx < r; ++r_idx) {
          for (int s_idx = 0; s_idx < s; ++s_idx) {
            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
              std::shuffle(&nnzb_indicator[0], &nnzb_indicator[4], std::default_random_engine(seed++));
              for (int in_chunk_idx = 0; in_chunk_idx < LogicalElemsAPerChunk; in_chunk_idx++) {
                if constexpr (is_tf32) {
                  int c_idx_0 = chunk_idx * LogicalElemsAPerChunk + in_chunk_idx;
                  int c_idx_1 = chunk_idx * LogicalElemsAPerChunk + in_chunk_idx + 1;
                  bool is_oob_0 = c_idx_0 >= c;
                  bool is_oob_1 = c_idx_1 >= c;
                  bool is_first_elem_zero = nnzb_indicator[0] == 0;
                  if (is_first_elem_zero) {
                    if (!is_oob_0) {
                      raw_tensor_a(k_idx, make_coord(c_idx_0, s_idx, r_idx, t_idx)) = ElementA(0);
                    }
                  }
                  else {
                    if (!is_oob_1) {
                      raw_tensor_a(k_idx, make_coord(c_idx_1, s_idx, r_idx, t_idx)) = ElementA(0);
                    }
                  }
                  break;
                }
                else {
                  int c_idx = chunk_idx * LogicalElemsAPerChunk + in_chunk_idx;
                  bool is_oob = c_idx >= c;
                  if (!is_oob) {
                    bool is_zero_this_elem = nnzb_indicator[in_chunk_idx/elems_per_subchunk] == 0;
                    if (is_zero_this_elem) {
                      raw_tensor_a(k_idx, make_coord(c_idx, s_idx, r_idx, t_idx)) = ElementA(0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  uint64_t get_tensor_A_bytes() const {
    return t * r * s * c * k * sizeof(ElementAMmaRaw) / ElementAMmaSparsity{};
  }

  uint64_t get_tensor_E_bytes() const {
    const int metadata_k = round_up(k, TensorEAlignmentM);
    const int metadata_c = round_up(c, TensorEAlignmentK);
    const int metadata_bytes = t * r * s * metadata_c * metadata_k * ExpandFactor * sizeof(ElementEMmaRaw) / ElementEMmaSparsity{};
    return metadata_bytes;
  }

private:
  int32_t k;
  int32_t t;
  int32_t r;
  int32_t s;
  int32_t c;
  
  int64_t stride_k;
  int64_t stride_t;
  int64_t stride_r;
  int64_t stride_s;
  int64_t stride_c;
};

template<
  class ProblemShape_,
  class ElementA_,
  class Sm1xxSparseConfig_>
struct SM100StructuredSparseConvCompressor {
  using ProblemShape = ProblemShape_;
  using ElementA = ElementA_;
  using Sm1xxSparseConfig = Sm1xxSparseConfig_;

  using ElementAMmaRaw = typename Sm1xxSparseConfig::TensorAType;
  using ElementEMmaRaw = typename Sm1xxSparseConfig::TensorEType;
  using ElementAMmaSparsity = Int<Sm1xxSparseConfig::TensorASparsity>;
  using ElementEMmaSparsity = Int<Sm1xxSparseConfig::TensorESparsity>;

  using UnreorderedAtom = typename Sm1xxSparseConfig::UnreorderedAtom;

  static constexpr int LogicalElemsAPerChunk = typename Sm1xxSparseConfig::LogicalElemsAPerChunk{};
  static constexpr int ExpandFactor = typename Sm1xxSparseConfig::ExpandFactor{};
  static constexpr bool is_tf32 = LogicalElemsAPerChunk == 2;
  // basic block is 16x64
  static constexpr int TensorEAlignmentK = Sm1xxSparseConfig::AlignmentK;
  static constexpr int TensorEAlignmentM = Sm1xxSparseConfig::AlignmentM;

  // Required by `device_kernel`
  static constexpr int MaxThreadsPerBlock = 1;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  using ArchTag = arch::Sm100;

  struct SharedStorage {
    /* empty, no smem needed */
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct TransformArguments {
    const void* ptr_A;
    void* ptr_ACompress;
    void* ptr_E;
  };

  struct TransformParams {
    const void* ptr_A;
    void* ptr_ACompress;
    void* ptr_E;
  };

  struct Arguments {
    ProblemShape problem_shape{};
    TransformArguments transform{};
    KernelHardwareInfo hw_info{};
  };

  struct Params {
    ProblemShape problem_shape{};
    TransformParams transform{};
    KernelHardwareInfo hw_info{};
    void* workspace = nullptr;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
      ProblemShape{args.problem_shape},
      TransformParams{args.transform.ptr_A, args.transform.ptr_ACompress, args.transform.ptr_E}, 
      KernelHardwareInfo{args.hw_info},
      workspace};
  }

  static Status
  can_implement(Arguments const& args) {
    return Status::kSuccess;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    auto problem_shape = args.problem_shape;
    const int k = cute::get<0>(problem_shape.shape_B);
    const int t = cute::get<1>(problem_shape.shape_B);
    const int r = cute::get<2>(problem_shape.shape_B);
    const int s = cute::get<3>(problem_shape.shape_B);
    const int c = cute::get<4>(problem_shape.shape_B);
    const int metadata_k = round_up(k, TensorEAlignmentM);
    const int metadata_c = round_up(c, TensorEAlignmentK);
    const int metadata_bytes = t * r * s * metadata_c * metadata_k * sizeof(ElementEMmaRaw) / ElementEMmaSparsity{};

    return metadata_bytes;
  }

  static Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    cudaError_t cuda_error;

    auto workspace_size = get_workspace_size(args);
    if (workspace_size == 0) {
      return Status::kSuccess;
    } else if (workspace == nullptr) {
      return Status::kErrorInternal;
    }

    cudaPointerAttributes attri;
    cuda_error = cudaPointerGetAttributes(&attri, workspace);
    if (cuda_error != cudaSuccess) {
      return Status::kErrorInternal;
    }

    if ( attri.type == cudaMemoryTypeDevice ) {
#if defined(CUTLASS_ENABLE_CUDA_HOST_ADAPTER) && CUTLASS_ENABLE_CUDA_HOST_ADAPTER
      CUTLASS_ASSERT(cuda_adapter);
      if (Status::kSuccess != cuda_adapter->memsetDevice(workspace, static_cast<uint8_t>(0), workspace_size, stream)) {
        return Status::kErrorInternal;
      }
#else
      cudaMemsetAsync(workspace, 0, workspace_size, stream);
      cuda_error = cudaGetLastError();
      if (cuda_error != cudaSuccess) {
        return Status::kErrorInternal;
      }
#endif
    } else {
      memset(workspace, 0, workspace_size);
    }

    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    return dim3(1, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(1, 1, 1);
  }

  CUTE_HOST_DEVICE
  void
  operator()(Params params, char* smem_buf = nullptr) {
    run(params, smem_buf);
  }

  CUTE_HOST_DEVICE
  static void
  run(Params params, char* smem_buf = nullptr) {
    do_compress_device_host(params);
  }

private:

  CUTE_HOST_DEVICE
  static void
  do_compress_device_host(Params params) { 
    auto [k, t, r, s, c] = params.problem_shape.shape_B;
    auto [ptr_A, ptr_ACompress, ptr_E] = params.transform;
    auto [stride_k, stride_t, stride_r, stride_s, stride_c] = params.problem_shape.stride_B;

    const int metadata_k = round_up(k, TensorEAlignmentM);
    const int metadata_c = round_up(c, TensorEAlignmentK);
    
    auto tensorA = make_tensor(
      recast_ptr<ElementA>(ptr_A),
      make_layout(
        make_shape(k, c, s, r, t),
        make_stride(stride_k, _1{}, stride_s, stride_r, stride_t)));

    auto tensorAc = make_tensor(
      recast_ptr<sparse_elem<ElementAMmaSparsity{},ElementAMmaRaw>>(ptr_ACompress),
      Sm1xxSparseConfig::fill_layoutFlt(params.problem_shape, int32_t(0), int64_t(0)));

    auto tensorE_raw = make_tensor(
      recast_ptr<sparse_elem<ElementEMmaSparsity{},ElementEMmaRaw>>(params.workspace),
      make_layout(
        make_shape(metadata_k, metadata_c, make_shape(s, r, t)),
        make_stride(
          metadata_c * s * r * t,
          _1{},
          make_stride(
            metadata_c,
            metadata_c * s,
            metadata_c * s * r
          )
        )
      )
    );

    auto tensorE = make_tensor(
      recast_ptr<sparse_elem<ElementEMmaSparsity{},ElementEMmaRaw>>(ptr_E),
      Sm1xxSparseConfig::fill_layoutMeta(params.problem_shape, int32_t(0), int64_t(0))
    );

    auto tensorE_tiled = make_tensor(
      recast_ptr<sparse_elem<ElementEMmaSparsity{},ElementEMmaRaw>>(ptr_E),
      Sm1xxSparseConfig::fill_reorder_layoutMeta(params.problem_shape, int32_t(0), int64_t(0))
    );

#if 0
    print("tensorA      :"); print(tensorA); print("\n");
    print("tensorAc     :"); print(tensorAc); print("\n");
    print("tensorE_raw  :"); print(tensorE_raw); print("\n");
    print("tensorE      :"); print(tensorE); print("\n");
#endif

    filter_init(tensorE_raw, tensorE);
    filter_raw_compress(tensorA, tensorAc, tensorE_raw);
    filter_reorder(tensorE_raw, tensorE_tiled);
  }


  template<
    class EngineERaw, class LayoutERaw,
    class EngineE,    class LayoutE
  >
  CUTE_HOST_DEVICE
  static void filter_init(Tensor<EngineERaw, LayoutERaw> tensorE_raw_, Tensor<EngineE, LayoutE> tensorE_) {
    auto tensorE_raw = recast<ElementEMmaRaw>(tensorE_raw_);
    auto tensorE     = recast<ElementEMmaRaw>(tensorE_);
    // Note: the default value of metadata is 0x44
    memset(    tensorE.data(), 0x44, cosize(    tensorE.layout()) * sizeof(ElementEMmaRaw));
    memset(tensorE_raw.data(), 0x44, cosize(tensorE_raw.layout()) * sizeof(ElementEMmaRaw));
  }

  template<class EngineA, class LayoutA,
           class EngineAc, class LayoutAc,
           class EngineE, class LayoutE>
  CUTE_HOST_DEVICE
  static void filter_raw_compress(Tensor<EngineA, LayoutA> tensorA_,
                                  Tensor<EngineAc, LayoutAc> tensorAc_,
                                  Tensor<EngineE, LayoutE> tensorE_) {
    
    auto tensorA  = recast<ElementAMmaRaw>(tensorA_);
    auto tensorAc = coalesce(recast<ElementAMmaRaw>(tensorAc_));
    auto tensorE  = coalesce(recast<ElementEMmaRaw>(tensorE_));

    using TileStepA = Int<LogicalElemsAPerChunk * 2>;
    using TileStepAc = Int<TileStepA{} / 2>;

    auto [k, c, s, r, t] = shape(tensorA);

    // K(CSRT)
    Tensor tensorATiled = logical_divide(tensorA, make_shape(_,TileStepA{},_,_,_));
    Tensor tensorAcTiled = logical_divide(tensorAc, make_shape(_,TileStepAc{},_,_,_));

  #if 0
    print("tensorAc       : "); print(tensorAc); print("\n");
    print("tensorE        : "); print(tensorE); print("\n");
    print("tensorATiled   : "); print(tensorATiled); print("\n");
    print("tensorAcTiled  : "); print(tensorAcTiled); print("\n");
  #endif

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      for (int t_idx = 0; t_idx < t; ++t_idx) {
        for (int r_idx = 0; r_idx < r; ++r_idx) {
          for (int s_idx = 0; s_idx < s; ++s_idx) {
            for (int tiler_c_idx = 0; tiler_c_idx < size<1,1>(tensorATiled); tiler_c_idx++) {
              int effective_elems = cute::min(TileStepA{}, c - (tiler_c_idx * TileStepA{}));
              detail::compress_two_chunks<Sm1xxSparseConfig>(tensorATiled(k_idx,make_coord(_,tiler_c_idx),s_idx,r_idx,t_idx),
                                                         tensorAcTiled(k_idx,make_coord(_,tiler_c_idx),s_idx,r_idx,t_idx),
                                                         tensorE(k_idx, tiler_c_idx,s_idx,r_idx,t_idx),
                                                         effective_elems);
            }
          }
        }
      }
    }
  }

  template<
    class TensorERaw,
    class TensorE
  >
  CUTE_HOST_DEVICE
  static void filter_reorder(TensorERaw tensorE_raw, TensorE tensorE_tiled)
  {
    auto tensorE_raw_tiled = zipped_divide(tensorE_raw, shape(UnreorderedAtom{}));

    auto tensorE_raw_u8 = recast<uint8_t>(tensorE_raw_tiled);
    auto tensorE_u8     = recast<uint8_t>(tensorE_tiled);
    copy(tensorE_raw_u8, tensorE_u8);

  #if 0
    print("tensorE_raw_tiled    : "); print(tensorE_raw_tiled); print("\n");
    print("tensorE_tiled        : "); print(tensorE_tiled); print("\n");
    print("tensorE_raw_u8       : "); print(tensorE_raw_u8); print("\n");
    print("tensorE_u8           : "); print(tensorE_u8); print("\n");
  #endif
  }
};

} // End namespace cutlass
