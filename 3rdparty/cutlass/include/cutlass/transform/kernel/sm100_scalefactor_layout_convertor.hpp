/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief ToBlockScaledLayout and ToFlatLayout kernels for sm100 scalefactor
*/

// {$nv-internal-release file}

#pragma once

#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/int_tuple.hpp"
#include "cute/util/debug.hpp"
#include "cute/config.hpp"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace cutlass::transform::kernel {

enum class ScalefactorLayoutConvertorKnob {
  ToBlockScaledLayout = 0,
  ToFlatLayout = 1
};

using namespace cute;

template<
  ScalefactorLayoutConvertorKnob SFLayoutConvertorKnob_,
  class ProblemShape_,
  class ElementSF_
>
class SM100ScalefactorLayoutConvertor {
public:
  static constexpr ScalefactorLayoutConvertorKnob SFLayoutConvertorKnob = SFLayoutConvertorKnob_;
  using ProblemShape = ProblemShape_;
  using ElementSF = ElementSF_;
  using LayoutSFTag = cutlass::layout::RowMajor;
  using StrideSF = cutlass::gemm::TagToStrideA_t<LayoutSFTag>;

  // Required by `device_kernel`
  static constexpr int MaxThreadsPerBlock = 128;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  using CtaTileShape = Shape<_128, _4>;
  using ArchTag = arch::Sm100;

  using SfNMajorAtom  = cutlass::detail::Sm1xxBlockScaledTensorConfig::SfAtom;
  using BLK_N = cutlass::detail::Sm1xxBlockScaledTensorConfig::Blk_N;
  using BLK_M = cutlass::detail::Sm1xxBlockScaledTensorConfig::Blk_M;

  struct SharedStorage { };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct TransformArguments {
    const void* ptr_src;
    void* ptr_dst;
    StrideSF dFlatTensorStride = {};
  };

  using TransformParams = TransformArguments;

  struct Arguments {
    ProblemShape problem_shape{};
    TransformArguments transform{};
    KernelHardwareInfo hw_info{};
  };

  struct Params {
    ProblemShape problem_shape{};
    TransformArguments transform{};
    KernelHardwareInfo hw_info{};
    void* workspace = nullptr;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
      ProblemShape{args.problem_shape},
      TransformParams{args.transform.ptr_src, args.transform.ptr_dst, args.transform.dFlatTensorStride},
      KernelHardwareInfo{args.hw_info},
      workspace
    };
  }

  static Status
  can_implement(Arguments const& args) {
    return Status::kSuccess;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    auto problem_shape_MNL = append<3>(params.problem_shape, 1);
    auto [M,N,L] = problem_shape_MNL;
    return dim3(ceil_div(M, size<0>(CtaTileShape{})), ceil_div(N, size<1>(CtaTileShape{})), L);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

private:
  CUTE_HOST_DEVICE
  auto
  get_src_tensor(Params const& params) {
    auto problem_shape_MNL = append<3>(params.problem_shape, 1);
    auto [M,N,L] = problem_shape_MNL;

    auto ptr_src = recast_ptr<ElementSF>(params.transform.ptr_src);
    if constexpr (SFLayoutConvertorKnob == ScalefactorLayoutConvertorKnob::ToBlockScaledLayout) {
      return make_tensor(make_gmem_ptr(ptr_src), make_layout(make_shape(M,N,L), params.transform.dFlatTensorStride));                    //(m,n,l)
    }
    else {
      return make_tensor(make_gmem_ptr(ptr_src), cutlass::detail::Sm1xxBlockScaledTensorConfig::tile_atom_to_shape(problem_shape_MNL));  //(m,n,l)
    }
  }


  CUTE_HOST_DEVICE
  auto
  get_dst_tensor(Params const& params) {
    auto problem_shape_MNL = append<3>(params.problem_shape, 1);
    auto [M,N,L] = problem_shape_MNL;

    auto ptr_dst = recast_ptr<ElementSF>(params.transform.ptr_dst);
    if constexpr (SFLayoutConvertorKnob == ScalefactorLayoutConvertorKnob::ToBlockScaledLayout) {
      return make_tensor(make_gmem_ptr(ptr_dst), cutlass::detail::Sm1xxBlockScaledTensorConfig::tile_atom_to_shape(problem_shape_MNL)); //(m,n,l)
    }
    else {
      return make_tensor(make_gmem_ptr(ptr_dst), make_layout(make_shape(M,N,L), params.transform.dFlatTensorStride));                   //(m,n,l)
    }
  }

public:
  CUTE_HOST_DEVICE
  void
  operator()(Params params, char* smem_buf = nullptr) {
    auto problem_shape_MNL = append<3>(params.problem_shape, 1);
    auto [M,N,L] = problem_shape_MNL;
    // Get the appropriate blocks for this thread block -- potential for thread block locality
    auto blk_shape = CtaTileShape{};                                                               // (BLK_M,BLK_N)
    auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    auto blk_coord_mnl = make_coord(int(m_coord), int(n_coord), int(l_coord));                           // (m,n,l)

    //////  Src tensor
    // Represent the full tensors
    Tensor mS_mnl = get_src_tensor(params);
    // Get batch slice
    Tensor mS_mn = mS_mnl(_,_,l_coord);                                                            // (m,k)
    // Slice to get the tiles this thread block is responsible for
    Tensor gS = local_tile(mS_mn, blk_shape, take<0,2>(blk_coord_mnl), Step<_1, _1>{});            // (BLK_M,BLK_N)
    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gS) * get<0>(blk_coord_mnl);                                    // M - BLK_M * m_coord
    auto n_max_coord = N - size<1>(gS) * get<1>(blk_coord_mnl);                                    // N - BLK_N * n_coord
    auto residue_mn = make_tuple(m_max_coord, n_max_coord);

    auto tiled_copy_g2r = make_tiled_copy(Copy_Atom<DefaultCopy, uint8_t>{},
                    Layout<Shape<_32, _4>, Stride< _4, _1>>{},
                    Layout<Shape <_1, _1>>{});

    auto thr_tiled_cp_g2r = tiled_copy_g2r.get_slice(threadIdx.x);
    auto tSgS = thr_tiled_cp_g2r.partition_S(gS);
    auto tSrS = make_fragment_like(tSgS);

    //////  Output tensor
    // Represent the full tensors
    Tensor mD_mnl = get_dst_tensor(params);
    // Get batch slice
    Tensor mD_mn = mD_mnl(_,_,l_coord);                                                            // (m,k)
    // Slice to get the tiles this thread block is responsible for
    Tensor gD = local_tile(mD_mn, blk_shape, take<0,2>(blk_coord_mnl), Step<_1, _1>{});            // (BLK_M,BLK_N)
    auto tiled_copy_r2g = make_tiled_copy_S(Copy_Atom<DefaultCopy, uint8_t>{}, tiled_copy_g2r);
    auto thr_tiled_cp_r2g = tiled_copy_r2g.get_slice(threadIdx.x);
    auto tDgD = thr_tiled_cp_r2g.partition_D(gD);
    auto tDrD = thr_tiled_cp_r2g.retile_S(tSrS);

    //////  Coord tensor
    auto cS = [&]() {
      if constexpr (SFLayoutConvertorKnob == ScalefactorLayoutConvertorKnob::ToBlockScaledLayout) {
        return make_identity_tensor(make_shape(unwrap(shape<0>(gS)), unwrap(shape<1>(gS))));
      }
      else {
        return make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
      }
    }();
    Tensor tCcS = thr_tiled_cp_g2r.partition_S(cS);

    #if 0
    if(thread0()) {
      print("problem_shape_MNL   ");print(problem_shape_MNL);print("\n");
      print("=============== Src Tensor ============== \n");
      print("mS_mnl     ");print(mS_mnl);print("\n");
      print("mS_mn      ");print(mS_mn);print("\n");
      print("gS         ");print(gS);print("\n");
      print("tSgS       ");print(tSgS);print("\n");
      print("tSrS       ");print(tSrS);print("\n");

      print("=============== Dst Tensor ============== \n");
      print("mD_mnl     ");print(mD_mnl);print("\n");
      print("mD_mn      ");print(mD_mn);print("\n");
      print("gD         ");print(gD);print("\n");
      print("tDgD       ");print(tDgD);print("\n");
      print("tDrD       ");print(tDrD);print("\n");

      print("=============== crd Tensor ============== \n");
      print("residue_mn ");print(residue_mn);print("\n");
      print("tCcS       ");print(tCcS);print("\n");
    }
    #endif

    Tensor tDpD = cute::lazy::transform(tCcS, [&] (auto const& c) { return elem_less(c, residue_mn); });
    if constexpr (SFLayoutConvertorKnob == ScalefactorLayoutConvertorKnob::ToBlockScaledLayout) {
      fill(tSrS, ElementSF(0));
      copy_if(tiled_copy_g2r, tDpD, tSgS, tSrS);
      copy(tiled_copy_r2g, tDrD, tDgD);             // w/o pred because the memory is 128*4 align
    }
    else {
      copy(tiled_copy_g2r, tSgS, tSrS);             // w/o pred because the memory is 128*4 align
      copy_if(tiled_copy_r2g, tDpD, tDrD, tDgD);
    }
  }
};

} // End namespace cutlass
