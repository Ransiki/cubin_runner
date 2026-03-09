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
#pragma once

/*! \file
  \brief W mode make_tma_copy
  // {$nv-internal-release file}
*/

#include "cute/arch/copy_sm100_tma.hpp"
#include "cute/atom/copy_traits_sm100_im2col.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cute/tensor.hpp"

#include "cute/algorithm/prefetch.hpp"
#include "cutlass/cuda_host_adapter.hpp"

namespace cute
{

// Utility for unpacking TMA_LOAD_W arguments into a CopyOp
template <class CopyOp>
struct TMA_LOAD_W_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src, // tile of the transformed global activation (A) tensor
              Tensor<TD,DLayout>                & dst) // shared memory tile
  {
    auto src_coord_cwhdn = src(Int<0>{});
    constexpr int NumTotalModes = decltype(rank(src_coord_cwhdn))::value;
    static_assert((NumTotalModes >= 3) || (NumTotalModes <= 5), " TMALDG(W) only support 3/4/5 modes.");

    // Append the packed_offset argument for the copyop
    auto src_coord_cwhdn_offset = append(append(src_coord_cwhdn, traits.w_halo_), traits.w_offset_);

    if constexpr (detail::is_prefetch<CopyOp>) {
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                   traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                   src_coord_cwhdn_offset, tuple_seq<decltype(src_coord_cwhdn_offset)>{});
    } else {
      static_assert(is_smem<TD>::value, "SM100_TMA_LOAD_W requires that the destination point to shared memory.");
      void* dst_ptr = raw_pointer_cast(dst.data());
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                   traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                   make_tuple(dst_ptr), seq<0>{},
                                   src_coord_cwhdn_offset, tuple_seq<decltype(src_coord_cwhdn_offset)>{});
    }
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD_W /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

using WModeTmaDescriptor = Im2ColTmaDescriptor;

struct SM100_TMA_LOAD_W_OP : SM100_TMA_LOAD_W {};

/// @brief Non-executable specialization of Copy_Traits for SM100
///   w mode TMA load, with TMA descriptor but no barrier.
///
/// Use `.with(memory_barrier, packed_offset)` to construct an executable version.
template <class NumBitsPerTMA, class BaseCorner, class GmemBasisStrides>
struct Copy_Traits<SM100_TMA_LOAD_W, NumBitsPerTMA, BaseCorner, GmemBasisStrides>
{
  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  WModeTmaDescriptor tma_desc_;
  // ArithmeticTuple representing the "zero" "pointer"
  BaseCorner base_corner_;
  // Basis strides for the TMA tensor
  GmemBasisStrides gbasis_strides_;

  CUTE_HOST_DEVICE constexpr
  WModeTmaDescriptor const *
  get_tma_descriptor() const
  {
    return &tma_desc_;
  }

  // XXX: consider removing packed_offset from with() when we know how dilation/stride will impact TMALDG.W
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_LOAD_W_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar, uint16_t w_halo, uint16_t w_offset) const
  {
    return {{}, {&tma_desc_, &tma_mbar}, w_halo, w_offset};
  }

  // Activition tma tensor:
  template <class TransformedActivationShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(TransformedActivationShape const& result_shape) const
  {
    // result_shape - ((Q,N),C,P,Z)
    // shape must be rank-4
    CUTE_STATIC_ASSERT_V(rank(result_shape) == Int<4>{});
    auto tensor_multimode = make_tensor(ArithmeticTupleIterator(base_corner_), result_shape, gbasis_strides_);
    auto tensor_linear = make_identity_tensor(
      make_shape(size<0>(result_shape), shape<1>(result_shape), shape<2>(result_shape), shape<3>(result_shape)));
    return make_tensor(tensor_multimode.data(),
                       composition(tensor_multimode.layout(),
                                   tensor_linear(Int<0>{}),
                                   tensor_linear.layout()));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD_W before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

/// @brief Executable specialization of Copy_Traits for SM100 w mode
///   TMA load, with TMA descriptor and barrier.
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_LOAD_W_OP, NumBitsPerTMA>
     : TMA_LOAD_W_Unpack<SM100_TMA_LOAD_W_OP>
{
  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  tuple<
  WModeTmaDescriptor const*,
  uint64_t* // smem mbarrier
  > const opargs_;
  uint16_t const w_halo_;
  uint16_t const w_offset_;
};

template <class NumBitsPerTMA, class... Args>
struct Copy_Traits<SM100_TMA_LOAD_W::PREFETCH, NumBitsPerTMA, Args...>
     : TMA_LOAD_W_Unpack<SM100_TMA_LOAD_W::PREFETCH>
{
  using ThrID = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  tuple<WModeTmaDescriptor const*> const opargs_;
  uint16_t const w_halo_;
  uint16_t const w_offset_;

  template <class... CopyArgs>
  CUTE_HOST_DEVICE
  Copy_Traits(Copy_Traits<CopyArgs...> const& traits)
    : opargs_({traits.tma_desc_}), w_offset_(traits.w_offset_), w_halo_(traits.w_halo_) {}
};

namespace detail {

template <class EngineA, class LayoutA,
          class SmemSwizzle,
          size_t N>
CUTE_HOST
auto
make_w_tma_copy_desc(
    Tensor<EngineA, LayoutA> const& tensor_cwhdn,       // (C,W,H,D,N)
    uint32_t                        range_c,            // TILE_C
    uint32_t                        range_wdn,          // TILE_WDN
    SmemSwizzle                     smem_swizzle,       // Swizzle
    cute::array<int32_t,1>   const& lower_w,
    cute::array<int32_t,1>   const& upper_w,
    cute::array<int32_t,N>   const& lower_srt,
    cute::array<int32_t,N>   const& traversal_stride)
{
  static_assert(is_gmem<EngineA>::value, "Tensor must point to GPU global memory.");
  using value_type = typename EngineA::value_type;

  constexpr uint32_t num_total_modes   = LayoutA::rank;
  constexpr int      num_spatial_modes = num_total_modes - 2;
  static_assert(num_spatial_modes == N);

  // Gmem starting address
  void* gmem_address = (void*) raw_pointer_cast(tensor_cwhdn.data());

  // Gmem extents are just the tensor shape
  cute::array<uint64_t, 5> gmem_prob_shape = {1,1,1,1,1};
  for_each(make_seq<num_total_modes>{}, [&](auto i) {
    gmem_prob_shape[i] = static_cast<uint64_t>(shape<i>(tensor_cwhdn));
  });

  // Gmem strides are byte strides of the activation tensor in CWHDN order
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
  for_each(make_seq<num_total_modes>{}, [&](auto i) {
    gmem_prob_stride[i] = sizeof(value_type) * stride<i>(tensor_cwhdn);
  });

  // Traversal strides are a function of the dilation shape
  // corresponding to spatial (WHD) modes.
  cute::array<uint32_t, 5> tma_traversal_strides = {1,1,1,1,1};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_traversal_strides[i+1] = static_cast<uint32_t>(get<i>(traversal_stride));
  });

  WModeTmaDescriptor tma_desc;

// TODO: check if it works well in the public TK https://jirasw.nvidia.com/browse/CFK-22323 // {$nv-release-never}
#if defined(CUTE_USE_PUBLIC_TMA_DESCRIPTOR) // {$nv-internal-release}
#if (__CUDACC_VER_MAJOR__ >= 12)

  CUtensorMapDataType     tma_format      = TMA::to_CUtensorMapDataType<value_type>();
  CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUtensorMapFloatOOBfill tma_oob_fill    = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
  TMA::SmemSwizzleBits    swizzle_bits    = detail::get_tma_swizzle_bits(smem_swizzle);
  TMA::SmemSwizzleBase    swizzle_base    = detail::get_tma_swizzle_base(smem_swizzle);
  CUtensorMapSwizzle      tma_swizzle     = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);

  CUresult encode_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeIm2col)(
      &tma_desc,
      tma_format,
      num_total_modes,
      gmem_address,
      gmem_prob_shape.data(),
      gmem_prob_stride.data() + 1, // gmem_prob_stride[0] implicitly sizeof(value_type)
      lower_w.data(),
      upper_w.data(),
      range_c,
      range_wdn,
      tma_traversal_strides.data(),
      tma_interleave,
      tma_swizzle,
      tma_l2Promotion,
      tma_oob_fill);

  // {$nv-internal-release begin}
  // CTK < 13.2 has a bug where the TMA desc OOB addr generation is not set correctly.
  // In order to do this properly we need to set bit 85 to 1 if the cosize
  // of the tensor is large enough (128KiB) and 0 if it is too small. 
  // On Hopper, since this is reserved it will do nothing
  // {$nv-internal-release end}
  
  int driver_version = 0;
  CUresult driver_version_result = cuDriverGetVersion(&driver_version);
  assert(driver_version_result == CUDA_SUCCESS);
  if (driver_version <= 13010) {
    if (cute::bits_to_bytes(
          cute::cosize(tensor_cwhdn.layout()) *
          cute::sizeof_bits<typename EngineA::value_type>::value) < 131072) {
      reinterpret_cast<uint64_t*>(&tma_desc)[1] &= ~(1llu << 21);
    }
  }

  // The extra asserts help indicate the error's cause.
  assert(encode_result != CUDA_ERROR_DEINITIALIZED);
  assert(encode_result != CUDA_ERROR_NOT_INITIALIZED);
  assert(encode_result != CUDA_ERROR_INVALID_CONTEXT);
  assert(encode_result != CUDA_ERROR_INVALID_VALUE);
  assert(encode_result == CUDA_SUCCESS);

#endif // (__CUDACC_VER_MAJOR__ >= 12)
#else // {$nv-internal-release begin}

  //
  // Construct the internal descriptor
  //

  // general info
  tma_desc.type_    = TMA::Type::IM2COL;
  tma_desc.version_ = 0;
  tma_desc.format_  = is_same<value_type, tfloat32_t>::value ? TMA::to_Format<float>() : TMA::to_Format<value_type>();
  tma_desc.toTF32_  = is_same<value_type, tfloat32_t>::value ? TMA::F32toTF32::ENABLE : TMA::F32toTF32::DISABLE;
  tma_desc.l2promo_ = TMA::L2Promotion::DISABLE;

  // gtensor info
  tma_desc.dim_           = num_total_modes - 1;
  tma_desc.interleaved_   = TMA::Interleave::DISABLE;
  tma_desc.oobfill_       = TMA::OOBFill::ZERO;
  tma_desc.start_address_ = reinterpret_cast<uint64_t>(gmem_address) >> 4;

  // TMA smem swizzle type
  tma_desc.swizzle_bits_ = detail::get_tma_swizzle_bits(smem_swizzle);
  tma_desc.swizzle_base_ = detail::get_tma_swizzle_base(smem_swizzle); // {$nv-internal-release}

  tma_desc.oob_addr_gen_mode_ = get_tma_oob_addr_mode(tensor_cwhdn);    // {$nv-internal-release}
  // TMA IM2COL parameters

  assert(range_c     >= (uint32_t(1)));                      // Size must be min 1
  assert(range_c     <= (uint32_t(1) << 8));                 // Size must be max 2^8
  assert(range_wdn  >= (uint32_t(1)));                       // Size must be min 1
  assert(range_wdn  <= (uint32_t(1) << 8));                  // Size must be max 2^8

  tma_desc.range_c_    = static_cast<uint8_t>(range_c   - 1);
  tma_desc.range_ndhw_ = static_cast<uint8_t>(range_wdn - 1);

  tma_desc.lower_corner_3d_w_ = int16_t(lower_w[0]);
  tma_desc.upper_corner_3d_w_ = int16_t(upper_w[0]);

  // TMA smem mode strides

  assert(tma_traversal_strides[0] >= (uint32_t(1)));         // Size must be min 1
  assert(tma_traversal_strides[0] <= (uint32_t(8)));         // Size must be max 8
  assert(tma_traversal_strides[1] >= (uint32_t(1)));         // Size must be min 1
  assert(tma_traversal_strides[1] <= (uint32_t(8)));         // Size must be max 8
  assert(tma_traversal_strides[2] >= (uint32_t(1)));         // Size must be min 1
  assert(tma_traversal_strides[2] <= (uint32_t(8)));         // Size must be max 8
  assert(tma_traversal_strides[3] >= (uint32_t(1)));         // Size must be min 1
  assert(tma_traversal_strides[3] <= (uint32_t(8)));         // Size must be max 8
  assert(tma_traversal_strides[4] >= (uint32_t(1)));         // Size must be min 1
  assert(tma_traversal_strides[4] <= (uint32_t(8)));         // Size must be max 8

  tma_desc.tstride0_ = uint8_t(tma_traversal_strides[0] - 1);
  tma_desc.tstride1_ = uint8_t(tma_traversal_strides[1] - 1);
  tma_desc.tstride2_ = uint8_t(tma_traversal_strides[2] - 1);
  tma_desc.tstride3_ = uint8_t(tma_traversal_strides[3] - 1);
  tma_desc.tstride4_ = uint8_t(tma_traversal_strides[4] - 1);

  // Address must be 16B-aligned
  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);

  assert(gmem_prob_shape[0] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));         // Size must be max 2^32

  // TMA gmem mode sizes
  tma_desc.size0_ = uint32_t(gmem_prob_shape[0] - 1);
  tma_desc.size1_ = uint32_t(gmem_prob_shape[1] - 1);
  tma_desc.size2_ = uint32_t(gmem_prob_shape[2] - 1);
  tma_desc.size3_ = uint32_t(gmem_prob_shape[3] - 1);
  tma_desc.size4_ = uint32_t(gmem_prob_shape[4] - 1);

  // TMA descriptor does not store the zeroth stride and assumes it is sizeof(T) == one element.
  assert(gmem_prob_stride[0] == sizeof(value_type) && "Majorness of smem doesn't match majorness of gmem");

  assert((gmem_prob_stride[1]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)

  // TMA gmem mode strides
  tma_desc.stride0_ = uint64_t(gmem_prob_stride[1] >> 4);
  tma_desc.stride1_ = uint64_t(gmem_prob_stride[2] >> 4);
  tma_desc.stride2_ = uint64_t(gmem_prob_stride[3] >> 4);
  tma_desc.stride3_ = uint64_t(gmem_prob_stride[4] >> 4);

#endif // {$nv-internal-release end}

  return tma_desc;
}

/// Make a Copy Atom for W mode TMA load to match dynamic CGA API.
///
/// @param copy_op The copy implementation: such as
///   SM100_TMA_LOAD_W
///
/// @param tensor_cwhdn The global tensor to use for W mode TMA loads.
///   For Fprop convolutions, this is the activation tensor.  This is
///   the "original tensor that points to global memory, not the
///   coordinate (W-transformed) tensor.
///
/// @param slayout Layout of shared memory tile.
///
/// @param traversal_stride The traversal strides convolution
///   parameter.
///
/// @return Copy Atom specialization for W TMA loads.
template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride,
          size_t N>
CUTE_HOST_RTC
auto
make_tma_copy_atom_w(CopyOp                  const& copy_op,
                     Tensor<GEngine,GLayout> const& gtensor_full,       // ((W,H,D,N),C)
                     SLayout                 const& slayout,
                     Layout<VShape,VStride>  const& cta_v_map,          // CTA vid -> gmem coord
                     cute::array<int32_t,1>  const& start_coord_w,
                     cute::array<int32_t,1>  const& lower_w,
                     cute::array<int32_t,1>  const& upper_w,
                     cute::array<int32_t,1>  const& lower_d,
                     cute::array<int32_t,N>  const& lower_srt,
                     cute::array<int32_t,N>  const& traversal_stride)   // [dQ,dP,dZ]
{

  //
  // TMA parameter checking
  //

  CUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
    "TMA requires CTA_Tile and SLayout top-level shape equivalence.");
  CUTE_STATIC_ASSERT_V(rank(get<0>(gtensor_full.shape())) == Int<4>{},
    "Tensor dim should be 5D(NDHWC)");

  //
  // TMA slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // trunc_smem_idx -> trunc_smem_coord

  // Map from smem idx to a gmem mode
  auto sidx_to_gmode = coalesce(composition(cta_v_map, inv_smem_layout));

#if 0
  print("gtensor_full     : "); print(gtensor_full.layout()); print("\n");
  print("s_layout         : "); print(slayout); print("\n");
  print("cta_v_map        : "); print(cta_v_map); print("\n");
  print("inv_smem         : "); print(inv_smem_layout); print("\n");
  print("sidx_to_gmode    : "); print(sidx_to_gmode); print("\n");
#endif

  //
  // TMA gtensor manipulation
  //

  // Get a gtensor without h dimension
  auto gtensor = make_layout(
      make_shape(remove<1>(shape<0>(gtensor_full)), shape<1>(gtensor_full)),
      make_stride(remove<1>(stride<0>(gtensor_full)), stride<1>(gtensor_full)));

  // Generate a TupleBasis for the gtensor
  // XXX: The product_each is the linearized multimode HACK    // {$nv-release-never}
  auto glayout_basis = make_identity_layout(product_each(shape(gtensor)));

  // Tile the modes of gtensor with the truncated cta_v_map o inv_smem_layout_trunc
  auto tma_layout_full = flatten(composition(glayout_basis, sidx_to_gmode));

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(tma_layout_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });

  static_assert(smem_rank >= 2, "W mode expects at least 2 modes of the smem to vectorize with gmem.");
  // W mode uses a maximum of 2 modes
  constexpr int smem_tma_rank = cute::min(int(smem_rank), 2);

  // Keep only the static-1 basis modes into gmem
  auto tma_layout_trunc = take<0,smem_tma_rank>(tma_layout_full);

#if 0
  print("glayout_basis   : "); print(glayout_basis); print("\n");
  print("tma_layout_full : "); print(tma_layout_full); print("\n");
  print("tma_layout_trunc: "); print(tma_layout_trunc); print("\n");
#endif

  // w mode not support mcast, so directly use tma_layout_trunc to get shape and basis.
  auto range_c   = size<0>(tma_layout_trunc);
  auto range_wdn = size<1>(tma_layout_trunc);

  Tensor gtensor_cwhdn = make_tensor(
      gtensor_full.data(),
      flatten(make_layout(make_layout(basis_get(stride<0>(tma_layout_trunc), gtensor_full.shape()),
                                      basis_get(stride<0>(tma_layout_trunc), gtensor_full.stride())),
                          make_layout(basis_get(stride<1>(tma_layout_trunc), gtensor_full.shape()),
                                      basis_get(stride<1>(tma_layout_trunc), gtensor_full.stride())))));

  auto tma_desc = make_w_tma_copy_desc(
      gtensor_cwhdn, range_c, range_wdn, get_swizzle_portion(slayout),
      lower_w, upper_w, lower_srt,
      traversal_stride);

  //
  // Construct the Copy_Traits
  //
  // Compute base coord
  // (C,W,H,D,N) offset starts at (0,-dW,0,-dD,0)
  auto corner_offset = start_coord_w[0];
  auto base_coord = as_arithmetic_tuple(flatten(
    cute::make_tuple(
      Int<0>{},
      flatten(corner_offset),
      Int<0>{},
      lower_d[0],
      Int<0>{})
  ));

  // Construct the strides of the coordinate tensor
  auto spatial_dim = rank(get<0>(gtensor_full.shape()));
  auto tensor_dim = rank(flatten(gtensor_full.shape()));
  auto basis = make_basis_like(base_coord);                                         // (C,Q,H,Z,N) => (C,W,H,D,N)
  auto qn_basis = append(get<1>(basis), get<4>(basis));                             // (Q,N)
  auto qn_stride = append(elem_scale(get<0>(traversal_stride), get<0>(qn_basis)),
                          get<1>(qn_basis));                                        // Scale with stride on spatial dim (w/o batch)
  auto c_basis = get<0>(basis);
  auto c_stride = c_basis;
  auto h_basis = get<2>(basis);
  auto h_stride = elem_scale(1, h_basis); // Do not need to scale with stride because it is H instead of P
  auto z_basis = get<3>(basis);
  auto z_stride = elem_scale(get<2>(traversal_stride), z_basis);
  auto gbasis_strides = make_stride(qn_stride, c_stride, h_stride, z_stride);

  using T = typename GEngine::value_type;
  constexpr int num_bits_per_tma = decltype(size(tma_layout_trunc))::value * sizeof(T) * 8;
  using Traits = Copy_Traits<CopyOp, cute::C<num_bits_per_tma>, decltype(base_coord), decltype(gbasis_strides)>;
  using Atom   = Copy_Atom<Traits, T>;

  // XXX: Save lower_corner in the Traits for get_tma_tensor?  // {$nv-release-never}

  Traits tma_traits{tma_desc, base_coord, gbasis_strides};

  // Return the Copy_Atom
  return Atom{tma_traits};
}

/// Make a TiledCopy for W mode TMA load.
///
/// @param copy_op The copy implementation: such as
///   SM100_TMA_LOAD_W
///
/// @param tensor_cwhdn The global tensor to use for W mode TMA loads.
///   For Fprop convolutions, this is the activation tensor.  This is
///   the "original tensor that points to global memory, not the
///   coordinate (W-transformed) tensor.
///
/// @param slayout Layout of shared memory tile.
///
/// @param traversal_stride The traversal strides convolution
///   parameter.
///
/// @return TiledCopy specialization for W TMA loads.
template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride,
          size_t N>
CUTE_HOST_RTC
auto
make_tma_copy_w(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor_full,       // ((W,H,D,N),C)
                SLayout                 const& slayout,
                Layout<TShape,TStride>  const& cta_t_map,          // CTA tid -> logical TMA tid
                Layout<VShape,VStride>  const& cta_v_map,          // CTA vid -> gmem coord
                cute::array<int32_t,1>  const& start_coord_w,
                cute::array<int32_t,1>  const& lower_w,
                cute::array<int32_t,1>  const& upper_w,
                cute::array<int32_t,1>  const& lower_d,
                cute::array<int32_t,N>  const& lower_srt,
                cute::array<int32_t,N>  const& traversal_stride)   // [dQ,dP,dZ]
{
  Copy_Atom atom = make_tma_copy_atom_w<TmaInternalType>(copy_op, gtensor_full, slayout, cta_v_map,
                                                         start_coord_w, lower_w, upper_w, lower_d,
                                                         lower_srt, traversal_stride);
  //
  // Construct the TiledCopy
  // May be deprecated as make_tma_copy_tiled // {$nv-internal-release}
  //

  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GEngine::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  // CTA T -> smem idx
  auto layout_t = make_layout(cosize(cta_t_map), safe_div(num_elems_per_tma, cosize(cta_t_map)));
  // CTA TID -> smem coord
  auto layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map));
  // Combine with the T mapping
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);

#if 0
  print("cta_tiler : "); print(cta_tiler); print("\n");
  print("layout_v : "); print(layout_v); print("\n");
  print("layout_V : "); print(layout_V); print("\n");
  print("layout_t : "); print(layout_t); print("\n");
  print("layout_T : "); print(layout_T); print("\n");
  print("layout_TV : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

} // end namespace detail

#if !defined(__CUDACC_RTC__)
/** Make a CuTe CTA-collective TiledCopy for a TMA operation.
 *  It works on nq tiled kernel's matrix B, Activation tensor.
 *
 * @param CopyOp The target copy operation: SM100_TMA_LOAD_W
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param cluster_tile The Cluster-local tile that each Cluster will be tiling GMEM with.
 *                     This is often the cluster_tile_shape that is used to tile the GMEM:
 *                       local_tile(gtensor, cluster_tile_shape, cluster_coord)
 *                         -> Cluster-local tile of GMEM
 * @param mma The TiledMMA that defines the Cluster-Tile to Block-Tile partitioning.
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tile,
          class... Args,
          size_t N>
CUTE_HOST
auto
make_w_tma_copy_B_sm100(CopyOp                  const& copy_op,
                        Tensor<GEngine,GLayout> const& gtensor,        // (N,K,...)
                        SLayout                 const& slayout,        // (MMA, MMA_N, MMA_K)
                        Cluster_Tile            const& cluster_tile,   // (TILE_M,TILE_N,TILE_K)
                        TiledMMA<Args...>       const& mma,
                        cute::array<int32_t,1>  const& start_coord_w,
                        cute::array<int32_t,1>  const& lower_w,
                        cute::array<int32_t,1>  const& upper_w,
                        cute::array<int32_t,1>  const& lower_d,
                        cute::array<int32_t,N>  const& lower_srt,
                        cute::array<int32_t,N>  const& traversal_stride)
{
  constexpr int R = GLayout::rank;
  // Keep only NK modes from MNK
  auto cluster_tile_shape = append<R>(make_shape(get<1>(cluster_tile), get<2>(cluster_tile)), Int<1>{});
  // cga tile coord -> gtensor coord
  auto cluster_layout = make_identity_layout(cluster_tile_shape);
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(cluster_layout))(_, repeat<R>(_));

  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, SM100_TMA_LOAD_W>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      // TODO: add multicast op https://jirasw.nvidia.com/browse/CFK-22323 // {$nv-release-never}
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides));

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_w<TmaType>(copy_op, gtensor, slayout,
                                          cta_t_map, cta_v_tile,
                                          start_coord_w, lower_w, upper_w,
                                          lower_d, lower_srt, traversal_stride);
}

#endif // !defined(__CUDACC_RTC__)

#if !defined(__CUDACC_RTC__)
/** Make a CuTe CTA-collective Copy Atom for a TMA operation.
 *  It works on nq tiled kernel's matrix B, Activation tensor.
 *
 * @param CopyOp The target copy operation: SM100_TMA_LOAD_W
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param mma_tiler The ClusterMMA tile (CTA tile * 1SM or 2SM) that each ClusterMMA will be tiling GMEM with
 * @param mma The TiledMMA that defines the ClusterMMA-Tile to Block-Tile partitioning.
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          size_t N>
CUTE_HOST
auto
make_w_tma_atom_B_sm100(CopyOp                  const& copy_op,
                        Tensor<GEngine,GLayout> const& gtensor,        // (N,K,...)
                        SLayout                 const& slayout,        // (MMA, MMA_N, MMA_K)
                        MMA_Tiler               const& mma_tiler,   // (TILE_M,TILE_N,TILE_K)
                        TiledMMA<Args...>       const& mma,
                        cute::array<int32_t,1>  const& start_coord_w,
                        cute::array<int32_t,1>  const& lower_w,
                        cute::array<int32_t,1>  const& upper_w,
                        cute::array<int32_t,1>  const& lower_d,
                        cute::array<int32_t,N>  const& lower_srt,
                        cute::array<int32_t,N>  const& traversal_stride)
{
  // Keep only NK modes from MNK
  auto mma_tiler_nk = remove<0>(mma_tiler);
  // cga tile coord -> gtensor coord
  auto g_tile = make_identity_layout(mma_tiler_nk);       // (TILE_N, TILE_K)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));  // (MMA, MMA_N, MMA_K)

  static_assert(is_same_v<CopyOp, SM100_TMA_LOAD_W>, "TMA should be SM100_TMA_LOAD_W");

  // No need to pass in num_multicast as we do not support mcast in w mode.
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom_w<TmaType>(copy_op, gtensor, slayout, cta_v_tile,
                                               start_coord_w, lower_w, upper_w,
                                               lower_d, lower_srt, traversal_stride);
}

#endif // !defined(__CUDACC_RTC__)

} // namespace cute
