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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/transform/collective/sm90_wgmma_transpose.hpp"
#include "cutlass/trace.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop that source A operand from registers
template <
  int Stages,
  int NumComputeMtxs,
  bool AllBands,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class ElementScalar_>
struct CollectiveMma<
    MainloopSm90TmaGmmaRmemAWarpSpecializedFastF32PreTransformed<Stages, NumComputeMtxs, false, AllBands, true, ClusterShape, KernelSchedule, ElementScalar_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {

  // Real/imag mode (0 for real part, 1 for imag part).
  // e.g.: complex<float> A(4,6) -> float A_(2,4,6) where A_(0,:,:) are real parts
  // Give it a name so that it doesn't look like a magic number
  static constexpr int ImMode = 2;

  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaRmemAWarpSpecializedFastF32PreTransformed<Stages, NumComputeMtxs, false, AllBands, true, ClusterShape, KernelSchedule, ElementScalar_>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ElementScalar = ElementScalar_;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  // Tiled Gmem access shape for B
  using TileShapeGmem = decltype(append(take<0,2>(TileShape{}), make_tuple(_1{},_1{},shape<2>(TileShape{}))));

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_same_v<ElementB, typename TiledMma::ValTypeB>, "B should directly feed int Mma.");
  static_assert(is_complex_v<typename TiledMma::ValTypeC>, "Complex TiledMma expected. Construct from MMA_Atom_PlanarComplex2Complex.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<ImMode>{}, Int<NumComputeMtxs>{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3,_4,_5>, Step<_1,_2,_3,_4,_5>>{}));
  // Smem layout for exponents
  using SmemLayoutAScale = decltype(make_layout(make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), _1{}), make_stride(_1{},_0{},_0{})));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                    cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  using InternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;
  using InternalElementB = uint_bit_t<sizeof_bits_v<ElementB>>;

  struct SharedStorage {
    struct TensorStorage {
      alignas(128) cute::array<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(128) cute::array<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
      // TODO: Use LDGSTS w/ alignments https://jirasw.nvidia.com/browse/CUT-1746 // {$nv-internal-release}
      cute::array<ElementScalar, shape<0>(TileShape{})> smem_A_scale;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    ElementScalar const* ptr_A_scale = nullptr;
    int64_t dA_scale = 0;
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  struct Params {
    // Modes for exponents: [M,(K),(S),L]: where one max exp is gathered across (K) to produce all (S)lices
    using EXP_A = decltype(local_tile(
        make_tensor(static_cast<ElementScalar const*>(nullptr), make_layout(make_shape(1,1,1), make_stride(_1{},0,int64_t(1)))),
        TileShape{},
        make_coord(_,_,_),
        Step<_1,X,_1>{}));
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_0{}),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_0{},_0{},_0{}),
        TileShapeGmem{},
        ClusterShape{}));
    EXP_A a_scale;
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor tensor_a_scale = local_tile(
        make_tensor(args.ptr_A_scale, make_layout(make_shape(M,K,L), make_stride(_1{},0,args.dA_scale))),
        TileShape{},
        make_coord(_,_,_),
        Step<_1,X,_1>{});

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
      GmemTiledCopyA{},
      tensor_a,
      SmemLayoutA{}(_,_,_0{}),
      TileShape{},
      ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
      GmemTiledCopyB{},
      tensor_b,
      SmemLayoutB{}(_,_,_0{},_0{},_0{}),
      TileShapeGmem{},
      ClusterShape{});
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;
    return {
      tensor_a_scale,
      tma_load_a,
      tma_load_b,
      transaction_bytes,
      transaction_bytes_mk,
      transaction_bytes_nk
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(make_shape(N,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr uint32_t TmaTransactionBytesMK = cutlass::bits_to_bytes(
    size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<InternalElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = cutlass::bits_to_bytes(
    size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * (ImMode * NumComputeMtxs) * static_cast<uint32_t>(sizeof_bits<InternalElementB>::value));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,(s,k),l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,(s,k),l)
  /// [s] is the additional mode for different compute matrices that would eventually be contracted
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L));
                                                                                                      // (n,(im,s,k),l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1,X,_1>{});         // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShapeGmem{}, make_coord(_,_,_), Step<X,_1,_1>{});
                                                                                          // (BLK_N,BLK_K,n,(im,s,k),l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter_a, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});   // (BLK_N,BLK_K,im,s,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      // Copy the exponents before any tile
      Tensor sScale = make_tensor(make_smem_ptr(shared_tensors.smem_A_scale.data()), SmemLayoutAScale{});
                                                                                                     // (BLK_M,BLK_K,1)
      if (mainloop_params.a_scale.data()) {
        Tensor gScale = mainloop_params.a_scale(_,_,m_coord,0,l_coord);                                // (BLK_M,BLK_K)
        copy(gScale(_,0), sScale(_,0,0));
      }
      else {
        // If nullptr is found in the scale factor field, initialize sScale with ones
        cute::transform(sScale(_,0,0), sScale(_,0,0), [&] (auto& x) { return static_cast<cute::remove_cvref_t<decltype(x)>>(1); });
      }

      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                              // (BLK_M,BLK_K,(im,s,k))

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                          // (TMA,TMA_N,TMA_K,(im,s,k))
      Tensor tBsB = block_tma_b.partition_D(sB);                                         // (TMA,TMA_N,TMA_K,im,s,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> CTA, value
      auto cta_layout_mnk = Layout<typename DispatchPolicy::ClusterShape>{};
      auto cta_coord_mnk = make_coord(cluster_local_block_id.x, cluster_local_block_id.y, Int<0>{});
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        mcast_mask_a = cute::create_tma_multicast_mask<1>(cta_layout_mnk, cta_coord_mnk);
      }
      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        mcast_mask_a = cute::create_tma_multicast_mask<0>(cta_layout_mnk, cta_coord_mnk);
      }

      // Infer iterator coordinates of B from A
      auto k_tile_iter_b = cute::make_coord_iterator(make_coord(0, 0, *k_tile_iter_a), shape<2>(gB));

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();

        // Copy A as-is
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter_a), tAsA(_,_,_,write_stage));
        ++k_tile_iter_a;

        // Copy all low-precision compute matrices from B
        CUTLASS_PRAGMA_UNROLL
        for (int mtx = 0; mtx < NumComputeMtxs; ++mtx) {
          CUTLASS_PRAGMA_UNROLL
          for (int im = 0; im < ImMode; ++im) {
            copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter_b), tBsB(_,_,_,im,mtx,write_stage));
            ++k_tile_iter_b;
          }
        }

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // Warpper over cute::gemm to compute all mtxs, real and imag
  template <
    class TiledMma,
    class TensorA,
    class TensorB,
    class TensorC
  >
  CUTLASS_DEVICE void
  dispatch_mma(
      TiledMma &tiled_mma,
      TensorA &&rA,
      TensorB &&rB,
      TensorC &accum) {

    CUTLASS_PRAGMA_UNROLL
    for (int mtx_a = 0; mtx_a < NumComputeMtxs; ++mtx_a) {
      CUTLASS_PRAGMA_UNROLL
      for (int mtx_b = 0; mtx_b < NumComputeMtxs; ++mtx_b) {
        if (AllBands || mtx_a + mtx_b < NumComputeMtxs) {
          cute::gemm(tiled_mma, rA(_,_,mtx_a), rB(_,_,mtx_b), accum);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
      }
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgCEngine, class FrgCLayout
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      Tensor<FrgCEngine, FrgCLayout>& accum_c32,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {

    static_assert(is_rmem<Tensor<FrgCEngine, FrgCLayout>>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout for A must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 5, "Smem layout for B must be rank 5.");
    static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops must specify a non-void copy atom for RF sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(is_complex_v<typename FrgCEngine::value_type>, "Complex accumulator expected.");

    // {$nv-internal-release begin}
    /// Ideally, exposure of downcasting complex<float> to float should happen within MMA_Op instead of in collectives,
    /// but the compiler would emit extra mov.b32 instructions that case and OCG will spill them to new registers.
    // {$nv-internal-release end}
    Tensor accum = recast<typename FrgCEngine::value_type::value_type>(accum_c32);                          // (2V,M,N)

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});     // (BLK_N,BLK_K,im,s,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCsA = thread_mma.partition_A(sA); // Compile-time checks only.                    // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = make_tensor<ElementA>(thread_mma.partition_A(sA(_,_,_0{})).shape());        // (MMA,MMA_M,MMA_K)
    Tensor tCrA = make_tensor<typename TiledMma::FrgTypeA>(append(tCrA_load.shape(), Int<NumComputeMtxs>{}));
                                                                                                 // (MMA,MMA_M,MMA_K,s)

    Tensor tCsB = thread_mma.partition_B(sB);                                            // (MMA,MMA_N,MMA_K,im,s,PIPE)
    Tensor tCrB_ = thread_mma.make_fragment_B(tCsB);                                     // (MMA,MMA_N,MMA_K,im,s,PIPE)

    static_assert(size<0>(tCrB_) == 1, "B fragment by ThreadMma must be a descriptor and have unit first dimension.");

    // MMA_Op has to be fed with real & imaginary parts. Move ImMode to MMA for B:
    Tensor tCrB = composition(tCrB_, make_ordered_layout(                                // (MMA_im,MMA_N,MMA_K,s,PIPE)
        tuple_cat(prepend(take<1,3>(tCrB_.shape()), Int<ImMode>{}), take<4,6>(tCrB_.shape())), Step<_3,_1,_2,_4,_5>{}));

    //
    // Copy Atom A retiling
    //

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);                                  // (CPY,CPY_M,CPY_K)
    Tensor tCsA_copy_view  = smem_thr_copy_A.partition_S(sA);                                 // (CPY,CPY_M,CPY_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCsA_copy_view) == size<1>(tCrA_copy_view));                                  // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA_copy_view) == size<2>(tCrA_copy_view));                                  // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                                                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                      // MMA_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    static_assert(size<3>(tCsA) == size<5>(tCsB));                                                              // PIPE
    static_assert(Int<DispatchPolicy::Stages>{} == size<4>(sB));                                                // PIPE
    static_assert(NumComputeMtxs == size<3>(tCsB));                                                   // NumComputeMtxs
    static_assert(NumComputeMtxs == size<2>(sB));                                                     // NumComputeMtxs
    static_assert(ImMode == size<4>(tCsB));                                                                // REAL/IMAG
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) >= _2{}, "RS loops require 2 or more MMA k-iterations for correctness");
    static constexpr int NumWGWait = const_min(7, const_max(0, size<2>(tCrA) - 2));

    Tensor sScale = make_tensor(make_smem_ptr(shared_tensors.smem_A_scale.data()), SmemLayoutAScale{});
                                                                                                     // (BLK_M,BLK_K,1)
    Tensor tsScale = thread_mma.partition_A(sScale);                                             // (MMA,MMA_M,MMA_K,1)
    Tensor trScale = make_tensor<ElementScalar>(make_tuple(size<0>(tCrA), size<1>(tCrA), _1{}));       // (MMA,MMA_M,1)
    auto smem_tiled_copy_scale = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopy, ElementScalar>{}, tiled_mma);
    auto smem_thr_copy_scale = smem_tiled_copy_scale.get_thread_slice(thread_idx);
    Tensor trScale_copy_view = smem_thr_copy_scale.retile_D(trScale);                                  // (CPY,CPY_M,1)

    //
    // PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Initialize ScaleOut
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    // first k tile
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      copy(smem_tiled_copy_scale, tsScale(_,_,0,0), trScale_copy_view(_,_,0));
      // Convert to reciprocal. No loss occurs here as scale factor should always be a power of 2
      cute::transform(trScale, trScale, cutlass::reciprocal_approximate<ElementScalar>{});

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      // copy smem->rmem for A operand
      copy(smem_tiled_copy_A, tCsA_copy_view(_,_,0,read_stage), tCrA_copy_view(_,_,0));
      transform_A_kblock(tCrA_load(_,_,0), trScale(_,_,0), tCrA(_,_,0,_));

      JETFIRE_MAC_LOOP_HEADER // {$nv-internal-release}
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA) - 1; ++k_block) {
        copy(smem_tiled_copy_A, tCsA_copy_view(_,_,k_block+1,read_stage), tCrA_copy_view(_,_,k_block+1));
        transform_A_kblock(tCrA_load(_,_,k_block+1), trScale(_,_,0), tCrA(_,_,k_block+1,_));
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        dispatch_mma(tiled_mma, tCrA(_,_,k_block,_), tCrB(_,_,k_block,_,read_stage), accum);
        warpgroup_commit_batch();
      }

      warpgroup_wait<NumWGWait>();

      warpgroup_arrive();
      const int final_k = size<2>(tCrA) - 1;
      // (V,M) x (V,N) => (V,M,N)
      dispatch_mma(tiled_mma, tCrA(_,_,final_k,_), tCrB(_,_,final_k,_,read_stage), accum);
      warpgroup_commit_batch();
      --k_tile_count;
      if (k_tile_count == 0) {
        return;
      }
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      copy(smem_tiled_copy_A, tCsA_copy_view(_,_,0,smem_pipe_read.index()), tCrA_copy_view(_,_,0));
      transform_A_kblock(tCrA_load(_,_,0), trScale(_,_,0), tCrA(_,_,0,_));
      warpgroup_wait<NumWGWait>();
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      warpgroup_fence_operand(accum);
      // Unroll the K mode manually to set scale D to 1
      JETFIRE_MAC_LOOP_HEADER // {$nv-internal-release}
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        if (k_block == 0) {
          barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        }
        if (k_block == size<2>(tCrA) - 1) {
          pipeline.consumer_wait(smem_pipe_read, barrier_token);
          copy(smem_tiled_copy_A, tCsA_copy_view(_,_,0,smem_pipe_read.index()), tCrA_copy_view(_,_,0));
          transform_A_kblock(tCrA_load(_,_,0), trScale(_,_,0), tCrA(_,_,0,_));
        }
        else {
          copy(smem_tiled_copy_A, tCsA_copy_view(_,_,k_block+1,read_stage), tCrA_copy_view(_,_,k_block+1));
          transform_A_kblock(tCrA_load(_,_,k_block+1), trScale(_,_,0), tCrA(_,_,k_block+1,_));
        }

        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        dispatch_mma(tiled_mma, tCrA(_,_,k_block,_), tCrB(_,_,k_block,_,read_stage), accum);
        warpgroup_commit_batch();
        warpgroup_wait<NumWGWait>();
        if (k_block == 0) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }
      }
      warpgroup_fence_operand(accum);

    }

    warpgroup_fence_operand(accum);

    // last k_tile (if there are 2 or more k_tiles in total)
    {
      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      warpgroup_fence_operand(accum);

      // Unroll the K mode manually to set scale D to 1
      JETFIRE_MAC_LOOP_HEADER // {$nv-internal-release}
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA) - 1; ++k_block) {
        copy(smem_tiled_copy_A, tCsA_copy_view(_,_,k_block+1,read_stage), tCrA_copy_view(_,_,k_block+1));
        transform_A_kblock(tCrA_load(_,_,k_block+1), trScale(_,_,0), tCrA(_,_,k_block+1,_));
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        dispatch_mma(tiled_mma, tCrA(_,_,k_block,_), tCrB(_,_,k_block,_,read_stage), accum);
        warpgroup_commit_batch();
        warpgroup_wait<NumWGWait>();
        if (k_block == 1) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }
      }

      warpgroup_arrive();
      const int final_k = size<2>(tCrA) - 1;
      // (V,M) x (V,N) => (V,M,N)
      dispatch_mma(tiled_mma, tCrA(_,_,final_k,_), tCrB(_,_,final_k,_,read_stage), accum);
      warpgroup_commit_batch();
      if constexpr (size<2>(tCrA) == 2) {
        warpgroup_wait<NumWGWait>();
        // release prior barrier: (num. k_block) == 2 has its k_block == 1 iteration dispatched here
        pipeline.consumer_release(smem_pipe_release);               // UNLOCK smem_pipe_release, done _computing_ on it
        ++smem_pipe_release;
      }
    }

    warpgroup_fence_operand(accum);

  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  // Transforms complex A into complex half_t mtxs.
  // Since we would like to feed A operands in a packed way, it is necessary to transform cuComplexFloat-equivalent inputs in the following way:
  // |f0_re|f0_im|f1_re|f1_im| -> |h0_re|h1_re|h0_im|h1_im|
  //                              \__________/ \__________/
  //                                regA_0_re    regA_0_im
  template <
    class EngineIn,
    class EngineScale,
    class EngineOut,
    class InLayout,
    class ScalarLayout,
    class OutputLayout,
    int N = cosize_v<InLayout>
  >
  CUTLASS_DEVICE void
  transform_A_kblock(
      Tensor<EngineIn,InLayout>&& in,
      Tensor<EngineScale,ScalarLayout>&& scale,
      Tensor<EngineOut,OutputLayout>&& out) {
    /// The inputs must be backed by registers & be statically sized so we can unroll the conversion loops.
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(cute::is_same_v<typename EngineIn::value_type, ElementA>, "Input engine must be same type as the A operand");
    static_assert(cute::is_same_v<typename EngineScale::value_type, ElementScalar>, "Scale engine must be same type as the Scale operand");
    static_assert(cute::is_same_v<typename EngineOut::value_type, typename TiledMma::ValTypeA>, "Output engine must be same type as the Mma input");
    static_assert(is_static_v<InLayout>, "Tensor layout for the conversion must be static");
    static_assert(is_static_v<OutputLayout>, "Tensor layout for the conversion must be static");
    static_assert(not cutlass::is_complex_v<ElementScalar>, "Scale is expected to be real.");
    static_assert(cutlass::is_complex_v<ElementA>, "A is expected to be complex.");
    static_assert(N % 2 == 0, "Could not vectorize FP16 operands");

    using In_Base = typename ElementA::value_type;
    using Out_x2 = typename TiledMma::ValTypeA;

    Tensor in_x4  = recast<Array<In_Base, 4>>(in); // Array<complex<float>,2>
    Tensor out_x4 = recast<Array<Out_x2, 2>>(out); // complex<Array<half_t,2>>
    Tensor tmp_x4 = make_tensor_like(in_x4);
    auto convert_in2out = [&] (Array<In_Base, 4> &x) {
      cutlass::NumericArrayConverter<typename Out_x2::Element, In_Base, 2, cutlass::FloatRoundStyle::round_to_nearest> convert;
      Array<Out_x2, 2> result;
      Array<In_Base, 2> x_re;
      Array<In_Base, 2> x_im;
      x_re[0] = std::move(x[0]);
      x_re[1] = std::move(x[2]);
      result[0] = std::move(convert(x_re));
      x_im[0] = std::move(x[1]);
      x_im[1] = std::move(x[3]);
      result[1] = std::move(convert(x_im));
      return result;
    };
    auto convert_out2in = [&] (Array<Out_x2, 2> &x) {
      cutlass::NumericArrayConverter<In_Base, typename Out_x2::Element, 2, cutlass::FloatRoundStyle::round_to_nearest> convert;
      Array<In_Base, 4> result;
      auto x_re = std::move(convert(x[0]));
      result[0] = std::move(x_re[0]);
      result[2] = std::move(x_re[1]);
      auto x_im = std::move(convert(x[1]));
      result[1] = std::move(x_im[0]);
      result[3] = std::move(x_im[1]);
      return result;
    };

    cute::transform(in, scale, in, [&] (ElementA &a, ElementScalar &b) { return a * b; });

    CUTE_UNROLL
    for (int mtx = 0; mtx < NumComputeMtxs; ++mtx) {
      cute::transform(in_x4, out_x4(_,_,mtx), convert_in2out);

      if (mtx < NumComputeMtxs - 1) {
        cute::transform(out_x4(_,_,mtx), tmp_x4, convert_out2in);
        cute::transform(in_x4, tmp_x4, in_x4, cutlass::minus<Array<In_Base, 4>>{});
      }
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
