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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
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

// WarpSpecialized Mainloop
template <
  int Stages,
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
  int NumComputeMtxs,
  bool AllBands
>
struct CollectiveMma<
    MainloopSm90TmaGmmaWarpSpecializedFastF32PreTransformed<Stages, NumComputeMtxs, false, AllBands, true, ClusterShape, KernelSchedule>,
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
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedFastF32PreTransformed<Stages, NumComputeMtxs, false, AllBands, true, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  // Tiled Gmem access shape
  using TileShapeGmem = decltype(append(take<0,2>(TileShape{}), make_tuple(_1{},_1{},shape<2>(TileShape{}))));

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(is_complex_v<typename TiledMma::ValTypeC>, "Complex TiledMma expected. Construct from MMA_Atom_PlanarComplex2Complex.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  // static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
  //               cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
  //               "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // Cast to size equivalent uint type to avoid any rounding by TMA.
  using InternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;
  using InternalElementB = uint_bit_t<sizeof_bits_v<ElementB>>;

  struct SharedStorage {
    struct TensorStorage {
      alignas(128) cute::array<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(128) cute::array<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShapeGmem{},
        ClusterShape{})); // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShapeGmem{},
        ClusterShape{})); // mcast along M mode for this N load, if any
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

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    /// NOTE: We classify real/imag as a K-mode since epilogue produces std::complex as ``elements''.
    /// The real/imag ``mode'' in pretransformed inputs are ``contracted'' is such sense.
    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShapeGmem{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShapeGmem{},
        ClusterShape{});
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = TmaTransactionBytes;

    return {
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
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static_assert(DispatchPolicy::Stages >= ImMode * NumComputeMtxs,
                "GemmEmulatedPreTransformed has Smem access backflows. Stages have to hold at least 2*NumComputeMtxs");
  static_assert(DispatchPolicy::Stages % ImMode == 0, "Stage number must be even to ensure meaningful real/imag-stride.");
  static constexpr int CStages = DispatchPolicy::Stages / ImMode; // Group real&imag parts into the a multi-stage
  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS_DEFAULT = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
    cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK =
    cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  // Reinterpret K-mode tile count in cooperative kernel context
  template <class K_Count, class K_Shape>
  CUTLASS_DEVICE
  static auto reinterpret_cooperative_k_count(K_Count k_count, K_Shape const& k_shape) {
    return crd2idx(make_coord(0, 0, k_count), k_shape);
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
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L));
                                                                                                      // (m,(im,s,k),l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,make_tuple(Int<ImMode>{},Int<NumComputeMtxs>{},K),L));
                                                                                                      // (n,(im,s,k),l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShapeGmem{}, make_coord(_,_,_), Step<_1, X,_1>{});
                                                                                          // (BLK_M,BLK_K,m,(im,s,k),l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShapeGmem{}, make_coord(_,_,_), Step< X,_1,_1>{});
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
      PipelineState &smem_pipe_write_,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    int lane_predicate = cute::elect_one_sync();
    PipelineState smem_pipe_write = smem_pipe_write_;

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                              // (BLK_M,BLK_K,(im,s,k))
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                              // (BLK_M,BLK_K,(im,s,k))

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                          // (TMA,TMA_M,TMA_K,(im,s,k))
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                          // (TMA,TMA_N,TMA_K,(im,s,k))
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

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

      // Mainloop
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        // Advance smem_pipe_write
        ++smem_pipe_write;
        ++k_tile_iter;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
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

  // Warpper over cute::gemm to compute all k_blocks, real and imag
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
      TensorC &accum,
      int stage_A,
      int stage_B) {

    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(rA); ++k_block) {
      cute::gemm(tiled_mma, rA(_,_,k_block,stage_A), rB(_,_,k_block,stage_B), accum);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  }

  template <class TiledMma, class TensorA, class TensorB, class TensorC>
  CUTLASS_DEVICE void
  dispatch_mma(TiledMma &tiled_mma, TensorA &&rA, TensorB &&rB, TensorC &&accum, int stage_A, int stage_B) {
    dispatch_mma(tiled_mma, rA, rB, accum, stage_A, stage_B);
  }

  // Warpper over cute::gemm to compute on correct mtxs according to the wavefront
  template <
    class TiledMma,
    class TensorA,
    class TensorB,
    class TensorC
  >
  CUTLASS_DEVICE void
  dispatch_mma_cases(
      TiledMma &tiled_mma,
      TensorA &&rA,
      TensorB &&rB,
      TensorC &accum,
      int stage,
      int mtx_wavefront) {

    switch (mtx_wavefront) {
      case 0: {
        dispatch_mma(tiled_mma, rA, rB, accum, stage, stage);
      }
      break;
      case 1: {
        int last1_stage = (stage + CStages - 1) % CStages;
        dispatch_mma(tiled_mma, rA, rB, accum, last1_stage, stage);
        dispatch_mma(tiled_mma, rA, rB, accum, stage, last1_stage);
        if constexpr (AllBands || NumComputeMtxs > 2) {
          dispatch_mma(tiled_mma, rA, rB, accum, stage, stage);
        }
      }
      break;
      case 2: {
        int last1_stage = (stage + CStages - 1) % CStages;
        int last2_stage = (stage + CStages - 2) % CStages;
        dispatch_mma(tiled_mma, rA, rB, accum, stage, last2_stage);
        dispatch_mma(tiled_mma, rA, rB, accum, last2_stage, stage);
        if constexpr (AllBands) {
          dispatch_mma(tiled_mma, rA, rB, accum, stage, last1_stage);
          dispatch_mma(tiled_mma, rA, rB, accum, last1_stage, stage);
          dispatch_mma(tiled_mma, rA, rB, accum, stage, stage);
        }
      }
      break;
      default:
      break;
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

    static_assert(is_rmem<FrgCEngine>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(is_complex_v<typename FrgCEngine::value_type>, "Complex accumulator expected.");

    // {$nv-internal-release begin}
    /// Ideally, exposure of downcasting complex<float> to float should happen within MMA_Op instead of in collectives,
    /// but the compiler would emit extra mov.b32 instructions that case and OCG will spill them to new registers.
    // {$nv-internal-release end}
    Tensor accum = recast<typename FrgCEngine::value_type::value_type>(accum_c32);                          // (2V,M,N)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA_ = thread_mma.make_fragment_A(tCsA);                                          // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB_ = thread_mma.make_fragment_B(tCsB);                                          // (MMA,MMA_N,MMA_K,PIPE)

    // MMA_Op has to be fed with real & imaginary parts, which are planars and come from PIPE mode:
    Tensor tCrA = composition(tCrA_, make_ordered_layout(make_shape(Int<ImMode>{}, shape<1>(tCrA_), shape<2>(tCrA_), Int<CStages>{}), Step<_3,_1,_2,_4>{}));
    Tensor tCrB = composition(tCrB_, make_ordered_layout(make_shape(Int<ImMode>{}, shape<1>(tCrB_), shape<2>(tCrB_), Int<CStages>{}), Step<_3,_1,_2,_4>{}));

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE
    static_assert(size<0>(tCrA_) == 1, "A fragment by ThreadMma must be a descriptor and have unit first dimension.");
    static_assert(size<0>(tCrB_) == 1, "B fragment by ThreadMma must be a descriptor and have unit first dimension.");

    //
    // PIPELINED MAIN LOOP
    //
    static constexpr int K_PIPE_MMAS = ImMode * NumComputeMtxs;
    static constexpr int MMA_WAIT_CNT = ImMode; // Each pipe is released per-mtx.
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Define wait & unlock functions
    auto wait_consumer = [&] (auto &pipeline, auto &smem_pipe_read) {
      int read_stage = smem_pipe_read.index();
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      ++smem_pipe_read;
      // Return in multi-stage representation
      return idx2crd(read_stage, Shape<Int<ImMode>, Int<CStages>>{});
    };

    auto unlock_consumer = [] (auto &accum, auto &pipeline, auto &smem_pipe_release) {
      warpgroup_wait<MMA_WAIT_CNT>();
      warpgroup_fence_operand(accum);
      // UNLOCK real&imag parts from one previous mtx
      CUTLASS_PRAGMA_UNROLL
      for (int reim = 0; reim < ImMode; ++reim) {
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;
      }
    };

    // Initialize ScaleOut
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; k_tile_prologue -= ImMode * NumComputeMtxs) {
      CUTLASS_PRAGMA_UNROLL
      for (int mtx_wavefront = 0; mtx_wavefront < NumComputeMtxs; ++mtx_wavefront) {
        int stage = get<1>(wait_consumer(pipeline, smem_pipe_read));
        wait_consumer(pipeline, smem_pipe_read);
        // Only start processing when both re & im arrives.
        warpgroup_arrive();

        // (V,M,K) x (V,N,K) => (V,M,N)
        dispatch_mma_cases(tiled_mma, tCrA, tCrB, accum, stage, mtx_wavefront);
        warpgroup_commit_batch();
      }
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; k_tile_count -= ImMode * NumComputeMtxs) {
      CUTLASS_PRAGMA_UNROLL
      for (int mtx_wavefront = 0; mtx_wavefront < NumComputeMtxs; ++mtx_wavefront) {
        warpgroup_fence_operand(accum);

        int stage = get<1>(wait_consumer(pipeline, smem_pipe_read));
        wait_consumer(pipeline, smem_pipe_read);
        // Only start processing when both re & im arrives.
        warpgroup_arrive();

        //
        // Compute on k_tile
        //
        dispatch_mma_cases(tiled_mma, tCrA, tCrB, accum, stage, mtx_wavefront);
        warpgroup_commit_batch();

        unlock_consumer(accum, pipeline, smem_pipe_release);
      }
    }

    warpgroup_fence_operand(accum);

  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    static constexpr int K_PIPE_MMAS = ImMode * NumComputeMtxs;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
