/******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 ******************************************************************************/
// {$nv-release-never file}

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/numeric_size.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

// K tile iterator that either increments (if `reverse` is false) or decrements (if `reverse` is true)
// The bidirectional nature of this iterator is release guarded because it is currently only used internally
template <class Shape>
struct KTileIterator {
  CUTLASS_HOST_DEVICE
  auto operator*() const {
    return cute::idx2crd(idx_, shape_);
  }

  CUTLASS_HOST_DEVICE
  KTileIterator& operator++() {
    if (reverse_) {
      idx_--;
    }
    else {
      idx_++;
    }
    return *this;
  }

  CUTLASS_HOST_DEVICE
  KTileIterator& operator--() {
    if (reverse_) {
      idx_++;
    }
    else {
      idx_--;
    }
    return *this;
  }

  Shape shape_{};
  int idx_ = 0;
  bool reverse_ = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helpers for representing sizs of operands for cache-aware heuristics
struct OperandSizeInfo {
  // Sizes of A and B operands and D output in bytes
  size_t A_ = 0;
  size_t B_ = 0;
  size_t D_ = 0;

  // Number of k tiles cmputed in the kernel
  size_t k_tiles_ = 0;

  // Whether the kernel uses sparsity for operand A
  bool is_sparse_ = false;
};

template <
  class ElementA,
  class ElementB,
  class ElementD,
  int SparsityA = 1,
  class ElementSFA = void,
  class LayoutSFA = void,
  class ElementSFB = void,
  class LayoutSFB = void,
  class ProblemShape,
  class TileShape
>
CUTLASS_HOST_DEVICE
static OperandSizeInfo
operand_sizes(ProblemShape problem_shape, TileShape tile_shape) {
  auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
  size_t m = cute::size<0>(problem_shape_mnkl);
  size_t n = cute::size<1>(problem_shape_mnkl);
  size_t k = cute::size<2>(problem_shape_mnkl);

  size_t total_bits_A = m * k * cute::sizeof_bits_v<ElementA>;
  total_bits_A /= SparsityA;

  size_t total_bits_B = k * n * cute::sizeof_bits_v<ElementB>;

  if constexpr (!cute::is_same_v<ElementSFA, void>) {
    total_bits_A += cute::size(LayoutSFA{}) * cute::sizeof_bits_v<ElementSFA>;
  }
  if constexpr (!cute::is_same_v<ElementSFB, void>) {
    total_bits_B += cute::size(LayoutSFB{}) * cute::sizeof_bits_v<ElementSFB>;
  }

  size_t total_bits_D = m * n * cute::sizeof_bits_v<ElementD>;
  size_t k_tiles = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));
  bool is_sparse = SparsityA > 1;

  return OperandSizeInfo{
    cutlass::bits_to_bytes<size_t>(total_bits_A),
    cutlass::bits_to_bytes<size_t>(total_bits_B),
    cutlass::bits_to_bytes<size_t>(total_bits_D),
    k_tiles,
    is_sparse
  };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
uint32_t
cage_swizzle_size_heuristic(
  int problem_ctas_m,
  int problem_ctas_n,
  int cluster_shape_m,
  int cluster_shape_n,
  KernelHardwareInfo hw_info,
  OperandSizeInfo operand_sizes) {

  // This method attempts to comute the optimal swizzle size for a problem based on:
  //   1) The number of available SMs
  //   2) The tile and cluster sizes used
  //   3) The amount of data loaded in each dimension of a cluster
  //
  // Without swizzling, tiles are assigned to CTAs solely based on rasterization order (i.e.,
  // whether to assign tiles along the M mode first, or along the N mode first). If assigning
  // along the M mode first, tiles from a single column are assigned to CTAs before moving to
  // the next column.
  //
  // Swizzling attempts to improve L2 cache use by assigning a tiles to CTAs in more of a
  // square shape so that reuse can be observed for both A and B matrices.
  //
  // We can define a swizzling configuration using a parameter known as the "swizzle_size,"
  // which determines the number of tiles in a column to assign during an along-N rasterization
  // before jumping to the next row (similarly, the number of tiles in a row to assign during
  // an along-M rasterization before jumping to the next column). That is, the swizzle size
  // determines the width or height of the box of tiles we attempt to assign.
  //
  // Our goal is to attempt to find the optimal swizzle size. Without loss of generality, we'll
  // consider an along-N rasterization, with the goal of finding the optimal number of tiles
  // in a row to assign before beginning to assign tiles of the next row.
  //
  // We can frame this as an optimization problem:
  //   - Given:
  //       - Cluster-tiled grid of size `clusters_M x clusters_N`
  //       - Chip with `total_clusters` available clusters based on SM count
  //   - Compute the column swizzle size col_swizzle
  //   - That minimizes:
  //       total_bytes = (bytes_per_cluster_A * clusters_M * clusters_N / col_swizzle) +
  //                     (bytes_per_cluster_B * clusters_M * clusters_N * col_swizzle / total_clusters)
  //
  // The first term in the equation to minimize accounts for the number of bytes of A that need
  // to be read assuming a swizzle size of `col_swizzle`: All tiles of A must be read for each
  // group of `col_swizzle` column clusters (there are `clusters_N / col_swizzle` of these groups).
  //
  // The second term accounts for the number of bytes of B that need to be read assuming a swizzle size
  // of `col_swizzle`. If the width of the swizzle box is of size `col_swizzle`, and the total area of
  // the swizzle box is `total_clusters`, then the height of the swizzle box is of size `total_clusters / col_swizzle`.
  // All tiles of B must be read for each group of `(total_clusters / col_swizzle)` row clusters (there are
  // `clusters_M / (total_clusters / col_swizzle) == clusters_M * col_swizzle / total_clusters` of these groups).
  //
  // To minimize, we take the derivative of this with respect to `col_swizzle`:
  //   d(total_bytes)/d(col_swizzle) =
  //       clusters_M * clusters_N * ((bytes_per_cluster_B / total_clusters) - (bytes_per_cluster_A / (col_swizzle^2)))
  // This is minimized when:
  //   col_swizzle = sqrt(total_clusters * (bytes_per_cluster_A / bytes_per_cluster_B))

  int available_clusters = hw_info.sm_count / (cluster_shape_m * cluster_shape_n);

  size_t total_bytes_min, total_bytes_max;
  int problem_clusters_min, problem_clusters_max;

  auto size_A = operand_sizes.A_;
  auto size_B = operand_sizes.B_;

  // If operand_sizes has been not been initialized specifically for this problem,
  // default sizes for A and B operands to be equal to one another.
  if (size_A == 0 || size_B == 0) {
    size_A = 1;
    size_B = 1;
  }

  if (size_A < size_B) {
    total_bytes_min = size_A;
    total_bytes_max = size_B;
    problem_clusters_min = ceil_div(problem_ctas_m, cluster_shape_m);
    problem_clusters_max = ceil_div(problem_ctas_n, cluster_shape_n);
  }
  else {
    total_bytes_min = size_B;
    total_bytes_max = size_A;
    problem_clusters_min = ceil_div(problem_ctas_n, cluster_shape_n);
    problem_clusters_max = ceil_div(problem_ctas_m, cluster_shape_m);
  }

  // Compute factor to account for skew in the number of bytes loaded in each cluster dimension.
  // We are computing:
  //   bytes_per_max_cluster_dim / bytes_per_min_cluster_dim
  //     = (total_bytes_max / problem_clusters_max) / (total_bytes_min / problem_clusters_min)
  //     = (total_bytes_max * problem_clusters_min) / (total_bytes_min * problem_clusters_max)
  float cluster_bytes_skew = static_cast<float>(total_bytes_max * problem_clusters_min) /
                              static_cast<float>(total_bytes_min * problem_clusters_max);

  // Temporarily decrease L2 capacity to account for skew in CTAs that limits inter-wave reuse.
  // This should be removed if a better solution to keeping CTAs in sync is implemented: https://jirasw.nvidia.com/browse/CFK-15576
  size_t l2_bytes = static_cast<size_t>(0.6f * static_cast<float>(hw_info.l2_capacity));

  // Determine the duplication factor of L2 due to duplication across uGPUs.
  // The code below currently assumes that a device with > 66 MB of L2 has two uGPUs,
  // since this is half of the 132 MB of a 2 uGPU B100.
  size_t ugpu_div_factor = l2_bytes > (66 << 20) ? 2 : 1;

  // Determine the amount of data that will be written to L2 per wave so that we can
  // determine how much L2 capacity can potentially be reused across waves.
  size_t wave_bytes_D = (operand_sizes.D_ * hw_info.sm_count) / (problem_ctas_m * problem_ctas_n);

  // Disable inter-wave reuse heuristic if size of D is not specified (operand_sizes.D_ == 0)
  size_t l2_bytes_reuse = operand_sizes.D_ > 0 && (l2_bytes > wave_bytes_D) ? (l2_bytes - wave_bytes_D) / ugpu_div_factor : size_t(0);

  // Goal is to reuse one of the inputs. Reuse the min dimension
  // The amount we can reuse for min depends on ratio of min:max since each are read equally. If min==max, then can get 50% of l2_bytes_reuse

  // We aim to reuse one of the inputs across waves, if possible. We will attempt to reuse the input
  // with the smaller footprint per wave. To understand how much of the remaining L2 capacity can be
  // reused for the input to be reused, we need to understand the skew in footprint between inputs.
  //
  // If we have fewer SMs along the minimum dimension than there are clusters along the min dimension,
  // we will rasterize linearly. We'll read `available_clusters` tiles of the min input per tile of the max.
  //
  // Otherwise, the maximum skew is if we read `problem_clusters_min` tiles of the min input and `available_clusters / problem_clusters_min`
  // of the max input (assuming a swizzle size of `problem_clusters_min`).
  float max_wave_shape_skew =
    available_clusters <= problem_clusters_min ?
    static_cast<float>(available_clusters) :
    static_cast<float>(problem_clusters_min) / static_cast<float>(ceil_div(available_clusters, problem_clusters_min));

  // Calculate the bytes of L2 that can be reused across the minimum dimension.
  // Assuming that tiles of the min and max dimension are the same size, then the fraction of L2
  // that can be reused by the minimum dimension is determined by wave_shape_skew:
  //   - If wave_shape_skew = 2, two tiles of the minimum dimension are loaded for every one of the max
  //   - Thus, the minimum dimension gets 2/3 of the L2 space.
  //   - This is generalized to `wave_shape_skew / (wave_shape_skew + 1)`
  //   - The total amount of L2 that can be reused is then `l2_reuse * wave_shape_skew / (wave_shape_skew + 1)`
  //     which is equivalent to `l2_reuse / (1 + 1/wave_shape_skew)`
  // We now must factor in that tiles of the min and max dimension may not be equal in size (i.e., `cluster_byte_skew`
  // may not equal one). Since `cluster_bytes_skew` is the ratio of the bytes of a tile in the max
  // to the bytes of a tile in the min dimension, the total skew ratio is given by `wave_shape_skew / cluster_bytes_skew`.
  // This new ratio can be used in place of `wave_shape_skew` in the calculation above to arrive at the final
  // amount of L2 that can be reused for the minimum dimension:
  //   l2_reuse / (1 + (1 / (wave_shape_skew / cluster_bytes_skew)))`
  // = l2_reuse / (1 + cluster_bytes_skew / wave_shape_skew)
  size_t l2_bytes_reuse_min = static_cast<size_t>(static_cast<float>(l2_bytes_reuse) / (1 + (cluster_bytes_skew / max_wave_shape_skew)));

  // Convert from the number of bytes to the number of tiles that can be reused
  int l2_tiles_reuse_min = static_cast<int>(l2_bytes_reuse_min * problem_clusters_min / total_bytes_min);

#if 0
  // If alternating K is not enabled, no partial reuse is assumed
  float l2_size_reuse = alternate_k_ ? static_cast<float>(l2_tiles_reuse_min) / static_cast<float>(problem_clusters_max) : 0;
#else
  float l2_size_reuse = 0;
#endif
  int swizzle_size_partial_reuse = static_cast<int>(
    ceil(fast_sqrt(static_cast<float>(available_clusters) * (cluster_bytes_skew + l2_size_reuse)))
  );

  // Compare partial reuse case (swizzle_size_partial_reuse) with full reuse case (l2_tiles_reuse_min)
  int swizzle_size_without_limits = platform::max(l2_tiles_reuse_min, swizzle_size_partial_reuse);

  // We have now found a swizzle size that we would select if there were not limits on the number
  // of available clusters or problem size. We now find bounds for swizzle sizes in which it is
  // beneficial to use a swizzle size > 1. The discussion below assumes that we are trying to determine the number of column
  // cluster-tiles to swizzle over.

  // If we have fewer cluster tiles along the M mode than there are available clusters, then
  // it does not make sense to use a swizzle size that is below `available_clusters / clusters_tiles_m`.
  // Suppose we have 32 available clusters and 4 cluster tiles along the M mode. If we use a swizzle
  // size of 2 columns, a wave will end up using four 4x2 swizzle boxes, which is equivalent to using
  // a swizzle size of 8 (which is 32 / 4). Any swizzle sizes <= 8 in this case will be equivalent to
  // not swizzling at all because we'll cover all of the rows in a column first.
  int swizzle_size_min = ceil_div(available_clusters, problem_clusters_max);

  // The maximum swizzle size that can be used is either the number of clusters tiles along the N mode
  // or the number of available clusters. Each of these require separate cases depending on which is larger.
  //
  // Suppose available_clusters > problem_clusters_min:
  //   Then, the maximum swizzle size that can be used is problem_clusters_min. However, doing so
  //   is equivalent to using a swizzle size of 1 and rasterizing in a row-major fashion.
  // Suppose available_clusters < problem_clusters_min:
  //   Then, the maximum swizzle size that can be used is available_clusters. Unlike the preceding
  //   case, this is not equivalent to using a swizzle size of 1 and a row-major rasterization.
  //   If we used a swizzle size of 1 and row-major rasterization, the first wave would cover
  //   only a fraction of the first row of tiles, and the second wave would continue in that
  //   row where the first wave left off. This leads to poor inter-wave reuse.
  //   If we instead use a swizzle size of available_clusters, we will get improved inter-wave reuse.
  //
  // Thus, we only set swizzle size to 1 if the selected swizzle size is greater than or equal
  // to problem_clusters_min.
  int swizzle_size_max = problem_clusters_min;

  int selected_swizzle_size;
  if (swizzle_size_without_limits <= swizzle_size_min || swizzle_size_without_limits >= swizzle_size_max) {
    // If the optimal swizzle size is less than the minimum swizzle size or greater than
    // the maximum swizzle size, the swizzle would be equivalent to a non-swizzled rasterization
    // along either the M or N mode.
    selected_swizzle_size = 1;
  }
  else {
    // The calculation for swizzle size above does not consider that swizzle size is capped by
    // the number of available clusters.
    swizzle_size_without_limits = min(swizzle_size_without_limits, available_clusters);

    #if 0 // {$nv-release-never begin}
    // For perf benchmarking, we can use the CAGE_NEW_SWIZZLE environment variable to enable/disable the
    // old swizzle size calculation.
    const char* env_var = "CAGE_NEW_SWIZZLE";
    bool new_swizzle = getenv(env_var) != nullptr && strcmp(getenv(env_var), "1") == 0;
    if (!new_swizzle) {
      // Round down swizzle size to a version that will evenly divide `problem_clusters_min` so that
      // we reduce residual tiles along the min dimension.
      auto swizzle_groups_along_min_dimension = ceil_div(problem_clusters_min, swizzle_size_without_limits);
      selected_swizzle_size = ceil_div(problem_clusters_min, swizzle_groups_along_min_dimension);
      return selected_swizzle_size;
    }
    #endif // {$nv-release-never end}

    // Adjust swizzle size to account for imperfections of swizzling implementation.
    // The implementation of swizzling currently performed for SM100 kernels only swizzles
    // the portion of the grid that is divisible by the swizzle size. This leaves "residual"
    // tiles in the major and minor modes that are not swizzled. This imperfect form of swizzling
    // was necessitated by overheads observed under a "perfect" implementation of swizzling --
    // see https://gitlab-master.nvidia.com/dlarch-fastkernels/kernel_store/-/merge_requests/2907 for details.
    //
    // The example below illustrates the residuals (tiles marked with R) that are left over after
    // swizzling with a swizzle size of 2.
    //
    //  <- S ->
    //  +--+--+--+--+--+
    //  |  |  |  |  | R|
    //  +--+--+--+--+--+
    //  |  |  |  |  | R|
    //  +--+--+--+--+--+
    //  |  |  |  |  | R|
    //  +--+--+--+--+--+
    //  | R| R| R| R| R|
    //  +--+--+--+--+--+
    //
    // The residual tiles mentioned above reduce L2 reuse. Thus, the number of residual tiles should be
    // accounted for in determining the swizzle size.
    //
    // The adjustments below do this by iterating through possible swizzle sizes less than the
    // selected swizzle size, and estimating the the total number of tiles loaded across the GEMM. 

    int min_swizz = 0;
    float min_score;

    auto cluster_tiles_major = problem_clusters_min;
    auto cluster_tiles_minor = problem_clusters_max;
    float f_cluster_tiles_major = static_cast<float>(cluster_tiles_major);
    float f_cluster_tiles_minor = static_cast<float>(cluster_tiles_minor);
    float f_swizz_block_size = static_cast<float>(available_clusters);

    for (int swizz = 1; swizz <= swizzle_size_without_limits; swizz++) {
      float swizz_block_dim_minor = static_cast<float>(swizz);
      float swizz_block_dim_major = f_swizz_block_size / swizz_block_dim_minor;

      auto resid_minor = cluster_tiles_minor % swizz;
      auto resid_major = cluster_tiles_major % swizz;

      // Compute the average number of residual tiles along the major mode per swizzle block.
      // With a swizzle size of S, the swizzling implementation works by making every set of S tiles
      // along the major mode into a set of S tiles along the minor mode (other than a residual).
      // Thus, when traversing the major mode, we encounter the residual space in the major mode roughly
      // every time we fully traverse the major mode.
      // Compute the average number of residual tiles along the major mode per swizzle block.
      float swizz_blocks_per_major_dim = f_swizz_block_size / f_cluster_tiles_major;
      float resid_major_per_swizz_block = swizz_blocks_per_major_dim * static_cast<float>(resid_major);

      // Because the pre-swizzling rasterization occurs in the major order first, once we've reached
      // the first residual tile in the minor mode, we will only be working with residual tiles. We
      // will account for the expected number of tiles read in that case separately.
      float minor_clusters_before_resid = f_cluster_tiles_minor - static_cast<float>(resid_minor);
      float swizz_block_before_minor_resid = f_cluster_tiles_major * minor_clusters_before_resid / f_swizz_block_size;
      float swizz_block_within_minor_resid = static_cast<float>(resid_minor) * f_cluster_tiles_major / f_swizz_block_size;

      // Determine how many tiles we expect to read from DRAM before reaching the minor residual.
      // CAGE assumes makes swizzle size such that S tiles remain in L2 between each wave. Thus,
      // the only tiles to be read from DRAM are the remainder of the swizzle box size and any residuals
      // in the major mode
      float tiles_read_per_block_before_minor_resid = swizz_block_dim_major + resid_major_per_swizz_block;
      float tiles_read_before_minor_resid = swizz_block_before_minor_resid * tiles_read_per_block_before_minor_resid;

      // Account for the initial reads of the S blocks that are cached across waves. We have as many of these
      // as there are tiles along the minor mode before reaching the residual space.
      tiles_read_before_minor_resid += minor_clusters_before_resid;

      // Determine how many tiles we expect to read from DRAM per wave within the minor residual space.
      // This assumes a basic along-major rasterization with no swizzling. Assume no inter-wave reuse.
      float wave_size = f_swizz_block_size;
      float tiles_read_per_block_within_minor_resid = wave_size < f_cluster_tiles_major
                                                    // Wave is smaller than major mode. We read only one tile along the minor mode
                                                    // and as many tiles along the mojor mode as our wave size.
                                                    ? wave_size + 1
                                                    // Wave exceeds major mode. We read as many tiles along the minor mode as
                                                    // are covered by a single wave and all tiles along the major mode.
                                                    : swizz_blocks_per_major_dim + f_cluster_tiles_major;

      float tiles_read_within_minor_resid = swizz_block_within_minor_resid * tiles_read_per_block_within_minor_resid;
      float score = tiles_read_before_minor_resid + tiles_read_within_minor_resid;
      if (min_swizz == 0 || score < min_score) {
        min_swizz = swizz;
        min_score = score;
      }
    }
    selected_swizzle_size = platform::max(1, min_swizz);
  }

  #if CUTLASS_CAGE_HOST_DEBUG_PRINT == 1
    printf("============================= GET_SWIZZLE =============================\n");
    printf("operand_sizes: A_=%lu B_=%lu D_=%lu\n", size_A, size_B, operand_sizes.D_);
    printf("swizzle_size_min=%d\n", (int)swizzle_size_min);
    printf("swizzle_size_max=%d\n", (int)swizzle_size_max);
    printf("wave_bytes_D=%d\n", (int)wave_bytes_D);
    printf("l2_bytes=%d\n", (int)l2_bytes);
    printf("l2_bytes_reuse=%d\n", (int)l2_bytes_reuse);
    printf("l2_bytes_reuse_min=%d\n", (int)l2_bytes_reuse_min);
    printf("l2_tiles_reuse_min=%d\n", (int)l2_tiles_reuse_min);
    printf("l2_size_reuse=%f\n", (float)l2_size_reuse);
    printf("swizzle_size_partial_reuse=%d\n", (int)swizzle_size_partial_reuse);
    printf("swizzle_size_without_limits=%d\n", (int)swizzle_size_without_limits);
    printf("selected_swizzle_size=%d\n", (int)selected_swizzle_size);
    printf("available_clusters=%d\n", (int)available_clusters);
    printf("cluster_bytes_skew=%f\n", cluster_bytes_skew);
    printf("max_wave_shape_skew=%f\n", max_wave_shape_skew);
    printf("cluster_shape_m=%d\n", (int)cluster_shape_m);
    printf("cluster_shape_n=%d\n", (int)cluster_shape_n);
    printf("problem_ctas_m=%d\n", (int)problem_ctas_m);
    printf("problem_ctas_n=%d\n", (int)problem_ctas_n);
    printf("problem_clusters_min=%d\n", (int)problem_clusters_min);
    printf("problem_clusters_max=%d\n", (int)problem_clusters_max);
  #endif
  return selected_swizzle_size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel::detail