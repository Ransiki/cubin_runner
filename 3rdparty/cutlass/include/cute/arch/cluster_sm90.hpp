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

#include <cute/config.hpp>
#include <cute/numeric/numeric_types.hpp>

// {$nv-internal-release begin}
#if !defined(__CUDACC_RTC__)
#define CU_INIT_UUID
#include <cute/arch/bringup/cluster_prototype.hpp>
#endif
// {$nv-internal-release end}

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
  ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#  define CUTE_ARCH_CLUSTER_SM90_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#  define CUTE_ARCH_ELECT_ONE_SM90_ENABLED
#endif

namespace cute {

CUTE_DEVICE void cluster_arrive_relaxed()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : );
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_arrive()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.aligned;\n" : : );
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_wait()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.wait.aligned;\n" : : );
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_sync()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  cluster_arrive();
  cluster_wait();
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

// Returns the dim3 grid size in terms of number of clusters.
CUTE_DEVICE dim3 cluster_grid_dims()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of gridDim with __CUDA_ARCH__.
  return gridDim;
#elif defined(_MSC_VER)
  CUTE_INVALID_CONTROL_PATH("cluster_grid_dims() can only be called on device");
  return {0, 0, 0};
#else
  return {0, 0, 0};
#endif
}

// Returns the dim3 cluster rank in the grid.
CUTE_DEVICE dim3 cluster_id_in_grid()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of blockIdx with __CUDA_ARCH__.
  return blockIdx;
#elif defined(_MSC_VER)
  CUTE_INVALID_CONTROL_PATH("cluster_id_in_grid() can only be called on device");
  return {0, 0, 0};
#else
  return {0, 0, 0};
#endif
}

// Returns the relative dim3 block rank local to the cluster.
CUTE_DEVICE dim3 block_id_in_cluster()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  return {0,0,0};
#endif
}

// Returns the dim3 cluster shape.
CUTE_DEVICE dim3 cluster_shape()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  return {1,1,1};
#endif
}

// Get 1D ctaid in a cluster.
CUTE_DEVICE uint32_t block_rank_in_cluster()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  return 0;
#endif
}

// Set the destination block-ID in cluster for a given SMEM Address
CUTE_DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank)
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
              : "=r"(result)
              : "r"(smemAddr), "r"(rank));
  return result;
#else
  return smemAddr;
#endif
}

// Elect one thread in the warp. The elected thread gets its predicate set to true, all others obtain false.
CUTE_HOST_DEVICE uint32_t elect_one_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
#elif defined(__CUDA_ARCH__)
  return (threadIdx.x % 32) == 0;
#else
  return true;
#endif
}

struct ElectOneLaneIdReturnType {
  uint32_t is_leader;
  uint32_t leader_lane_id;
};

CUTE_HOST_DEVICE
ElectOneLaneIdReturnType
elect_one_leader_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return {pred, laneid};
#elif defined(__CUDA_ARCH__)
  return {(threadIdx.x % 32) == 0, 0};
#else
  return {true, 0};
#endif
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void
store_shared_remote(uint32_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank)
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
#endif
}

// {$nv-internal-release begin}
#if !defined(__CUDACC_RTC__)
//
// Cluster launch utility
//
CUTE_HOST
bool
initialize_cluster_launch(void const* const kernel_function,
                          dim3 const& grid_dims,
                          dim3 const& cluster_dims,
                          cute::bringup::CUclusterPrototypeSchedulingPolicy cta_schedule_policy = cute::bringup::CU_CLUSTER_PROT_SCHEDULING_POLICY_DEFAULT)
{
  //
  // Validate cluster_dims
  //

  // Total number of cluster cannot be greater than 32 (hardware requirement)
  if (cluster_dims.x * cluster_dims.y * cluster_dims.z <= 0 ||
      cluster_dims.x * cluster_dims.y * cluster_dims.z > 32) {
    std::cout << "Invalid cluster dimensions: Attempting to init cluster (" << cluster_dims.x << "," << cluster_dims.y << "," << cluster_dims.z
              << ") [" << (cluster_dims.x * cluster_dims.y * cluster_dims.z) << "] which must be within (0,32]." << std::endl;
    return false;
  }

  // grid_dims should be divisible by cluster_dims
  if (grid_dims.x % cluster_dims.x != 0 ||
      grid_dims.y % cluster_dims.y != 0 ||
      grid_dims.z % cluster_dims.z != 0) {
    std::cout << "Invalid grid dimensions: Cluster (" << cluster_dims.x << "," << cluster_dims.y << "," << cluster_dims.z
              << ") does not divide Grid (" << grid_dims.x << "," << grid_dims.y << "," << grid_dims.z << ")." << std::endl;
    return false;
  }

  //
  // Driver internal functions to setup Cluster launch
  //

#ifndef __QNX__
  cudaFunction_t func;
  cute::bringup::CUetblClusterPrototype const* clusterProt = nullptr;

  cudaError_t result = cudaGetExportTable((const void**) &clusterProt,
                                          &cute::bringup::CU_ETID_ClusterPrototype);

  if (result != cudaSuccess) {
    std::cout << "cudaGetExportTable Failed!" << std::endl;
    return false;
  }

  result = cudaGetFuncBySymbol(&func, kernel_function);

  if (result != cudaSuccess) {
    std::cout << "cudaGetFuncBySymbol Failed!" << std::endl;
    return false;
  }

  clusterProt->SetFunctionClusterDim((CUfunction)func,
                                     cluster_dims.x, cluster_dims.y, cluster_dims.z);

  clusterProt->SetFunctionClusterNonPortableSizeSupport((CUfunction)func, 1);

  clusterProt->SetFunctionClusterSchedulingPolicy((CUfunction)func, cta_schedule_policy);
#endif

  return true;
}
#endif
// {$nv-internal-release end}
} // end namespace cute
