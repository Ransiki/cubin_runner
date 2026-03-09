/*
 * Copyright 2020 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

// {$nv-internal-release file}

#pragma once

#ifndef __CLUSTER_PROTOTYPE_H__
#define __CLUSTER_PROTOTYPE_H__

#include <cuda.h>

#include <cute/arch/bringup/cuda_uuid.h>


namespace cute {
namespace bringup {

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#if 0
} // To trick formatter
#endif

//------------------------------------------------------------------
// Prototype cluster APIs for functional testing
//------------------------------------------------------------------

//CU_DEFINE_UUID(CU_ETID_CLUSTER_PROTOTYPE, 
//    0x26dc3417, 0x0d80, 0x4547, 0x87, 0x26, 0xc0, 0xf1, 0xe7, 0xdd, 0x8b, 0xca);
static const CUuuid CU_ETID_CLUSTER_PROTOTYPE = {{((char)(651965463 & 255)), ((char)((651965463 >> 8) & 255)), ((char)((651965463 >> 16) & 255)), ((char)((651965463 >> 24) & 255)), ((char)(3456 & 255)), ((char)((3456 >> 8) & 255)), ((char)(17735 & 255)), ((char)((17735 >> 8) & 255)), ((char)(135 & 255)), ((char)(38 & 255)), ((char)(192 & 255)), ((char)(241 & 255)), ((char)(231 & 255)), ((char)(221 & 255)), ((char)(139 & 255)), ((char)(202 & 255))}}; 

#define CU_ETID_ClusterPrototype CU_ETID_CLUSTER_PROTOTYPE

typedef enum {
    CU_CLUSTER_PROT_SCHEDULING_POLICY_DEFAULT,
    CU_CLUSTER_PROT_SCHEDULING_POLICY_LOAD_BALANCING,
    CU_CLUSTER_PROT_SCHEDULING_POLICY_SPREAD
} CUclusterPrototypeSchedulingPolicy;

typedef struct CUetblClusterPrototype_st {
    size_t struct_size;

    // \brief Set the default cluster dimensions of a function. All
    // subsequent launches will default to use the set cluster size.
    // To reset, set all dimensions to 0.
    //
    CUresult (CUDAAPI *SetFunctionClusterDim)(CUfunction func, unsigned int clusterDimX, unsigned int clusterDimY, unsigned int clusterDimZ);

    // \brief Set the default cluster scheduling policy. All
    // subsequent launches of this kernel function will default to use
    // the set policy.
    // To reset, set the policy to CU_CLUSTER_PROT_SCHEDULING_POLICY_DEFAULT.
    //
    CUresult (CUDAAPI *SetFunctionClusterSchedulingPolicy)(CUfunction func, CUclusterPrototypeSchedulingPolicy policy);

    // \brief Allow the function to be launched with non-portable
    // cluster size.
    //
    CUresult (CUDAAPI *SetFunctionClusterNonPortableSizeSupport)(CUfunction func, unsigned int enable);

} CUetblClusterPrototype;

#if 0
{ // To trick formatter
#endif
#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __CLUSTER_PROTOTYPE_H__

} // end namespace bringup
} // end namespace cute
