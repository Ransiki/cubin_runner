# trtllmgen_runner — FP4 × BF16 MOE using trtllm-gen exported cubins
#
# Build:
#   mkdir -p build && cd build && cmake .. && make -j$(nproc)
#
# Libraries:
#   libmoe_mxfp4_cubin_lib_sm100a.so        — MxFP4 fused (172 cubins: route + SwiGLU)
#   libmoe_mxfp4_nofuse_cubin_lib_sm100a.so — MxFP4 nofuse (106 cubins: route, no SwiGLU)
#   libmoe_mxfp4_cubin_lib_sm103a.so        — MxFP4 fused for SM103

cmake_minimum_required(VERSION 3.18)
project(trtllmgen_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include")

set(ROUTING_SOURCES
    fi_routing/routing_wrapper.cu
    fi_routing/trtllm_fused_moe_routing_renormalize.cu
    fi_routing/trtllm_fused_moe_routing_deepseek.cu
    fi_routing/trtllm_fused_moe_dev_kernel.cu
)

function(add_moe_cubin_target TARGET_NAME EXPORT_DIR DTYPE_MODE_VAL)
    file(GLOB _CUBIN_SRCS "${EXPORT_DIR}/cubins/*.cpp")
    add_library(${TARGET_NAME} SHARED
        moe_cubin_lib.cu
        ${ROUTING_SOURCES}
        ${_CUBIN_SRCS}
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        TLLM_GEN_EXPORT_INTERFACE
        TLLM_ENABLE_CUDA
        DTYPE_MODE=${DTYPE_MODE_VAL}
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${EXPORT_DIR}
        ${EXPORT_DIR}/trtllm/gen
        ${CMAKE_CURRENT_SOURCE_DIR}/test_stubs
        ${CMAKE_CURRENT_SOURCE_DIR}/fi_routing
        ${CMAKE_CURRENT_SOURCE_DIR}/fi_routing/include
    )
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CUTLASS_DIR})
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_ARCHITECTURES "100a"
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

# MxFP4 fused (route + SwiGLU) — 172 cubins
add_moe_cubin_target(moe_mxfp4_cubin_lib_sm100a
    "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_mxfp4_export_sm100a" 1)

# MxFP4 nofuse (route, NO SwiGLU) — 106 cubins
add_moe_cubin_target(moe_mxfp4_nofuse_cubin_lib_sm100a
    "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_mxfp4_nofuse_export_sm100a" 1)
target_compile_definitions(moe_mxfp4_nofuse_cubin_lib_sm100a PRIVATE NOFUSE_MODE)

# MxFP4 fused for SM103
add_moe_cubin_target(moe_mxfp4_cubin_lib_sm103a
    "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_mxfp4_export_sm103a" 1)
