# trtllmgen_runner — FP4 × BF16 MOE using trtllm-gen exported cubins
#
# Build:
#   mkdir -p build && cd build && cmake .. && make -j$(nproc)
#
# Builds three libraries:
#   libmoe_cubin_lib.so              — NvFP4 (E2m1) cubins (all Blackwell)
#   libmoe_mxfp4_cubin_lib_sm100a.so — MxFP4 cubins for SM100/SM100a (GB200)
#   libmoe_mxfp4_cubin_lib_sm103a.so — MxFP4 cubins for SM103/SM103a

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

# NvFP4 (E2m1 × BF16)
add_moe_cubin_target(moe_cubin_lib
    "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_nvfp4_export" 0)

# MxFP4 (MxE2m1 × BF16, patchF2fp) — one .so per GPU arch variant
add_moe_cubin_target(moe_mxfp4_cubin_lib_sm100a
    "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_mxfp4_export_sm100a" 1)

add_moe_cubin_target(moe_mxfp4_cubin_lib_sm103a
    "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_mxfp4_export_sm103a" 1)
