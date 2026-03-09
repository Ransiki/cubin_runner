# trtllmgen_runner — NvFP4 × BF16 MOE using trtllm-gen exported cubins
#
# Build:
#   mkdir -p build && cd build && cmake .. && make -j$(nproc)
#
# Fully self-contained: all dependencies (cubins, cutlass headers) are bundled.

cmake_minimum_required(VERSION 3.18)
project(trtllmgen_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Cubin export directory (self-contained, shipped with this project)
set(BMM_EXPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtllmGen_bmm_export")

# cutlass headers (bundled in 3rdparty/)
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include")

# Collect cubin .cpp files
file(GLOB CUBIN_SOURCES "${BMM_EXPORT_DIR}/cubins/*.cpp")

add_library(moe_cubin_lib SHARED
    moe_cubin_lib.cu
    fi_routing/routing_wrapper.cu
    fi_routing/trtllm_fused_moe_routing_renormalize.cu
    fi_routing/trtllm_fused_moe_routing_deepseek.cu
    ${CUBIN_SOURCES}
)

target_compile_definitions(moe_cubin_lib PRIVATE
    TLLM_GEN_EXPORT_INTERFACE
    TLLM_ENABLE_CUDA
)

target_include_directories(moe_cubin_lib PRIVATE
    ${BMM_EXPORT_DIR}
    ${BMM_EXPORT_DIR}/trtllm/gen
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stubs
    ${CMAKE_CURRENT_SOURCE_DIR}/fi_routing
    ${CMAKE_CURRENT_SOURCE_DIR}/fi_routing/include
)

target_include_directories(moe_cubin_lib SYSTEM PRIVATE
    ${CUTLASS_DIR}
)

target_link_libraries(moe_cubin_lib PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(moe_cubin_lib PROPERTIES
    CUDA_ARCHITECTURES "100"
    POSITION_INDEPENDENT_CODE ON
)
