#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

namespace tensorrt_llm::common {

template <typename T1, typename T2>
inline size_t divUp(T1 const& a, T2 const& b) {
    return (static_cast<size_t>(a) + static_cast<size_t>(b) - 1) / static_cast<size_t>(b);
}

inline int roundUp(int a, int b) { return divUp(a, b) * b; }

inline int getMultiProcessorCount() {
    int deviceId, smCount;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId);
    return smCount;
}

} // namespace tensorrt_llm::common

#define sync_check_cuda_error(stream)
