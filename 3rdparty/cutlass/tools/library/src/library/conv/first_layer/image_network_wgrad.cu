// Standard Library include
#include <iostream>

//
// CUTLASS includes
//

#include "cutlass/library/conv/image_network_first_layer_sm75.h"

namespace cutlass_build_env {
namespace library {

class DeviceWorkspaceAlignment {
private:

    static constexpr uintptr_t N = 128;

public:

    // Increase allocation size to allow pointer alignment.
    static uintptr_t pad(uintptr_t size) {
        if (size != 0) {
            size += N - 1;
        }
        return size;
    }

    // Align a pointer.
    static void* align(const void* ptr) {
        auto asInt = (uintptr_t)ptr;
        asInt += N - 1;
        asInt &= -N;
        return (void*)asInt;
    }
};

cudaError_t image_network_first_layer_hmma_wgrad(cutlass::Tensor4DCoord input_tensor_size,
                              cutlass::Tensor4DCoord output_tensor_size,
                              cutlass::Tensor4DCoord conv_filter_size,
                              TensorRef<half_t, layout::TensorNHWC> ref_A,
                              TensorRef<half_t, layout::TensorNHWC> ref_B,
                              TensorRef<half_t, layout::TensorNHWC> ref_C) {
  int const kFltK = conv_filter_size.n();
  int const kFltR = conv_filter_size.h();
  int const kFltS = conv_filter_size.w();
  int const kFltC = conv_filter_size.c();

  typedef void (*pf)(cutlass::Tensor4DCoord input_tensor_size,
                     cutlass::Tensor4DCoord output_tensor_size,
                     cutlass::Tensor4DCoord conv_filter_size,
                     TensorRef<half_t, layout::TensorNHWC> ref_A,
                     TensorRef<half_t, layout::TensorNHWC> ref_B,
                     TensorRef<half_t, layout::TensorNHWC> ref_C,
                     int const kParamsPadTop,
                     int const kParamsPadLeft,
                     int const kRowsPerCTA,
                     int const kNumLocks,
                     int *gmem_locks,
                     int *gmem_retired_ctas,
                     uint16_t *gmem_red);

  int kRowsPerCTA = output_tensor_size.h();

  int const kNumLocks = 64;
  // 16 bytes alignment for reduction buffer stg
  int const alignment = 16; 
  int *gmem_locks = nullptr;
  int *gmem_retired_ctas = nullptr;
  uint16_t *gmem_red = nullptr;

  int const kParamsPadTop = (kFltR - 1) / 2;
  int const kParamsPadLeft = (kFltS - 1) / 2;

  int const kFltKPerCTA = kFltK; 
  int const kCtasPerK = (kFltK + kFltKPerCTA - 1) / kFltKPerCTA;

  long unsigned const kLocksSize = ((kNumLocks + 1) * kCtasPerK * kFltR * sizeof(int) + alignment - 1)
      / alignment * alignment;
  long unsigned const kReductionBufSize =
      kFltR * kFltS * 4 * kFltKPerCTA * kCtasPerK * kNumLocks * sizeof(uint16_t);
  long unsigned const kDeviceBufSize = kLocksSize + kReductionBufSize;
  void *device_buf_ = nullptr;
  // Allocate space
  cudaError_t result;
  result = cudaMalloc(reinterpret_cast<void **>(&device_buf_), kDeviceBufSize);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate memory for locks and reduction buffer on device: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }
  device_buf_ = DeviceWorkspaceAlignment::align(device_buf_);

  result = cudaMemset(device_buf_, 0, kDeviceBufSize);
  if (result != cudaSuccess) {
    cudaFree(device_buf_);
    std::cerr << "Failed to clear locks and reduction buffer: " << cudaGetErrorString(result)
              << std::endl;
    return result;
  }

  gmem_locks = reinterpret_cast<int *>(device_buf_);
  gmem_red = reinterpret_cast<uint16_t *>((char *)device_buf_ + kLocksSize);
  gmem_retired_ctas = &gmem_locks[kNumLocks * kCtasPerK * kFltR];

  const pf kernels[18] = {
      image_network_first_layer_hmma_wgrad_kernel<3, 4, 3, 3, 16, 16, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<5, 4, 5, 5, 16, 16, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<7, 4, 7, 7, 16, 16, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<3, 4, 3, 3, 32, 32, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<5, 4, 5, 5, 32, 32, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<7, 4, 7, 7, 32, 32, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<3, 4, 3, 3, 64, 64, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<5, 4, 5, 5, 64, 64, cutlass::conv::Stride<2, 2>>,
      image_network_first_layer_hmma_wgrad_kernel<7, 4, 7, 7, 64, 64, cutlass::conv::Stride<2, 2>>};

  int kernel_idx = 0;
  kernel_idx += (kFltC / 4 - 1) * 9;
  if (kFltK == 32) {
    kernel_idx += 3;
  } else if (kFltK == 64) {
    kernel_idx += 6;
  }
  kernel_idx += (kFltR - 3) / 2;

  int warps_per_cta = kFltR;

  dim3 block = dim3(warps_per_cta * 32);

  int ctas_in_row = (output_tensor_size.h() + kRowsPerCTA - 1) / kRowsPerCTA;
  int ctas_in_col = (output_tensor_size.w() + 31) / 32 * kCtasPerK;

  dim3 grid = dim3(ctas_in_col, ctas_in_row, output_tensor_size.n());
  // Launch kernel
  kernels[kernel_idx]<<<grid, block>>>(input_tensor_size,
                                       output_tensor_size,
                                       conv_filter_size,
                                       ref_A,
                                       ref_B,
                                       ref_C,
                                       kParamsPadTop,
                                       kParamsPadLeft,
                                       kRowsPerCTA,
                                       kNumLocks,
                                       gmem_locks,
                                       gmem_retired_ctas,
                                       gmem_red);
  return cudaSuccess;
}
}  // namespace library
}  // namespace cutlass_build_env
