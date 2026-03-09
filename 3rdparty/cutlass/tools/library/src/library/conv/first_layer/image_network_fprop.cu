//
// CUTLASS includes
//
#include "cutlass/library/conv/image_network_first_layer_sm75.h"

namespace cutlass_build_env {
namespace library {

void image_network_first_layer_hmma_fprop(cutlass::Tensor4DCoord input_tensor_size,
                       cutlass::Tensor4DCoord conv_filter_size,
                       cutlass::Tensor4DCoord output_tensor_size,
                       cutlass::Tensor4DCoord bias_tensor_size,
                       TensorRef<half_t, layout::TensorNHWC> ref_A,
                       TensorRef<half_t, layout::TensorNHWC> ref_B,
                       TensorRef<half_t, layout::TensorNHWC> ref_C,
                       TensorRef<half_t, layout::TensorNHWC> ref_bias) {
  int const kFltK = conv_filter_size.n();
  int const kFltR = conv_filter_size.h();
  int const kFltS = conv_filter_size.w();
  int const kFltC = conv_filter_size.c();
  half_t const kPosINF = cutlass::half_t::bitcast(0x7c00);
  half_t const kNagINF = cutlass::half_t::bitcast(0xfc00);

  int const kPadH = (kFltR - 1) / 2;
  int const kPadW = (kFltS - 1) / 2;

  int const kWithBias = 0;
  int const kWithRelu = 0;
  half_t const kRelu = kWithRelu ? half_t(0.0) : kNagINF;
  half_t const kUpperBound = kWithRelu ? half_t(10.0) : kPosINF;
  half_t const kAlpha = half_t(1.0);

  typedef void (*pf)(cutlass::Tensor4DCoord input_tensor_size,
                     cutlass::Tensor4DCoord conv_filter_size,
                     cutlass::Tensor4DCoord output_tensor_size,
                     cutlass::Tensor4DCoord bias_tensor_size,
                     TensorRef<half_t, layout::TensorNHWC> ref_A,
                     TensorRef<half_t, layout::TensorNHWC> ref_B,
                     TensorRef<half_t, layout::TensorNHWC> ref_C,
                     TensorRef<half_t, layout::TensorNHWC> ref_bias,
                     int pad_h,
                     int pad_w,
                     half_t alpha,
                     int withBias,
                     int withRelu,
                     half_t relu,
                     half_t upperBound);

  const pf kernels[18] = {//
                          // C = 4
                          //
                          image_network_first_layer_hmma_fprop_kernel<2, 4, 3, 3, 16, 16>,
                          image_network_first_layer_hmma_fprop_kernel<3, 4, 5, 5, 16, 16>,
                          image_network_first_layer_hmma_fprop_kernel<4, 4, 7, 7, 16, 16>,
                          image_network_first_layer_hmma_fprop_kernel<2, 4, 3, 3, 32, 32>,
                          image_network_first_layer_hmma_fprop_kernel<3, 4, 5, 5, 32, 32>,
                          image_network_first_layer_hmma_fprop_kernel<4, 4, 7, 7, 32, 32>,
                          image_network_first_layer_hmma_fprop_kernel<2, 4, 3, 3, 64, 64>,
                          image_network_first_layer_hmma_fprop_kernel<3, 4, 5, 5, 64, 64>,
                          image_network_first_layer_hmma_fprop_kernel<4, 4, 7, 7, 64, 64>,

                          //
                          // C = 8
                          //
                          image_network_first_layer_hmma_fprop_kernel<2, 8, 3, 3, 16, 16>,
                          image_network_first_layer_hmma_fprop_kernel<3, 8, 5, 5, 16, 16>,
                          image_network_first_layer_hmma_fprop_kernel<4, 8, 7, 7, 16, 16>,
                          image_network_first_layer_hmma_fprop_kernel<2, 8, 3, 3, 32, 32>,
                          image_network_first_layer_hmma_fprop_kernel<3, 8, 5, 5, 32, 32>,
                          image_network_first_layer_hmma_fprop_kernel<4, 8, 7, 7, 32, 16>,
                          image_network_first_layer_hmma_fprop_kernel<2, 8, 3, 3, 64, 64>,
                          image_network_first_layer_hmma_fprop_kernel<3, 8, 5, 5, 64, 32>,
                          image_network_first_layer_hmma_fprop_kernel<4, 8, 7, 7, 64, 16>};

  int kernel_idx = 0;
  kernel_idx += (kFltC / 4 - 1) * 9;
  if (kFltK == 32) {
    kernel_idx += 3;
  } else if (kFltK == 64) {
    kernel_idx += 6;
  }
  kernel_idx += (kFltR - 3) / 2;
  int warps_per_cta = 2;

  if (kFltR == 3 && kFltS == 3)
    warps_per_cta = 2;
  else if (kFltR == 5 && kFltS == 5)
    warps_per_cta = 3;
  else if (kFltR == 7 && kFltS == 7)
    warps_per_cta = 4;

  dim3 block = dim3(warps_per_cta * 32);

  int ctas_in_row = (output_tensor_size.h() + 15) / 16;
  int ctas_in_col = (output_tensor_size.w() + 63) / 64;

  if (kFltC == 8) {
    if (kFltR == 7 && kFltS == 7) {
      if (kFltK == 32) {
        ctas_in_col *= 2;
      } else if (kFltK == 64) {
        ctas_in_col *= 4;
      }
    } else if (kFltR == 5 && kFltS == 5) {
      if (kFltK == 64) {
        ctas_in_col *= 2;
      }
    }
  }
  dim3 grid = dim3(ctas_in_col, ctas_in_row, output_tensor_size.n());

  // Launch kernel
  kernels[kernel_idx]<<<grid, block>>>(
      input_tensor_size, conv_filter_size, output_tensor_size, bias_tensor_size, ref_A, ref_B, ref_C, ref_bias, kPadH, kPadW, kAlpha, kWithBias, kWithRelu, kRelu, kUpperBound);
}
}  // namespace library
}  // namespace cutlass_build_env

