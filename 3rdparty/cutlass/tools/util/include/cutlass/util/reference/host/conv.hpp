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
/*! \file
    \brief Reference implementation for CONV in host-side code.
*/
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"

#include "cute/tensor.hpp"

#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<class EngineAct, class LayoutAct>
bool
is_activation_in_bounds(
    cute::Tensor<EngineAct, LayoutAct> const& activation,
    int32_t n_, int32_t d_, int32_t h_, int32_t w_, int32_t c_, int32_t g_) {
  return ((g_ >= 0 && g_ < size<5>(activation)) &&
          (n_ >= 0 && n_ < size<4>(activation)) &&
          (d_ >= 0 && d_ < size<3>(activation)) &&
          (h_ >= 0 && h_ < size<2>(activation)) &&
          (w_ >= 0 && w_ < size<1>(activation)) &&
          (c_ >= 0 && c_ < size<0>(activation)));
}

template<class EngineAct, class LayoutAct>
bool
is_activation_in_bounds(
    cute::Tensor<EngineAct, LayoutAct> const& activation,
    int32_t n_, int32_t h_, int32_t w_, int32_t c_, int32_t g_) {
  return ((g_ >= 0 && g_ < size<4>(activation)) &&
          (n_ >= 0 && n_ < size<3>(activation)) &&
          (h_ >= 0 && h_ < size<2>(activation)) &&
          (w_ >= 0 && w_ < size<1>(activation)) &&
          (c_ >= 0 && c_ < size<0>(activation)));
}

template<class EngineAct, class LayoutAct>
bool
is_activation_in_bounds(
    cute::Tensor<EngineAct, LayoutAct> const& activation,
    int32_t n_, int32_t w_, int32_t c_, int32_t g_) {
  return ((g_ >= 0 && g_ < size<3>(activation)) &&
          (n_ >= 0 && n_ < size<2>(activation)) &&
          (w_ >= 0 && w_ < size<1>(activation)) &&
          (c_ >= 0 && c_ < size<0>(activation)));
}

} // namespace detail

// {$nv-internal-release begin}
template<
  class TensorBatchNormApply_,
  class TensorBatchNormStat_,
  class TensorFpropAct_,
  class DeActivationFunctor_,
  bool IsBatchNormApplySupported_ = false,
  bool IsBatchNormStatSupported_ = false,
  bool IsDualBatchNormSupported_ = false
>
struct BatchNormEpilogueFusionParams {
  using TensorBatchNormApply = TensorBatchNormApply_;
  using TensorBatchNormStat = TensorBatchNormStat_;
  using TensorFpropAct = TensorFpropAct_;
  using DeActivationFunctor = DeActivationFunctor_;

  using ElementBatchNormApply = typename TensorBatchNormApply::value_type;
  using ElementBatchNormStat = typename TensorBatchNormStat::value_type;
  using ElementFpropAct = typename TensorFpropAct::value_type;

  static constexpr bool IsBatchNormApplySupported = IsBatchNormApplySupported_;
  static constexpr bool IsBatchNormStatSupported = IsBatchNormStatSupported_;
  static constexpr bool IsDualBatchNormSupported = IsDualBatchNormSupported_;

  TensorFpropAct tensor_fprop_act{};
  TensorBatchNormApply tensor_fprop_alpha{};
  TensorBatchNormApply tensor_fprop_bias{};
  TensorBatchNormApply tensor_fprop_mean{};
  TensorBatchNormApply tensor_fprop_inv_stddev{};

  TensorFpropAct tensor_second_fprop_act{};
  TensorBatchNormApply tensor_second_fprop_alpha{};
  TensorBatchNormApply tensor_second_fprop_bias{};
  TensorBatchNormApply tensor_second_fprop_mean{};
  TensorBatchNormApply tensor_second_fprop_inv_stddev{};

  TensorBatchNormStat tensor_sum{};
  TensorBatchNormStat tensor_sum_of_square{};
  TensorBatchNormStat tensor_mean{};
  TensorBatchNormStat tensor_inv_stddev{};

  TensorBatchNormStat tensor_dbna_eq_dy_scale{};
  TensorBatchNormStat tensor_dbna_eq_x_scale{};
  TensorBatchNormStat tensor_dbna_eq_bias_scale{};

  TensorBatchNormStat tensor_second_sum{};
  TensorBatchNormStat tensor_second_sum_of_square{};
  TensorBatchNormStat tensor_second_dbna_eq_dy_scale{};
  TensorBatchNormStat tensor_second_dbna_eq_x_scale{};
  TensorBatchNormStat tensor_second_dbna_eq_bias_scale{};

  ElementBatchNormStat inv_count = ElementBatchNormStat(0);
  ElementBatchNormStat epsilon = ElementBatchNormStat(0);

  DeActivationFunctor epi_de_activation;

  void initialize() {
    if constexpr (IsBatchNormStatSupported) {
      for (int32_t i = 0; i < tensor_sum.size(); ++i) {
        tensor_sum(i) = ElementBatchNormStat(0.f);
        tensor_sum_of_square(i) = ElementBatchNormStat(0.f);
        if constexpr (IsDualBatchNormSupported) {
          tensor_second_sum_of_square(i) = ElementBatchNormStat(0.f);
        }
      }
    }
  }

  template<cutlass::conv::Operator ConvOp, class Coord, class ElementCompute>
  ElementCompute sum_reduce(Coord const& coord, ElementCompute acc) {
    ElementCompute result = acc;
    if constexpr (IsBatchNormStatSupported) {
      NumericConverter<ElementBatchNormStat, ElementCompute> batchnorm_stat_converter;
      int32_t coord0 = cute::size<0>(coord);
      if constexpr (ConvOp == cutlass::conv::Operator::kFprop) {
        tensor_sum(coord0) += batchnorm_stat_converter(result);
        tensor_sum_of_square(coord0) += batchnorm_stat_converter(result * result);
      } else if constexpr (ConvOp == cutlass::conv::Operator::kDgrad) {
        NumericConverter<ElementCompute, ElementBatchNormApply> batchnorm_apply_converter;
        NumericConverter<ElementCompute, ElementFpropAct> fprop_act_converter;
        if constexpr (IsBatchNormApplySupported) {
          // redo batchnorm apply in dgrad epilogue
          ElementCompute x_norm = (fprop_act_converter(tensor_fprop_act(coord)) - batchnorm_apply_converter(tensor_fprop_mean(coord0)))
            * batchnorm_apply_converter(tensor_fprop_inv_stddev(coord0)) * batchnorm_apply_converter(tensor_fprop_alpha(coord0))
            + batchnorm_apply_converter(tensor_fprop_bias(coord0));
          if constexpr (IsDualBatchNormSupported) {
            x_norm += (fprop_act_converter(tensor_second_fprop_act(coord)) - batchnorm_apply_converter(tensor_second_fprop_mean(coord0)))
              * batchnorm_apply_converter(tensor_second_fprop_inv_stddev(coord0)) * batchnorm_apply_converter(tensor_second_fprop_alpha(coord0))
              + batchnorm_apply_converter(tensor_second_fprop_bias(coord0));
          }
          // calculate de_act acoording to the result of batchnorm apply
          result = epi_de_activation(result, x_norm);
        }
        tensor_sum(coord0) += batchnorm_stat_converter(result);
        tensor_sum_of_square(coord0) += batchnorm_stat_converter((fprop_act_converter(tensor_fprop_act(coord))
          - batchnorm_apply_converter(tensor_fprop_mean(coord0))) * result);
        if constexpr (IsDualBatchNormSupported) {
          tensor_second_sum_of_square(coord0) += batchnorm_stat_converter((fprop_act_converter(tensor_second_fprop_act(coord))
          - batchnorm_apply_converter(tensor_second_fprop_mean(coord0))) * result);
        }
      } else {
        // do nothing for wgrad
      }
    }
    return result;
  }

  template<cutlass::conv::Operator ConvOp>
  void statistics() {
    if constexpr (IsBatchNormStatSupported) {
      for (int32_t i = 0; i < tensor_sum.size(); ++i) {
        if constexpr (ConvOp == cutlass::conv::Operator::kFprop) {
          // mean = sum / count
          // stddev = 1 / sqrt(sum_of_square / count - mean * mean + epsilon)
          tensor_mean(i) = tensor_sum(i) * inv_count;
          auto var = tensor_sum_of_square(i) * inv_count - tensor_mean(i) * tensor_mean(i);
          tensor_inv_stddev(i) = rsqrtf(var + epsilon);
        } else if constexpr (ConvOp == cutlass::conv::Operator::kDgrad) {
          // sum_of_square = sum_of_sqaure * inv_stddev
          // dbna_eq_dy_scale = alpha * inv_stddev
          // dbna_eq_x_scale = -dbna_eq_dy_scale * sum_of_square * inv_stddev / count
          // dbna_eq_bias_scale = dbna_eq_dy_scale * (mean * sum_of_square * inv_stddev - sum) / count
          tensor_sum_of_square(i) *= tensor_fprop_inv_stddev(i);
          tensor_dbna_eq_dy_scale(i) = tensor_fprop_alpha(i) * tensor_fprop_inv_stddev(i); 
          tensor_dbna_eq_x_scale(i) = -1.f * tensor_dbna_eq_dy_scale(i) * inv_count * tensor_sum_of_square(i)
            * tensor_fprop_inv_stddev(i);
          tensor_dbna_eq_bias_scale(i) = tensor_dbna_eq_dy_scale(i) * inv_count * (tensor_fprop_mean(i)
            * tensor_sum_of_square(i) * tensor_fprop_inv_stddev(i) - tensor_sum(i));
          if constexpr (IsDualBatchNormSupported) {
            // same with the first batchnorm
            tensor_second_sum(i) = tensor_sum(i);
            tensor_second_sum_of_square(i) *= tensor_second_fprop_inv_stddev(i);
            tensor_second_dbna_eq_dy_scale(i) = tensor_second_fprop_alpha(i) * tensor_second_fprop_inv_stddev(i); 
            tensor_second_dbna_eq_x_scale(i) = -1.f * tensor_second_dbna_eq_dy_scale(i)
              * inv_count * tensor_second_sum_of_square(i) * tensor_second_fprop_inv_stddev(i);
            tensor_second_dbna_eq_bias_scale(i) = tensor_second_dbna_eq_dy_scale(i) * inv_count * (tensor_second_fprop_mean(i)
              * tensor_second_sum_of_square(i) * tensor_second_fprop_inv_stddev(i) - tensor_sum(i));
          }
        } else {
          // do nothing for wgrad
        }
      }
    }
  }
};

template<
  class TensorAlpha_,
  class TensorBias_,
  template <class> class ActivationFunctor_ = cutlass::epilogue::thread::Identity,
  bool IsBatchNormApplySupported_ = false
>
struct BatchNormMainloopFusionParams {
  using TensorAlpha  = TensorAlpha_;
  using TensorBias   = TensorBias_;

  using ElementAlpha = typename TensorAlpha::value_type;
  using ElementBias  = typename TensorBias::value_type;

  static constexpr bool IsBatchNormApplySupported = IsBatchNormApplySupported_;

  TensorAlpha tensor_alpha{};
  TensorBias tensor_bias{};

  template<class Coord, class ElementCompute>
  ElementCompute apply(Coord const& coord, ElementCompute acc) {
    using ActivationFunctor = ActivationFunctor_<ElementCompute>;
    // Mainloop activation operation
    ActivationFunctor activation;

    ElementCompute result = acc;
    if constexpr (IsBatchNormApplySupported) {
      int32_t coord0 = cute::size<0>(coord);
      NumericConverter<ElementCompute, ElementAlpha> alpha_converter;
      NumericConverter<ElementCompute, ElementBias> bias_converter;
      result = activation(alpha_converter(tensor_alpha(coord0)) * acc
          + bias_converter(tensor_bias(coord0)));
    }
    return result;
  }
};

template<
  class BatchNormMainloopFusionParams_
>
struct ConvMainloopFusionParams {
  using BatchNormMainloopFusionParams = BatchNormMainloopFusionParams_;
  BatchNormMainloopFusionParams bn_fusion_params;
};

// {$nv-internal-release end}

template<
  class ElementAcc_,
  class ElementScalar_,
  class ElementCompute_,
  class ElementC_,
  class ElementOut_,
  bool ResidualAdd_,
  class TensorAlpha_,
  class TensorBeta_,
  class TensorBias_,
  class ActivationFunctor_ = cutlass::epilogue::thread::Identity<ElementCompute_>
  , // {$nv-internal-release}
  class BatchNormEpilogueFusionParams_ = BatchNormEpilogueFusionParams<TensorBias_, TensorBias_, TensorBias_, cutlass::first<ElementCompute_>> // {$nv-internal-release}
>
struct ConvEpilogueFusionParams {
  using ElementAcc = ElementAcc_;
  using ElementScalar = ElementScalar_;
  using ElementCompute = ElementCompute_;
  using ElementC = ElementC_;
  using ElementOut = ElementOut_;
  using TensorAlpha = TensorAlpha_;
  using TensorBeta = TensorBeta_;
  using TensorBias = TensorBias_;
  using ActivationFunctor = ActivationFunctor_;
  using BatchNormEpilogueFusionParams = BatchNormEpilogueFusionParams_; // {$nv-internal-release}

  static constexpr bool ResidualAdd = ResidualAdd_; // Source added after activation

  ElementScalar alpha = ElementScalar(1);
  ElementScalar beta = ElementScalar(0);

  TensorAlpha tensor_alpha{};
  TensorBeta tensor_beta{};
  TensorBias tensor_bias{};
  BatchNormEpilogueFusionParams bn_fusion_params{}; // {$nv-internal-release}
};

template<
  cutlass::conv::Operator ConvOp,
  int NumSpatialDims,
  class TensorA,
  class TensorB,
  class TensorC,
  class TensorD,
  class ShapePadding,
  class StrideTraversal,
  class ShapeDilation,
  class EpilogueFusionParams
  , // {$nv-internal-release}
  class MainloopFusionParams = ConvMainloopFusionParams<BatchNormMainloopFusionParams< // {$nv-internal-release}
                               decltype(make_tensor(make_gmem_ptr(recast_ptr<float>(nullptr)), make_layout(make_shape(0)))), // {$nv-internal-release}
                               decltype(make_tensor(make_gmem_ptr(recast_ptr<float>(nullptr)), make_layout(make_shape(0))))>> // {$nv-internal-release}
>
struct ConvReferenceImpl {
  // Hard code accumlulator type to float to avoid data lost in accumulating add.
  using ElementAcc = cutlass::platform::conditional_t<cutlass::platform::is_same_v<typename EpilogueFusionParams::ElementAcc, double>, double, float>;
  using ElementC = typename EpilogueFusionParams::ElementC;
  using ElementOut = typename EpilogueFusionParams::ElementOut;
  using ElementScalar = typename EpilogueFusionParams::ElementScalar;
  using ElementCompute = typename EpilogueFusionParams::ElementCompute;
  using ElementBias = typename EpilogueFusionParams::TensorBias::value_type;
  using ActivationFunctor = typename EpilogueFusionParams::ActivationFunctor;

  // Input related converter
  NumericConverter<ElementCompute, ElementAcc> acc_converter;
  NumericConverter<ElementCompute, ElementC> residual_converter;
  NumericConverter<ElementCompute, ElementBias> bias_converter;
  // Scale related converter
  NumericConverter<ElementCompute, ElementScalar> scale_converter;
  // Output related converter
  NumericConverter<ElementOut, ElementCompute> output_converter;

  EpilogueFusionParams& epi_fusion_params_;
  MainloopFusionParams& main_fusion_params_; // {$nv-internal-release}

  TensorA const& tensor_a_;
  TensorB const& tensor_b_;
  TensorC const& tensor_c_;
  TensorD& tensor_d_;

  ShapePadding const& padding_;
  StrideTraversal const& tstride_;
  ShapeDilation const& dilation_;

  // Epilogue activation operation
  ActivationFunctor epi_activation;

  ConvReferenceImpl(
    TensorA const& tensor_a,
    TensorB const& tensor_b,
    TensorC const& tensor_c,
    TensorD& tensor_d,
    ShapePadding const& padding,
    StrideTraversal const& tstride,
    ShapeDilation const& dilation,
    EpilogueFusionParams& epi_fusion_params)
  : tensor_a_(tensor_a),
    tensor_b_(tensor_b),
    tensor_c_(tensor_c),
    tensor_d_(tensor_d),
    padding_(padding),
    tstride_(tstride),
    dilation_(dilation),
    epi_fusion_params_(epi_fusion_params)
    , main_fusion_params_(reinterpret_cast<MainloopFusionParams&>(epi_fusion_params)) // {$nv-internal-release}
  {
    static_assert(rank(ShapePadding{}) == rank(ShapeDilation{}));
    static_assert(rank(ShapePadding{}) == rank(StrideTraversal{}));
  }

  // {$nv-internal-release begin}
  ConvReferenceImpl(
    TensorA const& tensor_a,
    TensorB const& tensor_b,
    TensorC const& tensor_c,
    TensorD& tensor_d,
    ShapePadding const& padding,
    StrideTraversal const& tstride,
    ShapeDilation const& dilation,
    EpilogueFusionParams& epi_fusion_params,
    MainloopFusionParams& main_fusion_params
    )
  : tensor_a_(tensor_a),
    tensor_b_(tensor_b),
    tensor_c_(tensor_c),
    tensor_d_(tensor_d),
    padding_(padding),
    tstride_(tstride),
    dilation_(dilation),
    epi_fusion_params_(epi_fusion_params),
    main_fusion_params_(main_fusion_params)
  {
    static_assert(rank(ShapePadding{}) == rank(ShapeDilation{}));
    static_assert(rank(ShapePadding{}) == rank(StrideTraversal{}));
  }
  // {$nv-internal-release end}

  void compute_reference() {
    if constexpr (ConvOp == cutlass::conv::Operator::kFprop) {
      fprop_reference(cute::Int<NumSpatialDims>{});
    }
    else if constexpr (ConvOp == cutlass::conv::Operator::kDgrad) {
      dgrad_reference(cute::Int<NumSpatialDims>{});
    }
    else {
      wgrad_reference(cute::Int<NumSpatialDims>{});
    }
  }

private:
  // Specialization for 1D fprop kernel
  void fprop_reference(cute::Int<1> spatial_dims) {
    int32_t G = size<3>(tensor_d_);
    int32_t N = size<2>(tensor_d_);
    int32_t Q = size<1>(tensor_d_);
    int32_t K = size<0>(tensor_d_);
    int32_t S = size<1>(tensor_b_);
    int32_t C = size<0>(tensor_b_);

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.initialize();
    // {$nv-internal-release end}

#if defined(_OPENMP)
  #pragma omp parallel for collapse(2)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t n = 0; n < N; ++n) {
        for (int32_t q = 0; q < Q; ++q) {
          for (int32_t k = 0; k < K; ++k) {
            auto accumulator = ElementAcc(0);
            for (int32_t s = 0; s < S; ++s) {
              for (int32_t c = 0; c < C; ++c) {
                int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                if (detail::is_activation_in_bounds(tensor_a_, n, w, c, g)) {
                  auto a = tensor_a_(c, w, n, g);
                  auto b = tensor_b_(c, s, k, g);
                  // {$nv-internal-release begin}
                  a = main_fusion_params_.bn_fusion_params.apply(cute::make_coord(c, w, n, g), a);
                  // {$nv-internal-release end}
                  accumulator += ElementAcc(a * b);
                }
              }
            }
            ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
              epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
            ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
              epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
            ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
            if (not EpilogueFusionParams::ResidualAdd) {
              output += scale_converter(beta) * residual_converter(tensor_c_(k, q, n, g));
            }
            if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
              output += bias_converter(epi_fusion_params_.tensor_bias[k]);
            }
            output = epi_activation(output);
            if (EpilogueFusionParams::ResidualAdd) {
              output += scale_converter(beta) * residual_converter(tensor_c_(k, q, n, g));
            }
            // {$nv-internal-release begin}
            output = epi_fusion_params_.bn_fusion_params.template sum_reduce<ConvOp>(
              cute::make_coord(k, q, n, g), output);
            // {$nv-internal-release end}
            tensor_d_(k, q, n, g) = output_converter(output);
          }
        }
      }
    }

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.template statistics<ConvOp>();
    // {$nv-internal-release end}
  }

  // Specialization for 2D fprop kernel
  void fprop_reference(cute::Int<2> spatial_dims) {
    int32_t G = size<4>(tensor_d_);
    int32_t N = size<3>(tensor_d_);
    int32_t P = size<2>(tensor_d_);
    int32_t Q = size<1>(tensor_d_);
    int32_t K = size<0>(tensor_d_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);
    int32_t C = size<0>(tensor_b_);

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.initialize();
    // {$nv-internal-release end}

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t n = 0; n < N; ++n) {
        for (int32_t p = 0; p < P; ++p) {
          for (int32_t q = 0; q < Q; ++q) {
            for (int32_t k = 0; k < K; ++k) {
              auto accumulator = ElementAcc(0);
              for (int32_t r = 0; r < R; ++r) {
                for (int32_t s = 0; s < S; ++s) {
                  for (int32_t c = 0; c < C; ++c) {
                    int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                    int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                    if (detail::is_activation_in_bounds(tensor_a_, n, h, w, c, g)) {
                      auto a = tensor_a_(c, w, h, n, g);
                      auto b = tensor_b_(c, s, r, k, g);
                      // {$nv-internal-release begin}
                      a = main_fusion_params_.bn_fusion_params.apply(cute::make_coord(c, w, h, n, g), a);
                      // {$nv-internal-release end}
                      accumulator += ElementAcc(a * b);
                    }
                  }
                }
              }
              ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
                epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
              ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
                epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
              ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
              if (not EpilogueFusionParams::ResidualAdd) {
                output += scale_converter(beta) * residual_converter(tensor_c_(k, q, p, n, g));
              }
              if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                output += bias_converter(epi_fusion_params_.tensor_bias[k]);
              }
              output = epi_activation(output);
              if (EpilogueFusionParams::ResidualAdd) {
                output += scale_converter(beta) * residual_converter(tensor_c_(k, q, p, n, g));
              }
              // {$nv-internal-release begin}
              output = epi_fusion_params_.bn_fusion_params.template sum_reduce<ConvOp>(
                cute::make_coord(k, q, p, n, g), output);
              // {$nv-internal-release end}
              tensor_d_(k, q, p, n, g) = output_converter(output);
            }
          }
        }
      }
    }

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.template statistics<ConvOp>();
    // {$nv-internal-release end}
  }

  // Specialization for 3D fprop kernel
  void fprop_reference(cute::Int<3> spatial_dims) {
    int32_t G = size<5>(tensor_d_);
    int32_t N = size<4>(tensor_d_);
    int32_t Z = size<3>(tensor_d_);
    int32_t P = size<2>(tensor_d_);
    int32_t Q = size<1>(tensor_d_);
    int32_t K = size<0>(tensor_d_);
    int32_t T = size<3>(tensor_b_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);
    int32_t C = size<0>(tensor_b_);

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.initialize();
    // {$nv-internal-release end}

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t n = 0; n < N; ++n) {
        for (int32_t z = 0; z < Z; ++z) {
          for (int32_t p = 0; p < P; ++p) {
            for (int32_t q = 0; q < Q; ++q) {
              for (int32_t k = 0; k < K; ++k) {
                auto accumulator = ElementAcc(0);
                for (int32_t t = 0; t < T; ++t) {
                  for (int32_t r = 0; r < R; ++r) {
                    for (int32_t s = 0; s < S; ++s) {
                      for (int32_t c = 0; c < C; ++c) {
                        int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                        int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                        int32_t d =  z * cute::get<2>(tstride_) - cute::get<2>(padding_) + t * cute::get<2>(dilation_);
                        if (detail::is_activation_in_bounds(tensor_a_, n, d, h, w, c, g)) {
                          auto a = tensor_a_(c, w, h, d, n, g);
                          auto b = tensor_b_(c, s, r, t, k, g);
                          // {$nv-internal-release begin}
                          a = main_fusion_params_.bn_fusion_params.apply(cute::make_coord(c, w, h, d, n, g), a);
                          // {$nv-internal-release end}
                          accumulator += ElementAcc(a * b);
                        }
                      }
                    }
                  }
                }
                ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
                  epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
                ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
                  epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
                ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
                if (not EpilogueFusionParams::ResidualAdd) {
                  output += scale_converter(beta) * residual_converter(tensor_c_(k, q, p, z, n, g));
                }
                if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                  output += bias_converter(epi_fusion_params_.tensor_bias[k]);
                }
                output = epi_activation(output);
                if (EpilogueFusionParams::ResidualAdd) {
                  output += scale_converter(beta) * residual_converter(tensor_c_(k, q, p, z, n, g));
                }
                // {$nv-internal-release begin}
                output = epi_fusion_params_.bn_fusion_params.template sum_reduce<ConvOp>(
                  cute::make_coord(k, q, p, z, n, g), output);
                // {$nv-internal-release end}
                tensor_d_(k, q, p, z, n, g) = output_converter(output);
              }
            }
          }
        }
      }
    }

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.template statistics<ConvOp>();
    // {$nv-internal-release end}

  }

  // Specialization for 1D dgrad kernel
  void dgrad_reference(cute::Int<1> spatial_dims) {
    int32_t G = size<3>(tensor_d_);
    int32_t N = size<2>(tensor_d_);
    int32_t W = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);
    int32_t K = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.initialize();
    // {$nv-internal-release end}

#if defined(_OPENMP)
   #pragma omp parallel for collapse(2)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t n = 0; n < N; ++n) {
        for (int32_t w = 0; w < W; ++w) {
          for (int32_t c = 0; c < C; ++c) {
            auto accumulator = ElementAcc(0);
            for (int32_t k = 0; k < K; ++k) {
              for (int32_t s = 0; s < S; ++s) {
                int32_t q = w + cute::get<0>(padding_) - s * cute::get<0>(dilation_);

                if (q % cute::get<0>(tstride_) == 0) {
                  q /= cute::get<0>(tstride_);
                } else {
                  continue;
                }

                if (detail::is_activation_in_bounds(tensor_a_, n, q, k, g)) {
                  accumulator += ElementAcc(tensor_a_(k, q, n, g) * tensor_b_(c, s, k, g));
                }
              }
            }
            ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data())
              ? epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
            ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data())
              ? epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;
            ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
            if (not EpilogueFusionParams::ResidualAdd) {
              output += scale_converter(beta) * residual_converter(tensor_c_(c, w, n, g));
            }
            if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
              output += bias_converter(epi_fusion_params_.tensor_bias[c]);
            }
            output = epi_activation(output);
            if (EpilogueFusionParams::ResidualAdd) {
              output += scale_converter(beta) * residual_converter(tensor_c_(c, w, n, g));
            }
            // {$nv-internal-release begin}
            output = epi_fusion_params_.bn_fusion_params.template sum_reduce<ConvOp>(
              cute::make_coord(c, w, n, g), output);
            // {$nv-internal-release end}
            tensor_d_(c, w, n, g) = output_converter(output);
          }
        }
      }
    }

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.template statistics<ConvOp>();
    // {$nv-internal-release end}
  }

  // Specialization for 2D dgrad kernel
  void dgrad_reference(cute::Int<2> spatial_dims) {
    int32_t G = size<4>(tensor_d_);
    int32_t N = size<3>(tensor_d_);
    int32_t H = size<2>(tensor_d_);
    int32_t W = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);
    int32_t K = size<3>(tensor_b_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.initialize();
    // {$nv-internal-release end}

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t n = 0; n < N; ++n) {
        for (int32_t h = 0; h < H; ++h) {
          for (int32_t w = 0; w < W; ++w) {
            for (int32_t c = 0; c < C; ++c) {
              auto accumulator = ElementAcc(0);
              for (int32_t k = 0; k < K; ++k) {
                for (int32_t r = 0; r < R; ++r) {
                  for (int32_t s = 0; s < S; ++s) {
                    int32_t q = w + cute::get<0>(padding_) - s * cute::get<0>(dilation_);
                    int32_t p = h + cute::get<1>(padding_) - r * cute::get<1>(dilation_);

                    if (q % cute::get<0>(tstride_) == 0) {
                      q /= cute::get<0>(tstride_);
                    } else {
                      continue;
                    }

                    if (p % cute::get<1>(tstride_) == 0) {
                      p /= cute::get<1>(tstride_);
                    } else {
                      continue;
                    }

                    if (detail::is_activation_in_bounds(tensor_a_, n, p, q, k, g)) {
                      accumulator += ElementAcc(tensor_a_(k, q, p, n, g) * tensor_b_(c, s, r, k, g));
                    }
                  }
                }
              }
              ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data())
                ? epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
              ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data())
                ? epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;
              ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
              if (not EpilogueFusionParams::ResidualAdd) {
                output += scale_converter(beta) * residual_converter(tensor_c_(c, w, h, n, g));
              }
              if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                output += bias_converter(epi_fusion_params_.tensor_bias[c]);
              }
              output = epi_activation(output);
              if (EpilogueFusionParams::ResidualAdd) {
                output += scale_converter(beta) * residual_converter(tensor_c_(c, w, h, n, g));
              }

              // {$nv-internal-release begin}
              output = epi_fusion_params_.bn_fusion_params.template sum_reduce<ConvOp>(
                cute::make_coord(c, w, h, n, g), output);
              // {$nv-internal-release end}
              tensor_d_(c, w, h, n, g) = output_converter(output);
            }
          }
        }
      }
    }

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.template statistics<ConvOp>();
    // {$nv-internal-release end}
  }

  // Specialization for 3D dgrad kernel
  void dgrad_reference(cute::Int<3> spatial_dims) {
    int32_t G = size<5>(tensor_d_);
    int32_t N = size<4>(tensor_d_);
    int32_t D = size<3>(tensor_d_);
    int32_t H = size<2>(tensor_d_);
    int32_t W = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);
    int32_t K = size<4>(tensor_b_);
    int32_t T = size<3>(tensor_b_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.initialize();
    // {$nv-internal-release end}

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t n = 0; n < N; ++n) {
        for (int32_t d = 0; d < D; ++d) {
          for (int32_t h = 0; h < H; ++h) {
            for (int32_t w = 0; w < W; ++w) {
              for (int32_t c = 0; c < C; ++c) {
                auto accumulator = ElementAcc(0);
                for (int32_t k = 0; k < K; ++k) {
                  for (int32_t t = 0; t < T; ++t) {
                    for (int32_t r = 0; r < R; ++r) {
                      for (int32_t s = 0; s < S; ++s) {
                        int32_t q = w + cute::get<0>(padding_) - s * cute::get<0>(dilation_);
                        int32_t p = h + cute::get<1>(padding_) - r * cute::get<1>(dilation_);
                        int32_t z = d + cute::get<2>(padding_) - t * cute::get<2>(dilation_);

                        if (q % cute::get<0>(tstride_) == 0) {
                          q /= cute::get<0>(tstride_);
                        } else {
                          continue;
                        }

                        if (p % cute::get<1>(tstride_) == 0) {
                          p /= cute::get<1>(tstride_);
                        } else {
                          continue;
                        }

                        if (z % cute::get<2>(tstride_) == 0) {
                          z /= cute::get<2>(tstride_);
                        } else {
                          continue;
                        }

                        if (detail::is_activation_in_bounds(tensor_a_, n, z, p, q, k, g)) {
                          accumulator += ElementAcc(tensor_a_(k, q, p, z, n, g) * tensor_b_(c, s, r, t, k, g));
                        }
                      }
                    }
                  }
                }
                ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data())
                  ? epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
                ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data())
                  ? epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;
                ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
                if (not EpilogueFusionParams::ResidualAdd) {
                  output += scale_converter(beta) * residual_converter(tensor_c_(c, w, h, d, n, g));
                }
                if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                  output += bias_converter(epi_fusion_params_.tensor_bias[c]);
                }
                output = epi_activation(output);
                if (EpilogueFusionParams::ResidualAdd) {
                  output += scale_converter(beta) * residual_converter(tensor_c_(c, w, h, d, n, g));
                }
                // {$nv-internal-release begin}
                output = epi_fusion_params_.bn_fusion_params.template sum_reduce<ConvOp>(
                  cute::make_coord(c, w, h, d, n, g), output);
                // {$nv-internal-release end}
                tensor_d_(c, w, h, d, n, g) = output_converter(output);
              }
            }
          }
        }
      }
    }

    // {$nv-internal-release begin}
    epi_fusion_params_.bn_fusion_params.template statistics<ConvOp>();
    // {$nv-internal-release end}

  }

  // Specialization for 1D wgrad kernel
  void wgrad_reference(cute::Int<1> spatial_dims) {
    static constexpr bool IsBatchNormApplySupported = MainloopFusionParams::BatchNormMainloopFusionParams::IsBatchNormApplySupported; // {$nv-internal-release}
    int32_t G = size<3>(tensor_d_);
    int32_t N =
        IsBatchNormApplySupported ? size<2>(tensor_b_) : // {$nv-internal-release}
        size<2>(tensor_a_);
    int32_t Q =
        IsBatchNormApplySupported ? size<1>(tensor_b_) : // {$nv-internal-release}
        size<1>(tensor_a_);
    int32_t K =
        IsBatchNormApplySupported ? size<0>(tensor_b_) : // {$nv-internal-release}
        size<0>(tensor_a_);
    int32_t S = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t k = 0; k < K; ++k) {
        for (int32_t s = 0; s < S; ++s) {
          for (int32_t c = 0; c < C; ++c) {
            auto accumulator = ElementAcc(0);
            for (int32_t n = 0; n < N; ++n) {
              for (int32_t q = 0; q < Q; ++q) {
                int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                bool is_in_bounds =
                    IsBatchNormApplySupported ? detail::is_activation_in_bounds(tensor_a_, n, w, c, g) : // {$nv-internal-release}
                    detail::is_activation_in_bounds(tensor_b_, n, w, c, g);
                if (is_in_bounds) {
                  auto act =
                      IsBatchNormApplySupported ? tensor_a_(c, w, n, g) : // {$nv-internal-release}
                      tensor_b_(c, w, n, g);
                  auto xformed_act =
                      IsBatchNormApplySupported ? tensor_b_(k, q, n, g) : // {$nv-internal-release}
                      tensor_a_(k, q, n, g);
                  // {$nv-internal-release begin}
                  act = main_fusion_params_.bn_fusion_params.apply(cute::make_coord(c, w, n, g), act);
                  // {$nv-internal-release end}
                  accumulator += ElementAcc(act * xformed_act);
                }
              }
            }

            ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
              epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
            ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
              epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;

            ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
            if (not EpilogueFusionParams::ResidualAdd) {
              output += scale_converter(beta) * residual_converter(tensor_c_(c, s, k, g));
            }
            if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
              output += bias_converter(epi_fusion_params_.tensor_bias[c]);
            }
            output = epi_activation(output);
            if (EpilogueFusionParams::ResidualAdd) {
              output += scale_converter(beta) * residual_converter(tensor_c_(c, s, k, g));
            }
            tensor_d_(c, s, k, g) = output_converter(output);
          }
        }
      }
    }
  }

  // Specialization for 2D wgrad kernel
  void wgrad_reference(cute::Int<2> spatial_dims) {
    static constexpr bool IsBatchNormApplySupported = MainloopFusionParams::BatchNormMainloopFusionParams::IsBatchNormApplySupported; // {$nv-internal-release}
    int32_t G = size<4>(tensor_d_);
    int32_t N =
        IsBatchNormApplySupported ? size<3>(tensor_b_) : // {$nv-internal-release}
        size<3>(tensor_a_);
    int32_t P =
        IsBatchNormApplySupported ? size<2>(tensor_b_) : // {$nv-internal-release}
        size<2>(tensor_a_);
    int32_t Q =
        IsBatchNormApplySupported ? size<1>(tensor_b_) : // {$nv-internal-release} 
        size<1>(tensor_a_);
    int32_t K =
        IsBatchNormApplySupported ? size<0>(tensor_b_) : // {$nv-internal-release}
        size<0>(tensor_a_);
    int32_t R = size<2>(tensor_d_);
    int32_t S = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t g = 0; g < G; ++g) {
      for (int32_t k = 0; k < K; ++k) {
        for (int32_t r = 0; r < R; ++r) {
          for (int32_t s = 0; s < S; ++s) {
            for (int32_t c = 0; c < C; ++c) {
              auto accumulator = ElementAcc(0);
              for (int32_t n = 0; n < N; ++n) {
                for (int32_t p = 0; p < P; ++p) {
                  for (int32_t q = 0; q < Q; ++q) {
                    int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                    int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                    bool is_in_bounds =
                        IsBatchNormApplySupported ? detail::is_activation_in_bounds(tensor_a_, n, h, w, c, g) : // {$nv-internal-release}
                        detail::is_activation_in_bounds(tensor_b_, n, h, w, c, g);
                    if (is_in_bounds) {
                      auto act =
                          IsBatchNormApplySupported ? tensor_a_(c, w, h, n, g) : // {$nv-internal-release}
                          tensor_b_(c, w, h, n, g);
                      auto xformed_act =
                          IsBatchNormApplySupported ? tensor_b_(k, q, p, n, g) : // {$nv-internal-release}
                          tensor_a_(k, q, p, n, g);
                      // {$nv-internal-release begin}
                      act = main_fusion_params_.bn_fusion_params.apply(cute::make_coord(c, w, h, n, g), act);
                      // {$nv-internal-release end}
                      accumulator += ElementAcc(act * xformed_act);
                    }
                  }
                }
              }

              ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
                epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
              ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
                epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;

              ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
              if (not EpilogueFusionParams::ResidualAdd) {
                output += scale_converter(beta) * residual_converter(tensor_c_(c, s, r, k, g));
              }
              if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                output += bias_converter(epi_fusion_params_.tensor_bias[c]);
              }
              output = epi_activation(output);
              if (EpilogueFusionParams::ResidualAdd) {
                output += scale_converter(beta) * residual_converter(tensor_c_(c, s, r, k, g));
              }
              tensor_d_(c, s, r, k, g) = output_converter(output);
            }
          }
        }
      }
    }
  }

  // Specialization for 3D wgrad kernel
  void wgrad_reference(cute::Int<3> spatial_dims) {
    static constexpr bool IsBatchNormApplySupported = MainloopFusionParams::BatchNormMainloopFusionParams::IsBatchNormApplySupported; // {$nv-internal-release}
    int32_t G = size<5>(tensor_d_);
    int32_t N =
        IsBatchNormApplySupported ? size<4>(tensor_b_) : // {$nv-internal-release}
        size<4>(tensor_a_);
    int32_t Z =
        IsBatchNormApplySupported ? size<3>(tensor_b_) : // {$nv-internal-release}
        size<3>(tensor_a_);
    int32_t P =
        IsBatchNormApplySupported ? size<2>(tensor_b_) : // {$nv-internal-release}
        size<2>(tensor_a_);
    int32_t Q =
        IsBatchNormApplySupported ? size<1>(tensor_b_) : // {$nv-internal-release} 
        size<1>(tensor_a_);
    int32_t K =
        IsBatchNormApplySupported ? size<0>(tensor_b_) : // {$nv-internal-release}
        size<0>(tensor_a_);
    int32_t T = size<3>(tensor_d_);
    int32_t R = size<2>(tensor_d_);
    int32_t S = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t g = 0 ; g < G; ++g) {
      for (int32_t k = 0; k < K; ++k) {
        for (int32_t t = 0; t < T; ++t) {
          for (int32_t r = 0; r < R; ++r) {
            for (int32_t s = 0; s < S; ++s) {
              for (int32_t c = 0; c < C; ++c) {
                auto accumulator = ElementAcc(0);
                for (int32_t n = 0; n < N; ++n) {
                  for (int32_t z = 0; z < Z; ++z) {
                    for (int32_t p = 0; p < P; ++p) {
                      for (int32_t q = 0; q < Q; ++q) {
                        int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                        int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                        int32_t d =  z * cute::get<2>(tstride_) - cute::get<2>(padding_) + t * cute::get<2>(dilation_);
                        bool is_in_bounds =
                            IsBatchNormApplySupported ? detail::is_activation_in_bounds(tensor_a_, n, d, h, w, c, g) : // {$nv-internal-release}
                            detail::is_activation_in_bounds(tensor_b_, n, d, h, w, c, g);
                        if (is_in_bounds) {
                          auto act =
                              IsBatchNormApplySupported ? tensor_a_(c, w, h, d, n, g) : // {$nv-internal-release}
                              tensor_b_(c, w, h, d, n, g);
                          auto xformed_act =
                              IsBatchNormApplySupported ? tensor_b_(k, q, p, z, n, g) : // {$nv-internal-release}
                              tensor_a_(k, q, p, z, n, g);
                          // {$nv-internal-release begin}
                          act = main_fusion_params_.bn_fusion_params.apply(cute::make_coord(c, w, h, d, n, g), act);
                          // {$nv-internal-release end}
                          accumulator += ElementAcc(act * xformed_act);
                        }
                      }
                    }
                  }
                }

                ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
                  epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
                ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
                  epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;

                ElementCompute output = scale_converter(alpha) * acc_converter(accumulator);
                if (not EpilogueFusionParams::ResidualAdd) {
                  output += scale_converter(beta) * residual_converter(tensor_c_(c, s, r, t, k, g));
                }
                if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                  output += bias_converter(epi_fusion_params_.tensor_bias[c]);
                }
                output = epi_activation(output);
                if (EpilogueFusionParams::ResidualAdd) {
                  output += scale_converter(beta) * residual_converter(tensor_c_(c, s, r, t, k, g));
                }
                tensor_d_(c, s, r, t, k, g) = output_converter(output);
              }
            }
          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
