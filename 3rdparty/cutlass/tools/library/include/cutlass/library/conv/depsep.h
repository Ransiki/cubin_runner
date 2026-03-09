#pragma once

#include <cstdio>

#include <iostream>
#include <cinttypes>
#include "cutlass/cutlass.h"

namespace cutlass {
namespace library {
namespace conv {

struct LaunchParams {
    void *input   = nullptr;
    void *weights = nullptr;
    void *output  = nullptr;
    void *bias    = nullptr;

    size_t n = 1, h = 16, w = 16, p = 14, q = 14;

    size_t r = 3, s = 3, k = 1, c = 1, g = 1;

    size_t u = 1;
    size_t v = 1;
    size_t padding_h = 0;
    size_t padding_w = 0;

    bool bias_enabled = false;

    size_t stride_w;
    size_t stride_h;
    size_t stride_c;
    size_t stride_gc;
    size_t stride_n_in;

    size_t stride_p;
    size_t stride_q;
    size_t stride_k;
    size_t stride_gk;
    size_t stride_n_out;

    void print() {
        printf("> Input Tensor NGCHW: [%zd %zd %zd %zd %zd]\n", n, g, c, h, w);
        printf("> Output Tensor NKPQ: [%zd %zd %zd %zd]\n", n, k, p, q);
        printf("> Kernel KCRS x UV: [%zd %zd %zd %zd x %zd %zd]\n", k, c, r, s, u, v);
        printf("> Stride N(in)  = %zd\n", stride_n_in);
        printf("> Stride N(out) = %zd\n", stride_n_out);
    }

    void calculate() {
        p = (h + 2 * padding_h - r) / u + 1;
        q = (w + 2 * padding_w - s) / v + 1;

        stride_w = 1;
        stride_h = w * stride_w;
        stride_c = h * stride_h;
        stride_gc = c * stride_c;
        stride_n_in = c * stride_c;

        stride_q = 1;
        stride_p = q * stride_q;
        stride_k = p * stride_p;
        stride_gk = k * stride_k;
        stride_n_out = k * stride_k;
    }
};

void depthwise_conv(const LaunchParams &params);
void depthwise_conv_fprop(const LaunchParams &params);
void depthwise_conv_wgrad(const LaunchParams &params);

}   // End namespace conv
}   // End namespace library
}   // End namespace cutlass
