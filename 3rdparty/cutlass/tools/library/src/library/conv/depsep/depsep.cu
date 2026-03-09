#include <cstdio>
#include <cinttypes>

#include "reducedMath.h"
using namespace cutlassRM;

#include "kernelUtils.h"
#include "cutlass/library/conv/depsep.h"

namespace cutlass {
namespace library {
namespace conv {

size_t divup(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

// Limitations:
// 1. Only support two CTA tile size: 20x20 and 10x10
// 2. TILE_Q_THREAD_ or stride should be a multiple of 2 to use large LDS/STS (.32 or .64)
template<typename T,
         int R_ = 3,
         int S_ = 3,
         int U_ = 1,
         int V_ = 1,
         /// Thread tile size
         int TILE_P_THREAD_ = 7,
         int TILE_Q_THREAD_ = 2,
         /// CTA tile size
         int TILE_P_CTA_ = 20,
         int TILE_Q_CTA_ = 20,
         int TILE_C_CTA_ = 8 >
__global__ __launch_bounds__(256)
void depthwise_conv_fprop_nchw_kernel(const LaunchParams params, rt::reduced_divisor tiles_p_)
{
    /// Params
    const T* __restrict srcPtr = reinterpret_cast<T*>(params.input);
    T* __restrict dstPtr = reinterpret_cast<T*>(params.output);
    const T* __restrict weightPtr = reinterpret_cast<T*>(params.weights);
    const T* __restrict biasPtr = reinterpret_cast<T*>(params.bias);
    const int h_ = params.h;
    const int w_ = params.w;
    const int p_ = params.p;
    const int q_ = params.q;
    const int ph_ = params.padding_h;
    const int pw_ = params.padding_w;
    const int lda_ = params.stride_c;
    const int ldc_ = params.stride_k;
    const int srcNStride = params.stride_n_in;
    const int dstNStride = params.stride_n_out;
    const bool biasEnabled = params.bias_enabled;

    // Threads per CTA
    const int THREADS_PER_CTA = TILE_C_CTA_ * 32;

    // Determine the input tile size
    const int TILE_W_CTA = (TILE_Q_CTA_ - 1) * V_ + S_;
    const int TILE_H_CTA = (TILE_P_CTA_ - 1) * U_ + R_;

    // Determine the shapre of each warp during convolution
    const int THREADS_Q_PER_WARP = (TILE_Q_CTA_ + TILE_Q_THREAD_ - 1) / TILE_Q_THREAD_;
    const int THREADS_P_PER_WARP = (TILE_P_CTA_ + TILE_P_THREAD_ - 1) / TILE_P_THREAD_;

    // Number LDGs/thread/C
    // Simplify the LDG pattern by just adding an offset in H dimension (waste some threads)
    const int IMG_HEIGHT_PER_CTA = THREADS_PER_CTA / TILE_W_CTA;
    const int IMG_LDGS_H = (TILE_H_CTA + IMG_HEIGHT_PER_CTA - 1) / IMG_HEIGHT_PER_CTA;
    const int IMG_STEP_H = (TILE_H_CTA + IMG_LDGS_H - 1) / IMG_LDGS_H;
    // Number LDGs/thread in C
    const int IMG_LDGS_C = TILE_C_CTA_;

    // Smem size
    // Height is larger to avoid boundary check in H dimension
    const int TILE_P_CTA_EXTENDED = TILE_P_THREAD_ * THREADS_P_PER_WARP;
    const int TILE_H_CTA_EXTENDED = (TILE_P_CTA_EXTENDED - 1) * U_ + R_;

    // Make sure SMem width is aligned to 4 to enable possible LDS.64
    const int SMEM_W_IN = (TILE_W_CTA + 3) / 4 * 4;
    __shared__ __half sIn[TILE_C_CTA_][TILE_H_CTA_EXTENDED][SMEM_W_IN];

    // Swzzile results in epilog before STG
    __shared__ __half sOut[TILE_C_CTA_][TILE_P_CTA_EXTENDED][TILE_Q_CTA_];

    // Initialize the load registers to 0
    __half img_fetch[IMG_LDGS_H][IMG_LDGS_C];
    for (int hi = 0; hi < IMG_LDGS_H; ++hi) {
        for (int ci = 0; ci < IMG_LDGS_C; ++ci) {
            img_fetch[hi][ci] = 0;
        }
    }

    // Get coords to load inputs
    const int tile_q_idx = blockIdx.x;
    const int n = blockIdx.z;
    // Get c and tile_p_idx since they were merged in one dimension (y)
    int tile_p_idx = blockIdx.y;
    int c_start = 0;
    tiles_p_.divmod(tile_p_idx, c_start, tile_p_idx);
    c_start *= TILE_C_CTA_;
    
    const int h_start = tile_p_idx * TILE_P_CTA_ * U_ - ph_;
    const int w_start = tile_q_idx * TILE_Q_CTA_ * V_ - pw_;

    // Determine the CTA's edge in H dimension 
    unsigned h_end = (unsigned)(h_start + TILE_H_CTA) > (unsigned)h_ ? (unsigned)h_ :
        (unsigned)(h_start + TILE_H_CTA);

    const __half* srcStart = srcPtr + n * srcNStride + c_start * lda_;

    const int linearIdx = threadIdx.y * 32 + threadIdx.x;
    const int hi = linearIdx / TILE_W_CTA;
    const int wi = linearIdx % TILE_W_CTA;
    if (linearIdx < IMG_STEP_H * TILE_W_CTA) {
#pragma unroll
        for (int iter = 0; iter < IMG_LDGS_H; ++iter) {
#pragma unroll
            for (int ci = 0; ci < IMG_LDGS_C; ++ci) {
                const int h_current = h_start + hi + iter * IMG_STEP_H;
                const int w_current = w_start + wi;
                if ((unsigned)h_current < h_end && (unsigned)w_current < (unsigned)w_) {
                    int img_offset = ci * lda_ + h_current * w_ + w_current;
                    img_fetch[iter][ci] = srcStart[img_offset];
                }
            }
        }
    }

    // Load weights
    __half weight[R_][S_];
    const __half* weightStart = weightPtr + (c_start + threadIdx.y) * R_ * S_;
    for (int r = 0; r < R_; r++) {
        for (int s = 0; s < S_; s++) {
            weight[r][s] = weightStart[r * S_ + s];
        }
    }

    // Preload bias
    __half bias = 0;
    if (biasEnabled) {
        bias = biasPtr[c_start + threadIdx.y];
    }

    // Write inputs to Smem
    if (linearIdx < IMG_STEP_H * TILE_W_CTA) {
#pragma unroll
        for (int iter = 0; iter < IMG_LDGS_H; ++iter) {
#pragma unroll
            for (int ci = 0; ci < IMG_LDGS_C; ++ci) {
                const int h_current = hi + iter * IMG_STEP_H;
                if (h_current < TILE_H_CTA) { // Boundary check in H dimension
                    sIn[ci][h_current][wi] = img_fetch[iter][ci];
                }
            }
        }
    }

    __syncthreads();

    // Simply disable the threads that won't work per warp
    const int ACTIVE_THREADS_PER_WARP = THREADS_Q_PER_WARP * THREADS_P_PER_WARP;
    if (threadIdx.x >= ACTIVE_THREADS_PER_WARP) {
        return;
    }

    // Change the shape of each warp
    const int out_x = threadIdx.x % THREADS_Q_PER_WARP;
    const int out_y = threadIdx.x / THREADS_Q_PER_WARP;
    const int thread_offset_x = out_x * TILE_Q_THREAD_;
    const int thread_offset_y = out_y * TILE_P_THREAD_;

    // How many registers required per row
    const int TILE_W_THREAD = (TILE_Q_THREAD_ - 1) * V_ + S_;
    const int TILE_H_THREAD = (TILE_P_THREAD_ - 1) * U_ + R_;
    const int REGS_PER_ROW = (TILE_W_THREAD + 1) / 2;

    // Pre load to registers to save SMem operations
    __half2 in[REGS_PER_ROW];
    float in_s[TILE_H_THREAD][TILE_W_THREAD];
#pragma unroll
    for (int hi = 0; hi < TILE_H_THREAD; ++hi) {
        // Use large LDS
#pragma unroll
        for (int reg = 0; reg < REGS_PER_ROW; ++reg) {
            in[reg] = reinterpret_cast<__half2*>(&sIn[threadIdx.y][thread_offset_y * U_ + hi]\
                [thread_offset_x * V_ + reg * 2])[0];
        }

        // Convert to fp32
#pragma unroll
        for (int wi = 0; wi < TILE_W_THREAD; ++wi) {
            in_s[hi][wi] = wi & 0x1 ? __high2float(in[wi / 2]) : __low2float(in[wi / 2]);
        }
    }

    // Fully unroll 
#pragma unroll
    for (int pi = 0; pi < TILE_P_THREAD_; ++pi) {
        // ACCs per row
        float conv[TILE_Q_THREAD_] = { 0.0f };
#pragma unroll
        for (int r = 0; r < R_; r++) {
#pragma unroll
            for (int s = 0; s < S_; s++) {
#pragma unroll
                for (int qi = 0; qi < TILE_Q_THREAD_; ++qi) {
                    conv[qi] += in_s[pi * U_ + r][qi * V_ + s] * __half2float(weight[r][s]);
/*                    if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0 && pi==0 && qi==0) {
		        printf("%f,%f,%f\n", in_s[pi * U_ + r][qi * V_ + s],__half2float(weight[r][s]),conv[qi]);
		    }
*/
		}
            }
        }

        // Swizzle in SMem before STG. Convert Accs to fp16 first
        __half convH[TILE_Q_THREAD_];
        for (int qi = 0; qi < TILE_Q_THREAD_; ++qi) {
            convH[qi] = __float2half(conv[qi] + __half2float(bias));
        }

        if (TILE_Q_THREAD_ & 0x1) { // Use STS.16
#pragma unroll
            for (int qi = 0; qi < TILE_Q_THREAD_; ++qi) {
                sOut[threadIdx.y][thread_offset_y + pi][thread_offset_x] = convH[qi];
            }
        }
        else {    // Use large STS
#pragma unroll
            for (int qi = 0; qi < TILE_Q_THREAD_ / 2; ++qi) {
                reinterpret_cast<int*>(&sOut[threadIdx.y][thread_offset_y + pi]\
                    [thread_offset_x + qi * 2])[0] = reinterpret_cast<int*>(&convH[qi * 2])[0];
            }
        }
    }

    __syncthreads();

    // Epilog
    // Determine the CTA's edge in P/Q dimension in epilog
    const int pEnd = (p_ - tile_p_idx * TILE_P_CTA_) > TILE_P_CTA_ ? TILE_P_CTA_ :
        (p_ - tile_p_idx * TILE_P_CTA_);
    const int qEnd = (q_ - tile_q_idx * TILE_Q_CTA_) > TILE_Q_CTA_ ? TILE_Q_CTA_ :
        (q_ - tile_q_idx * TILE_Q_CTA_);

    __half* dstStart = dstPtr + n * dstNStride + c_start * ldc_ + tile_p_idx * TILE_P_CTA_ * q_
        + tile_q_idx * TILE_Q_CTA_;

    // Use the whole CTA to store one C
    // Simplify the STG pattern by just adding an offset in P dimension (waste some threads)
    const int ACTIVE_THREADS_PER_CTA = ACTIVE_THREADS_PER_WARP * TILE_C_CTA_;
    const int OUT_HEIGHT_PER_CTA = ACTIVE_THREADS_PER_CTA / TILE_Q_CTA_;
    const int OUT_STGS_H = (TILE_P_CTA_ + OUT_HEIGHT_PER_CTA - 1) / OUT_HEIGHT_PER_CTA;
    const int OUT_STEP_H = (TILE_P_CTA_ + OUT_STGS_H - 1) / OUT_STGS_H;

    const int linearIdxOut = threadIdx.y * ACTIVE_THREADS_PER_WARP + threadIdx.x;
    const int qi = linearIdxOut % TILE_Q_CTA_;
    const int pi = linearIdxOut / TILE_Q_CTA_;
    if (linearIdxOut < OUT_STEP_H * TILE_Q_CTA_) {
#pragma unroll
        for (int iter = 0; iter < OUT_STGS_H; ++iter) {
            const int p_current = pi + iter * OUT_STEP_H;
            if (qi < qEnd && p_current < pEnd) {
#pragma unroll
                for (int ci = 0; ci < IMG_LDGS_C; ++ci) {
                    dstStart[ci * ldc_ + p_current * q_ + qi] = sOut[ci][p_current][qi];
                }
            }
        }
    }
}

void depthwise_conv_fprop(const LaunchParams &params)
{
    if (params.g % 8 != 0) {
        printf("Input channels should be a multiple of 8!\n");
        return;
    }

    typedef void(*pf)(const LaunchParams params, rt::reduced_divisor tiles_p_);
    const pf kernels[16] = {
        depthwise_conv_fprop_nchw_kernel<__half, 3, 3, 1, 1, 7, 2, 20, 20, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 3, 3, 1, 1, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 3, 3, 2, 2, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 3, 3, 1, 1, 4, 4, 10, 40, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 1, 1, 1, 1, 7, 2, 20, 20, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 1, 1, 1, 1, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 1, 1, 2, 2, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 1, 1, 1, 1, 4, 4, 10, 40, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 5, 5, 1, 1, 7, 2, 20, 20, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 5, 5, 1, 1, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 5, 5, 2, 2, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 5, 5, 1, 1, 4, 4, 10, 40, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 7, 7, 1, 1, 7, 2, 20, 20, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 7, 7, 1, 1, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 7, 7, 2, 2, 2, 2, 10, 10, 8>,
        depthwise_conv_fprop_nchw_kernel<__half, 7, 7, 1, 1, 4, 4, 10, 40, 8>
    };

    size_t tile_p = 20;
    size_t tile_q = 20;
    size_t tile_c = 8;
    int kernel_idx = 0;

    if (params.u == 2 && params.v == 2) {
        tile_p = 10;
        tile_q = 10;
        kernel_idx = 2;
    }
    else if (params.u == 1 && params.v == 1) {
        if (params.q <= 10) {
            tile_p = 10;
            tile_q = 10;
            kernel_idx = 1;
        }
        else if (params.q > 20) {
            tile_p = 10;
            tile_q = 40;
            kernel_idx = 3;
        }
    }
    else {
        printf("Stride not supported!\n");
        return;
    }
	
    if (params.r == 1 && params.s == 1) {
        kernel_idx += 4;
    } else if (params.r == 5 && params.s == 5) {
        kernel_idx += 8;
    } else if (params.r == 7 && params.s == 7) {
        kernel_idx += 12;
    }

    dim3 grid((unsigned int)(divup(params.q, tile_q)), (unsigned int)(divup(params.p, tile_p) * divup(params.g, tile_c)), (unsigned int)(params.n));
    dim3 block(32, (unsigned int)(tile_c), 1);
    kernels[kernel_idx] <<<grid, block>>>(params, rt::reduced_divisor((unsigned int)(divup(params.p, tile_p))));

    return;
}

}   // End namespace conv
}   // End namespace library
}   // End namespace cutlass
