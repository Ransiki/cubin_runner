#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<cstdint>
#include<iostream>
#include<cudnn.h>

#define cudaFunc(func) \
    if((func) != cudaSuccess){\
        std::cout << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }

#define cudnnFunc(func) \
    if((func) != CUDNN_STATUS_SUCCESS){\
        std::cout << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }


template<typename ElementType>
struct KernelParams{
    ElementType *gmem_image;
    ElementType *gmem_error;
    ElementType *gmem_filter;

    int N;
    int C;
    int D;
    int H;
    int W;

    int K;
    int T;
    int R;
    int S;

    int paddingFront;
    int paddingBack;
    int paddingTop;
    int paddingBottom;
    int paddingLeft;
    int paddingRight;

    int strideD;
    int strideH;
    int strideW;

    int dilationD;
    int dilationH;
    int dilationW;

    int O;
    int P;
    int Q;

    float alpha;
    float beta;

    bool isCrossCorrelation;

    int CTA_PER_O;
    int CTA_PER_P;
    int CTA_PER_Q;

    std::uint64_t imageStrideW; 
    std::uint64_t imageStrideH;
    std::uint64_t imageStrideD;
    std::uint64_t imageStrideN;

    std::uint64_t filterStrideS;
    std::uint64_t filterStrideR;
    std::uint64_t filterStrideT;
    std::uint64_t filterStrideK;

    std::uint64_t errorStrideQ; 
    std::uint64_t errorStrideP;
    std::uint64_t errorStrideO;
    std::uint64_t errorStrideN;

};

int GetOutputSize(int H, int paddingTop, int paddingBottom, int strideH, int R){
    return (H + paddingTop + paddingBottom - R)/strideH + 1;
}

template<typename ElementType>
void Dgrad_3d_ndhwc_cpu(KernelParams<ElementType> params){

    const int N = params.N;
    const int C = params.C;
    const int D = params.D;
    const int H = params.H;
    const int W = params.W;
    const int K = params.K;
    const int T = params.T;
    const int R = params.R;
    const int S = params.S;
    const int O = params.O;
    const int P = params.P;
    const int Q = params.Q;
    const int paddingFront = params.paddingFront;
    const int paddingTop = params.paddingTop;
    const int paddingLeft = params.paddingLeft;
    const int strideD = params.strideD;
    const int strideH = params.strideH;
    const int strideW = params.strideW;
    const int dilationD = params.dilationD;
    const int dilationH = params.dilationH;
    const int dilationW = params.dilationW;
    float alpha = params.alpha;
    float beta = params.beta;
    const bool isCrossCorrelation = params.isCrossCorrelation;

    constexpr std::uint64_t imageStrideC = 1;
    std::uint64_t imageStrideW = imageStrideC * C;
    std::uint64_t imageStrideH = imageStrideW * W;
    std::uint64_t imageStrideD = imageStrideH * H;
    std::uint64_t imageStrideN = imageStrideD * D;

    constexpr std::uint64_t filterStrideC = 1;
    std::uint64_t filterStrideS = filterStrideC * C;
    std::uint64_t filterStrideR = filterStrideS * S;
    std::uint64_t filterStrideT = filterStrideR * R;
    std::uint64_t filterStrideK = filterStrideT * T;

    constexpr std::uint64_t errorStrideK = 1;
    std::uint64_t errorStrideQ = errorStrideK * K;
    std::uint64_t errorStrideP = errorStrideQ * Q;
    std::uint64_t errorStrideO = errorStrideP * P;
    std::uint64_t errorStrideN = errorStrideO * O;

    for(int i = 0;i< N*C*D*H*W;++i){
        params.gmem_image[i] = params.gmem_image[i]*beta;
    }

    int d_base,h_base,w_base;
    int d,h,w;
    int real_t, real_r, real_s;
    std::uint64_t error_index, filter_index, image_index;
    for(int n=0;n<N;++n){
        for(int c=0;c<C;++c){
            for(int k=0;k<K;++k){
                for(int o=0;o<O;++o){
                    d_base = o*strideD - paddingFront;
                    for(int p=0;p<P;++p){
                        h_base = p*strideH - paddingTop;
                        for(int q=0;q<Q;++q){
                            w_base = q*strideW - paddingLeft;
                            error_index = n * errorStrideN 
                                + k * errorStrideK 
                                + o * errorStrideO 
                                + p * errorStrideP 
                                + q * errorStrideQ;
                            for(int t=0;t<T;++t){
                                real_t = (isCrossCorrelation?t:(T - 1 - t));
                                d = d_base + real_t * dilationD;
                                for(int r=0;r<R;++r){
                                    real_r = (isCrossCorrelation?r:(R - 1 - r));
                                    h = h_base + real_r * dilationH;
                                    for(int s=0;s<S;++s){
                                        filter_index = k * filterStrideK 
                                            + c * filterStrideC 
                                            + t * filterStrideT 
                                            + r * filterStrideR 
                                            + s * filterStrideS;
                                        real_s = (isCrossCorrelation?s:(S - 1 - s));
                                        w = w_base + real_s * dilationW;
                                        if((unsigned)d<D && (unsigned)h<H && (unsigned)w<W){
                                            image_index = n * imageStrideN
                                                + c * imageStrideC
                                                + d * imageStrideD
                                                + h * imageStrideH
                                                + w * imageStrideW;
                                            params.gmem_image[image_index] = params.gmem_image[image_index]
                                                +params.gmem_error[error_index]*params.gmem_filter[filter_index]*alpha;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

template<typename ElementType, int M, int N, int K>
__device__ inline void Clear(ElementType (&in)[M][N][K]){
#pragma unroll
    for(int m=0;m<M;++m){
#pragma unroll
        for(int n=0;n<N;++n){
#pragma unroll
            for(int k=0;k<K;++k){
                in[m][n][k] = 0;
            }
        }
    }
}

template<typename ElementType, int M, int N, int K>
__device__ inline void Copy(ElementType (&out)[M][N][K], ElementType (&in)[M][N][K]){
#pragma unroll
    for(int m=0;m<M;++m){
#pragma unroll
        for(int n=0;n<N;++n){
#pragma unroll
            for(int k=0;k<K;++k){
                out[m][n][k] = in[m][n][k];
            }
        }
    }
}

#define FLOAT
constexpr bool debug = false;
// Grid(ceil(Q/Q_PER_CTA)*ceil(P/P_PER_CTA)*ceil(O/O_PER_CTA), ceil(C/C_PER_CTA), N)
// Block(C_PER_CTA, 8, 1)
constexpr std::uint32_t FACTOR = 4;
static_assert(FACTOR%2==0,"FACTOR%2==0");

constexpr std::uint32_t C_PER_CTA=32;
#ifdef FLOAT
constexpr std::uint32_t SMEM_FILTER_C_FACTOR = 1;
#else
constexpr std::uint32_t SMEM_FILTER_C_FACTOR = 2;
#endif
constexpr std::uint32_t O_PER_CTA = FACTOR;
constexpr std::uint32_t P_PER_CTA = FACTOR;
constexpr std::uint32_t Q_PER_CTA = FACTOR;

constexpr std::uint32_t WARPS_PER_O_PER_CTA = 2;
constexpr std::uint32_t WARPS_PER_P_PER_CTA = 2;
constexpr std::uint32_t WARPS_PER_Q_PER_CTA = 2;

constexpr std::uint32_t K_PER_CTA_PER_LOOP=8;
static_assert(K_PER_CTA_PER_LOOP == 8, "K_PER_CTA_PER_LOOP == 8");

constexpr std::uint32_t T_PER_CTA=2;
constexpr std::uint32_t R_PER_CTA=2;
constexpr std::uint32_t S_PER_CTA=2;

constexpr std::uint32_t D_PER_CTA = O_PER_CTA*T_PER_CTA;
constexpr std::uint32_t H_PER_CTA = P_PER_CTA*R_PER_CTA;
constexpr std::uint32_t W_PER_CTA = Q_PER_CTA*S_PER_CTA;

constexpr std::uint32_t WARPS_PER_D_PER_CTA = WARPS_PER_O_PER_CTA;
constexpr std::uint32_t WARPS_PER_H_PER_CTA = WARPS_PER_P_PER_CTA;
constexpr std::uint32_t WARPS_PER_W_PER_CTA = WARPS_PER_Q_PER_CTA;

constexpr std::uint32_t O_PER_WARP = O_PER_CTA/WARPS_PER_O_PER_CTA;
constexpr std::uint32_t P_PER_WARP = P_PER_CTA/WARPS_PER_P_PER_CTA;
constexpr std::uint32_t Q_PER_WARP = Q_PER_CTA/WARPS_PER_Q_PER_CTA;
constexpr std::uint32_t D_PER_WARP = D_PER_CTA/WARPS_PER_D_PER_CTA;
constexpr std::uint32_t H_PER_WARP = H_PER_CTA/WARPS_PER_H_PER_CTA;
constexpr std::uint32_t W_PER_WARP = W_PER_CTA/WARPS_PER_W_PER_CTA;

constexpr std::uint32_t O_PER_THREAD = O_PER_WARP;
constexpr std::uint32_t P_PER_THREAD = P_PER_WARP;
constexpr std::uint32_t Q_PER_THREAD = Q_PER_WARP;
constexpr std::uint32_t D_PER_THREAD = D_PER_WARP;
constexpr std::uint32_t H_PER_THREAD = H_PER_WARP;
constexpr std::uint32_t W_PER_THREAD = W_PER_WARP;

constexpr std::uint32_t OPQ_PER_THREAD = O_PER_WARP*P_PER_WARP*Q_PER_WARP;
constexpr std::uint32_t TRS_PER_THREAD = T_PER_CTA * R_PER_CTA * S_PER_CTA;
constexpr std::uint32_t DHW_PER_THRAD = D_PER_THREAD*H_PER_THREAD*W_PER_THREAD;

constexpr std::uint32_t OFFSET_NUMBER = OPQ_PER_THREAD + TRS_PER_THREAD + DHW_PER_THRAD;

__constant__ std::uint64_t OFFSET_LUT[OFFSET_NUMBER];
template<typename ElementType, bool IS_CROSS_CORRELATION, int T, int R, int S, int STRIDE_D, int STRIDE_H, int STRIDE_W, int DILATION_D, int DILATION_H, int DILATION_W>
__global__ void Kernel_dgrad_3d_ndhwc(KernelParams<ElementType> params){

    static_assert(T==2 && R==2 && S==2,"T==2 && R==2 && S==2");
    static_assert(T==STRIDE_D && R==STRIDE_H && S==STRIDE_W,"T==STRIDE_D && R==STRIDE_H && S==STRIDE_W");
    static_assert(DILATION_D*DILATION_H*DILATION_W==1,"DILATION_D*DILATION_H*DILATION_W==1");

    const int C = params.C;
    const int D = params.D;
    const int H = params.H;
    const int W = params.W;
    const int K = params.K;
    const int O = params.O;
    const int P = params.P;
    const int Q = params.Q;
    const int paddingFront = params.paddingFront;
    const int paddingTop = params.paddingTop;
    const int paddingLeft = params.paddingLeft;
    if(debug && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
        printf("O = %d P = %d Q = %d\n",O,P,Q);
    }

    constexpr std::uint64_t imageStrideC = 1;
    const std::uint64_t imageStrideW = params.imageStrideW;
    const std::uint64_t imageStrideH = params.imageStrideH;
    const std::uint64_t imageStrideD = params.imageStrideD;
    const std::uint64_t imageStrideN = params.imageStrideN;

    constexpr std::uint64_t filterStrideC = 1;
    const std::uint64_t filterStrideK = params.filterStrideK;

    constexpr std::uint64_t errorStrideK = 1;
    const std::uint64_t errorStrideQ = params.errorStrideQ;
    const std::uint64_t errorStrideP = params.errorStrideP;
    const std::uint64_t errorStrideO = params.errorStrideO;
    const std::uint64_t errorStrideN = params.errorStrideN;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    int cta_c_begin = bidx * C_PER_CTA;
    int thread_c_index = cta_c_begin + tidx;
    int cta_n_index = bidz;
    const int thread_n_index = cta_n_index;
    int main_loop_k_begin = 0;
    std::uint32_t lane_id = tidx % K_PER_CTA_PER_LOOP;
    int thread_error_k_index = main_loop_k_begin + lane_id;

    int cta_q_begin = bidy % params.CTA_PER_Q * Q_PER_CTA;
    int cta_p_begin = bidy / params.CTA_PER_Q % params.CTA_PER_P * P_PER_CTA;
    int cta_o_begin = bidy / (params.CTA_PER_Q*params.CTA_PER_P) * O_PER_CTA;

    int smem_q_begin = tidy % WARPS_PER_Q_PER_CTA * Q_PER_WARP;
    int warp_q_begin = cta_q_begin + smem_q_begin;
    int smem_p_begin = tidy / WARPS_PER_Q_PER_CTA % WARPS_PER_P_PER_CTA * P_PER_WARP;
    int warp_p_begin = cta_p_begin + smem_p_begin;
    int smem_o_begin = tidy / (WARPS_PER_Q_PER_CTA*WARPS_PER_P_PER_CTA) * O_PER_WARP;
    int warp_o_begin = cta_o_begin + smem_o_begin;

    const int thread_q_begin = warp_q_begin;
    const int thread_p_begin = warp_p_begin;
    const int thread_o_begin = warp_o_begin;

    const std::uint64_t thread_k_error_ptr_delta = K_PER_CTA_PER_LOOP * errorStrideK;
    const std::uint64_t thread_k_filter_ptr_delta = K_PER_CTA_PER_LOOP * filterStrideK;

    //Prolog
    std::uint32_t k_boundary = min(K, main_loop_k_begin + K_PER_CTA_PER_LOOP);
    bool thread_error_k_valid = (thread_error_k_index < k_boundary);
    //LDG error
    ElementType *error_ptr = params.gmem_error + (thread_n_index * errorStrideN
            + thread_o_begin * errorStrideO
            + thread_p_begin * errorStrideP
            + thread_q_begin * errorStrideQ
            + thread_error_k_index * errorStrideK);
    ElementType error_reg[O_PER_THREAD][P_PER_THREAD][Q_PER_THREAD];
#pragma unroll
    for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
        for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
            for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
                if(thread_error_k_valid && thread_o_begin+o_index<O
                        && thread_p_begin+p_index<P
                        && thread_q_begin+q_index<Q){
                    error_reg[o_index][p_index][q_index] = __ldg((ElementType*)((std::uint64_t)error_ptr+OFFSET_LUT[q_index
                                +p_index*Q_PER_THREAD
                                +o_index*Q_PER_THREAD*P_PER_THREAD]));

                }else{
                    error_reg[o_index][p_index][q_index]  = 0.f;
                }
            }
        }
    }
    if(debug && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
        printf("Prolog ldg error finish\n");
    }
    //LDG filter
    bool c_valid = (thread_c_index<C);
    bool filter_ck_valid = (c_valid && main_loop_k_begin+tidy<k_boundary);
    ElementType *filter_ptr = params.gmem_filter +((main_loop_k_begin + tidy) * filterStrideK
            +thread_c_index * filterStrideC);
    ElementType filter_reg[T][R][S];
#pragma unroll
    for(int t_index=0;t_index<T;++t_index){
#pragma unroll
        for(int r_index=0;r_index<R;++r_index){
#pragma unroll
            for(int s_index=0;s_index<S;++s_index){
                if(filter_ck_valid){
                    filter_reg[t_index][r_index][s_index] = __ldg((ElementType*)((std::uint64_t)filter_ptr
                                +OFFSET_LUT[OPQ_PER_THREAD+s_index
                                +r_index*S
                                +t_index*S*R]));
                }
                else{
                    filter_reg[t_index][r_index][s_index] = 0.f;
                }
            }
        }
    }
    if(debug && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
        printf("Prolog ldg filter finish\n");
    }
    //Update ptr
    main_loop_k_begin += K_PER_CTA_PER_LOOP;
    thread_error_k_index += K_PER_CTA_PER_LOOP;
    k_boundary = min(K, main_loop_k_begin + K_PER_CTA_PER_LOOP);
    thread_error_k_valid = (thread_error_k_index < k_boundary);
    filter_ck_valid = (c_valid && main_loop_k_begin+tidy<k_boundary);
    error_ptr+=thread_k_error_ptr_delta;
    filter_ptr+=thread_k_filter_ptr_delta;
    //STS
    constexpr std::uint32_t SMEM_ERROR_K = C_PER_CTA;
    __shared__ ElementType smem_error[O_PER_CTA*P_PER_CTA*Q_PER_CTA*SMEM_ERROR_K];
    ElementType *smem_error_sts_ptr = smem_error + (tidx
            + smem_q_begin*SMEM_ERROR_K
            + smem_p_begin*SMEM_ERROR_K*Q_PER_CTA
            + smem_o_begin*SMEM_ERROR_K*Q_PER_CTA*P_PER_CTA);
    ElementType *smem_error_lds_ptr_base = smem_error_sts_ptr;
    ElementType *smem_error_lds_ptr = smem_error_lds_ptr_base;
#pragma unroll
    for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
        for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
            for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
                smem_error_sts_ptr[q_index*SMEM_ERROR_K
                    +p_index*SMEM_ERROR_K*Q_PER_CTA
                    +o_index*SMEM_ERROR_K*Q_PER_CTA*P_PER_CTA] = error_reg[o_index][p_index][q_index];
            }
        }
    }
    __shared__ ElementType smem_filter[C_PER_CTA*SMEM_FILTER_C_FACTOR*T_PER_CTA*R_PER_CTA*S_PER_CTA*K_PER_CTA_PER_LOOP];
    ElementType *smem_filter_sts_ptr = smem_filter + (tidx*SMEM_FILTER_C_FACTOR
            + tidy*C_PER_CTA * SMEM_FILTER_C_FACTOR * S_PER_CTA * R_PER_CTA * T_PER_CTA);
    ElementType *smem_filter_lds_ptr_base = smem_filter + tidx*SMEM_FILTER_C_FACTOR + lane_id*C_PER_CTA*SMEM_FILTER_C_FACTOR*T_PER_CTA*R_PER_CTA*S_PER_CTA;
    ElementType *smem_filter_lds_ptr = smem_filter_lds_ptr_base;
#pragma unroll
    for(int t_index=0;t_index<T;++t_index){
#pragma unroll
        for(int r_index=0;r_index<R;++r_index){
#pragma unroll
            for(int s_index=0;s_index<S;++s_index){
                smem_filter_sts_ptr[s_index*C_PER_CTA*SMEM_FILTER_C_FACTOR
                    +r_index*C_PER_CTA*SMEM_FILTER_C_FACTOR*S_PER_CTA
                    +t_index*C_PER_CTA*SMEM_FILTER_C_FACTOR*S_PER_CTA*R_PER_CTA] = filter_reg[t_index][r_index][s_index];
            }
        }
    }
    //Main loop
    if(debug && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
        printf("Main loop begin\n");
    }
    //Allocate math reg
    ElementType error_math[O_PER_THREAD][P_PER_THREAD][Q_PER_THREAD];
    ElementType filter_math[T][R][S];
    ElementType image_math[D_PER_THREAD][H_PER_THREAD][W_PER_THREAD];
    Clear<ElementType,D_PER_THREAD,H_PER_THREAD,W_PER_THREAD>(image_math);
    __syncthreads();
    std::uint32_t main_loop_k_boundary = ((K + K_PER_CTA_PER_LOOP - 1)/K_PER_CTA_PER_LOOP)*K_PER_CTA_PER_LOOP;
    for(;main_loop_k_begin<=main_loop_k_boundary;){
        if(debug && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
            printf("thread_error_k_index=%d\n",thread_error_k_index);
        }
        //LDG error
#pragma unroll
        for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
            for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
                for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
                    if(thread_error_k_valid && thread_o_begin+o_index<O
                            && thread_p_begin+p_index<P
                            && thread_q_begin+q_index<Q){
                        error_reg[o_index][p_index][q_index] = __ldg((ElementType*)((std::uint64_t)error_ptr+OFFSET_LUT[q_index
                                    +p_index*Q_PER_THREAD
                                    +o_index*Q_PER_THREAD*P_PER_THREAD]));
                    }else{
                        error_reg[o_index][p_index][q_index]  = 0.f;
                    }
                }
            }
        }
        //LDG filter
        for(int t_index=0;t_index<T;++t_index){
            for(int r_index=0;r_index<R;++r_index){
                for(int s_index=0;s_index<S;++s_index){
                    if(filter_ck_valid){
                        filter_reg[t_index][r_index][s_index] = __ldg((ElementType*)((std::uint64_t)filter_ptr
                                    +OFFSET_LUT[OPQ_PER_THREAD+s_index
                                    +r_index*S
                                    +t_index*S*R]));
                    }
                    else{
                        filter_reg[t_index][r_index][s_index] = 0.f;
                    }
                }
            }
        }
        for(int k_index=0;k_index<K_PER_CTA_PER_LOOP;++k_index){
            //LDS error
#pragma unroll
            for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
                for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
                    for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
                        error_math[o_index][p_index][q_index] = smem_error_lds_ptr[q_index*SMEM_ERROR_K
                            + p_index*SMEM_ERROR_K*Q_PER_CTA
                            + o_index*SMEM_ERROR_K*Q_PER_CTA*P_PER_CTA];
                    }
                }
            }
            //LDS filter
#pragma unroll
            for(int t_index = 0;t_index<T;++t_index){
#pragma unroll
                for(int r_index = 0;r_index<R;++r_index){
#pragma unroll
                    for(int s_index = 0;s_index<S;++s_index){
                        filter_math[t_index][r_index][s_index] = 
                            smem_filter_lds_ptr[s_index * C_PER_CTA * SMEM_FILTER_C_FACTOR
                            + r_index * C_PER_CTA * SMEM_FILTER_C_FACTOR * S_PER_CTA
                            + t_index * C_PER_CTA * SMEM_FILTER_C_FACTOR * S_PER_CTA * R_PER_CTA];
                    }
                }
            }
            //Math
#pragma unroll
            for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
                for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
                    for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
#pragma unroll
                        for(int t_index = 0;t_index<T;++t_index){
#pragma unroll
                            for(int r_index = 0;r_index<R;++r_index){
#pragma unroll
                                for(int s_index = 0;s_index<S;++s_index){
                                    image_math[o_index*STRIDE_D+(IS_CROSS_CORRELATION?t_index:(T-1-t_index))*DILATION_D]
                                        [p_index*STRIDE_H+(IS_CROSS_CORRELATION?r_index:(R-1-r_index))*DILATION_H]
                                        [q_index*STRIDE_W+(IS_CROSS_CORRELATION?s_index:(S-1-s_index))*DILATION_W] +=
                                            error_math[o_index][p_index][q_index]*filter_math[t_index][r_index][s_index];
                                }
                            }
                        }
                    }
                }
            }
            smem_error_lds_ptr = smem_error_lds_ptr_base + (k_index + 1);
            if(lane_id +k_index + 1>=K_PER_CTA_PER_LOOP){
                smem_error_lds_ptr -= K_PER_CTA_PER_LOOP;
            }
            smem_filter_lds_ptr = smem_filter_lds_ptr_base + (k_index + 1) * C_PER_CTA*SMEM_FILTER_C_FACTOR*S_PER_CTA*R_PER_CTA*T_PER_CTA;
            if(lane_id +k_index + 1>=K_PER_CTA_PER_LOOP){
                smem_filter_lds_ptr -= K_PER_CTA_PER_LOOP*C_PER_CTA*SMEM_FILTER_C_FACTOR*S_PER_CTA*R_PER_CTA*T_PER_CTA;
            }
        }
        __syncthreads();
        //Update ptr
        main_loop_k_begin += K_PER_CTA_PER_LOOP;
        k_boundary = min(K, main_loop_k_begin + K_PER_CTA_PER_LOOP);
        thread_error_k_index+=K_PER_CTA_PER_LOOP;
        thread_error_k_valid = (thread_error_k_index<k_boundary);
        filter_ck_valid = (c_valid && main_loop_k_begin+tidy<k_boundary);
        error_ptr+=thread_k_error_ptr_delta;
        filter_ptr+=thread_k_filter_ptr_delta;
        //STS
#pragma unroll
        for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
            for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
                for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
                    smem_error_sts_ptr[q_index*SMEM_ERROR_K
                        +p_index*SMEM_ERROR_K*Q_PER_CTA
                        +o_index*SMEM_ERROR_K*Q_PER_CTA*P_PER_CTA] = error_reg[o_index][p_index][q_index];
                }
            }
        }
#pragma unroll
    for(int t_index=0;t_index<T;++t_index){
#pragma unroll
        for(int r_index=0;r_index<R;++r_index){
#pragma unroll
            for(int s_index=0;s_index<S;++s_index){
                smem_filter_sts_ptr[s_index*C_PER_CTA*SMEM_FILTER_C_FACTOR
                    +r_index*C_PER_CTA*SMEM_FILTER_C_FACTOR*S_PER_CTA
                    +t_index*C_PER_CTA*SMEM_FILTER_C_FACTOR*S_PER_CTA*R_PER_CTA] = filter_reg[t_index][r_index][s_index];
            }
        }
    }
        __syncthreads();
    }
    //Epilog
    int thread_d_begin = thread_o_begin*STRIDE_D-paddingFront;
    int thread_h_begin = thread_p_begin*STRIDE_H-paddingTop;
    int thread_w_begin = thread_q_begin*STRIDE_W-paddingLeft;
    ElementType *image_ptr = params.gmem_image + thread_n_index*imageStrideN
        + thread_c_index*imageStrideC
        + thread_d_begin*imageStrideD
        + thread_h_begin*imageStrideH
        + thread_w_begin*imageStrideW;
#pragma unroll
    for(int d_index=0;d_index<D_PER_THREAD;++d_index){
#pragma unroll
        for(int h_index=0;h_index<H_PER_THREAD;++h_index){
#pragma unroll
            for(int w_index=0;w_index<W_PER_THREAD;++w_index){
#ifdef FLOAT
                image_math[d_index][h_index][w_index]*=params.alpha;
#else
                image_math[d_index][h_index][w_index] = __float2half(__half2float(image_math[d_index][h_index][w_index])*params.alpha);
#endif
            }
        }
    }
#pragma unroll
    for(int d_index=0;d_index<D_PER_THREAD;++d_index){
#pragma unroll
        for(int h_index=0;h_index<H_PER_THREAD;++h_index){
#pragma unroll
            for(int w_index=0;w_index<W_PER_THREAD;++w_index){
                if(c_valid && unsigned(thread_d_begin+d_index)<D
                        && unsigned(thread_h_begin+h_index)<H
                        && unsigned(thread_w_begin+w_index)<W){
                    *(ElementType*)((std::uint64_t)image_ptr+OFFSET_LUT[OPQ_PER_THREAD+K_PER_CTA_PER_LOOP+w_index+h_index*W_PER_THREAD+d_index*W_PER_THREAD*H_PER_THREAD]) = image_math[d_index][h_index][w_index];
                }
            }
        }
    }
    return;
}

int Ceil(int a,int b){
    return (a+b-1)/b;
}

template<typename ElementType>
void Fill(ElementType *in, int count){
    for(int i=0;i<count;++i){
        in[i] = static_cast<ElementType>(rand() % 11 - 5.0);
    }
    return;
}

template<typename ElementType, bool IS_CROSS_CORRELATION, int T, int R, int S, int STRIDE_D, int STRIDE_H, int STRIDE_W, int DILATION_D, int DILATION_H, int DILATION_W>
float Run_kernel_dgrad_3d_ndhwc(KernelParams<ElementType> &params, int runs){
    cudaEvent_t start,stop;
    constexpr std::uint64_t imageStrideC = 1;
    params.imageStrideW = imageStrideC * params.C;
    params.imageStrideH = params.imageStrideW * params.W;
    params.imageStrideD = params.imageStrideH * params.H;
    params.imageStrideN = params.imageStrideD * params.D;

    constexpr std::uint64_t filterStrideC = 1;
    params.filterStrideS = filterStrideC * params.C;
    params.filterStrideR = params.filterStrideS * params.S;
    params.filterStrideT = params.filterStrideR * params.R;
    params.filterStrideK = params.filterStrideT * params.T;

    constexpr std::uint64_t errorStrideK = 1;
    params.errorStrideQ = errorStrideK * params.K;
    params.errorStrideP = params.errorStrideQ * params.Q;
    params.errorStrideO = params.errorStrideP * params.P;
    params.errorStrideN = params.errorStrideO * params.O;

    params.CTA_PER_O = Ceil(params.O, O_PER_CTA);
    params.CTA_PER_P = Ceil(params.P, P_PER_CTA);
    params.CTA_PER_Q = Ceil(params.Q, Q_PER_CTA);

    std::uint64_t OFFSET_LUT_HOST[OFFSET_NUMBER];
    int offset_index(0);
#pragma unroll
    for(int o_index = 0;o_index<O_PER_THREAD;++o_index){
#pragma unroll
        for(int p_index = 0;p_index<P_PER_THREAD;++p_index){
#pragma unroll
            for(int q_index = 0;q_index<Q_PER_THREAD;++q_index){
                OFFSET_LUT_HOST[offset_index++] = sizeof(ElementType)*(o_index*params.errorStrideO
                        +  p_index*params.errorStrideP
                        +  q_index*params.errorStrideQ);
            }
        }
    }
#pragma unroll
    for(int t_index = 0;t_index<T;++t_index){
#pragma unroll
        for(int r_index = 0;r_index<R;++r_index){
#pragma unroll
            for(int s_index = 0;s_index<S;++s_index){
                OFFSET_LUT_HOST[offset_index++] = sizeof(ElementType)*(t_index*params.filterStrideT
                        +  r_index*params.filterStrideR
                        +  s_index*params.filterStrideS);
            }
        }
    }
#pragma unroll
    for(int d_index=0;d_index<D_PER_THREAD;++d_index){
#pragma unroll
        for(int h_index=0;h_index<H_PER_THREAD;++h_index){
#pragma unroll
            for(int w_index=0;w_index<W_PER_THREAD;++w_index){
                OFFSET_LUT_HOST[offset_index++] = sizeof(ElementType)*(w_index*params.imageStrideW
                        +h_index*params.imageStrideH
                        +d_index*params.imageStrideD);
            }
        }
    }
    cudaFunc(cudaMemcpyToSymbol(OFFSET_LUT,OFFSET_LUT_HOST, sizeof(std::uint64_t)*OFFSET_NUMBER));

    dim3 grid(Ceil(params.C, C_PER_CTA), params.CTA_PER_O*params.CTA_PER_P*params.CTA_PER_Q, params.N);
    dim3 block(C_PER_CTA, WARPS_PER_O_PER_CTA*WARPS_PER_P_PER_CTA*WARPS_PER_Q_PER_CTA, 1);
    if(debug){
        std::cout << "grid( " << grid.x << "," << grid.y << "," << grid.z << " )" << std::endl;
        std::cout << "block( " << block.x << "," << block.y << "," << block.z << " )" << std::endl;
        std::cout << "cta_per_o = " << params.CTA_PER_O << std::endl;
        std::cout << "cta_per_p = " << params.CTA_PER_P << std::endl;
        std::cout << "cta_per_q = " << params.CTA_PER_Q << std::endl;
    }
    cudaFunc(cudaFuncSetCacheConfig((void *)Kernel_dgrad_3d_ndhwc<ElementType,IS_CROSS_CORRELATION,T,R,S,STRIDE_D,STRIDE_H,STRIDE_W,DILATION_D,DILATION_H,DILATION_W>, cudaFuncCachePreferEqual));
    cudaFunc(cudaEventCreate(&start));
    cudaFunc(cudaEventCreate(&stop));
    cudaFunc(cudaEventRecord(start));
    for(int i=0;i<runs;++i){
        Kernel_dgrad_3d_ndhwc<ElementType,IS_CROSS_CORRELATION,T,R,S,STRIDE_D,STRIDE_H,STRIDE_W,DILATION_D,DILATION_H,DILATION_W><<<grid,block>>>(params);
    }
    cudaFunc(cudaEventRecord(stop));
    cudaFunc(cudaEventSynchronize(stop));

    float timeUsed;
    cudaFunc(cudaEventElapsedTime(&timeUsed,start,stop));
    std::cout << (params.isCrossCorrelation?"cross-correlation":"convolution") << ',';
    std::cout << params.N << ',';
    std::cout << params.C << ',';
    std::cout << params.D << ',';
    std::cout << params.H << ',';
    std::cout << params.W << ',';
    std::cout << params.K << ',';
    std::cout << params.T << ',';
    std::cout << params.R << ',';
    std::cout << params.S << ',';
    std::cout << params.paddingFront << ',';
    std::cout << params.paddingBack << ',';
    std::cout << params.paddingTop << ',';
    std::cout << params.paddingBottom << ',';
    std::cout << params.paddingLeft << ',';
    std::cout << params.paddingRight << ',';
    std::cout << params.strideD << ',';
    std::cout << params.strideH << ',';
    std::cout << params.strideW << ',';
    std::cout << params.dilationD << ',';
    std::cout << params.dilationH << ',';
    std::cout << params.dilationW << ',';
    std::cout << params.O << ',';
    std::cout << params.P << ',';
    std::cout << params.Q << ',';
    std::uint64_t dram_footprint_bytes = (params.N*params.C*params.D*params.H*params.W
            +params.N*params.K*params.O*params.P*params.Q
            +params.K*params.C*params.T*params.R*params.S)*sizeof(ElementType);
    std::cout << dram_footprint_bytes << ',';
    std::cout << runs << ',';
    std::cout << timeUsed/runs << ',';
    double dram_sol_time = dram_footprint_bytes/8980480.0;
    double hfma_sol_cycle = 1.0*params.N*params.C*params.K*params.T*params.R*params.S*params.O*params.P*params.Q/80/64;
    double hfma_sol_time = hfma_sol_cycle/(1290*1024*1024)*100000;
    if(dram_sol_time>hfma_sol_time){
        std::cout << "dram" << ',';
    }else{
        std::cout << "math" << ',';
    }
    std::cout << dram_sol_time/(timeUsed/runs) << ',';
    std::cout << hfma_sol_time/(timeUsed/runs) << ',';
    return timeUsed/runs;
}

template<typename ElementType>
int Run_test(int N,
        int C,
        int D,
        int H,
        int W,
        int K,
        int T,
        int R,
        int S,
        int paddingFront,
        int paddingBack,
        int paddingTop,
        int paddingBottom,
        int paddingLeft,
        int paddingRight,
        int strideD,
        int strideH,
        int strideW,
        int dilationD,
        int dilationH,
        int dilationW,
        bool isCrossCorrelation,
        float alpha,
        float beta,
        int runs,
        bool crc_check){
            int O = GetOutputSize(D,paddingFront,paddingBack,strideD,T);
            int P = GetOutputSize(H,paddingTop,paddingBottom,strideH,R);
            int Q = GetOutputSize(W,paddingLeft,paddingRight,strideW,S);
            if(debug){
                std::cout << "O = " << O << " P = " << P << " Q = " << Q << std::endl;
            }
            std::uint64_t NKOPQ = N*K*O*P*Q;
            ElementType *gmem_error_h = new ElementType[NKOPQ];
            Fill(gmem_error_h,NKOPQ);
            std::uint64_t KCTRS = K*C*T*R*S;
            ElementType *gmem_filter_h = new ElementType[KCTRS];
            Fill(gmem_filter_h,KCTRS);
            std::uint64_t NCDHW = N*C*D*H*W;
            ElementType *gmem_image_h = new ElementType[NCDHW];
            memset((void *)gmem_image_h,0,NCDHW*sizeof(ElementType));

            ElementType *gmem_error_d;
            cudaFunc(cudaMalloc((void**)&gmem_error_d, NKOPQ*sizeof(ElementType)));
            ElementType *gmem_filter_d;
            cudaFunc(cudaMalloc((void**)&gmem_filter_d, KCTRS*sizeof(ElementType)));
            ElementType *gmem_image_d;
            cudaFunc(cudaMalloc((void**)&gmem_image_d, NCDHW*sizeof(ElementType)));

            cudaFunc(cudaMemcpy(gmem_error_d,gmem_error_h,NKOPQ*sizeof(ElementType),cudaMemcpyHostToDevice));
            cudaFunc(cudaMemcpy(gmem_filter_d,gmem_filter_h,KCTRS*sizeof(ElementType),cudaMemcpyHostToDevice));
            cudaFunc(cudaMemcpy(gmem_image_d,gmem_image_h,NCDHW*sizeof(ElementType),cudaMemcpyHostToDevice));

            KernelParams<ElementType> params = {
                &gmem_image_d[0],
                &gmem_error_d[0],
                &gmem_filter_d[0],
                N,
                C,
                D,
                H,
                W,
                K,
                T,
                R,
                S,
                paddingFront,
                paddingBack,
                paddingTop,
                paddingBottom,
                paddingLeft,
                paddingRight,
                strideD,
                strideH,
                strideW,
                dilationD,
                dilationH,
                dilationW,
                O,
                P,
                Q,
                alpha,
                beta,
                isCrossCorrelation,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            };
            if(debug){
                std::cout << "Start run kernel" << std::endl;
            }
            float custom_kernel_time;
            if(T == 2 && strideD == 2){
                if(isCrossCorrelation){
                    custom_kernel_time = Run_kernel_dgrad_3d_ndhwc<ElementType,true,2,2,2,2,2,2,1,1,1>(params, runs);
                }
                else{
                    custom_kernel_time = Run_kernel_dgrad_3d_ndhwc<ElementType,false,2,2,2,2,2,2,1,1,1>(params, runs);
                }
            }
            else{
                std::cout << "T and strideD are not instanced!" << std::endl;
            }
            cudaFunc(cudaMemcpy(gmem_image_h,gmem_image_d,NCDHW*sizeof(ElementType),cudaMemcpyDeviceToHost));

            if(!crc_check){
                //Running cudnn
                cudnnHandle_t handle;
                cudnnFunc(cudnnCreate(&handle));

                cudnnConvolutionDescriptor_t conv_desc;
                cudnnFunc(cudnnCreateConvolutionDescriptor(&conv_desc));

                cudnnConvolutionMode_t conv_mode = (params.isCrossCorrelation?CUDNN_CROSS_CORRELATION:CUDNN_CONVOLUTION);

                int conv_pad[] = {params.paddingFront, params.paddingTop, params.paddingLeft};

                int conv_stride[] = {params.strideD, params.strideH, params.strideW};

                int conv_dilation[] = {params.dilationD, params.dilationH, params.dilationW};

                cudnnDataType_t data_type;
                if(std::is_same<ElementType, float>::value){
                    data_type = CUDNN_DATA_FLOAT;
                }else{
                    data_type = CUDNN_DATA_HALF;
                }

                cudnnFunc(cudnnSetConvolutionNdDescriptor(conv_desc,
                            3,
                            conv_pad,
                            conv_stride,
                            conv_dilation,
                            conv_mode,
                            CUDNN_DATA_FLOAT));
                cudnnFunc(cudnnSetConvolutionMathType(conv_desc,CUDNN_DEFAULT_MATH));

                cudnnTensorDescriptor_t error_desc;
                cudnnFunc(cudnnCreateTensorDescriptor(&error_desc));
                int error_dim[] = {params.N, params.K,params.O,params.P,params.Q};
                cudnnFunc(cudnnSetTensorNdDescriptorEx(error_desc,CUDNN_TENSOR_NCHW,data_type,5,error_dim));

                cudnnFilterDescriptor_t filter_desc;
                cudnnFunc(cudnnCreateFilterDescriptor(&filter_desc));
                int filter_dim[]={params.K,params.C, params.T,params.R,params.S};
                cudnnFunc(cudnnSetFilterNdDescriptor(filter_desc,data_type,CUDNN_TENSOR_NCHW,5,filter_dim));

                cudnnTensorDescriptor_t image_desc;
                cudnnFunc(cudnnCreateTensorDescriptor(&image_desc));
                int image_dim[] = {params.N, params.C,params.D,params.H,params.W};
                cudnnFunc(cudnnSetTensorNdDescriptorEx(image_desc,CUDNN_TENSOR_NCHW,data_type,5,image_dim));

                cudnnConvolutionBwdDataAlgo_t conv_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

                size_t workspace_sz =0;
                cudnnStatus_t cudnn_status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                        filter_desc,
                        error_desc,
                        conv_desc,
                        image_desc,
                        conv_algo,
                        &workspace_sz);

                if(cudnn_status!=CUDNN_STATUS_SUCCESS){
                    std::cout << cudnnGetErrorString(cudnn_status) <<std::endl;
                    exit(EXIT_FAILURE);
                }

                void *workspace_d;
                cudaFunc(cudaMalloc((void**) &workspace_d, workspace_sz));

                cudaEvent_t start,stop;
                cudaFunc(cudaEventCreate(&start));
                cudaFunc(cudaEventCreate(&stop));
                cudaFunc(cudaEventRecord(start));
                for( int i = 0; i < runs; ++i ){
                    cudnnFunc(cudnnConvolutionBackwardData(handle,
                                &(params.alpha),
                                filter_desc,
                                gmem_filter_d,
                                error_desc,
                                gmem_error_d,
                                conv_desc,
                                conv_algo,
                                workspace_d,
                                workspace_sz,
                                &(params.beta),
                                image_desc,
                                gmem_image_d));
                }
                cudaFunc(cudaEventRecord(stop));
                cudaFunc(cudaDeviceSynchronize());
                float cudnn_time;
                cudaFunc(cudaEventElapsedTime(&cudnn_time, start, stop));
                std::cout << cudnn_time/runs/custom_kernel_time << "  " << std::endl;
                cudaFunc(cudaFree(workspace_d));

                cudaFunc(cudaFree(gmem_error_d));
                cudaFunc(cudaFree(gmem_filter_d));
                cudaFunc(cudaFree(gmem_image_d));
            }
            else{
                std::cout << "  " << std::endl;
            }

#ifdef FLOAT
            if(crc_check && std::is_same<ElementType,float>::value){
                ElementType *gmem_image_h_ref = new ElementType[NCDHW];
                params.gmem_image = gmem_image_h_ref;
                params.gmem_error = gmem_error_h;
                params.gmem_filter = gmem_filter_h;
                Dgrad_3d_ndhwc_cpu(params);
                int error_position = -1;
                for(int i = 0;i<NCDHW;++i){
                    if(fabs(gmem_image_h_ref[i]-gmem_image_h[i])>0.1){
                        error_position = i;
                        break;
                    }
                }
                int element_position = error_position;
                if(error_position==-1){
                    std::cout << "Pass  " << std::endl;
                }
                else{
                    int error_n,error_c,error_d,error_h,error_w;
                    error_c = error_position%C;
                    error_position/=C;
                    error_w = error_position%W;
                    error_position/=W;
                    error_h = error_position%H;
                    error_position/=H;
                    error_d = error_position%D;
                    error_n = error_position/D;
                    std::cout << "n=" << error_n << " "
                        << "c=" << error_c << " "
                        << "d=" << error_d << " "
                        << "h=" << error_h << " "
                        << "w=" << error_w << " " << std::endl;
                    std::cout << "ref = " << gmem_image_h_ref[element_position] << std::endl;
                    std::cout << "val = " << gmem_image_h[element_position] << std::endl;
                }
                delete[] gmem_image_h_ref;
            }
#endif
            delete[] gmem_error_h;
            delete[] gmem_filter_h;
            delete[] gmem_image_h;

            return 0;
        }

int main(int argc, char *argv[]){
    if(argc != 25){
        std::cout << "mode N C D H W K T R S paddingFront paddingBack paddingTop paddingBottom paddingLeft paddingRight strideD strideH strideW dilationD dilationH dilationW runs crc_check" << std::endl;
        return 0;
    }
    int index = 0;
    bool isCrossCorrelation = (std::string(argv[++index])=="cross");
    int N = atoi(argv[++index]);
    int C = atoi(argv[++index]);
    int D = atoi(argv[++index]);
    int H = atoi(argv[++index]);
    int W = atoi(argv[++index]);
    int K = atoi(argv[++index]);
    int T = atoi(argv[++index]);
    int R = atoi(argv[++index]);
    int S = atoi(argv[++index]);
    int paddingFront = atoi(argv[++index]);
    int paddingBack = atoi(argv[++index]);
    int paddingTop = atoi(argv[++index]);
    int paddingBottom = atoi(argv[++index]);
    int paddingLeft = atoi(argv[++index]);
    int paddingRight = atoi(argv[++index]);
    int strideD = atoi(argv[++index]);
    int strideH = atoi(argv[++index]);
    int strideW = atoi(argv[++index]);
    int dilationD = atoi(argv[++index]);
    int dilationH = atoi(argv[++index]);
    int dilationW = atoi(argv[++index]);
    int runs = atoi(argv[++index]);
    bool crc_check = (atoi(argv[++index])>0);

    if(debug){
        std::cout << (isCrossCorrelation?"cross":"conv") << '_';
        std::cout << N << '_';
        std::cout << C << '_';
        std::cout << D << '_';
        std::cout << H << '_';
        std::cout << W << '_';
        std::cout << K << '_';
        std::cout << T << '_';
        std::cout << R << '_';
        std::cout << S << '_';
        std::cout << paddingFront << '_';
        std::cout << paddingBack << '_';
        std::cout << paddingTop << '_';
        std::cout << paddingBottom << '_';
        std::cout << paddingLeft << '_';
        std::cout << paddingRight << '_';
        std::cout << strideD << '_';
        std::cout << strideH << '_';
        std::cout << strideW << '_';
        std::cout << dilationD << '_';
        std::cout << dilationH << '_';
        std::cout << dilationW << std::endl;
    }

#ifdef FLOAT
    Run_test<float>(N,
#else
    Run_test<__half>(N,
#endif
        C,
        D,
        H,
        W,
        K,
        T,
        R,
        S,
        paddingFront,
        paddingBack,
        paddingTop,
        paddingBottom,
        paddingLeft,
        paddingRight,
        strideD,
        strideH,
        strideW,
        dilationD,
        dilationH,
        dilationW,
        isCrossCorrelation,
        0.5,
        0.0,
        runs,
        crc_check);
    return 0;
}
