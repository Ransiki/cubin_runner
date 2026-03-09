
/*
 * moe_cubin_lib.cu — C API for launching trtllm-gen BatchedGemm exported cubins.
 *
 * ============================================================================
 * OVERVIEW
 * ============================================================================
 *
 * This library provides a thin C API for Python (via ctypes) to:
 *   1. Select the best FC1/FC2 cubin from 48 pre-compiled kernels
 *   2. Launch FC1 (fused Permute + GEMM + SwiGLU) and FC2 (plain GEMM) cubins
 *
 * The cubins are embedded as C++ byte arrays (from trtllmGen_bmm_export/cubins/*.cpp)
 * and registered in KernelMetaInfo.h. At runtime, BatchedGemmInterface handles:
 *   - Loading cubin bytes into CUmodule via cuModuleLoadData
 *   - Setting up TMA (Tensor Memory Accelerator) descriptors for Blackwell
 *   - Packing kernel parameters
 *   - Computing grid/block dimensions
 *   - Launching the kernel
 *
 * ============================================================================
 * CUBIN SELECTION
 * ============================================================================
 *
 * Each cubin is identified by a BatchedGemmConfig containing BatchedGemmOptions.
 * Selection matches on these fields (all must match):
 *
 *   dtypeA      = E2m1 (NvFP4 weights)
 *   dtypeB      = Bfloat16 (activations)
 *   dtypeC      = Bfloat16 (output)
 *   routeAct    = true (FC1: fused token permutation) / false (FC2: plain GEMM)
 *   fusedAct    = true (FC1: fused SwiGLU) / false (FC2: no activation)
 *   tileN       = tile size in the token dimension (8/16/32/64)
 *   transposeMmaOutput = true
 *   useShuffledMatrix  = true
 *   epilogueTileM      = 128
 *
 * tileN is chosen based on average tokens per expert:
 *   tileN = clamp(next_pow2(num_tokens * topK / num_experts), 8, 64)
 *
 * ============================================================================
 * FC1 CUBIN: PermuteGemm + SwiGLU
 * ============================================================================
 *
 * The FC1 cubin performs three fused operations in one kernel launch:
 *
 *   1. Permute: reads tokens from scattered positions in hidden_states
 *      using permutedIdxToTokenIdx (routeAct=ldgsts: Load Global / Store Shared)
 *   2. GEMM: multiplied with FP4 expert weights, dequantized on-the-fly
 *   3. SwiGLU: silu(gate) * up activation applied in the epilogue
 *
 * Dimensions (with transposeMmaOutput=true, batchN mode):
 *   M = 2 * intermediate_size   (weight rows: gate + up projections)
 *   N = tokens_per_expert        (batched along this dimension)
 *   K = hidden_size              (reduction dimension)
 *
 * ============================================================================
 * FC2 CUBIN: Plain Batched GEMM
 * ============================================================================
 *
 * The FC2 cubin is a straightforward batched GEMM:
 *   output = fc2_weights × fc1_output  (per expert)
 *
 * Dimensions:
 *   M = hidden_size               (weight rows: down projection)
 *   N = tokens_per_expert         (batched)
 *   K = intermediate_size         (reduction dimension)
 *
 * ============================================================================
 * BatchedGemmData STRUCTURE
 * ============================================================================
 *
 * The cubin launch requires filling BatchedGemmData with:
 *
 * ProblemDimensions:
 *   mM, mN, mK              — logical GEMM dimensions
 *   mNumBatches              — number of experts
 *   mNumTokens               — total expanded tokens (num_tokens * topK)
 *   mBatchedN                — per-expert token counts (host-side vector)
 *   mMaxNumCtasInTokenDim    — max CTA tiles across all experts
 *
 * InputBuffers:
 *   mPtrA                    — FP4 weights [E, M, K/2] uint8 (shuffled layout)
 *   mPtrSfA                  — weight scale factors [E, M, K/16] fp8 (128x4 interleaved)
 *   mPtrB                    — activations [T, K] bf16 (FC1: original hidden_states)
 *   mPtrScaleC               — output scale [E] float32 (dequant compensation)
 *   mPtrScaleGate            — gate scale [E] float32 (FC1 only)
 *   mPtrBias                 — bias [E, M] float32 (optional)
 *   mPtrGatedActAlpha/Beta   — SwiGLU params (optional)
 *   mPtrClampLimit           — clamp limit (optional)
 *   mPtrRouteMap             — permutedIdxToTokenIdx [max_
 padded] int32 (FC1 only)
 *   mPtrTotalNumPaddedTokens — [1] int32 (from routing kernel)
 *   mPtrCtaIdxXyToBatchIdx   — [max_ctas] int32 (CTA → expert mapping)
 *   mPtrCtaIdxXyToMnLimit    — [max_ctas] int32 (CTA → valid token limit)
 *   mPtrNumNonExitingCtas    — [1] int32 (active CTAs for early exit)
 *
 * OutputBuffers:
 *   mPtrC                    — output [padded_tokens, M] bf16
 */

#ifndef TLLM_GEN_EXPORT_INTERFACE
#define TLLM_GEN_EXPORT_INTERFACE
#endif
#ifndef TLLM_ENABLE_CUDA
#define TLLM_ENABLE_CUDA
#endif

#include "BatchedGemmInterface.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    auto err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[moe_cubin] CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

namespace bg = ::batchedGemm::batchedGemm;
namespace tg = ::batchedGemm::trtllm::gen;

using bg::BatchedGemmInterface;
using bg::BatchedGemmConfig;
using bg::BatchedGemmData;
using bg::BatchedGemmOptions;

/* ─── Global state ─────────────────────────────────────────────────────────── */

static std::mutex g_mutex;
static bool g_device_init = false;
static int g_device = 0;
static cudaDeviceProp g_prop;

struct CubinKernel {
    BatchedGemmInterface iface;
    BatchedGemmConfig config;
    bool valid = false;
};

static CubinKernel g_fc1;
static CubinKernel g_fc2;
static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static int ensure_device() {
    if (!g_device_init) {
        CHECK_CUDA(cudaGetDevice(&g_device));
        CHECK_CUDA(cudaGetDeviceProperties(&g_prop, g_device));
        g_device_init = true;
    }
    return 0;
}

static int ensure_workspace(size_t needed) {
    if (needed > g_workspace_size) {
        if (g_workspace) cudaFree(g_workspace);
        auto err = cudaMalloc(&g_workspace, needed);
        if (err != cudaSuccess) { g_workspace = nullptr; g_workspace_size = 0; return -1; }
        g_workspace_size = needed;
    }
    if (g_workspace && needed > 0) cudaMemset(g_workspace, 0, needed);
    return 0;
}

/* ─── Cubin selection ──────────────────────────────────────────────────────── */

/*
 * Iterate through all 48 registered cubins (from KernelMetaInfo.h) and find
 * the first one matching the requested dtype/routing/tileN combination.
 *
 * The matching criteria are identical to FlashInfer's TrtllmGenBatchedGemmRunner
 * constructor (trtllm_batched_gemm_runner.cu:93-118).
 */
static int find_best_config(
    BatchedGemmInterface& iface,
    BatchedGemmConfig& out_config,
    bool is_fc1,
    int tile_n,
    int M, int N, int K,
    int num_experts, int num_tokens)
{
    auto const* configs = iface.getBatchedGemmConfigs();
    size_t num_configs = iface.getNumBatchedGemmConfigs();

    fprintf(stderr, "[moe_cubin] Searching %zu configs for %s (tile_n=%d, M=%d, N=%d, K=%d)\n",
            num_configs, is_fc1 ? "FC1" : "FC2", tile_n, M, N, K);

    /* Build a dummy BatchedGemmData for isValidConfig checks */
    BatchedGemmData data;
    data.mProblemDimensions.mM = M;
    data.mProblemDimensions.mN = N;
    data.mProblemDimensions.mK = K;
    data.mProblemDimensions.mValidM = M;
    data.mProblemDimensions.mValidN = N;
    data.mProblemDimensions.mValidK = K;
    data.mProblemDimensions.mNumBatches = num_experts;
    data.mProblemDimensions.mNumTokens = num_tokens;
    data.mProblemDimensions.mBatchM = false;
    data.mProblemDimensions.mRank = 0;
    data.mProblemDimensions.mWorldSize = 1;
    for (int i = 0; i < num_experts; i++)
        data.mProblemDimensions.mBatchedN.push_back(N);
    data.mProblemDimensions.mMaxNumCtasInTokenDim = (N + tile_n - 1) / tile_n * num_experts;

    int best_idx = -1;
    for (size_t i = 0; i < num_configs; i++) {
        auto const& opt = configs[i].mOptions;

        /* dtype match: NvFP4 weights × BF16 activations → BF16 output */
        if (opt.mDtypeA != tg::Dtype::E2m1) continue;
        if (opt.mDtypeB != tg::Dtype::Bfloat16) continue;
        if (opt.mDtypeC != tg::Dtype::Bfloat16) continue;
        if (!opt.mTransposeMmaOutput) continue;
        if (!opt.mUseShuffledMatrix) continue;

        /* FC1 cubins have routeAct + fusedAct; FC2 cubins have neither */
        bool cubin_has_route = !bg::doesRouteImplUseNoRoute(opt.mRouteImpl);
        if (is_fc1 && (!cubin_has_route || !opt.mFusedAct)) continue;
        if (!is_fc1 && (cubin_has_route || opt.mFusedAct)) continue;

        /* tileN must match exactly */
        if (opt.mTileN != tile_n) continue;

        /* Final validation: check dimensions, pipeline stages, etc. */
        if (iface.isValidConfig(configs[i], data)) {
            best_idx = (int)i;
            break;
        }
    }

    if (best_idx < 0) {
        fprintf(stderr, "[moe_cubin] ERROR: No matching %s cubin for tile_n=%d\n",
                is_fc1 ? "FC1" : "FC2", tile_n);
        return -1;
    }

    out_config = configs[best_idx];
    fprintf(stderr, "[moe_cubin] Selected %s kernel: %s\n",
            is_fc1 ? "FC1" : "FC2", out_config.mFunctionName);
    return 0;
}

/* ─── Public C API ─────────────────────────────────────────────────────────── */

extern "C" {

/*
 * Select FC1 and FC2 cubins for the given tile_n and problem shape.
 * Call once before run (or when tile_n changes).
 */
int moe_cubin_init(
    int tile_n, int hidden_size, int intermediate_size,
    int num_experts, int tokens_per_expert)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_device() != 0) return -1;

    int num_tokens = num_experts * tokens_per_expert;

    g_fc1.iface = BatchedGemmInterface();
    int rc = find_best_config(g_fc1.iface, g_fc1.config, true, tile_n,
                              2 * intermediate_size, tokens_per_expert, hidden_size,
                              num_experts, num_tokens);
    if (rc != 0) return rc;
    g_fc1.valid = true;

    g_fc2.iface = BatchedGemmInterface();
    rc = find_best_config(g_fc2.iface, g_fc2.config, false, tile_n,
                          hidden_size, tokens_per_expert, intermediate_size,
                          num_experts, num_tokens);
    if (rc != 0) return rc;
    g_fc2.valid = true;

    fprintf(stderr, "[moe_cubin] Init complete. GPU: %s (%d SMs)\n",
            g_prop.name, g_prop.multiProcessorCount);
    return 0;
}

/*
 * Launch FC1 cubin: fused Permute + GEMM + SwiGLU.
 *
 * All pointer arguments are GPU device pointers unless noted otherwise.
 * batched_n_host is a HOST pointer to int32 array of per-expert token counts.
 */
int moe_cubin_fc1_run(
    /* weights (GPU) */
    void* weights,              /* [E, 2*I, H/2]  uint8: shuffled NvFP4 packed */
    void* weights_sf,           /* [E, 2*I, H/16] fp8:   shuffled 128x4 interleaved scales */
    /* activations (GPU) */
    void* input,                /* [T, H]          bf16:  original hidden_states (NOT permuted) */
    void* output,               /* [max_padded, I]  bf16: pre-allocated output buffer */
    /* per-expert scales (GPU) */
    float* scale_c,             /* [E] float32: output dequant scale = c_gsf / w_gsf / act_gsf */
    float* scale_gate,          /* [E] float32: gate dequant scale = 1 / w_gsf / act_gsf */
    /* optional per-expert params (GPU, nullptr if unused) */
    float* bias,                /* [E, 2*I] float32 */
    float* alpha,               /* [E] float32: SwiGLU alpha */
    float* beta,                /* [E] float32: SwiGLU beta */
    float* clamp_limit,         /* [E] float32: SwiGLU clamp */
    /* dimensions */
    int hidden_size, int intermediate_size,
    int num_experts, int num_tokens,   /* num_tokens = T * topK (expanded) */
    /* routing metadata (GPU, from FlashInfer routing kernel) */
    int* permuted_idx_to_token_idx,    /* [max_padded] cubin reads token at position[i] */
    int* cta_idx_xy_to_batch_idx,      /* [max_ctas]   which expert each CTA handles */
    int* cta_idx_xy_to_mn_limit,       /* [max_ctas]   cumulative valid token count per CTA */
    int* num_non_exiting_ctas,         /* [1]          total active CTAs */
    int* total_num_padded_tokens,      /* [1]          total tokens after padding */
    /* per-expert token counts (HOST pointer) */
    int* batched_n_host, int n_batched_n,
    /* CUDA stream */
    void* cuda_stream)
{
    if (!g_fc1.valid) { fprintf(stderr, "[moe_cubin] FC1 not initialized\n"); return -1; }
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    BatchedGemmData data;
    data.mProblemDimensions.mM = 2 * intermediate_size;
    data.mProblemDimensions.mN = g_fc1.config.mOptions.mTileN;
    data.mProblemDimensions.mK = hidden_size;
    data.mProblemDimensions.mValidM = 2 * intermediate_size;
    data.mProblemDimensions.mValidN = data.mProblemDimensions.mN;
    data.mProblemDimensions.mValidK = hidden_size;
    data.mProblemDimensions.mNumBatches = num_experts;
    data.mProblemDimensions.mNumTokens = num_tokens;
    data.mProblemDimensions.mBatchM = false;
    data.mProblemDimensions.mRank = 0;
    data.mProblemDimensions.mWorldSize = 1;
    for (int i = 0; i < n_batched_n; i++)
        data.mProblemDimensions.mBatchedN.push_back(batched_n_host[i]);
    int tile_n = g_fc1.config.mOptions.mTileN;
    int max_ctas = 0;
    for (int i = 0; i < n_batched_n; i++)
        max_ctas += (batched_n_host[i] + tile_n - 1) / tile_n;
    data.mProblemDimensions.mMaxNumCtasInTokenDim = max_ctas;

    data.mInputBuffers.mPtrA = weights;
    data.mInputBuffers.mPtrSfA = weights_sf;
    data.mInputBuffers.mPtrB = input;
    data.mInputBuffers.mPtrSfB = nullptr;
    data.mInputBuffers.mPtrScaleC = scale_c;
    data.mInputBuffers.mPtrScaleAct = scale_c;
    data.mInputBuffers.mPtrScaleGate = scale_gate;
    data.mInputBuffers.mPtrBias = bias;
    data.mInputBuffers.mPtrGatedActAlpha = alpha;
    data.mInputBuffers.mPtrGatedActBeta = beta;
    data.mInputBuffers.mPtrClampLimit = clamp_limit;
    data.mInputBuffers.mPtrPerTokenSfA = nullptr;
    data.mInputBuffers.mPtrPerTokenSfB = nullptr;
    data.mInputBuffers.mPtrRouteMap = permuted_idx_to_token_idx;
    data.mInputBuffers.mPtrTotalNumPaddedTokens = total_num_padded_tokens;
    data.mInputBuffers.mPtrCtaIdxXyToBatchIdx = cta_idx_xy_to_batch_idx;
    data.mInputBuffers.mPtrCtaIdxXyToMnLimit = cta_idx_xy_to_mn_limit;
    data.mInputBuffers.mPtrNumNonExitingCtas = num_non_exiting_ctas;

    data.mOutputBuffers.mPtrC = output;
    data.mOutputBuffers.mPtrSfC = nullptr;

    size_t ws = g_fc1.iface.getWorkspaceSizeInBytes(g_fc1.config, data);
    if (ensure_workspace(ws) != 0) return -1;

    g_fc1.iface.runInitBeforeWorldSync(g_fc1.config, data, (void*)stream);
    return (int)g_fc1.iface.run(g_fc1.config, g_workspace, data, (void*)stream,
                                 g_prop.multiProcessorCount);
}

/*
 * Launch FC2 cubin: plain batched GEMM (no routing, no activation).
 * Same calling convention as FC1 but without routing/activation params.
 */
int moe_cubin_fc2_run(
    void* weights, void* weights_sf,
    void* input, void* output,
    float* scale_c,
    int hidden_size, int intermediate_size,
    int num_experts, int num_tokens,
    int* cta_idx_xy_to_batch_idx,
    int* cta_idx_xy_to_mn_limit,
    int* num_non_exiting_ctas,
    int* total_num_padded_tokens,
    int* batched_n_host, int n_batched_n,
    void* cuda_stream)
{
    if (!g_fc2.valid) { fprintf(stderr, "[moe_cubin] FC2 not initialized\n"); return -1; }
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    BatchedGemmData data;
    data.mProblemDimensions.mM = hidden_size;
    data.mProblemDimensions.mN = g_fc2.config.mOptions.mTileN;
    data.mProblemDimensions.mK = intermediate_size;
    data.mProblemDimensions.mValidM = hidden_size;
    data.mProblemDimensions.mValidN = data.mProblemDimensions.mN;
    data.mProblemDimensions.mValidK = intermediate_size;
    data.mProblemDimensions.mNumBatches = num_experts;
    data.mProblemDimensions.mNumTokens = num_tokens;
    data.mProblemDimensions.mBatchM = false;
    data.mProblemDimensions.mRank = 0;
    data.mProblemDimensions.mWorldSize = 1;
    for (int i = 0; i < n_batched_n; i++)
        data.mProblemDimensions.mBatchedN.push_back(batched_n_host[i]);
    int tile_n = g_fc2.config.mOptions.mTileN;
    int max_ctas = 0;
    for (int i = 0; i < n_batched_n; i++)
        max_ctas += (batched_n_host[i] + tile_n - 1) / tile_n;
    data.mProblemDimensions.mMaxNumCtasInTokenDim = max_ctas;

    data.mInputBuffers.mPtrA = weights;
    data.mInputBuffers.mPtrSfA = weights_sf;
    data.mInputBuffers.mPtrB = input;
    data.mInputBuffers.mPtrSfB = nullptr;
    data.mInputBuffers.mPtrScaleC = scale_c;
    data.mInputBuffers.mPtrScaleAct = scale_c;
    data.mInputBuffers.mPtrScaleGate = nullptr;
    data.mInputBuffers.mPtrBias = nullptr;
    data.mInputBuffers.mPtrGatedActAlpha = nullptr;
    data.mInputBuffers.mPtrGatedActBeta = nullptr;
    data.mInputBuffers.mPtrClampLimit = nullptr;
    data.mInputBuffers.mPtrRouteMap = nullptr;
    data.mInputBuffers.mPtrPerTokenSfA = nullptr;
    data.mInputBuffers.mPtrPerTokenSfB = nullptr;
    data.mInputBuffers.mPtrTotalNumPaddedTokens = total_num_padded_tokens;
    data.mInputBuffers.mPtrCtaIdxXyToBatchIdx = cta_idx_xy_to_batch_idx;
    data.mInputBuffers.mPtrCtaIdxXyToMnLimit = cta_idx_xy_to_mn_limit;
    data.mInputBuffers.mPtrNumNonExitingCtas = num_non_exiting_ctas;

    data.mOutputBuffers.mPtrC = output;
    data.mOutputBuffers.mPtrSfC = nullptr;

    size_t ws = g_fc2.iface.getWorkspaceSizeInBytes(g_fc2.config, data);
    if (ensure_workspace(ws) != 0) return -1;

    g_fc2.iface.runInitBeforeWorldSync(g_fc2.config, data, (void*)stream);
    return (int)g_fc2.iface.run(g_fc2.config, g_workspace, data, (void*)stream,
                                 g_prop.multiProcessorCount);
}

int moe_cubin_get_sm_count() {
    if (ensure_device() != 0) return -1;
    return g_prop.multiProcessorCount;
}

} /* extern "C" */
