/*
 * moe_cubin_lib.cu — C API for launching trtllm-gen BatchedGemm exported cubins.
 *
 * ============================================================================
 * API OVERVIEW
 * ============================================================================
 *
 * Lifecycle:
 *   moe_cubin_find_valid_configs()  → enumerate all valid cubin indices for a shape
 *   moe_cubin_autotune()           → benchmark all valid cubins, return best index
 *   moe_cubin_run()                → launch a specific cubin by config_index
 *
 * The Python wrapper calls autotune() once per (shape, tile_n, is_fc1) combo,
 * caches the best config_index, and uses run() for subsequent calls.
 *
 * ============================================================================
 * CUBIN SELECTION
 * ============================================================================
 *
 * Each cubin is identified by its index in KernelMetaInfo.h's tllmGenBatchedGemmList[].
 * Selection matches on: dtypeA/B/C, routeAct, fusedAct, tileN, transposeMmaOutput,
 * useShuffledMatrix, epilogueTileM. See find_valid_configs() for details.
 *
 * ============================================================================
 * L2 CACHE FLUSHING
 * ============================================================================
 *
 * During autotuning, we flush L2 cache between benchmark iterations by writing
 * to a scratch buffer larger than L2 (64MB). This prevents artificially low
 * latency from cached weight data and gives realistic cold-cache timings.
 */

#ifndef TLLM_GEN_EXPORT_INTERFACE
#define TLLM_GEN_EXPORT_INTERFACE
#endif
#ifndef TLLM_ENABLE_CUDA
#define TLLM_ENABLE_CUDA
#endif

#include "BatchedGemmInterface.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
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

static std::mutex g_mutex;
static bool g_device_init = false;
static cudaDeviceProp g_prop;
static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

/* L2 cache flush buffer (64MB, larger than B200's 50MB L2) */
static void* g_l2_flush_buf = nullptr;
static const size_t L2_FLUSH_SIZE = 64 * 1024 * 1024;

static int ensure_device() {
    if (!g_device_init) {
        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));
        CHECK_CUDA(cudaGetDeviceProperties(&g_prop, dev));
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
        fprintf(stderr, "[workspace] Allocated %zu bytes (%.1f MB)\n", needed, needed/1e6);
    }
    /* Only memset on first allocation; skip on subsequent calls for performance */
    return 0;
}

static void flush_l2_cache(cudaStream_t stream) {
    if (!g_l2_flush_buf) {
        cudaMalloc(&g_l2_flush_buf, L2_FLUSH_SIZE);
    }
    if (g_l2_flush_buf) {
        cudaMemsetAsync(g_l2_flush_buf, 0, L2_FLUSH_SIZE, stream);
    }
}

/* ─── Helper: fill BatchedGemmData from arguments ──────────────────────────── */

static BatchedGemmData make_gemm_data(
    bool is_fc1, int M, int K, int tile_n,
    int num_experts, int num_tokens,
    void* weights, void* weights_sf,
    void* input, void* output,
    float* scale_c, float* scale_gate,
    float* bias, float* alpha, float* beta, float* clamp_limit,
    int* permuted_idx_to_token_idx,
    int* cta_idx_xy_to_batch_idx, int* cta_idx_xy_to_mn_limit,
    int* num_non_exiting_ctas, int* total_num_padded_tokens,
    int* batched_n_host, int n_batched_n)
{
    BatchedGemmData data;
    data.mProblemDimensions.mM = M;
    data.mProblemDimensions.mN = tile_n;
    data.mProblemDimensions.mK = K;
    data.mProblemDimensions.mValidM = M;
    data.mProblemDimensions.mValidN = tile_n;
    data.mProblemDimensions.mValidK = K;
    data.mProblemDimensions.mNumBatches = num_experts;
    data.mProblemDimensions.mNumTokens = num_tokens;
    data.mProblemDimensions.mBatchM = false;
    data.mProblemDimensions.mRank = 0;
    data.mProblemDimensions.mWorldSize = 1;
    int max_ctas = 0;
    for (int i = 0; i < n_batched_n; i++) {
        data.mProblemDimensions.mBatchedN.push_back(batched_n_host[i]);
        max_ctas += (batched_n_host[i] + tile_n - 1) / tile_n;
    }
    data.mProblemDimensions.mMaxNumCtasInTokenDim = max_ctas;

    data.mInputBuffers.mPtrA = weights;
    data.mInputBuffers.mPtrSfA = weights_sf;
    data.mInputBuffers.mPtrB = input;
    data.mInputBuffers.mPtrSfB = nullptr;
    data.mInputBuffers.mPtrScaleC = scale_c;
    data.mInputBuffers.mPtrScaleAct = scale_c;
    data.mInputBuffers.mPtrScaleGate = is_fc1 ? scale_gate : nullptr;
    data.mInputBuffers.mPtrBias = bias;
    data.mInputBuffers.mPtrGatedActAlpha = alpha;
    data.mInputBuffers.mPtrGatedActBeta = beta;
    data.mInputBuffers.mPtrClampLimit = clamp_limit;
    data.mInputBuffers.mPtrPerTokenSfA = nullptr;
    data.mInputBuffers.mPtrPerTokenSfB = nullptr;
    data.mInputBuffers.mPtrRouteMap = is_fc1 ? permuted_idx_to_token_idx : nullptr;
    data.mInputBuffers.mPtrTotalNumPaddedTokens = total_num_padded_tokens;
    data.mInputBuffers.mPtrCtaIdxXyToBatchIdx = cta_idx_xy_to_batch_idx;
    data.mInputBuffers.mPtrCtaIdxXyToMnLimit = cta_idx_xy_to_mn_limit;
    data.mInputBuffers.mPtrNumNonExitingCtas = num_non_exiting_ctas;
    data.mOutputBuffers.mPtrC = output;
    data.mOutputBuffers.mPtrSfC = nullptr;
    return data;
}

extern "C" {

int moe_cubin_get_sm_count() {
    if (ensure_device() != 0) return -1;
    return g_prop.multiProcessorCount;
}

/*
 * Find all valid cubin config indices for a given shape.
 * Returns number found. Writes indices to out_indices (max_results entries).
 */
int moe_cubin_find_valid_configs(
    bool is_fc1, int tile_n,
    int M, int N, int K,
    int num_experts, int num_tokens,
    int* out_indices, int max_results)
{
    if (ensure_device() != 0) return 0;

    BatchedGemmInterface iface;
    auto const* configs = iface.getBatchedGemmConfigs();
    size_t num_configs = iface.getNumBatchedGemmConfigs();

    /* Build dummy data for isValidConfig check */
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

    int found = 0;
    for (size_t i = 0; i < num_configs && found < max_results; i++) {
        auto const& opt = configs[i].mOptions;
#if DTYPE_MODE == 1
        if (opt.mDtypeA != tg::Dtype::MxE2m1) continue;
#else
        if (opt.mDtypeA != tg::Dtype::E2m1) continue;
#endif
        if (opt.mDtypeB != tg::Dtype::Bfloat16) continue;
        if (opt.mDtypeC != tg::Dtype::Bfloat16) continue;
        if (!opt.mTransposeMmaOutput) continue;
        if (!opt.mUseShuffledMatrix) continue;
        bool has_route = !bg::doesRouteImplUseNoRoute(opt.mRouteImpl);
        if (is_fc1 && (!has_route || !opt.mFusedAct)) continue;
        if (!is_fc1 && (has_route || opt.mFusedAct)) continue;
        if (opt.mTileN != tile_n) continue;
        if (opt.mClusterDimX > 1 && opt.mClusterDimZ > 1) continue;
        if (iface.isValidConfig(configs[i], data)) {
            out_indices[found++] = (int)i;
        }
    }
    return found;
}

/*
 * Autotune: benchmark all valid cubins for a shape, return the best config_index.
 *
 * Uses L2 cache flushing between iterations for realistic timings.
 * All buffer pointers are GPU (can be dummy/zero data — only timing matters).
 * batched_n_host is a HOST pointer.
 *
 * Returns best config_index, or -1 on failure.
 */
int moe_cubin_autotune(
    bool is_fc1, int tile_n,
    int M, int K,
    int num_experts, int num_tokens,
    void* weights, void* weights_sf,
    void* input, void* output,
    float* scale_c, float* scale_gate,
    int* permuted_idx_to_token_idx,
    int* cta_idx_xy_to_batch_idx,
    int* cta_idx_xy_to_mn_limit,
    int* num_non_exiting_ctas,
    int* total_num_padded_tokens,
    int* batched_n_host, int n_batched_n,
    int n_warmup, int n_bench,
    void* cuda_stream)
{
    if (ensure_device() != 0) return -1;
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    /* Find all valid configs */
    int valid_indices[128];
    int n_valid = moe_cubin_find_valid_configs(
        is_fc1, tile_n, M, tile_n, K, num_experts, num_tokens,
        valid_indices, 128);

    if (n_valid == 0) {
        fprintf(stderr, "[autotune] No valid %s cubins for tile_n=%d M=%d K=%d\n",
                is_fc1 ? "FC1" : "FC2", tile_n, M, K);
        return -1;
    }
    if (n_valid == 1) {
        fprintf(stderr, "[autotune] Only one valid %s cubin: config[%d]\n",
                is_fc1 ? "FC1" : "FC2", valid_indices[0]);
        return valid_indices[0];
    }

    fprintf(stderr, "[autotune] Benchmarking %d %s cubins (warmup=%d, bench=%d)...\n",
            n_valid, is_fc1 ? "FC1" : "FC2", n_warmup, n_bench);

    /* Use persistent interface to cache loaded CUmodules across configs */
    static BatchedGemmInterface s_tune_iface;
    static BatchedGemmInterface::ModuleCache s_tune_cache;
    static bool s_tune_init = false;
    if (!s_tune_init) { s_tune_iface = BatchedGemmInterface(); s_tune_init = true; }
    auto& iface = s_tune_iface;
    auto const* configs = iface.getBatchedGemmConfigs();

    auto data = make_gemm_data(is_fc1, M, K, tile_n, num_experts, num_tokens,
                                weights, weights_sf, input, output,
                                scale_c, scale_gate,
                                nullptr, nullptr, nullptr, nullptr,
                                permuted_idx_to_token_idx,
                                cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit,
                                num_non_exiting_ctas, total_num_padded_tokens,
                                batched_n_host, n_batched_n);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int best_idx = -1;
    float best_us = 1e30f;
    const int N_ROUNDS = 3;

    for (int v = 0; v < n_valid; v++) {
        int ci = valid_indices[v];
        auto const& cfg = configs[ci];
        auto const& opt = cfg.mOptions;

        size_t ws = iface.getWorkspaceSizeInBytes(cfg, data);
        if (ensure_workspace(ws) != 0) continue;

        iface.runInitBeforeWorldSync(cfg, data, (void*)stream);

        /* Warmup with L2 flush */
        for (int w = 0; w < n_warmup; w++) {
            flush_l2_cache(stream);
            iface.run(cfg, g_workspace, data, (void*)stream, g_prop.multiProcessorCount,
                       true, nullptr, s_tune_cache);
        }
        cudaStreamSynchronize(stream);

        /* Multi-round benchmark: per-iteration L2 flush, take median across rounds */
        float round_us[N_ROUNDS];
        for (int r = 0; r < N_ROUNDS; r++) {
            cudaEventRecord(ev_start, stream);
            for (int b = 0; b < n_bench; b++) {
                flush_l2_cache(stream);
                iface.run(cfg, g_workspace, data, (void*)stream, g_prop.multiProcessorCount,
                           true, nullptr, s_tune_cache);
            }
            cudaEventRecord(ev_stop, stream);
            cudaEventSynchronize(ev_stop);
            float elapsed_ms = 0;
            cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
            round_us[r] = (elapsed_ms * 1000.0f) / n_bench;
        }
        std::sort(round_us, round_us + N_ROUNDS);
        float us = round_us[N_ROUNDS / 2];

        const char* sched = (opt.mTileScheduler == ::batchedGemm::gemm::TileScheduler::Persistent) ? "persistent" : "static";

        fprintf(stderr, "[autotune]   config[%2d] %7.1f us  tile=%dx%dx%d stages=%d/%d %s%s%s\n",
                ci, us, opt.mTileM, opt.mTileN, opt.mTileK,
                opt.mNumStages, opt.mNumStagesMma, sched,
                opt.mUseUnrollLoop2xForMma ? " u2" : "",
                (us < best_us) ? "  <-- new best" : "");

        if (us < best_us) {
            best_us = us;
            best_idx = ci;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "[autotune] Best %s: config[%d] = %.1f us\n",
            is_fc1 ? "FC1" : "FC2", best_idx, best_us);
    return best_idx;
}

/*
 * Launch a specific cubin by config_index.
 * config_index must be obtained from find_valid_configs() or autotune().
 *
 * This is the core run function — called for every forward pass after autotuning.
 */
/* Persistent interface + module cache — avoids reloading cubins on every call */
static BatchedGemmInterface g_run_iface;
static BatchedGemmInterface::ModuleCache g_module_cache;
static bool g_run_iface_init = false;

int moe_cubin_run(
    int config_index,
    bool is_fc1,
    void* weights, void* weights_sf,
    void* input, void* output,
    float* scale_c, float* scale_gate,
    float* bias, float* alpha, float* beta, float* clamp_limit,
    int hidden_size, int intermediate_size,
    int num_experts, int num_tokens,
    int* permuted_idx_to_token_idx,
    int* cta_idx_xy_to_batch_idx,
    int* cta_idx_xy_to_mn_limit,
    int* num_non_exiting_ctas,
    int* total_num_padded_tokens,
    int* batched_n_host, int n_batched_n,
    void* cuda_stream)
{
    if (ensure_device() != 0) return -1;
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    if (!g_run_iface_init) {
        g_run_iface = BatchedGemmInterface();
        g_run_iface_init = true;
    }
    auto& iface = g_run_iface;
    auto const* configs = iface.getBatchedGemmConfigs();
    size_t num_configs = iface.getNumBatchedGemmConfigs();

    if (config_index < 0 || config_index >= (int)num_configs) {
        fprintf(stderr, "[moe_cubin] Invalid config_index %d (have %zu)\n",
                config_index, num_configs);
        return -1;
    }

    int M = is_fc1 ? 2 * intermediate_size : hidden_size;
    int K = is_fc1 ? hidden_size : intermediate_size;
    int tile_n = configs[config_index].mOptions.mTileN;

    auto data = make_gemm_data(is_fc1, M, K, tile_n, num_experts, num_tokens,
                                weights, weights_sf, input, output,
                                scale_c, scale_gate,
                                bias, alpha, beta, clamp_limit,
                                permuted_idx_to_token_idx,
                                cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit,
                                num_non_exiting_ctas, total_num_padded_tokens,
                                batched_n_host, n_batched_n);

    size_t ws = iface.getWorkspaceSizeInBytes(configs[config_index], data);
    if (ensure_workspace(ws) != 0) return -1;

    iface.runInitBeforeWorldSync(configs[config_index], data, (void*)stream);

    static int call_count = 0;
    call_count++;
    if (call_count <= 5) {
        fprintf(stderr, "[run] call %d: config_index=%d, cache size=%zu, func=%s\n",
                call_count, config_index, g_module_cache.size(),
                configs[config_index].mFunctionName);
    }

    return (int)iface.run(configs[config_index], g_workspace, data, (void*)stream,
                           g_prop.multiProcessorCount,
                           /*usePdl=*/true, /*pinnedHostBuffer=*/nullptr,
                           g_module_cache);
}

/*
 * Get human-readable info for a config index.
 */
/*
 * Fused pipeline: routing → FC1 → FC2 in one call, zero CPU sync in the middle.
 *
 * All routing metadata stays on GPU. No host readback. No Python overhead.
 * The only CPU→GPU boundary is this single ctypes call.
 */
int moe_cubin_fused_run(
    int fc1_config_index,
    int fc2_config_index,
    /* routing inputs (GPU) */
    void* routing_logits,        /* [T, E] bf16 or float32 */
    int num_tokens,
    int num_experts,
    int top_k,
    int tile_n,
    int routing_method,          /* 0=default, 1=renormalize */
    /* weights (GPU) */
    void* fc1_weights, void* fc1_weights_sf,
    void* fc2_weights, void* fc2_weights_sf,
    /* activations (GPU) */
    void* hidden_states,         /* [T, H] bf16 */
    /* output (GPU, pre-allocated) */
    void* fc2_output,            /* [max_padded, H] bf16 */
    /* scales (GPU) */
    float* scale_c_fc1, float* scale_gate_fc1, float* scale_c_fc2,
    /* optional (GPU, can be nullptr) */
    float* fc1_bias, float* fc1_alpha, float* fc1_beta, float* fc1_clamp,
    /* dimensions */
    int hidden_size, int intermediate_size,
    /* routing output buffers (GPU, pre-allocated by caller) */
    int* expert_indexes,         /* [T*K] */
    int* expert_count_hist,      /* [2*E] */
    int* permuted_idx_size,      /* [1] */
    int* expanded_to_perm,       /* [T*K] */
    int* perm_to_expanded,       /* [max_padded] */
    int* perm_to_token,          /* [max_padded] */
    void* expert_weights,        /* [T*K] float32 */
    int* cta_to_batch,           /* [max_ctas] */
    int* cta_to_mn,              /* [max_ctas] */
    int* num_non_exit,           /* [1] */
    /* FC1 intermediate (GPU, pre-allocated) */
    void* fc1_output,            /* [max_padded, I] bf16 */
    /* stream */
    void* cuda_stream)
{
    if (ensure_device() != 0) return -1;
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    int expanded = num_tokens * top_k;

    /* ── Step 1: Routing (all async on stream) ── */
    {
        extern int routing_renormalize_run(
            void*, int32_t, int32_t, int32_t, int32_t,
            int32_t, int32_t, int32_t,
            int32_t*, int32_t*, int32_t*,
            int32_t*, int32_t*, int32_t*,
            void*,
            int32_t*, int32_t*, int32_t*,
            void*);

        int rc = routing_renormalize_run(
            routing_logits, num_tokens, num_experts, top_k, tile_n,
            0, num_experts, routing_method,
            expert_indexes, expert_count_hist, permuted_idx_size,
            expanded_to_perm, perm_to_expanded, perm_to_token,
            expert_weights,
            cta_to_batch, cta_to_mn, num_non_exit,
            cuda_stream);
        if (rc != 0) return -100;
    }

    /*
     * ── Step 2: Build dummy batched_n (NO sync needed) ──
     *
     * The cubin with earlyExit+dynamic batch uses ctaIdxXyToBatchIdx/MnLimit
     * at runtime, NOT mBatchedN. mBatchedN is only consumed by
     * getOptionsFromConfigAndData for validation and workspace sizing.
     * Since autotune already validated the config and workspace is
     * pre-allocated to max size, we can pass tile_n for all experts.
     */
    static std::vector<int32_t> h_batched_n;
    if ((int)h_batched_n.size() != num_experts) {
        h_batched_n.resize(num_experts);
    }
    for (int e = 0; e < num_experts; e++) h_batched_n[e] = tile_n;

    /* ── Step 4: FC1 cubin (async) ── */
    if (!g_run_iface_init) { g_run_iface = BatchedGemmInterface(); g_run_iface_init = true; }

    int fc1_M = 2 * intermediate_size;
    int fc1_K = hidden_size;
    {
        auto data = make_gemm_data(true, fc1_M, fc1_K, tile_n, num_experts, expanded,
                                    fc1_weights, fc1_weights_sf, hidden_states, fc1_output,
                                    scale_c_fc1, scale_gate_fc1,
                                    fc1_bias, fc1_alpha, fc1_beta, fc1_clamp,
                                    perm_to_token, cta_to_batch, cta_to_mn,
                                    num_non_exit, permuted_idx_size,
                                    h_batched_n.data(), num_experts);
        auto& cfg = g_run_iface.getBatchedGemmConfigs()[fc1_config_index];
        size_t ws = g_run_iface.getWorkspaceSizeInBytes(cfg, data);
        if (ensure_workspace(ws) != 0) return -2;
        g_run_iface.runInitBeforeWorldSync(cfg, data, (void*)stream);
        int rc = (int)g_run_iface.run(cfg, g_workspace, data, (void*)stream,
                                       g_prop.multiProcessorCount, true, nullptr, g_module_cache);
        if (rc != 0) return -3;
    }

    /* ── Step 5: FC2 cubin (async) ── */
    int fc2_M = hidden_size;
    int fc2_K = intermediate_size;
    {
        auto data = make_gemm_data(false, fc2_M, fc2_K, tile_n, num_experts, expanded,
                                    fc2_weights, fc2_weights_sf, fc1_output, fc2_output,
                                    scale_c_fc2, nullptr,
                                    nullptr, nullptr, nullptr, nullptr,
                                    nullptr, cta_to_batch, cta_to_mn,
                                    num_non_exit, permuted_idx_size,
                                    h_batched_n.data(), num_experts);
        auto& cfg = g_run_iface.getBatchedGemmConfigs()[fc2_config_index];
        size_t ws = g_run_iface.getWorkspaceSizeInBytes(cfg, data);
        if (ensure_workspace(ws) != 0) return -4;
        g_run_iface.runInitBeforeWorldSync(cfg, data, (void*)stream);
        int rc = (int)g_run_iface.run(cfg, g_workspace, data, (void*)stream,
                                       g_prop.multiProcessorCount, true, nullptr, g_module_cache);
        if (rc != 0) return -5;
    }

    return 0;
}

/*
 * Run the FlashInfer fused finalize kernel: unpermute + scale + reduce in one launch.
 * Replaces the 7-step PyTorch finalize (gather, cast, mul, sum, cast).
 */
int moe_cubin_finalize(
    void* fc2_output,              /* [max_padded, H] bf16 — input from FC2 */
    void* output,                  /* [T, H] bf16 — final output (pre-allocated) */
    void* expert_weights,          /* [T*K] float32 — routing weights */
    int* expanded_idx_to_permuted, /* [T*K] int32 */
    int* total_num_padded_tokens,  /* [1] int32 (GPU) */
    int num_tokens,
    int num_experts,
    int top_k,
    int hidden_size,
    void* cuda_stream)
{
    ::moe::dev::finalize::Data fdata;
    fdata.mDtypeElt = tg::Dtype::Bfloat16;
    fdata.mDtypeExpW = tg::Dtype::Fp32;
    fdata.mUsePdl = false;
    fdata.mUseDeepSeekFp8 = false;
    fdata.inPtr = fc2_output;
    fdata.outPtr = output;
    fdata.inDqSfsPtr = nullptr;
    fdata.outDqSfsPtr = nullptr;
    fdata.expertWeightsPtr = expert_weights;
    fdata.expandedIdxToPermutedIdx = expanded_idx_to_permuted;
    fdata.numTokens = num_tokens;
    fdata.numExperts = num_experts;
    fdata.topK = top_k;
    fdata.hiddenDim = hidden_size;
    fdata.hiddenDimPadded = hidden_size;
    fdata.totalNumPaddedTokens = total_num_padded_tokens;

    ::moe::dev::finalize::run(fdata, cuda_stream);
    return 0;
}

void moe_cubin_get_config_info(int config_index,
    int* out_tileM, int* out_tileN, int* out_tileK,
    int* out_numStages, int* out_numStagesMma, int* out_isPersistent, int* out_isUnroll2x)
{
    BatchedGemmInterface iface;
    auto const* configs = iface.getBatchedGemmConfigs();
    auto const& o = configs[config_index].mOptions;
    *out_tileM = o.mTileM;
    *out_tileN = o.mTileN;
    *out_tileK = o.mTileK;
    *out_numStages = o.mNumStages;
    *out_numStagesMma = o.mNumStagesMma;
    *out_isPersistent = (o.mTileScheduler == ::batchedGemm::gemm::TileScheduler::Persistent) ? 1 : 0;
    *out_isUnroll2x = o.mUseUnrollLoop2xForMma ? 1 : 0;
}

void moe_cubin_get_config_info_ext(int config_index,
    int* out_tileM, int* out_tileN, int* out_tileK,
    int* out_numStages, int* out_numStagesMma,
    int* out_isPersistent, int* out_isUnroll2x,
    int* out_clusterDimX, int* out_splitK, int* out_mmaM)
{
    BatchedGemmInterface iface;
    auto const* configs = iface.getBatchedGemmConfigs();
    auto const& o = configs[config_index].mOptions;
    *out_tileM = o.mTileM;
    *out_tileN = o.mTileN;
    *out_tileK = o.mTileK;
    *out_numStages = o.mNumStages;
    *out_numStagesMma = o.mNumStagesMma;
    *out_isPersistent = (o.mTileScheduler == ::batchedGemm::gemm::TileScheduler::Persistent) ? 1 : 0;
    *out_isUnroll2x = o.mUseUnrollLoop2xForMma ? 1 : 0;
    *out_clusterDimX = o.mClusterDimX;
    *out_splitK = o.mNumSlicesForSplitK;
    *out_mmaM = o.mMmaM;
}

} /* extern "C" */
