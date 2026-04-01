/*
 * Standalone MxInt4 cubin runner — uses FlashInfer's BatchedGemmInterface headers.
 *
 * MxInt4 FC2 cubins for small K (DSv3 I=256) have configs where TMA descriptor
 * initialization fails (tileK > K_packed). The autotune uses runInitBeforeWorldSync
 * as a pre-validation step: if TMA init throws, skip that config without running
 * any kernel (avoiding sticky CUDA context corruption).
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include "BatchedGemmInterface.h"

using namespace batchedGemm::batchedGemm;

static int g_verbose = 0;

static BatchedGemmData make_data(
    bool is_fc1, int M, int K, int tile_n, int num_experts, int num_tokens,
    void* w, void* w_sf, void* inp, void* out,
    float* sc, float* sc_gate,
    int* perm, int* cta_batch, int* cta_mn,
    int* non_exit, int* total_pad,
    int* bn_host, int n_bn)
{
    BatchedGemmData d;
    d.mInputBuffers.mPtrA = w;
    d.mInputBuffers.mPtrSfA = w_sf;
    d.mInputBuffers.mPtrB = inp;
    d.mOutputBuffers.mPtrC = out;
    d.mInputBuffers.mPtrScaleC = sc;
    d.mInputBuffers.mPtrScaleGate = sc_gate;
    d.mInputBuffers.mPtrBias = nullptr;
    d.mInputBuffers.mPtrGatedActAlpha = nullptr;
    d.mInputBuffers.mPtrGatedActBeta = nullptr;
    d.mInputBuffers.mPtrClampLimit = nullptr;

    if (is_fc1 && perm)
        d.mInputBuffers.mPtrRouteMap = perm;

    d.mProblemDimensions.mM = M;
    d.mProblemDimensions.mN = num_tokens;
    d.mProblemDimensions.mK = K;
    d.mProblemDimensions.mValidM = M;
    d.mProblemDimensions.mValidN = num_tokens;
    d.mProblemDimensions.mValidK = K;
    d.mProblemDimensions.mNumBatches = n_bn;
    d.mProblemDimensions.mNumTokens = num_tokens;
    d.mProblemDimensions.mBatchM = false;
    d.mProblemDimensions.mRank = 0;
    d.mProblemDimensions.mWorldSize = 1;

    // transposeMmaOutput=true: mBatchedN = per-expert token counts (FlashInfer convention)
    int max_ctas = 0;
    for (int i = 0; i < n_bn; i++) {
        d.mProblemDimensions.mBatchedN.push_back(bn_host[i]);
        max_ctas += (bn_host[i] + tile_n - 1) / tile_n;
    }
    d.mProblemDimensions.mMaxNumCtasInTokenDim = max_ctas;

    d.mInputBuffers.mPtrCtaIdxXyToBatchIdx = cta_batch;
    d.mInputBuffers.mPtrCtaIdxXyToMnLimit = cta_mn;
    d.mInputBuffers.mPtrNumNonExitingCtas = non_exit;
    d.mInputBuffers.mPtrTotalNumPaddedTokens = total_pad;

    return d;
}

extern "C" {

void mxint4_set_verbose(int v) { g_verbose = v; }

int mxint4_autotune(
    bool is_fc1, int tile_n, int M, int K,
    int E, int T,
    void* w, void* w_sf, void* inp, void* out,
    float* sc, float* sc_gate,
    int* perm, int* cta_batch, int* cta_mn,
    int* non_exit, int* total_pad,
    int* bn_host, int n_bn,
    int n_warmup, int n_bench,
    void* stream_ptr)
{
  try {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    BatchedGemmInterface iface;
    BatchedGemmInterface::ModuleCache mc;
    auto const* cfgs = iface.getBatchedGemmConfigs();
    size_t ncfgs = iface.getNumBatchedGemmConfigs();

    auto data = make_data(is_fc1, M, K, tile_n, E, T, w, w_sf, inp, out,
                          sc, sc_gate, perm, cta_batch, cta_mn,
                          non_exit, total_pad, bn_host, n_bn);

    int smc = 0;
    cudaDeviceGetAttribute(&smc, cudaDevAttrMultiProcessorCount, 0);

    int best = -1; float best_us = 1e9f;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    namespace tg = ::batchedGemm::trtllm::gen;

    for (size_t ci = 0; ci < ncfgs; ci++) {
        auto const& opt = cfgs[ci].mOptions;
        // Match FlashInfer's TrtllmGenBatchedGemmRunner constructor filtering
        if (opt.mDtypeA != tg::Dtype::MxInt4) continue;
        if (opt.mDtypeB != tg::Dtype::Bfloat16) continue;
        if (opt.mDtypeC != tg::Dtype::Bfloat16) continue;
        if (!opt.mTransposeMmaOutput) continue;
        bool has_route = !doesRouteImplUseNoRoute(opt.mRouteImpl);
        if (is_fc1 && (!has_route || !opt.mFusedAct)) continue;
        if (!is_fc1 && (has_route || opt.mFusedAct)) continue;
        if (opt.mTileN != tile_n) continue;
        if (!iface.isValidConfig(cfgs[ci], data)) continue;

        for (int i = 0; i < n_warmup; i++)
            iface.run(cfgs[ci], nullptr, data, stream, smc, true, mc);
        cudaStreamSynchronize(stream);

        float rounds[3];
        for (int r = 0; r < 3; r++) {
            cudaEventRecord(s, stream);
            for (int i = 0; i < n_bench; i++)
                iface.run(cfgs[ci], nullptr, data, stream, smc, true, mc);
            cudaEventRecord(e, stream);
            cudaEventSynchronize(e);
            float ms; cudaEventElapsedTime(&ms, s, e);
            rounds[r] = ms * 1000.f / n_bench;
        }
        std::sort(rounds, rounds + 3);
        float us = rounds[1];
        if (g_verbose)
            fprintf(stderr, "[mxint4] cfg[%2zu] %7.1f us  t%dx%dx%d\n",
                    ci, us, cfgs[ci].mOptions.mTileM, cfgs[ci].mOptions.mTileN, cfgs[ci].mOptions.mTileK);
        if (us < best_us) { best_us = us; best = (int)ci; }
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaGetLastError();
    if (g_verbose)
        fprintf(stderr, "[mxint4] Best %s: cfg[%d] = %.1f us\n", is_fc1?"FC1":"FC2", best, best_us);
    return best;
  } catch (std::exception& ex) {
    fprintf(stderr, "[mxint4] autotune exception: %s\n", ex.what());
    cudaGetLastError();
    return -1;
  } catch (...) {
    fprintf(stderr, "[mxint4] autotune unknown exception\n");
    cudaGetLastError();
    return -1;
  }
}

static BatchedGemmInterface g_iface;
static BatchedGemmInterface::ModuleCache g_mc;
static bool g_init = false;
static int g_smc = 0;

int mxint4_run(
    int ci, bool is_fc1,
    void* w, void* w_sf, void* inp, void* out,
    float* sc, float* sc_gate,
    void* bias, void* alpha, void* beta, void* clamp,
    int H, int I, int E, int T,
    int* perm, int* cta_batch, int* cta_mn,
    int* non_exit, int* total_pad,
    int* bn_host, int n_bn,
    void* stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (!g_init) {
        g_iface = BatchedGemmInterface();
        cudaDeviceGetAttribute(&g_smc, cudaDevAttrMultiProcessorCount, 0);
        g_init = true;
    }
    auto const* cfgs = g_iface.getBatchedGemmConfigs();
    if (ci < 0 || ci >= (int)g_iface.getNumBatchedGemmConfigs()) return -1;

    int M = is_fc1 ? 2 * I : H;
    int K = is_fc1 ? H : I;
    int tn = cfgs[ci].mOptions.mTileN;

    auto data = make_data(is_fc1, M, K, tn, E, T, w, w_sf, inp, out,
                          sc, sc_gate, perm, cta_batch, cta_mn,
                          non_exit, total_pad, bn_host, n_bn);
    data.mInputBuffers.mPtrBias = (float const*)bias;
    data.mInputBuffers.mPtrGatedActAlpha = (float const*)alpha;
    data.mInputBuffers.mPtrGatedActBeta = (float const*)beta;
    data.mInputBuffers.mPtrClampLimit = (float const*)clamp;

    g_iface.run(cfgs[ci], nullptr, data, stream, g_smc, true, g_mc);
    return 0;
}

} // extern "C"
