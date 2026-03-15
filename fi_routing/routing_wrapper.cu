/*
 * C API wrapper for FlashInfer's routing CUDA kernel.
 *
 * Cherry-picked from FlashInfer's trtllm_fused_moe_routing_renormalize.cu
 * with TVM FFI dependencies stubbed out.
 *
 * Provides: routing_renormalize_run() — computes topK routing + all metadata
 * needed by trtllm-gen BatchedGemm cubins.
 */

#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"
#include "tvm_ffi_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace btg = batchedGemm::trtllm::gen;

static inline int32_t computeLog2(int32_t val) {
    int32_t n = val, out = 0;
    while (n >>= 1) ++out;
    if ((1 << out) != val) out = -1;
    return out;
}

extern "C" {

/*
 * Run the Renormalize routing kernel (Softmax→TopK or TopK→Softmax).
 *
 * Input:
 *   routing_logits  [numTokens, numExperts] bf16 or float32 (GPU)
 *
 * Outputs (all GPU, pre-allocated by caller):
 *   expert_indexes           [numTokens * topK] int32 — packed score+idx
 *   expert_count_histogram   [2 * numExperts] int32 — workspace
 *   permuted_idx_size        [1] int32 — total padded token count
 *   expanded_idx_to_permuted [numTokens * topK] int32
 *   permuted_idx_to_expanded [max_padded_tokens] int32
 *   permuted_idx_to_token    [max_padded_tokens] int32
 *   expert_weights           [numTokens * topK] bf16 — routing weights
 *   cta_idx_to_batch         [max_ctas] int32
 *   cta_idx_to_mn_limit      [max_ctas] int32
 *   num_non_exiting_ctas     [1] int32
 *
 * routing_method: 0=softmax→topK (default), 1=topK→softmax (renormalize)
 *
 * Returns 0 on success.
 */
int routing_renormalize_run(
    void* routing_logits,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t top_k,
    int32_t tile_tokens_dim,
    int32_t local_expert_offset,
    int32_t local_num_experts,
    int32_t routing_method,  /* 0=softmax→topK, 1=topK→softmax */
    /* outputs (GPU) */
    int32_t* expert_indexes,
    int32_t* expert_count_histogram,
    int32_t* permuted_idx_size,
    int32_t* expanded_idx_to_permuted,
    int32_t* permuted_idx_to_expanded,
    int32_t* permuted_idx_to_token,
    void* expert_weights,
    int32_t* cta_idx_to_batch,
    int32_t* cta_idx_to_mn_limit,
    int32_t* num_non_exiting_ctas,
    void* cuda_stream)
{
    try {
        cudaStream_t stream = (cudaStream_t)cuda_stream;

        moe::dev::routing::routingRenormalize::Data data;

        data.mDtypeExpW = btg::Dtype::Fp32;
        data.mUsePdl = true;

        /* routing_method:
         *   0 = Default (Softmax → TopK) — mDoSoftmaxBeforeTopK=false, mApplySoftmaxAfterTopK=false
         *   1 = Renormalize (TopK → Softmax) — mDoSoftmaxBeforeTopK=false, mApplySoftmaxAfterTopK=true
         *   2 = RenormalizeNaive (Softmax → TopK → Renorm) — mDoSoftmaxBeforeTopK=true, mNormTopkProb=true
         */
        data.mDoSoftmaxBeforeTopK = (routing_method == 2);
        data.mNormTopkProb = (routing_method == 2);
        data.mApplySoftmaxAfterTopK = (routing_method == 1);

        data.mPtrScores = routing_logits;

        data.mPtrTopKPacked = expert_indexes;
        data.mPtrExpertCounts = expert_count_histogram;
        data.mPtrPermutedIdxSize = permuted_idx_size;
        data.mPtrExpandedIdxToPermutedIdx = expanded_idx_to_permuted;
        data.mPtrPermutedIdxToExpandedIdx = permuted_idx_to_expanded;
        data.mPtrPermutedIdxToTokenIdx = permuted_idx_to_token;
        data.mPtrTopKWeights = expert_weights;

        data.mPtrCtaIdxXyToBatchIdx = cta_idx_to_batch;
        data.mPtrCtaIdxXyToMnLimit = cta_idx_to_mn_limit;
        data.mPtrNumNonExitingCtas = num_non_exiting_ctas;

        data.mNumTokens = num_tokens;
        data.mNumExperts = num_experts;
        data.mTopK = top_k;
        data.mPaddingLog2 = computeLog2(tile_tokens_dim);
        data.mTileTokensDim = tile_tokens_dim;
        data.mLocalExpertsStartIdx = local_expert_offset;
        data.mLocalExpertsStrideLog2 = 0;
        data.mNumLocalExperts = local_num_experts;

        moe::dev::routing::routingRenormalize::run(data, stream);

        return 0;
    } catch (std::exception const& e) {
        fprintf(stderr, "[routing] Error: %s\n", e.what());
        return -1;
    }
}

/*
 * Get the maximum number of padded tokens for buffer allocation.
 */
int routing_get_max_padded_tokens(int num_tokens, int top_k, int num_experts, int tile_tokens_dim) {
    int max_ctas = 0;
    int remaining = num_tokens * top_k;
    int filled = (remaining < num_experts) ? remaining : num_experts;
    max_ctas += filled;
    remaining -= filled;
    if (remaining > 0) {
        max_ctas += remaining / tile_tokens_dim;
    }
    return max_ctas * tile_tokens_dim;
}

int routing_get_max_ctas(int num_tokens, int top_k, int num_experts, int tile_tokens_dim) {
    int remaining = num_tokens * top_k;
    int filled = (remaining < num_experts) ? remaining : num_experts;
    int max_ctas = filled;
    remaining -= filled;
    if (remaining > 0) {
        max_ctas += remaining / tile_tokens_dim;
    }
    return max_ctas;
}

} /* extern "C" */
