"""
NvFP4 × BF16 MOE using trtllm-gen BatchedGemm exported cubins.

Pipeline:
  1. Routing:  FlashInfer CUDA kernel → topK experts + permutation metadata
  2. FC1:      trtllm-gen cubin (fused permute + GEMM + SwiGLU)
  3. FC2:      trtllm-gen cubin (plain batched GEMM)
  4. Finalize: PyTorch (unpermute + weighted sum)

On first call for each (tile_n, shape) combination, autotuning benchmarks
all valid cubins with L2 cache flushing and selects the fastest one.
Subsequent calls reuse the cached best config.
"""
import ctypes
import math
import os
import torch
import torch.nn.functional as F
from typing import Optional, List
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# Library loading
# ═════════════════════════════════════════════════════════════════════════════

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    for p in [
        Path(__file__).parent / "libmoe_cubin_lib.so",
        Path(__file__).parent / "build" / "libmoe_cubin_lib.so",
    ]:
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            _setup_signatures(_lib)
            return _lib
    raise RuntimeError("libmoe_cubin_lib.so not found. Build with cmake.")


def _setup_signatures(lib):
    """Declare ctypes function signatures."""
    # autotune
    lib.moe_cubin_autotune.argtypes = [
        ctypes.c_bool, ctypes.c_int,  # is_fc1, tile_n
        ctypes.c_int, ctypes.c_int,   # M, K
        ctypes.c_int, ctypes.c_int,   # num_experts, num_tokens
        ctypes.c_void_p, ctypes.c_void_p,  # weights, weights_sf
        ctypes.c_void_p, ctypes.c_void_p,  # input, output
        ctypes.c_void_p, ctypes.c_void_p,  # scale_c, scale_gate
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # routing meta
        ctypes.c_void_p, ctypes.c_void_p,  # num_non_exit, total_padded
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # batched_n, n
        ctypes.c_int, ctypes.c_int,   # n_warmup, n_bench
        ctypes.c_void_p,              # stream
    ]
    lib.moe_cubin_autotune.restype = ctypes.c_int

    # run
    lib.moe_cubin_run.argtypes = [
        ctypes.c_int, ctypes.c_bool,  # config_index, is_fc1
        ctypes.c_void_p, ctypes.c_void_p,  # weights, weights_sf
        ctypes.c_void_p, ctypes.c_void_p,  # input, output
        ctypes.c_void_p, ctypes.c_void_p,  # scale_c, scale_gate
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # bias,alpha,beta,clamp
        ctypes.c_int, ctypes.c_int,        # hidden_size, intermediate_size
        ctypes.c_int, ctypes.c_int,        # num_experts, num_tokens
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # routing meta
        ctypes.c_void_p, ctypes.c_void_p,  # num_non_exit, total_padded
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # batched_n, n
        ctypes.c_void_p,              # stream
    ]
    lib.moe_cubin_run.restype = ctypes.c_int

    # routing
    lib.routing_renormalize_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ] + [ctypes.c_void_p] * 11
    lib.routing_renormalize_run.restype = ctypes.c_int32


def _ptr(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())

def _host_int_array(t):
    cpu = t.cpu().to(torch.int32).contiguous().numpy()
    return (ctypes.c_int * len(cpu))(*cpu)


# ═════════════════════════════════════════════════════════════════════════════
# Autotuning cache (in-memory + disk persistent)
# ═════════════════════════════════════════════════════════════════════════════
#
# Cache key: (is_fc1, tile_n, M, K, num_experts)
# Cache value: best config_index (int)
#
# On first call for a new key:
#   1. Check disk cache (~/.cache/trtllmgen_runner/autotune.json)
#   2. If miss, run autotune on GPU, save to disk + memory
#   3. Subsequent calls use memory cache (zero overhead)

_autotune_cache = {}
_AUTOTUNE_WARMUP = 5
_AUTOTUNE_BENCH = 20


def _get_best_config(
    lib, is_fc1: bool, tile_n: int,
    M: int, K: int, num_experts: int, num_tokens: int,
    meta: dict, device: torch.device,
) -> int:
    """Get the best cubin config_index for this shape, autotuning if needed."""
    cache_key = (is_fc1, tile_n, M, K, num_experts)

    if cache_key in _autotune_cache:
        return _autotune_cache[cache_key]

    # Allocate dummy buffers for autotuning (zero data, only timing matters)
    alloc_padded = meta["max_padded"]
    w = torch.zeros(num_experts, M, K // 2, device=device, dtype=torch.uint8)
    w_sf = torch.ones(num_experts, M, K // 16, device=device, dtype=torch.float8_e4m3fn)
    inp = torch.zeros(num_tokens if is_fc1 else alloc_padded, K, device=device, dtype=torch.bfloat16)
    out_dim = M // 2 if is_fc1 else M  # FC1 outputs intermediate_size, FC2 outputs hidden_size
    out = torch.zeros(alloc_padded, out_dim, device=device, dtype=torch.bfloat16)
    sc = torch.ones(num_experts, device=device, dtype=torch.float32)

    batched_n_host = _host_int_array(meta["batched_n"])
    stream = torch.cuda.current_stream(device).cuda_stream

    best = lib.moe_cubin_autotune(
        is_fc1, tile_n, M, K, num_experts, num_tokens,
        _ptr(w), _ptr(w_sf), _ptr(inp), _ptr(out),
        _ptr(sc), _ptr(sc),
        _ptr(meta["permuted_idx_to_token_idx"]),
        _ptr(meta["cta_idx_xy_to_batch_idx"]),
        _ptr(meta["cta_idx_xy_to_mn_limit"]),
        _ptr(meta["num_non_exiting_ctas"]),
        _ptr(meta["total_num_padded_tokens"]),
        batched_n_host, num_experts,
        _AUTOTUNE_WARMUP, _AUTOTUNE_BENCH,
        ctypes.c_void_p(stream),
    )

    if best < 0:
        raise RuntimeError(f"Autotune failed for {'FC1' if is_fc1 else 'FC2'} "
                           f"tile_n={tile_n} M={M} K={K}")

    _autotune_cache[cache_key] = best
    return best


# ═════════════════════════════════════════════════════════════════════════════
# Routing — FlashInfer CUDA kernel
# ═════════════════════════════════════════════════════════════════════════════

def compute_routing_cuda(
    routing_logits, num_experts, top_k, tile_n, routing_method=1,
):
    """Run FlashInfer CUDA routing kernel. See docstring in previous version."""
    lib = _load_lib()
    device = routing_logits.device
    T = routing_logits.shape[0]
    expanded = T * top_k
    max_padded = lib.routing_get_max_padded_tokens(T, top_k, num_experts, tile_n)
    max_ctas = lib.routing_get_max_ctas(T, top_k, num_experts, tile_n)

    expert_indexes    = torch.zeros(expanded, dtype=torch.int32, device=device)
    expert_count_hist = torch.zeros(2 * num_experts, dtype=torch.int32, device=device)
    permuted_idx_size = torch.zeros(1, dtype=torch.int32, device=device)
    expanded_to_perm  = torch.full((expanded,), -1, dtype=torch.int32, device=device)
    perm_to_expanded  = torch.full((max_padded,), -1, dtype=torch.int32, device=device)
    perm_to_token     = torch.full((max_padded,), -1, dtype=torch.int32, device=device)
    expert_weights    = torch.zeros(expanded, dtype=torch.bfloat16, device=device)
    cta_to_batch      = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    cta_to_mn         = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    num_non_exit      = torch.zeros(1, dtype=torch.int32, device=device)

    logits_bf16 = routing_logits.bfloat16().contiguous()
    stream = torch.cuda.current_stream(device).cuda_stream
    rc = lib.routing_renormalize_run(
        _ptr(logits_bf16), T, num_experts, top_k, tile_n,
        0, num_experts, routing_method,
        _ptr(expert_indexes), _ptr(expert_count_hist), _ptr(permuted_idx_size),
        _ptr(expanded_to_perm), _ptr(perm_to_expanded), _ptr(perm_to_token),
        _ptr(expert_weights),
        _ptr(cta_to_batch), _ptr(cta_to_mn), _ptr(num_non_exit),
        ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if rc != 0:
        raise RuntimeError(f"routing kernel failed (rc={rc})")

    actual_padded = permuted_idx_size[0].item()
    actual_ctas = num_non_exit[0].item()

    # Extract per-expert token counts from CTA mapping (vectorized)
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if actual_ctas > 0:
        cb = cta_to_batch[:actual_ctas]
        cm = cta_to_mn[:actual_ctas]
        ctas_per_expert = torch.bincount(cb.long(), minlength=num_experts).to(torch.int32)
        eco = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        eco[1:] = torch.cumsum(ctas_per_expert, dim=0)
        has = ctas_per_expert > 0
        last = eco[1:] - 1
        start = eco[:num_experts] * tile_n
        tokens_per_expert = torch.where(has, cm[last.long().clamp(min=0)] - start,
                                         torch.zeros_like(ctas_per_expert))

    return {
        "topk_weights": expert_weights.float().reshape(T, top_k),
        "tokens_per_expert": tokens_per_expert,
        "permuted_idx_to_token_idx": perm_to_token,
        "expanded_idx_to_permuted": expanded_to_perm,
        "cta_idx_xy_to_batch_idx": cta_to_batch,
        "cta_idx_xy_to_mn_limit": cta_to_mn,
        "num_non_exiting_ctas": num_non_exit,
        "total_num_padded_tokens": permuted_idx_size,
        "batched_n": tokens_per_expert.clone(),
        "total_padded": actual_padded,
        "max_padded": max_padded,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main API
# ═════════════════════════════════════════════════════════════════════════════

def _select_tile_n(num_tokens, top_k, num_experts):
    avg = (num_tokens * top_k) / num_experts
    return max(8, min(64, 2 ** math.ceil(math.log2(max(1, avg)))))


def nvfp4_moe_cubin(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    routing_bias: Optional[torch.Tensor] = None,
    hidden_states_scale: Optional[torch.Tensor] = None,
    gemm1_bias: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    gemm2_bias: Optional[torch.Tensor] = None,
    output1_scale_scalar: Optional[torch.Tensor] = None,
    output1_scale_gate_scalar: Optional[torch.Tensor] = None,
    output2_scale_scalar: Optional[torch.Tensor] = None,
    do_finalize: bool = True,
    routing_method_type: int = 1,
) -> List[torch.Tensor]:
    """
    NvFP4 × BF16 MOE forward pass with automatic cubin autotuning.

    First call per shape triggers autotuning (benchmarks all valid cubins with
    L2 cache flushing). Subsequent calls use cached best config.
    """
    lib = _load_lib()
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    device = hidden_states.device

    tile_n = _select_tile_n(num_tokens, top_k, num_experts)
    expanded = num_tokens * top_k
    stream = torch.cuda.current_stream(device).cuda_stream

    # Autotune on first call (needs routing metadata, so runs routing once)
    fc1_M, fc1_K = 2 * intermediate_size, hidden_size
    fc2_M, fc2_K = hidden_size, intermediate_size
    ck1, ck2 = (True, tile_n, fc1_M, fc1_K, num_experts), (False, tile_n, fc2_M, fc2_K, num_experts)

    if ck1 not in _autotune_cache or ck2 not in _autotune_cache:
        torch.cuda.nvtx.range_push("autotune")
        meta = compute_routing_cuda(routing_logits, num_experts, top_k, tile_n, routing_method_type)
        _get_best_config(lib, True, tile_n, fc1_M, fc1_K, num_experts, expanded, meta, device)
        _get_best_config(lib, False, tile_n, fc2_M, fc2_K, num_experts, expanded, meta, device)
        torch.cuda.nvtx.range_pop()

    fc1_config = _autotune_cache[ck1]
    fc2_config = _autotune_cache[ck2]

    # Pre-allocate GPU buffers (reuse across calls)
    max_padded = lib.routing_get_max_padded_tokens(num_tokens, top_k, num_experts, tile_n)
    max_ctas = lib.routing_get_max_ctas(num_tokens, top_k, num_experts, tile_n)
    bk = (num_tokens, num_experts, top_k, tile_n, hidden_size, intermediate_size)

    if not hasattr(nvfp4_moe_cubin, '_bufs') or nvfp4_moe_cubin._bk != bk:
        nvfp4_moe_cubin._bufs = {
            "ei": torch.empty(expanded, dtype=torch.int32, device=device),
            "hist": torch.empty(2 * num_experts, dtype=torch.int32, device=device),
            "ps": torch.empty(1, dtype=torch.int32, device=device),
            "e2p": torch.empty(expanded, dtype=torch.int32, device=device),
            "p2e": torch.empty(max_padded, dtype=torch.int32, device=device),
            "p2t": torch.empty(max_padded, dtype=torch.int32, device=device),
            "ew": torch.empty(expanded, dtype=torch.bfloat16, device=device),
            "cb": torch.empty(max_ctas, dtype=torch.int32, device=device),
            "cm": torch.empty(max_ctas, dtype=torch.int32, device=device),
            "ne": torch.empty(1, dtype=torch.int32, device=device),
            "f1": torch.empty(max_padded, intermediate_size, dtype=torch.bfloat16, device=device),
            "f2": torch.empty(max_padded, hidden_size, dtype=torch.bfloat16, device=device),
        }
        nvfp4_moe_cubin._bk = bk

        # Setup fused call signature once
        lib.moe_cubin_fused_run.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ] + [ctypes.c_void_p] * 12
        lib.moe_cubin_fused_run.restype = ctypes.c_int

    b = nvfp4_moe_cubin._bufs

    # Fused call: routing → FC1 → FC2 (one ctypes call, minimal CPU sync)
    logits_bf16 = routing_logits.bfloat16().contiguous()
    torch.cuda.nvtx.range_push("fused_routing_fc1_fc2")
    rc = lib.moe_cubin_fused_run(
        fc1_config, fc2_config,
        _ptr(logits_bf16),
        num_tokens, num_experts, top_k, tile_n, routing_method_type,
        _ptr(gemm1_weights), _ptr(gemm1_weights_scale),
        _ptr(gemm2_weights), _ptr(gemm2_weights_scale),
        _ptr(hidden_states), _ptr(b["f2"]),
        _ptr(output1_scale_scalar), _ptr(output1_scale_gate_scalar), _ptr(output2_scale_scalar),
        _ptr(gemm1_bias), _ptr(gemm1_alpha), _ptr(gemm1_beta), _ptr(gemm1_clamp_limit),
        hidden_size, intermediate_size,
        _ptr(b["ei"]), _ptr(b["hist"]), _ptr(b["ps"]), _ptr(b["e2p"]),
        _ptr(b["p2e"]), _ptr(b["p2t"]), _ptr(b["ew"]),
        _ptr(b["cb"]), _ptr(b["cm"]), _ptr(b["ne"]),
        _ptr(b["f1"]),
        ctypes.c_void_p(stream))
    torch.cuda.nvtx.range_pop()

    if rc != 0:
        raise RuntimeError(f"fused_run failed (rc={rc})")

    # Finalize: fused CUDA kernel (cherry-picked from FlashInfer)
    torch.cuda.nvtx.range_push("finalize")
    if do_finalize:
        # Pre-allocate output buffer (reuse)
        if "out" not in b or b["out"].shape != (num_tokens, hidden_size):
            b["out"] = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

        if not hasattr(lib, '_fin_setup'):
            lib.moe_cubin_finalize.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.moe_cubin_finalize.restype = ctypes.c_int
            lib._fin_setup = True

        lib.moe_cubin_finalize(
            _ptr(b["f2"]), _ptr(b["out"]), _ptr(b["ew"]),
            _ptr(b["e2p"]), _ptr(b["ps"]),
            num_tokens, num_experts, top_k, hidden_size,
            ctypes.c_void_p(stream))
        torch.cuda.nvtx.range_pop()
        return [b["out"]]
    else:
        torch.cuda.nvtx.range_pop()
        return [b["f2"], b["ew"].float().reshape(num_tokens, top_k), b["e2p"]]
