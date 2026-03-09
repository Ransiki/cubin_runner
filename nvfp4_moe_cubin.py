"""
NvFP4 × BF16 MOE using trtllm-gen BatchedGemm exported cubins.

Pipeline:
  1. Routing:  FlashInfer CUDA kernel → topK experts + permutation metadata
  2. FC1:      trtllm-gen cubin (fused permute + GEMM + SwiGLU)
  3. FC2:      trtllm-gen cubin (plain batched GEMM)
  4. Finalize: PyTorch (unpermute + weighted sum)

Usage:
    from nvfp4_moe_cubin import nvfp4_moe_cubin
    output = nvfp4_moe_cubin(routing_logits, hidden_states,
                              gemm1_weights, gemm1_weights_scale,
                              gemm2_weights, gemm2_weights_scale,
                              num_experts=E, top_k=K, intermediate_size=I,
                              output1_scale_scalar=scale_c,
                              output1_scale_gate_scalar=scale_g,
                              output2_scale_scalar=scale_fc2)
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
        Path(__file__).parent.parent / "trtllm-gen" / "build" / "moe_runner" / "libmoe_cubin_lib.so",
    ]:
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            _setup_signatures(_lib)
            return _lib
    raise RuntimeError("libmoe_cubin_lib.so not found. Build with cmake.")


def _setup_signatures(lib):
    """Declare ctypes function signatures for the C API."""
    lib.moe_cubin_init.argtypes = [ctypes.c_int] * 5
    lib.moe_cubin_init.restype = ctypes.c_int

    _fc1_args = [ctypes.c_void_p] * 4 + [ctypes.c_void_p] * 6 + \
                [ctypes.c_int] * 4 + [ctypes.c_void_p] * 5 + \
                [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    lib.moe_cubin_fc1_run.argtypes = _fc1_args
    lib.moe_cubin_fc1_run.restype = ctypes.c_int

    _fc2_args = [ctypes.c_void_p] * 4 + [ctypes.c_void_p] + \
                [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4 + \
                [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    lib.moe_cubin_fc2_run.argtypes = _fc2_args
    lib.moe_cubin_fc2_run.restype = ctypes.c_int

    lib.routing_renormalize_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.routing_renormalize_run.restype = ctypes.c_int32


def _ptr(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())

def _host_int_array(t):
    cpu = t.cpu().to(torch.int32).contiguous().numpy()
    return (ctypes.c_int * len(cpu))(*cpu)


# ═════════════════════════════════════════════════════════════════════════════
# Routing — FlashInfer CUDA kernel
# ═════════════════════════════════════════════════════════════════════════════

def compute_routing_cuda(
    routing_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    tile_n: int,
    routing_method: int = 1,
) -> dict:
    """
    Run FlashInfer's CUDA routing kernel. Computes topK expert selection and
    all metadata buffers needed by the FC1/FC2 cubins in a single kernel launch.

    Args:
        routing_logits: [T, E] bf16 or float32 (GPU)
        routing_method: 0=softmax→topK, 1=topK→softmax(renormalize)

    Returns dict:
        topk_weights              [T, K] float32 — routing weights
        tokens_per_expert         [E]   int32 — actual tokens per expert
        permuted_idx_to_token_idx [max_padded] int32 — FC1 cubin reads token[i] from here
        expanded_idx_to_permuted  [T*K] int32 — for finalize unpermute
        cta_idx_xy_to_batch_idx   [max_ctas] int32 — CTA→expert mapping
        cta_idx_xy_to_mn_limit    [max_ctas] int32 — cumulative token limits
        num_non_exiting_ctas      [1] int32
        total_num_padded_tokens   [1] int32
        batched_n                 [E] int32 — same as tokens_per_expert
        total_padded              int — scalar for buffer allocation
        max_padded                int — max possible padded tokens
    """
    lib = _load_lib()
    device = routing_logits.device
    T = routing_logits.shape[0]
    expanded = T * top_k
    max_padded = lib.routing_get_max_padded_tokens(T, top_k, num_experts, tile_n)
    max_ctas = lib.routing_get_max_ctas(T, top_k, num_experts, tile_n)

    # Allocate output buffers for routing kernel
    expert_indexes     = torch.zeros(expanded, dtype=torch.int32, device=device)
    expert_count_hist  = torch.zeros(2 * num_experts, dtype=torch.int32, device=device)
    permuted_idx_size  = torch.zeros(1, dtype=torch.int32, device=device)
    expanded_to_perm   = torch.full((expanded,), -1, dtype=torch.int32, device=device)
    perm_to_expanded   = torch.full((max_padded,), -1, dtype=torch.int32, device=device)
    perm_to_token      = torch.full((max_padded,), -1, dtype=torch.int32, device=device)
    expert_weights     = torch.zeros(expanded, dtype=torch.bfloat16, device=device)
    cta_to_batch       = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    cta_to_mn          = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    num_non_exit       = torch.zeros(1, dtype=torch.int32, device=device)

    logits_in = routing_logits.contiguous()
    stream = torch.cuda.current_stream(device).cuda_stream

    rc = lib.routing_renormalize_run(
        _ptr(logits_in), T, num_experts, top_k, tile_n, 0, num_experts,
        routing_method,
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
        expert_cta_offset = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        expert_cta_offset[1:] = torch.cumsum(ctas_per_expert, dim=0)
        has_ctas = ctas_per_expert > 0
        last_cta_idx = expert_cta_offset[1:] - 1
        expert_padded_start = expert_cta_offset[:num_experts] * tile_n
        last_mn = cm[last_cta_idx.long().clamp(min=0)]
        tokens_per_expert = torch.where(has_ctas, last_mn - expert_padded_start,
                                         torch.zeros_like(last_mn))

    topk_weights = expert_weights.float().reshape(T, top_k)

    return {
        "topk_weights": topk_weights,
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
# Finalize — PyTorch
# ═════════════════════════════════════════════════════════════════════════════

def finalize(fc2_output, topk_weights, expanded_idx_to_permuted,
             num_tokens, hidden_size, top_k):
    """Unpermute expert outputs and weighted-sum across top_k experts."""
    gathered = fc2_output[expanded_idx_to_permuted.long()].float()
    gathered = gathered.reshape(num_tokens, top_k, hidden_size)
    return (gathered * topk_weights.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


# ═════════════════════════════════════════════════════════════════════════════
# Main API
# ═════════════════════════════════════════════════════════════════════════════

_initialized_tile_n = None

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
    NvFP4 × BF16 MOE forward pass using trtllm-gen exported cubins.

    Weight preparation (must be done beforehand, e.g. with TRT-LLM):
      1. fp4_quantize(weights, global_sf, sf_vec_size=16, use_ue8m0=False, is_sf_swizzled=False)
      2. reorder_rows_for_gated_act_gemm(fp4_weights)  [FC1 only]
      3. shuffle_matrix_a(weights, epilogue_tile_m=128)
      4. shuffle_matrix_sf_a(scales, epilogue_tile_m=128)

    Scale factors:
      scale_c_fc1     = c_global_sf / gemm1_global_sf / hidden_states_global_sf
      scale_gate_fc1  = 1.0 / gemm1_global_sf / hidden_states_global_sf
      scale_c_fc2     = 1.0 / c_global_sf / gemm2_global_sf
      (For W4A16 with bf16 activations: hidden_states_global_sf=1.0, c_global_sf=1.0)

    Returns:
      [output]  if do_finalize=True,  output shape [T, H] bf16
      [fc2_out, topk_weights, expanded_idx_to_permuted]  otherwise
    """
    global _initialized_tile_n
    lib = _load_lib()

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    device = hidden_states.device

    tile_n = _select_tile_n(num_tokens, top_k, num_experts)
    if _initialized_tile_n != tile_n:
        est = max(tile_n, (num_tokens * top_k + num_experts - 1) // num_experts)
        rc = lib.moe_cubin_init(tile_n, hidden_size, intermediate_size, num_experts, est)
        if rc != 0:
            raise RuntimeError(f"moe_cubin_init failed (rc={rc})")
        _initialized_tile_n = tile_n

    # Step 1: CUDA routing
    meta = compute_routing_cuda(routing_logits, num_experts, top_k, tile_n, routing_method_type)
    alloc_padded = meta["max_padded"]
    batched_n_host = _host_int_array(meta["batched_n"])
    stream = torch.cuda.current_stream(device).cuda_stream

    # Step 2: FC1 cubin
    fc1_output = torch.zeros(alloc_padded, intermediate_size, dtype=torch.bfloat16, device=device)
    rc = lib.moe_cubin_fc1_run(
        _ptr(gemm1_weights), _ptr(gemm1_weights_scale),
        _ptr(hidden_states), _ptr(fc1_output),
        _ptr(output1_scale_scalar), _ptr(output1_scale_gate_scalar),
        _ptr(gemm1_bias), _ptr(gemm1_alpha), _ptr(gemm1_beta), _ptr(gemm1_clamp_limit),
        hidden_size, intermediate_size, num_experts, num_tokens * top_k,
        _ptr(meta["permuted_idx_to_token_idx"]),
        _ptr(meta["cta_idx_xy_to_batch_idx"]),
        _ptr(meta["cta_idx_xy_to_mn_limit"]),
        _ptr(meta["num_non_exiting_ctas"]),
        _ptr(meta["total_num_padded_tokens"]),
        batched_n_host, num_experts,
        ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"FC1 cubin failed (rc={rc})")

    # Step 3: FC2 cubin
    fc2_output = torch.zeros(alloc_padded, hidden_size, dtype=torch.bfloat16, device=device)
    rc = lib.moe_cubin_fc2_run(
        _ptr(gemm2_weights), _ptr(gemm2_weights_scale),
        _ptr(fc1_output), _ptr(fc2_output),
        _ptr(output2_scale_scalar),
        hidden_size, intermediate_size, num_experts, num_tokens * top_k,
        _ptr(meta["cta_idx_xy_to_batch_idx"]),
        _ptr(meta["cta_idx_xy_to_mn_limit"]),
        _ptr(meta["num_non_exiting_ctas"]),
        _ptr(meta["total_num_padded_tokens"]),
        batched_n_host, num_experts,
        ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"FC2 cubin failed (rc={rc})")

    # Step 4: Finalize
    if do_finalize:
        output = finalize(fc2_output, meta["topk_weights"],
                          meta["expanded_idx_to_permuted"],
                          num_tokens, hidden_size, top_k)
        return [output]
    else:
        return [fc2_output, meta["topk_weights"], meta["expanded_idx_to_permuted"]]
