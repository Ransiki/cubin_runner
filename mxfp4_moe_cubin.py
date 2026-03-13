"""MxFP4 × BF16 MOE using trtllm-gen BatchedGemm exported cubins."""
import ctypes, math, torch
from typing import Optional, List
from pathlib import Path

_lib = None


def _get_mxfp4_lib_name():
    """Select the correct MxFP4 .so based on GPU compute capability."""
    cc = torch.cuda.get_device_capability()
    sm = cc[0] * 10 + cc[1]
    if sm >= 103:
        return "libmoe_mxfp4_cubin_lib_sm103a.so"
    else:
        return "libmoe_mxfp4_cubin_lib_sm100a.so"


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    lib_name = _get_mxfp4_lib_name()
    for d in [Path(__file__).parent, Path(__file__).parent / "build"]:
        p = d / lib_name
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            _setup_signatures(_lib)
            return _lib
    raise RuntimeError(f"{lib_name} not found. Build with cmake. "
                       f"GPU compute capability: {torch.cuda.get_device_capability()}")

def _setup_signatures(lib):
    lib.moe_cubin_autotune.argtypes = [
        ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.moe_cubin_autotune.restype = ctypes.c_int
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

_autotune_cache = {}

def _get_best_config(lib, is_fc1, tile_n, M, K, num_experts, num_tokens, meta, device):
    cache_key = (is_fc1, tile_n, M, K, num_experts, num_tokens)
    if cache_key in _autotune_cache:
        return _autotune_cache[cache_key]
    ap = meta["max_padded"]
    w = torch.zeros(num_experts, M, K // 2, device=device, dtype=torch.uint8)
    w_sf = torch.ones(num_experts, M, K // 16, device=device, dtype=torch.float8_e4m3fn)
    inp = torch.zeros(num_tokens if is_fc1 else ap, K, device=device, dtype=torch.bfloat16)
    out = torch.zeros(ap, M // 2 if is_fc1 else M, device=device, dtype=torch.bfloat16)
    sc = torch.ones(num_experts, device=device, dtype=torch.float32)
    bn = _host_int_array(meta["batched_n"])
    stream = torch.cuda.current_stream(device).cuda_stream
    best = lib.moe_cubin_autotune(
        is_fc1, tile_n, M, K, num_experts, num_tokens,
        _ptr(w), _ptr(w_sf), _ptr(inp), _ptr(out), _ptr(sc), _ptr(sc),
        _ptr(meta["permuted_idx_to_token_idx"]), _ptr(meta["cta_idx_xy_to_batch_idx"]),
        _ptr(meta["cta_idx_xy_to_mn_limit"]), _ptr(meta["num_non_exiting_ctas"]),
        _ptr(meta["total_num_padded_tokens"]), bn, num_experts, 5, 20,
        ctypes.c_void_p(stream))
    if best < 0:
        raise RuntimeError(f"Autotune failed")
    _autotune_cache[cache_key] = best
    return best

def compute_routing_cuda(routing_logits, num_experts, top_k, tile_n, routing_method=1):
    lib = _load_lib()
    device = routing_logits.device
    T = routing_logits.shape[0]
    expanded = T * top_k
    max_padded = lib.routing_get_max_padded_tokens(T, top_k, num_experts, tile_n)
    max_ctas = lib.routing_get_max_ctas(T, top_k, num_experts, tile_n)
    ei = torch.zeros(expanded, dtype=torch.int32, device=device)
    hist = torch.zeros(2 * num_experts, dtype=torch.int32, device=device)
    ps = torch.zeros(1, dtype=torch.int32, device=device)
    e2p = torch.full((expanded,), -1, dtype=torch.int32, device=device)
    p2e = torch.full((max_padded,), -1, dtype=torch.int32, device=device)
    p2t = torch.full((max_padded,), -1, dtype=torch.int32, device=device)
    ew = torch.zeros(expanded, dtype=torch.bfloat16, device=device)
    cb = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    cm = torch.zeros(max_ctas, dtype=torch.int32, device=device)
    ne = torch.zeros(1, dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream
    rc = lib.routing_renormalize_run(
        _ptr(routing_logits.contiguous()), T, num_experts, top_k, tile_n,
        0, num_experts, routing_method,
        _ptr(ei), _ptr(hist), _ptr(ps), _ptr(e2p), _ptr(p2e), _ptr(p2t),
        _ptr(ew), _ptr(cb), _ptr(cm), _ptr(ne), ctypes.c_void_p(stream))
    torch.cuda.synchronize()
    if rc != 0:
        raise RuntimeError(f"routing failed rc={rc}")
    tpe = torch.zeros(num_experts, dtype=torch.int32, device=device)
    ac = ne[0].item()
    if ac > 0:
        cpe = torch.bincount(cb[:ac].long(), minlength=num_experts).to(torch.int32)
        eco = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        eco[1:] = torch.cumsum(cpe, dim=0)
        has = cpe > 0
        last = eco[1:] - 1
        start = eco[:num_experts] * tile_n
        tpe = torch.where(has, cm[last.long().clamp(min=0)] - start, torch.zeros_like(cpe))
    return {"topk_weights": ew.float().reshape(T, top_k), "tokens_per_expert": tpe,
            "permuted_idx_to_token_idx": p2t, "expanded_idx_to_permuted": e2p,
            "cta_idx_xy_to_batch_idx": cb, "cta_idx_xy_to_mn_limit": cm,
            "num_non_exiting_ctas": ne, "total_num_padded_tokens": ps,
            "batched_n": tpe.clone(), "total_padded": ps[0].item(), "max_padded": max_padded}

def _select_tile_n(num_tokens, top_k, num_experts):
    avg = (num_tokens * top_k) / num_experts
    return max(8, min(64, 2 ** math.ceil(math.log2(max(1, avg)))))

def mxfp4_moe_cubin(routing_logits, hidden_states, gemm1_weights, gemm1_weights_scale,
                     gemm2_weights, gemm2_weights_scale, num_experts, top_k, intermediate_size,
                     routing_bias=None, hidden_states_scale=None,
                     gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
                     gemm2_bias=None, output1_scale_scalar=None, output1_scale_gate_scalar=None,
                     output2_scale_scalar=None, do_finalize=True, routing_method_type=1):
    lib = _load_lib()
    T = hidden_states.shape[0]; H = hidden_states.shape[1]; device = hidden_states.device
    tile_n = _select_tile_n(T, top_k, num_experts)
    expanded = T * top_k; stream = torch.cuda.current_stream(device).cuda_stream
    fc1_M, fc1_K = 2 * intermediate_size, H
    fc2_M, fc2_K = H, intermediate_size
    ck1 = (True, tile_n, fc1_M, fc1_K, num_experts, expanded)
    ck2 = (False, tile_n, fc2_M, fc2_K, num_experts, expanded)
    if ck1 not in _autotune_cache or ck2 not in _autotune_cache:
        meta = compute_routing_cuda(routing_logits, num_experts, top_k, tile_n, routing_method_type)
        _get_best_config(lib, True, tile_n, fc1_M, fc1_K, num_experts, expanded, meta, device)
        _get_best_config(lib, False, tile_n, fc2_M, fc2_K, num_experts, expanded, meta, device)
    fc1_cfg = _autotune_cache[ck1]; fc2_cfg = _autotune_cache[ck2]
    max_padded = lib.routing_get_max_padded_tokens(T, top_k, num_experts, tile_n)
    max_ctas = lib.routing_get_max_ctas(T, top_k, num_experts, tile_n)
    bk = (T, num_experts, top_k, tile_n, H, intermediate_size)
    if not hasattr(mxfp4_moe_cubin, '_bufs') or mxfp4_moe_cubin._bk != bk:
        mxfp4_moe_cubin._bufs = {
            "ei": torch.empty(expanded, dtype=torch.int32, device=device),
            "hist": torch.empty(2*num_experts, dtype=torch.int32, device=device),
            "ps": torch.empty(1, dtype=torch.int32, device=device),
            "e2p": torch.empty(expanded, dtype=torch.int32, device=device),
            "p2e": torch.empty(max_padded, dtype=torch.int32, device=device),
            "p2t": torch.empty(max_padded, dtype=torch.int32, device=device),
            "ew": torch.empty(expanded, dtype=torch.bfloat16, device=device),
            "cb": torch.empty(max_ctas, dtype=torch.int32, device=device),
            "cm": torch.empty(max_ctas, dtype=torch.int32, device=device),
            "ne": torch.empty(1, dtype=torch.int32, device=device),
            "f1": torch.empty(max_padded, intermediate_size, dtype=torch.bfloat16, device=device),
            "f2": torch.empty(max_padded, H, dtype=torch.bfloat16, device=device),
        }
        mxfp4_moe_cubin._bk = bk
        lib.moe_cubin_fused_run.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 12
        lib.moe_cubin_fused_run.restype = ctypes.c_int
    b = mxfp4_moe_cubin._bufs
    torch.cuda.nvtx.range_push("mxfp4_fused")
    rc = lib.moe_cubin_fused_run(
        fc1_cfg, fc2_cfg, _ptr(routing_logits.contiguous()),
        T, num_experts, top_k, tile_n, routing_method_type,
        _ptr(gemm1_weights), _ptr(gemm1_weights_scale),
        _ptr(gemm2_weights), _ptr(gemm2_weights_scale),
        _ptr(hidden_states), _ptr(b["f2"]),
        _ptr(output1_scale_scalar), _ptr(output1_scale_gate_scalar), _ptr(output2_scale_scalar),
        _ptr(gemm1_bias), _ptr(gemm1_alpha), _ptr(gemm1_beta), _ptr(gemm1_clamp_limit),
        H, intermediate_size,
        _ptr(b["ei"]), _ptr(b["hist"]), _ptr(b["ps"]), _ptr(b["e2p"]),
        _ptr(b["p2e"]), _ptr(b["p2t"]), _ptr(b["ew"]),
        _ptr(b["cb"]), _ptr(b["cm"]), _ptr(b["ne"]), _ptr(b["f1"]),
        ctypes.c_void_p(stream))
    torch.cuda.nvtx.range_pop()
    if rc != 0:
        raise RuntimeError(f"fused_run failed rc={rc}")
    torch.cuda.nvtx.range_push("mxfp4_finalize")
    if do_finalize:
        if "out" not in b or b["out"].shape != (T, H):
            b["out"] = torch.empty(T, H, dtype=torch.bfloat16, device=device)
        if not hasattr(lib, '_fin_setup'):
            lib.moe_cubin_finalize.argtypes = [
                ctypes.c_void_p]*5 + [ctypes.c_int]*4 + [ctypes.c_void_p]
            lib.moe_cubin_finalize.restype = ctypes.c_int
            lib._fin_setup = True
        lib.moe_cubin_finalize(_ptr(b["f2"]), _ptr(b["out"]), _ptr(b["ew"]),
            _ptr(b["e2p"]), _ptr(b["ps"]), T, num_experts, top_k, H,
            ctypes.c_void_p(stream))
        torch.cuda.nvtx.range_pop()
        return [b["out"]]
    torch.cuda.nvtx.range_pop()
    return [b["f2"], b["ew"].float().reshape(T, top_k), b["e2p"]]
