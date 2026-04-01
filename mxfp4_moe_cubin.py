"""MxFP4 × BF16 MOE using trtllm-gen BatchedGemm exported cubins."""
import ctypes, math, sys, torch
from typing import Optional, List
from pathlib import Path

_lib = None
_nofuse_lib = None

def _is_verbose():
    """Check C-level verbose flag."""
    if _lib is not None and hasattr(_lib, 'moe_cubin_get_verbose'):
        return _lib.moe_cubin_get_verbose() > 0
    return True


def _get_sm_suffix():
    cc = torch.cuda.get_device_capability()
    sm = cc[0] * 10 + cc[1]
    return "sm103a" if sm >= 103 else "sm100a"


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    lib_name = f"libmoe_mxfp4_cubin_lib_{_get_sm_suffix()}.so"
    for d in [Path(__file__).parent, Path(__file__).parent / "build"]:
        p = d / lib_name
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            _setup_signatures(_lib)
            return _lib
    raise RuntimeError(f"{lib_name} not found. Build with cmake. "
                       f"GPU compute capability: {torch.cuda.get_device_capability()}")


def _load_nofuse_lib():
    """Load the nofuse .so (routeAct=true, fusedAct=false cubins)."""
    global _nofuse_lib
    if _nofuse_lib is not None:
        return _nofuse_lib
    lib_name = f"libmoe_mxfp4_nofuse_cubin_lib_{_get_sm_suffix()}.so"
    for d in [Path(__file__).parent, Path(__file__).parent / "build"]:
        p = d / lib_name
        if p.exists():
            _nofuse_lib = ctypes.CDLL(str(p))
            _setup_signatures(_nofuse_lib)
            return _nofuse_lib
    raise RuntimeError(f"{lib_name} not found. Build nofuse target with cmake.")

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
    lib.moe_cubin_find_valid_configs.argtypes = [
        ctypes.c_bool, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.moe_cubin_find_valid_configs.restype = ctypes.c_int
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

def _get_best_config(lib, is_fc1, tile_n, M, K, num_experts, num_tokens, meta, device,
                     top_k=None):
    cache_key = (is_fc1, tile_n, M, K, num_experts, num_tokens)
    if cache_key in _autotune_cache:
        return _autotune_cache[cache_key]
    ap = meta["max_padded"]
    w = torch.zeros(num_experts, M, K // 2, device=device, dtype=torch.uint8)
    w_sf = torch.ones(num_experts, M, K // 32, device=device, dtype=torch.uint8)
    inp = torch.zeros(num_tokens if is_fc1 else ap, K, device=device, dtype=torch.bfloat16)
    out = torch.zeros(ap, M // 2 if is_fc1 else M, device=device, dtype=torch.bfloat16)
    sc = torch.ones(num_experts, device=device, dtype=torch.float32)

    # Use worst-case batched_n matching fused_run, so autotune evaluates
    # the same cubin variant that fused_run will actually launch.
    if top_k is not None and top_k > 0:
        T_local = num_tokens // top_k
        max_ctas = lib.routing_get_max_ctas(T_local, top_k, num_experts, tile_n)
        cpe = (max_ctas + num_experts - 1) // num_experts
        bn_val = tile_n
        while bn_val < cpe * tile_n:
            bn_val *= 2
        bn = (ctypes.c_int * num_experts)(*([bn_val] * num_experts))
    else:
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
    logits_bf16 = routing_logits.bfloat16().contiguous()
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
        _ptr(logits_bf16), T, num_experts, top_k, tile_n,
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


def _select_tile_n_candidates(num_tokens, top_k, num_experts):
    primary = _select_tile_n(num_tokens, top_k, num_experts)
    candidates = [primary]
    avg = (num_tokens * top_k) / num_experts
    if primary == 8 and avg >= 0.5:
        candidates.append(16)
    return candidates


_best_tile_n_cache = {}


def mxfp4_moe_cubin(routing_logits, hidden_states, gemm1_weights, gemm1_weights_scale,
                     gemm2_weights, gemm2_weights_scale, num_experts, top_k, intermediate_size,
                     routing_bias=None, hidden_states_scale=None,
                     gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
                     gemm2_bias=None, output1_scale_scalar=None, output1_scale_gate_scalar=None,
                     output2_scale_scalar=None, do_finalize=True, routing_method_type=1):
    lib = _load_lib()
    T = hidden_states.shape[0]; H = hidden_states.shape[1]; device = hidden_states.device
    expanded = T * top_k; stream = torch.cuda.current_stream(device).cuda_stream
    fc1_M, fc1_K = 2 * intermediate_size, H
    fc2_M, fc2_K = H, intermediate_size
    shape_key = (T, num_experts, top_k, H, intermediate_size)

    if shape_key not in _best_tile_n_cache:
        import sys, os
        if os.environ.get("MXFP4_MULTI_TILE", "1") == "0":
            candidates = [_select_tile_n(T, top_k, num_experts)]
        else:
            candidates = _select_tile_n_candidates(T, top_k, num_experts)
        best_tile_n = candidates[0]
        best_total_us = float('inf')

        if not hasattr(lib, '_fused_setup'):
            lib.moe_cubin_fused_run.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 12
            lib.moe_cubin_fused_run.restype = ctypes.c_int
            lib._fused_setup = True

        for tn in candidates:
            ck1 = (True, tn, fc1_M, fc1_K, num_experts, expanded)
            ck2 = (False, tn, fc2_M, fc2_K, num_experts, expanded)
            if ck1 not in _autotune_cache or ck2 not in _autotune_cache:
                n_fc1 = lib.moe_cubin_find_valid_configs(
                    True, tn, fc1_M, tn, fc1_K, num_experts, expanded, (ctypes.c_int * 1)(), 1)
                n_fc2 = lib.moe_cubin_find_valid_configs(
                    False, tn, fc2_M, tn, fc2_K, num_experts, expanded, (ctypes.c_int * 1)(), 1)
                if n_fc1 == 0 or n_fc2 == 0:
                    if _is_verbose(): print(f"[multi-tile] tile_n={tn}: skip (FC1={n_fc1}, FC2={n_fc2})", file=sys.stderr)
                    continue
                meta = compute_routing_cuda(routing_logits, num_experts, top_k, tn, routing_method_type)
                _get_best_config(lib, True, tn, fc1_M, fc1_K, num_experts, expanded, meta, device, top_k=top_k)
                _get_best_config(lib, False, tn, fc2_M, fc2_K, num_experts, expanded, meta, device, top_k=top_k)
            if ck1 not in _autotune_cache or ck2 not in _autotune_cache:
                continue
            fc1_c = _autotune_cache[ck1]; fc2_c = _autotune_cache[ck2]
            mp = lib.routing_get_max_padded_tokens(T, top_k, num_experts, tn)
            mc = lib.routing_get_max_ctas(T, top_k, num_experts, tn)
            db = {
                "ei": torch.empty(expanded, dtype=torch.int32, device=device),
                "hist": torch.empty(2*num_experts, dtype=torch.int32, device=device),
                "ps": torch.empty(1, dtype=torch.int32, device=device),
                "e2p": torch.empty(expanded, dtype=torch.int32, device=device),
                "p2e": torch.empty(mp, dtype=torch.int32, device=device),
                "p2t": torch.empty(mp, dtype=torch.int32, device=device),
                "ew": torch.empty(expanded, dtype=torch.bfloat16, device=device),
                "cb": torch.empty(mc, dtype=torch.int32, device=device),
                "cm": torch.empty(mc, dtype=torch.int32, device=device),
                "ne": torch.empty(1, dtype=torch.int32, device=device),
            }
            f1 = torch.empty(mp, intermediate_size, dtype=torch.bfloat16, device=device)
            f2 = torch.empty(mp, H, dtype=torch.bfloat16, device=device)
            logits_bf16_tn = routing_logits.bfloat16().contiguous()
            fargs = (fc1_c, fc2_c, _ptr(logits_bf16_tn),
                T, num_experts, top_k, tn, routing_method_type,
                _ptr(gemm1_weights), _ptr(gemm1_weights_scale),
                _ptr(gemm2_weights), _ptr(gemm2_weights_scale),
                _ptr(hidden_states), _ptr(f2),
                _ptr(output1_scale_scalar), _ptr(output1_scale_gate_scalar), _ptr(output2_scale_scalar),
                _ptr(gemm1_bias), _ptr(gemm1_alpha), _ptr(gemm1_beta), _ptr(gemm1_clamp_limit),
                H, intermediate_size,
                _ptr(db["ei"]), _ptr(db["hist"]), _ptr(db["ps"]), _ptr(db["e2p"]),
                _ptr(db["p2e"]), _ptr(db["p2t"]), _ptr(db["ew"]),
                _ptr(db["cb"]), _ptr(db["cm"]), _ptr(db["ne"]), _ptr(f1),
                ctypes.c_void_p(stream))
            for _ in range(3):
                lib.moe_cubin_fused_run(*fargs)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10):
                lib.moe_cubin_fused_run(*fargs)
            e.record(); torch.cuda.synchronize()
            total_us = s.elapsed_time(e) / 10 * 1000
            if _is_verbose(): print(f"[multi-tile] tile_n={tn}: {total_us:.1f}us (FC1=config[{fc1_c}], FC2=config[{fc2_c}])",
                  file=sys.stderr)
            if total_us < best_total_us:
                best_total_us = total_us; best_tile_n = tn
        _best_tile_n_cache[shape_key] = best_tile_n
        if _is_verbose(): print(f"[multi-tile] BEST tile_n={best_tile_n} ({best_total_us:.1f}us)", file=sys.stderr)

    tile_n = _best_tile_n_cache[shape_key]
    ck1 = (True, tile_n, fc1_M, fc1_K, num_experts, expanded)
    ck2 = (False, tile_n, fc2_M, fc2_K, num_experts, expanded)
    if ck1 not in _autotune_cache or ck2 not in _autotune_cache:
        raise RuntimeError(f"No valid cubins for tile_n={tile_n} M={fc1_M}/{fc2_M} K={fc1_K}/{fc2_K} E={num_experts}")
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
    logits_bf16 = routing_logits.bfloat16().contiguous()
    torch.cuda.nvtx.range_push("mxfp4_fused")
    rc = lib.moe_cubin_fused_run(
        fc1_cfg, fc2_cfg, _ptr(logits_bf16),
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
