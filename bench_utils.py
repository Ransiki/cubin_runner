"""Benchmark-only utilities for E2E MoE experiments.

This module intentionally isolates benchmarking helpers from production paths.
It contains:
1) config translation (CLI network args -> per-GPU GEMM shapes)
2) deterministic balanced-routing builders and validation helpers
3) kineto timing utility used by bench_e2e
4) subprocess wrappers for isolated autotune/measurement
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass

import torch


def drain_cuda_errors() -> None:
    """Best-effort clear sticky async CUDA errors."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class GemmConfig:
    H: int
    I: int
    E: int
    K: int
    expanded: int
    active: int
    ep_size: int
    local_K: int
    local_T: int
    tokens_per_expert: list[int]


def make_gemm_config(H: int, moe_I: int, E: int, K: int, tp: int, ep: int, bs: int) -> GemmConfig:
    if K % ep != 0:
        raise ValueError(f"global top_k={K} must be divisible by ep={ep}")
    if E % ep != 0:
        raise ValueError(f"num_experts={E} must be divisible by ep={ep}")
    if moe_I % tp != 0:
        raise ValueError(f"moe_I={moe_I} must be divisible by tp={tp}")
    I = moe_I // tp
    local_E = E // ep
    local_K = K // ep
    expanded = (bs * K) // ep
    if bs * local_K != expanded:
        raise ValueError(f"inconsistent routing shape bs={bs} local_K={local_K} expanded={expanded}")
    base, rem = divmod(expanded, local_E)
    tpe = [base + 1] * rem + [base] * (local_E - rem)
    active = local_E if base > 0 else rem
    return GemmConfig(
        H=H,
        I=I,
        E=local_E,
        K=K,
        expanded=expanded,
        active=active,
        ep_size=ep,
        local_K=local_K,
        local_T=bs,
        tokens_per_expert=tpe,
    )


def auto_tile_n(cfg: GemmConfig) -> int:
    avg = cfg.expanded / max(1, cfg.E)
    return max(8, min(64, 2 ** math.ceil(math.log2(max(1.0, avg)))))


def rand_fp4_weight(shape: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)


def _median_kernel_us(run_fn, warmup: int = 3, iters: int = 9) -> float:
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_fn()
        e.record()
        torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) * 1000.0)
    vals.sort()
    return vals[len(vals) // 2]


def make_balanced_logits(T: int, E: int, K: int, device: str = "cuda") -> torch.Tensor:
    logits = torch.full((T, E), -1e4, device=device, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            expert_id = (t * K + k) % E
            logits[t, expert_id] = 100.0 - 0.1 * k
    return logits


def verify_balanced_routing(T: int, E: int, K: int, tile_n: int, device: str = "cuda") -> dict:
    from mxfp4_moe_cubin import compute_routing_cuda

    logits = make_balanced_logits(T, E, K, device=device)
    meta = compute_routing_cuda(logits, E, K, tile_n, routing_method=1)
    tpe = meta["tokens_per_expert"]
    expected = (T * K) // E
    remainder = (T * K) % E
    if remainder == 0:
        if not (tpe == expected).all():
            raise RuntimeError(f"routing not balanced, expect {expected}, got {tpe.unique().tolist()}")
    else:
        if not (((tpe == expected) | (tpe == expected + 1)).all()):
            raise RuntimeError(f"routing not near-balanced, got {tpe.unique().tolist()}")
    return meta


def bench_kineto(fn, kernel_name: str, num_tests: int = 20, flush_l2: bool = True, exclude: str | None = None) -> float:
    """Return average kernel time in seconds."""
    flush_size = int(8e9) // 4
    fn()
    sched = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=sched,
    ) as prof:
        for _step in range(2):
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_size, dtype=torch.int, device="cuda").zero_()
                fn()
            prof.step()

    total_us = 0.0
    kn = kernel_name.lower()
    ex = exclude.lower() if exclude else None
    for evt in prof.key_averages():
        key = evt.key.lower()
        if kn not in key:
            continue
        if ex and ex in key:
            continue
        if evt.self_device_time_total > 0:
            total_us += evt.self_device_time_total
    return total_us / num_tests / 1e6 if total_us > 0 else 0.0


WORKER_BOOTSTRAP = r"""
import json
import traceback
try:
    from bench_e2e import _worker_entry
    out = _worker_entry(json.loads(__BENCH_PAYLOAD__))
    print(json.dumps({"ok": True, "result": out}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e), "traceback": traceback.format_exc()}))
"""


def _parse_last_json_line(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    raise RuntimeError("No JSON line found in worker output")


def _run_worker_subprocess(payload: dict, timeout_s: int = 1800) -> dict:
    script = WORKER_BOOTSTRAP.replace("__BENCH_PAYLOAD__", json.dumps(payload).replace("\\", "\\\\").replace('"', '\\"'))
    cp = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    msg = _parse_last_json_line(out)
    if not msg.get("ok"):
        raise RuntimeError(msg.get("error", "worker failed"))
    return msg["result"]


def measure_fused_in_subprocess(cfg_dict: dict, num_tests: int = 20, flush_l2: bool = True) -> dict:
    return _run_worker_subprocess(
        {
            "op": "measure_fused",
            "cfg": cfg_dict,
            "num_tests": num_tests,
            "flush_l2": flush_l2,
        }
    )


def measure_nofuse_in_subprocess(cfg_dict: dict, num_tests: int = 20, flush_l2: bool = True) -> dict:
    return _run_worker_subprocess(
        {
            "op": "measure_nofuse",
            "cfg": cfg_dict,
            "num_tests": num_tests,
            "flush_l2": flush_l2,
        }
    )


def build_fused_runtime_state(cfg: GemmConfig, device: str = "cuda") -> dict:
    import mxfp4_moe_cubin as mxfp4_mod
    from mxfp4_moe_cubin import mxfp4_moe_cubin

    E, H, I, T, local_K = cfg.E, cfg.H, cfg.I, cfg.local_T, cfg.local_K
    expanded = T * local_K
    logits = make_balanced_logits(T, E, local_K, device)
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.01
    w1 = rand_fp4_weight((E, 2 * I, H // 2), device)
    s1 = torch.randint(0, 255, (E, 2 * I, H // 32), dtype=torch.uint8, device=device)
    w2 = rand_fp4_weight((E, H, I // 2), device)
    s2 = torch.randint(0, 255, (E, H, I // 32), dtype=torch.uint8, device=device)
    sc = torch.ones(E, dtype=torch.float32, device=device)

    # Keep fused autotune logic aligned with nofuse: exhaustive tile_n traversal.
    os.environ["MXFP4_MULTI_TILE"] = "1"
    mxfp4_mod._select_tile_n_candidates = lambda _t, _k, _e: [8, 16, 32, 64]

    mxfp4_moe_cubin(
        logits,
        hidden,
        w1,
        s1,
        w2,
        s2,
        E,
        local_K,
        I,
        output1_scale_scalar=sc,
        output1_scale_gate_scalar=sc,
        output2_scale_scalar=sc,
        do_finalize=True,
        routing_method_type=1,
    )
    torch.cuda.synchronize()
    # Read fused autotune selection from module caches for CSV traceability.
    shape_key = (T, E, local_K, H, I)
    tile_n = mxfp4_mod._best_tile_n_cache.get(shape_key)
    fc1_cfg = None
    fc2_cfg = None
    if tile_n is not None:
        ck1 = (True, tile_n, 2 * I, H, E, expanded)
        ck2 = (False, tile_n, H, I, E, expanded)
        fc1_cfg = mxfp4_mod._autotune_cache.get(ck1)
        fc2_cfg = mxfp4_mod._autotune_cache.get(ck2)
    return {
        "logits": logits,
        "hidden": hidden,
        "w1": w1,
        "s1": s1,
        "w2": w2,
        "s2": s2,
        "sc": sc,
        "tile_n": tile_n,
        "fc1_cfg": fc1_cfg,
        "fc2_cfg": fc2_cfg,
    }


def build_nofuse_runtime_state(cfg: GemmConfig, device: str = "cuda") -> dict:
    from mxfp4_moe_cubin import _load_lib, _load_nofuse_lib, _ptr, compute_routing_cuda

    E, H, I, T, local_K = cfg.E, cfg.H, cfg.I, cfg.local_T, cfg.local_K
    expanded = T * local_K
    logits = make_balanced_logits(T, E, local_K, device)
    tn_candidates = [8, 16, 32, 64]
    nofuse_lib = _load_nofuse_lib()
    fused_lib = _load_lib()
    nofuse_lib.moe_cubin_set_verbose(0)
    fused_lib.moe_cubin_set_verbose(0)
    sc = torch.ones(E, dtype=torch.float32, device=device)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    best = None

    for cur_tn in tn_candidates:
        meta = compute_routing_cuda(logits, E, local_K, cur_tn, routing_method=1)
        mp = meta["max_padded"]
        max_ctas = nofuse_lib.routing_get_max_ctas(T, local_K, E, cur_tn)
        cpe = (max_ctas + E - 1) // E
        bn_val = cur_tn
        while bn_val < cpe * cur_tn:
            bn_val *= 2
        bn_host = (ctypes.c_int * E)(*([bn_val] * E))
        hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.01
        w1 = rand_fp4_weight((E, 2 * I, H // 2), device)
        s1 = torch.randint(0, 255, (E, 2 * I, H // 32), dtype=torch.uint8, device=device)
        w2 = rand_fp4_weight((E, H, I // 2), device)
        s2 = torch.randint(0, 255, (E, H, I // 32), dtype=torch.uint8, device=device)
        fc1_out = torch.empty(mp, 2 * I, dtype=torch.bfloat16, device=device)
        fc2_in = torch.empty(mp, I, dtype=torch.bfloat16, device=device)
        fc2_out = torch.empty(mp, H, dtype=torch.bfloat16, device=device)

        fc1_cfg = nofuse_lib.moe_cubin_autotune(
            True, cur_tn, 2 * I, H, E, expanded,
            _ptr(w1), _ptr(s1), _ptr(hidden), _ptr(fc1_out),
            _ptr(sc), _ptr(sc), _ptr(meta["permuted_idx_to_token_idx"]),
            _ptr(meta["cta_idx_xy_to_batch_idx"]), _ptr(meta["cta_idx_xy_to_mn_limit"]),
            _ptr(meta["num_non_exiting_ctas"]), _ptr(meta["total_num_padded_tokens"]),
            bn_host, E, 5, 20, stream,
        )
        drain_cuda_errors()
        fc2_cfg = fused_lib.moe_cubin_autotune(
            False, cur_tn, H, I, E, expanded,
            _ptr(w2), _ptr(s2), _ptr(fc2_in), _ptr(fc2_out),
            _ptr(sc), _ptr(sc), None, _ptr(meta["cta_idx_xy_to_batch_idx"]),
            _ptr(meta["cta_idx_xy_to_mn_limit"]), _ptr(meta["num_non_exiting_ctas"]),
            _ptr(meta["total_num_padded_tokens"]), bn_host, E, 5, 20, stream,
        )
        drain_cuda_errors()
        if fc1_cfg < 0 or fc2_cfg < 0:
            continue

        def run_fc1():
            nofuse_lib.moe_cubin_run(
                fc1_cfg, True,
                _ptr(w1), _ptr(s1), _ptr(hidden), _ptr(fc1_out),
                _ptr(sc), _ptr(sc), None, None, None, None,
                H, I, E, expanded,
                _ptr(meta["permuted_idx_to_token_idx"]),
                _ptr(meta["cta_idx_xy_to_batch_idx"]),
                _ptr(meta["cta_idx_xy_to_mn_limit"]),
                _ptr(meta["num_non_exiting_ctas"]),
                _ptr(meta["total_num_padded_tokens"]),
                bn_host, E, stream,
            )

        def run_fc2():
            fused_lib.moe_cubin_run(
                fc2_cfg, False,
                _ptr(w2), _ptr(s2), _ptr(fc2_in), _ptr(fc2_out),
                _ptr(sc), _ptr(sc), None, None, None, None,
                H, I, E, expanded,
                None,
                _ptr(meta["cta_idx_xy_to_batch_idx"]),
                _ptr(meta["cta_idx_xy_to_mn_limit"]),
                _ptr(meta["num_non_exiting_ctas"]),
                _ptr(meta["total_num_padded_tokens"]),
                bn_host, E, stream,
            )

        fc1_us = _median_kernel_us(run_fc1)
        fc2_us = _median_kernel_us(run_fc2)
        total_us = fc1_us + fc2_us
        if best is None or total_us < best[5]:
            best = (cur_tn, fc1_cfg, fc2_cfg, bn_host, meta, total_us)

    if best is None:
        raise RuntimeError("nofuse autotune failed for all tile_n")
    tn, fc1_cfg, fc2_cfg, bn_host, meta, _ = best
    max_padded = meta["max_padded"]
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.01
    w1 = rand_fp4_weight((E, 2 * I, H // 2), device)
    s1 = torch.randint(0, 255, (E, 2 * I, H // 32), dtype=torch.uint8, device=device)
    w2 = rand_fp4_weight((E, H, I // 2), device)
    s2 = torch.randint(0, 255, (E, H, I // 32), dtype=torch.uint8, device=device)
    fc1_out = torch.empty(max_padded, 2 * I, dtype=torch.bfloat16, device=device)
    fc2_in = torch.empty(max_padded, I, dtype=torch.bfloat16, device=device)
    fc2_out = torch.empty(max_padded, H, dtype=torch.bfloat16, device=device)

    # Retune cfg indices on the final runtime buffers under selected tile_n.
    fc1_cfg = nofuse_lib.moe_cubin_autotune(
        True, tn, 2 * I, H, E, expanded,
        _ptr(w1), _ptr(s1), _ptr(hidden), _ptr(fc1_out),
        _ptr(sc), _ptr(sc), _ptr(meta["permuted_idx_to_token_idx"]),
        _ptr(meta["cta_idx_xy_to_batch_idx"]), _ptr(meta["cta_idx_xy_to_mn_limit"]),
        _ptr(meta["num_non_exiting_ctas"]), _ptr(meta["total_num_padded_tokens"]),
        bn_host, E, 5, 20, stream,
    )
    drain_cuda_errors()
    fc2_cfg = fused_lib.moe_cubin_autotune(
        False, tn, H, I, E, expanded,
        _ptr(w2), _ptr(s2), _ptr(fc2_in), _ptr(fc2_out),
        _ptr(sc), _ptr(sc), None, _ptr(meta["cta_idx_xy_to_batch_idx"]),
        _ptr(meta["cta_idx_xy_to_mn_limit"]), _ptr(meta["num_non_exiting_ctas"]),
        _ptr(meta["total_num_padded_tokens"]), bn_host, E, 5, 20, stream,
    )
    drain_cuda_errors()
    if fc1_cfg < 0 or fc2_cfg < 0:
        raise RuntimeError(f"nofuse retune failed on selected tile_n={tn}")

    return {
        "tn": tn,
        "fc1_cfg": fc1_cfg,
        "fc2_cfg": fc2_cfg,
        "expanded": expanded,
        "p2t": meta["permuted_idx_to_token_idx"],
        "cb": meta["cta_idx_xy_to_batch_idx"],
        "cm": meta["cta_idx_xy_to_mn_limit"],
        "ne": meta["num_non_exiting_ctas"],
        "tp_gpu": meta["total_num_padded_tokens"],
        "bn_host": bn_host,
        "hidden": hidden,
        "w1": w1,
        "s1": s1,
        "w2": w2,
        "s2": s2,
        "fc1_out": fc1_out,
        "fc2_in": fc2_in,
        "fc2_out": fc2_out,
        "sc": sc,
    }

