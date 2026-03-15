"""
Unified MOE Benchmark — NvFP4 and MxFP4 with per-kernel BW analysis.

Usage:
  python3 bench_moe.py                                    # MxFP4 TP8 BS=1,16,128
  python3 bench_moe.py --dtype nvfp4 --tp 1               # NvFP4 TP1
  python3 bench_moe.py --dtype mxfp4 --tp 8 --tokens 128  # single BS

nsys profiling:
  nsys profile --capture-range=cudaProfilerApi -o mxfp4_tp8 \\
    python3 bench_moe.py --nsys --dtype mxfp4 --tp 8 --tokens 128
"""
import sys, os, math, ctypes
sys.path.insert(0, os.path.dirname(__file__))
import torch
import argparse

parser = argparse.ArgumentParser(description="MOE Cubin Benchmark")
parser.add_argument("--model", default="dsv3", choices=["dsv3", "kimi_k2"])
parser.add_argument("--dtype", default="mxfp4", choices=["nvfp4", "mxfp4"])
parser.add_argument("--tp", type=int, default=8)
parser.add_argument("--tokens", default="1,16,128", help="comma-separated batch sizes")
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--nsys", action="store_true", help="enable cudaProfilerApi for nsys capture")
args = parser.parse_args()

MODELS = {
    "dsv3": {"hidden_size": 7168, "intermediate_size": 2048, "num_experts": 256, "top_k": 8},
    "kimi_k2": {"hidden_size": 7168, "intermediate_size": 2048, "num_experts": 384, "top_k": 8},
}
mcfg = MODELS[args.model]
H = mcfg["hidden_size"]
I = mcfg["intermediate_size"] // args.tp
E = mcfg["num_experts"]
K = mcfg["top_k"]
B200_BW = 8.0
token_list = [int(t) for t in args.tokens.split(",")]

if args.dtype == "nvfp4":
    from nvfp4_moe_cubin import nvfp4_moe_cubin as moe_fn, _load_lib, _autotune_cache
    from nvfp4_moe_cubin import compute_routing_cuda, _ptr, _host_int_array
    sf_block, sf_dtype = 16, torch.float8_e4m3fn
else:
    from mxfp4_moe_cubin import mxfp4_moe_cubin as moe_fn, _load_lib, _autotune_cache
    from mxfp4_moe_cubin import compute_routing_cuda, _ptr, _host_int_array
    sf_block, sf_dtype = 32, torch.uint8

lib = _load_lib()
device = "cuda"


def get_kernel_info(config_index):
    tM = ctypes.c_int(); tN = ctypes.c_int(); tK = ctypes.c_int()
    nS = ctypes.c_int(); nSM = ctypes.c_int(); isP = ctypes.c_int(); isU = ctypes.c_int()
    lib.moe_cubin_get_config_info(config_index,
        ctypes.byref(tM), ctypes.byref(tN), ctypes.byref(tK),
        ctypes.byref(nS), ctypes.byref(nSM), ctypes.byref(isP), ctypes.byref(isU))
    sched = "persistent" if isP.value else "static"
    u2 = " u2" if isU.value else ""
    return f"t{tM.value}x{tN.value}x{tK.value} s{nS.value}/{nSM.value} {sched}{u2}"


def get_cache_key(is_fc1, tile_n, M, K_dim):
    if args.dtype == "nvfp4":
        return (is_fc1, tile_n, M, K_dim, E)
    else:
        return (is_fc1, tile_n, M, K_dim, E, -1)


def bench_one(T):
    torch.manual_seed(42)
    expanded = T * K
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
    logits = torch.randn(T, E, device=device, dtype=torch.bfloat16).float()
    g1_w = torch.zeros(E, 2*I, H//2, device=device, dtype=torch.uint8)
    g1_sf = torch.ones(E, 2*I, H//sf_block, device=device, dtype=sf_dtype)
    g2_w = torch.zeros(E, H, I//2, device=device, dtype=torch.uint8)
    g2_sf = torch.ones(E, H, I//sf_block, device=device, dtype=sf_dtype)
    sc = torch.ones(E, device=device, dtype=torch.float32)
    kw = dict(routing_logits=logits, hidden_states=hidden,
              gemm1_weights=g1_w, gemm1_weights_scale=g1_sf,
              gemm2_weights=g2_w, gemm2_weights_scale=g2_sf,
              num_experts=E, top_k=K, intermediate_size=I,
              output1_scale_scalar=sc, output1_scale_gate_scalar=sc,
              output2_scale_scalar=sc, routing_method_type=1)

    # Warmup (triggers autotune on first call)
    print(f"  BS={T}: warmup ({args.warmup} iters, autotune on first call)...", flush=True)
    for _ in range(args.warmup):
        moe_fn(**kw); torch.cuda.synchronize()
    print(f"  BS={T}: warmup done. Benchmarking ({args.iters} iters)...", flush=True)

    # Pipeline timing
    if args.nsys:
        ctypes.CDLL('libcudart.so').cudaProfilerStart()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.iters):
        moe_fn(**kw)
    e.record(); torch.cuda.synchronize()
    if args.nsys:
        ctypes.CDLL('libcudart.so').cudaProfilerStop()
    pipeline_ms = s.elapsed_time(e) / args.iters

    # Get routing metadata for active expert count and per-kernel bench
    tile_n_primary = max(8, min(64, 2 ** math.ceil(math.log2(max(1, (T*K)/E)))))
    if args.dtype == "mxfp4" and hasattr(sys.modules.get('mxfp4_moe_cubin', None), '_best_tile_n_cache'):
        import mxfp4_moe_cubin as m
        shape_key = (T, E, K, H, I)
        tile_n = m._best_tile_n_cache.get(shape_key, tile_n_primary)
    else:
        tile_n = tile_n_primary

    meta = compute_routing_cuda(logits, E, K, tile_n, 1)
    active = (meta["tokens_per_expert"] > 0).sum().item()
    max_padded = meta["max_padded"]
    bn = _host_int_array(meta["batched_n"])
    stream_val = torch.cuda.current_stream(device).cuda_stream

    # Read autotune selection
    fc1_M, fc1_K = 2*I, H
    fc2_M, fc2_K = H, I
    if args.dtype == "nvfp4":
        fc1_idx = _autotune_cache.get((True, tile_n, fc1_M, fc1_K, E), -1)
        fc2_idx = _autotune_cache.get((False, tile_n, fc2_M, fc2_K, E), -1)
    else:
        fc1_idx = _autotune_cache.get((True, tile_n, fc1_M, fc1_K, E, expanded), -1)
        fc2_idx = _autotune_cache.get((False, tile_n, fc2_M, fc2_K, E, expanded), -1)

    fc1_info = get_kernel_info(fc1_idx) if fc1_idx >= 0 else "N/A"
    fc2_info = get_kernel_info(fc2_idx) if fc2_idx >= 0 else "N/A"

    # Per-kernel timing via moe_cubin_run
    if not hasattr(lib, '_run_setup'):
        lib.moe_cubin_run.argtypes = [
            ctypes.c_int, ctypes.c_bool,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        lib.moe_cubin_run.restype = ctypes.c_int
        lib._run_setup = True

    print(f"  BS={T}: pipeline={pipeline_ms:.3f}ms. Measuring per-kernel...", flush=True)

    fc1_us = fc2_us = 0
    if fc1_idx >= 0 and fc2_idx >= 0:
        fc1_out = torch.zeros(max_padded, I, device=device, dtype=torch.bfloat16)
        fc2_out = torch.zeros(max_padded, H, device=device, dtype=torch.bfloat16)

        def _bench_kernel(ci, is_fc1):
            if is_fc1:
                run_args = (ci, True, _ptr(g1_w), _ptr(g1_sf), _ptr(hidden), _ptr(fc1_out),
                    _ptr(sc), _ptr(sc), None, None, None, None, H, I, E, expanded,
                    _ptr(meta["permuted_idx_to_token_idx"]),
                    _ptr(meta["cta_idx_xy_to_batch_idx"]), _ptr(meta["cta_idx_xy_to_mn_limit"]),
                    _ptr(meta["num_non_exiting_ctas"]), _ptr(meta["total_num_padded_tokens"]),
                    bn, E, ctypes.c_void_p(stream_val))
            else:
                run_args = (ci, False, _ptr(g2_w), _ptr(g2_sf), _ptr(fc1_out), _ptr(fc2_out),
                    _ptr(sc), None, None, None, None, None, H, I, E, expanded,
                    None,
                    _ptr(meta["cta_idx_xy_to_batch_idx"]), _ptr(meta["cta_idx_xy_to_mn_limit"]),
                    _ptr(meta["num_non_exiting_ctas"]), _ptr(meta["total_num_padded_tokens"]),
                    bn, E, ctypes.c_void_p(stream_val))
            for _ in range(5):
                lib.moe_cubin_run(*run_args)
            torch.cuda.synchronize()
            ks = torch.cuda.Event(enable_timing=True); ke = torch.cuda.Event(enable_timing=True)
            ks.record()
            for _ in range(args.iters):
                lib.moe_cubin_run(*run_args)
            ke.record(); torch.cuda.synchronize()
            return ks.elapsed_time(ke) / args.iters * 1000

        fc1_us = _bench_kernel(fc1_idx, True)
        fc2_us = _bench_kernel(fc2_idx, False)

    # BW calculation
    fc1_w_bytes = active * 2*I * (H//2) + active * 2*I * (H//sf_block)
    fc1_act_bytes = expanded * H * 2 + expanded * I * 2
    fc1_bw = ((fc1_w_bytes + fc1_act_bytes) / 1e12) / (fc1_us / 1e6) if fc1_us > 0 else 0

    fc2_w_bytes = active * H * (I//2) + active * H * (I//sf_block)
    fc2_act_bytes = expanded * I * 2 + expanded * H * 2
    fc2_bw = ((fc2_w_bytes + fc2_act_bytes) / 1e12) / (fc2_us / 1e6) if fc2_us > 0 else 0

    return {
        "T": T, "pipeline_ms": pipeline_ms, "active": active, "tile_n": tile_n,
        "fc1_us": fc1_us, "fc1_bw": fc1_bw, "fc1_info": fc1_info,
        "fc2_us": fc2_us, "fc2_bw": fc2_bw, "fc2_info": fc2_info,
    }


def main():
    gpu_name = torch.cuda.get_device_name()
    sm_count = lib.moe_cubin_get_sm_count()

    print(f"\n{'='*70}")
    print(f"  {args.model.upper()} TP{args.tp} — {args.dtype.upper()} MOE Benchmark")
    print(f"{'='*70}")
    print(f"  GPU:     {gpu_name} ({sm_count} SMs)")
    print(f"  Model:   H={H} I={I} E={E} K={K}")
    print(f"  Iters:   {args.iters} (warmup={args.warmup})")
    print()

    for T in token_list:
        r = bench_one(T)
        fc1_pct = r["fc1_bw"] / B200_BW * 100 if r["fc1_bw"] > 0 else 0
        fc2_pct = r["fc2_bw"] / B200_BW * 100 if r["fc2_bw"] > 0 else 0
        print(f"  BS={r['T']:>4}:")
        print(f"    Pipeline:   {r['pipeline_ms']:.3f}ms")
        print(f"    FC1: {r['fc1_us']:>7.1f}us  BW={r['fc1_bw']:.2f}T/s ({fc1_pct:.0f}%)  {r['fc1_info']}")
        print(f"    FC2: {r['fc2_us']:>7.1f}us  BW={r['fc2_bw']:.2f}T/s ({fc2_pct:.0f}%)  {r['fc2_info']}")
        print(f"    Active: {r['active']}/{E}  tile_n={r['tile_n']}")
        print()


if __name__ == "__main__":
    main()
