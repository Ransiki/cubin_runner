"""
Unified MOE Benchmark — NvFP4 and MxFP4, end-to-end pipeline latency.

Usage:
  python3 bench_moe.py                                    # MxFP4 TP8 BS=1,16,128
  python3 bench_moe.py --dtype nvfp4 --tp 1               # NvFP4 TP1
  python3 bench_moe.py --dtype mxfp4 --tp 8 --tokens 128  # single BS
  python3 bench_moe.py --balanced --tokens 128             # perfect expert balance

nsys profiling (use nsys_analyze.sh for per-kernel bandwidth):
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
parser.add_argument("--balanced", action="store_true",
                    help="use perfectly balanced routing (each expert gets exactly expanded/E tokens)")
parser.add_argument("--force-fc1", type=int, default=-1, metavar="IDX",
                    help="force a specific cubin config index for FC1 (bypass autotune)")
parser.add_argument("--force-fc2", type=int, default=-1, metavar="IDX",
                    help="force a specific cubin config index for FC2 (bypass autotune)")
parser.add_argument("--list-configs", action="store_true",
                    help="list all valid cubin configs for the first token count, then exit")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="show autotune logs (default: suppressed)")
parser.add_argument("--cuda-graph", action="store_true",
                    help="use CUDA Graph capture+replay to eliminate launch overhead")
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
lib.moe_cubin_set_verbose(1 if args.verbose else 0)
device = "cuda"


def get_kernel_info_ext(config_index):
    """Extended config info including clusterDimX, splitK, mmaM."""
    tM = ctypes.c_int(); tN = ctypes.c_int(); tK = ctypes.c_int()
    nS = ctypes.c_int(); nSM = ctypes.c_int(); isP = ctypes.c_int(); isU = ctypes.c_int()
    cX = ctypes.c_int(); sK = ctypes.c_int(); mM = ctypes.c_int()
    lib.moe_cubin_get_config_info_ext(config_index,
        ctypes.byref(tM), ctypes.byref(tN), ctypes.byref(tK),
        ctypes.byref(nS), ctypes.byref(nSM), ctypes.byref(isP), ctypes.byref(isU),
        ctypes.byref(cX), ctypes.byref(sK), ctypes.byref(mM))
    sched = "persistent" if isP.value else "static"
    u2 = " u2" if isU.value else ""
    cdx = f" c{cX.value}" if cX.value > 1 else ""
    sk = f" splitK={sK.value}" if sK.value > 1 else ""
    return (f"t{tM.value}x{tN.value}x{tK.value} s{nS.value}/{nSM.value} "
            f"mma{mM.value} {sched}{u2}{cdx}{sk}")


def make_balanced_logits(T, E, K):
    """Create logits that produce perfectly balanced routing (each expert gets T*K/E tokens)."""
    logits = torch.full((T, E), -1e4, device=device, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            expert_id = (t * K + k) % E
            logits[t, expert_id] = 100.0 - k * 0.1
    return logits


def list_valid_configs(T, is_fc1, tile_n, M, K_dim):
    """List all valid cubin configs for a shape."""
    expanded = T * K
    buf = (ctypes.c_int * 256)()
    n = lib.moe_cubin_find_valid_configs(is_fc1, tile_n, M, tile_n, K_dim, E, expanded, buf, 256)
    configs = []
    for i in range(n):
        idx = buf[i]
        info = get_kernel_info_ext(idx)
        configs.append((idx, info))
    return configs


def bench_one(T):
    torch.manual_seed(42)
    expanded = T * K
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
    if args.balanced:
        logits = make_balanced_logits(T, E, K)
    else:
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

    # End-to-end pipeline timing
    if args.nsys:
        ctypes.CDLL('libcudart.so').cudaProfilerStart()

    if args.cuda_graph:
        # CUDA Graph: capture once, replay many times
        # All buffers already allocated by warmup; pointers are stable
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe_fn(**kw)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            graph.replay()
        e.record(); torch.cuda.synchronize()
    else:
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            moe_fn(**kw)
        e.record(); torch.cuda.synchronize()

    if args.nsys:
        ctypes.CDLL('libcudart.so').cudaProfilerStop()
    pipeline_us = s.elapsed_time(e) / args.iters * 1000  # microseconds

    # Get active expert count from routing
    tile_n_primary = max(8, min(64, 2 ** math.ceil(math.log2(max(1, (T*K)/E)))))
    if args.dtype == "mxfp4" and hasattr(sys.modules.get('mxfp4_moe_cubin', None), '_best_tile_n_cache'):
        import mxfp4_moe_cubin as m
        shape_key = (T, E, K, H, I)
        tile_n = m._best_tile_n_cache.get(shape_key, tile_n_primary)
    else:
        tile_n = tile_n_primary

    meta = compute_routing_cuda(logits, E, K, tile_n, 1)
    active = (meta["tokens_per_expert"] > 0).sum().item()

    # Read autotune selection
    fc1_M, fc1_K = 2*I, H
    fc2_M, fc2_K = H, I
    if args.dtype == "nvfp4":
        fc1_idx = _autotune_cache.get((True, tile_n, fc1_M, fc1_K, E), -1)
        fc2_idx = _autotune_cache.get((False, tile_n, fc2_M, fc2_K, E), -1)
    else:
        fc1_idx = _autotune_cache.get((True, tile_n, fc1_M, fc1_K, E, expanded), -1)
        fc2_idx = _autotune_cache.get((False, tile_n, fc2_M, fc2_K, E, expanded), -1)
    if args.force_fc1 >= 0:
        fc1_idx = args.force_fc1
    if args.force_fc2 >= 0:
        fc2_idx = args.force_fc2

    fc1_info = get_kernel_info_ext(fc1_idx) if fc1_idx >= 0 else "N/A"
    fc2_info = get_kernel_info_ext(fc2_idx) if fc2_idx >= 0 else "N/A"

    # Data volume for bandwidth estimation (same formula as nsys_analyze.sh)
    fc1_w_bytes = active * (2*I) * (H // 2) + active * (2*I) * (H // sf_block)
    fc1_act_bytes = expanded * H * 2 + expanded * I * 2
    fc2_w_bytes = active * H * (I // 2) + active * H * (I // sf_block)
    fc2_act_bytes = expanded * I * 2 + expanded * H * 2
    total_bytes = fc1_w_bytes + fc1_act_bytes + fc2_w_bytes + fc2_act_bytes
    pipe_bw = (total_bytes / 1e12) / (pipeline_us / 1e6) if pipeline_us > 0 else 0

    return {
        "T": T, "pipeline_us": pipeline_us, "active": active, "tile_n": tile_n,
        "fc1_info": fc1_info, "fc1_idx": fc1_idx,
        "fc2_info": fc2_info, "fc2_idx": fc2_idx,
        "total_mb": total_bytes / 1e6, "pipe_bw": pipe_bw,
    }


def main():
    gpu_name = torch.cuda.get_device_name()
    sm_count = lib.moe_cubin_get_sm_count()

    if args.list_configs:
        T = token_list[0]
        for tn in [8, 16, 32]:
            print(f"\n{'='*60}")
            print(f"  Valid configs for BS={T} tile_n={tn}:")
            print(f"{'='*60}")
            for label, is_fc1, M, K_dim in [("FC1", True, 2*I, H), ("FC2", False, H, I)]:
                configs = list_valid_configs(T, is_fc1, tn, M, K_dim)
                if configs:
                    print(f"\n  {label} (M={M} K={K_dim}): {len(configs)} configs")
                    for idx, info in configs:
                        print(f"    [{idx:>3}] {info}")
        return

    routing_mode = "balanced" if args.balanced else "random"
    print(f"\n{'='*70}")
    print(f"  {args.model.upper()} TP{args.tp} — {args.dtype.upper()} MOE Benchmark")
    print(f"{'='*70}")
    print(f"  GPU:     {gpu_name} ({sm_count} SMs)")
    print(f"  Model:   H={H} I={I} E={E} K={K}")
    print(f"  Routing: {routing_mode}")
    if args.force_fc1 >= 0:
        print(f"  Force FC1: [{args.force_fc1}] {get_kernel_info_ext(args.force_fc1)}")
    if args.force_fc2 >= 0:
        print(f"  Force FC2: [{args.force_fc2}] {get_kernel_info_ext(args.force_fc2)}")
    print(f"  Iters:   {args.iters} (warmup={args.warmup})")
    print()

    results = []
    for T in token_list:
        r = bench_one(T)
        results.append(r)

    # Print results table
    print(f"  {'BS':>4} {'Active':>10} {'Pipe(us)':>9} {'BW':>9} {'Data':>7} {'tN':>3}"
          f" │ {'FC1 config':<42} │ {'FC2 config':<42}")
    print(f"  {'────':>4} {'──────────':>10} {'─────────':>9} {'─────────':>9} {'───────':>7} {'───':>3}"
          f" ┼ {'──────────────────────────────────────────':<42} ┼ {'──────────────────────────────────────────':<42}")
    for r in results:
        fc1_tag = f"[{r['fc1_idx']:>3}] {r['fc1_info']}" if r['fc1_idx'] >= 0 else "N/A"
        fc2_tag = f"[{r['fc2_idx']:>3}] {r['fc2_info']}" if r['fc2_idx'] >= 0 else "N/A"
        print(f"  {r['T']:>4} {r['active']:>4}/{E:<4} {r['pipeline_us']:>9.1f}"
              f" {r['pipe_bw']:>7.3f}T/s {r['total_mb']:>5.0f}MB {r['tile_n']:>3}"
              f" │ {fc1_tag:<42} │ {fc2_tag:<42}")
    print()


if __name__ == "__main__":
    main()
