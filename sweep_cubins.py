"""
Benchmark all valid cubin configs for FC1/FC2 at specific problem shapes.

Usage:
  python3 sweep_cubins.py                    # default DSv3 TP8
  python3 sweep_cubins.py --bs 1 16 128      # specific batch sizes
"""
import sys, os, ctypes, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from nvfp4_moe_cubin import _load_lib, _ptr, _host_int_array, compute_routing_cuda

def sweep(lib, is_fc1, tile_n, M, N, K, num_experts, num_tokens,
          meta, hidden, alloc_padded, n_warmup=10, n_bench=50):
    """Benchmark all valid cubins for one (FC1 or FC2) configuration."""
    device = "cuda"
    label = "FC1" if is_fc1 else "FC2"

    # Allocate dummy buffers with correct shapes
    if is_fc1:
        weights = torch.zeros(num_experts, M, K // 2, device=device, dtype=torch.uint8)
        weights_sf = torch.ones(num_experts, M, K // 16, device=device, dtype=torch.float8_e4m3fn)
        output = torch.zeros(alloc_padded, M // 2, device=device, dtype=torch.bfloat16)  # intermediate_size
    else:
        weights = torch.zeros(num_experts, M, K // 2, device=device, dtype=torch.uint8)
        weights_sf = torch.ones(num_experts, M, K // 16, device=device, dtype=torch.float8_e4m3fn)
        output = torch.zeros(alloc_padded, M, device=device, dtype=torch.bfloat16)

    inp = hidden if is_fc1 else torch.zeros(alloc_padded, K, device=device, dtype=torch.bfloat16)
    scale_c = torch.ones(num_experts, device=device, dtype=torch.float32)
    scale_g = torch.ones(num_experts, device=device, dtype=torch.float32)
    batched_n_host = _host_int_array(meta["batched_n"])
    stream = torch.cuda.current_stream(device).cuda_stream

    max_results = 48
    out_idx = (ctypes.c_int * max_results)()
    out_us = (ctypes.c_float * max_results)()

    lib.moe_cubin_benchmark_all.argtypes = [
        ctypes.c_bool,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.moe_cubin_benchmark_all.restype = ctypes.c_int

    n_found = lib.moe_cubin_benchmark_all(
        is_fc1, tile_n, M, N, K, num_experts, num_tokens,
        _ptr(weights), _ptr(weights_sf),
        _ptr(inp), _ptr(output),
        _ptr(scale_c), _ptr(scale_g),
        _ptr(meta["permuted_idx_to_token_idx"]),
        _ptr(meta["cta_idx_xy_to_batch_idx"]),
        _ptr(meta["cta_idx_xy_to_mn_limit"]),
        _ptr(meta["num_non_exiting_ctas"]),
        _ptr(meta["total_num_padded_tokens"]),
        batched_n_host, num_experts,
        n_warmup, n_bench,
        out_idx, out_us, max_results,
        ctypes.c_void_p(stream),
    )

    # Collect results
    lib.moe_cubin_get_config_info.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 7
    results = []
    for j in range(n_found):
        idx = out_idx[j]
        us = out_us[j]
        tM, tN, tK, nS, nSm, isPers, isU2 = [ctypes.c_int() for _ in range(7)]
        lib.moe_cubin_get_config_info(idx,
            ctypes.byref(tM), ctypes.byref(tN), ctypes.byref(tK),
            ctypes.byref(nS), ctypes.byref(nSm), ctypes.byref(isPers), ctypes.byref(isU2))
        results.append({
            "idx": idx, "us": us,
            "tileM": tM.value, "tileN": tN.value, "tileK": tK.value,
            "stages": nS.value, "stagesMma": nSm.value,
            "persistent": bool(isPers.value), "unroll2x": bool(isU2.value),
        })

    results.sort(key=lambda r: r["us"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--inter", type=int, default=256, help="intermediate_size per TP shard")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bench", type=int, default=50)
    args = parser.parse_args()

    lib = _load_lib()
    device = "cuda"

    print("=" * 80)
    print("  Cubin Performance Sweep — DeepSeek-V3 TP=8")
    print("=" * 80)
    print(f"  hidden={args.hidden}, inter={args.inter}, experts={args.experts}, topK={args.topk}")
    print(f"  batch sizes: {args.bs}")
    print(f"  warmup={args.warmup}, bench={args.bench}")
    print()

    H = args.hidden
    I = args.inter
    E = args.experts
    K_top = args.topk

    for bs in args.bs:
        T = bs
        num_tokens_expanded = T * K_top
        avg_per_expert = num_tokens_expanded / E
        tile_n_candidates = set()
        for tn in [8, 16, 32, 64]:
            tile_n_candidates.add(tn)

        print(f"{'='*80}")
        print(f"  BS={T}, expanded={num_tokens_expanded}, avg/expert={avg_per_expert:.1f}")
        print(f"{'='*80}")

        for tile_n in sorted(tile_n_candidates):
            # Generate routing metadata for this bs
            torch.manual_seed(42)
            logits = torch.randn(T, E, device=device, dtype=torch.bfloat16)
            hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1

            meta = compute_routing_cuda(logits, E, K_top, tile_n, routing_method=1)
            alloc_padded = meta["max_padded"]

            # FC1: M=2*I, K=H
            fc1_M = 2 * I
            fc1_K = H
            print(f"\n  --- FC1 tile_n={tile_n} (M={fc1_M}, K={fc1_K}) ---")
            fc1_results = sweep(lib, True, tile_n, fc1_M, tile_n, fc1_K,
                                E, num_tokens_expanded, meta, hidden, alloc_padded,
                                args.warmup, args.bench)
            if not fc1_results:
                print(f"    No valid FC1 cubins for tile_n={tile_n}")
            else:
                for i, r in enumerate(fc1_results):
                    tag = " *** BEST ***" if i == 0 else ""
                    sched = "persistent" if r["persistent"] else "static"
                    u2 = " u2" if r["unroll2x"] else ""
                    print(f"    [{r['us']:8.1f} us] tile={r['tileM']}x{r['tileN']}x{r['tileK']} "
                          f"stages={r['stages']}/{r['stagesMma']} {sched}{u2}{tag}")

            # FC2: M=H, K=I
            fc2_M = H
            fc2_K = I
            print(f"\n  --- FC2 tile_n={tile_n} (M={fc2_M}, K={fc2_K}) ---")
            fc2_results = sweep(lib, False, tile_n, fc2_M, tile_n, fc2_K,
                                E, num_tokens_expanded, meta, hidden, alloc_padded,
                                args.warmup, args.bench)
            if not fc2_results:
                print(f"    No valid FC2 cubins for tile_n={tile_n}")
            else:
                for i, r in enumerate(fc2_results):
                    tag = " *** BEST ***" if i == 0 else ""
                    sched = "persistent" if r["persistent"] else "static"
                    u2 = " u2" if r["unroll2x"] else ""
                    print(f"    [{r['us']:8.1f} us] tile={r['tileM']}x{r['tileN']}x{r['tileK']} "
                          f"stages={r['stages']}/{r['stagesMma']} {sched}{u2}{tag}")

        print()


if __name__ == "__main__":
    main()
