"""
Benchmark MxFP4 vs NvFP4 MOE cubin.

Usage:
  python3 bench_mxfp4_moe.py --model dsv3 --tp 8 --tokens 128
  python3 bench_mxfp4_moe.py --model dsv3 --tp 1 --tokens 1,16,128
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="dsv3", choices=["kimi_k2", "dsv3"])
parser.add_argument("--tokens", default="1,16,128")
parser.add_argument("--tp", type=int, default=8)
parser.add_argument("--iters", type=int, default=20)
parser.add_argument("--warmup", type=int, default=5)
args = parser.parse_args()

CONFIGS = {
    "kimi_k2": {"hidden_size": 7168, "intermediate_size": 2048, "num_experts": 384, "top_k": 8},
    "dsv3": {"hidden_size": 7168, "intermediate_size": 2048, "num_experts": 256, "top_k": 8},
}
cfg = CONFIGS[args.model]
H = cfg["hidden_size"]; I = cfg["intermediate_size"] // args.tp
E = cfg["num_experts"]; K = cfg["top_k"]
token_list = [int(t) for t in args.tokens.split(",")]

def bench_one(moe_fn, T, H, I, E, K, device, sf_block, sf_dtype):
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
    for _ in range(args.warmup):
        moe_fn(**kw); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.iters):
        moe_fn(**kw)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / args.iters
    flops = 2*T*K*(2*I)*H + 2*T*K*H*I
    return ms, flops/(ms/1e3)/1e12

def main():
    device = "cuda"
    print(f"{'='*70}\n  {args.model} TP{args.tp} — MxFP4 vs NvFP4  H={H} I={I} E={E} K={K}\n{'='*70}")
    from nvfp4_moe_cubin import nvfp4_moe_cubin
    from mxfp4_moe_cubin import mxfp4_moe_cubin
    results = []
    for T in token_list:
        nv_ms, nv_tf = bench_one(nvfp4_moe_cubin, T, H, I, E, K, device, 16, torch.float8_e4m3fn)
        mx_ms, mx_tf = bench_one(mxfp4_moe_cubin, T, H, I, E, K, device, 32, torch.uint8)
        sp = nv_ms / mx_ms if mx_ms > 0 else 0
        print(f"  BS={T:>4}: NvFP4={nv_ms:.3f}ms MxFP4={mx_ms:.3f}ms speedup={sp:.2f}x")
        results.append((T, nv_ms, mx_ms, sp))
    print(f"\n{'BS':>5}  {'NvFP4(ms)':>10}  {'MxFP4(ms)':>10}  {'Speedup':>8}")
    for T, nv, mx, sp in results:
        print(f"{T:>5}  {nv:>10.3f}  {mx:>10.3f}  {sp:>7.2f}x")

if __name__ == "__main__":
    main()
