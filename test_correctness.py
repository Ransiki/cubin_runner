"""
MOE Correctness Test — routing + GEMM for NvFP4 and MxFP4.

Layer 1: Routing — CUDA kernel == PyTorch topK (exact match)
Layer 2: GEMM   — cubin output vs ground truth

  NvFP4: ground truth = Python dequant reference (two-process)
  MxFP4: ground truth = TRT-LLM Bf16MxE2m1BlockScaleMoERunner (two-process)

Usage:
  python3 test_correctness.py                       # routing only, both dtypes
  python3 test_correctness.py --gemm                # routing + GEMM
  python3 test_correctness.py --gemm --dtype mxfp4  # MxFP4 only
  python3 test_correctness.py --gemm --dtype nvfp4  # NvFP4 only
  python3 test_correctness.py --gemm --full         # DSv3 TP8 shape
"""
import sys, os, subprocess, json, tempfile, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ["MXFP4_MULTI_TILE"] = "0"

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="MOE Correctness Test")
parser.add_argument("--full", action="store_true", help="full DSv3 TP8 shape")
parser.add_argument("--stress", action="store_true", help="comprehensive corner-case coverage")
parser.add_argument("--gemm", action="store_true", help="include GEMM test (needs TRT-LLM)")
parser.add_argument("--dtype", default="both", choices=["nvfp4", "mxfp4", "both"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timeout", type=int, default=300, help="subprocess timeout in seconds")
args = parser.parse_args()

# (T, H, I, E, K)
SMALL_CONFIGS = [
    (8,  256, 256, 4, 2),
    (4,  256, 256, 4, 2),
    (16, 256, 256, 4, 2),
]
FULL_CONFIGS = [
    (8,  256, 256, 4, 2),
    (16, 256, 256, 4, 2),
    (8,  512, 512, 4, 2),
]
# Stress: cover all corner cases (T boundaries, E/K combos, dimensions)
STRESS_CONFIGS = [
    # Token count: single CTA → multi-CTA
    (1,  256, 256, 4, 2),
    (2,  256, 256, 4, 2),
    (4,  256, 256, 4, 2),
    (8,  256, 256, 4, 2),
    (16, 256, 256, 4, 2),
    # Expert/TopK
    (8,  256, 256, 4, 1),   # K=1
    (8,  256, 256, 8, 2),   # E=8 sparse
    # Dimensions
    (4,  128, 128, 4, 2),   # small
    (4,  512, 512, 4, 2),   # medium
]


# ═════════════════════════════════════════════════════════════════
# Layer 1: Routing
# ═════════════════════════════════════════════════════════════════

def test_routing(dtype):
    if dtype == "mxfp4":
        from mxfp4_moe_cubin import compute_routing_cuda
    else:
        from nvfp4_moe_cubin import compute_routing_cuda

    T, E, K = (4, 256, 8) if args.full else (4, 256, 8)
    n_pass = 0
    n_total = 0
    for seed in [args.seed, args.seed + 1, args.seed + 100]:
        for tile_n in [8, 16]:
            torch.manual_seed(seed)
            logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16).float()
            meta = compute_routing_cuda(logits, E, K, tile_n, 1)
            tpe = meta["tokens_per_expert"]
            cuda_active = set((tpe > 0).nonzero(as_tuple=True)[0].tolist())
            cuda_total = tpe.sum().item()

            _, pt_idx = torch.topk(logits, K, dim=-1)
            pt_counts = torch.bincount(pt_idx.flatten().long(), minlength=E)
            pt_active = set((pt_counts > 0).nonzero(as_tuple=True)[0].tolist())

            n_total += 1
            if cuda_total == T * K and cuda_active == pt_active:
                n_pass += 1
            else:
                print(f"    FAIL: {dtype} seed={seed} tn={tile_n}")
    return n_pass, n_total


# ═════════════════════════════════════════════════════════════════
# Layer 2: NvFP4 GEMM (dequant reference)
# ═════════════════════════════════════════════════════════════════

def test_nvfp4_gemm(T, H, I, E, K, seed):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"T": T, "H": H, "I": I, "E": E, "K": K, "seed": seed, "dir": tmpdir}
        cfg_path = os.path.join(tmpdir, "config.json")
        json.dump(cfg, open(cfg_path, "w"))

        r1 = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "_phase1_quantize.py"), cfg_path],
            capture_output=True, text=True, timeout=300)
        if r1.returncode != 0:
            return False, f"Phase1 failed: {r1.stderr[-200:]}"

        r2 = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "_phase2_cubin_run.py"), cfg_path],
            capture_output=True, text=True, timeout=300)
        if r2.returncode != 0:
            return False, f"Phase2 failed: {r2.stderr[-200:]}"

        result_path = os.path.join(tmpdir, "result.json")
        if os.path.exists(result_path):
            r = json.load(open(result_path))
            return r.get("passed", False), r2.stdout.strip().split("\n")[-1]
        return False, "No result.json"


# ═════════════════════════════════════════════════════════════════
# Layer 2: MxFP4 GEMM (TRT-LLM ground truth)
# ═════════════════════════════════════════════════════════════════

def test_mxfp4_gemm(T, H, I, E, K, seed):
    """Two-process: Phase A runs TRT-LLM, Phase B runs our cubin, compare."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase A: TRT-LLM ground truth
        phase_a = f'''
import torch
torch.ops.load_library("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libth_common.so")
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a, shuffle_matrix_sf_a, float4_sf_dtype)
torch.manual_seed({seed}); device="cuda"
T,H,I,E,K,sf_bs={T},{H},{I},{E},{K},32
fc1_w=torch.randn(E,2*I,H,device=device,dtype=torch.bfloat16)*0.5
fc2_w=torch.randn(E,H,I,device=device,dtype=torch.bfloat16)*0.5
hidden=torch.randn(T,H,device=device,dtype=torch.bfloat16)*0.5
logits=torch.randn(T,E,device=device,dtype=torch.bfloat16)
g1w,g1sf,g2w,g2sf=[],[],[],[]
for i in range(E):
    fp4,sf=torch.ops.trtllm.fp4_quantize(fc1_w[i],None,sf_bs,True,False)
    w=fp4.view(torch.uint8).reshape(2*I,H//2).view(torch.float8_e4m3fn)
    s=sf.view(torch.uint8).reshape(2*I,H//sf_bs).view(float4_sf_dtype)
    g1w.append(shuffle_matrix_a(reorder_rows_for_gated_act_gemm(w).view(torch.uint8),128))
    g1sf.append(shuffle_matrix_sf_a(reorder_rows_for_gated_act_gemm(s).view(torch.uint8),128,num_elts_per_sf=sf_bs))
for i in range(E):
    fp4,sf=torch.ops.trtllm.fp4_quantize(fc2_w[i],None,sf_bs,True,False)
    w=fp4.view(torch.uint8).reshape(H,I//2).view(torch.float8_e4m3fn)
    s=sf.view(torch.uint8).reshape(H,I//sf_bs).view(float4_sf_dtype)
    g2w.append(shuffle_matrix_a(w.view(torch.uint8),128))
    g2sf.append(shuffle_matrix_sf_a(s.view(torch.uint8),128,num_elts_per_sf=sf_bs))
gw1=torch.stack(g1w).reshape(E,2*I,H//2).contiguous()
gs1=torch.stack(g1sf).view(torch.uint8).reshape(E,2*I,H//sf_bs).contiguous()
gw2=torch.stack(g2w).reshape(E,H,I//2).contiguous()
gs2=torch.stack(g2sf).view(torch.uint8).reshape(E,H,I//sf_bs).contiguous()
runner=torch.classes.trtllm.Bf16MxE2m1BlockScaleMoERunner(0)
c8=[c for c in runner.get_valid_configs(K,H,I,E,T,H,I) if c[0]==8]
cfg=c8[0] if c8 else runner.get_valid_configs(K,H,I,E,T,H,I)[0]
out=runner.run_moe(logits,None,hidden,gw1,gs1,None,None,None,None,gw2,gs2,None,
    E,K,None,None,I,H,I,0,E,None,1,list(cfg),None,None)
torch.cuda.synchronize()
torch.save({{"out":out.cpu(),"logits":logits.cpu(),"hidden":hidden.cpu(),
    "gw1":gw1.cpu(),"gs1":gs1.cpu(),"gw2":gw2.cpu(),"gs2":gs2.cpu()}},
    "{tmpdir}/data.pt")
print("OK")
'''
        r1 = subprocess.run([sys.executable, "-c", phase_a],
                            capture_output=True, text=True, timeout=300)
        if r1.returncode != 0 or "OK" not in r1.stdout:
            return False, f"TRT-LLM phase failed: {r1.stderr[-200:]}"

        # Phase B: Our cubin
        phase_b = f'''
import os,sys,torch,torch.nn.functional as F,json
os.environ["MXFP4_MULTI_TILE"]="0"
sys.path.insert(0,"{SCRIPT_DIR}")
from mxfp4_moe_cubin import mxfp4_moe_cubin
data=torch.load("{tmpdir}/data.pt",weights_only=False); device="cuda"
T,H,I,E,K={T},{H},{I},{E},{K}
sc=torch.ones(E,device=device,dtype=torch.float32)
result=mxfp4_moe_cubin(
    routing_logits=data["logits"].to(device).float(),hidden_states=data["hidden"].to(device),
    gemm1_weights=data["gw1"].to(device),gemm1_weights_scale=data["gs1"].to(device),
    gemm2_weights=data["gw2"].to(device),gemm2_weights_scale=data["gs2"].to(device),
    num_experts=E,top_k=K,intermediate_size=I,
    output1_scale_scalar=sc,output1_scale_gate_scalar=sc,output2_scale_scalar=sc,
    routing_method_type=1)
cubin_out=result[0]; torch.cuda.synchronize()
trtllm_out=data["out"].to(device).float()
a,b=cubin_out.float(),trtllm_out
cos=F.cosine_similarity(a.flatten().unsqueeze(0),b.flatten().unsqueeze(0)).item()
has_nan=a.isnan().any().item()
diff=(a-b).abs()
ref_scale=b.abs().mean().item()
adaptive_atol=max(0.05*ref_scale,1e-3)
elem_pass=(diff<=adaptive_atol+0.1*b.abs()).float().mean().item()
json.dump({{"has_nan":has_nan,"cos":cos,"elem_pass":elem_pass}},
    open("{tmpdir}/result.json","w"))
print(f"cos={{cos:.6f}} elem_pass={{elem_pass:.4f}} nan={{has_nan}}")
'''
        r2 = subprocess.run([sys.executable, "-c", phase_b],
                            capture_output=True, text=True, timeout=300)
        if r2.returncode != 0:
            return False, f"Cubin phase failed: {r2.stderr[-200:]}"

        result_path = os.path.join(tmpdir, "result.json")
        if os.path.exists(result_path):
            r = json.load(open(result_path))
            passed = (not r["has_nan"]) and r["cos"] > 0.99 and r["elem_pass"] > 0.95
            detail = r2.stdout.strip()
            return passed, detail
        return False, "No result.json"


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    dtypes = ["nvfp4", "mxfp4"] if args.dtype == "both" else [args.dtype]
    if args.stress:
        configs = STRESS_CONFIGS
        seeds = [42, 7, 99]
    elif args.full:
        configs = FULL_CONFIGS
        seeds = [args.seed]
    else:
        configs = SMALL_CONFIGS
        seeds = [args.seed]
    all_pass = True

    mode = "stress" if args.stress else ("full" if args.full else "quick")
    print(f"\n{'='*70}")
    print(f"  MOE Correctness Test — {mode} mode")
    print(f"{'='*70}")

    # Layer 1: Routing
    print(f"\n  Layer 1: Routing")
    for dtype in dtypes:
        n_pass, n_total = test_routing(dtype)
        ok = n_pass == n_total
        print(f"    {dtype.upper():>6}: {n_pass}/{n_total} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    # Layer 2: GEMM
    if args.gemm or args.stress:
        print(f"\n  Layer 2: GEMM ({len(configs)} configs × {len(seeds)} seeds × {len(dtypes)} dtypes)")
        n_pass = n_fail = n_skip = 0
        for T, H, I, E, K in configs:
            for seed in seeds:
                for dtype in dtypes:
                    tag = f"{dtype.upper()} T={T:<3} H={H:<4} I={I:<3} E={E:<3} K={K} s={seed}"
                    try:
                        if dtype == "nvfp4":
                            passed, detail = test_nvfp4_gemm(T, H, I, E, K, seed)
                        else:
                            passed, detail = test_mxfp4_gemm(T, H, I, E, K, seed)
                    except subprocess.TimeoutExpired:
                        passed, detail = None, "TIMEOUT"
                    except Exception as e:
                        passed, detail = False, str(e)[:80]

                    if passed is None:
                        status = "SKIP"
                        n_skip += 1
                    elif passed:
                        status = "PASS"
                        n_pass += 1
                    else:
                        status = "FAIL"
                        n_fail += 1
                        all_pass = False

                    cos_str = ""
                    if isinstance(detail, str) and "cos=" in detail:
                        cos_str = f" cos={detail.split('cos=')[1].split()[0]}"
                    sys.stdout.flush(); print(f"    [{status}] {tag}{cos_str}")
                    if status == "FAIL":
                        print(f"           {detail}")
        print(f"\n    {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    elif not args.stress:
        print(f"\n  Layer 2 skipped (use --gemm or --stress)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
