"""Phase 2: Load MxFP4 quantized weights and run cubin pipeline (NO TRT-LLM loaded)."""
import sys, os, json
os.environ["MXFP4_MULTI_TILE"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

import torch
from mxfp4_moe_cubin import mxfp4_moe_cubin


def main():
    cfg = json.load(open(sys.argv[1]))
    T, H, I, E, K = cfg["T"], cfg["H"], cfg["I"], cfg["E"], cfg["K"]
    d = cfg["dir"]

    data = torch.load(f"{d}/data.pt", weights_only=False)
    device = "cuda"

    hidden = data["hidden"].to(device)
    logits = data["logits"].to(device).float()
    g1_w = data["g1_w"].to(device)
    g1_sf = data["g1_sf"].to(device)
    g2_w = data["g2_w"].to(device)
    g2_sf = data["g2_sf"].to(device)
    scale_c_fc1 = data["scale_c_fc1"].to(device)
    scale_gate_fc1 = data["scale_gate_fc1"].to(device)
    scale_c_fc2 = data["scale_c_fc2"].to(device)
    ref_out = data["ref_out"].to(device)

    try:
        result = mxfp4_moe_cubin(
            routing_logits=logits,
            hidden_states=hidden,
            gemm1_weights=g1_w,
            gemm1_weights_scale=g1_sf,
            gemm2_weights=g2_w,
            gemm2_weights_scale=g2_sf,
            num_experts=E, top_k=K, intermediate_size=I,
            output1_scale_scalar=scale_c_fc1,
            output1_scale_gate_scalar=scale_gate_fc1,
            output2_scale_scalar=scale_c_fc2,
            routing_method_type=1,  # renormalize: topK→softmax
        )
        cubin_out = result[0]
        torch.cuda.synchronize()
    except Exception as e:
        print(f"CUBIN ERROR: {e}")
        import traceback; traceback.print_exc()
        json.dump({"passed": False, "error": str(e)}, open(f"{d}/result.json", "w"))
        return

    a = cubin_out.float()
    b = ref_out.float()
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_scale = b.abs().mean().item()

    # TRT-LLM tolerance: atol=0.1, rtol=0.85, 92.5% pass rate
    atol, rtol, percent = 0.1, 0.85, 0.925
    left = diff
    right = atol + rtol * b.abs()
    mismatch = (left > right).float().mean().item()
    trtllm_pass = mismatch <= (1 - percent)

    print(f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} ref_scale={ref_scale:.6f}")
    print(f"TRT-LLM check: mismatch={mismatch:.4f} (need <={1-percent:.4f}) -> {'PASS' if trtllm_pass else 'FAIL'}")
    print(f"cubin[0,:4]: {cubin_out[0,:4].tolist()}")
    print(f"ref[0,:4]:   {ref_out[0,:4].tolist()}")

    passed = trtllm_pass
    print(f"{'PASS' if passed else 'FAIL'}")
    json.dump({"passed": passed, "max_diff": max_diff, "mean_diff": mean_diff, "mismatch": mismatch}, open(f"{d}/result.json", "w"))


if __name__ == "__main__":
    main()
