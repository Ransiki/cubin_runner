"""Phase 1: Quantize weights + prepare shuffled layout using TRT-LLM ops."""
import sys, json, torch
import torch.nn.functional as F

torch.ops.load_library('/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libth_common.so')
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a, shuffle_matrix_sf_a)


def quant_fp4(a):
    # MxFP4 (MxE2m1) uses gsf=1.0: the E8M0 block scales capture the full
    # dynamic range.  trtllm-gen's own test confirms this — any other gsf
    # value is silently ignored during MxE2m1 quantisation.
    global_sf = torch.tensor(1.0, device=a.device)
    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a.cuda(), global_sf.cuda(), 32, True, False)
    return a_fp4, a_sf, global_sf


def main():
    cfg = json.load(open(sys.argv[1]))
    T, H, I, E, K, seed = cfg["T"], cfg["H"], cfg["I"], cfg["E"], cfg["K"], cfg["seed"]
    d = cfg["dir"]
    device = "cuda"

    torch.manual_seed(seed)
    w_scale = cfg.get("weight_scale", 0.5)
    fc1_w = torch.randn(E, 2*I, H, device=device, dtype=torch.bfloat16) * w_scale
    fc2_w = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * w_scale
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.5
    logits = torch.randn(T, E, device=device, dtype=torch.bfloat16)

    epilogue_tile_m = 128
    I2 = 2 * I

    # Quantize weights
    g1_gsf_list, g2_gsf_list = [], []
    g1_fp4_all, g1_sf_all = [], []
    g2_fp4_all, g2_sf_all = [], []
    for i in range(E):
        fp4, sf, gsf = quant_fp4(fc1_w[i])
        g1_fp4_all.append(fp4)
        g1_sf_all.append(sf)
        g1_gsf_list.append(gsf.item())
    for i in range(E):
        fp4, sf, gsf = quant_fp4(fc2_w[i])
        g2_fp4_all.append(fp4)
        g2_sf_all.append(sf)
        g2_gsf_list.append(gsf.item())

    g1_fp4 = torch.stack(g1_fp4_all).view(torch.uint8).reshape(E, I2, H//2)
    g1_sf = torch.stack(g1_sf_all).view(torch.uint8).reshape(E, I2, H//32)
    g2_fp4 = torch.stack(g2_fp4_all).view(torch.uint8).reshape(E, H, I//2)
    g2_sf = torch.stack(g2_sf_all).view(torch.uint8).reshape(E, H, I//32)
    g1_gsf = torch.tensor(g1_gsf_list, device=device)
    g2_gsf = torch.tensor(g2_gsf_list, device=device)

    # FC1: reorder + shuffle (view as float8_e4m3fn for reorder_rows API compatibility)
    g1s_w, g1s_sf = [], []
    for i in range(E):
        wr = reorder_rows_for_gated_act_gemm(g1_fp4[i].clone().view(torch.float8_e4m3fn))
        sr = reorder_rows_for_gated_act_gemm(g1_sf[i].clone().view(torch.float8_e4m3fn))
        g1s_w.append(shuffle_matrix_a(wr.view(torch.uint8), epilogue_tile_m))
        g1s_sf.append(shuffle_matrix_sf_a(sr.view(torch.uint8), epilogue_tile_m))
    g1s_w = torch.stack(g1s_w).reshape(E, I2, H//2)
    g1s_sf = torch.stack(g1s_sf).view(torch.uint8).reshape(E, I2, H//32)

    # FC2: shuffle only
    g2s_w, g2s_sf = [], []
    for i in range(E):
        g2s_w.append(shuffle_matrix_a(g2_fp4[i].view(torch.uint8), epilogue_tile_m))
        g2s_sf.append(shuffle_matrix_sf_a(g2_sf[i].view(torch.uint8), epilogue_tile_m))
    g2s_w = torch.stack(g2s_w).reshape(E, H, I//2)
    g2s_sf = torch.stack(g2s_sf).view(torch.uint8).reshape(E, H, I//32)

    # For MxFP4 W4A16: gsf=1.0 so all per-expert scales are 1.0.
    # The E8M0 block scales already encode the full weight magnitude.
    scale_c_fc1 = torch.ones(E, device=device, dtype=torch.float32)
    scale_gate_fc1 = torch.ones(E, device=device, dtype=torch.float32)
    scale_c_fc2 = torch.ones(E, device=device, dtype=torch.float32)

    # Dequantize the FP4 weights back to float for reference computation
    # This is the TRT-LLM approach: compare cubin vs dequant(quant(weights))
    # so quantization error is factored out — only GEMM numerical error remains
    # mxfp4_dequantize_unswizzled already returns values at original scale
    # (gsf is baked into the block scale factors by fp4_quantize).
    # Do NOT multiply by inv_gsf — that was a double-scaling bug.
    def dequant_mxfp4(fp4_bytes, sf_bytes, rows=-1):
        fp4_2d = fp4_bytes.cpu().reshape(rows, -1) if rows > 0 else fp4_bytes.cpu()
        sf_2d = sf_bytes.cpu().reshape(fp4_2d.shape[0], -1)
        return torch.ops.trtllm.mxfp4_dequantize_unswizzled(fp4_2d, sf_2d, 32).float()

    fc1_deq_list, fc2_deq_list = [], []
    for i in range(E):
        fc1_deq_list.append(dequant_mxfp4(g1_fp4_all[i], g1_sf_all[i], rows=I2).cuda())
        fc2_deq_list.append(dequant_mxfp4(g2_fp4_all[i], g2_sf_all[i], rows=H).cuda())

    fc1_deq = torch.stack(fc1_deq_list).reshape(E, 2*I, H)
    fc2_deq = torch.stack(fc2_deq_list).reshape(E, H, I)

    # Reference using dequantized weights + renormalize routing
    _, topk_indices = torch.topk(logits.float(), K, dim=-1)
    topk_logits = logits.float().gather(-1, topk_indices)
    topk_weights = F.softmax(topk_logits, dim=-1)

    ref_out = torch.zeros(T, H, dtype=torch.float32, device=device)
    for t in range(T):
        for k in range(K):
            eid = topk_indices[t, k].item()
            w = topk_weights[t, k].item()
            x = hidden[t:t+1].float()
            proj = x @ fc1_deq[eid].float().t()
            gate, up = proj[:, :I], proj[:, I:]
            act = F.silu(gate) * up
            out = act @ fc2_deq[eid].float().t()
            ref_out[t] += w * out.squeeze(0)
    ref_out = ref_out.to(torch.bfloat16)

    # Save everything
    torch.save({
        "hidden": hidden.cpu(), "logits": logits.cpu(),
        "g1_w": g1s_w.cpu(), "g1_sf": g1s_sf.cpu(),
        "g2_w": g2s_w.cpu(), "g2_sf": g2s_sf.cpu(),
        "scale_c_fc1": scale_c_fc1.cpu(), "scale_gate_fc1": scale_gate_fc1.cpu(),
        "scale_c_fc2": scale_c_fc2.cpu(),
        "ref_out": ref_out.cpu(),
    }, f"{d}/data.pt")

    print(f"Quantized & saved. g1_gsf={g1_gsf[:2].tolist()}, g2_gsf={g2_gsf[:2].tolist()}")
    print(f"ref_out range: [{ref_out.min().item():.6f}, {ref_out.max().item():.6f}]")


if __name__ == "__main__":
    main()
