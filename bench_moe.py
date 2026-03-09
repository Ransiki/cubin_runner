"""
Profile NvFP4 MOE cubin with Kimi-K2 MOE shape.

Kimi-K2 config (from HuggingFace moonshotai/Kimi-K2-Base):
  hidden_size = 7168
  moe_intermediate_size = 2048
  n_routed_experts = 384
  num_experts_per_tok = 8

Run with nsys:
  nsys profile -o kimi_k2_moe --force-overwrite true \
    python3 profile.py

Or standalone:
  python3 profile.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import time

# --- Model config (switch between presets) ---
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--model", default="dsv3", choices=["kimi_k2", "dsv3"])
_parser.add_argument("--tokens", type=int, default=128)
_args, _ = _parser.parse_known_args()

CONFIGS = {
    "kimi_k2": {  # moonshotai/Kimi-K2-Base
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 384,
        "top_k": 8,
    },
    "dsv3": {  # deepseek-ai/DeepSeek-V3
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 256,
        "top_k": 8,
    },
}

_cfg = CONFIGS[_args.model]
HIDDEN_SIZE = _cfg["hidden_size"]
INTERMEDIATE_SIZE = _cfg["intermediate_size"]
NUM_EXPERTS = _cfg["num_experts"]
TOP_K = _cfg["top_k"]
NUM_TOKENS = _args.tokens

NUM_ITERS = 10
NUM_WARMUP = 3


def main():
    device = "cuda"
    print("=" * 60)
    print(f"  {_args.model} MOE Profile — NvFP4 × BF16 Cubin")
    print("=" * 60)
    print(f"  hidden_size      = {HIDDEN_SIZE}")
    print(f"  intermediate_size= {INTERMEDIATE_SIZE}")
    print(f"  num_experts      = {NUM_EXPERTS}")
    print(f"  top_k            = {TOP_K}")
    print(f"  num_tokens       = {NUM_TOKENS}")
    print(f"  warmup iters     = {NUM_WARMUP}")
    print(f"  profile iters    = {NUM_ITERS}")
    print()

    # We need TRT-LLM for weight prep — run in same process since
    # we're profiling, not doing correctness (no symbol conflict for cubin selection
    # since we load our .so first before TRT-LLM)
    # Actually to avoid conflict, just use dummy shuffled weights.

    from nvfp4_moe_cubin import _load_lib

    lib = _load_lib()
    print(f"  Library loaded: {lib._name}")
    print(f"  SM count: {lib.moe_cubin_get_sm_count()}")

    # Create dummy quantized weights (already "shuffled")
    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE
    E = NUM_EXPERTS
    K = TOP_K
    T = NUM_TOKENS

    torch.manual_seed(42)

    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1
    logits = torch.randn(T, E, device=device, dtype=torch.bfloat16)

    # Dummy FP4 weights — zero-initialized (correct shape, won't give meaningful output)
    g1_w = torch.zeros(E, 2 * I, H // 2, device=device, dtype=torch.uint8)
    g1_sf = torch.ones(E, 2 * I, H // 16, device=device, dtype=torch.float8_e4m3fn)
    g2_w = torch.zeros(E, H, I // 2, device=device, dtype=torch.uint8)
    g2_sf = torch.ones(E, H, I // 16, device=device, dtype=torch.float8_e4m3fn)
    scale_c = torch.ones(E, device=device, dtype=torch.float32)
    scale_g = torch.ones(E, device=device, dtype=torch.float32)

    print(f"\n  Weight memory:")
    g1_mb = g1_w.nelement() * g1_w.element_size() / 1e6
    g1_sf_mb = g1_sf.nelement() * g1_sf.element_size() / 1e6
    g2_mb = g2_w.nelement() * g2_w.element_size() / 1e6
    g2_sf_mb = g2_sf.nelement() * g2_sf.element_size() / 1e6
    print(f"    FC1 weights: {g1_w.shape} = {g1_mb:.1f} MB")
    print(f"    FC1 scales:  {g1_sf.shape} = {g1_sf_mb:.1f} MB")
    print(f"    FC2 weights: {g2_w.shape} = {g2_mb:.1f} MB")
    print(f"    FC2 scales:  {g2_sf.shape} = {g2_sf_mb:.1f} MB")
    print(f"    Total: {g1_mb + g1_sf_mb + g2_mb + g2_sf_mb:.1f} MB")

    from nvfp4_moe_cubin import nvfp4_moe_cubin

    print(f"\n  Warming up ({NUM_WARMUP} iters)...")
    for i in range(NUM_WARMUP):
        result = nvfp4_moe_cubin(
            routing_logits=logits.float(),
            hidden_states=hidden,
            gemm1_weights=g1_w,
            gemm1_weights_scale=g1_sf,
            gemm2_weights=g2_w,
            gemm2_weights_scale=g2_sf,
            num_experts=E, top_k=K, intermediate_size=I,
            output1_scale_scalar=scale_c,
            output1_scale_gate_scalar=scale_g,
            output2_scale_scalar=scale_c,
            routing_method_type=1,
        )
        torch.cuda.synchronize()
    print("  Warmup done.")

    # CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"\n  Profiling ({NUM_ITERS} iters)...")

    # Mark for nsys
    torch.cuda.nvtx.range_push("kimi_k2_moe_profile")

    start_event.record()
    for i in range(NUM_ITERS):
        torch.cuda.nvtx.range_push(f"moe_iter_{i}")
        result = nvfp4_moe_cubin(
            routing_logits=logits.float(),
            hidden_states=hidden,
            gemm1_weights=g1_w,
            gemm1_weights_scale=g1_sf,
            gemm2_weights=g2_w,
            gemm2_weights_scale=g2_sf,
            num_experts=E, top_k=K, intermediate_size=I,
            output1_scale_scalar=scale_c,
            output1_scale_gate_scalar=scale_g,
            output2_scale_scalar=scale_c,
            routing_method_type=1,
        )
        torch.cuda.nvtx.range_pop()
    end_event.record()

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    per_iter_ms = elapsed_ms / NUM_ITERS

    # Compute FLOPS
    # FC1: 2 * T * K * (2*I) * H = 2 * 128 * 8 * 4096 * 7168
    # FC2: 2 * T * K * H * I     = 2 * 128 * 8 * 7168 * 2048
    flops_fc1 = 2 * T * K * (2 * I) * H
    flops_fc2 = 2 * T * K * H * I
    total_flops = flops_fc1 + flops_fc2
    tflops = total_flops / (per_iter_ms / 1000) / 1e12

    print(f"\n  Results:")
    print(f"    Total time ({NUM_ITERS} iters): {elapsed_ms:.2f} ms")
    print(f"    Per iteration: {per_iter_ms:.3f} ms")
    print(f"    FC1 FLOPS: {flops_fc1/1e9:.1f} GFLOPS")
    print(f"    FC2 FLOPS: {flops_fc2/1e9:.1f} GFLOPS")
    print(f"    Total FLOPS: {total_flops/1e9:.1f} GFLOPS")
    print(f"    Throughput: {tflops:.2f} TFLOPS")
    print(f"    Output shape: {result[0].shape}")
    print()


if __name__ == "__main__":
    main()
