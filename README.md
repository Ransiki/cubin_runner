# trtllmgen_runner

NvFP4 (E2m1) × BF16 MOE on Blackwell (SM100) using trtllm-gen exported cubins.

## Build

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

Self-contained — all dependencies bundled (cubins, cutlass, FlashInfer kernels).

## Quick Start

```python
from nvfp4_moe_cubin import nvfp4_moe_cubin

output = nvfp4_moe_cubin(
    routing_logits, hidden_states,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    num_experts=256, top_k=8, intermediate_size=256,
    output1_scale_scalar=scale_c_fc1,
    output1_scale_gate_scalar=scale_gate_fc1,
    output2_scale_scalar=scale_c_fc2,
)
```

First call per shape triggers autotuning (~1s). Subsequent calls use cached best cubin config.

## Performance (DSv3 TP=8, B200)

| BS | Latency | Throughput |
|----|---------|------------|
| 1 | 0.095 ms | — |
| 16 | 0.31 ms | — |
| 128 | 0.62 ms | 18.2 TFLOPS |

## Weight Preparation

Requires TRT-LLM ops (run in a separate process):
```python
fp4, sf = torch.ops.trtllm.fp4_quantize(w, global_sf, 16, False, False)
fp4 = shuffle_matrix_a(reorder_rows_for_gated_act_gemm(fp4), 128)
sf = shuffle_matrix_sf_a(reorder_rows_for_gated_act_gemm(sf), 128)
```

## Tests

```bash
python3 test_correctness.py    # 10-config correctness vs dequant reference
python3 bench_moe.py --model dsv3 --tp 8 --tokens 128
python3 sweep_cubins.py --bs 1 16 128
```

## Architecture

| Component | Source |
|-----------|--------|
| FC1/FC2 cubins (72 kernels) | trtllm-gen export |
| Routing kernel | FlashInfer (cherry-picked) |
| Finalize kernel | FlashInfer (cherry-picked) |
| Cubin interface (BatchedGemmInterface) | trtllm-gen export |
| Autotuning + fused pipeline | This project |
| Weight quantization + shuffle | TRT-LLM ops |

See `context.md` for full technical details.
