# trtllmgen_runner

NvFP4 (E2m1) × BF16 MOE on Blackwell using trtllm-gen exported cubins.

## Build

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

Fully self-contained. Requires only CUDA 13+ and an SM100 GPU.

## Usage

```python
from nvfp4_moe_cubin import nvfp4_moe_cubin

output = nvfp4_moe_cubin(
    routing_logits, hidden_states,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    num_experts=E, top_k=K, intermediate_size=I,
    output1_scale_scalar=scale_c_fc1,
    output1_scale_gate_scalar=scale_gate_fc1,
    output2_scale_scalar=scale_c_fc2,
)
```

Weights must be pre-processed with TRT-LLM's quantization utilities:
`fp4_quantize` → `reorder_rows_for_gated_act_gemm` → `shuffle_matrix_a` / `shuffle_matrix_sf_a`.

## Files

| File | Purpose |
|------|---------|
| `moe_cubin_lib.cu` | C API: cubin selection + FC1/FC2 launch |
| `nvfp4_moe_cubin.py` | Python API: routing + cubin calls + finalize |
| `fi_routing/` | FlashInfer routing kernel (cherry-picked) |
| `bench_moe.py` | Performance profiling (Kimi-K2, DSv3 configs) |
| `test_correctness.py` | Two-process correctness test |

## Module Sources

| Component | Source |
|-----------|--------|
| FC1/FC2 cubins (48 kernels) | trtllm-gen export |
| BatchedGemmInterface | trtllm-gen export |
| Routing kernel | FlashInfer (cherry-picked) |
| Weight quantization + shuffle | TRT-LLM `torch.ops.trtllm.*` |
| Python wrapper + finalize | This project |

## Limitations

- TRT-LLM and this library cannot be loaded in the same process (symbol conflict). Weight prep and cubin execution must run in separate processes.
- SM100 (Blackwell) only.
- NvFP4 (E2m1) weights only. MxFP4 (MxE2m1) requires different cubins.
