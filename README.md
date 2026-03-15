# trtllmgen_runner

FP4 × BF16 MOE on Blackwell using trtllm-gen exported cubins.

Supports two weight formats:
- **NvFP4 (E2m1)**: 16-element block scales, E4M3 scale dtype
- **MxFP4 (MxE2m1)**: 32-element block scales, E8M0 scale dtype, hardware F2FP upcast via SASS patching

## Build

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

Builds three libraries:
- `libmoe_cubin_lib.so` — NvFP4 (all Blackwell)
- `libmoe_mxfp4_cubin_lib_sm100a.so` — MxFP4 for SM100 (GB200)
- `libmoe_mxfp4_cubin_lib_sm103a.so` — MxFP4 for SM103

## Quick Start

```python
from mxfp4_moe_cubin import mxfp4_moe_cubin

output = mxfp4_moe_cubin(
    routing_logits, hidden_states,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    num_experts=256, top_k=8, intermediate_size=256,
    output1_scale_scalar=scale_c_fc1,
    output1_scale_gate_scalar=scale_gate_fc1,
    output2_scale_scalar=scale_c_fc2,
)
```

First call triggers autotuning (tries multiple tile_n values including 2CTA). Subsequent calls use cached best config.

## Benchmark

```bash
# MxFP4 TP8 (default)
python3 bench_moe.py

# NvFP4 TP1
python3 bench_moe.py --dtype nvfp4 --tp 1

# Single batch size
python3 bench_moe.py --tokens 128

# All options
python3 bench_moe.py --model dsv3 --dtype mxfp4 --tp 8 --tokens 1,16,128 --iters 20 --warmup 5
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `dsv3` | Model config: `dsv3` or `kimi_k2` |
| `--dtype` | `mxfp4` | Weight format: `nvfp4` or `mxfp4` |
| `--tp` | `8` | Tensor parallelism degree |
| `--tokens` | `1,16,128` | Comma-separated batch sizes |
| `--iters` | `20` | Benchmark iterations |
| `--warmup` | `5` | Warmup iterations (includes autotune) |
| `--nsys` | off | Enable `cudaProfilerStart/Stop` for nsys capture |

## nsys Profiling

```bash
nsys profile --capture-range=cudaProfilerApi -o mxfp4_tp8_bs128 \
  python3 bench_moe.py --nsys --dtype mxfp4 --tp 8 --tokens 128
```

The `--nsys` flag inserts `cudaProfilerStart/Stop` around the profiled iterations. Combined with `--capture-range=cudaProfilerApi`, nsys only captures steady-state kernel execution (autotune excluded).

## Performance (DSv3 TP8, B300 SM103, 80 SMs)

### MxFP4 (patchF2fp, 134 cubins, multi-tile_n autotune)

| BS | Pipeline | FC1 | FC1 BW | FC2 | FC2 BW | Strategy |
|----|----------|-----|--------|-----|--------|----------|
| 1 | 0.055ms | 10.8us | 1.46 TB/s (18%) | 13.9us | 0.57 TB/s (7%) | tile_n=8, splitK=2 |
| 16 | 0.115ms | 61.4us | 3.36 TB/s (42%) | 38.4us | 2.72 TB/s (34%) | tile_n=16, 2CTA |
| 128 | 0.170ms | 98.2us | 5.16 TB/s (65%) | 55.6us | 4.70 TB/s (59%) | tile_n=16, 2CTA |

### NvFP4 (baseline)

| BS | Pipeline | FC1 | FC2 |
|----|----------|-----|-----|
| 1 | 0.088ms | 52.5us | 22.3us |
| 16 | 0.418ms | 179.6us | 98.2us |
| 128 | 0.838ms | 382.2us | 210.3us |

### MxFP4 vs NvFP4 Speedup

| BS | NvFP4 | MxFP4 | Speedup |
|----|-------|-------|---------|
| 1 | 0.088ms | 0.055ms | 1.6x |
| 16 | 0.418ms | 0.115ms | 3.6x |
| 128 | 0.838ms | 0.170ms | 4.9x |

## Correctness Tests

```bash
python3 test_correctness.py          # NvFP4 (10 configs)
python3 test_mxfp4_correctness.py    # MxFP4 (10 configs)
```

## Weight Preparation

Requires TRT-LLM ops (run in a separate process to avoid symbol conflicts):

NvFP4:
```python
fp4, sf = torch.ops.trtllm.fp4_quantize(w, global_sf, 16, False, False)
```

MxFP4:
```python
fp4, sf = torch.ops.trtllm.fp4_quantize(w, global_sf, 32, True, False)
```

Both require `reorder_rows_for_gated_act_gemm` (FC1 only) and `shuffle_matrix_a/sf_a`.

## Architecture

| Component | Count | Source |
|-----------|-------|--------|
| NvFP4 cubins | 122 | trtllm-gen export |
| MxFP4 cubins | 134 (per arch) | trtllm-gen + SASS patch |
| Routing kernel | 1 | FlashInfer (cherry-picked) |
| Finalize kernel | 1 | FlashInfer (cherry-picked) |
| Cubin interface | — | trtllm-gen BatchedGemmInterface |
| Autotuning + fused pipeline | — | This project |

See `context.md` for full technical details.
