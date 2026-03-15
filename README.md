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

`bench_moe.py` measures **end-to-end MOE pipeline latency** (routing → FC1 → FC2 → finalize):

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

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `dsv3` | Model config: `dsv3` or `kimi_k2` |
| `--dtype` | `mxfp4` | Weight format: `nvfp4` or `mxfp4` |
| `--tp` | `8` | Tensor parallelism degree |
| `--tokens` | `1,16,128` | Comma-separated batch sizes |
| `--iters` | `20` | Benchmark iterations |
| `--warmup` | `5` | Warmup iterations (includes autotune) |
| `--balanced` | off | Perfectly balanced routing across experts |
| `--nsys` | off | Enable `cudaProfilerStart/Stop` for nsys capture |
| `--force-fc1` | — | Force specific cubin config index for FC1 |
| `--force-fc2` | — | Force specific cubin config index for FC2 |
| `--list-configs` | — | List all valid cubin configs, then exit |

## nsys Kernel Bandwidth

`nsys_analyze.sh` profiles the MOE pipeline under nsys and computes **per-kernel achieved memory bandwidth** from real GPU kernel execution times. It accepts the same CLI as `bench_moe.py`:

```bash
# Same args as bench_moe.py
./nsys_analyze.sh --dtype mxfp4 --tp 8 --tokens 1,16,128
./nsys_analyze.sh --dtype mxfp4 --tp 8 --tokens 128 --balanced
./nsys_analyze.sh --dtype nvfp4 --tp 1 --tokens 128
```

Example output (B300, DSv3 TP8 MxFP4):

```
    BS     Active  Pipe(us) │  FC1(us)    FC1 BW  FC1% │  FC2(us)    FC2 BW  FC2%
  ──── ────────── ───────── ┼ ──────── ───────── ───── ┼ ──────── ───────── ─────
     1    8/256      126.5 │     11.5   1.372T/s   17% │     13.5   0.585T/s    7%
    16  102/256      120.8 │     48.2   4.162T/s   52% │     33.1   3.060T/s   38%
   128  251/256      207.3 │    104.7   4.817T/s   60% │     64.1   4.058T/s   51%
```

**How bandwidth is computed:**
- Kernel execution time comes from nsys `cuda_gpu_kern_sum` (real GPU time, no launch overhead)
- FC1 data = `active × 2I × (H/2 + H/sf_block)` (weights) + `expanded × (H + I) × 2` (activations)
- FC2 data = `active × H × (I/2 + I/sf_block)` (weights) + `expanded × (I + H) × 2` (activations)
- `active` = experts with ≥1 token (only active experts' weights are loaded from HBM)
- BW = total_bytes / kernel_time

**Why nsys instead of CUDA events:** CUDA event timing wraps the Python→C→kernel launch loop, which includes CPU-side overhead. For short kernels (BS=1, kernel <15us), this gap dominates and inflates the measured time by 20-80us. nsys measures pure GPU execution time and gives accurate bandwidth at all batch sizes.


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

