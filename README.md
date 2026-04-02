# trtllmgen_runner

FP4 × BF16 MoE kernels and benchmark workflow on Blackwell.

Supports two weight formats:

- **NvFP4 (E2m1)**: 16-element block scales, E4M3 scale dtype
- **MxFP4 (MxE2m1)**: 32-element block scales, E8M0 scale dtype, F2FP hardware upcast via SASS patching

## Build

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

Build outputs include:

- `libmoe_mxfp4_cubin_lib_sm100a.so` (fused path)
- `libmoe_mxfp4_nofuse_cubin_lib_sm100a.so` (nofuse path)

## Quick Start

```python
from mxfp4_moe_cubin import mxfp4_moe_cubin

output = mxfp4_moe_cubin(
    routing_logits, hidden_states,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    num_experts=256, top_k=8, intermediate_size=256,
    output1_scale_scalar=scale_c,
    output1_scale_gate_scalar=scale_gate,
    output2_scale_scalar=scale_c_fc2,
)
```

First call triggers autotuning for the shape.

## Benchmark Usage

### 1) One-model end-to-end benchmark (recommended)

Run from `trtllmgen_runner`:

```bash
# dsv3 (with mxint4)
bash benchmark_example/benchmark_dsv3_with_mxint4.sh \
  /ran/4bit/benchmark_results/$(date +%Y%m%d_%H%M%S)

# kimi_k2 (with mxint4)
bash benchmark_example/benchmark_kimi_k2_with_mxint4.sh \
  /ran/4bit/benchmark_results/$(date +%Y%m%d_%H%M%S)

```

Each script will:

- Run `run_bench.py` for both `TP8/EP1` and `TP1/EP8`
- Sweep batch sizes `1..8192` (powers of 2)
- Write CSV files (`tp8_ep1_e2e.csv`, `tp1_ep8_e2e.csv`)
- Generate plots with `plot_results.py`

Default env:

- `ITERS=20`
- persistent cache under `trtllmgen_runner/.benchmark_cache`

### 2) Run custom config directly

```bash
python3 run_bench.py \
  --H 7168 --moe-I 2048 --E 256 --K 8 \
  --tp 8 --ep 1 \
  --bs "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192" \
  --backend "mxfp4,mxfp4_nf,deepgemm" \
  --iters 20 \
  --csv /ran/4bit/benchmark_results/custom_run/dsv3/tp8_ep1_e2e.csv

python3 plot_results.py --input-dir /ran/4bit/benchmark_results/custom_run/dsv3
```

### 3) `benchmark_example` scripts (dsv3 / kimi_k2 with `mxint4`)

The example scripts are under `trtllmgen_runner/benchmark_example/`.
Run from `trtllmgen_runner`:

```bash
# dsv3 + mxint4
bash benchmark_example/benchmark_dsv3_with_mxint4.sh \
  /ran/4bit/benchmark_results/$(date +%Y%m%d_%H%M%S)_example

# kimi_k2 + mxint4
bash benchmark_example/benchmark_kimi_k2_with_mxint4.sh \
  /ran/4bit/benchmark_results/$(date +%Y%m%d_%H%M%S)_example
```

Defaults used by both scripts:

- `BACKEND=mxfp4,mxfp4_nf,mxint4,deepgemm`
- `ITERS=20`
- `BS=1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192`

## Correctness

```bash
python3 test_correctness.py
python3 test_correctness.py --gemm
python3 test_correctness.py --stress --dtype mxfp4
```
