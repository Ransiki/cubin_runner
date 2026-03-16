# trtllmgen_runner

FP4 × BF16 MOE on Blackwell using trtllm-gen exported cubins.

Supports two weight formats:
- **NvFP4 (E2m1)**: 16-element block scales, E4M3 scale dtype
- **MxFP4 (MxE2m1)**: 32-element block scales, E8M0 scale dtype, F2FP hardware upcast via SASS patching

## Build

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

Builds:
- `libmoe_cubin_lib.so` — NvFP4
- `libmoe_mxfp4_cubin_lib_sm100a.so` — MxFP4 (172 F2FP patched cubins)

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

First call triggers autotuning. Use `-v` flag on bench_moe.py to see autotune details.

## Weight Preparation

MxFP4 (gsf=None, block_size=32, E8M0 scales):
```python
fp4, sf = torch.ops.trtllm.fp4_quantize(w, None, 32, True, False)
```

NvFP4 (gsf computed, block_size=16, E4M3 scales):
```python
gsf = (448.0 * 6.0) / w.float().abs().nan_to_num().max()
fp4, sf = torch.ops.trtllm.fp4_quantize(w, gsf, 16, False, False)
```

Both require `reorder_rows_for_gated_act_gemm` (FC1), `shuffle_matrix_a`, and `shuffle_matrix_sf_a(num_elts_per_sf=32|16)` from TRT-LLM's `fp4_utils`.

## Benchmark

```bash
# End-to-end pipeline latency + bandwidth (CUDA events timing)
python3 bench_moe.py                                       # MxFP4 TP8 DSv3
python3 bench_moe.py --cuda-graph                           # with CUDA Graph (faster BS=1)
python3 bench_moe.py -v                                     # show autotune logs

# Per-kernel bandwidth via nsys (pure GPU time, L2 flush between iters)
./nsys_analyze.sh --dtype mxfp4 --tp 8 --tokens 1,16,128
```

Reports saved to `nsysrep/<model>_<dtype>_tp<N>_<timestamp>/`.

### Reference Performance (Blackwell 148 SMs, DSv3 TP8 MxFP4, F2FP patched)

**nsys per-kernel** (pure GPU time with L2 flush, most accurate):
```
    BS     Active  Pipe(us) │  FC1(us)    FC1 BW  FC1% │  FC2(us)    FC2 BW  FC2%
  ──── ────────── ───────── ┼ ──────── ───────── ───── ┼ ──────── ───────── ─────
     1    8/256       74.7 │     13.6   1.158T/s   14% │     13.4   0.591T/s    7%
    16  102/256      103.7 │     50.4   3.986T/s   50% │     31.3   3.236T/s   40%
   128  251/256      204.1 │    107.0   4.716T/s   59% │     62.0   4.190T/s   52%
```

**CUDA Graph effect on BS=1** (`bench_moe.py --cuda-graph`):
```
               default     --cuda-graph    speedup
    BS=1       57 us       40 us           -30%
    BS=16      99 us       99 us           (same)
    BS=128    199 us      199 us           (same)
```

CUDA Graph eliminates per-kernel CPU launch overhead (~17us for 4 kernels). Significant for BS=1 where launch overhead is 30% of total; negligible for BS≥16 where GPU kernel time dominates.

**How bandwidth is computed:**
- Kernel time from nsys `cuda_gpu_kern_sum` (real GPU time, no launch overhead)
- FC1 data = `active × 2I × (H/2 + H/sf_block)` + `expanded × (H + I) × 2`
- FC2 data = `active × H × (I/2 + I/sf_block)` + `expanded × (I + H) × 2`
- `active` = experts with ≥1 token, `expanded` = T × top_k
- BW = total_bytes / kernel_time

## Correctness

```bash
python3 test_correctness.py                        # routing only (fast)
python3 test_correctness.py --gemm                 # routing + GEMM
python3 test_correctness.py --stress --dtype mxfp4  # 27-case stress test
```

MxFP4 GEMM verified against TRT-LLM `Bf16MxE2m1BlockScaleMoERunner` (cos ≥ 0.999995).
NvFP4 GEMM verified against Python dequant reference.
