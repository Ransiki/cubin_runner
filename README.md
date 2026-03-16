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
# End-to-end pipeline latency + bandwidth
python3 bench_moe.py                                       # MxFP4 TP8 DSv3
python3 bench_moe.py --dtype nvfp4 --tp 1                  # NvFP4 TP1
python3 bench_moe.py --tokens 128                           # single BS
python3 bench_moe.py -v                                     # show autotune logs

# CUDA Graph: eliminate kernel launch overhead (significant for BS=1)
python3 bench_moe.py --cuda-graph --tokens 1,16,128

# Per-kernel bandwidth via nsys
./nsys_analyze.sh --dtype mxfp4 --tp 8 --tokens 1,16,128
```

Reports saved to `nsysrep/<model>_<dtype>_tp<N>_<timestamp>/`.

### Reference Performance (Blackwell 148 SMs, DSv3 TP8 MxFP4, F2FP patched)

**bench_moe.py** (end-to-end):
```
    BS     Active  Pipe(us)        BW    Data  tN │ FC1 config                                  │ FC2 config
  ──── ────────── ───────── ───────── ─────── ─── ┼ ──────────────────────────────────────────── ┼ ──────────────────────────────────────────
     1    8/256       57.0   0.414T/s    24MB   8 │ t128x8x128 s5/2 persistent u2 splitK=2       │ t128x8x256 s3/2 persistent
    16  102/256       98.8   3.058T/s   302MB  16 │ t128x16x256 s3/2 mma256 persistent u2 c2     │ t128x16x256 s3/2 persistent
   128  251/256      198.5   3.852T/s   764MB  16 │ t128x16x256 s3/2 mma256 persistent u2 c2     │ t128x16x256 s3/2 persistent
```

**bench_moe.py --cuda-graph** (CUDA Graph replay, eliminates launch overhead):
```
    BS     Active  Pipe(us)        BW    Data  tN │ FC1 config                                  │ FC2 config
  ──── ────────── ───────── ───────── ─────── ─── ┼ ──────────────────────────────────────────── ┼ ──────────────────────────────────────────
     1    8/256       39.9   0.592T/s    24MB   8 │ t128x8x128 s5/2 persistent u2 splitK=2       │ t128x8x256 s3/2 persistent
    16  102/256       98.6   3.065T/s   302MB  16 │ t128x16x256 s3/2 mma256 persistent u2 c2     │ t128x16x256 s3/2 persistent
   128  251/256      256.4   2.981T/s   764MB  16 │ t128x16x256 s3/2 mma256 persistent u2 c2     │ t128x16x256 s3/2 persistent
```

CUDA Graph reduces BS=1 pipeline from 57us to 40us (**-30%**) by eliminating per-kernel CPU launch overhead. No effect on BS≥16 where kernel time dominates.

**nsys_analyze.sh** (per-kernel):
```
    BS     Active  Pipe(us) │  FC1(us)    FC1 BW  FC1% │  FC2(us)    FC2 BW  FC2%
  ──── ────────── ───────── ┼ ──────── ───────── ───── ┼ ──────── ───────── ─────
     1    8/256       74.7 │     13.6   1.160T/s   14% │     13.4   0.591T/s    7%
    16  102/256      107.2 │     50.5   3.978T/s   50% │     32.6   3.109T/s   39%
   128  251/256      204.2 │    107.0   4.714T/s   59% │     62.0   4.191T/s   52%
```

## Correctness

```bash
python3 test_correctness.py                        # routing only (fast)
python3 test_correctness.py --gemm                 # routing + GEMM
python3 test_correctness.py --stress --dtype mxfp4  # 27-case stress test
```

MxFP4 GEMM verified against TRT-LLM `Bf16MxE2m1BlockScaleMoERunner` (cos ≥ 0.999995).
NvFP4 GEMM verified against Python dequant reference.
