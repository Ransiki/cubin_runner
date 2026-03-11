# TRT-LLM Gen Cubin Runner — Full Context

## 1. Project Goal

Build a standalone Python-callable **NvFP4 (E2m1) × BF16 MOE** pipeline using pre-compiled CUDA kernels (cubins) from trtllm-gen, targeting **Blackwell SM100** GPUs.

This dtype combination (NvFP4 weights × BF16 activations) is **not natively supported** by FlashInfer or TRT-LLM — both only support MxFP4×BF16 or NvFP4×NvFP4. Our cubins fill this gap.

## 2. MOE Computation Flow

```
hidden_states [T, H] bf16
     │
     ▼ Routing kernel (FlashInfer cherry-pick)
topk_weights [T, K] bf16  +  permutation metadata (GPU buffers)
     │
     ▼ FC1 cubin (fused permute + GEMM + SwiGLU)
     │  Reads tokens from scattered positions via routeAct=ldgsts
     │  Weight: [E, 2I, H/2] NvFP4  →  Output: [padded, I] bf16
     │
     ▼ FC2 cubin (plain batched GEMM)
     │  Weight: [E, H, I/2] NvFP4  →  Output: [padded, H] bf16
     │
     ▼ Finalize kernel (FlashInfer cherry-pick)
     │  Unpermute + scale by routing weight + reduce across top_k
     │
output [T, H] bf16
```

Dimensions (transposeMmaOutput=true, batchN mode):
- FC1: M=2*I (gate+up), N=tokens_per_expert (batched), K=H (reduction)
- FC2: M=H (down), N=tokens_per_expert (batched), K=I (reduction)

## 3. Architecture

```
Python (nvfp4_moe_cubin.py)
  │
  ├─ Autotune: benchmark all valid cubins per shape (first call only)
  │   └─ L2 flush before warmup for realistic timing
  │   └─ Result cached in-memory (dict keyed by shape)
  │
  └─ moe_cubin_fused_run() ← single ctypes call, zero CPU-GPU sync mid-pipeline
      │
      C++ (moe_cubin_lib.cu)
      ├─ Step 1: FlashInfer routing kernel (async on stream)
      ├─ Step 2: Dummy batched_n (no sync needed — cubin uses CTA mapping at runtime)
      ├─ Step 3: FC1 cubin launch (async)
      ├─ Step 4: FC2 cubin launch (async)
      └─ Return to Python
  │
  └─ moe_cubin_finalize() ← FlashInfer fused finalize kernel
      Unpermute + weighted sum in one kernel (replaces 7 PyTorch ops)
```

## 4. Cubin Details

### 72 cubins (extended config)

Generated from `batched_gemm_tllm_config_ext_safe.json`:
- FC1: 48 variants (tileN={8,16,32,64} × tileK={128,256} × scheduler={static,persistent} × stages × unroll)
- FC2: 24 variants

All cubins: `dtypeA=E2m1, dtypeB=BF16, dtypeC=BF16, smVersion=100f, transposeMmaOutput=true, useShuffledMatrix=true`

### Cubin selection (autotuning)

1. `moe_cubin_find_valid_configs()` — filter by dtype, routeAct, fusedAct, tileN
2. `moe_cubin_autotune()` — benchmark each valid cubin (warmup + bench iterations), L2 flush before warmup
3. Cache best config_index per (is_fc1, tile_n, M, K, num_experts)
4. `moe_cubin_run()` — launch by config_index with persistent ModuleCache (avoids cuModuleLoadData per call)

### Key insight: dummy batched_n

For cubins with `earlyExit=true` (all our cubins), `BatchedGemmInterface::run()` uses `mMaxNumCtasInBatchDim` for grid sizing, NOT `mBatchedN`. So we pass dummy `batched_n = [tile_n, tile_n, ...]` and skip the GPU→CPU sync that would be needed to compute real per-expert token counts. This eliminates ~70μs bubble between routing and FC1.

## 5. Weight Preparation

Weights must be pre-processed with TRT-LLM ops (in a separate process due to symbol conflicts):

```python
# 1. Quantize to NvFP4
global_sf = (448 * 6) / weights.float().abs().max()
fp4, sf = torch.ops.trtllm.fp4_quantize(weights, global_sf, 16, False, False)

# 2. Reorder rows for gated activation (FC1 only)
fp4 = reorder_rows_for_gated_act_gemm(fp4)
sf = reorder_rows_for_gated_act_gemm(sf)

# 3. Shuffle matrix for TMA hardware
fp4 = shuffle_matrix_a(fp4.view(torch.uint8), 128)
sf = shuffle_matrix_sf_a(sf.view(torch.uint8), 128)
```

Scale factors for runtime:
```
scale_c_fc1     = c_global_sf / gemm1_global_sf / hidden_states_global_sf
scale_gate_fc1  = 1.0 / gemm1_global_sf / hidden_states_global_sf
scale_c_fc2     = 1.0 / c_global_sf / gemm2_global_sf
For W4A16 (bf16 activations): hidden_states_global_sf=1.0, c_global_sf=1.0
```

## 6. Cherry-picked Components

| Component | Source | Files | Modifications |
|-----------|--------|-------|---------------|
| Routing kernel | FlashInfer | `fi_routing/trtllm_fused_moe_routing_*.cu`, `RoutingKernel.cuh/.h` | TVM FFI macros → assert stubs |
| Finalize kernel | FlashInfer | `fi_routing/trtllm_fused_moe_dev_kernel.cu`, `DevKernel.h` | Same stubs |
| Cubin interface | trtllm-gen export | `trtllmGen_bmm_export/BatchedGemmInterface.h` | None (used as-is) |
| Cubin data | trtllm-gen export | `trtllmGen_bmm_export/cubins/*.cpp` | None |
| cutlass | trtllm-gen 3rdparty | `3rdparty/cutlass/` | None |

## 7. Performance (DSv3 TP=8, B200 80 SMs)

### Latency breakdown (BS=128)

```
Routing kernel:     12 μs
CPU gap:            13 μs  (BatchedGemmData construction + cuLaunchKernel)
FC1 kernel:        382 μs  (autotuned: tileK=128 stages=4/2 persistent u2)
FC2 kernel:        210 μs  (autotuned: tileK=128 stages=5/2 persistent u2)
Finalize kernel:    ~5 μs  (fused FlashInfer kernel)
────────────────────────
Total:             620 μs
```

### Optimization history

| Version | Per-iter | Improvement |
|---------|----------|-------------|
| Initial (separate routing/FC1/FC2/finalize) | 1.18 ms | baseline |
| + Fused C pipeline (routing→FC1→FC2 one call) | 0.76 ms | -36% |
| + Eliminate CPU-GPU sync (dummy batched_n) | 0.70 ms | -8% |
| + Fused finalize kernel | 0.62 ms | -11% |
| **Total** | **0.62 ms** | **-47%** |

### Bandwidth analysis

| BS | Time | Achieved BW | B200 util |
|----|------|-------------|-----------|
| 1 | 0.095ms | 8.35 TB/s | 104% (L2 cached) |
| 16 | 0.311ms | 2.56 TB/s | 32% |
| 128 | 0.619ms | 1.33 TB/s | 17% |
| 1024 | 1.210ms | 0.85 TB/s | 11% |

Low utilization at large BS due to: 256 experts × fragmented CTA launches, small per-expert GEMM size.

### Config sweep findings (DSv3 TP=8)

- **FC1**: static scheduler beats persistent for small batch; tileK=128 > tileK=256 (K=7168, deeper pipeline)
- **FC2**: persistent scheduler 20-30% faster than static; tileK=128 >> tileK=256 when K=256 (2 tiles vs 1 tile pipeline)
- **Extended cubins** (tileK=128 for FC2): 34-38% faster than original config

## 8. Correctness

Validated using TRT-LLM methodology:
- Reference: `dequant(quant(weights))` → BF16 MOE → compare vs cubin output
- Criterion: `|a-b| ≤ 0.1 + 0.85*|b|` for 92.5% of elements
- 10 test configs all PASS (including DSv3 TP8 BS=1/16/128, K2 TP8, edge cases)

## 9. Known Limitations

1. **Symbol conflict**: `libmoe_cubin_lib.so` and TRT-LLM's `libth_common.so` cannot coexist (duplicate `BatchedGemmOptions`). Weight prep must run in separate process.
2. **SM100 only**: Cubins target Blackwell. Won't work on Hopper.
3. **NvFP4 (E2m1) only**: MxFP4 (MxE2m1) needs different cubins.
4. **earlyExit assumption**: Dummy batched_n trick only works for earlyExit=true cubins. staticBatch cubins would need real batched_n (with GPU→CPU sync).

## 10. File Layout

```
trtllmgen_runner/
├── CMakeLists.txt              Self-contained build
├── README.md
├── context.md                  This file
├── moe_cubin_lib.cu            C API: autotune + fused run + finalize
├── nvfp4_moe_cubin.py          Python API with autotuning cache
├── bench_moe.py                Performance benchmark (--model dsv3/kimi_k2 --tp N --tokens N)
├── sweep_cubins.py             Cubin config sweep tool
├── test_correctness.py         Two-process correctness test (10 configs)
├── _phase1_quantize.py         TRT-LLM weight quantization (Phase 1)
├── _phase2_cubin_run.py        Cubin execution (Phase 2)
├── trtllmGen_bmm_export/       72 cubins + interface headers
├── 3rdparty/cutlass/           cutlass headers (BSD-3-Clause)
├── test_stubs/                 Namespace macro stubs
└── fi_routing/                 Cherry-picked FlashInfer kernels
    ├── routing_wrapper.cu      Routing C API
    ├── trtllm_fused_moe_routing_renormalize.cu
    ├── trtllm_fused_moe_routing_deepseek.cu
    ├── trtllm_fused_moe_dev_kernel.cu    Finalize + activation kernels
    ├── tvm_ffi_utils.h         TVM FFI stubs
    └── include/                Headers (RoutingKernel, DevKernel, etc.)
```
