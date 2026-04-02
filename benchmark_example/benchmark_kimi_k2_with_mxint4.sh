#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RUNNER_DIR/.." && pwd)"

cd "$RUNNER_DIR"

export TRTLLMGEN_BENCH_CACHE_BASE="$RUNNER_DIR/.benchmark_cache"
export FLASHINFER_WORKSPACE_BASE="$TRTLLMGEN_BENCH_CACHE_BASE/flashinfer"
export FLASHINFER_CUBIN_DIR="$FLASHINFER_WORKSPACE_BASE/cubins"
export TORCH_EXTENSIONS_DIR="$TRTLLMGEN_BENCH_CACHE_BASE/torch_extensions"
export XDG_CACHE_HOME="$TRTLLMGEN_BENCH_CACHE_BASE"

OUT_ROOT="${1:-$WORKSPACE_ROOT/benchmark_results/$(date +%Y%m%d_%H%M%S)_example}"
OUT_DIR="$OUT_ROOT/kimi_k2"
mkdir -p "$OUT_DIR"

BS="${BS:-1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192}"
ITERS="${ITERS:-20}"
BACKEND="${BACKEND:-mxfp4,mxfp4_nf,mxint4,deepgemm}"

python3 run_bench.py --H 7168 --moe-I 2048 --E 384 --K 8 \
  --tp 8 --ep 1 --bs "$BS" --backend "$BACKEND" --iters "$ITERS" \
  --csv "$OUT_DIR/tp8_ep1_e2e.csv"

python3 run_bench.py --H 7168 --moe-I 2048 --E 384 --K 8 \
  --tp 1 --ep 8 --bs "$BS" --backend "$BACKEND" --iters "$ITERS" \
  --csv "$OUT_DIR/tp1_ep8_e2e.csv"

python3 plot_results.py --input-dir "$OUT_DIR"
echo "done: $OUT_DIR"
