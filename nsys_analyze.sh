#!/bin/bash
# nsys_analyze.sh — Run bench_moe under nsys, compute FC1/FC2 kernel bandwidth.
#
# Same CLI as bench_moe.py so you can compare directly:
#   python3 bench_moe.py --dtype mxfp4 --tp 8 --tokens 1,16,128
#   ./nsys_analyze.sh    --dtype mxfp4 --tp 8 --tokens 1,16,128
#
# Usage:
#   ./nsys_analyze.sh                                          # mxfp4 tp8 bs=1,16,128
#   ./nsys_analyze.sh --dtype mxfp4 --tp 8 --tokens 128
#   ./nsys_analyze.sh --dtype mxfp4 --tp 8 --tokens 1,16,128 --balanced
#   ./nsys_analyze.sh --dtype nvfp4 --tp 1 --tokens 128
#   ./nsys_analyze.sh --force-fc1 42 --force-fc2 31 --tokens 128

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults (same as bench_moe.py) ──
DTYPE="mxfp4"
TP=8
TOKENS="1,16,128"
ITERS=20
WARMUP=5
MODEL="dsv3"
BALANCED=""
FORCE_FC1=""
FORCE_FC2=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)      DTYPE="$2";      shift 2;;
        --tp)         TP="$2";         shift 2;;
        --tokens)     TOKENS="$2";     shift 2;;
        --iters)      ITERS="$2";      shift 2;;
        --warmup)     WARMUP="$2";     shift 2;;
        --model)      MODEL="$2";      shift 2;;
        --balanced)   BALANCED="--balanced"; shift;;
        --force-fc1)  FORCE_FC1="--force-fc1 $2"; shift 2;;
        --force-fc2)  FORCE_FC2="--force-fc2 $2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ── Model parameters ──
case "$MODEL" in
    dsv3)    H=7168; I_FULL=2048; E=256; K=8;;
    kimi_k2) H=7168; I_FULL=2048; E=384; K=8;;
    *) echo "Unknown model: $MODEL"; exit 1;;
esac
I=$((I_FULL / TP))
case "$DTYPE" in
    mxfp4) SF_BLOCK=32;;
    nvfp4) SF_BLOCK=16;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${SCRIPT_DIR}/nsysrep/${MODEL}_${DTYPE}_tp${TP}_${TIMESTAMP}"
mkdir -p "$OUTDIR"
TMPDIR="$OUTDIR"

# ── Collect results per BS ──
IFS=',' read -ra BS_LIST <<< "$TOKENS"
RESULTS_FILE="${TMPDIR}/results.jsonl"
> "$RESULTS_FILE"

for BS in "${BS_LIST[@]}"; do
    REPORT="${TMPDIR}/bs${BS}"
    BENCH_OUTPUT="${TMPDIR}/bench_bs${BS}.log"

    >&2 echo "  Profiling BS=${BS} ..."

    nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        -o "$REPORT" \
        --force-overwrite=true \
        --stats=false \
        python3 "${SCRIPT_DIR}/bench_moe.py" \
            --nsys \
            --dtype "$DTYPE" \
            --tp "$TP" \
            --tokens "$BS" \
            --model "$MODEL" \
            --iters "$ITERS" \
            --warmup "$WARMUP" \
            $BALANCED $FORCE_FC1 $FORCE_FC2 \
        > "$BENCH_OUTPUT" 2>&1

    # Parse from table format: "  128  251/256      255.9 ..."
    ACTIVE=$(grep -oP '\s+\d+/\d+' "$BENCH_OUTPUT" | head -1 | grep -oP '\d+(?=/)' || echo "0")
    PIPELINE_US=$(grep -oP '\d+/\d+\s+\K[0-9.]+' "$BENCH_OUTPUT" | head -1 || echo "0")

    KERN_CSV="${TMPDIR}/kern_bs${BS}.csv"
    nsys stats -r cuda_gpu_kern_sum -f csv --timeunit nsec "${REPORT}.nsys-rep" \
        > "$KERN_CSV" 2>/dev/null

    # Extract FC1/FC2 kernel times and write one JSON line
    python3 - "$KERN_CSV" "$BS" "$ACTIVE" "$PIPELINE_US" \
             "$H" "$I" "$E" "$K" "$SF_BLOCK" >> "$RESULTS_FILE" <<'PYEOF'
import csv, io, sys, json

kern_csv_path = sys.argv[1]
T       = int(sys.argv[2])
active  = int(sys.argv[3])
pipe_us = float(sys.argv[4])
H, I, E, K, SF = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
expanded = T * K

with open(kern_csv_path) as f:
    text = f.read()
lines = text.strip().split('\n')
csv_start = next((i for i, l in enumerate(lines) if 'Time (%)' in l), 0)
csv_text = '\n'.join(lines[csv_start:])

by_cat = {}
for row in csv.DictReader(io.StringIO(csv_text)):
    name = row.get("Name", "")
    nl = name.lower()
    if "bmm_" not in nl:
        continue
    cat = "fc1" if "swiglu" in nl else "fc2"
    avg = float(row.get("Avg (ns)", row.get("Avg", 0)))
    if cat not in by_cat or avg > by_cat[cat]["avg"]:
        by_cat[cat] = {"avg": avg, "med": float(row.get("Med (ns)", row.get("Med", 0))),
                        "n": int(row.get("Instances", 0)), "name": name}

fc1_w = active * (2*I) * (H // 2) + active * (2*I) * (H // SF)
fc1_act = expanded * H * 2 + expanded * I * 2
fc2_w = active * H * (I // 2) + active * H * (I // SF)
fc2_act = expanded * I * 2 + expanded * H * 2

r = {"T": T, "active": active, "E": E, "pipe_us": pipe_us}
for cat, w, act in [("fc1", fc1_w, fc1_act), ("fc2", fc2_w, fc2_act)]:
    k = by_cat.get(cat, {})
    avg_us = k.get("avg", 0) / 1e3
    med_us = k.get("med", 0) / 1e3
    total = w + act
    bw = (total / 1e12) / (avg_us / 1e6) if avg_us > 0 else 0
    r[cat] = {"avg_us": round(avg_us, 1), "med_us": round(med_us, 1),
              "w_mb": round(w/1e6, 1), "act_mb": round(act/1e6, 1),
              "total_mb": round(total/1e6, 1), "bw": round(bw, 3)}
print(json.dumps(r))
PYEOF

    rm -f "${REPORT}.sqlite"
done

# ── Print summary table ──
ROUTING_MODE="random"
[[ -n "$BALANCED" ]] && ROUTING_MODE="balanced"

python3 - "$RESULTS_FILE" "$MODEL" "$TP" "$DTYPE" "$H" "$I" "$E" "$K" "$SF_BLOCK" \
         "$ITERS" "$WARMUP" "$ROUTING_MODE" <<'PYEOF'
import sys, json

results_file = sys.argv[1]
MODEL    = sys.argv[2].upper()
TP       = int(sys.argv[3])
DTYPE    = sys.argv[4].upper()
H, I, E, K, SF = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
ITERS    = int(sys.argv[10])
WARMUP   = int(sys.argv[11])
ROUTING  = sys.argv[12]
PEAK_BW  = 8.0

with open(results_file) as f:
    results = [json.loads(line) for line in f if line.strip()]

print()
print(f"{'='*86}")
print(f"  {MODEL} TP{TP} — {DTYPE} MOE nsys Kernel Bandwidth")
print(f"{'='*86}")
print(f"  Model:   H={H} I={I} E={E} K={K}  sf_block={SF}")
print(f"  Routing: {ROUTING}   Iters: {ITERS} (warmup={WARMUP})")
print()

# Summary table
hdr  = f"  {'BS':>4} {'Active':>10} {'Pipe(us)':>9}"
hdr += f" │ {'FC1(us)':>8} {'FC1 BW':>9} {'FC1%':>5}"
hdr += f" │ {'FC2(us)':>8} {'FC2 BW':>9} {'FC2%':>5}"
sep  = f"  {'─'*4} {'─'*10} {'─'*9}"
sep += f" ┼ {'─'*8} {'─'*9} {'─'*5}"
sep += f" ┼ {'─'*8} {'─'*9} {'─'*5}"

print(hdr)
print(sep)

for r in results:
    T = r["T"]; active = r["active"]; E_ = r["E"]
    pipe = r["pipe_us"]
    f1 = r["fc1"]; f2 = r["fc2"]
    f1_pct = f1["bw"] / PEAK_BW * 100 if f1["bw"] > 0 else 0
    f2_pct = f2["bw"] / PEAK_BW * 100 if f2["bw"] > 0 else 0
    line  = f"  {T:>4} {active:>4}/{E_:<4} {pipe:>9.1f}"
    line += f" │ {f1['avg_us']:>8.1f} {f1['bw']:>7.3f}T/s {f1_pct:>4.0f}%"
    line += f" │ {f2['avg_us']:>8.1f} {f2['bw']:>7.3f}T/s {f2_pct:>4.0f}%"
    print(line)

print()

# Data volume detail
print(f"  Data volumes (weight = active_experts × per-expert, act = expanded × dim × 2B):")
for r in results:
    T = r["T"]; active = r["active"]; expanded = T * K
    f1 = r["fc1"]; f2 = r["fc2"]
    print(f"    BS={T:<4} active={active:>3}/{E}  "
          f"FC1: w={f1['w_mb']:.1f}+a={f1['act_mb']:.1f}={f1['total_mb']:.1f}MB  "
          f"FC2: w={f2['w_mb']:.1f}+a={f2['act_mb']:.1f}={f2['total_mb']:.1f}MB")
print()
PYEOF

echo "  Reports saved to: ${OUTDIR}/"
ls -1 "${OUTDIR}"/*.nsys-rep 2>/dev/null | while read f; do echo "    $(basename $f)"; done
