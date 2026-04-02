"""Subprocess-isolated E2E benchmark runner."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from benchmark_env import ensure_persistent_benchmark_env


def _parse_last_json_line(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("No JSON row found in bench_e2e output")


def _run_one_bs(args, bs: int) -> dict:
    cmd = [
        sys.executable,
        "bench_e2e.py",
        "--H", str(args.H),
        "--moe-I", str(args.moe_I),
        "--E", str(args.E),
        "--K", str(args.K),
        "--tp", str(args.tp),
        "--ep", str(args.ep),
        "--bs", str(bs),
        "--backend", args.backend,
        "--iters", str(args.iters),
        "--json-row",
    ]
    if args.no_l2_flush:
        cmd.append("--no-l2-flush")
    cp = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
        env=dict(__import__("os").environ),
    )
    merged = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    if cp.returncode != 0:
        raise RuntimeError(f"bench_e2e failed at bs={bs}\n{merged}")
    return _parse_last_json_line(merged)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = set()
    for r in rows:
        keys.update(r.keys())
    base = [k for k in ["bs", "expanded", "active", "E", "H", "I", "K"] if k in keys]
    tail = sorted(k for k in keys if k not in base)
    cols = base + tail
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda x: int(x.get("bs", 0))):
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Run E2E in isolated subprocesses")
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--moe-I", type=int, required=True, dest="moe_I")
    p.add_argument("--E", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--ep", type=int, required=True)
    p.add_argument("--bs", type=str, required=True)
    p.add_argument("--backend", default="all")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--no-l2-flush", action="store_true")
    args = p.parse_args()

    ensure_persistent_benchmark_env()
    bs_list = [int(x) for x in args.bs.split(",") if x.strip()]
    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for bs in bs_list:
        try:
            row = _run_one_bs(args, bs)
            rows.append(row)
            _write_csv(out_path, rows)  # incremental checkpoint
            print(f"[OK] bs={bs}")
        except Exception as e:
            print(f"[ERR] bs={bs} {e}")
            _write_csv(out_path, rows)

    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()

