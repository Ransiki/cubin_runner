"""Unified plot tool, aligned with legacy plot style."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")


BACKENDS = ["mxfp4", "mxfp4_nf", "mxint4", "dg_masked", "dg_psum"]
BACKEND_LABELS = {
    "mxfp4": "MxFP4 fused",
    "mxfp4_nf": "MxFP4 nofuse",
    "mxint4": "MxInt4",
    "dg_masked": "DG masked",
    "dg_psum": "DG psum",
}
COLORS = {
    "mxfp4": "#1f77b4",
    "mxfp4_nf": "#9467bd",
    "mxint4": "#2ca02c",
    "dg_masked": "#d62728",
    "dg_psum": "#ff7f0e",
}
PEAK_BW = 8.0


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                if k is None:
                    continue
                clean[k.strip()] = (v.strip() if isinstance(v, str) else v)
            rows.append(clean)
    return rows


def calc_dram_bytes(backend: str, fc: str, row: dict) -> float:
    active = int(float(row.get("active", 0)))
    expanded = int(float(row.get("expanded", 0)))
    h = int(float(row.get("H", 0)))
    i = int(float(row.get("I", 0)))
    a, x = active, expanded

    if backend == "mxfp4":
        sf_block = 32
        if fc == "fc1":
            return a * 2 * i * (h // 2 + h // sf_block) + x * h * 2 + x * i * 2
        return a * h * (i // 2 + i // sf_block) + x * i * 2 + x * h * 2
    if backend == "mxfp4_nf":
        sf_block = 32
        if fc == "fc1":
            return a * 2 * i * (h // 2 + h // sf_block) + x * h * 2 + x * 2 * i * 2
        return a * h * (i // 2 + i // sf_block) + x * i * 2 + x * h * 2
    if backend == "mxint4":
        if fc == "fc1":
            return a * 2 * i * (h // 2 + h // 32 * 2) + x * h * 2 + x * i * 2
        return a * h * (i // 2 + i // 32 * 2) + x * i * 2 + x * h * 2
    if backend in ("dg_masked", "dg_psum"):
        if fc == "fc1":
            w = a * 2 * i * (h // 2 + h // 32 * 4)
            act = x * h + x * (h // 128) * 4
            out = x * 2 * i * 2
            return w + act + out
        w = a * h * (i // 2 + i // 32 * 4)
        act = x * i + x * (i // 128) * 4
        out = x * h * 2
        return w + act + out
    return 0.0


def _extract_cfg_from_name(path: Path) -> tuple[int, int]:
    m = re.search(r"tp(\d+)_ep(\d+)_e2e\.csv$", path.name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _sorted_cfg_csvs(folder: Path) -> list[tuple[int, int, Path]]:
    items = []
    for p in folder.glob("*_e2e.csv"):
        tp, ep = _extract_cfg_from_name(p)
        if tp > 0 and ep > 0:
            items.append((tp, ep, p))
    # Keep historical order when present.
    preferred = {(8, 1): 0, (1, 8): 1}
    items.sort(key=lambda x: preferred.get((x[0], x[1]), 99))
    return items


def _series_latency(rows: list[dict], backend: str, fc: str) -> tuple[np.ndarray, np.ndarray]:
    col = f"{backend}_{fc}_us"
    xs, ys = [], []
    for r in rows:
        v = r.get(col, "")
        if not v:
            continue
        t_us = float(v)
        if t_us <= 0:
            continue
        xs.append(int(r["bs"]))
        ys.append(t_us)
    return np.array(xs), np.array(ys)


def _series_bw(rows: list[dict], backend: str, fc: str) -> tuple[np.ndarray, np.ndarray]:
    col = f"{backend}_{fc}_us"
    xs, ys = [], []
    for r in rows:
        v = r.get(col, "")
        if not v:
            continue
        t_us = float(v)
        if t_us <= 0:
            continue
        bw = calc_dram_bytes(backend, fc, r) / (t_us * 1e-6) / 1e12
        xs.append(int(r["bs"]))
        ys.append(bw)
    return np.array(xs), np.array(ys)


def _plot_mode(folder: Path, mode: str) -> Path:
    """Plot FC1/FC2 together in one figure."""
    cfgs = _sorted_cfg_csvs(folder)
    fcs = ("fc1", "fc2")
    nrows = len(fcs)
    ncols = max(1, len(cfgs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)

    fig.subplots_adjust(top=0.90, wspace=0.22, hspace=0.28, left=0.06, right=0.97, bottom=0.08)
    if mode == "bandwidth":
        fig.suptitle("E2E: FC1/FC2 Bandwidth  (L2-cold, kineto)", fontsize=15, fontweight="bold")
    else:
        fig.suptitle("E2E: FC1/FC2 Latency  (L2-cold, kineto)", fontsize=15, fontweight="bold")

    for row_idx, fc in enumerate(fcs):
        for col_idx, (tp, ep, csv_path) in enumerate(cfgs):
            ax = axes[row_idx][col_idx]
            rows = read_csv(csv_path)
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: int(r["bs"]))
            e_local = int(rows[0].get("E", "0"))
            ax.set_title(f"{fc.upper()} | TP{tp} EP{ep}  (E_local={e_local})", fontsize=11.5, pad=6)

            for be in BACKENDS:
                if mode == "bandwidth":
                    x, y = _series_bw(rows, be, fc)
                else:
                    x, y = _series_latency(rows, be, fc)
                if len(x) == 0:
                    continue
                ax.plot(
                    x,
                    y,
                    "-D",
                    color=COLORS[be],
                    markersize=5,
                    linewidth=2.0,
                    label=f"{BACKEND_LABELS[be]} e2e",
                    zorder=5,
                )

            if mode == "bandwidth":
                ax.axhline(y=PEAK_BW, color="#888888", linestyle=":", linewidth=1, alpha=0.5)
                ax.text(1.1, PEAK_BW * 1.02, f"Peak {PEAK_BW} TB/s", fontsize=7.5, color="#888888", va="bottom")
                ax.set_ylabel(f"{fc.upper()} Bandwidth (TB/s)", fontsize=10)
                ax.set_ylim(bottom=0, top=PEAK_BW + 1.0)
            else:
                ax.set_ylabel(f"{fc.upper()} Latency (us)", fontsize=10)
                ax.set_ylim(bottom=0)

            ax.set_xlabel("Batch Size", fontsize=10)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=9)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ncol = 3 if len(handles) > 4 else 2
                ax.legend(fontsize=7, ncol=ncol, loc="upper left", framealpha=0.85, edgecolor="none")

    out = folder / f"e2e_{mode}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot benchmark CSVs in folder")
    p.add_argument("--input-dir", type=str, required=True)
    args = p.parse_args()
    folder = Path(args.input_dir)
    outs = []
    outs.append(_plot_mode(folder, "latency"))
    outs.append(_plot_mode(folder, "bandwidth"))
    for o in outs:
        print(f"[plot] {o.name}")


if __name__ == "__main__":
    main()

