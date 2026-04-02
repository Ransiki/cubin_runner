"""E2E MoE benchmark entrypoint."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "DeepGEMM"))

from benchmark_env import ensure_persistent_benchmark_env
from bench_utils import bench_kineto, build_fused_runtime_state, build_nofuse_runtime_state, make_balanced_logits, make_gemm_config


class MxFP4E2E:
    name = "mxfp4"

    def setup(self, cfg, device):
        from mxfp4_moe_cubin import mxfp4_moe_cubin
        self.cfg = cfg
        self.state = build_fused_runtime_state(cfg, device=device)
        self.fn = mxfp4_moe_cubin

    def run_pipeline(self):
        c, s = self.cfg, self.state
        self.fn(
            s["logits"], s["hidden"], s["w1"], s["s1"], s["w2"], s["s2"],
            c.E, c.local_K, c.I,
            output1_scale_scalar=s["sc"],
            output1_scale_gate_scalar=s["sc"],
            output2_scale_scalar=s["sc"],
            do_finalize=True, routing_method_type=1,
        )

    def cubin_meta(self):
        s = self.state
        return {
            "tile_n": s.get("tile_n"),
            "fc1_cfg": s.get("fc1_cfg"),
            "fc2_cfg": s.get("fc2_cfg"),
        }


class MxFP4NofuseE2E:
    name = "mxfp4_nf"

    def setup(self, cfg, device):
        from mxfp4_moe_cubin import _load_lib, _load_nofuse_lib
        self.cfg = cfg
        self.state = build_nofuse_runtime_state(cfg, device=device)
        self.fused_lib = _load_lib()
        self.nofuse_lib = _load_nofuse_lib()
        self.fused_lib.moe_cubin_set_verbose(0)
        self.nofuse_lib.moe_cubin_set_verbose(0)
        self.stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)

    def _swiglu(self):
        s = self.state
        s["fc2_in"].copy_(s["fc1_out"][..., 0::2] * torch.nn.functional.silu(s["fc1_out"][..., 1::2]))

    def run_pipeline(self):
        self.run_fc1()
        self.run_fc2()

    def run_fc1(self):
        from mxfp4_moe_cubin import _ptr
        c, s = self.cfg, self.state
        self.nofuse_lib.moe_cubin_run(
            s["fc1_cfg"], True,
            _ptr(s["w1"]), _ptr(s["s1"]), _ptr(s["hidden"]), _ptr(s["fc1_out"]),
            _ptr(s["sc"]), _ptr(s["sc"]), None, None, None, None,
            c.H, c.I, c.E, s["expanded"],
            _ptr(s["p2t"]), _ptr(s["cb"]), _ptr(s["cm"]),
            _ptr(s["ne"]), _ptr(s["tp_gpu"]),
            s["bn_host"], c.E, self.stream,
        )

    def run_fc2(self):
        from mxfp4_moe_cubin import _ptr
        c, s = self.cfg, self.state
        self._swiglu()
        self.fused_lib.moe_cubin_run(
            s["fc2_cfg"], False,
            _ptr(s["w2"]), _ptr(s["s2"]), _ptr(s["fc2_in"]), _ptr(s["fc2_out"]),
            _ptr(s["sc"]), _ptr(s["sc"]), None, None, None, None,
            c.H, c.I, c.E, s["expanded"],
            None, _ptr(s["cb"]), _ptr(s["cm"]),
            _ptr(s["ne"]), _ptr(s["tp_gpu"]),
            s["bn_host"], c.E, self.stream,
        )

    def cubin_meta(self):
        s = self.state
        return {
            "tile_n": s.get("tn"),
            "fc1_cfg": s.get("fc1_cfg"),
            "fc2_cfg": s.get("fc2_cfg"),
        }


def _mxint4_quantize(x, sf_vec_size=32):
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].float()
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].float()
    amax = torch.where(x_max * 8.0 / 7.0 > -x_min, x_max * 8.0 / 7.0, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    packed = x_int4.reshape(*x.shape[:-1], x.shape[-1] // 2).view(torch.uint8)
    sf = scales.reshape(*x.shape[:-1], x.shape[-1] // sf_vec_size).to(torch.bfloat16)
    return packed, sf


class MxInt4E2E:
    name = "mxint4"

    def setup(self, cfg, device):
        from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe
        E, H, I, T, local_K = cfg.E, cfg.H, cfg.I, cfg.local_T, cfg.local_K
        self.logits = make_balanced_logits(T, E, local_K, device)
        self.hidden = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.01
        w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.01
        w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device) * 0.01
        self.w1, self.s1 = _mxint4_quantize(w1, 32)
        self.w2, self.s2 = _mxint4_quantize(w2, 32)
        self.fn = trtllm_mxint4_block_scale_moe
        self.kw = dict(num_experts=E, top_k=local_K, n_group=1, topk_group=1, intermediate_size=I, local_expert_offset=0, local_num_experts=E, routed_scaling_factor=None, routing_method_type=1, tune_max_num_tokens=max(T, 128))
        self.run_pipeline()
        torch.cuda.synchronize()

    def run_pipeline(self):
        self.fn(self.logits, None, self.hidden, self.w1, self.s1, None, None, None, self.w2, self.s2, **self.kw)


class DeepGEMME2E:
    def __init__(self, layout="masked"):
        self.layout = layout
        self.name = f"dg_{layout}"

    def setup(self, cfg, device):
        import deep_gemm
        from deep_gemm.utils import ceil_div, get_mk_alignment_for_contiguous_layout, per_token_cast_to_fp8, per_token_cast_to_fp4
        self.dg = deep_gemm
        self.ceil_div = ceil_div
        self.fp8 = per_token_cast_to_fp8
        self.fp4 = per_token_cast_to_fp4
        self.mk_align = get_mk_alignment_for_contiguous_layout()
        E, H, I = cfg.E, cfg.H, cfg.I
        GA, GB = 128, 32
        self.GA, self.I, self.E = GA, I, E
        self.ra, self.rb = (1, GA), (1, GB)
        self.masked_m = torch.tensor(list(cfg.tokens_per_expert), device=device, dtype=torch.int32)
        self.expected_m = max(1, cfg.expanded // max(1, cfg.active))
        b1 = torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.1
        b2 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * 0.1
        self.b1 = self._quant_b(b1, GB)
        self.b2 = self._quant_b(b2, GB)
        self.use_psum = self.layout == "psum"

        if self.use_psum:
            from deep_gemm.utils import align
            psum, off = [], 0
            for m in self.masked_m.tolist():
                off += m
                psum.append(off)
                off = align(off, self.mk_align)
            total_m = off
            self.psum_m = torch.tensor(psum, device=device, dtype=torch.int32)
            a1 = torch.randn(total_m, H, device=device, dtype=torch.bfloat16) * 0.1
            self.a1 = self.fp8(a1, use_ue8m0=True, gran_k=GA)
            self.d1 = torch.zeros(total_m, 2 * I, device=device, dtype=torch.bfloat16)
            self.a2 = (torch.empty(total_m, I, device=device, dtype=torch.float8_e4m3fn), torch.empty(total_m, self.ceil_div(I, GA), device=device, dtype=torch.float))
            self.d2 = torch.empty(total_m, H, device=device, dtype=torch.bfloat16)
        else:
            max_m = max(max(self.masked_m.tolist()), 1)
            a1 = torch.randn(E, max_m, H, device=device, dtype=torch.bfloat16) * 0.1
            self.a1 = self._quant_a(a1, GA)
            self.d1 = torch.zeros(E, max_m, 2 * I, device=device, dtype=torch.bfloat16)
            self.a2 = (torch.empty(E, max_m, I, device=device, dtype=torch.float8_e4m3fn), torch.empty(E, max_m, self.ceil_div(I, GA), device=device, dtype=torch.float))
            self.d2 = torch.empty(E, max_m, H, device=device, dtype=torch.bfloat16)

        self.run_pipeline()
        torch.cuda.synchronize()

    def _quant_a(self, a, gran_k):
        ng, m, k = a.shape
        d = torch.empty(ng, m, k, device=a.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(ng, m, self.ceil_div(k, gran_k), device=a.device, dtype=torch.float)
        for i in range(ng):
            d[i], s[i] = self.fp8(a[i], use_ue8m0=True, gran_k=gran_k)
        return d, s

    def _quant_b(self, b, gran_k):
        ng, n, k = b.shape
        d = torch.empty(ng, n, k // 2, device=b.device, dtype=torch.uint8)
        s = torch.empty(ng, n, self.ceil_div(k, gran_k), device=b.device, dtype=torch.float)
        for i in range(ng):
            d[i], s[i] = self.fp4(b[i], use_ue8m0=True, gran_k=gran_k)
        return d, s

    def _swiglu_and_quant(self):
        x = torch.nan_to_num(
            self.d1[..., :self.I] * torch.nn.functional.silu(self.d1[..., self.I:]),
            nan=0.1,
            posinf=1.0,
            neginf=-1.0,
        )
        if self.use_psum:
            if torch.count_nonzero(x) == 0:
                x = x + 1e-6
            self.a2 = self.fp8(x, use_ue8m0=True, gran_k=self.GA)
        else:
            for i in range(self.E):
                m = int(self.masked_m[i].item())
                if m > 0:
                    xi = torch.nan_to_num(x[i, :m], nan=0.1, posinf=1.0, neginf=-1.0)
                    if torch.count_nonzero(xi) == 0:
                        xi = xi + 1e-6
                    self.a2[0][i, :m], self.a2[1][i, :m] = self.fp8(xi, use_ue8m0=True, gran_k=self.GA)

    def run_pipeline(self):
        if self.use_psum:
            self.dg.m_grouped_fp8_fp4_gemm_nt_contiguous(self.a1, self.b1, self.d1, self.psum_m, disable_ue8m0_cast=False, use_psum_layout=True, expected_m_for_psum_layout=self.expected_m, recipe=None, recipe_a=self.ra, recipe_b=self.rb)
            self._swiglu_and_quant()
            self.dg.m_grouped_fp8_fp4_gemm_nt_contiguous(self.a2, self.b2, self.d2, self.psum_m, disable_ue8m0_cast=False, use_psum_layout=True, expected_m_for_psum_layout=self.expected_m, recipe=None, recipe_a=self.ra, recipe_b=self.rb)
        else:
            self.dg.m_grouped_fp8_fp4_gemm_nt_masked(self.a1, self.b1, self.d1, self.masked_m, self.expected_m, disable_ue8m0_cast=False, recipe=None, recipe_a=self.ra, recipe_b=self.rb)
            self._swiglu_and_quant()
            self.dg.m_grouped_fp8_fp4_gemm_nt_masked(self.a2, self.b2, self.d2, self.masked_m, self.expected_m, disable_ue8m0_cast=False, recipe=None, recipe_a=self.ra, recipe_b=self.rb)


def measure_backend(be, cfg, num_tests, flush_l2):
    fn = be.run_pipeline
    if isinstance(be, DeepGEMME2E):
        fc1_t = bench_kineto(fn, f"{2 * cfg.I}u, {cfg.H}u", num_tests=num_tests, flush_l2=flush_l2)
        fc2_t = bench_kineto(fn, f"{cfg.H}u, {cfg.I}u", num_tests=num_tests, flush_l2=flush_l2)
    elif isinstance(be, MxFP4E2E):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        bmm = [e.key for e in prof.key_averages() if e.device_time_total > 0 and "bmm_" in e.key]
        if len(bmm) < 2:
            raise RuntimeError(f"expected >=2 bmm kernels, got {len(bmm)}")
            
        fc1 = [k for k in bmm if "_tmaSf_" in k or "_ldgstsSf_" in k]
        fc1_name = fc1[0] if fc1 else bmm[0]
        fc2_name = [k for k in bmm if k != fc1_name][0]
        fc1_t = bench_kineto(fn, fc1_name, num_tests=num_tests, flush_l2=flush_l2)
        fc2_t = bench_kineto(fn, fc2_name, num_tests=num_tests, flush_l2=flush_l2)
    elif isinstance(be, MxFP4NofuseE2E):
        # Profile each phase directly to avoid dependency on distinct bmm names.
        fc1_t = bench_kineto(be.run_fc1, "bmm_", num_tests=num_tests, flush_l2=flush_l2)
        fc2_t = bench_kineto(be.run_fc2, "bmm_", num_tests=num_tests, flush_l2=flush_l2)
    else:
        fc1_t = bench_kineto(fn, "swiglu", num_tests=num_tests, flush_l2=flush_l2)
        fc2_t = bench_kineto(fn, "bmm_", num_tests=num_tests, flush_l2=flush_l2, exclude="swiglu")
    fc1_us, fc2_us = fc1_t * 1e6, fc2_t * 1e6
    return {"fc1_us": round(fc1_us, 1), "fc2_us": round(fc2_us, 1), "total_us": round(fc1_us + fc2_us, 1)}


def _worker_entry(payload):
    c = payload["cfg"]
    cfg = make_gemm_config(c["H"], c["moe_I"], c["E"], c["K"], c["tp"], c["ep"], c["bs"])
    be = MxFP4E2E() if payload["op"] == "measure_fused" else MxFP4NofuseE2E()
    be.setup(cfg, "cuda")
    return measure_backend(be, cfg, payload.get("num_tests", 20), payload.get("flush_l2", True))


def main():
    ensure_persistent_benchmark_env()
    p = argparse.ArgumentParser(description="E2E benchmark")
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--moe-I", type=int, required=True, dest="moe_I")
    p.add_argument("--E", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--ep", type=int, required=True)
    p.add_argument("--bs", type=str, required=True)
    p.add_argument("--backend", default="all")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-l2-flush", action="store_true")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--json-row", action="store_true")
    args = p.parse_args()

    bs_list = [int(x) for x in args.bs.split(",") if x.strip()]
    flush = not args.no_l2_flush
    be_set = set(args.backend.split(",")) if args.backend != "all" else {"mxfp4", "mxfp4_nf", "mxint4", "deepgemm"}
    backends = []
    if "mxfp4" in be_set: backends.append(MxFP4E2E())
    if "mxfp4_nf" in be_set: backends.append(MxFP4NofuseE2E())
    if "mxint4" in be_set: backends.append(MxInt4E2E())
    if "deepgemm" in be_set: backends.extend([DeepGEMME2E("masked"), DeepGEMME2E("psum")])

    rows = []
    for bs in bs_list:
        cfg = make_gemm_config(args.H, args.moe_I, args.E, args.K, args.tp, args.ep, bs)
        row = {"bs": bs, "expanded": cfg.expanded, "active": cfg.active, "E": cfg.E, "H": cfg.H, "I": cfg.I, "K": cfg.K}
        for be in backends:
            torch.cuda.empty_cache()
            be.setup(cfg, "cuda")
            m = measure_backend(be, cfg, args.iters, flush)
            row[f"{be.name}_fc1_us"] = m["fc1_us"]
            row[f"{be.name}_fc2_us"] = m["fc2_us"]
            row[f"{be.name}_total_us"] = m["total_us"]
            # Record selected cubin metadata for non-DeepGEMM backends.
            if not isinstance(be, DeepGEMME2E) and hasattr(be, "cubin_meta"):
                meta = be.cubin_meta()
                for k, v in meta.items():
                    row[f"{be.name}_{k}"] = v
            print(f"BS={bs:5d} {be.name:10s} FC1={m['fc1_us']:8.1f}us FC2={m['fc2_us']:8.1f}us total={m['total_us']:8.1f}us")
        rows.append(row)
        if args.json_row and len(bs_list) == 1:
            print(json.dumps(row))

    if args.csv:
        keys = set()
        for r in rows: keys.update(r.keys())
        base = [k for k in ["bs", "expanded", "active", "E", "H", "I", "K"] if k in keys]
        cols = base + sorted(k for k in keys if k not in base)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
