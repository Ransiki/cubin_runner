"""
Correctness test v2: TRT-LLM quantization + our cubin execution.

Split into two phases to avoid symbol conflicts:
  Phase 1: Quantize + prepare weights with TRT-LLM ops (save to disk)
  Phase 2: Load weights + run cubin pipeline (separate from TRT-LLM)
"""
import sys, os, subprocess, tempfile, json
sys.path.insert(0, os.path.dirname(__file__))

PHASE1_SCRIPT = os.path.join(os.path.dirname(__file__), "_phase1_mxfp4_quantize.py")
PHASE2_SCRIPT = os.path.join(os.path.dirname(__file__), "_phase2_mxfp4_cubin_run.py")


def run_test(T, H, I, E, K, seed=42):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"T": T, "H": H, "I": I, "E": E, "K": K, "seed": seed, "dir": tmpdir}
        cfg_path = os.path.join(tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"\n--- T={T} H={H} I={I} E={E} K={K} ---")

        # Phase 1: quantize with TRT-LLM
        print("  Phase 1: Quantizing with TRT-LLM...")
        r1 = subprocess.run(
            [sys.executable, PHASE1_SCRIPT, cfg_path],
            capture_output=True, text=True, timeout=300)
        if r1.returncode != 0:
            print(f"  Phase 1 FAILED:\n{r1.stderr[-500:]}")
            return False
        for line in r1.stdout.strip().split("\n"):
            print(f"    {line}")

        # Phase 2: run cubin pipeline
        print("  Phase 2: Running cubin pipeline...")
        r2 = subprocess.run(
            [sys.executable, PHASE2_SCRIPT, cfg_path],
            capture_output=True, text=True, timeout=600)
        if r2.returncode != 0:
            print(f"  Phase 2 FAILED:\n{r2.stderr[-500:]}")
            return False
        for line in r2.stdout.strip().split("\n"):
            print(f"    {line}")

        result_path = os.path.join(tmpdir, "result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            return result.get("passed", False)
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("  MxFP4 MOE Cubin Correctness Test")
    print("  Phase 1: TRT-LLM quantization (separate process)")
    print("  Phase 2: Cubin execution (separate process)")
    print("=" * 70)

    configs = [
        # (T, H, I, E, K)
        # --- Basic shapes ---
        (8,   256,  256,  4,  2),
        (16,  512,  512,  4,  2),
        (8,  1024, 1024,  4,  2),
        (8,  1024, 1024,  4,  1),    # K=1: single expert per token
        # --- DSv3 TP=8 shapes (customer config) ---
        (1,  7168,  256, 256, 8),    # BS=1 decode
        (16, 7168,  256, 256, 8),    # BS=16 decode
        (128,7168,  256, 256, 8),    # BS=128 decode
        # --- Kimi K2 TP=8 shape (smaller to fit timeout) ---
        (16, 7168,  256, 384, 8),    # K2 TP=8
        # --- Edge cases ---
        (4,   256,  256,  8,  2),    # very few tokens
        (32,  512,  256, 128, 4),    # many experts, moderate K
    ]

    results = []
    for cfg in configs:
        passed = run_test(*cfg)
        results.append((cfg, passed))

    print("\n" + "=" * 70)
    all_pass = all(p for _, p in results)
    for cfg, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] T={cfg[0]} H={cfg[1]} I={cfg[2]} E={cfg[3]} K={cfg[4]}")
    print("=" * 70)
    print("  ALL PASSED" if all_pass else "  SOME FAILED")
    sys.exit(0 if all_pass else 1)
