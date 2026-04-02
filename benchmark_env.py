"""Persistent benchmark cache setup for FlashInfer and torch extensions."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TRTLLMGEN_BENCH_CACHE_BASE = REPO_ROOT / ".benchmark_cache"
FLASHINFER_WORKSPACE_BASE = TRTLLMGEN_BENCH_CACHE_BASE / "flashinfer"
FLASHINFER_CUBIN_DIR = FLASHINFER_WORKSPACE_BASE / "cubins"
TORCH_EXTENSIONS_DIR = TRTLLMGEN_BENCH_CACHE_BASE / "torch_extensions"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copytree_if_missing(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    shutil.copytree(src, dst)


def _remove_stale_ninja(path: Path) -> None:
    if not path.exists():
        return
    for root, _, files in os.walk(path):
        if "build.ninja" in files:
            file_path = Path(root) / "build.ninja"
            try:
                txt = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Only remove files that still hardcode default user cache paths.
            # Do NOT remove valid ninjas in our persistent benchmark cache,
            # whose paths may also contain ".cache" segments.
            home_cache = str(Path.home() / ".cache")
            has_default_cache_ref = (home_cache in txt) or ("/root/.cache" in txt)
            has_benchmark_cache_ref = str(TRTLLMGEN_BENCH_CACHE_BASE) in txt
            if has_default_cache_ref and not has_benchmark_cache_ref:
                try:
                    file_path.unlink()
                except OSError:
                    pass


def ensure_persistent_benchmark_env() -> None:
    """Create and export a repository-local persistent benchmark cache."""
    _safe_mkdir(TRTLLMGEN_BENCH_CACHE_BASE)
    _safe_mkdir(FLASHINFER_WORKSPACE_BASE)
    _safe_mkdir(FLASHINFER_CUBIN_DIR)
    _safe_mkdir(TORCH_EXTENSIONS_DIR)

    # Best-effort one-time migration from default cache.
    home = Path.home()
    old_fi_cubins = home / ".cache" / "flashinfer" / "cubins"
    old_torch_ext = home / ".cache" / "torch_extensions"
    _copytree_if_missing(old_fi_cubins, FLASHINFER_CUBIN_DIR)
    _copytree_if_missing(old_torch_ext, TORCH_EXTENSIONS_DIR)

    # Drop stale ninja files that reference old absolute cache paths.
    _remove_stale_ninja(FLASHINFER_WORKSPACE_BASE)
    _remove_stale_ninja(TORCH_EXTENSIONS_DIR)

    os.environ["TRTLLMGEN_BENCH_CACHE_BASE"] = str(TRTLLMGEN_BENCH_CACHE_BASE)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(FLASHINFER_WORKSPACE_BASE)
    os.environ["FLASHINFER_CUBIN_DIR"] = str(FLASHINFER_CUBIN_DIR)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(TORCH_EXTENSIONS_DIR)
    # Keep legacy ~/.cache aligned so lower-level libraries that do not honor
    # the custom variables still land in a persistent location.
    os.environ.setdefault("XDG_CACHE_HOME", str(TRTLLMGEN_BENCH_CACHE_BASE))

