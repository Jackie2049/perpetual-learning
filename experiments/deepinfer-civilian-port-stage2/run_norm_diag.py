#!/usr/bin/env python3
"""Diagnostic: run norm.py upstream tests with nosync patch, dump linecache state if
inspect.getsourcelines fails (the observed intermittent OSError in the full 六续 run)."""
import importlib.util
import inspect
import linecache
import os
import sys

REPO = "/home/zxw/deepinfer-civilian"
sys.path.insert(0, REPO)
CUDA_DIR = os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops/cuda")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"


def _hide():
    hidden = []
    for f in sorted(os.listdir(CUDA_DIR)):
        if f.endswith("_pdlfree.py") or f.endswith("_nosync.py"):
            src = os.path.join(CUDA_DIR, f)
            dst = src + ".hidden"
            os.rename(src, dst)
            hidden.append((src, dst))
    return hidden


def _restore(h):
    for src, dst in h:
        os.rename(dst, src)


h = _hide()
try:
    from deepinfer.model_executor.layers.tilelang_ops import ops
finally:
    _restore(h)

var = importlib.util.spec_from_file_location(
    f"{PKG}.norm_nosync", os.path.join(CUDA_DIR, "norm_nosync.py")
)
mod = importlib.util.module_from_spec(var)
sys.modules[f"{PKG}.norm_nosync"] = mod
var.loader.exec_module(mod)
ops.norm_tl = mod.norm_tl

fp = os.path.join(CUDA_DIR, "norm_nosync.py")
print("file exists:", os.path.exists(fp), flush=True)
print("linecache cached:", fp in linecache.cache, flush=True)

orig_gl = inspect.getsourcelines


def diag_gl(obj):
    try:
        return orig_gl(obj)
    except OSError as e:
        code = getattr(obj, "__code__", None)
        cf = code.co_filename if code else None
        print("=== GETSOURCELINES FAILED ===", flush=True)
        print("obj:", obj, flush=True)
        print("co_filename:", cf, "exists:", os.path.exists(cf) if cf else None, flush=True)
        if cf:
            try:
                st = os.stat(cf)
                print("stat size/mtime:", st.st_size, st.st_mtime, flush=True)
                print("mode:", oct(st.st_mode), flush=True)
            except OSError as se:
                print("stat failed:", se, flush=True)
            entry = linecache.cache.get(cf)
            print(
                "linecache entry:", type(entry).__name__,
                ("len=%d" % len(entry)) if isinstance(entry, (list, tuple)) else entry,
                flush=True,
            )
            print(
                "cached keys:", [k for k in linecache.cache if "norm_nosync" in k],
                flush=True,
            )
        raise


inspect.getsourcelines = diag_gl

import pytest  # noqa: E402

tl = os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops")
sys.exit(pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", os.path.join(tl, "norm.py")]))
