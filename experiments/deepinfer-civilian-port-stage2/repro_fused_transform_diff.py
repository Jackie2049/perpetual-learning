#!/usr/bin/env python3
"""Reproduce one failing fused_transform config and locate actual-vs-reference mismatches."""
import importlib.util
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


def _load_variant(mod_name, fn_name):
    spec = importlib.util.spec_from_file_location(
        f"{PKG}.{mod_name}_nosync", os.path.join(CUDA_DIR, f"{mod_name}_nosync.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{PKG}.{mod_name}_nosync"] = mod
    spec.loader.exec_module(mod)
    setattr(ops, fn_name, getattr(mod, fn_name))


# fused_transform + its inner norm dependency
_load_variant("fused_transform", "fused_norm_rope_hadamard_cast_store_kvcache_tl")
_load_variant("norm", "norm_tl")

import torch  # noqa: E402
from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (  # noqa: E402
    fused_norm_rope_hadamard_cast_store_kvcache,
    fused_norm_rope_hadamard_cast_store_kvcache_ref,
    _make_case,
    _clone_out_args,
    _resolve_cache_layout,
    _guarded_store_cache,
)
from deepinfer.model_executor.layers.tilelang_ops.fused_transform import KVCacheOutArgs  # noqa: E402
from dataclasses import replace  # noqa: E402


def _assert_diff(actual, expected, label=""):
    if isinstance(actual, tuple):
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_diff(a, e, f"{label}[{i}]")
        return
    if actual is None:
        return
    if not torch.equal(actual, expected):
        diff = (actual != expected)
        # account for exact bitwise view comparisons
        a_b = actual.view(torch.uint8) if actual.dtype == torch.float8_e4m3fn else actual
        e_b = expected.view(torch.uint8) if expected.dtype == torch.float8_e4m3fn else expected
        neq = (a_b != e_b)
        n_mismatch = neq.sum().item()
        flat_idx = neq.view(-1).nonzero().view(-1)
        print(f"{label}: mismatch={n_mismatch}/{neq.numel()}", flush=True)
        for idx in flat_idx[:5]:
            unraveled = torch.unravel_index(idx, neq.shape)
            vals = tuple(int(v) for v in unraveled)
            av = a_b[vals].item() if a_b.dtype.is_floating_point else int(a_b[vals].item())
            ev = e_b[vals].item() if e_b.dtype.is_floating_point else int(e_b[vals].item())
            print(f"   at {vals}: actual={av} expected={ev}", flush=True)
        if actual.dtype.is_floating_point and expected.dtype.is_floating_point:
            max_abs = (actual - expected).abs().max().item()
            print(f"   max_abs_diff={max_abs} actual_dtype={actual.dtype}", flush=True)
    else:
        print(f"{label}: equal", flush=True)


# reproduce: norm-bf16-out-h128-d512-sn73728-n2333 (first failing config)
# _CORRECTNESS_CASES contains (case, n); find the one with that id
import pytest  # noqa: E402
from deepinfer.model_executor.layers.tilelang_ops.fused_transform import _CORRECTNESS_CASES  # noqa: E402

target_id = "norm-bf16-out-h128-d512-sn73728-n2333"
for case, n in _CORRECTNESS_CASES:
    if f"{case.operation}-n{n}" == target_id:
        torch.manual_seed(0)
        x, norm_args, rope_args, hadamard_args, cast_args, out_args = _make_case(case, n=n, preallocate=False)
        expected_out_args = _clone_out_args(out_args)
        actual = fused_norm_rope_hadamard_cast_store_kvcache(x, norm_args, rope_args, hadamard_args, cast_args, out_args)
        expected = fused_norm_rope_hadamard_cast_store_kvcache_ref(x, norm_args, rope_args, hadamard_args, cast_args, expected_out_args)
        torch.cuda.synchronize()
        if isinstance(actual, tuple):
            actual = tuple(v.contiguous().view(torch.uint8) if v.dtype == torch.float8_e4m3fn else v for v in actual)
            expected = tuple(v.contiguous().view(torch.uint8) if v.dtype == torch.float8_e4m3fn else v for v in expected)
        elif isinstance(actual, torch.Tensor) and actual.dtype == torch.float8_e4m3fn:
            actual = actual.contiguous().view(torch.uint8)
            expected = expected.contiguous().view(torch.uint8)
        print(f"target: {target_id}", flush=True)
        _assert_diff(actual, expected, "result")
        break
