"""fused_transform 数值差异复验：向量化统计 + 差异位置采样，两遍对照确定性"""
import importlib.util, os, sys
import torch
torch.cuda.set_device(0)
REPO = "/home/zxw/deepinfer-civilian"
CUDA_DIR = os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops/cuda")
PKG = "deepinfer.model_executor.layers.tilelang_ops.cuda"

def hide_variants():
    hidden = []
    for f in sorted(os.listdir(CUDA_DIR)):
        if f.endswith("_pdlfree.py") or f.endswith("_nosync.py"):
            src = os.path.join(CUDA_DIR, f); dst = src + ".hidden"
            os.rename(src, dst); hidden.append((src, dst))
    return hidden
def restore(hidden):
    for src, dst in hidden: os.rename(dst, src)

hidden = hide_variants()
try:
    from deepinfer.model_executor.layers.tilelang_ops import ops
finally:
    restore(hidden)

def load_variant(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# patch all 43 like the six-run
import run_nosync_upstream_tests as rt
rt.VARIANT_SUFFIX = os.environ.get("FT_VARIANT", "nosync")
for mod_name, fn_name in rt.CASES:
    var_mod = load_variant(f"{PKG}.{mod_name}_{rt.VARIANT_SUFFIX}",
                           os.path.join(CUDA_DIR, f"{mod_name}_{rt.VARIANT_SUFFIX}.py"))
    setattr(ops, fn_name, getattr(var_mod, fn_name))

import sys as _sys
spec = importlib.util.spec_from_file_location(
    "ft_wrapper2", os.path.join(REPO, "deepinfer/model_executor/layers/tilelang_ops/fused_transform.py"))
ft = importlib.util.module_from_spec(spec)
_sys.modules["ft_wrapper2"] = ft
spec.loader.exec_module(ft)

fn = ft.fused_norm_rope_hadamard_cast_store_kvcache
ref = ft.fused_norm_rope_hadamard_cast_store_kvcache_ref

case = n = None
for c, nn in ft._CORRECTNESS_CASES:
    if (getattr(c, "operation", None) == "norm-rope-bf16-out-h128-d512-r64-sn73728"
            and nn == int(os.environ.get("FT_N", "32769"))):
        case, n = c, nn
        break

def run_once(seed):
    torch.manual_seed(seed)
    x, norm_args, rope_args, hadamard_args, cast_args, out_args = ft._make_case(case, n=n, preallocate=False)
    from dataclasses import replace
    expected_out_args = ft._clone_out_args(out_args)
    actual = fn(x, norm_args, rope_args, hadamard_args, cast_args, out_args)
    expected = ref(x, norm_args, rope_args, hadamard_args, cast_args, expected_out_args)
    if isinstance(actual, tuple):
        a, e = actual[0], expected[0]
    else:
        a, e = actual, expected
    total = a.numel()
    a = a.reshape(-1)
    e = e.reshape(-1)
    CH = 1 << 22
    n_diff = 0; max_err = 0.0; mean = 0.0
    first8 = []
    for i in range(0, total, CH):
        aa = a[i:i+CH]; ee = e[i:i+CH]
        neq = (aa != ee)
        nd = int(neq.sum())
        n_diff += nd
        d = (aa - ee).abs()
        if d.numel():
            max_err = max(max_err, float(d.max()))
            mean += float(d.sum())
        if nd and len(first8) < 8:
            idx = neq.nonzero()[:8-len(first8)].squeeze(1) + i
            first8.extend(idx.tolist())
    mean /= total
    # magnitude histogram on all diffs (vectorized via gather of neq positions is costly; do cheap)
    dv_max = 0.0; dv_min = float("inf"); dv_mean = 0.0; n_ge1 = 0; n_ge2 = 0; n_ge6 = 0
    for i in range(0, total, CH):
        aa = a[i:i+CH]; ee = e[i:i+CH]
        d = (aa - ee).abs()
        nd = int((d > 0).sum())
        if nd:
            dv_max = max(dv_max, float(d.max()))
            m = d[d > 0]
            dv_min = min(dv_min, float(m.min()))
            dv_mean += float(m.sum())
            n_ge1 += int((m >= 0.01).sum()); n_ge2 += int((m >= 0.02).sum()); n_ge6 += int((m >= 0.06).sum())
    if n_diff: dv_mean /= n_diff
    return n_diff, max_err, mean, total, first8, (dv_max, dv_min, dv_mean, n_ge1, n_ge2, n_ge6)

n1, mx1, mn1, total, f1, h1 = run_once(0)
torch.cuda.empty_cache()
n2, mx2, mn2, total2, f2, h2 = run_once(0)
print(f"case={case.operation} n={n} total={total}")
print(f"run1: n_diff={n1} ({100*n1/total:.4f}%) max_err={mx1:.6g} mean={mn1:.6g}")
print(f"run2: n_diff={n2} ({100*n2/total:.4f}%) max_err={mx2:.6g} mean={mn2:.6g}")
print(f"deterministic(same n_diff): {n1 == n2}; first8 same: {f1 == f2}")
print("run1 first8:", f1, "lastdim(512):", [p % 512 for p in f1])
print(f"run1 diff errs: max={h1[0]:.6g} min={h1[1]:.6g} mean={h1[2]:.6g}")
print(f"run1 errs>=0.01: {h1[3]} >=0.02: {h1[4]} >=0.06: {h1[5]}")
