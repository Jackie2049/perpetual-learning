#!/usr/bin/env python3
"""DeepInfer 去 PDL 改写数值对照扩展：load/store_kv_cache 全矩阵（4090-1 SM89，M5）。

方法：monkeypatch ``ops.load_paged_kv_cache_tl`` / ``ops.store_kv_cache_tl`` 指向
``cuda/*_pdlfree.py`` 变体（工厂名与签名一致，只把 PDL 调用替换为 sync_grid/pass），
然后直接复用上游 wrapper 模块里的原始测试函数（``test_load_paged_kv_cache`` /
``test_store_cache``，内部与 torch 参考 ``load_paged_kv_cache_ref`` /
``store_kv_cache_ref`` 逐字节对照），覆盖 KV_CACHE_TEST_CASES 全矩阵：
5 种布局（V41 e4m3 / V41 FP4 / FP4 index / FP8-fp32-sf index / MXFP8 index）
x block_size(64,133|32,64) x num_tokens(1,233,67890) + store 三次 update
（负值/幅度扫描/全零量化）。

等价于在真实 SM89 上跑"PDL-free 版 kernel 的官方测试套件"，对照目标为 torch
参考实现而非自写探针参考，证据强度高于手工构造。
"""

import importlib.util
import os
import sys

import torch

_CUDA_DIR = "/home/zxw/deepinfer-civilian/deepinfer/model_executor/layers/tilelang_ops/cuda"
_VARIANT = os.environ.get("KV_PROBE_VARIANT", "pdlfree")  # pdlfree | nosync


def _hide_pdlfree_variants():
    """ops 模块遍历 backend 包时会与同名 *_tl 工厂冲突，先临时隐藏变体文件。"""
    hidden = []
    for f in sorted(os.listdir(_CUDA_DIR)):
        if f.endswith("_pdlfree.py"):
            src = os.path.join(_CUDA_DIR, f)
            dst = src + ".hidden"
            os.rename(src, dst)
            hidden.append((src, dst))
    return hidden


def _restore(hidden):
    for src, dst in hidden:
        os.rename(dst, src)


def _load_variant(module_name: str, file_path: str) -> object:
    """以包内模块名从文件加载变体，保持相对导入可用。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- 前置：把 ops 的两个 kv-cache 工厂指向 _pdlfree 变体 ----
hidden = _hide_pdlfree_variants()
try:
    from deepinfer.model_executor.layers.tilelang_ops import ops  # noqa: E402

    _restore(hidden)
    hidden = []

    pkg = "deepinfer.model_executor.layers.tilelang_ops.cuda"
    # pdlfree 变体在 cuda 包内；nosync 变体在 /tmp（避免与包内 *_pdlfree 混淆）
    _var_dir = _CUDA_DIR if _VARIANT == "pdlfree" else "/tmp"
    load_path = os.path.join(_var_dir, f"load_paged_kv_cache_{_VARIANT}.py")
    store_path = os.path.join(_var_dir, f"store_kv_cache_{_VARIANT}.py")
    load_var = _load_variant(f"{pkg}.load_paged_kv_cache_{_VARIANT}", load_path)
    store_var = _load_variant(f"{pkg}.store_kv_cache_{_VARIANT}", store_path)
    _pdlfree_load_tl = load_var.load_paged_kv_cache_tl
    _pdlfree_store_tl = store_var.store_kv_cache_tl

    assert ops.load_paged_kv_cache_tl is not _pdlfree_load_tl
    assert ops.store_kv_cache_tl is not _pdlfree_store_tl
    ops.load_paged_kv_cache_tl = _pdlfree_load_tl
    ops.store_kv_cache_tl = _pdlfree_store_tl
finally:
    if hidden:
        _restore(hidden)

# ---- 复用上游原始测试函数（parametrize 包装前） ----
from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import (  # noqa: E402
    _LOAD_TEST_CASES,
    test_load_paged_kv_cache,
)
from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import (  # noqa: E402
    _STORE_TEST_CASES,
    test_store_cache,
)

raw_load = getattr(test_load_paged_kv_cache, "__wrapped__", test_load_paged_kv_cache)
raw_store = getattr(test_store_cache, "__wrapped__", test_store_cache)


def run_case(name: str, fn, args) -> bool:
    try:
        fn(*args)
        print(f"PASS {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    torch.cuda.set_device(0)
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    print(f"monkeypatched ops: load={ops.load_paged_kv_cache_tl.__module__} store={ops.store_kv_cache_tl.__module__}")

    load_cases = [_values(p) for p in _LOAD_TEST_CASES]
    store_cases = [_values(p) for p in _STORE_TEST_CASES]
    print(f"load cases: {len(load_cases)}, store cases: {len(store_cases)}")

    results: dict[str, tuple[int, int]] = {"load": (0, 0), "store": (0, 0)}
    for case in load_cases:
        # (layout, block_size, num_tokens, pattern)
        ok = run_case("load " + _id(case), raw_load, case)
        results["load"] = (results["load"][0] + ok, results["load"][1] + 1)
    for case in store_cases:
        # (layout, block_size, n, slot_pattern)
        ok = run_case("store " + _id(case), raw_store, case)
        results["store"] = (results["store"][0] + ok, results["store"][1] + 1)

    for k, (passed, total) in results.items():
        print(f"== {k}: {passed}/{total} passed ==")
    return 0 if all(p == t for p, t in results.values()) else 1


def _values(p):
    # pytest.param objects expose .values; plain tuples are fine as-is.
    return getattr(p, "values", tuple(p))


def _id(case) -> str:
    # layout note + block_size + tokens (+ slot pattern), short readable label
    layout = case[0]
    parts = [getattr(layout, "note", str(layout))[:20]]
    for v in case[1:]:
        parts.append(str(v))
    return "|".join(parts)


if __name__ == "__main__":
    sys.exit(main())
