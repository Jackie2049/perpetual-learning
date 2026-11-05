#!/usr/bin/env python3
"""端到端：真实 `GELUGemmOp.__init__` / `forward` 源码 + 真实 triton `fused_gemm_gelu`，
只桩掉 `inference_module`（C++ 内核）与 `matmul_ext`，在 CPU 上确认 fp16+triton 时的实际行为。

这不是 inspect 层面的推断：走的是 `forward` 的**真实函数体**，函数指针也指向**真实 triton 函数**。

三档读数：
  A. dtype=float16 + HAS_TRITON + use_triton=True     → 期望 TypeError（缺 q_int8）
  B. dtype=float16 + use_triton=False                  → 指针指向 C++ 桩（8 实参对齐）→ 正常返回
  C. dtype=bfloat16 + use_triton=True                  → 指针指向 C++ 桩（bf16 分支不查 triton）→ 正常返回

    python3 probe_gelu_e2e.py --repo /tmp/tm025-deepspeed
    python3 probe_gelu_e2e.py --repo ... --selftest
"""
import argparse
import ast
import inspect
import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_gelu_argshift import TRITON_OPS, GELU_BINDING, load, find_def, build_triton_fn, _Recorder  # noqa: E402


class _TorchStub:
    """只为 `self.config.dtype in [torch.float16, torch.int8]` 与 `== torch.int8` 求值。"""

    float16 = "FLOAT16"
    bfloat16 = "BFLOAT16"
    int8 = "INT8"
    float32 = "FLOAT32"
    Tensor = "TORCH_TENSOR"  # 只用于类型注解

    @staticmethod
    def empty(n):
        return f"<torch.empty({n})>"


def _load_gelu_class(repo: Path, dtype, has_triton: bool, use_triton: bool):
    """执行真实的 GELUGemmOp 类体（只桩掉外部依赖）。"""
    src = load(repo, GELU_BINDING)
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GELUGemmOp")
    cls_src = textwrap.dedent(ast.get_source_segment(src, cls))

    calls = []

    def cpp_kernel(*a, **kw):
        calls.append(("cpp", len(a), [type(x).__name__ for x in a]))
        return "CPP_OUTPUT"

    class _InferenceModule:
        fused_gemm_gelu_fp16 = staticmethod(cpp_kernel)
        fused_gemm_gelu_bf16 = staticmethod(cpp_kernel)
        fused_gemm_gelu_fp32 = staticmethod(cpp_kernel)
        fused_gemm_gelu_int8 = staticmethod(cpp_kernel)

    class _BaseOp:
        def __init__(self, config):
            self.config = config
            self.inference_module = _InferenceModule()

    rec = _Recorder()
    triton_fn, _ = build_triton_fn(repo, rec)

    class _TritonOpsModule:
        fused_gemm_gelu = staticmethod(triton_fn)

    # 真实类体里 `from deepspeed.ops...triton.ops import fused_gemm_gelu as _x` 用的是这个替身
    import types as _t

    fake_pkg = {"deepspeed": _t.SimpleNamespace(
        HAS_TRITON=has_triton,
        ops=_t.SimpleNamespace(transformer=_t.SimpleNamespace(inference=_t.SimpleNamespace(
            triton=_t.SimpleNamespace(ops=_TritonOpsModule)))),
    )}

    ns = {
        "torch": _TorchStub,
        "deepspeed": fake_pkg["deepspeed"],
        "BaseOp": _BaseOp,
        "DeepSpeedInferenceConfig": object,
        "F": None,
    }
    # 类体里的 import 语句要能执行：用自定义 __import__ 拦截
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "deepspeed":
            return fake_pkg["deepspeed"]
        if name.startswith("deepspeed.ops.transformer.inference.triton.ops"):
            return _TritonOpsModule()
        return real_import(name, *args, **kwargs)

    ns["__builtins__"] = dict(__builtins__) if isinstance(__builtins__, dict) else dict(vars(__builtins__))
    ns["__builtins__"]["__import__"] = fake_import

    exec(compile(cls_src, "<gelu_gemm>", "exec"), ns)

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.dtype = dtype
    cfg.use_triton = use_triton
    cfg.transposed_mode = False
    return ns["GELUGemmOp"], cfg, rec, calls


def run_arm(repo: Path, label, dtype, has_triton, use_triton):
    cls, cfg, rec, calls = _load_gelu_class(repo, dtype, has_triton, use_triton)
    op = cls(cfg)
    ptr = getattr(op, "fused_gemm_gelu", None)
    ptr_name = getattr(ptr, "__name__", repr(ptr))
    try:
        # 真实 forward：8 个位置实参，input/weight 等用字符串占位（matmul 是桩）
        out = op.forward("IN", "W", "B", "WOUT")
        return {"arm": label, "pointer": ptr_name, "outcome": "returned", "output": out,
                "triton_calls": len(rec.calls)}
    except TypeError as e:
        return {"arm": label, "pointer": ptr_name, "outcome": "TypeError", "error": str(e),
                "triton_calls": len(rec.calls)}
    except Exception as e:  # noqa: BLE001
        return {"arm": label, "pointer": ptr_name, "outcome": type(e).__name__, "error": str(e),
                "triton_calls": len(rec.calls)}


def analyze(repo: Path) -> dict:
    arms = [
        run_arm(repo, "A_fp16_triton_on", _TorchStub.float16, True, True),
        run_arm(repo, "B_fp16_triton_off", _TorchStub.float16, True, False),
        run_arm(repo, "C_bf16_triton_on", _TorchStub.bfloat16, True, True),
        run_arm(repo, "D_fp16_triton_on_no_triton_installed", _TorchStub.float16, False, True),
    ]
    by = {a["arm"]: a for a in arms}
    checks = [
        {"id": "E1.A_selects_triton", "ok": "fused_gemm_gelu" == by["A_fp16_triton_on"]["pointer"],
         "detail": f"A 臂指针 = {by['A_fp16_triton_on']['pointer']}"},
        {"id": "E2.A_raises_TypeError", "ok": by["A_fp16_triton_on"]["outcome"] == "TypeError",
         "detail": f"A 臂结果 = {by['A_fp16_triton_on']['outcome']}: {by['A_fp16_triton_on'].get('error')}"},
        {"id": "E3.B_succeeds", "ok": by["B_fp16_triton_off"]["outcome"] == "returned",
         "detail": f"B 臂（关 triton）结果 = {by['B_fp16_triton_off']['outcome']}"},
        {"id": "E4.C_succeeds", "ok": by["C_bf16_triton_on"]["outcome"] == "returned",
         "detail": f"C 臂（bf16）结果 = {by['C_bf16_triton_on']['outcome']}"},
        {"id": "E5.D_falls_back", "ok": by["D_fp16_triton_on_no_triton_installed"]["outcome"] in ("returned",),
         "detail": f"D 臂（未装 triton）结果 = {by['D_fp16_triton_on_no_triton_installed']['outcome']}，"
                   f"指针 = {by['D_fp16_triton_on_no_triton_installed']['pointer']}"},
    ]
    ok = all(c["ok"] for c in checks)
    return {"arms": arms, "checks": checks,
            "verdict": "E2E_CONFIRMED" if ok else "CHANGED",
            "failed": [c["id"] for c in checks if not c["ok"]]}


def selftest(repo: Path):
    base = analyze(repo)
    res = [("baseline_confirmed", base["verdict"] == "E2E_CONFIRMED", base["failed"])]

    def with_mut(rel, old, new, label):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "r"
            shutil.copytree(repo, tmp, ignore=shutil.ignore_patterns(".git"))
            p = tmp / rel
            s = p.read_text(encoding="utf-8")
            if old not in s:
                res.append((label, False, "注入目标不存在"))
                return
            p.write_text(s.replace(old, new, 1), encoding="utf-8")
            r = analyze(tmp)
            res.append((label, r["verdict"] != "E2E_CONFIRMED", r["failed"]))

    # I1: 把调用点补成 11 个位置实参 ⇒ A 臂不该再 TypeError
    with_mut(GELU_BINDING,
             "            self.config.dtype == torch.int8,\n            self.config.transposed_mode)",
             "            self.config.dtype == torch.int8,\n            self.config.transposed_mode,\n"
             "            self.config.dtype == torch.int8,\n            False,\n            self.config.transposed_mode)",
             "inj_arity_fixed")
    # I2: triton 分支改成永远不选 ⇒ A 臂指针变 C++ 桩
    with_mut(GELU_BINDING,
             "if deepspeed.HAS_TRITON and self.config.use_triton and self.config.dtype == torch.float16:",
             "if False:", "inj_triton_unselectable")

    print(json.dumps([{"id": a, "caught": b, "failed": c} for a, b, c in res], indent=1))
    bad = [a for a, b, _ in res if not b]
    if bad:
        print(f"SELFTEST FAILED: {bad}", file=sys.stderr)
        return 2
    print(f"SELFTEST OK: {len(res)}/{len(res)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()

    if a.selftest:
        return selftest(repo)

    rep = analyze(repo)
    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(txt, encoding="utf-8")
    for arm in rep["arms"]:
        print(f"[ARM] {arm['arm']:<42} ptr={arm['pointer']:<22} {arm['outcome']}"
              f"{'  ' + arm['error'] if arm.get('error') else ''}")
    for c in rep["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['id']:<32} {c['detail']}")
    print(f"\nverdict: {rep['verdict']}  ({len(rep['checks'])} checks, {len(rep['failed'])} failed)")
    return 0 if rep["verdict"] == "E2E_CONFIRMED" else 1


if __name__ == "__main__":
    sys.exit(main())
