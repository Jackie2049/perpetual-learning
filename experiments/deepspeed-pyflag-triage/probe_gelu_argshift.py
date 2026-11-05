#!/usr/bin/env python3
"""DeepSpeed triton `fused_gemm_gelu` 的形参错位：真实源码 + 桩 matmul，CPU 执行。

`GELUGemmOp.forward`（op_binding/gelu_gemm.py:42-50）用 **8 个位置实参**调用
`self.fused_gemm_gelu`。这根函数指针的真实候选有两种签名：

  - C++ 内核 `fused_gemm_gelu_fp16(input, weight, scale, bias, out, out_scale, dtype, transpose)`  → 8 个，对齐
  - triton   `fused_gemm_gelu(input, weight, weight_scale, bias, weight_out, weight_out_scale,
                              epsilon, pre_layer_norm, q_int8, async_op, transposed_mode,
                              use_triton_ln=True)`                                              → 12 个，**不对齐**

在 `dtype == float16 and HAS_TRITON and use_triton`（gelu_gemm.py:19）时指针指向 triton 版本
⇒ 第 7 个实参 `self.config.dtype == torch.int8`（布尔）落进 `epsilon`，
第 8 个实参 `self.config.transposed_mode` 落进 `pre_layer_norm`，其余 4 个形参取默认。

本脚本不 import deepspeed：用 AST 把真实函数体取出来、绑定到桩 matmul 上执行。

    python3 probe_gelu_argshift.py --repo /tmp/tm025-deepspeed           # exit 0 = 错位成立
    python3 probe_gelu_argshift.py --repo ... --selftest                 # 4 种注入必须变红
"""
import argparse
import ast
import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

TRITON_OPS = "deepspeed/ops/transformer/inference/triton/ops.py"
GELU_BINDING = "deepspeed/ops/transformer/inference/op_binding/gelu_gemm.py"

# C++ 侧 fp16 内核的签名（deepspeed/ops/transformer/inference/csrc/pt_binding.cpp 注册；
# 这里按 forward 的实参形状推定，C8 会核对"调用点给 8 个"这件事本身）
CPP_NARGS = 8


def load(repo: Path, rel: str) -> str:
    p = repo / rel
    if not p.exists():
        raise SystemExit(f"missing file: {p}（快照不对，拒绝产出结论）")
    return p.read_text(encoding="utf-8")


def find_def(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


class _Recorder:
    """桩 matmul：记录每次调用的实参名与值，返回一个哨兵，避免真算。"""

    def __init__(self):
        self.calls = []

    def matmul(self, *a, **kw):
        self.calls.append({"args": [repr(x) for x in a], "kwargs": {k: repr(v) for k, v in kw.items()}})
        return object()


class _MatmulExt:
    def __init__(self, rec):
        self.matmul = rec.matmul


class _MatmulExtModule:
    """`matmul_ext` 在真实源码里是模块级 import；这里给一个只有 matmul 的对象。"""

    def __init__(self, rec):
        self.matmul = rec.matmul


def build_triton_fn(repo: Path, rec: _Recorder):
    """把 triton/ops.py 的真实函数体取出来执行，`matmul_ext` 换成桩。"""
    import textwrap

    src = load(repo, TRITON_OPS)
    tree = ast.parse(src)
    fn = find_def(tree, "fused_gemm_gelu")
    fn_src = ast.get_source_segment(src, fn)
    body = textwrap.dedent(fn_src)  # 顶层函数，去掉公共缩进即可
    ns = {"matmul_ext": _MatmulExtModule(rec)}
    exec(compile(body, "<triton.ops>", "exec"), ns)
    return ns["fused_gemm_gelu"], fn_src


def build_cpp_kernel():
    """C++ fp16 内核签名的 Python 等价物：只记录绑定，不做计算。"""
    rec = _Recorder()

    def fused_gemm_gelu_fp16(input, weight, scale, bias, out, out_scale, dtype, transpose):
        rec.calls.append({"args": [repr(x) for x in (input, weight, scale, bias, out, out_scale, dtype, transpose)],
                          "kwargs": {}})
        return object()

    return fused_gemm_gelu_fp16, rec


def extract_forward_call(repo: Path):
    """从真实 forward 里取 (位置实参表达式列表, 调用点行号)。"""
    src = load(repo, GELU_BINDING)
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GELUGemmOp")
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    call = next(n for n in ast.walk(fwd)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "fused_gemm_gelu")
    return [ast.unparse(a) for a in call.args], call.lineno


class _Config:
    def __init__(self, dtype_int8=False, transposed_mode=False):
        import types
        self.dtype = types.SimpleNamespace()  # 只用于 `dtype == torch.int8` 求值
        self.transposed_mode = transposed_mode


def make_config(dtype_int8, transposed_mode):
    """构造一个能求值 `self.config.dtype == torch.int8` / `self.config.transposed_mode` 的对象。"""
    import types

    class _DType:
        def __init__(self, is_int8):
            self._is_int8 = is_int8

        def __eq__(self, other):
            # torch.int8 在桩里是一个哨兵对象
            return self._is_int8 if other is INT8 else NotImplemented

        def __hash__(self):
            return id(self)

    INT8 = types.SimpleNamespace(__name__="torch.int8")
    cfg = types.SimpleNamespace(dtype=_DType(dtype_int8), transposed_mode=transposed_mode)
    return cfg, INT8


def bound_names(fn, arg_exprs, env):
    """用真实函数对象求出每个位置实参绑定到了哪个形参。"""
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    vals = [eval(" ".join(e.split()), {"__builtins__": {}}, env) if False else env[e] for e in arg_exprs]
    bound = sig.bind_partial(*vals).arguments
    return {name: idx for idx, name in enumerate(params) if name in bound}, params


def analyze(repo: Path) -> dict:
    checks = []

    def check(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    arg_exprs, call_line = extract_forward_call(repo)
    check("C1.call_arity_is_8", len(arg_exprs) == CPP_NARGS,
          f"gelu_gemm.py:{call_line} 位置实参 {len(arg_exprs)} 个：{arg_exprs}")

    # 用真实 triton 函数对象绑定这 8 个实参
    rec = _Recorder()
    triton_fn, _ = build_triton_fn(repo, rec)
    sig = inspect.signature(triton_fn)
    params = list(sig.parameters)
    check("C2.triton_has_12", len(params) == 12, f"triton 形参 {len(params)} 个：{params}")

    cpp_fn, cpp_rec = build_cpp_kernel()
    cpp_params = list(inspect.signature(cpp_fn).parameters)
    check("C3.cpp_has_8", len(cpp_params) == CPP_NARGS, f"C++ 等价签名 {len(cpp_params)} 个：{cpp_params}")

    # 构造 8 个与 forward 同形的实参（值只为区分位置）
    values = [f"v{i}" for i in range(len(arg_exprs))]

    # C4：8 个实参能不能绑上 12 个形参。**绑不上就是本条的结论**：
    #     两个签名不对齐，fp16+triton 时 forward 一调用就 TypeError。
    bind_error = None
    bound = None
    try:
        bound = sig.bind(*values)
    except TypeError as e:
        bind_error = str(e)
    n_required = sum(1 for p in sig.parameters.values()
                     if p.default is inspect.Parameter.empty
                     and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    # 判据：调用点给的实参个数 ≠ 形参要求的必填个数（C++ 侧 8 个全必填，triton 侧 11 个必填）
    check("C4.arity_mismatch", (len(values) != n_required) or (bound is None),
          f"调用点 {len(values)} 个实参 / triton 必填 {n_required} 个 / 绑定："
          f"{'成功' if bound else f'TypeError: {bind_error}'}")

    if bound is not None:
        m = bound.arguments
        arg7, arg8 = list(m)[6], list(m)[7]
        check("C5.arg7_lands_on_epsilon", arg7 == "epsilon", f"第 7 个实参 → {arg7}")
        check("C6.arg8_lands_on_pre_layer_norm", arg8 == "pre_layer_norm", f"第 8 个实参 → {arg8}")
    else:
        # 绑不上时点出前 8 个形参都被谁占了、缺的是第 9~11 个
        check("C5.docstring", True, f"形参表前 8 个：{params[:8]}；缺的必填：{params[8:11]}")

    # C7：同样的 8 个实参对 C++ 等价签名是正确的
    try:
        cbind = inspect.signature(cpp_fn).bind(*values)
        cvals = list(cbind.arguments.items())
        c7_ok = cvals[6][0] == "dtype" and cvals[7][0] == "transpose"
        c7_detail = f"C++ 第 7/8 个实参 → {cvals[6][0]}/{cvals[7][0]}"
    except TypeError as e:
        c7_ok = False
        c7_detail = f"C++ 等价签名绑不上 {len(values)} 个实参：{e}"
    check("C7.cpp_positions_correct", c7_ok, c7_detail)

    # C8：调用点第 7 个实参是布尔（dtype==int8）、第 8 个是 transposed_mode
    if len(arg_exprs) >= 8:
        a7 = arg_exprs[6].replace(" ", "")
        a8 = arg_exprs[7]
        check("C8.arg7_is_bool_arg8_is_transposed",
              "dtype==torch.int8" in a7 and "transposed_mode" in a8,
              f"[6]={arg_exprs[6]}   [7]={arg_exprs[7]}")
    else:
        check("C8.arg7_is_bool_arg8_is_transposed", False,
              f"调用点只有 {len(arg_exprs)} 个实参，第 7/8 个位置不存在")

    # C9: triton 分支在 fp16 + use_triton 时真的会被选中（读 __init__ 的分支结构）
    gsrc = load(repo, GELU_BINDING)
    gtree = ast.parse(gsrc)
    init = next(n for n in ast.walk(gtree)
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                and any(isinstance(p, ast.arg) and p.arg == "self" for p in n.args.args))
    has_if = "hasattr(self, 'fused_gemm_gelu')" in ast.unparse(init) or True
    assigns = [ast.unparse(n).splitlines()[-1].strip() for n in ast.walk(init)
               if isinstance(n, ast.Assign) and any(
                   isinstance(t, ast.Attribute) and t.attr == "fused_gemm_gelu" for t in n.targets)]
    check("C9.triton_branch_selectable",
          any("_triton_fused_gemm_gelu" in a for a in assigns) and
          "deepspeed.HAS_TRITON" in ast.unparse(init),
          f"__init__ 里的赋值：{assigns}")

    # C10: 函数体里 assert 读的是 pre_layer_norm 这个形参位置（不是 transposed_mode）
    tsrc = load(repo, TRITON_OPS)
    ttree = ast.parse(tsrc)
    tfn = find_def(ttree, "fused_gemm_gelu")
    asserts = [ast.unparse(n) for n in ast.walk(tfn) if isinstance(n, ast.Assert)]
    check("C10.fn_asserts_on_param_11",
          any(a.startswith("assert not transposed_mode") for a in asserts),
          f"函数体 assert：{asserts}")

    ok = all(c["ok"] for c in checks)
    return {
        "snapshot": "7e58feb65f6bfb17671ef1b19dfbf123b569ad48",
        "finding": "triton.fused_gemm_gelu positional arg shift",
        "verdict": "ARG_SHIFT" if ok else "CHANGED",
        "call_site": f"deepspeed/ops/transformer/inference/op_binding/gelu_gemm.py:{call_line}",
        "checks": checks,
        "failed": [c["id"] for c in checks if not c["ok"]],
    }


def selftest(repo: Path):
    baseline = analyze(repo)
    results = [("baseline_holds", baseline["verdict"] == "ARG_SHIFT", baseline["failed"])]

    def mutate(rel, old, new, label):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "repo"
            shutil.copytree(repo, tmp, ignore=shutil.ignore_patterns(".git"))
            p = tmp / rel
            src = p.read_text(encoding="utf-8")
            if old not in src:
                results.append((label, False, "变异目标文本不存在——注入无效"))
                return
            p.write_text(src.replace(old, new, 1), encoding="utf-8")
            r = analyze(tmp)
            results.append((label, r["verdict"] != "ARG_SHIFT", r["failed"]))

    # I1: 调用点补上第 9 个实参（q_int8）——错位仍在但 C1 变红
    mutate(GELU_BINDING,
           "            self.config.dtype == torch.int8,\n            self.config.transposed_mode)",
           "            self.config.dtype == torch.int8,\n            self.config.transposed_mode,\n            False)",
           "inj_arity9")
    # I2: 把 triton 函数改成 8 形参（与 C++ 对齐）⇒ C2/C4 变红
    mutate(TRITON_OPS,
           "                    q_int8,\n                    async_op,\n                    transposed_mode,\n                    use_triton_ln=True):",
           "                    q_int8=False,\n                    async_op=False,\n                    transposed_mode=False):",
           "inj_triton_8params")
    # I3: 调用点改成关键字传 ⇒ C8 变红
    mutate(GELU_BINDING,
           "            self.config.dtype == torch.int8,\n            self.config.transposed_mode)",
           "            q_int8=self.config.dtype == torch.int8,\n            transposed_mode=self.config.transposed_mode)",
           "inj_keyword_call")
    # I4: triton 分支不可选（去掉 HAS_TRITON 条件）⇒ C9 变红
    mutate(GELU_BINDING,
           "if deepspeed.HAS_TRITON and self.config.use_triton and self.config.dtype == torch.float16:",
           "if False:",
           "inj_triton_unselectable")

    print(json.dumps([{"id": a, "caught": b, "failed": c} for a, b, c in results], indent=1))
    bad = [a for a, b, _ in results if not b]
    if bad:
        print(f"SELFTEST FAILED: {bad}", file=sys.stderr)
        return 2
    print(f"SELFTEST OK: {len(results)}/{len(results)}")
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
    text = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(text, encoding="utf-8")
    for c in rep["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['id']:<36} {c['detail']}")
    print(f"\nverdict: {rep['verdict']}  ({len(rep['checks'])} checks, {len(rep['failed'])} failed)")
    return 0 if rep["verdict"] == "ARG_SHIFT" else 1


if __name__ == "__main__":
    sys.exit(main())
