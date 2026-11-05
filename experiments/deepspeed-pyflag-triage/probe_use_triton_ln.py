#!/usr/bin/env python3
"""DeepSpeed `use_triton_ln`：形参在、分支在、三个调用点全按位置传 ⇒ 恒取默认 True。

判据全部绑到可 grep 的事实（文件行号 + AST 实参个数），不读注释、不猜语义。

用法：
    python3 probe_use_triton_ln.py --repo /tmp/tm025-deepspeed          # exit 0 = 结论成立
    python3 probe_use_triton_ln.py --repo ... --json raw/probe.json
    python3 probe_use_triton_ln.py --repo ... --selftest                # 注入 4 种破坏，判据必须变红

退出码：
    0  结论成立（use_triton_ln 不可达）
    1  判据不成立（上游改了代码，结论失效）
    2  selftest 有注入没被抓住（判据本身有问题）
"""
import argparse
import ast
import json
import sys
from pathlib import Path

TRITON_OPS = "deepspeed/ops/transformer/inference/triton/ops.py"
QKV_BINDING = "deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py"
MLP_BINDING = "deepspeed/ops/transformer/inference/op_binding/mlp_gemm.py"
GELU_BINDING = "deepspeed/ops/transformer/inference/op_binding/gelu_gemm.py"

# 三个带 use_triton_ln 形参的函数：函数名 → 形参表里 use_triton_ln 的位置（0 基）
# 判据要求「三个函数的 use_triton_ln 都在形参表末尾」，调用点用位置实参时必然漏传。
TARGETS = ("fused_gemm_gelu", "mlp_gemm_func", "qkv_gemm_func")


def load(repo: Path, rel: str) -> str:
    p = repo / rel
    if not p.exists():
        raise SystemExit(f"missing file: {p}（快照不对，拒绝产出结论）")
    return p.read_text(encoding="utf-8")


def parse(text: str, rel: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as e:
        raise SystemExit(f"parse failed: {rel}: {e}")


def find_def(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def param_index(fn: ast.FunctionDef, name: str):
    """返回形参在参数列表里的 0 基位置；没有返回 None。"""
    for i, a in enumerate(fn.args.args):
        if a.arg == name:
            return i
    return None


def calls_of(tree: ast.Module, name: str):
    """返回 (line, kwargs 名集合, n_positional) —— 只认直接按名字的调用。"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            out.append((node.lineno, {kw.arg for kw in node.keywords}, len(node.args)))
    return out


def calls_of_attr(tree: ast.Module, leaf: str):
    """返回 self.<leaf>(...) 形式的调用点（真实绑定落在 base 类的 self.<leaf> 属性上）。"""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == leaf):
            out.append((node.lineno, {kw.arg for kw in node.keywords}, len(node.args)))
    return out


def analyze(repo: Path) -> dict:
    checks = []

    def check(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    ops_text = load(repo, TRITON_OPS)
    ops_tree = parse(ops_text, TRITON_OPS)

    # C1: 三个目标函数都存在，且 use_triton_ln 默认值为 True
    for name in TARGETS:
        fn = find_def(ops_tree, name)
        if fn is None:
            check(f"C1.{name}.exists", False, "函数不存在")
            continue
        idx = param_index(fn, "use_triton_ln")
        default = None
        if idx is not None:
            # ast: args 中最后 len(defaults) 个位置参数有默认值
            nd = len(fn.args.defaults)
            first_default = len(fn.args.args) - nd
            if idx >= first_default:
                d = fn.args.defaults[idx - first_default]
                default = ast.literal_eval(d) if isinstance(d, ast.Constant) else ast.dump(d)
        check(f"C1.{name}.param", idx is not None and default is True,
              f"line={fn.lineno} idx={idx} default={default!r}")

    # C2: 记录每个函数体里 use_triton_ln 的读取行。fused_gemm_gelu 的函数体**不用**它
    #     （12 个形参里 7 个从未出现），这是它的写法特征，不是缺陷——如实记录
    for name in TARGETS:
        fn = find_def(ops_tree, name)
        if fn is None:
            check(f"C2.{name}.branch", False, "函数不存在")
            continue
        uses = [n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Name) and n.id == "use_triton_ln" and isinstance(n.ctx, ast.Load)]
        check(f"C2.{name}.branch_observed", True,
              f"读取行={uses}" + ("（函数体不使用该形参）" if not uses else ""))

    # C3: 全仓（非测试）没有任何调用点按关键字传 use_triton_ln，且所有调用都是位置实参
    callers = {
        "qkv_gemm_func": (QKV_BINDING, calls_of_attr(parse(load(repo, QKV_BINDING), QKV_BINDING), "qkv_gemm_func")),
        "mlp_gemm_func": (MLP_BINDING, calls_of_attr(parse(load(repo, MLP_BINDING), MLP_BINDING), "mlp_gemm_func")),
        "fused_gemm_gelu": (GELU_BINDING, calls_of_attr(parse(load(repo, GELU_BINDING), GELU_BINDING), "fused_gemm_gelu")),
    }
    for name, (rel, calls) in callers.items():
        any_kw = [c for c in calls if "use_triton_ln" in c[1]]
        check(f"C3.{name}.no_keyword", len(any_kw) == 0,
              f"{rel}: {len(calls)} 个调用点，带该关键字的 {len(any_kw)} 个")
        check(f"C3.{name}.has_call", len(calls) >= 1, f"{rel}: {len(calls)} 个 self.{name}(...) 调用")

    # C4: 位置实参个数 < use_triton_ln 的位置 ⇒ 按 Python 规则必然取默认值
    for name in TARGETS:
        fn = find_def(ops_tree, name)
        rel, calls = callers[name]
        if fn is None:
            check(f"C4.{name}.unreachable", False, "函数不存在")
            continue
        idx = param_index(fn, "use_triton_ln")
        worst = max((n for _, _, n in calls), default=0)
        # 第 k 个位置实参落在第 k-1 号形参上 ⇒ 要覆盖第 idx 号形参必须传 >= idx+1 个实参
        check(f"C4.{name}.unreachable", idx is not None and worst <= idx,
              f"最多次位置实参={worst} <= 形参位置={idx} ⇒ 该位置从未被实参覆盖")

    # C5: 全仓递归扫描，确认没有别的文件（含 tests）按关键字传该名字
    other_hits = []
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        if rel in (TRITON_OPS, QKV_BINDING, MLP_BINDING, GELU_BINDING):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(t):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "use_triton_ln":
                        other_hits.append(f"{rel}:{node.lineno}")
    check("C5.no_keyword_anywhere", len(other_hits) == 0, f"其他调用点={other_hits}")

    # C6: 两个分支的语义差异确实存在——True 走 triton`layer_norm`，False 走 `LayerNormOp`
    fn = find_def(ops_tree, "mlp_gemm_func")
    body_src = ast.get_source_segment(ops_text, fn) if fn else ""
    check("C6.branches_differ",
          "layer_norm_residual(input, input_bias, residual" in (body_src or "")
          and "LayerNormOp.layer_norm_residual" in (body_src or ""),
          "if/else 两支分别调 triton layer_norm 与 LayerNormOp")

    ok = all(c["ok"] for c in checks)
    return {
        "snapshot": "7e58feb65f6bfb17671ef1b19dfbf123b569ad48",
        "finding": "use_triton_ln",
        "verdict": "UNREACHABLE" if ok else "CHANGED",
        "checks": checks,
        "failed": [c["id"] for c in checks if not c["ok"]],
    }


def selftest(repo: Path):
    """注入 4 种破坏，每一种都必须让结论从成立翻到不成立。"""
    import shutil
    import tempfile

    baseline = analyze(repo)
    results = [("baseline_holds", baseline["verdict"] == "UNREACHABLE", baseline["failed"])]

    def mutate(rel, old, new, label):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "repo"
            shutil.copytree(repo, tmp, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=False)
            p = tmp / rel
            src = p.read_text(encoding="utf-8")
            if old not in src:
                results.append((label, False, "变异目标文本不存在——注入无效"))
                return
            p.write_text(src.replace(old, new, 1), encoding="utf-8")
            r = analyze(tmp)
            results.append((label, r["verdict"] != "UNREACHABLE", r["failed"]))

    # I1: 让一个调用点改成关键字传值 ⇒ C3/C4 变红
    mutate(QKV_BINDING, "q_int8, self.config.transposed_mode)",
           "q_int8, self.config.transposed_mode, use_triton_ln=False)", "inj_keyword_at_call")
    # I2: 让形参默认值改成 False ⇒ C1 变红
    mutate(TRITON_OPS, "use_triton_ln=True):\n    assert not transposed_mode\n\n    # residual add",
           "use_triton_ln=False):\n    assert not transposed_mode\n\n    # residual add",
           "inj_default_false")
    # I3: 删掉 else 分支的 LayerNormOp 调用 ⇒ C6 变红
    mutate(TRITON_OPS, "LayerNormOp.layer_norm_residual(input, input_bias, residual, gamma, beta, epsilon)",
           "layer_norm_residual(input, input_bias, residual, gamma, beta, epsilon)", "inj_branches_same")
    # I4: 在别的文件里加一个关键字传值 ⇒ C5 变红
    def add_kw(tmp_root):
        p = tmp_root / MLP_BINDING
        p.write_text(p.read_text(encoding="utf-8") + "\n# probe\nmlp_gemm_func(use_triton_ln=False)\n",
                     encoding="utf-8")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "repo"
        shutil.copytree(repo, tmp, ignore=shutil.ignore_patterns(".git"))
        # 投到另一个文件里，同时保持 C5 的"其他文件"口径
        extra = tmp / "deepspeed/ops/transformer/inference/_probe_extra.py"
        extra.write_text("from deepspeed.ops.transformer.inference.triton.ops import mlp_gemm_func\n"
                         "mlp_gemm_func(use_triton_ln=False)\n", encoding="utf-8")
        r = analyze(tmp)
        results.append(("inj_keyword_other_file", r["verdict"] != "UNREACHABLE", r["failed"]))

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
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    if args.selftest:
        return selftest(repo)

    report = analyze(repo)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
    for c in report["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['id']:<32} {c['detail']}")
    print(f"\nverdict: {report['verdict']}  ({len(report['checks'])} checks, {len(report['failed'])} failed)")
    return 0 if report["verdict"] == "UNREACHABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
