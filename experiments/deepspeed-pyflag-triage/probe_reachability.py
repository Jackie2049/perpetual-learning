#!/usr/bin/env python3
"""可达性：`attn_nw is None` 在 `use_triton=True` 时到底能不能发生。

上一步证明了「fp16 + triton + attn_nw is None」会 TypeError。本脚本回答的是
「这条三元条件在真实代码里能否同时成立」——这是决定它该写成"用户能撞到的崩溃"
还是"结构性死路"的关键，不能靠推断。

判据来源（全部 AST，只读）：
  R1 `TritonMLP.__init__` 无条件给 attn_nw 赋 `nn.Parameter` ⇒ 在 triton MLP 上恒不为 None
  R2 `DeepSpeedMLP.__init__` 只在 `set_empty_params` 时置 None，其它情况赋 Parameter
  R3 `use_triton=True` 时 `ds_transformer` 选的是 `TritonMLP`（不是 DeepSpeedMLP）
  R4 因此「triton MLP ∧ attn_nw is None」**在同一实例上不可同时成立**
  R5 但 `GELUGemmOp` 的指针选择只看 dtype/use_triton，与 MLP 类无关 ⇒
     在 `DeepSpeedMLP + set_empty_params + dtype=fp16 + use_triton=True` 上仍然会 TypeError

    python3 probe_reachability.py --repo /tmp/tm025-deepspeed
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_gelu_argshift import load  # noqa: E402

TRITON_MLP = "deepspeed/ops/transformer/inference/triton/mlp.py"
DS_MLP = "deepspeed/ops/transformer/inference/ds_mlp.py"
DS_TRANSFORMER = "deepspeed/model_implementations/transformers/ds_transformer.py"


def _class(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def _init(cls):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")


def _assigns_attn_nw(init):
    """返回 [(行号, 右值源码, 是否在 if 块内的条件)]。"""
    out = []

    def walk(node, guard=None):
        if isinstance(node, list):
            for sub in node:
                walk(sub, guard)
            return
        if isinstance(node, ast.If):
            walk(node.body, ast.unparse(node.test) if guard is None else f"({guard}) and ({ast.unparse(node.test)})")
            walk(node.orelse, guard)
            return
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "attn_nw":
                    out.append((node.lineno, ast.unparse(node.value), guard))
            return
        for child in ast.iter_child_nodes(node):
            walk(child, guard)

    walk(init)
    return out


def analyze(repo: Path) -> dict:
    checks = []

    def check(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    tsrc = load(repo, TRITON_MLP)
    triton_assigns = _assigns_attn_nw(_init(_class(ast.parse(tsrc), "TritonMLP")))
    check("R1.triton_mlp_attn_nw_never_none",
          len(triton_assigns) == 1 and triton_assigns[0][2] is None
          and "Parameter" in triton_assigns[0][1],
          f"TritonMLP 的 attn_nw 赋值：{triton_assigns}")

    msrc = load(repo, DS_MLP)
    ds_assigns = _assigns_attn_nw(_init(_class(ast.parse(msrc), "DeepSpeedMLP")))
    none_assigns = [(l, v, g) for l, v, g in ds_assigns if v == "None"]
    has_param = any("Parameter" in v for _, v, _ in ds_assigns)
    # 收紧：None 那条必须带 `set_empty_params` 守卫，否则等于"恒为 None"
    none_guarded = bool(none_assigns) and all(g and "set_empty_params" in g for _, _, g in none_assigns)
    check("R2.ds_mlp_none_only_if_set_empty",
          none_guarded and has_param,
          f"DeepSpeedMLP 的 attn_nw 赋值：{[(l, v, g) for l, v, g in ds_assigns]}")

    s = load(repo, DS_TRANSFORMER)
    stree = ast.parse(s)
    src = ast.unparse(stree)
    # use_triton=True 分支选 TritonMLP，else 选 DeepSpeedMLP
    check("R3.triton_selects_triton_mlp",
          "if deepspeed.HAS_TRITON and self.config.use_triton" in src
          and "TritonMLP(self.config)" in src and "DeepSpeedMLP(" in src,
          "ds_transformer 里 use_triton 分支 → TritonMLP，else → DeepSpeedMLP")

    # R4: 结论——triton MLP 上 attn_nw 恒非 None
    check("R4.triton_path_unreachable", triton_assigns[0][2] is None,
          "⇒ 'triton 路径 ∧ attn_nw is None' 在同一实例上不可同时成立")

    # R5: GELUGemmOp 的指针只取决于 dtype/use_triton，与 MLP 类无关
    g = load(repo, "deepspeed/ops/transformer/inference/op_binding/gelu_gemm.py")
    gsrc = ast.unparse(ast.parse(g))
    check("R5.geluop_ignores_mlp_class",
          "HAS_TRITON and self.config.use_triton" in gsrc and "attn_nw" not in gsrc,
          "gelu_gemm.py 全文不出现 attn_nw ⇒ 指针选择与 MLP 类无关")

    # R6: 谁能造出 DeepSpeedMLP + set_empty_params + use_triton=True
    he = load(repo, "deepspeed/runtime/hybrid_engine.py")
    check("R6.hybrid_engine_sets_empty_params", "set_empty_params=True" in he,
          "hybrid_engine 的 DeepSpeedInferenceConfig 带 set_empty_params=True")
    check("R7.hybrid_engine_does_not_set_use_triton",
          "use_triton" not in he,
          "hybrid_engine 不传 use_triton（默认 False）⇒ 该组合需用户显式打开 triton")

    # R8: 决定性——`use_triton=True` 时容器层**只允许 BERT**，否则直接 raise
    base = load(repo, "deepspeed/module_inject/containers/base.py")
    bsrc = ast.unparse(ast.parse(base))
    check("R8.triton_only_bert",
          'if self.use_triton and deepspeed.HAS_TRITON' in bsrc
          and 'DS_BERTContainer' in bsrc
          and 'Triton kernels are only for BERT-like models yet' in bsrc,
          "containers/base.py：use_triton 且非 BERT ⇒ raise NotImplementedError")

    # R9: BERT 容器走的是 LayerNorm（不是 attn_nw is None 的那一支）
    bert = repo / "deepspeed/module_inject/containers/bert.py"
    bert_src = load(repo, "deepspeed/module_inject/containers/bert.py") if bert.exists() else ""
    check("R9.bert_has_layernorm",
          "layernorm" in bert_src.lower() or "LayerNorm" in bert_src,
          f"bert.py 存在={bert.exists()}，含 LayerNorm={('LayerNorm' in bert_src) or ('layernorm' in bert_src.lower())}")

    ok = all(c["ok"] for c in checks)
    return {
        "checks": checks,
        "verdict": "UNREACHABLE_IN_PRACTICE" if ok else "CHANGED",
        "summary": "TypeError 真实存在（E2E 已证），但 user-facing 路径上不可达："
                   "use_triton=True 只允许 BERT 容器（否则 NotImplementedError），"
                   "而 TritonMLP 的 attn_nw 恒为 Parameter",
        "failed": [c["id"] for c in checks if not c["ok"]],
    }


def selftest(repo: Path):
    import shutil
    import tempfile

    base = analyze(repo)
    res = [("baseline_holds", base["verdict"] == "UNREACHABLE_IN_PRACTICE", base["failed"])]

    def with_tmp(fn, label):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "r"
            shutil.copytree(repo, tmp, ignore=shutil.ignore_patterns(".git"))
            if not fn(tmp):
                res.append((label, False, "注入目标不存在"))
                return
            r = analyze(tmp)
            res.append((label, r["verdict"] != "UNREACHABLE_IN_PRACTICE", r["failed"]))

    def replace_in(rel, old, new):
        def inner(tmp):
            p = tmp / rel
            s2 = p.read_text(encoding="utf-8")
            if old not in s2:
                return False
            p.write_text(s2.replace(old, new, 1), encoding="utf-8")
            return True
        return inner

    # I1: 让 TritonMLP 的 attn_nw 也可能为 None ⇒ R1/R4 变红
    old1 = ('        self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size, '
            'dtype=data_type_fp, device=device),\n                                    requires_grad=False)')
    new1 = ('        if self.config.set_empty_params:\n            self.attn_nw = None\n'
            '        else:\n            self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size, '
            'dtype=data_type_fp, device=device),\n                                        requires_grad=False)')
    with_tmp(replace_in(TRITON_MLP, old1, new1), "inj_triton_mlp_noneable")

    # I2: 去掉 BERT 限制 ⇒ R8 变红
    with_tmp(replace_in("deepspeed/module_inject/containers/base.py",
                        'raise NotImplementedError("Triton kernels are only for BERT-like models yet")',
                        "pass"), "inj_bert_limit_removed")

    # I3: 让 DeepSpeedMLP 恒置 None ⇒ R2 的"只在 set_empty_params 时"变红
    with_tmp(replace_in(DS_MLP, "if self.config.set_empty_params:\n            self.attn_nw = None",
                        "if True:\n            self.attn_nw = None"), "inj_ds_mlp_always_none")

    print(json.dumps([{"id": a, "caught": b, "failed": c} for a, b, c in res], indent=1))
    bad = [a for a, b, _ in res if not b]
    print(f"SELFTEST {'OK' if not bad else 'FAILED'}: {len(res)-len(bad)}/{len(res)}")
    return 0 if not bad else 2


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
    for c in rep["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['id']:<34} {c['detail']}")
    print(f"\nverdict: {rep['verdict']}  ({len(rep['checks'])} checks, {len(rep['failed'])} failed)")
    print(rep["summary"])
    return 0 if rep["verdict"] != "CHANGED" else 1


if __name__ == "__main__":
    sys.exit(main())
