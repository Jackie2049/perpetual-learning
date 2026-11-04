#!/usr/bin/env python3
"""把 #6067 删除的 worker 接口/文档引用，逐条钉成"真断 / 假阳性 / 需真机"的判定表。

`scan_api_surface.py` 只做文本级定位（50 个名字 → 13 条"main 无定义但 active 文档还在引用"）。
本脚本用**逐条写死的判据**复核这些候选，并额外收进几条扫描器**看不见**、靠人读代码才发现的项。

每条给出：
  - 旧代码在哪个文件/行；
  - 当前 main 的证据（正则 / `main_uses` 断言 / "无"）；
  - 结论：`gone_but_doc_only`（代码已无此名，文档还在用——按文档走会走空）
           / `renamed`（换了名字照样生效，文档只是用词旧）
           / `doc_drift`（代码在、文档的映射/参数名过时）
           / `frontend_intact`（前端照常触发，但后端这一环被删）
           / `needs_hardware`（静态只能说到这里，读数要 NPU + msprobe）

判据说硬就硬：
  - `gone_*` 类必须同时满足「main 的 verl/** 无该名字的 def」+「active 文档里有引用」；
  - `doc_drift` 类必须给出「文档说的名字」与「main 里的实际名字」两侧的证据；
  - `main_uses` 是负向自检的落点：往 main 种一个同名 def，期望结论**必须**翻转。

用法：
    python3 verify_findings.py --repo <verl 仓库>
    python3 verify_findings.py --selftest        # 在临时副本里注入，期望判定翻转

退出码：0 = 全部判定与声明一致；1 = 有判定对不上（main 变了，结论要重核）；2 = 参数问题。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# 期望结论 → 判据类型
#   ("gone", N)             : main 的 verl/** 里该名字的 def 数 == N
#   ("doc", regex, N)       : 指定文件里 regex 命中数 >= N
#   ("main_uses", N)        : 指定文件里 `name(` 调用出现数 >= N
CASES = [
    dict(
        name="update_policy",
        old="verl/workers/actor/{base,dp_actor,megatron_actor}.py（base:54、dp:513、megatron:769）",
        expect="gone_but_doc_only",
        gone=0,
        docs=[
            ("docs/perf/perf_tuning.rst", r"bwd of ``update_policy`` and ``update_critic``"),
            ("docs/advance/dpo_extension.rst", r"``update_policy`` in the actor_rollout model"),
        ],
        note="代码侧统一改成 `update_actor`（engine_workers.py:656），文档仍写 actor 的方法名",
    ),
    dict(
        name="update_critic",
        old="verl/workers/critic/{base,dp_critic,megatron_critic}.py + fsdp_workers.py:1698",
        expect="needs_hardware",
        gone=0,
        docs=[
            ("docs/workers/ray_trainer.rst", r"self\.critic_wg\.update_critic\(batch\)"),
            ("docs/perf/perf_tuning.rst", r"bwd of ``update_policy`` and ``update_critic``"),
        ],
        note="两件事：① `update_actor` 有同名 RPC 但 `update_critic` 没有，"
             "trainer 改成 `critic_wg.train_mini_batch`（ray_trainer.py:1359）；"
             "② 更要紧的是 profiler role：旧 worker 的 `@DistProfiler.annotate(role=\"critic_update\")`"
             "随文件删除，engine_workers 只有 actor_update / actor_compute_log_prob /"
             "ref_compute_log_prob 三个 role ⇒ 文档里 critic_update 这类 stage 在前端**仍被接受**"
             f"（_STAGE_TO_ROLE 里还在，precision_debugger_profile.py:31），但没有任何装饰器会触发它。"
             "实际有没有 dump 要 NPU + msprobe 才能读",
    ),
    dict(
        name="compute_values",
        old="verl/workers/critic/{base,dp_critic,megatron_critic}.py + fsdp_workers.py:1678"
            "（旧 worker 的 RPC，带 role=\"compute_values\" 装饰器）",
        expect="needs_hardware",
        gone=0,
        docs=[
            ("docs/ascend_tutorial/dev_guide/precision_analysis/precision_debugger_zh.md", r"- compute_values$"),
            ("docs/perf/perf_tuning.rst", r"fwd computation of ``compute_values``"),
            ("docs/workers/ray_trainer.rst", r"self\.critic_wg\.compute_values\(batch\)"),
        ],
        note="同上：值推理改成 `critic_wg.infer_batch`（ray_trainer.py:1221），doc 里的 stage 名"
             "仍在 _STAGE_TO_ROLE 里但无装饰器；critic 的**前端 stage 名**在 `_STAGE_TO_ROLE` 里留着是"
             "有意为之的话需要 maintainer 判断",
    ),
    dict(
        name="critic_update",
        old="verl/workers/fsdp_workers.py:1697 `@DistProfiler.annotate(role=\"critic_update\")`；"
            "megatron_workers.py 的 update_critic 同",
        expect="drift_check_decorator",
        gone=None,
        docs=[
            ("docs/ascend_tutorial/dev_guide/precision_analysis/precision_debugger_zh.md", r"`critic_update`"),
        ],
        note="这条扫描器看不见（`critic_update` 从来不是一个函数名，只是一个装饰器 role）。"
             "判据：main 的 verl/** 里 `role=\"critic_update\"` / `role=\"compute_values\"` 出现次数必须为 0",
    ),
    dict(
        name="workers/fsdp_workers.py#L120-L122",
        old="docs/perf/perf_tuning.rst 里指向 `fsdp_workers.py#L120-L122` 的两个行号锚点",
        expect="renamed",
        gone=None,
        docs=[("docs/perf/perf_tuning.rst", r"engine_workers\.py>`_")],
        note="#6067 把链接整体替换成 `verl/workers/engine_workers.py`（无行号）。"
             "旧锚点指向的是 `get_sharding_strategy()`，而 #6067 的说明写的是'batch size normalization'"
             "——改指向新文件后，那个文件里其实没有 normaliz* / 整除逻辑（`git grep ppo_mini_batch_size` 在 "
             "engine_workers.py 只有 :511 一处赋值）。文档本身没坏（链接可达），坏的是它点名的内容",
    ),
    dict(
        name="critic-infer-token-cap",
        old="docs/perf/perf_tuning.rst:140-141 —— 告诉用户 values 前向的 token 上限由 "
            "`critic.forward_micro_batch_size_per_gpu` 控制",
        expect="doc_drift",
        gone=None,
        docs=[("docs/perf/perf_tuning.rst", r"``critic\.forward_micro_batch_size_per_gpu``")],
        main_uses=("verl/trainer/config/critic/critic.yaml", r"^forward_max_token_len_per_gpu:"),
        note="文档把 token 上限挂在 `forward_micro_batch_size_per_gpu`（个数）上；main 的 critic 配置里"
             "另有 `forward_max_token_len_per_gpu`（critic.yaml:49）才是 token 上限，而文档全文一次都没提它。"
             "属存量文档漂移，**不是 #6067 造成的**（#6067 只改了 perf_tuning 的两处链接）",
    ),
    dict(
        name="mkdocs/ray_trainer.rst-critic-doc",
        old="docs/workers/ray_trainer.rst 的 `critic_wg.compute_values/update_critic` 调用示例",
        expect="doc_drift",
        gone=None,
        docs=[("docs/workers/ray_trainer.rst", r"self\.critic_wg\.(compute_values|update_critic)\(")],
        main_uses=("verl/trainer/ppo/ray_trainer.py", r"self\.critic_wg\.(infer_batch|train_mini_batch)\("),
        note="ray_trainer.rst 展示了 critic 的调用序列；main 里实际是 infer_batch / train_mini_batch。"
             "这份文档 #6067 根本没碰（不是回归，是**重构后没跟上的存量文档**）",
    ),
]


def grep_count(root: pathlib.Path, pattern: str, label: str) -> int:
    """返回 pattern（多行）在文件里的命中数。"""
    path = root / label
    if not path.is_file():
        return -1
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(pattern, text, re.MULTILINE))


def def_count(root: pathlib.Path, name: str) -> int:
    n = 0
    for py in (root / "verl").rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        n += len(re.findall(rf"^\s*def {re.escape(name)}\(", text, re.MULTILINE))
    return n


def check_case(root: pathlib.Path, case: dict) -> dict:
    name = case["name"]
    detail: dict = {}
    expect = case["expect"]

    if expect in ("gone_but_doc_only",):
        n = def_count(root, name)
        doc_counts = {label: grep_count(root, pat, label) for label, pat in case["docs"]}
        detail = {"defs": n, "docs": doc_counts}
        ok = n == case["gone"] and all(c > 0 for c in doc_counts.values())

    elif expect in ("needs_hardware", "renamed"):
        n = def_count(root, name) if case.get("gone") is not None else None
        doc_counts = {label: grep_count(root, pat, label) for label, pat in case["docs"]}
        detail = {"defs": n, "docs": doc_counts}
        ok = (n == case["gone"]) if case.get("gone") is not None else True
        ok = ok and all(c > 0 for c in doc_counts.values())

    elif expect in ("doc_drift",):
        doc_counts = {label: grep_count(root, pat, label) for label, pat in case["docs"]}
        label, pat = case["main_uses"]
        uses = grep_count(root, pat, label)
        detail = {"docs": doc_counts, "main_uses": {label: uses}}
        ok = all(c > 0 for c in doc_counts.values()) and uses > 0

    elif expect == "drift_check_decorator":
        missing = 0
        for pat in (r'role="critic_update"', r'role="compute_values"'):
            missing += sum(
                len(re.findall(pat, p.read_text(encoding="utf-8", errors="replace")))
                for p in (root / "verl").rglob("*.py")
            )
        detail = {"annotate_role_hits": missing}
        ok = missing == 0

    else:  # pragma: no cover - 防拼写错误
        raise SystemExit(f"未知 expect: {expect}")

    return {"name": name, "expect": expect, "ok": ok, "detail": detail, "note": case["note"],
            "old": case["old"]}


def run(repo: pathlib.Path) -> dict:
    rows = [check_case(repo, c) for c in CASES]
    mism = [r["name"] for r in rows if not r["ok"]]
    return {"n": len(rows), "n_mismatch": len(mism), "mismatch": mism, "rows": rows}


def selftest(repo: pathlib.Path) -> int:
    """负向对照：往 main 种一个与候选同名的 def/装饰器，被检项必须翻转。"""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="verl-6067-api-selftest-"))
    try:
        # 只拷需要的最小树
        for sub in ("verl", "docs"):
            shutil.copytree(repo / sub, tmp / sub, ignore=shutil.ignore_patterns("__pycache__"))
        # 保证依赖 git 的部分不参与（本脚本不需要 git）
        planted = tmp / "verl" / "_selftest_planted.py"
        planted.write_text(
            "def update_policy(self):\n    pass\n\n"
            "def update_critic(self):\n    pass\n\n"
            "def compute_values(self):\n    pass\n\n"
            "# role=\"critic_update\"\n# role=\"compute_values\"\n",
            encoding="utf-8",
        )
        after = run(tmp)
        still_ok = [r["name"] for r in after["rows"] if r["ok"] and r["name"] in
                    ("update_policy", "update_critic", "compute_values", "critic_update")]
        print(f"注入后仍然 ok 的用例：{still_ok}")
        # update_policy 用 `gone_but_doc_only`，def 计数一变就应该翻
        flipped = {r["name"]: r["ok"] for r in after["rows"]}
        expect_flip = ["update_policy", "update_critic", "compute_values", "critic_update"]
        bad = [n for n in expect_flip if flipped.get(n, True)]
        if bad:
            print(f"自检失败：注入 {bad} 的同名定义后，这些用例仍报 ok（判据没读到定义）")
            return 1
        print("自检通过：注入同名定义后四条用例全部翻转")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="verl 仓库路径")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        if not args.repo:
            print("--selftest 需要 --repo", file=sys.stderr)
            return 2
        return selftest(pathlib.Path(args.repo).expanduser().resolve())

    if not args.repo:
        print("需要 --repo", file=sys.stderr)
        return 2
    repo = pathlib.Path(args.repo).expanduser().resolve()
    result = run(repo)
    print(json.dumps({k: result[k] for k in ("n", "n_mismatch", "mismatch")}, ensure_ascii=False, indent=2))
    for r in result["rows"]:
        print(f"  [{'ok' if r['ok'] else 'MISMATCH'}] {r['name']:<28} expect={r['expect']:<20} {r['detail']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["n_mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
