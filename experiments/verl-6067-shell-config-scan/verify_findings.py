#!/usr/bin/env python3
"""逐条核对 `scan_shell_refs.py` 的四条判定 + 三条结构性自证。

判据一律**写死在这里**（不在被测脚本里），跑一遍必须 0 不一致 exit 0。
`--selftest` 往仓库副本里种三种扰动，验证判据会翻转；任一不翻转即 exit 1。

判定与口径（对外材料只能引用这几条）：
- `COMPOSE_FAIL`：`compose()` 阶段就抛 `ConfigCompositionException`。已用真实 Hydra 1.3.2
  在两个最小树上复现过（见 readme「真实 Hydra 复现」），不是机制推断。
- `SILENT_OK`：加了 `+` 前缀，键被接受、**不报错**。这一档必须写清"不报错 ≠ 生效"。
- `DOC_TABLE_CELL`：文档表格里仍把该键当作可配参数列出。
- `UNKNOWN`：链条没接上，不得写成"没有该键"。

用法：
    python3 verify_findings.py --repo <verl 仓库> [--json findings.json]
    python3 verify_findings.py --repo <verl 仓库> --selftest
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scan_shell_refs as S  # noqa: E402

KEY = "trainer.use_legacy_worker_impl"

# 写死的判定表：文件:行 → (kind, consumed_by)
EXPECTED = {
    ("docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md", 368): ("DOC_TABLE_CELL", None),
    ("examples/grpo_trainer/run_qwen3_30b_veomni.sh", 102): ("INVOKE", "SILENT_OK"),
    ("tests/special_e2e/run_fully_async_policy_genrm.sh", 177): ("INVOKE", "COMPOSE_FAIL"),
    ("tests/special_e2e/run_fully_async_policy_opd.sh", 240): ("INVOKE", "SILENT_OK"),
}

# 结构性自证：这些事实必须同时成立，否则上面四条判定的含义就不一样
STRUCTURAL = {
    "removed_key_absent_in_ppo_trainer": lambda repo: KEY not in S.yaml_leaf_paths(
        (repo / "verl/trainer/config/ppo_trainer.yaml").read_text(encoding="utf-8")
    ),
    "removed_key_absent_in_generated_config": lambda repo: KEY not in S.yaml_leaf_paths(
        (repo / "verl/trainer/config/_generated_ppo_trainer.yaml").read_text(encoding="utf-8")
    ),
    "opd_script_is_in_ci_paths": lambda repo: "run_fully_async_policy_opd.sh" in (
        repo / ".github/workflows/e2e_fully_async_policy_opd.yml"
    ).read_text(encoding="utf-8"),
}


def collect(repo: pathlib.Path) -> dict:
    hits = S.scan_refs(repo, KEY)
    log: list[str] = []
    for h in hits:
        if h["kind"] != "INVOKE":
            continue
        ch = S.config_chain(repo, h["pkg_entry"], log)
        root = KEY.split(".", 1)[0]
        prefix = "+" in h["text"]
        if ch is None or root not in ch["root_keys"]:
            h["consumed_by"] = "UNKNOWN"
        elif ch["has_record_root"] and not prefix:
            h["consumed_by"] = "COMPOSE_FAIL"
        elif ch["has_record_root"] and prefix:
            h["consumed_by"] = "SILENT_OK"
        else:
            h["consumed_by"] = "FREE"
    return {"hits": hits, "log": log}


def check(repo: pathlib.Path) -> list[str]:
    problems: list[str] = []
    got = {(h["file"], h["line"]): (h["kind"], h.get("consumed_by")) for h in collect(repo)["hits"]}
    for where, want in EXPECTED.items():
        if where not in got:
            problems.append(f"缺：{where[0]}:{where[1]} 没有命中")
        elif got[where] != want:
            problems.append(f"不一致：{where[0]}:{where[1]} 期望 {want}，实得 {got[where]}")
    for where in got:
        if where not in EXPECTED:
            problems.append(f"多出：{where[0]}:{where[1]} → {got[where]}（未登记，先判断是不是新增引用）")
    for name, fn in STRUCTURAL.items():
        try:
            ok = fn(repo)
        except FileNotFoundError as exc:
            ok, problem = False, f"读不到文件：{exc.filename}"
            problems.append(f"自证 {name}：{problem}")
            continue
        if not ok:
            problems.append(f"自证 {name} 不成立")
    return problems


def selftest() -> int:
    """四种扰动，每种都必须让判定翻转，否则判据是"怎么写都过"的。

    ② 的落点是**当前树**的 ppo_trainer.yaml：`removed_keys()` 读的是两个历史提交，
    本脚本改不了它们；而"今天把键加回来"这件事，等价于让扫描报空集的判据是
    "当前树里没有这个键"，所以用它来验证该判据不会永远报同一个答案。
    """
    src = pathlib.Path(__file__).resolve().parent.parent.parent / "verl"
    if not src.is_dir():
        print(f"找不到 verl 仓库：{src}", file=sys.stderr)
        return 2
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        def clone() -> pathlib.Path:
            dst = (pathlib.Path(tmp) / "verl").resolve()
            if dst.exists():
                shutil.rmtree(dst)
            subprocess.run(["git", "-C", str(src), "worktree", "list"], capture_output=True)
            subprocess.run(["cp", "-R", str(src), str(dst)], check=True)
            return dst

        # ① 把 opd 那行去掉 `+` ⇒ COMPOSE_FAIL（不报错的档要能翻成会报错的档）
        repo = clone()
        p = repo / "tests/special_e2e/run_fully_async_policy_opd.sh"
        text = p.read_text(encoding="utf-8").replace("+trainer.use_legacy_worker_impl=disable", "trainer.use_legacy_worker_impl=disable")
        p.write_text(text, encoding="utf-8")
        after = {(h["file"], h["line"]): h.get("consumed_by") for h in collect(repo)["hits"]}
        if after.get(("tests/special_e2e/run_fully_async_policy_opd.sh", 240)) != "COMPOSE_FAIL":
            failures.append("扰动①（去掉 +）后 opd 仍不是 COMPOSE_FAIL")

        # ② 把键写回 ppo_trainer.yaml ⇒ 当前树的键空间判据必须能读到它
        repo = clone()
        y = repo / "verl/trainer/config/ppo_trainer.yaml"
        original = y.read_text(encoding="utf-8")
        patched = original.replace("device: cuda\n", "device: cuda\n  use_legacy_worker_impl: disable\n", 1)
        y.write_text(patched, encoding="utf-8")
        # 自检的前提：替换真的落到了文件里（replace 没命中时 write_text 照样成功）
        if patched == original:
            failures.append("扰动②注入没落到 ppo_trainer.yaml（替换锚点已变）")
        if patched != original and KEY not in S.yaml_leaf_paths(y.read_text(encoding="utf-8")):
            failures.append("扰动②（键写回）后当前树里仍读不到该键")

        # ③ 把文档表格行改成注释 ⇒ DOC_TABLE_CELL 消失（判据不是"见到文件名就算"）
        repo = clone()
        d = repo / "docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md"
        lines = d.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[367] = "<!-- " + lines[367].rstrip("\n") + " -->\n"
        d.write_text("".join(lines), encoding="utf-8")
        kinds = {(h["file"], h["line"]): h["kind"] for h in collect(repo)["hits"]}
        if kinds.get(("docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md", 368)) == "DOC_TABLE_CELL":
            failures.append("扰动③（表格行改注释）后仍判 DOC_TABLE_CELL")

    for f in failures:
        print("SELFTEST FAIL", f)
    if not failures:
        print("自检通过：3 个扰动全部被判定捕捉")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    repo = pathlib.Path(args.repo).resolve()
    problems = check(repo)
    result = {"key": KEY, "n_problems": len(problems), "problems": problems,
              "hits": collect(repo)["hits"]}
    for p in problems:
        print("MISMATCH", p)
    print(f"\n判定 {len(EXPECTED)} 条 / 自证 {len(STRUCTURAL)} 条：{len(problems)} 处不一致")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
