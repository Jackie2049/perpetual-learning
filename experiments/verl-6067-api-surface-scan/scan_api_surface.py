#!/usr/bin/env python3
"""扫 #6067（044bbba2，"deprecate workers, migrate to engines"）删掉的**函数/类接口**与
**文档引用**，看它们在当前 main 上还有没有对应物。

与同目录族的另一份 `verl-6067-config-consumer-scan/scan_consumers.py` 分工：
那份查"配置读取点"（用户能设、但没人读）；这份查"**代码/文档还在提一个已经不存在的名字**"。

做法（纯静态，不跑 verl）：
1. 从 `044bbba2^` 里被整文件删除的生产模块抽 **公开方法名** 与 **公开函数名**；
2. 在当前 main 的 `verl/**` 里查同名定义（`def <name>(`）；
3. 当前 main 的 `docs/**` 里查同名**引用**，并把"引用所在文件"分成
   - `historical`：路径含 `blog/` 或 `release` —— 发布博客/变更日志写"当年有的东西"，
     引用已删接口是**正确**的，不参与判定；
   - `active`：其余文档 —— 用户照着做会走空，参与判定。

判定口径（写死，避免"看起来像问题"）：
- `still_defined`：main 里有同名定义 ⇒ 接口没消失，不算问题（`__init__` 这种跨文件重名的
  一律按"有定义"处理，宁可漏报）。
- `gone_no_doc`：main 无定义、且没有 active 文档引用 ⇒ 正常删除，不报。
- `gone_referenced`：main 无定义、**但 active 文档还在引用** ⇒ 候选（用户照文档走会失败）。
- `gone_doc_historical_only`：只被 blog/发布说明引用 ⇒ 不算候选。

用法：
    python3 scan_api_surface.py --repo <verl 仓库路径> [--parent 044bbba2] [--json out.json]

退出码：0 = 扫描完成；2 = 参数/仓库问题。
**本脚本只做文本级定位，逐条结论必须人工读代码/文档确认后才写进文档。**
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

# 044bbba2 里被整文件删除的生产模块（tests/ 与 docs 另算）
DELETED_MODULES = [
    "verl/workers/actor/base.py",
    "verl/workers/actor/dp_actor.py",
    "verl/workers/actor/megatron_actor.py",
    "verl/workers/critic/base.py",
    "verl/workers/critic/dp_critic.py",
    "verl/workers/critic/megatron_critic.py",
    "verl/workers/fsdp_workers.py",
    "verl/workers/megatron_workers.py",
    "verl/workers/sharding_manager/base.py",
    "verl/workers/sharding_manager/fsdp_ulysses.py",
]

# 判定时忽略的名字：协议样板，跨文件重名率高，留着只会制造噪声
IGNORED_NAMES = {"__init__", "__enter__", "__exit__", "forward", "training_step", "validation_step"}

# 这些前缀的文档视为历史材料（发布博客、变更日志），引用已删接口是对的
HISTORICAL_DOC_PREFIXES = ("docs/blog/",)


def git(repo: pathlib.Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败\n{out.stderr}")
    return out.stdout


def extract_api(src: str) -> dict[str, list[int]]:
    """抽出「公开方法名」与「模块级公开函数名」，返回 {name: [行号...]}。"""
    names: dict[str, list[int]] = {}
    for lineno, line in enumerate(src.splitlines(), 1):
        m = re.match(r"^(\s*)def ([A-Za-z_]\w*)\(", line)
        if not m:
            continue
        indent, name = m.group(1), m.group(2)
        # 只收方法（4 空格缩进）与模块级函数（0 缩进）
        if indent not in ("", "    "):
            continue
        if name.startswith("_") or name in IGNORED_NAMES:
            continue
        names.setdefault(name, []).append(lineno)
    return names


def current_definitions(repo: pathlib.Path) -> dict[str, list[str]]:
    """当前 main 的 verl/** 里每个名字被哪些文件 `def` 过。"""
    defs: dict[str, list[str]] = {}
    for py in sorted((repo / "verl").rglob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        for name in set(re.findall(r"^\s*def ([A-Za-z_]\w*)\(", text, re.M)):
            defs.setdefault(name, []).append(str(py.relative_to(repo)))
    return defs


def doc_references(repo: pathlib.Path) -> dict[str, list[str]]:
    """当前 main 的 docs/** 里每个名字出现在哪些文件（`name(` 或 `` `name` `` 两种写法）。"""
    refs: dict[str, list[str]] = {}
    docs = repo / "docs"
    if not docs.exists():
        return refs
    for doc in sorted(docs.rglob("*")):
        if doc.suffix not in (".md", ".rst", ".txt"):
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        rel = str(doc.relative_to(repo))
        for name in set(re.findall(r"[A-Za-z_]\w*", text)):
            if re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", text) or f"`{name}`" in text:
                refs.setdefault(name, []).append(rel)
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="verl 仓库路径")
    ap.add_argument("--parent", default="044bbba2^", help="#6067 的父提交")
    ap.add_argument("--json", default=None, help="把结果写到这个文件")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).expanduser().resolve()
    if not (repo / "verl").is_dir():
        print(f"错误：{repo} 不像 verl 仓库", file=sys.stderr)
        return 2

    defs = current_definitions(repo)
    docs = doc_references(repo)

    rows = []
    for path in DELETED_MODULES:
        src = git(repo, "show", f"{args.parent}:{path}")
        for name, lines in sorted(extract_api(src).items()):
            still = defs.get(name, [])
            doc_hits = docs.get(name, [])
            active = [d for d in doc_hits if not d.startswith(HISTORICAL_DOC_PREFIXES)]
            historical = [d for d in doc_hits if d.startswith(HISTORICAL_DOC_PREFIXES)]
            if still:
                verdict = "still_defined"
            elif active:
                verdict = "gone_referenced"
            elif historical:
                verdict = "gone_doc_historical_only"
            else:
                verdict = "gone_no_doc"
            rows.append(
                {
                    "module": path,
                    "name": name,
                    "old_lines": lines,
                    "current_definitions": still,
                    "doc_references_active": active,
                    "doc_references_historical": historical,
                    "verdict": verdict,
                }
            )

    cand = [r for r in rows if r["verdict"] == "gone_referenced"]
    summary = {
        "repo": str(repo),
        "parent": args.parent,
        "n_names": len(rows),
        "n_still_defined": sum(1 for r in rows if r["verdict"] == "still_defined"),
        "n_gone_referenced": len(cand),
        "n_gone_doc_historical_only": sum(1 for r in rows if r["verdict"] == "gone_doc_historical_only"),
        "n_gone_no_doc": sum(1 for r in rows if r["verdict"] == "gone_no_doc"),
        "candidates": sorted({r["name"] for r in cand}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
