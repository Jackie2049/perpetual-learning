#!/usr/bin/env python3
"""扫 #6067（044bbba2，"deprecate workers, migrate to engines"）删掉的 worker 文件里，
哪些配置读取点在当前 main 的引擎侧没有消费者。

做法（静态，不跑 verl）：
1. 从 044bbba2^:verl/workers/fsdp_workers.py、megatron_workers.py 里抽出全部
   `config.<role>.<key>` 读取点（正则，行号一并保留）。
2. 对每个 (role, key) 判定当前 main 是否有消费者：
   - **数据类字段**：在 verl/workers/config/*.py 里有没有该字段的声明；
   - **引擎读取**：在 verl/workers/engine/**/*.py 里有没有读取；
   - **trainer 读取**：在 verl/trainer/**/*.py 里有没有读取（只读不建模型的情形）。
3. 输出一张表：字段是否仍在声明 → 引擎是否读 → 判定。

判定口径（写死，避免"看起来像问题"）：
- 只把"**字段仍在配置里（用户能设）但引擎侧无任何读取**"列为候选失效；
- 字段已从数据类删除的，记为「已消失」，不算失效（用户设了会被 omega 拒掉，报错可见）；
- 引擎侧无读取但 trainer 侧有读取的，记为「trainer 消费」，不算失效。

用法：
    python3 scan_consumers.py --repo <verl 仓库路径> [--parent 044bbba2] [--json out.json]

退出码：0 = 扫描完成（不论有无候选）；2 = 参数/仓库问题。
**本脚本只做文本级定位，逐条结论必须人工读代码确认后才写进文档。**
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

OLD_FILES = {
    "actor_ref": "verl/workers/fsdp_workers.py",
    "megatron": "verl/workers/megatron_workers.py",
}

# 这些 key 是"结构性"读取（get 整个 role、to_dataclass 等），不参与失效判定
SKIP_KEYS = {"get", "items", "keys", "copy", "update", "pop", "to_container"}


def git_show(repo: pathlib.Path, rev: str, path: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"读不到 {rev}:{path}\n{out.stderr}")
    return out.stdout


def extract_reads(src: str) -> dict[str, list[int]]:
    """返回 {"actor.fsdp_config": [行号...], ...}。"""
    reads: dict[str, list[int]] = {}
    for lineno, line in enumerate(src.splitlines(), 1):
        for m in re.finditer(r"\bconfig\.((?:actor|ref|critic|model|rollout|profiler)[\w\.]*)", line):
            path = m.group(1)
            parts = path.split(".")
            if any(p in SKIP_KEYS for p in parts):
                continue
            reads.setdefault(path, []).append(lineno)
    return reads


def collect_field_names(repo: pathlib.Path) -> set[str]:
    """verl/workers/config/*.py 里宣布的所有 dataclass 字段名（不带 role 前缀）。"""
    names: set[str] = set()
    for py in sorted((repo / "verl/workers/config").glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(r"^\s{4}([a-zA-Z_]\w*)\s*:\s*[^=\n]+=", text, re.MULTILINE):
            names.add(m.group(1))
    return names


def collect_engine_text(repo: pathlib.Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for py in sorted((repo / "verl/workers/engine").rglob("*.py")):
        files[str(py.relative_to(repo))] = py.read_text(encoding="utf-8", errors="replace")
    for py in sorted((repo / "verl/trainer").rglob("*.py")):
        files[str(py.relative_to(repo))] = py.read_text(encoding="utf-8", errors="replace")
    return files


def where_read(files: dict[str, str], role: str, key: str) -> list[str]:
    """在引擎/trainer 文本里找 `.<key>` 的出现（`key` 是叶子名，role 只用来说明语义）。"""
    leaf = key.split(".")[-1]
    pat = re.compile(r"\b" + re.escape(leaf) + r"\b")
    hits = []
    for path, text in files.items():
        if pat.search(text):
            hits.append(path)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="verl 仓库路径")
    ap.add_argument("--parent", default="044bbba2^", help="#6067 的父提交")
    ap.add_argument("--json", default=None, help="把结果写到这个文件")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).expanduser().resolve()
    if not (repo / ".git").exists() and not (repo / "verl").exists():
        print(f"错误：{repo} 不像 verl 仓库", file=sys.stderr)
        return 2

    fields = collect_field_names(repo)
    current = collect_engine_text(repo)
    if not current:
        print("错误：当前树里没找到 verl/workers/engine 或 verl/trainer .py", file=sys.stderr)
        return 2

    rows = []
    for label, old_path in OLD_FILES.items():
        src = git_show(repo, args.parent, old_path)
        for key, lines in sorted(extract_reads(src).items()):
            leaf = key.split(".")[-1]
            still_declared = leaf in fields
            hits = where_read(current, key.split(".")[0], key)
            if not still_declared:
                verdict = "字段已消失"
            elif hits:
                verdict = "有读取"
            else:
                verdict = "候选失效（字段在、引擎/trainer 不读）"
            rows.append(
                {
                    "source": label,
                    "key": key,
                    "old_lines": lines,
                    "leaf_still_declared": still_declared,
                    "current_readers": hits,
                    "verdict": verdict,
                }
            )

    cand = [r for r in rows if r["verdict"].startswith("候选失效")]
    summary = {
        "repo": str(repo),
        "parent": args.parent,
        "n_reads": len(rows),
        "n_candidates": len(cand),
        "candidates": [f"{r['source']}:{r['key']}" for r in cand],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
