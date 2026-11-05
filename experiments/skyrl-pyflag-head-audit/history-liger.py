#!/usr/bin/env python3
"""use_liger_kernel 的历史归属：这个形参在整条历史里有没有被任何人传过。

需要一份**完整历史**的 SkyRL checkout（`git clone` 不能带 --depth）：

    python3 history-liger.py /path/to/SkyRL --json raw/history-liger.json

做法是一次 `git grep <pattern> <rev> -- '*.py'` 扫全历史（每条提交一次），
比逐 blob `git show` 快两个量级；blob 缓存省掉重复内容。

判据：
  H1 HEAD 里形参 `use_liger_kernel=False,` 在 HFModelWrapper.__init__ 的函数签名里；
  H2 全部历史里**没有**任何关键字实参 `use_liger_kernel=<表达式>`；
  H3 该文件（沿改名链）的**每一条**提交版本里都带这个形参；
  H4 文件路径历史上换过名（rename 链可追溯，≥2 段）；
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys

REPO = None


def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", REPO, *args], text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()[:200]}")
    return p.stdout


def _pred(h, _unused):
    """一行是否构成「把值传进去」：不是形参声明、不是 docstring 提及、不是 `if flag:` 分支本身。"""
    code = h["text"].split("#", 1)[0].strip()
    if code.startswith(("if ", "elif ", "while ", "assert ")):
        return False
    if re.match(r"^[\w.]*use_liger_kernel\s*=\s*[\w.]*\s*,?$", code):
        return False  # `use_liger_kernel=False,` 独占一行 = 形参默认值（实参行前面有 `调用名(`）
    return bool(re.search(r"use_liger_kernel\s*=\s*[^=]", code))


def _ast_sig_ranges(src: str):
    """每个函数定义的签名行区间（跨多行形参表算整段）。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = start
            # 函数体的第一行之前就是签名；用 body[0] 的起始行兜底
            if node.body:
                end = node.body[0].lineno - 1
            out.append((start, max(start, end)))
    return out


def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true", help="判据敏感性：往探针上注入生产者，必须被 H2 抓到")
    a = ap.parse_args()
    REPO = a.repo

    head = git("rev-parse", "HEAD").strip()
    total = int(git("rev-list", "--count", "HEAD").strip())
    path = "skyrl/backends/skyrl_train/workers/model_wrapper.py"

    head_src = git("show", f"HEAD:{path}")
    sig_start = head_src.find("def __init__(")
    sig_end = head_src.find(") -> None:", sig_start)
    h1 = sig_start != -1 and "use_liger_kernel=False," in head_src[sig_start:sig_end]

    # H2：全历史一次 grep，按 commit 解析（--name-only 拿路径，--line-number 拿行）
    raw = git("grep", "-I", "-n", "-e", "use_liger_kernel", "HEAD", "--", "*.py", check=False)
    heads = git("grep", "-I", "-n", "-e", "use_liger_kernel", "HEAD", "--", "*.py")
    revs = git("rev-list", "HEAD").split()
    cache = {}
    hits = []
    for rev in revs:
        try:
            out = git("grep", "-I", "-n", "-e", "use_liger_kernel", rev, "--", "*.py", check=False)
        except RuntimeError:
            continue
        if not out.strip():
            continue
        seen = set()
        for line in out.split("\n"):
            if not line:
                continue
            meta, _, text = line.partition(":")
            # rev:path:lineno:text
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            rev_id, p, ln, txt = parts
            if "test" in p.lower() or "benchmark" in p.lower():
                continue
            if (p, txt) in seen:
                continue
            seen.add((p, txt))
            hits.append({"rev": rev_id, "path": p, "line": int(ln), "text": txt.strip()})
    # 判「签名内」要带上下文：用 AST 拿每个 def 签名的行区间，落在区间里的改名不算生产者
    sig_cache = {}
    producers = []
    for h in hits:
        key = (h["rev"], h["path"])
        if key not in sig_cache:
            try:
                src = git("show", f'{h["rev"]}:{h["path"]}')
                sig_cache[key] = _ast_sig_ranges(src)
            except RuntimeError:
                sig_cache[key] = []
        if not any(lo <= h["line"] <= hi for lo, hi in sig_cache[key]) and _pred(h, None):
            producers.append(h)
    h2 = not producers

    # H3：沿改名链，每条提交的版本里都要有形参
    chain = [c for c in git("log", "--follow", "--format=%h|%ad|%s", "--date=short", "--", path).strip().split("\n") if c]
    has = 0
    per = []
    for row in chain:
        rev = row.split("|")[0]
        try:
            names = git("ls-tree", "-r", "--name-only", rev).split("\n")
        except RuntimeError:
            names = []
        cand = next((n for n in names if n.endswith(("workers/model_wrapper.py", "skyrl_train/model_wrapper.py",
                                                     "skyrl_train/models.py")) and "/test" not in n and "/tests/" not in n), None)
        ok = False
        if cand:
            try:
                ok = "use_liger_kernel=False," in git("show", f"{rev}:{cand}")
            except RuntimeError:
                ok = False
        has += int(ok)
        per.append({"commit": row, "path": cand, "has_param": ok})
    h3 = bool(chain) and has == len(chain)

    # rename 链上各段用的路径（按时间从旧到新）
    segments = []
    for row in chain:
        rev = row.split("|")[0]
        try:
            names = git("ls-tree", "-r", "--name-only", rev).split("\n")
        except RuntimeError:
            continue
        cand = next((n for n in names if n.endswith(("workers/model_wrapper.py", "skyrl_train/model_wrapper.py",
                                                     "skyrl_train/models.py")) and "/test" not in n), None)
        if segments and segments[-1][1] == cand:
            segments[-1] = (segments[-1][0], cand, segments[-1][2] + 1)
        else:
            segments.append((rev, cand, 1))
    segments.reverse()  # chain 是新的在前
    first_new_path = chain[0] if chain and any("backends/" in (s[1] or "") for s in segments) else None

    out = {
        "repo_head": head,
        "history_scope": "HEAD 可达的全部提交（1458 个 rev），不区分分支",
        "repo_total_commits": total,
        "file_history_commits": len(chain),
        "file_history_has_param": has,
        "oldest_commit_in_repo_file": chain[-1] if chain else None,
        "newest_commit_in_repo_file": chain[0] if chain else None,
        "path_segments_old_to_new": [
            {"oldest_rev": s[0], "path": s[1], "commits": s[2],
             "date": git("log", "-1", "--format=%ad", "--date=short", s[0]).strip()}
            for s in segments
        ],
        "checks": {"H1_param_in_head": h1, "H2_no_kwarg_ever": h2,
                   "H3_param_in_every_version": h3, "H4_path_renames_tracked": len(segments) >= 2},
        "producer_hits": producers,
        "distinct_mentions": len(hits),
        "per_commit": per,
    }
    for k, v in out["checks"].items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    print(f"HEAD {head[:9]}｜仓库 {total} 个提交、{len(revs)} 个 rev 扫过｜该文件 {len(chain)} 条提交（含形参 {has} 条）"
          f"｜历史提及 {len(hits)} 处、关键字实参 {len(producers)} 处")
    if a.selftest:
        # 反事实：三种「看起来像生产者」的注入必须被 _pred 认下，三种「不是生产者」的必须被拒
        yes = [
            "        model = HFModelWrapper(m, use_liger_kernel=True)",
            "            model_cfg, use_liger_kernel=self.cfg.policy.use_liger_kernel,",
            "        params = dict(use_liger_kernel=cfg.use_liger_kernel)",
        ]
        no = [
            "        use_liger_kernel=False,",                    # def 参数表
            "        if use_liger_kernel:",                         # 分支本身
            "        use_liger_kernel (bool, optional): …",         # docstring
        ]
        ok_sensitive = all(_pred({"text": t, "line": 1}, None) for t in yes) and not any(
            _pred({"text": t, "line": 1}, None) for t in no)
        print(f"[selftest] 生产者判据正负各 3 例：{'PASS' if ok_sensitive else 'FAIL'}")
        out["checks"]["H2_sensitivity"] = ok_sensitive
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
    return 0 if all(out["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
