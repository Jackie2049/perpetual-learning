#!/usr/bin/env python3
"""SGLang pyflag 第二批：把 34 条 `silently_inert` **自动分桶**，只把源码层候选交给人工取证。

为什么先分桶
------------
第一批（TM-013，`fused_moe_router_shim`）是按 reads 排序逐条人眼看出来的。
SGLang 报 34 条、reads 最高 236，靠人眼排优先级会先撞上一堆**结构性不可达**的条目：

  * **内核启动器**（`srt/kernels/**`、`srt/layers/attention/**`）：形参是 Triton/cute
    核的 `constexpr`，由 `@triton.jit` 编译期专用化，走的是字典或 kwargs 展开。
    TM-013 是**特例**——那个形参用普通 `if` 分支两条路径，所以真的会分叉。
  * **配置 dataclass**（`srt/configs/**`）：形参记录进 `self.*`，真正的消费者是模型。
  * **`__init__` 里按形参分支**：`*_ENABLED` 一类的模块级常量（底下一层常是
    `SGLANG_*` 环境变量）或 `**kwargs` 才是真正的生产者。这些不是用户可配的前端选项 ⇒
    即使零生产者，对外也只是"内部开关被固化了"，不是用户缺陷。**注意这是线索不是结论**：
    `use_dsa` 同样是 `__init__` 里的形参，但 `DSATokenToKVPool` 内部真的传了 `use_dsa=True`。

本脚本只做**第一层分桶**（可复核、不需要读懂内核），产出 `triage-buckets.json`；
每条候选的最终判定在 `findings.json` / readme 里人工给出，并逐条绑到可 grep 的事实。

用法
----
    python3 build_triage.py <sglang-snapshot> --pyflag-json <scan-sglang-main.json> \
        [--json out.json]

退出码
------
0 = 分桶完成
1 = 输入对不上（pyflag 报告里的 consumer 文件在快照里找不到 / 计数与预期不符）
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import pathlib
import sys

# 内核实现目录：形参在这两个目录里，几乎都是编译期专用化 + 字典/kwargs 传参
KERNEL_DIRS = (
    "/srt/kernels/",
    "/srt/layers/attention/",
    "/batch_invariant_ops/",
)

EXPECTED_INERT = 34


def is_kernel_path(rel: str) -> bool:
    return any(d in rel for d in KERNEL_DIRS)


def name_read(root: pathlib.Path, flag: str, files: set[str]) -> dict:
    """这些文件里，**不是**形参声明、也不是按形参分支的那些行——即可能是生产者的行。

    只是为了把候选缩到几行好复核，不做判定：`**kwargs` 里的名字、环境变量常量、
    模块级 dict 都长这样，判定仍然要人读代码。
    """
    out = {}
    for rel in sorted(files):
        p = root / rel
        if not p.exists():
            out[rel] = None
            continue
        hits = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if flag not in line or line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            # 形参声明：`flag: bool = True,`
            if stripped.startswith(f"{flag}:") or stripped.startswith(f"{flag} ="):
                continue
            hits.append({"line": i, "text": stripped})
        out[rel] = hits
    return out


def classify_consumer(root: pathlib.Path, consumer: dict) -> dict:
    """看一个消费者函数，把它分到内核 / dataclass / 自赋值 / 其它。"""
    rel = consumer["file"]
    line = consumer["line"]
    flag = consumer["flag"]
    p = root / rel
    if not p.exists():
        raise SystemExit(f"[FAIL] 快照里找不到 {rel}（快照版本或布局已变，先核对 ref）")
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line:
            fn = node
            break
    if fn is None:
        raise SystemExit(f"[FAIL] {rel}:{line} 不是一个函数定义，pyflag 的行号口径变了")

    # 形参是怎么被用的：存进 self.* 还是进 if 测试
    stored_in_self = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and t.attr == flag
                ):
                    stored_in_self = True

    in_init = fn.name == "__init__"
    in_configs = "/srt/configs/" in rel
    kind = (
        "kernel_launcher"
        if is_kernel_path(rel)
        else "config_dataclass"
        if in_configs
        else "init_stored_flag"
        if (in_init and stored_in_self)
        else "source_branch"
    )
    return {
        "file": rel,
        "line": line,
        "func": fn.name,
        "call_name": consumer["call_name"],
        "param_default": consumer["param_default"],
        "stored_in_self": stored_in_self,
        "kind": kind,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--pyflag-json", type=pathlib.Path, required=True)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    report = json.loads(args.pyflag_json.read_text(encoding="utf-8"))
    inert = [f for f in report["flags"] if f.get("silently_inert")]
    if len(inert) != EXPECTED_INERT:
        print(f"[FAIL] pyflag 报告里 inert={len(inert)}，本批基线是 {EXPECTED_INERT}"
              "（上游已修 / 扫描器改过口径，先别复用本批结论）")
        return 1

    rows = []
    for f in inert:
        cons = [
            classify_consumer(args.root, dict(c, flag=f["flag"]))
            for c in f["consumers_nontest"]
        ]
        kinds = collections.Counter(c["kind"] for c in cons)
        consumer_files = {c["file"] for c in cons}
        rows.append(
            {
                "flag": f["flag"],
                "mode": f["mode"],
                "reads_nontest": f["reads_nontest"],
                "defaults": f["defaults"],
                "consumers": cons,
                "name_reads": name_read(args.root, f["flag"], consumer_files),
                "n_src_consumers": kinds["source_branch"] + kinds["init_stored_flag"],
                "n_kernel_consumers": kinds["kernel_launcher"],
                "n_config_consumers": kinds["config_dataclass"],
                "bucket": (
                    "source_candidate"
                    if kinds["source_branch"] > 0
                    else "runtime_kill_switch"
                    if kinds["init_stored_flag"] > 0
                    else "config_dataclass"
                    if kinds["config_dataclass"] > 0
                    else "kernel_launcher"
                ),
            }
        )

    bucket_counts = collections.Counter(r["bucket"] for r in rows)
    out = {
        "root": args.root.name,
        "n_flags": len(report["flags"]),
        "n_inert": len(inert),
        "buckets": dict(sorted(bucket_counts.items())),
        "rows": sorted(rows, key=lambda r: (r["bucket"], -r["reads_nontest"])),
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
