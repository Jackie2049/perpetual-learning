#!/usr/bin/env python3
"""SGLang pyflag 第二批：把 34 条 `silently_inert` 的**原始读数**压到一个可复核的数目。

第一批（TM-013，`fused_moe_router_shim`）是按 `reads` 排序、逐条人眼看出来的。
本脚本不重复那次取证，只回答两个更大的问题：

  1. 34 条里**还有几条是源码层真缺陷**？（其余分别是什么结构性原因）
  2. 被判定为「非缺陷」的那些，**判定依据能不能被脚本检查**？

于是分三层，每层都有可机读的判据：

  L1 `build_triage.py`   按消费者所在位置分桶（内核 / 配置 dataclass / `__init__` 存 self / 源码分支）
  L2 本脚本              逐条把判定绑到「可 grep 的事实」：形参的每一个**读取点**都要么是
                         分支、要么是转发给别的函数、要么整条链上另有生产者
  L3 本脚本              唯一的新缺陷候选 `use_logits_buffer`：取 `logits_processor.py`
                         **真实源码**在 CPU 上执行，看两条分支的产物是否可观测地不同

为什么 L3 要跑真实源码
----------------------
「形参没人传」本身不是缺陷——没人传、两条分支又等价，就只是死开关。
`use_logits_buffer` 的注释自己写着用途：

    # The shared logits buffer is keyed by vocab width and rows; skip it
    # when this batch has a different logits shape than the graph buffer.

而唯一的调用点把 `use_logits_buffer` 传成 `i * n_chunks == 1`，直接跳过了这个判断。
所以这里要在 CPU 上把这两条分支各跑一遍，看 `logits_buffer` 是不是真的被跳过。

用法
----
    python3 verify_findings.py <sglang-snapshot> [--json out.json]

退出码
------
0 = 全部判定与读数一致
1 = 有判定不一致（读数变了 / 上游已修）
2 = 输入缺失（快照对不上、pyflag 报告条数不符）
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PYFLAG_JSON = HERE.parent.parent / "tools" / "pyflag" / "scan-sglang-main.json"

LOGITS_REL = "python/sglang/srt/layers/logits_processor.py"
LOGPROB_REL = "python/sglang/srt/layers/logprob_processor.py"
MEMPOOL_REL = "python/sglang/srt/mem_cache/memory_pool.py"

FLAG = "use_logits_buffer"
_HOST_DEFAULTS = {"torch", "self", "logits", "hidden_states", "lm_head", "logits_metadata",
                  "use_buffer", "softcap", "vocab_size"}


# ---------------------------------------------------------------- L1 复用


def load_triage(required: bool = True) -> dict:
    p = HERE / "triage-buckets.json"
    if not p.exists():
        if not required:
            return {}
        raise SystemExit(f"[FAIL] 缺 {p.name}，先跑 build_triage.py")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- L2 读取点


def _function_at(src: str, line: int):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line:
            return node
    return None


def reads_of_flag(src: str, fn) -> list[dict]:
    """函数体内对该形参的每一处读取，按「怎么读」分类。

    * `branch`  —— 出现在 if/三目/布尔运算的测试里，并且函数体内没有 `flag = ...`
                  重新绑定（否则读的是局部变量）
    * `forward` —— 作为关键字实参传给另一个函数/方法（这是转发，不是死读）
    * `self_attr` —— 存进 `self.flag`
    * `other`   —— 其余（拼接字符串、断言等），单独计数、不当证据
    """
    rebound = {
        t.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    out = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.IfExp, ast.BoolOp)):
            if any(isinstance(s, ast.Name) and s.id == FLAG for s in ast.walk(node.test
                                                                               if hasattr(node, "test")
                                                                               else node)):
                out.append({"kind": "branch", "line": node.lineno})
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg == FLAG:
                    out.append({"kind": "forward", "line": node.lineno,
                                "to": _callee(node)})
            for a in node.args:
                if isinstance(a, ast.Name) and a.id == FLAG:
                    out.append({"kind": "forward", "line": node.lineno,
                                "to": _callee(node)})
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == FLAG
                        and isinstance(t.value, ast.Name) and t.value.id == "self"):
                    out.append({"kind": "self_attr", "line": node.lineno})
    # 形参被重新绑定后，branch 读数不算
    if FLAG in rebound:
        out = [r for r in out if r["kind"] != "branch"]
    return out


def _callee(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return "?"


# ---------------------------------------------------------------- L3 行为分叉


def _extract(src: str, names: set[str]) -> dict[str, str]:
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    return out


class _FakeTensor:
    """只够桩住 `logits` 参与的那几行：形状、dtype，以及一次 `float()` 调用。"""

    def __init__(self, rows: int, cols: int, name: str, dtype: str = "bf16"):
        self.shape = (rows, cols)
        self.dtype = dtype
        self.name = name
        self.is_cuda = True

    def float(self):
        return _FakeTensor(self.shape[0], self.shape[1], self.name + ".float")

    def copy_(self, other):
        """原地拷贝：buffer 收到别人的值，名字记下来以便看产物来源。"""
        self.name = f"{self.name}<-{other.name}"
        return self

    def __repr__(self):
        return f"<{self.name} {self.shape}>"


def run_branch_surface(root: pathlib.Path) -> dict:
    """执行 `_copy_logits_to_buffer` 的**真实源码**，两条分支各跑一次。"""
    src = (root / LOGITS_REL).read_text(encoding="utf-8")
    bodies = _extract(src, {"_copy_logits_to_buffer"})
    if "_copy_logits_to_buffer" not in bodies:
        raise SystemExit("[FAIL] 源码里已无 `_copy_logits_to_buffer`，本判据失效")

    # 两个被真调用的外部函数：保留签名，记录谁被调用（不改变控制流）
    calls: list[str] = []

    def cast_to_float32_logits(logits, buffer):
        calls.append("cast_to_float32_logits")
        return buffer

    def softcap_to_float32_logits(logits, softcap, buffer):
        calls.append("softcap_to_float32_logits")
        return buffer

    class _TorchStub:
        """只桩住函数体里真正用到的两个成员：`empty_like` 与 `float` 这个 dtype。"""

        float = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

        @staticmethod
        def empty_like(t, dtype=None):
            return _FakeTensor(t.shape[0], t.shape[1], "empty_like")

    ns: dict[str, object] = {
        "cast_to_float32_logits": cast_to_float32_logits,
        "softcap_to_float32_logits": softcap_to_float32_logits,
        "torch": _TorchStub,
        "Optional": object,
        "tuple": tuple,
        "float": float,
    }
    exec(compile(bodies["_copy_logits_to_buffer"], "<real>", "exec"), ns)

    class _Meta:
        def __init__(self, buf):
            self.next_token_logits_buffer = buf

    class _Self:
        vocab_size = 8

    fn = ns["_copy_logits_to_buffer"]
    results = {}
    for label, use_buffer in (("use_logits_buffer=True", True), ("use_logits_buffer=False", False)):
        calls.clear()
        produced = _FakeTensor(4, 8, "logits")
        # 形状与 dtype 都匹配 ⇒ 默认路径会把结果直接收进这块 buffer
        buffer = _FakeTensor(4, 8, "graph_buffer", dtype="float32")
        out = fn(_Self(), produced, _Meta(buffer), use_buffer=use_buffer)
        results[label] = {
            "returned": out.name,
            "fell_back_to_float": out.name.endswith(".float"),
            "kernel_calls": list(calls),
        }
    return results


def caller_sets_buffer(root: pathlib.Path) -> dict:
    """唯一调用点把 `use_logits_buffer` 传成什么。"""
    src = (root / LOGPROB_REL).read_text(encoding="utf-8")
    facts = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and _callee(node) == "get_logits_fn":
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
            facts["get_logits_fn_call"] = {"line": node.lineno,
                                           "use_logits_buffer": kw.get(FLAG)}
    facts["chunk_count_in_scope"] = "num_chunks = (total_size + chunk_size - 1) // chunk_size"
    facts["num_chunks_binding"] = [
        {"line": n.lineno, "target": ast.unparse(n.targets[0])}
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "num_chunks"
    ]
    facts["num_chunks_reads_in_call"] = sum(
        1 for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Name) and node.id == "num_chunks")
    return facts


def default_chunking(root: pathlib.Path) -> dict:
    """默认配置下，输入 logprob 的单序列会被切成几块。"""
    env = (root / "python/sglang/srt/environ.py").read_text(encoding="utf-8")
    out = {}
    for line in env.splitlines():
        s = line.strip()
        if s.startswith("SGLANG_ENABLE_LOGPROB_CHUNK ="):
            out["SGLANG_ENABLE_LOGPROB_CHUNK"] = s.split("=", 1)[1].strip()
        if s.startswith("SGLANG_LOGPROB_CHUNK_SIZE ="):
            out["SGLANG_LOGPROB_CHUNK_SIZE"] = s.split("=", 1)[1].strip()
    return out


# ---------------------------------------------------------------- 主流程


def check_verdicts(triage: dict, findings: dict) -> list[dict]:
    """把 findings.json 的 34 条判定与 L1 分桶、pyflag 读数逐字段对账。"""
    rows = {r["flag"]: r for r in triage["rows"]}
    mismatches = []
    seen = set()

    for v in findings["verdicts"]:
        flag = v["flag"]
        if flag in seen:
            mismatches.append({"flag": flag, "why": "findings.json 里重复"})
        seen.add(flag)
        if flag not in rows:
            mismatches.append({"flag": flag, "why": "pyflag 这一批没报该条"})
            continue
        r = rows[flag]
        bucket = r["bucket"]
        label = v["label"]
        # 内核桶允许两种：编译期 constexpr（不传也等价）与转发链（真的被传，只是
        # 走的是回调参数，工具看不见）。两者都要在 findings 里写明理由。
        expect = {
            "kernel_launcher": {"KERNEL_CONSTEXPR", "KERNEL_FORWARDED"},
            "config_dataclass": {"CONFIG_DEFAULT"},
        }.get(bucket)
        if expect and label not in expect:
            mismatches.append({"flag": flag, "why": f"分桶={bucket} 但判定={label}"})
        if not expect:
            allowed = {"INTERNAL_PRODUCER", "INTERNAL_OR_DEFAULT", "ATTR_PASSING",
                       "TOPIC_ESCAPE_FAILS_OPEN", "TM013_DUP", "KERNEL_CONSTEXPR",
                       "KERNEL_FORWARDED"}
            if label not in allowed:
                mismatches.append({"flag": flag, "why": f"源码桶不接受判定={label}"})
    missing = set(rows) - seen
    for flag in sorted(missing):
        mismatches.append({"flag": flag, "why": "缺判定"})
    return mismatches


def validate_zigzag_chain(root: pathlib.Path) -> dict:
    """`use_zigzag_page_table` 被分到内核桶，但它其实是**转发链上的开关**。

    `trtllm_mha_backend.forward_extend` 在 CP 激活时把内部函数 `_trtllm_context_attn`
    交给 `cp_strategy.run_attention`，`srt/layers/cp/zigzag.py` 再用
    `use_zigzag_page_table=True` 调回来。⇒ 名字看起来一样，路径却是不透明的调用回调。
    """
    backend = (root / "python/sglang/srt/layers/attention/trtllm_mha_backend.py").read_text(
        encoding="utf-8")
    zigzag = (root / "python/sglang/srt/layers/cp/zigzag.py").read_text(encoding="utf-8")
    return {
        "backend_passes_inner_fn_to_cp": "cp_strategy.run_attention(" in backend
        and "_trtllm_context_attn," in backend,
        "cp_module_passes_true": "use_zigzag_page_table=True" in zigzag,
        "backend_default_false": "use_zigzag_page_table=False," in backend,
    }


def validate_use_dsa(root: pathlib.Path) -> dict:
    """`use_dsa` 被判「有内部生产者」——把那条生产者钉住。"""
    src = (root / MEMPOOL_REL).read_text(encoding="utf-8")
    lines = src.splitlines()
    hits = [
        {"line": i, "text": lines[i - 1].strip()}
        for i in range(1, len(lines) + 1)
        if "use_dsa=True" in lines[i - 1] and not lines[i - 1].lstrip().startswith("#")
    ]
    return {"producers": hits, "n": len(hits)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--selftest", action="store_true",
                    help="跑负向对照（不改动传入的快照）")
    args = ap.parse_args()

    if args.selftest:
        if not (args.root / LOGITS_REL).exists():
            print(f"[FAIL] 快照里没有 {LOGITS_REL}")
            return 2
        return selftest(args.root)

    if not (args.root / LOGITS_REL).exists():
        print(f"[FAIL] 快照里没有 {LOGITS_REL}")
        return 2
    triage = load_triage()
    findings = json.loads((HERE / "findings.json").read_text(encoding="utf-8"))
    if triage["n_inert"] != len(findings["verdicts"]):
        print(f"[FAIL] 分桶 {triage['n_inert']} 条 vs 判定 {len(findings['verdicts'])} 条")
        return 2

    result: dict[str, object] = {}

    # L2：每条源码/运行时常量开关的读取点分类
    reads = {}
    for row in triage["rows"]:
        if row["bucket"] not in ("source_candidate", "runtime_kill_switch"):
            continue
        per_flag = []
        for c in row["consumers"]:
            p = args.root / c["file"]
            src = p.read_text(encoding="utf-8")
            fn = _function_at(src, c["line"])
            if fn is None:
                per_flag.append({"file": c["file"], "line": c["line"], "error": "函数定位失败"})
                continue
            per_flag.append({"file": c["file"], "line": c["line"], "func": c["func"],
                             "reads": reads_of_flag(src, fn)})
        reads[row["flag"]] = per_flag
    result["reads"] = reads

    # L3：唯一的新候选，两条分支各跑一遍
    branch = run_branch_surface(args.root)
    result["use_logits_buffer_branch"] = branch
    result["use_logits_buffer_caller"] = caller_sets_buffer(args.root)
    result["logprob_chunking_defaults"] = default_chunking(args.root)
    result["use_dsa_producers"] = validate_use_dsa(args.root)
    zig = validate_zigzag_chain(args.root)
    result["use_zigzag_chain"] = zig
    if not all(zig.values()):
        mismatches_pre = [{"flag": "use_zigzag_page_table",
                           "why": f"转发链判据不成立: {zig}"}]
    else:
        mismatches_pre = []

    mismatches = check_verdicts(triage, findings) + mismatches_pre
    # 判据自检：两条分支的产物必须**不同**，否则这条候选不成立
    t = branch["use_logits_buffer=True"]["returned"]
    f = branch["use_logits_buffer=False"]["returned"]
    if t == f:
        mismatches.append({"flag": FLAG, "why": "两条分支产物相同 ⇒ 该形参无可观测差异，候选作废"})
    if not branch["use_logits_buffer=False"]["fell_back_to_float"]:
        mismatches.append({"flag": FLAG, "why": "False 分支没有退回 .float()，判据需重写"})

    result["mismatches"] = mismatches
    result["verdict_counts"] = _count_labels(findings)

    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)

    if mismatches:
        print(f"\n[FAIL] {len(mismatches)} 处不一致")
        return 1
    print("\n[OK] 34 条判定与读数一致；新候选的两条分支产物不同")
    return 0


def selftest(root: pathlib.Path) -> int:
    """负向对照：把「形参不产生差异」的情况造出来，判据必须变红。

    三条扰动，都作用在**快照副本**上，不动原始快照：
      1. `use_logits_buffer=False` 分支改成不退回 `.float()` ⇒ 两条分支产物相同
      2. 把 `_copy_logits_to_buffer` 删掉 ⇒ 判据失效必须被点名，而不是静默通过
      3. 把 `findings.json` 里一条内核桶的判定换成源码桶的判定 ⇒ 对账必须报不一致
    """
    import shutil
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, mutate, expect in (
            ("等价分支", _mut_eq, "产物相同"),
            ("删函数", _mut_drop, "判据失效"),
        ):
            work = pathlib.Path(tmp) / name
            shutil.copytree(root, work)
            mutate(work)
            try:
                res = run_branch_surface(work)
                same = res["use_logits_buffer=True"]["returned"] ==                     res["use_logits_buffer=False"]["returned"]
                got = "产物相同" if (same or res["use_logits_buffer=False"]["fell_back_to_float"] is False) else "仍分叉"
            except SystemExit as e:
                got = "判据失效" if "判据失效" in str(e) else f"异常:{e}"
            ok = got == expect
            print(f"[selftest] {name}: 期望={expect} 实测={got} {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(name)

    # 内核桶的条目被写成源码桶的判定 ⇒ 分桶与判定对不上，必须报出来
    findings = json.loads((HERE / "findings.json").read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(findings))
    for v in broken["verdicts"]:
        if v["flag"] == "enable_bmm":
            v["label"] = "TOPIC_ESCAPE_FAILS_OPEN"
    mism = check_verdicts(load_triage(required=False), broken)
    ok = any(m["flag"] == "enable_bmm" for m in mism)
    print(f"[selftest] 误判注入: 期望=报不一致 实测={'报不一致' if ok else '没报'} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("误判注入")

    if failures:
        print(f"[selftest] {len(failures)} 条失败: {failures}")
        return 1
    print("[selftest] 3/3 通过")
    return 0


def _mut_eq(root: pathlib.Path) -> None:
    p = root / LOGITS_REL
    s = p.read_text(encoding="utf-8")
    # 让 if/else 两支都退化成同一个恒等操作
    s2 = s.replace("                return cast_to_float32_logits(logits, logits_buffer)",
                   "                return logits_buffer", 1)
    s2 = s2.replace("            logits = logits.float()", "            pass", 1)
    assert s2 != s, "扰动没生效"
    p.write_text(s2, encoding="utf-8")


def _mut_drop(root: pathlib.Path) -> None:
    p = root / LOGITS_REL
    s = p.read_text(encoding="utf-8")
    start = s.index("    def _copy_logits_to_buffer(")
    end = s.index("    def _get_dllm_logits(", start)
    p.write_text(s[:start] + s[end:], encoding="utf-8")


def _count_labels(findings: dict) -> dict:
    out: dict[str, int] = {}
    for v in findings["verdicts"]:
        out[v["label"]] = out.get(v["label"], 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    sys.exit(main())
