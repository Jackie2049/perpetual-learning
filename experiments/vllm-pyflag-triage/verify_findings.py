#!/usr/bin/env python3
"""vLLM pyflag 第二批：把 12 条 `silently_inert` 逐条绑到可 grep 的事实。

只读静态分析，不导入、不运行 vLLM（torch/nixl 本机没有也不该为这件事装）。
每条判定落到具体文件的具体行号；锚点找不到就报 MISMATCH。

用法：
    UPSTREAM=/tmp/vllm-scan-20261105 python3 verify_findings.py       # exit 0 = 读数与判定一致
    UPSTREAM=... python3 verify_findings.py --json                    # 另落盘 findings.json
    UPSTREAM=... python3 verify_findings.py --selftest                # 注入必须让对应判据变红

退出码：0 一致 / 1 锚点漂移（上游已变，结论需重判）/ 2 自检失败（工具坏了，读数作废）。

两个易错点（本脚本自己踩过，判据里都留了痕）：
  * 形参默认值声明行 `def f(use_x=False)` 不是生产者；
  * `text.count("name")` 会把 docstring、告警文案、`if name:` 分支一起算进去，
    所以"某处没传 / 没有赋值"必须用**带上下文的正则**判，不能靠子串计数。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CHECKS: list[tuple[str, bool, str]] = []
INJ: set[str] = set()


def check(name: str, cond: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(cond), detail))
    return bool(cond)


def inj(name: str) -> bool:
    return name in INJ


def inj_fail(name: str, cond: bool) -> bool:
    """注入对应缺陷时，这条判据必须变红。"""
    return bool(cond) and not inj(name)


def read(root: Path, rel: str) -> str:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def lineno_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return 0 if idx < 0 else text.count("\n", 0, idx) + 1


def lines(text: str) -> list[str]:
    return text.splitlines()


# ======================================================================================
# F1  enable_kv_events —— secondary tier 的逐层事件开关无生产者
# ======================================================================================
def f1_enable_kv_events(root: Path) -> dict:
    fs = read(root, "vllm/v1/kv_offload/tiering/fs/manager.py")
    obj = read(root, "vllm/v1/kv_offload/tiering/obj/manager.py")
    spec = read(root, "vllm/v1/kv_offload/tiering/spec.py")
    factory = read(root, "vllm/v1/kv_offload/tiering/factory.py")
    docs = read(root, "docs/features/kv_offloading_usage.md")

    fs_decl = lineno_of(fs, "enable_kv_events: bool = False,")
    obj_decl = lineno_of(obj, "enable_kv_events: bool = False,")
    fs_branch = lineno_of(fs, "if enable_kv_events:")
    obj_branch = lineno_of(obj, "if enable_kv_events:")
    check("F1.fs_declares_parameter", fs_decl > 0, f"fs/manager.py:{fs_decl}")
    check("F1.obj_declares_parameter", obj_decl > 0, f"obj/manager.py:{obj_decl}")
    check("F1.fs_branches_on_flag", inj_fail("kv_events_producer", fs_branch > 0),
          f"fs/manager.py:{fs_branch}")
    check("F1.obj_branches_on_flag", obj_branch > 0, f"obj/manager.py:{obj_branch}")

    n_create = spec.count("create_secondary_tier(")
    check("F1.single_construction_path", n_create == 1,
          f"spec.py:{lineno_of(spec, 'create_secondary_tier(')}（共 {n_create} 处）")
    check("F1.factory_expands_tier_config", "**config," in factory,
          f"factory.py:{lineno_of(factory, '**config,')}")

    doc_line = lineno_of(docs, "| `enable_kv_events` |")
    check("F1.docs_advertise_key", doc_line > 0,
          f"docs/features/kv_offloading_usage.md:{doc_line}")

    occurrences, assigners = [], []
    for p in (root / "vllm").rglob("*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "enable_kv_events" not in txt:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if "enable_kv_events" not in line:
                continue
            occurrences.append((str(p.relative_to(root)), i, line.strip()))
            s = line.strip()
            if re.match(r"enable_kv_events\s*:", s):      # 形参声明
                continue
            if s.startswith("if "):                       # 分支读取
                continue
            if s.startswith('"'):                         # 告警文案
                continue
            if re.search(r"(?<![\w.])enable_kv_events\s*=\s*[^=]", line):
                assigners.append((str(p.relative_to(root)), i, s))
    check("F1.no_assignment_anywhere", inj_fail("kv_events_producer", not assigners),
          f"{len(occurrences)} occurrences, {len(assigners)} assignments")

    check("F1.no_alternative_passthrough",
          fs.count("enable_kv_events") == 4 and obj.count("enable_kv_events") == 4,
          f"fs×{fs.count('enable_kv_events')} obj×{obj.count('enable_kv_events')}"
          "（声明+docstring+分支+告警文案）")

    ok = all(c for n, c, _ in CHECKS if n.startswith("F1."))
    return {"id": "F1", "flag": "enable_kv_events",
            "verdict": "TRUE_DEFECT" if ok else "DECISION_FLIPPED",
            "anchors": {"fs_param_line": fs_decl, "obj_param_line": obj_decl,
                        "fs_branch_line": fs_branch, "obj_branch_line": obj_branch,
                        "docs_line": doc_line},
            "occurrences_outside_tests": len(occurrences),
            "assignments_outside_tests": len(assigners)}


# ======================================================================================
# F2  no_thinking —— Jina re-ranker 的输出模板里 `</think` 恒被补上
# ======================================================================================
def f2_no_thinking(root: Path) -> dict:
    rel = "vllm/entrypoints/pooling/scoring/io_processor.py"
    src = read(root, rel)
    decl = lineno_of(src, "no_thinking: bool = True,")
    branch = lineno_of(src, "if no_thinking:")
    think = lineno_of(src, 'suffix += "<think>\\n\\n</think>\\n\\n"')
    check("F2.parameter_exists", decl > 0, f"{rel}:{decl}")
    check("F2.branch_exists", branch > 0, f"{rel}:{branch}")
    check("F2.think_block_reachable", think > 0, f"{rel}:{think}")

    call_lines = [i for i, line in enumerate(lines(src), 1)
                  if "self.format_docs_prompts_func(" in line]
    passing = [i for i in call_lines if re.search(r"no_thinking\s*=", src)]
    check("F2.calls_present", len(call_lines) >= 4, f"{len(call_lines)} call sites")
    check("F2.no_call_passes_flag", inj_fail("thinking_kwarg", not passing),
          f"kwarg 传参处: {passing}")

    embed = read(root, "vllm/entrypoints/pooling/embed/io_processor.py")
    check("F2.not_overridden_elsewhere", "def format_docs_prompts_func" not in embed,
          "embed/io_processor.py 只调用、不重定义")

    ok = all(c for n, c, _ in CHECKS if n.startswith("F2."))
    return {"id": "F2", "flag": "no_thinking",
            "verdict": "TEMPLATE_FLAG_UNREACHABLE" if ok else "DECISION_FLIPPED",
            "anchors": {"param_line": decl, "branch_line": branch, "think_line": think,
                        "call_sites": call_lines}}


# ======================================================================================
# F3  use_page_cache —— 对照组：同样「默认值 + 分支」，但它其实是接通的
# ======================================================================================
def f3_use_page_cache(root: Path) -> dict:
    conn = read(root, "vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py")
    worker = read(root, "vllm/v1/simple_kv_offload/worker.py")
    backend = read(root, "vllm/v1/simple_kv_offload/disk_backend.py")
    read_line = lineno_of(conn, 'use_page_cache = bool(extra_config.get("use_page_cache", False))')
    fwd = lineno_of(conn, "use_page_cache=use_page_cache,")
    store = lineno_of(worker, "self.use_page_cache = use_page_cache")
    branch = lineno_of(backend, "if not use_page_cache:")
    check("F3.config_read_by_producer", read_line > 0, f"connector:{read_line}")
    check("F3.producer_forwards_it", fwd > 0, f"connector:{fwd}")
    check("F3.consumer_stores_it", store > 0, f"worker.py:{store}")
    check("F3.consumer_branches_on_it", inj_fail("page_cache_unwired", branch > 0),
          f"disk_backend.py:{branch}")
    ok = all(c for n, c, _ in CHECKS if n.startswith("F3."))
    return {"id": "F3", "flag": "use_page_cache",
            "verdict": "WIRED_NOT_A_DEFECT" if ok else "DECISION_FLIPPED",
            "anchors": {"read_line": read_line, "forward_line": fwd,
                        "store_line": store, "branch_line": branch}}


# ======================================================================================
# F4  enable_trace_replay —— 对照组 2：pyflag 看不见的生产者是 config 字段的**属性**
# ======================================================================================
def f4_enable_trace_replay(root: Path) -> dict:
    model = read(root, "vllm/config/model.py")
    runner = read(root, "vllm/v1/worker/gpu/model_runner.py")
    sampler = read(root, "vllm/v1/worker/gpu/sample/sampler.py")
    field = lineno_of(model, "enable_trace_replay: bool = False")
    producer = lineno_of(runner, '"enable_trace_replay": self.model_config.enable_trace_replay')
    consume = lineno_of(sampler, "TraceReplayState(req_states) if enable_trace_replay else None")
    check("F4.config_field_exists", field > 0, f"config/model.py:{field}")
    check("F4.runner_reads_config_field", producer > 0, f"model_runner.py:{producer}")
    check("F4.flag_reaches_consumer", inj_fail("trace_replay_unwired", consume > 0),
          f"sample/sampler.py:{consume}")
    ok = all(c for n, c, _ in CHECKS if n.startswith("F4."))
    return {"id": "F4", "flag": "enable_trace_replay",
            "verdict": "INDIRECT_PRODUCER" if ok else "DECISION_FLIPPED",
            "anchors": {"field_line": field, "producer_line": producer,
                        "consumer_line": consume}}


# ======================================================================================
# F5  use_mix / use_abs_pos / use_sliding_window / use_grads —— 模型实现内部的构造期开关
# ======================================================================================
def f5_constructor_flags(root: Path) -> dict:
    cases = {
        "use_mix": ("vllm/models/qwen4_exp/common/hyperconnection.py",
                    "use_mix: bool = True,", "if use_mix:"),
        "use_abs_pos": ("vllm/model_executor/models/deepencoder.py",
                        "use_abs_pos: bool = True,", "if use_abs_pos:"),
        "use_sliding_window": ("vllm/models/flint/configs.py",
                               "use_sliding_window: bool = True,",
                               "if use_sliding_window else None"),
        "use_grads": ("vllm/transformers_utils/processors/cohere_asr.py",
                      "use_grads=False,", "if not use_grads:"),
    }
    out, ok_all = {}, True
    for flag, (rel, decl, branch) in cases.items():
        src = read(root, rel)
        d, b = lineno_of(src, decl), lineno_of(src, branch)
        ok = (d > 0 and b > 0) and not inj("f5_drop_branch")
        ok_all &= ok
        check(f"F5.{flag}.decl_and_branch", ok, f"{rel}:{d}/{b}")
        out[flag] = {"file": rel, "decl_line": d, "branch_line": b}
    return {"id": "F5", "verdict": "MODEL_INTERNAL_DEFAULT" if ok_all else "DECISION_FLIPPED",
            "flags": out}


# ======================================================================================
# F6  use_fused_tilelang / use_pre_gemm_splits —— 生产者是同文件的小表，不是前端配置
# ======================================================================================
def f6_tilelang_table(root: Path) -> dict:
    rel = "vllm/model_executor/kernels/mhc/tilelang_kernels.py"
    src = read(root, rel)
    fused_decl = lineno_of(src, "use_fused_tilelang: bool = False,")
    splits_decl = lineno_of(src, "use_pre_gemm_splits: bool = False,")
    fused_true = src.count("use_fused_tilelang=True,")
    fused_false = src.count("use_fused_tilelang=False,")
    splits_local = (src.count("use_pre_gemm_splits=include_pre_gemm_splits,")
                    + src.count("use_pre_gemm_splits=include_broadcast_splits,"))
    splits_false = src.count("use_pre_gemm_splits=False,")
    check("F6.fused_declared", fused_decl > 0, f"{rel}:{fused_decl}")
    check("F6.splits_declared", splits_decl > 0, f"{rel}:{splits_decl}")
    check("F6.fused_has_literal_producers",
          inj_fail("f6_strip_producers", fused_true >= 1 and fused_false >= 1),
          f"True×{fused_true} False×{fused_false}")
    check("F6.splits_fed_from_local_vars", splits_local >= 2 and splits_false >= 1,
          f"local×{splits_local} False×{splits_false}")
    ok = all(c for n, c, _ in CHECKS if n.startswith("F6."))
    return {"id": "F6", "verdict": "INTERNAL_TABLE_LOCAL" if ok else "DECISION_FLIPPED",
            "flags": {"use_fused_tilelang": {"decl_line": fused_decl,
                                             "producers": fused_true + fused_false},
                      "use_pre_gemm_splits": {"decl_line": splits_decl,
                                              "producers": splits_local + splits_false}}}


# ======================================================================================
# F7  use_heuristic —— 只有测试传，两条分支都是合法路径
# ======================================================================================
def f7_use_heuristic(root: Path) -> dict:
    rel = "vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py"
    src = read(root, rel)
    decl = lineno_of(src, "use_heuristic=True,")
    branch = lineno_of(src, "if use_heuristic:")
    calls = [i for i, line in enumerate(lines(src), 1)
             if "triton_scaled_mm(" in line and not line.strip().startswith("def ")]
    if inj("f7_prod_caller"):
        calls.append(1)  # 注入：假装库内多了一处调用点
    test = read(root, "tests/kernels/quantization/test_triton_scaled_mm.py")
    check("F7.declared", decl > 0, f"{rel}:{decl}")
    check("F7.branch", branch > 0, f"{rel}:{branch}")
    check("F7.not_called_from_library", inj_fail("f7_prod_caller", not calls),
          f"库内非 def 行的调用点: {calls or '无'}（只有 `def` 那条）")
    check("F7.producer_is_test_only", "use_heuristic=use_heuristic" in test,
          "测试里 True/False 两档都跑")
    ok = all(c for n, c, _ in CHECKS if n.startswith("F7."))
    return {"id": "F7", "verdict": "TEST_ONLY_KNOB" if ok else "DECISION_FLIPPED",
            "anchors": {"decl_line": decl, "branch_line": branch, "call_lines": calls}}


# ======================================================================================
# F8  enable_telemetry —— 生产者是 extra_config 的**字典键**
# ======================================================================================
def f8_enable_telemetry(root: Path) -> dict:
    rel = "vllm/v1/kv_offload/tiering/kvcr/manager.py"
    src = read(root, rel)
    read_cfg = lineno_of(src, 'extra_config.get("enable_telemetry", False)')
    decl = lineno_of(src, "enable_telemetry: bool = False,")
    fwd = lineno_of(src, "enable_telemetry=enable_telemetry")
    check("F8.reads_extra_config_key", inj_fail("f8_drop_read", read_cfg > 0),
          f"{rel}:{read_cfg}")
    check("F8.param_declared", decl > 0, f"{rel}:{decl}")
    check("F8.forwarded_internally", fwd > 0, f"{rel}:{fwd}")
    ok = all(c for n, c, _ in CHECKS if n.startswith("F8."))
    return {"id": "F8", "verdict": "CONFIG_DICT_PRODUCER" if ok else "DECISION_FLIPPED",
            "anchors": {"read_line": read_cfg, "decl_line": decl, "forward_line": fwd}}


def run(root: Path) -> dict:
    CHECKS.clear()
    findings = [f1_enable_kv_events(root), f2_no_thinking(root), f3_use_page_cache(root),
                f4_enable_trace_replay(root), f5_constructor_flags(root),
                f6_tilelang_table(root), f7_use_heuristic(root), f8_enable_telemetry(root)]
    return {"upstream": str(root),
            "checks": [{"name": n, "ok": c, "detail": d} for n, c, d in CHECKS],
            "n_checks": len(CHECKS),
            "n_failed": sum(1 for _, c, _ in CHECKS if not c),
            "findings": findings}


INJECTIONS = {
    "kv_events_producer": "F1.no_assignment_anywhere",
    "thinking_kwarg": "F2.no_call_passes_flag",
    "page_cache_unwired": "F3.consumer_branches_on_it",
    "trace_replay_unwired": "F4.flag_reaches_consumer",
    "f5_drop_branch": "F5.use_mix.decl_and_branch",
    "f6_strip_producers": "F6.fused_has_literal_producers",
    "f7_prod_caller": "F7.not_called_from_library",
    "f8_drop_read": "F8.reads_extra_config_key",
}


def selftest(root: Path) -> int:
    global INJ
    INJ = set()
    base = run(root)
    ok = base["n_failed"] == 0
    print(f"[selftest] baseline: {base['n_checks']} checks, {base['n_failed']} failed")
    for c in base["checks"]:
        if not c["ok"]:
            print("   baseline RED:", c["name"], c["detail"])
    for name, target in INJECTIONS.items():
        INJ = {name}
        r = run(root)
        red = [c["name"] for c in r["checks"] if not c["ok"]]
        caught = target in red
        ok &= caught
        print(f"[selftest] inject={name:22s} target={target:34s} caught={caught} red={red}")
    INJ = set()
    print(f"[selftest] {'PASS' if ok else 'FAIL'} {len(INJECTIONS)}/{len(INJECTIONS)}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.environ.get("UPSTREAM", "/tmp/vllm-scan-20261105"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    root = Path(a.upstream)
    if not (root / "vllm").is_dir():
        print(f"not a vLLM tree: {root}", file=sys.stderr)
        return 3
    if a.selftest:
        return selftest(root)
    r = run(root)
    for c in r["checks"]:
        print(f"[{'ok' if c['ok'] else 'MISMATCH'}] {c['name']:40s} {c['detail']}")
    for f in r["findings"]:
        print(f"  -> {f['id']:3s} {f['verdict']}")
    print(f"{r['n_checks']} checks, {r['n_failed']} failed")
    if a.json:
        (Path(__file__).parent / "findings.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))
    return 0 if r["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
