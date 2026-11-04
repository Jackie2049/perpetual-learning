#!/usr/bin/env python3
"""enable_kv_events 的行为分叉：把 tier 的真实事件缓冲逻辑跑两遍。

静态读到的是"没有任何调用点传它"，但静态证明不了"传了会有区别"——万一那条分支
什么都干呢。这里把 `fs/manager.py` 里 `__init__` 的那 8 行（`self.events = None` /
`if enable_kv_events:` …）按**真实源码文本**取出，在 CPU 上跑 False / True 两遍，
比较 `self.events` 与 logger 输出（vLLM 自己的分支里有一条告警，必须也能看到）。

不导入 vllm（无 torch/nixl），只用标准库。

Run:
    UPSTREAM=/tmp/vllm-scan-20261105 python3 kv_events_fork.py
    UPSTREAM=... python3 kv_events_fork.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

TIER_FILES = {
    "fs": "vllm/v1/kv_offload/tiering/fs/manager.py",
    "obj": "vllm/v1/kv_offload/tiering/obj/manager.py",
}


class _Logger:
    def __init__(self):
        self.records = []

    def warning(self, msg, *a):
        self.records.append(("warning", msg % a if a else msg))

    def info(self, msg, *a):
        self.records.append(("info", msg % a if a else msg))


def extract_branch(src: str) -> str:
    """取出 `self.events: ... = None` 到 `if enable_kv_events:` 分支结束的真实代码。"""
    start = src.index("        self.events: list[OffloadingEvent] | None = None")
    end = src.index("# Track in-flight store jobs", start)
    return textwrap.dedent(src[start:end])


def run_case(branch_src: str, enable: bool, global_events: bool) -> dict:
    logger = _Logger()

    class OffloadingEvent:  # noqa: D401 - 只用于类型注解的替身
        pass

    class Tier:
        def __init__(self, tier_type):
            self.tier_type = tier_type

    spec = type("Spec", (), {"kv_events_config": type("E", (), {"enable_kv_cache_events": global_events})()})()
    ns = {"logger": logger, "OffloadingEvent": OffloadingEvent}
    obj = Tier("fs")
    fn_src = ("def _init(self, offloading_spec, enable_kv_events, tier_type):\n"
              + textwrap.indent(branch_src, "    ") + "\n")
    exec(compile(fn_src, "<tier-init>", "exec"), ns)
    ns["_init"](obj, spec, enable, "fs")
    return {"enable_kv_events": enable, "global_kv_events": global_events,
            "events": None if obj.events is None else list(obj.events),
            "logs": logger.records}


def selftest(root: Path) -> int:
    ok = True
    for tier, rel in TIER_FILES.items():
        branch = extract_branch((root / rel).read_text(encoding="utf-8"))
        off = run_case(branch, False, True)
        on = run_case(branch, True, True)
        gate = run_case(branch, True, False)
        # 正向对照：开与不开必须不同
        differs = off["events"] != on["events"]
        # 第二个门：全局关着时 tier 自己开也不缓冲，且必须告警
        gated = gate["events"] is None and any(
            lvl == "warning" and "KV cache events are disabled globally" in msg
            for lvl, msg in gate["logs"])
        ok &= differs and gated
        print(f"[{tier}] enable=False -> events={off['events']}   "
              f"enable=True -> events={on['events']}   differs={differs}   "
              f"global-off gated+warned={gated}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'} 2/2 tiers")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.environ.get("UPSTREAM", "/tmp/vllm-scan-20261105"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    root = Path(a.upstream)
    if a.selftest:
        return selftest(root)
    out = {}
    for tier, rel in TIER_FILES.items():
        branch = extract_branch((root / rel).read_text(encoding="utf-8"))
        rows = sorted((run_case(branch, e, g) for e in (False, True) for g in (True, False)),
                  key=lambda r: (r["global_kv_events"], r["enable_kv_events"]))
        out[tier] = {"source_lines": len(branch.splitlines()), "cases": rows}
        for r in rows:
            print(f"[{tier}] enable={r['enable_kv_events']!s:5s} global={r['global_kv_events']!s:5s} "
                  f"events={r['events']} logs={r['logs']}")
    # 判据：**全局开** 时（也就是文档宣称能用事件的那个条件），enable 两档必须不同；
    # 全局关的档位本来就是「两个都是 None、只多一条告警」，不能拿它当分叉证据。
    def pick(tier, enable, global_events):
        for r in out[tier]["cases"]:
            if r["enable_kv_events"] is enable and r["global_kv_events"] is global_events:
                return r
        raise KeyError((tier, enable, global_events))

    forked = all(pick(t, False, True)["events"] != pick(t, True, True)["events"]
                 for t in TIER_FILES)
    gated = all(pick(t, True, False)["events"] is None and any(
        "disabled globally" in m for _, m in pick(t, True, False)["logs"])
        for t in TIER_FILES)
    print(f"behavioral_fork(global ON) = {forked}；"
          f"global OFF 时两个档位同为 None 且带告警 = {gated}")
    if a.json:
        (Path(__file__).parent / "kv-events-fork.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if forked and gated else 1


if __name__ == "__main__":
    sys.exit(main())
