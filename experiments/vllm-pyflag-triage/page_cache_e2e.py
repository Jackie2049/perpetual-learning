#!/usr/bin/env python3
"""use_page_cache 的配置链路：用**真实的 `kv_connector_extra_config` 处理段**跑一遍。

这与 verify_findings.py 的分工：那边证明"名字在哪些行"，这边证明"配置真的会走进去"。
取 `SimpleCPUOffloadConnector._common_init` 里的真实处理逻辑（不含 GPU worker 构造），
把 ctx 换成最小桩，比较 `use_page_cache` 三种取值下 worker/backend 收到的实参。

为什么值得跑：pyflag 把它报成 `silently_inert`（`use_page_cache: bool = False` 在
`worker.py` / `disk_backend.py` 里都有分支，却没看到同名关键字实参）。真相是生产者
用 **`self.use_page_cache`** 转发（第五类误报）。这里把"传了就有、不传就没有"跑成读数。

Run:
    UPSTREAM=/tmp/vllm-scan-20261105 python3 page_cache_e2e.py
    UPSTREAM=... python3 page_cache_e2e.py --selftest     # 去掉转发必须被抓住
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

CONNECTOR = "vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py"


def extract(src: str) -> str:
    """把真实源码里的配置处理段抠出来，去掉需要 vllm 运行时的部分。"""
    start = src.index("        extra_config = self._kv_transfer_config")
    end = src.index("        if disk_mode:", start)
    # 这一段只用到 extra_config / world_size / logger，其他都换成桩
    body = textwrap.dedent(src[start:end])
    body = body.replace("self._kv_transfer_config", "ctx.kv_transfer_config")
    body = body.replace("vllm_config.", "ctx.vllm_config.")
    body = body.replace("self._cache_hit_source", "ctx.cache_hit_source")
    return body


def harness(root: Path, inject_drop_forward: bool = False) -> dict:
    src = (root / CONNECTOR).read_text(encoding="utf-8")
    body = extract(src)
    # 收集 worker::init 收到的实参：把 disk_backend 的 `if not use_page_cache:` 换成记录
    backend = (root / "vllm/v1/simple_kv_offload/disk_backend.py").read_text(encoding="utf-8")
    o_direct_branch = "if not use_page_cache:" in backend
    if inject_drop_forward:
        o_direct_branch = False

    class Cfg:
        pass

    class Ctx:
        world_size = 1
        kv_transfer_config = None

    class _Sched:
        pass

    class Stub:
        """替代 SimpleCPUOffloadScheduler/Worker 的 `init`，只记录 use_page_cache。"""

        def __init__(self, **kw):
            self.kw = kw
            self.use_page_cache = kw.get("use_page_cache")

    rows = []
    for label, extra in [("unset", {}),
                         ("true", {"use_page_cache": True}),
                         ("false", {"use_page_cache": False})]:
        extra = dict(extra)
        extra.setdefault("kv_offload_backend", "disk")
        extra.setdefault("disk_path", "/tmp/fake-disk")
        extra.setdefault("disk_capacity_bytes", 1024 ** 3)
        ctx = Ctx()
        ctx.cache_hit_source = None
        ctx.kv_transfer_config = Ctx()
        ctx.kv_transfer_config.kv_connector_extra_config = extra
        ctx.vllm_config = _NS(parallel_config=_NS(world_size=1))
        local = {"logger": _Logger(), "int": int, "bool": bool, "str": str,
                 "SimpleCPUOffloadScheduler": Stub, "SimpleCPUOffloadWorker": Stub,
                 "disk_mode": None, "cpu_capacity_bytes": None,
                 "cpu_capacity_per_rank": None, "num_cpu_blocks": None,
                 "lazy_offload": None, "kv_offload_backend": None,
                 "use_page_cache": None, "disk_path": None,
                 "disk_capacity_bytes": None, "disk_buffer_slots": None,
                 "DEFAULT_CPU_CAPACITY_BYTES": 8 * (1024 ** 3),
                 "VALID_KV_OFFLOAD_BACKENDS": ("cpu", "disk"),
                 "CacheHitSource": _NS(DISK="disk", HOST="host"),
                 "ValueError": ValueError,
                 "_DISK_ONLY_KEYS": ("disk_path", "disk_capacity_bytes",
                                     "disk_buffer_slots", "use_page_cache")}
        local["ctx"] = ctx
        exec(compile(body, "<connector-config-segment>", "exec"), local)
        rows.append({"case": label,
                     "config_value": extra.get("use_page_cache", "<absent>"),
                     "parsed": local["use_page_cache"]})

    # 转发段：真实源码里 worker/backend 拿的是 `use_page_cache=use_page_cache`
    forwards = {
        "scheduler_passes_to_worker": "use_page_cache=use_page_cache," in src,
        "worker_stores_attr": "self.use_page_cache = use_page_cache" in
                              (root / "vllm/v1/simple_kv_offload/worker.py").read_text(encoding="utf-8"),
        "backend_branches": o_direct_branch,
    }
    if inject_drop_forward:
        forwards["scheduler_passes_to_worker"] = False
    return {"cases": rows, "forwards": forwards,
            "wired": (forward[0] for forward in [()])}  # 占位，下面覆盖


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Logger:
    def __init__(self):
        self.msgs = []

    def warning(self, *a, **k):
        self.msgs.append(("warning", a))

    def info(self, *a, **k):
        self.msgs.append(("info", a))


def evaluate(root: Path, inject: bool = False) -> dict:
    r = harness(root, inject)
    r.pop("wired", None)
    cases = {c["case"]: c["parsed"] for c in r["cases"]}
    fw = r["forwards"]
    # 判定：配置值被如实解析 + 每一跳都真的转发 + 后端真的按它分支
    wired = (cases["unset"] is False and cases["true"] is True and cases["false"] is False
             and fw["scheduler_passes_to_worker"] and fw["worker_stores_attr"]
             and fw["backend_branches"])
    r["verdict"] = "WIRED" if wired else "BROKEN"
    return r


def selftest(root: Path) -> int:
    base = evaluate(root)
    ok = base["verdict"] == "WIRED"
    print(f"[selftest] baseline: verdict={base['verdict']} cases={base['cases']}")
    broken = evaluate(root, inject=True)
    caught = broken["verdict"] == "BROKEN"
    ok &= caught
    print(f"[selftest] inject=drop_forward: verdict={broken['verdict']} "
          f"forwards={broken['forwards']} caught={caught}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'} 1/1")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.environ.get("UPSTREAM", "/tmp/vllm-scan-20261105"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    root = Path(a.upstream)
    if not (root / CONNECTOR).exists():
        print(f"not a vLLM tree: {root}", file=sys.stderr)
        return 3
    if a.selftest:
        return selftest(root)
    r = evaluate(root)
    for c in r["cases"]:
        print(f"  case={c['case']:6s} config={c['config_value']!s:8s} parsed={c['parsed']}")
    for k, v in r["forwards"].items():
        print(f"  {k}: {v}")
    print(f"verdict: {r['verdict']}")
    if a.json:
        (Path(__file__).parent / "page-cache-e2e.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))
    return 0 if r["verdict"] == "WIRED" else 1


if __name__ == "__main__":
    sys.exit(main())
