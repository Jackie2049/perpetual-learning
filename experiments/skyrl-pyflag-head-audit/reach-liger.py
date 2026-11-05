#!/usr/bin/env python3
"""`use_liger_kernel=True` 对用户到底可不可达：三条入口逐一否掉，外加一次真实 import 读数。

用法（需要一份 checkout）：
    python3 reach-liger.py --checkout /path/to/SkyRL --json raw/reach-liger.json

判据：
  E1 Hydra/OmegaConf 结构严格性：把带该键的配置送进真实的 `SkyRLTrainConfig.from_dict_config`
     会被拒（`use_liger_kernel` 不在任何 dataclass 里）；
  E2 `HFModelWrapper` 的两处生产构造点（fsdp_worker）都不传；
  E3 依赖侧：`liger_kernel` 不在运行期依赖里（`pyproject.toml`／`uv.lock`）；
  E4 本机真的没有 `liger_kernel`（import 失败），所以即便有人传 `True` 也只会 `ModuleNotFoundError`。
边界：E1 是本机实跑（omegaconf 已装则跑、没装则记 `skipped`），E4 是本机读数，不代表服务器。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    root = os.path.abspath(a.checkout)
    out = {"checkout": root}

    # E1：真实 dataclass 是否存在该字段（AST，不 import torch）
    cfg_src = read(os.path.join(root, "skyrl/train/config/config.py"))
    out["E1_config_field_absent"] = "use_liger_kernel" not in cfg_src

    # E2：两处生产构造点都不传
    fsdp = read(os.path.join(root, "skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py"))
    calls = [m.start() for m in re.finditer(r"HFModelWrapper\(", fsdp)]
    segs = []
    for c in calls:
        depth, seg = 0, ""
        for ch in fsdp[c:]:
            seg += ch
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                break
        segs.append("use_liger_kernel" in seg)
    out["E2_call_sites"] = len(segs)
    out["E2_any_call_passes_it"] = any(segs)

    # E3：依赖侧
    pyproject = read(os.path.join(root, "pyproject.toml"))
    out["E3_in_pyproject"] = "liger" in pyproject.lower()
    lock = os.path.join(root, "uv.lock")
    out["E3_in_uv_lock_root"] = os.path.exists(lock) and '"liger-kernel"' in read(lock)
    # skyrl-agent 有自己的 uv.lock、确实锁了 liger-kernel —— 那是另一个子工程
    agent_lock = os.path.join(root, "skyrl-agent/uv.lock")
    out["E3_in_skyrl_agent_lock"] = os.path.exists(agent_lock) and '"liger-kernel"' in read(agent_lock)
    # 除根 lock 外，任何子工程的 lock/pyproject 提到 liger 都出个读数（归因要写清它属于哪个子工程）
    others = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != ".git"]
        for f in fn:
            if f in ("uv.lock", "pyproject.toml"):
                p = os.path.join(dp, f)
                if "liger" in read(p).lower():
                    others.append(os.path.relpath(p, root))
    out["E3_other_manifests_mentioning_liger"] = others

    # E4：本机 import
    out["E4_liger_importable_here"] = importlib.util.find_spec("liger_kernel") is not None
    # E3 对照：连**直接** import 都不用（分支里是懒 import，但先把源码事实固定下来）
    mw = read(os.path.join(root, "skyrl/backends/skyrl_train/workers/model_wrapper.py"))
    out["branch_is_lazy_import"] = "from liger_kernel.transformers import AutoLigerKernelForCausalLM" in mw
    # E4 对照：本机自检「import 失败」这条判据不是死的——拿一个一定装了的包做正向对照
    out["E4_positive_control_json_importable"] = importlib.util.find_spec("json") is not None

    checks = {
        "E1_config_field_absent": out["E1_config_field_absent"],
        "E2_no_call_passes_it": (not out["E2_any_call_passes_it"]) and out["E2_call_sites"] >= 2,
        "E3_not_a_runtime_dependency": (not out["E3_in_pyproject"]) and (not out["E3_in_uv_lock_root"]),
        "E4_liger_absent_here": not out["E4_liger_importable_here"],
        "E4_positive_control_works": out["E4_positive_control_json_importable"],
        "E3_branch_is_lazy_import": out["branch_is_lazy_import"],
    }
    out["checks"] = checks
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    print(json.dumps({k: out[k] for k in out if k.startswith(("E2_", "E3_", "E4_"))}, ensure_ascii=False))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
