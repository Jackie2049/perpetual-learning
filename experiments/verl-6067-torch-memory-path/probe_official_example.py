#!/usr/bin/env python3
"""照官方示例脚本 `examples/profile/run_qwen2_5_vl_7b_torch_memory.sh` 的 overrides 渲染真实配置，
看它在当前 main 上会落到什么状态。

示例脚本只给 `global_profiler.tool=torch_memory` + `save_path` + 两个 tool_config 值，
**没有** `*.profiler.enable=True`。而当前 main 的 torch_memory 快照只在
`DistProfiler.start()` 里推进窗口（`check_enable()` 为 False 时直接跳过），
`enable` 为 False ⇒ 一步都不会写。

第二组探针：手工把 `actor.profiler.enable=True` 打开，看快照**写到哪**——决定路径的是角色级
`actor.profiler.save_path`（默认继承 `global_profiler.save_path`），不是示例脚本里
`PROFILE_SAVE_PATH` 这个环境变量本身。

用法：
    python probe_official_example.py --repo <verl 仓库> [--json example-result.json]
退出码：0 = 读数与判定一致；1 = 读数变了（示例可能已修，需重查）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 与示例脚本 EXTRA 数组逐行对应（examples/profile/run_qwen2_5_vl_7b_torch_memory.sh）
EXAMPLE_OVERRIDES = [
    "global_profiler.tool=torch_memory",
    "global_profiler.save_path=./mem_snapshots",
    "global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=100000",
    "global_profiler.global_tool_config.torch_memory.stack_depth=32",
]
ROLES = ("actor_rollout_ref.actor", "actor_rollout_ref.rollout", "critic")
EXAMPLE_SAVE_PATH = "./mem_snapshots"  # 示例脚本的 PROFILE_SAVE_PATH 默认值，同时也写进 global_profiler.save_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=os.path.join(args.repo, "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=EXAMPLE_OVERRIDES)
    OmegaConf.resolve(cfg)

    roles = {}
    for sec in ROLES:
        p = OmegaConf.select(cfg, sec + ".profiler")
        if p is None:
            continue
        roles[sec] = {"tool": p.get("tool"), "enable": p.get("enable"), "save_path": p.get("save_path")}

    # 用真实 DistProfiler 验证"enable=False ⇒ 即使到了 scheduled step 也不进后端"
    from copy import deepcopy

    from verl.utils.profiler.profile import DistProfiler

    actor_prof = deepcopy(cfg.actor_rollout_ref.actor.profiler)
    # trainer 实际会传显式 tool_config（逐角色取自己的 tool_config.<tool>），这里一样处理
    tool_config = (
        actor_prof.get("tool_config", {}).get(actor_prof.get("tool")) if actor_prof.get("tool") else None
    )
    dist = DistProfiler(rank=0, config=actor_prof, tool_config=tool_config)
    enable = dist.check_enable()
    dumps: list = []
    if enable:  # 只有在 enable=True 时才有必要继续观察
        from verl.utils.profiler import torch_memory_profile as T

        orig = T.MemorySnapshotSampler.dump_memory_snapshot
        T.MemorySnapshotSampler.dump_memory_snapshot = lambda self, **kw: dumps.append(kw) or None
        try:
            dist.start(profile_step=1)
            dist.stop(run_command=False)
        finally:
            T.MemorySnapshotSampler.dump_memory_snapshot = orig

    # 第二组：手工打开 enable，看写到哪里（示例给了 save_path=./mem_snapshots）
    with initialize_config_dir(config_dir=os.path.join(args.repo, "verl/trainer/config"), version_base=None):
        cfg2 = compose(
            config_name="ppo_trainer",
            overrides=EXAMPLE_OVERRIDES
            + ["actor_rollout_ref.actor.profiler.enable=True", "actor_rollout_ref.actor.profiler.ranks=[0]"],
        )
    OmegaConf.resolve(cfg2)
    from verl.utils.config import omega_conf_to_dataclass

    from verl.utils.profiler import torch_memory_profile as T

    actor_prof2 = omega_conf_to_dataclass(cfg2.actor_rollout_ref.actor.profiler)
    dumps2: list = []
    T.TorchMemoryProfiler._memory_history_enabled = True
    orig = T.MemorySnapshotSampler.dump_memory_snapshot
    T.MemorySnapshotSampler.dump_memory_snapshot = lambda self, **kw: dumps2.append(kw) or None
    try:
        dist2 = DistProfiler(rank=0, config=actor_prof2, tool_config=actor_prof2.tool_config.get("torch_memory"))
        dist2.start(role="train", profile_step=1)
        dist2.stop(run_command=False)
    finally:
        T.MemorySnapshotSampler.dump_memory_snapshot = orig

    report = {
        "overrides": EXAMPLE_OVERRIDES,
        "roles": roles,
        "dist_profiler": {"enable": enable, "dumps": dumps},
        "all_roles_enable_false": all(v["enable"] is False for v in roles.values()),
        "enable_forced": {
            "actor_save_path": getattr(actor_prof2, "save_path", None),
            "global_save_path": cfg2.global_profiler.save_path,
            "dumps": dumps2,
        },
    }
    checks = [
        {
            "name": "示例脚本的 overrides 让三个角色的 profiler.enable 都是 False",
            "ok": report["all_roles_enable_false"],
            "evidence": roles,
        },
        {
            "name": "该配置下 DistProfiler.check_enable() 为 False（start 不会推进窗口，后端不会被调用）",
            "ok": enable is False and dumps == [],
            "evidence": {"enable": enable, "dumps": dumps},
        },
        {
            "name": "手工打开 enable 后快照写到角色级 save_path，与 global_profiler.save_path 一致",
            "ok": len(dumps2) == 1
            and dumps2[0]["out_dir"] == getattr(actor_prof2, "save_path", None)
            and dumps2[0]["out_dir"] == EXAMPLE_SAVE_PATH,
            "evidence": {
                "dumps": dumps2,
                "expected": EXAMPLE_SAVE_PATH,
                "profile_save_path_env_var_used": False,
            },
        },
    ]
    report["checks"] = checks
    report["n_fail"] = sum(1 for c in checks if not c["ok"])

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["name"], file=sys.stderr)
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
