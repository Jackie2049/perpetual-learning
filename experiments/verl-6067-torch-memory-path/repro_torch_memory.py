#!/usr/bin/env python3
"""#6067 engine 重构之后，`global_profiler.tool=torch_memory` 还灵不灵。

可查的提交事实（脚本会逐条重算成布尔值，不靠人眼）：
- #3099（2025-08）让 trainer 在每步后调 `actor_rollout_wg.dump_memory_snapshot(...)`；
- #6198（2026-04-29，收 #6181）删掉这些调用，PR 说明写的是
  "relevant capabilities are fully integrated in DistProfiler ... Collect memory_snapshot via
  global_profiler.tool=torch_memory + global_profiler.steps=[...] + actor.profiler.enable=True"；
- 而 `DistProfiler` 那条链在 #6067（2026-04-20，早于 #6198）之后只存在于 engine worker 上，
  engine worker 的方法集合里**没有** dump 入口。

两件事，都在 CPU 上做、不跑训练：
  A. 入口面：父提交与当前 main 上两个 worker 类的公开方法差集（AST，不看注释）。
  B. 可执行性：用真实 `ppo_trainer.yaml`（hydra compose + 与 `run_ppo` 相同的 `OmegaConf.resolve`）
     造出 trainer 视角的 profiler 配置，再走真实 `omega_conf_to_dataclass`；然后按
     `DistProfilerExtension.start_profile` → `DistProfiler.start` → 后端 `start/stop` 的真实调用链
     驱动，用间谍替换唯一写文件的那一步，看**有没有发生、写到哪**。

三条探针（每条都有对应的负向对照）：
  1. `tool=torch_memory` 的真配置：**走通了**，每一步写一张快照到 `outputs/profile/step<N>/`；
  2. 同配置换成**旧 worker 那种写法**（trainer 在循环里显式调 `dump_memory_snapshot`）：
     后端根本没有这个方法 → `AttributeError`（这正是 #6198 修掉的那个 bug 形态）；
  3. **带 `_target_` 的 profiler 配置**（`run_ppo` 的 `run_ppo_v1` 分支与 `ActorRolloutRefWorker`
     都这么转）：`tool_config` 退化成 `dict` → 后端构造期 `AttributeError`。

用法：
    python repro_torch_memory.py --repo <verl 仓库> [--json repro-result.json] [--selftest]
退出码：0 = 全部断言成立；1 = 有断言不成立（结论可能失效，需重查）。

依赖（只用它们在 CPU 上做配置解析与调用链）：ray omegaconf hydra-core torch transformers tensordict codetiming
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys

HEAD = "74543265"  # 2026-11-04 fetch 的 verl main
REFACTOR = "044bbba2"  # #6067 BREAKING: deprecate workers, migrate to engines
PARENT = f"{REFACTOR}^"
REVERT_PR = "78afaa5b"  # #6198 remove actor.dump_memory_snapshot
SEPARATION_PR = "84ac897d"  # #6075 remove _fit_torch_memory for separation trainer
WORKER_FILE_OLD = "verl/workers/fsdp_workers.py"
WORKER_FILE_NEW = "verl/workers/engine_workers.py"
CLASSES_OLD = ("ActorRolloutRefWorker", "CriticWorker")
CLASSES_NEW = ("ActorRolloutRefWorker", "TrainingWorker")


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


# ---------------------------------------------------------------- A. 入口面（AST）
def public_methods(repo: str, rev: str, path: str, class_names: tuple[str, ...]) -> dict[str, list[str]]:
    tree = ast.parse(_git(repo, "show", f"{rev}:{path}"))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            out[node.name] = sorted(
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
            )
    return out


def dump_entry_points(repo: str) -> dict:
    old = public_methods(repo, PARENT, WORKER_FILE_OLD, CLASSES_OLD)
    new = public_methods(repo, HEAD, WORKER_FILE_NEW, CLASSES_NEW)
    old_dump = sorted({m for ms in old.values() for m in ms if "snapshot" in m})
    new_dump = sorted({m for ms in new.values() for m in ms if "snapshot" in m})
    return {
        "old_classes": old,
        "new_classes": new,
        "old_snapshot_methods": old_dump,
        "new_snapshot_methods": new_dump,
        "removed_snapshot_methods": sorted(set(old_dump) - set(new_dump)),
    }


# ---------------------------------------------------------------- 配置：真实 yaml 解析
def resolved_cfg(repo: str, tool: str | None = "torch_memory", enable_roles: bool = True):
    """hydra compose 真实 ppo_trainer.yaml + 与 run_ppo 相同的 OmegaConf.resolve。

    `tool=torch_memory` 是官方文档与示例教的用法；传 None 看 yaml 原值。
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = ["global_profiler.steps=[1,2]"]
    if tool is not None:
        overrides.append(f"global_profiler.tool={tool}")
    if enable_roles:
        overrides += [
            "critic.profiler.enable=True",
            "actor_rollout_ref.actor.profiler.enable=True",
        ]
    with initialize_config_dir(config_dir=os.path.join(repo, "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def to_dataclass(prof_cfg):
    from verl.utils.config import omega_conf_to_dataclass

    return omega_conf_to_dataclass(prof_cfg)


def torch_memory_tool_config(profiler_config):
    tc = getattr(profiler_config, "tool_config", None)
    if not tc:
        return None
    return tc.get("torch_memory") if hasattr(tc, "get") else None


# ---------------------------------------------------------------- 探针
class _DumpSpy:
    """替身：只记录"要写哪"，不碰设备、不落盘。"""

    def __init__(self):
        self.calls: list[dict] = []

    def __enter__(self):
        from verl.utils.profiler import torch_memory_profile

        self._mod = torch_memory_profile
        self._orig = torch_memory_profile.MemorySnapshotSampler.dump_memory_snapshot
        calls = self.calls

        def spy(self_sampler, out_dir="./mem_snapshots", tag="snapshot", sub_dir=None, synchronize=True):
            calls.append({"out_dir": out_dir, "tag": tag, "sub_dir": sub_dir})
            return None

        torch_memory_profile.MemorySnapshotSampler.dump_memory_snapshot = spy
        # memory-history 的开在真机上也是 best-effort（无卡静默跳过）；这里显式置位，
        # 免得"没写"被误读成"因为没卡所以没写"。
        torch_memory_profile.TorchMemoryProfiler._memory_history_enabled = True
        torch_memory_profile.TorchMemoryProfiler._oom_observer_attached = True
        return self

    def __exit__(self, *exc):
        self._mod.MemorySnapshotSampler.dump_memory_snapshot = self._orig


def probe_dist_profiler_path(prof, tool_config, role: str | None, step: int = 1) -> dict:
    """探针 1：trainer 的 start_profile/stop_profile → DistProfiler → 后端 stop 写快照。"""
    from verl.utils.profiler.profile import DistProfiler

    with _DumpSpy() as spy:
        try:
            dist = DistProfiler(rank=0, config=prof, tool_config=tool_config, save_file_prefix=role or "")
        except Exception as exc:  # 见探针 3
            return {"error": f"{type(exc).__name__}: {exc}", "dumps": spy.calls}
        kwargs = {"profile_step": step}
        if role is not None:
            kwargs["role"] = role
        dist.start(**kwargs)
        dist.stop(run_command=False)
        return {"impl": type(dist._impl).__name__, "start_kwargs": kwargs, "dumps": spy.calls}


def probe_old_worker_call_shape(prof, tool_config) -> dict:
    """探针 2：旧 worker 的 `dump_memory_snapshot(tag=..., sub_dir=...)` 直接打到同一后端。"""
    from verl.utils.profiler.profile import DistProfiler

    with _DumpSpy() as spy:
        dist = DistProfiler(rank=0, config=prof, tool_config=tool_config)
        impl = dist._impl
        try:
            getattr(impl, "dump_memory_snapshot")(tag="post_update_step1", sub_dir="step1")
            return {"error": None, "dumps": spy.calls}
        except AttributeError as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "dumps": spy.calls}


def probe_target_instantiated(repo: str) -> dict:
    """探针 3：走 `_target_` instantiate（trainer 的 v1 分支 / ActorRolloutRefWorker 的做法）。

    `omega_conf_to_dataclass(cfg.profiler)` 不带 `dataclass_type`，拿到的 `tool_config` 是**容器 dict**
    （键是工具名）；`DistProfiler` 在没显式传 `tool_config` 时会把整个容器当参数交给后端。
    """
    from verl.utils.profiler.profile import DistProfiler

    cfg = resolved_cfg(repo)
    prof = to_dataclass(cfg.critic.profiler)
    tc = getattr(prof, "tool_config", None)
    inner = tc.get(prof.tool) if (tc is not None and getattr(prof, "tool", None)) else None
    out = {
        "tool": getattr(prof, "tool", None),
        "tool_config_container_type": type(tc).__name__ if tc is not None else None,
        "inner_type": type(inner).__name__ if inner is not None else None,
    }
    try:
        dist = DistProfiler(rank=0, config=prof)  # 不传 tool_config ⇒ DistProfiler 回落到 config.tool_config
        out.update(ok=True, impl=type(dist._impl).__name__)
    except Exception as exc:
        out.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="verl 仓库路径（需含 HEAD 与 044bbba2 两个提交）")
    ap.add_argument("--json", help="结果写到该文件")
    ap.add_argument("--selftest", action="store_true", help="注入扰动，验证判据对输入敏感")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)

    report: dict = {
        "repo": args.repo,
        "commits": {
            "head": _git(args.repo, "rev-parse", HEAD).strip(),
            "refactor": _git(args.repo, "rev-parse", REFACTOR).strip(),
            "parent": _git(args.repo, "rev-parse", PARENT).strip(),
            "revert_pr": _git(args.repo, "rev-parse", REVERT_PR).strip(),
            "separation_pr": _git(args.repo, "rev-parse", SEPARATION_PR).strip(),
        },
    }

    report["entry_points"] = dump_entry_points(args.repo)

    cfg = resolved_cfg(args.repo)
    critic_prof = to_dataclass(cfg.critic.profiler)
    actor_prof = to_dataclass(cfg.actor_rollout_ref.actor.profiler)
    n_steps = getattr(critic_prof, "tool_config", {}).get("torch_memory")
    report["config"] = {
        "global_tool": cfg.global_profiler.tool,
        "global_save_path": cfg.global_profiler.save_path,
        "critic": {
            "type": type(critic_prof).__name__,
            "tool": getattr(critic_prof, "tool", None),
            "enable": getattr(critic_prof, "enable", None),
            "save_path": getattr(critic_prof, "save_path", None),
            "tool_config_torch_memory_type": type(n_steps).__name__ if n_steps is not None else None,
            "memory_snapshot_num_steps": getattr(n_steps, "memory_snapshot_num_steps", None),
        },
        "actor": {
            "tool": getattr(actor_prof, "tool", None),
            "enable": getattr(actor_prof, "enable", None),
            "save_path": getattr(actor_prof, "save_path", None),
        },
    }

    p1 = probe_dist_profiler_path(critic_prof, torch_memory_tool_config(critic_prof), role="train")
    p2 = probe_old_worker_call_shape(critic_prof, torch_memory_tool_config(critic_prof))
    p3 = probe_target_instantiated(args.repo)
    report["probes"] = {"dist_profiler_chain": p1, "old_worker_call_shape": p2, "target_instantiate": p3}

    checks: list[dict] = []

    def check(name: str, ok: bool, evidence) -> None:
        checks.append({"name": name, "ok": bool(ok), "evidence": evidence})

    ep = report["entry_points"]
    check("父提交的 worker 类有 dump_memory_snapshot", ep["old_snapshot_methods"] == ["dump_memory_snapshot"], ep["old_snapshot_methods"])
    check("当前 main 的 worker 类没有任何 snapshot 方法", ep["new_snapshot_methods"] == [], ep["new_snapshot_methods"])

    c = report["config"]["critic"]
    check("critic 的 profiler 配置走到了 TorchMemoryProfiler", p1.get("impl") == "TorchMemoryProfiler", p1.get("impl"))
    check(
        "role 级 tool_config 里有 torch_memory 的 dataclass（带 memory_snapshot_num_steps）",
        c["tool_config_torch_memory_type"] == "TorchMemoryToolConfig",
        c["tool_config_torch_memory_type"],
    )
    check("start_profile/stop_profile 这一环**确实能**写出快照", len(p1["dumps"]) == 1, p1["dumps"])
    check(
        "写出的路径是 save_path 下的 step 子目录（不是哨兵 '???'）",
        len(p1["dumps"]) == 1
        and p1["dumps"][0]["sub_dir"] == "step1"
        and p1["dumps"][0]["out_dir"] == cfg.global_profiler.save_path,
        p1["dumps"],
    )
    check(
        "旧 worker 那种显式 dump 调用打在新后端上会 AttributeError（#6198 修的就是它）",
        p2.get("error", "").startswith("AttributeError"),
        p2.get("error"),
    )
    check(
        "走 _target_ instantiate 且不显式传 tool_config 时，容器 dict 被当参数传入 ⇒ 后端构造期 AttributeError",
        p3.get("ok") is False
        and p3.get("tool_config_container_type") == "dict"
        and p3.get("inner_type") == "TorchMemoryToolConfig"
        and "AttributeError" in p3.get("error", ""),
        p3,
    )

    if args.selftest:
        # 扰动 1：不选工具（yaml 原值）时后端应是 _NoOpProfiler、且不写
        cfg0 = resolved_cfg(args.repo, tool=None, enable_roles=False)
        prof0 = to_dataclass(cfg0.critic.profiler)
        p0 = probe_dist_profiler_path(prof0, torch_memory_tool_config(prof0), role=None)
        checks.append(
            {
                "name": "[selftest] 不选工具 ⇒ 后端是 _NoOpProfiler 且不写（证明能写不是探针自说自话）",
                "ok": p0.get("impl") == "_NoOpProfiler" and p0["dumps"] == [],
                "evidence": p0,
            }
        )
        # 扰动 2：把 memory_snapshot_num_steps 调成 2，只走一步就不该写（窗口未满）
        cfg2 = resolved_cfg(args.repo)
        prof2 = to_dataclass(cfg2.critic.profiler)
        tm2 = torch_memory_tool_config(prof2)
        try:
            tm2.memory_snapshot_num_steps = 2
        except Exception:
            object.__setattr__(tm2, "memory_snapshot_num_steps", 2)
        p2b = probe_dist_profiler_path(prof2, tm2, role="train")
        checks.append(
            {
                "name": "[selftest] memory_snapshot_num_steps=2 时单步不写（窗口判据确实在生效）",
                "ok": p2b["dumps"] == [],
                "evidence": p2b["dumps"],
            }
        )

    report["checks"] = checks
    report["n_fail"] = sum(1 for ck in checks if not ck["ok"])
    report["not_covered"] = [
        "快照文件是否真的落盘（CPU 上 MemorySnapshotSampler 直接返回 None）",
        "OOM 观察者是否真的挂到分配器上（需要 CUDA/NPU）",
        "端到端 run_ppo 冒烟（需要 GPU + 数据）",
    ]

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    for ck in checks:
        print(("PASS " if ck["ok"] else "FAIL ") + ck["name"], file=sys.stderr)
    print(f"{len(checks) - report['n_fail']}/{len(checks)} PASS", file=sys.stderr)
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
