#!/usr/bin/env python3
"""Does `run-critic-pdbg.sh`'s override list actually compose on the tree under test?

The critic arm has not run on a GPU yet. Its three runs differ only in
`precision_debugger.stages`, so a single bad override key would kill all three at
Hydra composition — before any training step, i.e. the GPU window would be wasted
without a reading. This script composes the real `ppo_trainer` config with the real
override list (CPU only, no verl import) and records what the keys resolve to.

Two checks per arm:
  * compose succeeds and the keys the probe cares about carry the intended values
  * negative control: an invented key (`critic.fsdp.no_such_field`) must FAIL
    composition — otherwise "compose ok" would not mean anything

Usage:
  compose_check.py --tree /tmp/verl-3fdf2b5 --out raw/compose-check.json
  compose_check.py --selftest          # proves the negative control fires
"""

import argparse
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Overrides in the ORDER run-critic-pdbg.sh passes them: base recipe arrays first
# (examples/grpo_trainer/run_qwen3_4b_fsdp.sh), then EXTRA, then the trailing "$@" args.
# Order is load-bearing: the base recipe sets adv_estimator=grpo and EXTRA sets =gae,
# and Hydra lets the last occurrence win.
BASE_RECIPE = [
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.actor.use_dynamic_bsz=True",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000",
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    'trainer.logger=\'["console","wandb"]\'',
    "trainer.critic_warmup=0",
]

PROBE_EXTRA = [
    "global_profiler.tool=precision_debugger",
    "global_profiler.steps=[1]",
    "global_profiler.save_path=/tmp/m5-pdbg-critic/prof-cu",
    "global_profiler.global_tool_config.precision_debugger.config_path=/tmp/m5-msprobe/config.json",
    "global_profiler.global_tool_config.precision_debugger.strict=False",
    "actor_rollout_ref.actor.profiler.enable=True",
    "critic.profiler.enable=True",
    "algorithm.adv_estimator=gae",  # later than the base recipe's grpo -> this is the one in effect
    "critic.enable=True",
    "critic.model.path=/home/zxw/models/Qwen2.5-0.5B-Instruct",
    "critic.ppo_mini_batch_size=8",
    "critic.ppo_micro_batch_size_per_gpu=1",
    "critic.ppo_max_token_len_per_gpu=3000",
    "critic.fsdp.optimizer_offload=True",
    "critic.fsdp.model_dtype=bf16",
    "trainer.total_training_steps=2",
    "++transfer_queue.backend.SimpleStorage.total_storage_size=100000",
]

PROBE_ARGS = [
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256",
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
    "+actor_rollout_ref.rollout.max_model_len=2048",
    'trainer.logger=["console"]',
    "trainer.default_local_dir=/tmp/m5-pdbg-critic/ckpt-XX",
]

COMMON = BASE_RECIPE + PROBE_EXTRA + PROBE_ARGS

ARMS = {"au": "[actor_update]", "cu": "[critic_update]", "cv": "[compute_values]"}

# What the probe's claims depend on; read back after composition.
READBACK = [
    "global_profiler.tool",
    "global_profiler.steps",
    "global_profiler.global_tool_config.precision_debugger.stages",
    "global_profiler.global_tool_config.precision_debugger.strict",
    "actor_rollout_ref.actor.profiler.enable",
    "critic.profiler.enable",
    "algorithm.adv_estimator",
    "critic.enable",
    "critic.model.path",
    "critic.fsdp.model_dtype",
    "critic.fsdp.optimizer_offload",
    "critic.ppo_mini_batch_size",
    "critic.ppo_max_token_len_per_gpu",
    "critic.forward_max_token_len_per_gpu",
    "critic.use_dynamic_bsz",
    "trainer.total_training_steps",
    "actor_rollout_ref.model.override_config.attn_implementation",
]

NEGATIVE_CONTROL = "critic.fsdp.no_such_field=1"


def git_commit(tree):
    """Record which commit was composed -- a worktree path alone is not a version."""
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "<unknown>"


def compose_arm(config_dir, overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def readback(cfg):
    from omegaconf import OmegaConf

    out = {}
    for key in READBACK:
        node = cfg
        try:
            for part in key.split("."):
                node = node[part]
            if OmegaConf.is_config(node):
                node = OmegaConf.to_container(node, resolve=True)
            out[key] = node
        except Exception as exc:  # missing key is a reading, not a crash
            out[key] = f"<missing: {type(exc).__name__}>"
    return out


def run_arm(config_dir, arm):
    overrides = COMMON + [f"global_profiler.global_tool_config.precision_debugger.stages={ARMS[arm]}"]
    overrides = [o.replace("prof-cu", f"prof-{arm}").replace("ckpt-XX", f"ckpt-{arm}") for o in overrides]
    res = {"arm": arm, "stages": ARMS[arm], "composed": False}
    try:
        cfg = compose_arm(config_dir, overrides)
        res["composed"] = True
        res["readback"] = readback(cfg)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    return res


def run_negative_control(config_dir, overrides):
    try:
        compose_arm(config_dir, overrides + [NEGATIVE_CONTROL])
        return {"fired": False, "note": "composed with an invented key -> check is blind"}
    except Exception as exc:
        return {"fired": True, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", help="verl checkout to compose (e.g. /tmp/verl-3fdf2b5)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        # A throwaway config dir with a typed field proves the negative control fires
        # without needing a verl checkout: one clean override must compose, one invented
        # key must not.
        import tempfile

        import yaml

        tmp = tempfile.mkdtemp()
        Path(tmp, "ppo_trainer.yaml").write_text(
            yaml.safe_dump({"critic": {"fsdp": {"model_dtype": "fp32", "optimizer_offload": False}}})
        )
        clean_overrides = ["critic.fsdp.model_dtype=bf16"]
        clean_ok = True
        try:
            cfg = compose_arm(tmp, clean_overrides)
            clean_ok = cfg.critic.fsdp.model_dtype == "bf16"
        except Exception:
            clean_ok = False
        bad = run_negative_control(tmp, clean_overrides)
        ok = bool(bad["fired"] and clean_ok)
        print(json.dumps({"clean_override_composes": clean_ok,
                          "negative_control_fires": bad["fired"],
                          "negative_control_error": bad.get("error", "")[:120],
                          "ok": ok}, indent=2))
        return 0 if ok else 1

    if not args.tree:
        ap.error("--tree is required unless --selftest")

    config_dir = str(Path(args.tree).resolve() / "verl" / "trainer" / "config")
    if not Path(config_dir, "ppo_trainer.yaml").exists():
        print(f"no ppo_trainer.yaml under {config_dir}", file=sys.stderr)
        return 2

    import hydra
    import omegaconf

    result = {
        "env": {"python": sys.version.split()[0], "hydra": hydra.__version__,
                "omegaconf": omegaconf.__version__},
        "tree": str(Path(args.tree).resolve()),
        "commit": git_commit(args.tree),
        "config_dir": config_dir,
        "arms": [run_arm(config_dir, a) for a in ARMS],
        "negative_control": run_negative_control(config_dir, COMMON),
    }
    result["all_composed"] = all(a["composed"] for a in result["arms"])
    result["ok"] = result["all_composed"] and result["negative_control"]["fired"]

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
