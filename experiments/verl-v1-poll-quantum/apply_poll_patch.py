#!/usr/bin/env python3
"""Make ``poll_interval`` reachable from config and decouple the idle heuristic.

Local evidence patch for TM-008 (verl V1 replay buffer sample-wait poll).
Applies to a copy of the verl tree; every replacement must match exactly once.
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1]).resolve()

EDITS = [
    (
        "verl/trainer/config/ppo_trainer.yaml",
        "      # Custom sampler config\n      custom_sampler:\n",
        "      # Poll interval (seconds) for the built-in replay buffer's sample wait.\n"
        "      # Lower values reduce sample-detection latency at the cost of more\n"
        "      # TransferQueue metadata polls. Ignored by custom samplers.\n"
        "      poll_interval: 2.0\n"
        "\n"
        "      # Custom sampler config\n      custom_sampler:\n",
    ),
    (
        "verl/trainer/ppo/v1/trainer_base.py",
        "            replay_buffer_kwargs.update(\n"
        "                filter_groups_metric=filter_groups_metric,\n"
        "                sync_refill_failed_groups=sync_refill_failed_groups,\n"
        "            )\n",
        "            replay_buffer_kwargs.update(\n"
        "                poll_interval=sampler_config.get(\"poll_interval\", 2.0),\n"
        "                filter_groups_metric=filter_groups_metric,\n"
        "                sync_refill_failed_groups=sync_refill_failed_groups,\n"
        "            )\n",
    ),
    (
        "verl/trainer/ppo/v1/trainer_separate_async.py",
        "        poll_interval = getattr(self.replay_buffer, \"poll_interval\", 2.0)\n"
        "        return self._step_sample_wait_seconds > poll_interval\n",
        "        # A shorter sample poll must not turn every short wait into an idle step,\n"
        "        # so the idle heuristic keeps a floor independent of the poll cadence.\n"
        "        poll_interval = getattr(self.replay_buffer, \"poll_interval\", 2.0)\n"
        "        return self._step_sample_wait_seconds > max(poll_interval, 0.5)\n",
    ),
]


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


for rel, old, new in EDITS:
    path = ROOT / rel
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"FAIL {rel}: anchor matched {count} times")
    before = md5(path)
    path.write_text(text.replace(old, new, 1))
    print(f"OK {rel}: {before} -> {md5(path)}")

print("patch applied")
