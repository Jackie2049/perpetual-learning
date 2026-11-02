#!/usr/bin/env python3
"""Additive timing instrumentation for the gen two-state isolation experiment.

Targets (server-side local patch, NOT for upstream):
  verl/workers/rollout/llm_server.py   -- per-request client-side timing with the
                                          load-balancer server_id (which replica)
  verl/trainer/ppo/ray_trainer.py      -- per-step phase markers around
                                          generate_sequences() and sleep_replicas(),
                                          which both live inside timing_s/gen

Records are appended as JSONL only when M5_GEN_TIMING_DIR is set; otherwise every
inserted line is a no-op (one env lookup). Files are per-pid, so Ray worker
processes and the driver never interleave writes.

Usage (on 4090-1):
  python3 instrument-verl-patch.py apply      # backup + patch + py_compile
  python3 instrument-verl-patch.py revert     # restore backup + verify md5
  python3 instrument-verl-patch.py status
"""

import hashlib
import os
import py_compile
import shutil
import sys

SRC = "/home/zxw/venvs/rollout-g8-src/verl"
BACKUP = "/tmp/m5-gen-timing-backup"
MARK = "M5_GEN_TIMING_DIR"

FILES = [
    "verl/workers/rollout/llm_server.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/trainer/ppo/v1/trainer_base.py",
    "verl/trainer/ppo/v1/trainer_sync.py",
    "verl/trainer/ppo/v1/replay_buffer.py",
]

HELPER = '''

def _m5_log(kind, **kw):
    """M5 local instrumentation: append one JSON line when M5_GEN_TIMING_DIR is set."""
    d = os.environ.get("M5_GEN_TIMING_DIR")
    if not d:
        return
    try:
        kw.update(kind=kind, pid=os.getpid(), wall=time.time())
        path = os.path.join(d, "%s-%d.jsonl" % (kind, os.getpid()))
        with open(path, "a") as f:
            f.write(json.dumps(kw, default=str) + "\\n")
    except Exception:
        pass
'''

# ---------------------------------------------------------------- llm_server.py
LLM_IMPORT_OLD = """import asyncio
import logging
import os
"""
LLM_IMPORT_NEW = """import asyncio
import json
import logging
import os
import time
"""

LLM_HELPER_ANCHOR = 'logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))'

LLM_GEN_OLD = """        server_id, server = await self._acquire_server(request_id)
        try:
            multimodal_kwargs = {}"""
LLM_GEN_NEW = """        _m5_t0 = time.monotonic()
        server_id, server = await self._acquire_server(request_id)
        _m5_ta = time.monotonic()
        _m5_out = None
        try:
            multimodal_kwargs = {}"""

LLM_RET_OLD = """            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            self._release_server(server_id)"""
LLM_RET_NEW = """            output.extra_fields.setdefault("max_global_steps", global_steps)
            _m5_out = output
            return output
        finally:
            self._release_server(server_id)
            _m5_log(
                "m5req",
                t0=_m5_t0,
                ta=_m5_ta,
                t1=time.monotonic(),
                sid=str(server_id),
                rid=str(request_id),
                np=len(prompt_ids),
                nout=(-1 if _m5_out is None else len(_m5_out.token_ids)),
                stop=(None if _m5_out is None else _m5_out.stop_reason),
                pre=(None if _m5_out is None else _m5_out.num_preempted),
            )"""

# --------------------------------------------------------------- ray_trainer.py
TR_IMPORT_OLD = """import json
import os
import uuid
"""
TR_IMPORT_NEW = """import json
import os
import time
import uuid
"""

TR_HELPER_OLD = """from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty"""
TR_HELPER_NEW = """from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
""" + HELPER + """

def apply_kl_penalty"""

TR_GEN_OLD = """                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()"""
TR_GEN_NEW = """                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        _m5_log("m5step", phase="gen_begin", gs=self.global_steps, n=len(combined_gen_batch))
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        _m5_log("m5step", phase="gen_generate_end", gs=self.global_steps)
                        self.checkpoint_manager.sleep_replicas()
                        _m5_log("m5step", phase="gen_end", gs=self.global_steps)"""


# ------------------------------------------------------- v1/trainer_base.py
V1_IMPORT_OLD = """import json
import logging
import math
import os
import uuid
"""
V1_IMPORT_NEW = """import json
import logging
import math
import os
import time
import uuid
"""

V1_HELPER_OLD = """from transfer_queue import KVBatchMeta
"""
V1_HELPER_NEW = """from transfer_queue import KVBatchMeta
""" + HELPER + """
"""

V1_GEN_OLD = """        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()"""
V1_GEN_NEW = """        with marked_timer("gen", timing_raw, color="red"):
            _m5_log("m5v1", phase="gen_begin", gs=self.global_steps)
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            _m5_log("m5v1", phase="sample_end", gs=self.global_steps)
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()
            _m5_log("m5v1", phase="gen_end", gs=self.global_steps)"""

V1_PREP_OLD = """    def prepare_step(self) -> dict:
        self._add_batch_to_generate()
        return {}"""
V1_PREP_NEW = """    def prepare_step(self) -> dict:
        _m5_log("m5v1", phase="prepare_begin", gs=self.global_steps)
        self._add_batch_to_generate()
        _m5_log("m5v1", phase="prepare_end", gs=self.global_steps)
        return {}"""

# ------------------------------------------------------- v1/trainer_sync.py
TS_IMPORT_OLD = """from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer"""
TS_IMPORT_NEW = """from verl.trainer.ppo.v1.trainer_base import PPOTrainer, _m5_log, register_trainer"""

TS_STEP_OLD = """    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)"""
TS_STEP_NEW = """    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            _m5_log("m5v1", phase="update_weights_begin", gs=self.global_steps)
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            _m5_log("m5v1", phase="update_weights_end", gs=self.global_steps)"""

TS_SAMPLE_OLD = """    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()"""
TS_SAMPLE_NEW = """    def on_sample_end(self):
        _m5_log("m5v1", phase="sleep_begin", gs=self.global_steps)
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        _m5_log("m5v1", phase="sleep_end", gs=self.global_steps)"""

# --------------------------------------------------- v1/replay_buffer.py
RB_POLL_OLD = """        self.poll_interval = poll_interval
"""
RB_POLL_NEW = """        self.poll_interval = poll_interval
        if os.environ.get("M5_REPLAY_POLL_INTERVAL"):
            # M5 local intervention knob: shrink the replay-buffer poll quantum.
            self.poll_interval = float(os.environ["M5_REPLAY_POLL_INTERVAL"])
"""

EDITS = {
    "verl/workers/rollout/llm_server.py": [
        (LLM_IMPORT_OLD, LLM_IMPORT_NEW),
        (LLM_HELPER_ANCHOR, LLM_HELPER_ANCHOR + HELPER),
        (LLM_GEN_OLD, LLM_GEN_NEW),
        (LLM_RET_OLD, LLM_RET_NEW),
    ],
    "verl/trainer/ppo/ray_trainer.py": [
        (TR_IMPORT_OLD, TR_IMPORT_NEW),
        (TR_HELPER_OLD, TR_HELPER_NEW),
        (TR_GEN_OLD, TR_GEN_NEW),
    ],
    "verl/trainer/ppo/v1/trainer_base.py": [
        (V1_IMPORT_OLD, V1_IMPORT_NEW),
        (V1_HELPER_OLD, V1_HELPER_NEW),
        (V1_GEN_OLD, V1_GEN_NEW),
        (V1_PREP_OLD, V1_PREP_NEW),
    ],
    "verl/trainer/ppo/v1/trainer_sync.py": [
        (TS_IMPORT_OLD, TS_IMPORT_NEW),
        (TS_STEP_OLD, TS_STEP_NEW),
        (TS_SAMPLE_OLD, TS_SAMPLE_NEW),
    ],
    "verl/trainer/ppo/v1/replay_buffer.py": [
        (RB_POLL_OLD, RB_POLL_NEW),
    ],
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply():
    os.makedirs(BACKUP, exist_ok=True)
    for rel in FILES:
        src = os.path.join(SRC, rel)
        dst = os.path.join(BACKUP, os.path.basename(rel))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        text = open(src).read()
        if all(new in text for _, new in EDITS[rel]):
            print(f"skip (already patched): {rel}")
            continue
        for old, new in EDITS[rel]:
            n = text.count(old)
            assert n == 1, f"{rel}: anchor found {n} times, expected 1: {old[:60]!r}"
            text = text.replace(old, new)
        open(src, "w").write(text)
        py_compile.compile(src, doraise=True)
        print(f"patched {rel}  md5={md5(src)}  backup_md5={md5(dst)}")


def revert():
    ok = True
    for rel in FILES:
        src = os.path.join(SRC, rel)
        dst = os.path.join(BACKUP, os.path.basename(rel))
        if not os.path.exists(dst):
            print(f"NO BACKUP for {rel} -- refusing to touch")
            ok = False
            continue
        shutil.copy2(dst, src)
        py_compile.compile(src, doraise=True)
        print(f"reverted {rel}  md5={md5(src)}")
    print("revert completed" if ok else "revert INCOMPLETE")


def status():
    for rel in FILES:
        src = os.path.join(SRC, rel)
        dst = os.path.join(BACKUP, os.path.basename(src))
        text = open(src).read()
        print(
            f"{rel}: patched={'_m5_log' in text} md5={md5(src)} "
            f"backup={'missing' if not os.path.exists(dst) else md5(dst)}"
        )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"apply": apply, "revert": revert, "status": status}[cmd]()
