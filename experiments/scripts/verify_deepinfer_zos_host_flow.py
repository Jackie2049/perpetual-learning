"""Run original DeepInfer host control flow with fake device/event boundaries.

Usage: python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer
Baseline: 8b16149741a4935e1dec8cea27853f273b374c5b.
Requires Python 3.9+ and numpy; no torch.
Three Worker method bodies are compiled unchanged from AST, with decorators
removed and hardware dependencies replaced by explicit doubles. Metadata
correction is injected, not executed. This cannot validate CUDA completion,
stream ordering, allocator lifetimes, kernels, or distributed correctness.
"""
import argparse
import ast
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


class Event:
    def __init__(self, log, label):
        self.log, self.label = log, label

    def record(self):
        self.log.append(f"record:{self.label}")

    def synchronize(self):
        self.log.append(f"wait:{self.label}")


def load_worker(path, log, env, tick):
    source = path.read_text()
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "Worker")
    names = {"execute_model_step", "_post_process_step", "flush_inflight_reqs"}
    methods = [node for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    cls.body, cls.bases, cls.keywords, cls.decorator_list = methods, [], [], []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0), cls,
    ], type_ignores=[])
    namespace = {
        "np": np, "envs": env, "Movable": lambda x: x,
        "WorkerStepState": NS,
        "cpu_timer": NS(record=lambda *args, **kwargs: nullcontext()),
        "cuda_graph_timer": NS(flush_out=lambda: None),
        "cuda_timer": NS(latest_event_pair=lambda *args: None),
        "torch": NS(cuda=NS(Event=lambda **kwargs: Event(log, tick[0]))),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    print(f"worker.py sha256={hashlib.sha256(source.encode()).hexdigest()}")
    return namespace["Worker"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    root = parser.parse_args().source_root
    log, tick = [], [0]
    env = NS(DEEPINFER_ZERO_OVERHEAD_SCHEDULING=True,
             DEEPINFER_ENABLE_MTP_VARLEN=False)
    worker = load_worker(root / "deepinfer/worker/worker.py", log, env, tick)
    worker.is_driver_worker, worker.prev_step_state, worker.nextn = True, None, 5
    worker.gpu_cache = None
    correction = [None]
    worker._correct_seqlens_qk = lambda *args: correction[0]
    worker.maybe_gc_collect = lambda: None

    def forward(*args, **kwargs):
        log.append(f"forward:{tick[0]}")
        return NS(graph_size=0, input_metadata=None, model_output=None)

    def sample(*args):
        log.append(f"sample:{tick[0]}")
        return NS(tick=tick[0])

    def post(state):
        step = state.sampler_output.tick
        log.append(f"post:{step}")
        return np.array([] if state.is_placeholder else [step], dtype=np.uint64)

    worker.model_runner = NS(
        execute_main_model=forward, execute_sample=sample,
        _placeholder_seq_metadata=lambda prompt: NS(seqlen_q=1),
        post_process=lambda *args: None,
    )
    worker.engine_callbacks = NS(
        post_process_fn=post,
        pre_sampler_sync_fn=lambda *args: log.append(f"release:{tick[0]}"),
    )
    scheduler = NS(
        num_batched_tokens=6,
        update_metadata_for_sampling_zos=lambda *args: log.append(f"metadata:{tick[0]}"),
    )

    def run(step, prompt=False, placeholder=False):
        tick[0] = step
        return worker.execute_model_step(
            prompt, placeholder, [NS(seqlen_q=1)], scheduler
        ).tolist()

    def before(first, second):
        assert log.index(first) < log.index(second), (first, second, log)

    assert run(1) == [] and worker.prev_step_state is not None
    assert "wait:1" not in log
    assert run(2) == [1]
    before("forward:2", "wait:1")
    before("wait:1", "post:1")
    before("post:1", "sample:2")
    before("sample:2", "release:2")
    before("release:2", "record:2")
    print("ZOS decode: current forward precedes previous wait/post; current is retained.")

    assert run(3, prompt=True) == [2, 3]
    assert worker.prev_step_state is None
    before("sample:3", "wait:3")
    assert run(4) == []
    assert worker.flush_inflight_reqs().tolist() == [4]
    previous_log = list(log)
    assert worker.flush_inflight_reqs().tolist() == [] and log == previous_log
    env.DEEPINFER_ZERO_OVERHEAD_SCHEDULING = False
    assert run(5) == [5] and worker.prev_step_state is None
    before("sample:5", "wait:5")
    print("Prompt/non-ZOS post immediately; flush clears pending state and is idempotent.")

    env.DEEPINFER_ZERO_OVERHEAD_SCHEDULING = True
    assert run(6) == []
    assert run(7, placeholder=True) == [6]
    assert worker.prev_step_state.is_placeholder
    assert worker.flush_inflight_reqs().tolist() == []
    assert worker.prev_step_state is None
    print("Placeholder forward drains prior real output; placeholder itself emits none.")

    assert run(8) == []
    env.DEEPINFER_ENABLE_MTP_VARLEN = True
    correction[0] = NS(
        metadata_zos=NS(remap_cpu=None), seqlens_q_cpu=None,
        d2h_evt=Event(log, "length9"),
    )
    assert run(9) == [8]
    before("post:8", "wait:length9")
    before("wait:length9", "metadata:9")
    before("metadata:9", "sample:9")
    assert worker.flush_inflight_reqs().tolist() == [9]
    print("Injected length correction: wait D2H, refresh metadata, then sample current.")
    print("PASS: host ordering and lifecycle scenarios. GPU events are doubles.")


if __name__ == "__main__":
    main()
