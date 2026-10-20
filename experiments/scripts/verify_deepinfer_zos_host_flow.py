"""Run original DeepInfer host control flow with fake device/event boundaries.

Usage: python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer
Baseline: 8b16149741a4935e1dec8cea27853f273b374c5b.
PR #54 event-pool model (head 70381d28): add --check-event-pool.
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
from contextlib import contextmanager
from collections import defaultdict
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


def load_worker(path, log, env, tick, timer=None, event_factory=None):
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
        "cuda_timer": timer if timer is not None else NS(latest_event_pair=lambda *args: None),
        "torch": NS(cuda=NS(Event=event_factory if event_factory is not None else lambda **kwargs: Event(log, tick[0]))),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    print(f"worker.py sha256={hashlib.sha256(source.encode()).hexdigest()}")
    return namespace["Worker"]()


def check_event_pool(root):
    """PR #54 adversarial host model: current forward is already complete at flush.

    Real Timer/Worker/engine timing-reader AST is retained. Device completion,
    durations, model calls and communication are explicit test doubles.
    """
    timer_path = root / "deepinfer/timer.py"
    engine_path = root / "deepinfer/engine/llm_engine.py"
    timer_nodes = [n for n in ast.parse(timer_path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name in {"BaseTimer", "CUDATimer"}]
    assert len(timer_nodes) == 2
    engine = next(n for n in ast.parse(engine_path.read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == "LLMEngine")
    post = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "post_process")
    branch = next(n for n in post.body if isinstance(n, ast.If) and "forward_events" in ast.unparse(n.test))
    reader = ast.parse("def read_runtime(self, step_state):\n    pass").body[0]
    seq_binding = next(n for n in post.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "seq_metadata_list" for t in n.targets))
    reader.body = [seq_binding, branch]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    reader_ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, reader], type_ignores=[])), str(engine_path), "exec"), reader_ns)

    def scenario(reuse):
        device = NS(ms=0.0)
        completions = []

        class DeviceEvent:
            def __init__(self, enable_timing=False):
                self.enable_timing = enable_timing
                self.generation = self.waited = 0
                self.stamp = None
                if not enable_timing:
                    completions.append(self)

            def record(self):
                if not self.enable_timing and self.generation:
                    assert self.waited == self.generation, "completion marker re-recorded before consumption"
                self.generation += 1
                self.stamp = device.ms

            def query(self):
                return True  # injected valid schedule: all submitted work has completed

            def synchronize(self):
                self.waited = self.generation

            def elapsed_time(self, other):
                return other.stamp - self.stamp

        namespace = {"defaultdict": defaultdict, "contextmanager": contextmanager,
                     "torch": NS(cuda=NS(Event=DeviceEvent))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *timer_nodes], type_ignores=[])), str(timer_path), "exec"), namespace)
        timer = namespace["CUDATimer"]()
        assert hasattr(timer, "_event_pool"), "this mode expects PR #54 event-pool source"
        if not reuse:
            # Explicit test control, not an upstream fix or a modified source body.
            timer._acquire_event_pair = lambda: (DeviceEvent(True), DeviceEvent(True))
        env = NS(DEEPINFER_ZERO_OVERHEAD_SCHEDULING=True, DEEPINFER_ENABLE_MTP_VARLEN=True)
        tick = [0]
        worker = load_worker(root / "deepinfer/worker/worker.py", [], env, tick, timer, DeviceEvent)
        worker.is_driver_worker, worker.nextn = True, 5
        worker.prev_step_state, worker._sample_end_evt, worker.gpu_cache = None, None, None
        worker._correct_seqlens_qk = lambda *args: None
        worker.maybe_gc_collect = lambda: None
        observed = []

        def forward(*args, **kwargs):
            with timer.record("device_forward"):
                device.ms += 100 * tick[0]
            return NS(graph_size=8, input_metadata=None, model_output=None)

        def sample(*args, **kwargs):
            with timer.record("device_sample"):
                device.ms += 10 * tick[0]
            return NS(tick=tick[0])

        def post_process(state):
            reader_ns["read_runtime"](NS(scheduler=NS(update_mtp_runtime=lambda n, g, f, s: observed.append((state.sampler_output.tick, f, s)))), state)
            return np.array([state.sampler_output.tick], dtype=np.uint64)

        worker.model_runner = NS(execute_main_model=forward, execute_sample=sample,
                                 post_process=lambda *args: timer.flush_out(non_blocking=True))
        worker.engine_callbacks = NS(post_process_fn=post_process, pre_sampler_sync_fn=lambda *args: None)
        scheduler = NS(num_batched_tokens=6, num_active_so=0)
        for step in (1, 2, 3):
            tick[0] = step
            worker.execute_model_step(False, False, [NS(seqlen_q=1)], scheduler)
        worker.flush_inflight_reqs()
        assert len(completions) == 1 and completions[0].generation == completions[0].waited == 3
        return observed

    expected = [(1, 100.0, 10.0), (2, 200.0, 20.0), (3, 300.0, 30.0)]
    pooled, fresh = scenario(True), scenario(False)
    assert fresh == expected, ("fresh-event control failed", fresh)
    assert pooled != expected and pooled[0] == expected[0], ("counterexample not reproduced", pooled)
    for path in (timer_path, engine_path):
        print(f"{path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"expected timing feedback: {expected}")
    print(f"PR pooled feedback:      {pooled}")
    print(f"reuse-disabled control:  {fresh}")
    print("PASS: host-model counterexample reproduced; completion marker consumed before re-record in both scenarios.")
    print("Not a CUDA runtime, model, performance or device-race reproduction; no upstream source changed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--check-event-pool", action="store_true", help="PR #54 timing-alias host model; no device execution")
    args = parser.parse_args()
    root = args.source_root
    if args.check_event_pool:
        check_event_pool(root)
        return
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
