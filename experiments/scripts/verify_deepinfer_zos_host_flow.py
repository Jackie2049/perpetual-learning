"""Run original DeepInfer host control flow with fake device/event boundaries.

Usage: python3 experiments/scripts/verify_deepinfer_zos_host_flow.py /path/to/DeepInfer
Baseline: 8b16149741a4935e1dec8cea27853f273b374c5b.
PR #54 event-pool model (head 70381d28): add --check-event-pool.
PR #54 structured-output count model: add --check-so-counter.
PR #54 output-array ownership probe: add --check-output-copy --pybind-include
/path/to/pybind11/include [--python-include /path/to/Python/Headers].
This mode compiles a tiny CPU extension, not DeepInfer or its output pool.
CP lifecycle/control-plane model: add --check-worker-lifecycle (no Linux/device execution).
Native shared-memory cleanup: add --check-shm-cleanup (torch zero-fill adapter;
not the full upstream Gloo/ZMQ test suite).
Requires Python 3.9+ and numpy; no torch.
Three Worker method bodies are compiled unchanged from AST, with decorators
removed and hardware dependencies replaced by explicit doubles. Metadata
correction is injected, not executed. This cannot validate CUDA completion,
stream ordering, allocator lifetimes, kernels, or distributed correctness.
"""
import argparse
import ast
import hashlib
import importlib.util
import mmap
import pickle
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import nullcontext
from contextlib import contextmanager
from contextlib import suppress
from collections import defaultdict
from itertools import product
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch

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


def check_so_counter(root):
    """Original prefill promotion and sampler bodies; scheduler/ledger are models."""
    engine_path = root / "deepinfer/engine/llm_engine.py"
    sampler_path = root / "deepinfer/model_executor/layers/sampler.py"
    scheduler_path = root / "csrc/engine/scheduler.hpp"
    # Guard the exact snapshot predicate assumed by the model, not C++ execution.
    assert re.search(r"if\s*\(seq->structured_output\.status\s*>\s*0\)\s*\{\s*"
                     r"scheduler_outputs\.num_active_so\s*\+=\s*1;\s*\}", scheduler_path.read_text())
    engine = next(n for n in ast.parse(engine_path.read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == "LLMEngine")
    step = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    promotion = next(n for n in ast.walk(step) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "scheduler_outputs.prompt_run"
                     and any(isinstance(k, ast.AugAssign) and isinstance(k.target, ast.Attribute)
                             and k.target.attr == "num_active_so" for k in ast.walk(n)))
    wrapper = ast.parse("def promote(scheduler_outputs, seq_metadata_list):\n    pass").body[0]
    wrapper.body = [promotion]
    sampler = next(n for n in ast.parse(sampler_path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "Sampler")
    get_mask = next(n for n in sampler.body if isinstance(n, ast.FunctionDef) and n.name == "_get_so_mask")
    get_mask.decorator_list = []

    class Tensor:
        def __init__(self, data=None, shape=None):
            self.data = data
            self.shape = shape
        def cuda(self, **kwargs):
            return self
        def record_stream(self, stream):
            pass

    class Marker:
        def record(self, *args):
            pass

    class Matcher:
        def __init__(self, aborted=False):
            self.aborted = aborted
        def is_aborted(self):
            return self.aborted
        def constrained(self):
            return True
        def fill_vocab_mask(self, *args):
            return True
        def accept_token(self, token):
            return True
        def rollback(self, n):
            pass
        def set_aborted(self, reason):
            self.aborted = True

    stream = NS(wait_event=lambda *args: None)
    def checked_zip(a, b, strict=False):
        if strict and len(a) != len(b):
            raise ValueError("zip length")
        return zip(a, b)

    namespace = {
        "zip": checked_zip,
        "SequenceStatus": NS(FINISHED_SO_FAILED="failed"),
        "logger": NS(error=lambda *args: None, warning=lambda *args: None, exception=lambda *args: None),
        "allocate_vocab_mask": lambda n, v, **kwargs: Tensor(shape=(n, v)),
        "torch": NS(int32="int32", tensor=lambda data, **kwargs: Tensor(list(data)),
                    cuda=NS(current_stream=lambda: stream, Event=Marker, Stream=lambda: stream, stream=lambda s: nullcontext())),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, wrapper], type_ignores=[])), str(engine_path), "exec"), namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, get_mask], type_ignores=[])), str(sampler_path), "exec"), namespace)

    def mask(metadata, count):
        state = NS(bitmasks_cpu=[], vocab_size=32, max_MTP_value=2, so_copy_stream=None)
        indices, bitmasks, statuses, event = namespace["_get_so_mask"](state, metadata, count)
        return None if indices is None else [t.data for t in indices], statuses

    # C++ schedule's snapshot predicate and allowed ledger transitions are an
    # explicit model, not an execution/proof of the C++ scheduler or pybind.
    successors = {-1: (-1,), 0: (0,), 1: (-1, 1, 2), 2: (-1, 2)}
    cases = overestimates = 0
    for n in (1, 2, 3):
        for snapshot in product((-1, 0, 1, 2), repeat=n):
            for refreshed in product(*(successors[s] for s in snapshot)):
                for policy in ("decode", "prompt-partial", "prompt-final", "missing-matcher", "no-so", "mixed"):
                    metadata, seqs = [], []
                    before = sum(s > 0 for s in snapshot)
                    for i, status in enumerate(refreshed):
                        request = f"r{i}"
                        last = policy not in ("decode", "prompt-partial") and (policy != "mixed" or i % 2 == 0)
                        so_args = None if policy == "no-so" else object()
                        matcher = None if policy == "missing-matcher" else Matcher()
                        metadata.append(NS(request_id=request, so_status=status, so_matcher=matcher,
                                           is_last_prompt_chunk=last, sampling_params=NS(so_args=so_args),
                                           last_output_token=-1 if policy == "prompt-partial" else 7,
                                           last_accepted_token=[3], last_next_n_token=[4, 5]))
                        seqs.append(NS(request_id=request, status="running"))
                    output = NS(prompt_run=policy != "decode", scheduled_seqs=seqs, num_active_so=before)
                    namespace["promote"](output, metadata)
                    active = sum(m.so_status > 0 for m in metadata)
                    assert output.num_active_so >= active, (snapshot, refreshed, policy)
                    overestimates += output.num_active_so > active
                    indices, statuses = mask(metadata, output.num_active_so)
                    legacy_indices, legacy_statuses = mask(metadata, None)
                    if active:
                        assert (indices, statuses) == (legacy_indices, legacy_statuses)
                    else:
                        assert indices is None or not any(indices), "inactive batch unexpectedly gets masks"
                        if statuses is not None:
                            assert all(s <= 0 for s in statuses)
                    cases += 1

    # Sensitivity check: omission of the required 0->1 increment is unsafe.
    item = NS(request_id="fault", so_status=0, so_matcher=Matcher(), is_last_prompt_chunk=True,
              sampling_params=NS(so_args=object()), last_output_token=7, last_accepted_token=[], last_next_n_token=[])
    output = NS(prompt_run=True, scheduled_seqs=[NS(request_id="fault", status="running")], num_active_so=0)
    namespace["promote"](output, [item])
    assert item.so_status == 1 and output.num_active_so == 1
    assert mask([item], output.num_active_so)[0][0] == [0]
    assert mask([item], 0) == (None, None), "fault-injected stale zero did not expose skipped mask"
    aborted = NS(**{**vars(item), "so_matcher": Matcher(aborted=True), "so_status": 2})
    assert mask([aborted], 1)[1] == [-1]
    assert mask(None, 0) == mask(None, 1) == mask([], 0) == (None, None)
    assert overestimates > 0
    for path in (engine_path, sampler_path, scheduler_path):
        print(f"{path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"PASS SO upper-bound model: {cases} cases; {overestimates} conservative overestimates; stale-zero fault skips an active mask.")
    print("Scope: original Python promotion/mask bodies; mocked matcher/tensors/events; scheduler snapshot and ledger transitions modeled, not executed.")


def check_output_copy(root, pybind_include, python_include, cxx, cxx_flags):
    """Probe the exact return expression, not the full torch-dependent builder.

    The C++ vector contains integer identities, never real RequestOutput pointers.
    An explicit capsule-backed array is the genuinely borrowed negative control.
    Python collect/release methods are original AST, with pools replaced by doubles.
    """
    engine_path = root / "deepinfer/engine/llm_engine.py"
    outputs_path = root / "csrc/engine/outputs.hpp"
    source = outputs_path.read_text()
    returns = re.findall(r"return (py::array_t<uint64_t>\(request_outputs\.size\(\), request_outputs\.data\(\)\));", source)
    assert len(returns) == 1, "reassess the builder's constructor before using this probe"
    header = pybind_include / "pybind11/numpy.h"
    assert header.is_file() and (python_include / "Python.h").is_file()
    for path in (outputs_path, engine_path, header):
        print(f"{path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    cls = next(n for n in ast.parse(engine_path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "DeferredRelease")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"collect_request_outputs", "release"}]
    assert len(cls.body) == 2
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"np": np, "free_deferred_pool_handles": lambda: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(engine_path), "exec"), namespace)

    cpp = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <vector>
namespace py = pybind11;
static std::vector<uint64_t> request_outputs(32);
py::array_t<uint64_t> build(int count, uint64_t start, bool borrowed) {
    if (count < 0 || count > 32) throw py::value_error("fixture capacity");
    request_outputs.clear();
    for (int i = 0; i < count; ++i) request_outputs.emplace_back(start + i);
    if (borrowed) {
        // Explicit control: static storage never reallocates, capsule does not free it.
        auto owner = py::capsule(request_outputs.data(), [](void *) {});
        return py::array_t<uint64_t>(request_outputs.size(), request_outputs.data(), owner);
    }
    return ACTUAL_RETURN;
}
PYBIND11_MODULE(_deepinfer_array_probe, m) {
    m.def("build", &build);
    m.def("storage_address", []() { return reinterpret_cast<uint64_t>(request_outputs.data()); });
}
'''.replace("ACTUAL_RETURN", returns[0])
    with tempfile.TemporaryDirectory(prefix="deepinfer-array-probe-") as temp:
        # Generated test artifact, not an edit to upstream sources.
        fixture = Path(temp) / "probe.cpp"
        fixture.write_text(cpp)
        extension = Path(temp) / ("_deepinfer_array_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
        command = [cxx, "-std=c++17", "-O0", "-shared", "-fPIC", "-I" + str(pybind_include),
                   "-I" + str(python_include), str(fixture), "-o", str(extension)]
        if sys.platform == "darwin":
            command.extend(["-undefined", "dynamic_lookup"])
        command.extend(cxx_flags)
        print("compile:", " ".join(command))
        subprocess.run(command, check=True, timeout=60)
        spec = importlib.util.spec_from_file_location("_deepinfer_array_probe", extension)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        first = probe.build(3, 100, False)
        second = probe.build(3, 200, False)
        assert first.flags.owndata and first.base is None
        assert first.ctypes.data != probe.storage_address()
        assert not np.shares_memory(first, second)
        assert first.tolist() == [100, 101, 102]
        borrowed = probe.build(3, 300, True)
        snapshot = borrowed.copy()
        probe.build(3, 400, False)
        assert not borrowed.flags.owndata
        assert borrowed.ctypes.data == probe.storage_address()
        assert borrowed.tolist() == [400, 401, 402]
        assert snapshot.tolist() == [300, 301, 302]
        print("No-base constructor owns a copy; capsule control aliases static storage.")

        cases = 0
        for borrowed_input, n, m in product((False, True), range(5), range(5)):
            released = []
            collector = namespace["DeferredRelease"]()
            collector.request_outputs_to_release = np.empty(0, dtype=np.uint64)
            collector.sequence_metadata_to_release = []
            collector.clone_unpins_to_release = None
            collector.request_output_pool = NS(release=lambda a: released.append(a.tolist()))
            expected = list(range(100, 100 + n)) + list(range(200, 200 + m))
            for count, start in ((n, 100), (m, 200)):
                array = probe.build(count, start, borrowed_input)
                old = collector.request_outputs_to_release
                collector.collect_request_outputs(array)
                if not count:
                    assert collector.request_outputs_to_release is old
                elif not old.size:
                    assert not np.shares_memory(collector.request_outputs_to_release, array)
            probe.build(4, 900, False)  # overwrite storage before deferred release
            assert collector.request_outputs_to_release.tolist() == expected
            collector.release()
            assert released == [expected] and collector.request_outputs_to_release.size == 0
            collector.collect_request_outputs(probe.build(2, 500, borrowed_input))
            probe.build(2, 600, False)
            collector.release()
            assert released[-1] == [500, 501]
            cases += 1
        print(f"PASS output-copy probe: {cases} owned/borrowed collection, empty/concat/release/reset cases.")
        print("No DeepInfer build, torch, actual pointers, destructor, device or performance measurement.")


def check_worker_lifecycle(root):
    """Run original control-plane bodies with in-memory ring/clock/resource doubles.

    This checks ordering, dispatch and timeout forwarding, not shared-memory
    visibility, concurrent processes, prctl, collectives or kernel cleanup.
    """
    base_path = root / "deepinfer/worker/base.py"
    parallel_path = root / "deepinfer/model_executor/parallel_utils/parallel_state.py"
    queue_path = root / "deepinfer/distributed/shm_broadcast.py"
    trees = {}
    for path in (base_path, parallel_path, queue_path):
        raw = path.read_bytes()
        trees[path] = ast.parse(raw)
        print(f"{path.name} sha256={hashlib.sha256(raw).hexdigest()}")
    base = next(n for n in trees[base_path].body if isinstance(n, ast.ClassDef) and n.name == "WorkerBase")
    init = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    stages = [ast.unparse(n.func) for n in sorted(ast.walk(init), key=lambda n: getattr(n, "lineno", 0))
              if isinstance(n, ast.Call) and ast.unparse(n.func) in
              {"self._init_torch", "self.init_model", "self.init_dist", "self.start_slave_worker_loop"}]
    assert stages == ["self._init_torch", "self.init_model", "self.init_dist", "self.start_slave_worker_loop"]
    print("Static initializer order:", " -> ".join(stages))
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"start_slave_worker_loop", "stop", "run_workers"}]
    assert len(base.body) == 3
    queue = next(n for n in trees[queue_path].body if isinstance(n, ast.ClassDef) and n.name == "MessageQueue")
    queue.body = [n for n in queue.body if isinstance(n, ast.FunctionDef)
                  and n.name in {"_spin_wait", "acquire_write", "acquire_read", "enqueue", "dequeue"}]
    assert len(queue.body) == 5
    broadcast = next(n for n in trees[parallel_path].body
                     if isinstance(n, ast.FunctionDef) and n.name == "cp_broadcast_object")
    trace, clock = [], NS(now=0, bound=False)

    class ProbeExit(Exception):
        pass

    class ProbeBound(Exception):
        pass

    def fake_exit(code):
        trace.append(("exit", code))
        raise ProbeExit

    def fake_yield():
        clock.now += 1
        if clock.bound and clock.now >= 6:
            raise ProbeBound  # stop a default-unbounded source loop without hanging

    pg = NS(cpu_cp_group=NS(size=lambda: 2), mq=None,
            cleanup_shm_broadcast=lambda: trace.append("shm.cleanup"))
    ns = {
        "contextmanager": contextmanager, "pickle": pickle, "PG": pg,
        "os": NS(_exit=fake_exit, sched_yield=fake_yield),
        "time": NS(monotonic=lambda: clock.now),
        "envs": NS(DEEPINFER_RINGBUFFER_WARNING_INTERVAL=2),
        "logger": NS(debug=lambda *args: trace.append("warning-log")),
        "signal": NS(SIGKILL=9),
        "set_pdeathsig": lambda sig: trace.append(("pdeath-register", sig)),
        "dist": NS(barrier=lambda: trace.append("barrier"),
                   destroy_process_group=lambda: trace.append("dist.destroy")),
        "userbuffer": NS(close=lambda: trace.append("userbuffer.close")),
        "cpu_timer": NS(record=lambda *args, **kwargs: nullcontext()),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, base, queue, broadcast], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<DeepInfer CP control-plane AST>", "exec"), ns)

    class Ring:
        n_reader, max_chunks, max_chunk_bytes = 1, 1, 128

        def __init__(self):
            self.metadata, self.data = bytearray(2), bytearray(128)

        @contextmanager
        def get_metadata(self, index):
            assert index == 0
            yield memoryview(self.metadata)

        @contextmanager
        def get_data(self, index):
            assert index == 0
            yield memoryview(self.data)

    ring = Ring()

    def make_queue(writer):
        q = ns["MessageQueue"]()
        q._is_writer, q._is_local_reader, q._is_remote_reader = writer, not writer, False
        q.n_local_reader, q.n_remote_reader, q.local_reader_rank = 1, 0, 0
        q.buffer, q.current_idx = ring, 0
        return q

    writer, reader = make_queue(True), make_queue(False)
    worker = ns["WorkerBase"]()
    worker.is_driver_worker = True
    worker.model_runner = NS(destroy_cuda_graphs=lambda: trace.append("graphs.destroy"))
    worker.probe = lambda **kwargs: trace.append(("driver-method", kwargs)) or "driver-result"
    pg.mq = writer
    assert worker.run_workers("probe", driver_kwargs={"local": 9}, token=7) == "driver-result"
    assert list(ring.metadata) == [1, 0]  # driver returned before reader consumed command
    assert reader.dequeue() == ("probe", {"token": 7})
    assert list(ring.metadata) == [1, 1]
    assert trace == [("driver-method", {"local": 9, "token": 7})]
    print("Dispatch: driver result precedes reader consumption; read flag is not method completion.")
    trace.clear()
    pg.cpu_cp_group = NS(size=lambda: 1)
    worker.run_workers("probe", token=8)
    assert list(ring.metadata) == [1, 1] and trace == [("driver-method", {"token": 8})]
    pg.cpu_cp_group = NS(size=lambda: 2)

    trace.clear()
    writer.enqueue(("stop", {}))
    pg.mq, worker.is_driver_worker = reader, False
    try:
        worker.start_slave_worker_loop()
    except ProbeExit:
        pass
    else:
        raise AssertionError("slave stop must leave the command loop")
    cleanup = ["barrier", "userbuffer.close", "barrier", "graphs.destroy", "shm.cleanup", "dist.destroy"]
    assert trace == [("pdeath-register", 9), *cleanup, ("exit", 0)]
    trace.clear()
    worker.is_driver_worker = True
    worker.stop()
    assert trace == cleanup
    trace.clear()

    def failed_close():
        trace.append("userbuffer.close")
        raise RuntimeError("injected close failure")

    ns["userbuffer"].close = failed_close
    try:
        worker.stop()
    except RuntimeError as error:
        assert str(error) == "injected close failure"
    else:
        raise AssertionError("expected injected resource failure")
    assert trace == ["barrier", "userbuffer.close"]
    print("Stop: register before receive; two barriers precede graph/shm/dist teardown; injected error propagates.")

    trace.clear()
    ring.metadata[:] = bytes([1, 0])  # unread slot models stalled/absent consumer
    clock.now = 0
    try:
        writer.enqueue("blocked", timeout=2)
    except TimeoutError:
        pass
    else:
        raise AssertionError("explicit ring timeout was not used")
    assert clock.now == 3 and list(ring.metadata) == [1, 0]
    trace.clear()
    clock.now, clock.bound, pg.mq = 0, True, writer
    try:
        ns["cp_broadcast_object"](("stop", {}))
    except ProbeBound:
        pass
    else:
        raise AssertionError("probe must bound the default no-timeout loop")
    assert clock.now == 6 and trace.count("warning-log") == 2
    print("Timeout: explicit deadline raises; wrapper default keeps spinning after warning intervals (bounded by probe).")
    print("PASS CP host-control contract. Fake ring/clock/prctl/resources; no concurrency, Linux, sockets or devices.")


def check_shm_cleanup(root):
    """Original ring/PG cleanup plus original cleanup assertions on real OS shm.

    Only torch.frombuffer(...).fill_(0) is adapted to numpy for initialization.
    Native handles are tracked solely to clean this probe's own segments on error.
    """
    ring_path = root / "deepinfer/distributed/shm_broadcast.py"
    test_path = root / "tests/distributed/test_shm_broadcast.py"
    pg_path = root / "deepinfer/model_executor/parallel_utils/parallel_state.py"
    trees = {}
    for path in (ring_path, test_path, pg_path):
        raw = path.read_bytes()
        trees[path] = ast.parse(raw)
        print(f"{path.name} sha256={hashlib.sha256(raw).hexdigest()}")
    ring_class = next(n for n in trees[ring_path].body
                      if isinstance(n, ast.ClassDef) and n.name == "ShmRingBuffer")
    cleanup_test = next(n for n in trees[test_path].body
                        if isinstance(n, ast.FunctionDef) and n.name == "test_shm_ring_buffer_cleanup")
    cleanup_test.decorator_list = []
    pg_class = next(n for n in trees[pg_path].body
                    if isinstance(n, ast.ClassDef) and n.name == "ParallelProcessGroup")
    pg_class.body = [n for n in pg_class.body if isinstance(n, ast.FunctionDef)
                     and n.name == "cleanup_shm_broadcast"]
    assert len(pg_class.body) == 1
    handles = []

    def tracked_shm(*args, **kwargs):
        handle = shared_memory.SharedMemory(*args, **kwargs)
        handles.append((handle, kwargs.get("create", False)))
        if not kwargs.get("create", False):
            print(f"Attached native mapping size={handle.size}")
        return handle

    ns = {
        "contextmanager": contextmanager, "suppress": suppress, "patch": patch,
        "shared_memory": NS(SharedMemory=tracked_shm),
        "torch": NS(uint8=np.uint8, frombuffer=lambda buffer, dtype:
                    NS(fill_=np.frombuffer(buffer, dtype=dtype).fill)),
        "pytest": NS(raises=TestCase().assertRaises),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, ring_class, pg_class, cleanup_test], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<DeepInfer native shm cleanup AST>", "exec"), ns)
    ring_type, pg_type = ns["ShmRingBuffer"], ns["ParallelProcessGroup"]
    escaped = None
    try:
        for already_unlinked in (False, True):
            ns["test_shm_ring_buffer_cleanup"](already_unlinked)
        print("Original cleanup assertions: normal and already-unlinked cases pass; double destructor tolerated.")
        # macOS rounds native shm sizes to a page on attach. Match that size
        # rather than editing the upstream exact-size assertion to force a pass.
        owner = ring_type(1, mmap.PAGESIZE - 2, 1)
        print(f"Attachment fixture: requested={owner.total_bytes_of_buffer}, page={mmap.PAGESIZE}")
        name = owner.shared_memory.name
        peer = ring_type(*owner.handle())
        with owner.get_metadata(0) as metadata:
            assert bytes(metadata) == bytes(2)
        with TestCase().assertRaises(ValueError):
            len(metadata)
        with owner.get_data(0) as view:
            view[:4] = b"ring"
        with TestCase().assertRaises(ValueError):
            len(view)  # original yielded view is released by its context manager
        with peer.get_data(0) as peer_view:
            assert bytes(peer_view[:4]) == b"ring"
        peer_pg = pg_type()
        peer_pg.mq = NS(buffer=peer)
        peer_pg.cleanup_shm_broadcast()
        peer_pg.cleanup_shm_broadcast()
        assert peer_pg.mq is None and peer.shared_memory.buf is None
        reopened = ring_type(*owner.handle())  # closing the noncreator did not unlink
        reopened.__del__()
        owner_pg = pg_type()
        owner_pg.mq = NS(buffer=owner)
        owner_pg.cleanup_shm_broadcast()
        owner_pg.cleanup_shm_broadcast()
        assert owner_pg.mq is None and owner.shared_memory.buf is None
        with TestCase().assertRaises(FileNotFoundError):
            tracked_shm(name=name)
        print("Real mappings: peer sees data; noncreator close preserves name; creator cleanup removes it; PG repeat is a no-op.")

        retained = ring_type(1, 32, 1)
        retained_name = retained.shared_memory.name
        if retained.shared_memory.size != retained.total_bytes_of_buffer:
            print(f"Native granularity: requested={retained.total_bytes_of_buffer}, mapped={retained.shared_memory.size}")
            with TestCase().assertRaises(AssertionError):
                ring_type(*retained.handle())
            print("Unaligned attachment rejects exact-size assertion on this platform; aligned fixture is separate.")
        with retained.get_data(0) as view:
            view[:4] = b"hold"
            escaped = view[:4]  # deliberate additional export, unlike ordinary scoped use
            copied = bytes(view[:4])
        assert bytes(escaped) == copied == b"hold"
        with TestCase().assertRaises(BufferError):
            retained.__del__()
        # close failed before unlink; prove only the name of this probe's own segment exists.
        with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
            observer = tracked_shm(name=retained_name)
        observer.close()
        escaped.release()
        escaped = None
        retained.__del__()
        retained.__del__()
        assert copied == b"hold"  # detached bytes do not keep a shared-memory export alive
        with TestCase().assertRaises(FileNotFoundError):
            tracked_shm(name=retained_name)
        print("Retained derived view: close raises BufferError; releasing export permits retry; detached bytes remain valid.")
    finally:
        if escaped is not None:
            escaped.release()
        for handle, creator in reversed(handles):
            handle.close()
            if creator:
                with suppress(FileNotFoundError):
                    handle.unlink()
    creators = [h for h, creator in handles if creator]
    assert all(h.buf is None for h, _ in handles)
    for handle in creators:
        with TestCase().assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.name)
    print(f"PASS native shm lifetime checks; {len(creators)} probe-created segments closed/unlinked.")
    print("Real local mappings, but no worker/Gloo/ZMQ/concurrent-process or device validation; no upstream patch.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-event-pool", action="store_true", help="PR #54 timing-alias host model; no device execution")
    modes.add_argument("--check-so-counter", action="store_true", help="PR #54 SO upper-bound model; no grammar/device execution")
    modes.add_argument("--check-output-copy", action="store_true", help="CPU pybind constructor probe plus original collect/release")
    modes.add_argument("--check-worker-lifecycle", action="store_true", help="CP control-plane AST with ring/clock/resource doubles")
    modes.add_argument("--check-shm-cleanup", action="store_true", help="original ring/cleanup assertions on native local shared memory")
    parser.add_argument("--pybind-include", type=Path)
    parser.add_argument("--python-include", type=Path, default=Path(sysconfig.get_path("include")))
    parser.add_argument("--cxx", default=shutil.which("c++") or "c++")
    parser.add_argument("--cxx-flag", action="append", default=[], help="extra compiler flag, repeat as needed")
    args = parser.parse_args()
    root = args.source_root
    if args.check_shm_cleanup:
        check_shm_cleanup(root)
        return
    if args.check_worker_lifecycle:
        check_worker_lifecycle(root)
        return
    if args.check_output_copy:
        if args.pybind_include is None:
            parser.error("--check-output-copy requires --pybind-include; use the target header version")
        check_output_copy(root, args.pybind_include, args.python_include, args.cxx, args.cxx_flag)
        return
    if args.check_event_pool:
        check_event_pool(root)
        return
    if args.check_so_counter:
        check_so_counter(root)
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
