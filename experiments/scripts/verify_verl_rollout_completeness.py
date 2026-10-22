#!/usr/bin/env python3
"""Pinned #6641: original CPU replay tests with their original metadata fake.

Python >=3.10, numpy and pytest. Reads Git objects, AST-isolates the entire
ReplayBuffer and helpers; no package initialization, poll thread, Ray or TQ.
Run: python experiments/scripts/verify_verl_rollout_completeness.py /path/to/verl
"""
import __future__
import argparse
import ast
from collections import defaultdict
import hashlib
import linecache
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pytest

HEAD = "9e4e85da050cc1cbf156cd8c9d89217c68c4ea66"
BASE = "8911f243022aac3558ec7fa8e444cc960ea8d93c"
SOURCE = "verl/trainer/main_ppo_sync.py"
TESTS = "tests/trainer/test_main_ppo_sync_replay_buffer_on_cpu.py"
HELPERS = {"_parse_agent_loop_output_key", "_format_rollout_batch_error", "_get_transfer_queue_cleanup_keys"}
FAILURE_TESTS = {
    "test_replay_buffer_fails_on_terminal_prompt_failure",
    "test_replay_buffer_fails_on_terminal_prompt_failure_without_expected_rollouts",
    "test_replay_buffer_fails_when_finished_prompt_has_missing_rollout",
    "test_replay_buffer_counts_unique_sessions_not_output_rows",
}


def read(root, rev, path):
    data = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(rev[:8], path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    filename = str(root / ".git-object-tests" / rev / path)
    text = data.decode()
    linecache.cache[filename] = (len(data), None, text.splitlines(True), filename)
    return ast.parse(text), filename


def execute(nodes, filename, name, namespace):
    module = ModuleType(name)
    module.__file__ = filename
    module.__dict__.update(namespace)
    sys.modules[name] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), module.__dict__)
    return module


def original_tests(root, rev):
    tree, filename = read(root, rev, SOURCE)
    wanted = {"ReplayBuffer"} | (HELPERS if rev == HEAD else set())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    native = execute(nodes, filename, "_native_rollout_" + rev[:8],
                     dict(defaultdict=defaultdict, threading=threading, time=time, os=os,
                          logger=logging.getLogger("rollout-check"), KVBatchMeta=None))
    tree, filename = read(root, HEAD, TESTS)
    # Preserve the original fake, fixtures and test bodies, not verl imports.
    # Two TensorDict/manager tests are intentionally unselected, not simulated.
    selected = []
    test_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not (isinstance(node, ast.ImportFrom) and node.module.startswith("verl")):
                selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "FakeKVBatchMeta":
            selected.append(node)
        elif isinstance(node, ast.FunctionDef):
            if node.name in {"mock_kv_batch_meta", "replay_buffer"}:
                selected.append(node)
            elif node.name.startswith("test_replay_buffer_") and (rev == HEAD or node.name in FAILURE_TESTS):
                selected.append(node)
                test_names.add(node.name)
    assert len(test_names) == (7 if rev == HEAD else 4)
    namespace = dict(main_ppo_sync=native, ReplayBuffer=native.ReplayBuffer)
    if rev == HEAD:
        namespace["_get_transfer_queue_cleanup_keys"] = native._get_transfer_queue_cleanup_keys
    tests = execute(selected, filename, "_original_rollout_tests_" + rev[:8], namespace)

    class OriginalModule(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.calls, self.failures, self.skips = 0, [], []

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalModule.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1
            if report.failed:
                self.failures.append((report.nodeid.split("::")[-1], report.when, str(report.longrepr)))
            if report.skipped:
                self.skips.append(report.nodeid)

    adapter = Adapter()
    folder = str(Path(__file__).resolve().parent)
    code = pytest.main(["-q", "--noconftest", "-c", os.devnull, "--rootdir", folder,
                        "--confcutdir", folder, "-p", "no:cacheprovider", str(Path(__file__).resolve())],
                       plugins=[adapter])
    assert adapter.calls == len(test_names) and not adapter.skips
    if rev == HEAD:
        assert code == 0 and not adapter.failures, adapter.failures
    else:
        assert code == 1 and {n for n, _, _ in adapter.failures} == FAILURE_TESTS
        assert all(when == "call" and "DID NOT RAISE" in detail for _, when, detail in adapter.failures)
    return native, tests.FakeKVBatchMeta


def boundaries(native, meta):
    native.KVBatchMeta = meta

    def sample(items):
        replay = native.ReplayBuffer.__new__(native.ReplayBuffer)
        replay.partitions = defaultdict(dict)
        replay.lock, replay.poll_interval = threading.Lock(), 0
        replay.add("train", items)
        return replay.sample("train", global_steps=1)

    def tag(status, **extra):
        return dict(global_steps=1, status=status, **extra)

    with pytest.raises(RuntimeError, match="expected 2, got 0"):
        sample({"prompt_a": tag("finished", expected_rollouts=2)})
    print("empty outputs: expected2/got0 rejected")
    legacy = sample({"prompt_a": tag("finished"), "prompt_a_0_0": tag("success")})
    assert legacy.keys == ["prompt_a_0_0"] and legacy.extra_info["rollout_root_keys"] == []
    print("legacy finished marker without expected count: still accepted, not included in root cleanup")
    excess = sample({"prompt_a": tag("finished", expected_rollouts=1),
                     "prompt_a_0_0": tag("success"), "prompt_a_1_0": tag("success")})
    assert len(excess.keys) == 2
    orphan = sample({"prompt_a_0_0": tag("success")})
    assert len(orphan.keys) == 1 and orphan.extra_info["rollout_root_keys"] == []
    print("metadata contract: minimum count, not exact count or missing-root validation")
    empty = sample({"prompt_a": tag("finished", expected_rollouts=0),
                    "old": dict(global_steps=0, status="failure")})
    assert empty.keys == [] and empty.extra_info["rollout_root_keys"] == ["prompt_a"]
    print("expected0 accepts empty; failures at another step do not affect this sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    root = parser.parse_args().source.resolve()
    _, _ = original_tests(root, BASE)
    native, meta = original_tests(root, HEAD)
    boundaries(native, meta)
    assert not any(name in sys.modules for name in ("ray", "transfer_queue", "torch", "verl.utils.device"))
    print("Python", sys.version.split()[0], "numpy", np.__version__, "pytest", pytest.__version__)
    print("Baseline: four original tests fail as expected; candidate: seven original tests pass; five boundary cases pass")
    print("NOT RUN: two TensorDict/manager tests, full package/conftest, poll thread, Ray/TQ, trainer/model/devices")


if __name__ == "__main__":
    main()
