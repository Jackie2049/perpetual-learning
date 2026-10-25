"""Execute #7382's four original async test bodies with their original state fakes.

Source: verl-project/verl, Apache-2.0, ByteDance contributors.
Only the original helper/constants and original tests are AST-loaded.
No Ray import, cluster, actor, engine, GPU or new failure matrix.
"""

import __future__
import argparse
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest


HEAD = "8ebc3558c39d5192a6bf312a806c889a2197f293"
SOURCE = "verl/workers/rollout/replica.py"
TEST = "tests/workers/rollout/test_rollout_server_schedule_guard_on_cpu.py"
CALLERS = ["verl/workers/rollout/" + name for name in (
    "vllm_rollout/vllm_async_server.py", "sglang_rollout/async_sglang_server.py",
    "trtllm_rollout/trtllm_async_server.py")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(path):
        raw = subprocess.check_output(["git", "show", HEAD + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    tree = source(SOURCE)
    names = {"ROLLOUT_SERVER_SCHEDULE_TIMEOUT_S", "_UNSCHEDULED_ACTOR_STATES"}
    selected = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                            for target in node.targets)) or
        getattr(node, "name", None) == "wait_rollout_servers_scheduled"]
    assert len(selected) == 3
    replica = ModuleType("_original_replica_helper")
    replica.__dict__.update(asyncio=asyncio, time=time, get_actor=None)
    exec(compile(ast.Module(body=selected, type_ignores=[]), SOURCE, "exec",
                 flags=__future__.annotations.compiler_flag), replica.__dict__)
    tests = {"pytest": pytest, "SimpleNamespace": SimpleNamespace, "replica": replica}
    test_tree = source(TEST)
    definitions = [node for node in test_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), TEST, "exec",
                 flags=__future__.annotations.compiler_flag), tests)
    passed = []
    for node in definitions:
        if not node.name.startswith("test_"):
            continue
        with pytest.MonkeyPatch.context() as monkeypatch:
            asyncio.run(tests[node.name](monkeypatch))
        passed.append(node.name)
    assert len(passed) == 4
    wiring = {}
    for path in CALLERS:
        calls = [node for node in ast.walk(source(path)) if isinstance(node, ast.Await)
                 and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "wait_rollout_servers_scheduled"]
        assert len(calls) == 1
        wiring[path] = {"guard_await_line": calls[0].lineno, "check": "AST only; caller not executed"}
    assert "ray" not in sys.modules and "torch" not in sys.modules
    print(json.dumps({"revision": HEAD, "source_sha256": hashes,
                      "passed_original_test_bodies": passed, "caller_wiring": wiring,
                      "environment": {"python": platform.python_version(), "pytest": pytest.__version__,
                                      "system": platform.system()},
                      "scope": "unchanged original helper/test bodies, original get_actor replacements; "
                               "real asyncio/time and pytest MonkeyPatch/raises; "
                               "not pytest collection, Ray State API, scheduling or backend launch"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
