"""Pinned #7382 readiness revision: original test bodies and local coroutine controls.

../deepinfer-eplb-venv/bin/python -B experiments/verl-server-schedule-guard/verify_ready.py ../verl-packaging-readonly
No Ray/torch import, actor, cluster, HTTP request or accelerator initialization.
"""
import __future__
import ast
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

MIDDLE = "6047ae650be2c3fa84be40aaac4626e6d0d69961"
HEAD = "466adbad3433e66beb10ad448a82fa3f8cceab73"
SOURCE = "verl/workers/rollout/replica.py"
TEST = "tests/workers/rollout/test_rollout_server_schedule_guard_on_cpu.py"
CALLERS = ["verl/workers/rollout/" + p for p in ("vllm_rollout/vllm_async_server.py", "sglang_rollout/async_sglang_server.py", "trtllm_rollout/trtllm_async_server.py")]


def server(name, ready):
    return SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: name), __ray_ready__=SimpleNamespace(remote=ready))


async def call(module, servers, timeout):
    if module.revision == HEAD:
        await module.wait_rollout_servers_scheduled(servers, [f"target-{i}" for i in range(len(servers))], timeout_s=timeout)
    else:
        await module.wait_rollout_servers_scheduled(servers, timeout_s=timeout)


async def init_failure(module):
    """Own control: real asyncio tasks, not Ray actor cancellation semantics."""
    started = asyncio.Event()
    tasks = []

    async def pending():
        tasks.append(asyncio.current_task())
        started.set()
        await asyncio.Event().wait()

    async def failing():
        await started.wait()
        raise ValueError("own init failure")

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must not run on init failure")

    module.get_actor = forbidden
    begin = time.monotonic()
    try:
        with pytest.raises(ValueError, match="own init failure"):
            await call(module, [server("failed", failing), server("pending", pending)], 0.05)
        elapsed = time.monotonic() - begin
        await asyncio.sleep(0)
        live = sum(not task.done() for task in tasks)
        cancelled = sum(task.cancelled() for task in tasks)
        assert live == (1 if module.revision == MIDDLE else 0)
        return {"revision": module.revision, "deadline_s": 0.05, "elapsed_s": elapsed, "pending_local_tasks_after_failure": live, "cancelled_local_tasks": cancelled}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def caller_cancel(module):
    started = asyncio.Event()
    tasks = []

    async def pending():
        tasks.append(asyncio.current_task())
        started.set()
        await asyncio.Event().wait()

    outer = asyncio.create_task(call(module, [server("pending", pending)], 1.0))
    try:
        await started.wait()
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        await asyncio.sleep(0)
        live = sum(not task.done() for task in tasks)
        assert live == 1
        return {"revision": module.revision, "outer_cancelled": outer.cancelled(), "readiness_local_tasks_still_live": live, "probe_cleanup": "explicitly cancel and gather the owned local task"}
    finally:
        outer.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(outer, *tasks, return_exceptions=True)


async def diagnostic_boundary(module):
    loop_thread = threading.get_ident()
    calls = []
    cancelled = asyncio.Event()

    async def pending():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    def slow_state(actor_id, timeout):
        calls.append({"actor_id": actor_id, "api_timeout_s": timeout, "off_event_loop_thread": threading.get_ident() != loop_thread})
        time.sleep(0.04)
        return {"name": "already-placed", "state": "ALIVE", "node_id": "reported-node", "required_resources": {"CPU": 0.5}}

    module.get_actor = slow_state
    begin = time.monotonic()
    with pytest.raises(RuntimeError) as error:
        await call(module, [server("late-ready", pending)], 0.005)
    elapsed = time.monotonic() - begin
    message = str(error.value)
    assert "target_node=target-0 state=ALIVE" in message
    assert "has no free slot" in message
    assert cancelled.is_set() and calls[0]["off_event_loop_thread"]
    return {"deadline_s": 0.005, "injected_diagnostic_delay_s": 0.04, "elapsed_s": elapsed, "calls": calls, "readiness_local_task_cancelled": cancelled.is_set(), "message": message, "scope": "Artificial ALIVE snapshot with unresolved readiness; not a real actor diagnosis or timing benchmark"}


def main():
    hashes, trees = {}, {}
    for revision, paths in ((MIDDLE, (SOURCE,)), (HEAD, (SOURCE, TEST, *CALLERS))):
        for path in paths:
            raw = subprocess.check_output(["git", "-C", sys.argv[1], "show", f"{revision}:{path}"], env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            compile(raw, path, "exec")
            hashes[f"{revision}:{path}"] = hashlib.sha256(raw).hexdigest()
            trees[revision, path] = ast.parse(raw)
    modules, ast_hashes = {}, {}
    for revision in (MIDDLE, HEAD):
        nodes = [node for node in trees[revision, SOURCE].body if getattr(node, "name", None) in {"wait_rollout_servers_scheduled", "_describe_unready_actor"} or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROLLOUT_SERVER_SCHEDULE_TIMEOUT_S" for t in node.targets)]
        assert len(nodes) == 3
        module = modules[revision] = ModuleType("original_schedule_guard")
        module.__dict__.update(asyncio=asyncio, get_actor=None, revision=revision)
        # Compile original nodes directly; no decorator, body or constant is rewritten.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), SOURCE, "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
        ast_hashes[revision] = {getattr(n, "name", "timeout_constant"): hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in nodes}
    tests = {"asyncio": asyncio, "pytest": pytest, "SimpleNamespace": SimpleNamespace, "replica": modules[HEAD]}
    definitions = [n for n in trees[HEAD, TEST].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), TEST, "exec", flags=__future__.annotations.compiler_flag), tests)
    passed = []
    for node in definitions:
        if node.name.startswith("test_"):
            with pytest.MonkeyPatch.context() as patch:
                asyncio.run(tests[node.name](patch))
            passed.append(node.name)
    assert len(passed) == 4
    controls = {"init_failure": [asyncio.run(init_failure(modules[rev])) for rev in (MIDDLE, HEAD)], "caller_cancellation": asyncio.run(caller_cancel(modules[HEAD])), "diagnostic_boundary": asyncio.run(diagnostic_boundary(modules[HEAD]))}
    wiring = {}
    for path in CALLERS:
        tree = trees[HEAD, path]
        guards = [n for n in ast.walk(tree) if isinstance(n, ast.Await) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "wait_rollout_servers_scheduled"]
        assert len(guards) == 1 and len(guards[0].value.args) == 2
        wiring[path] = {"guard_line": guards[0].lineno, "positional_arguments": [ast.unparse(n) for n in guards[0].value.args], "scope": "AST wiring only, launch not executed"}
    assert "ray" not in sys.modules and "torch" not in sys.modules
    print(json.dumps({"revision": HEAD, "environment": {"python": platform.python_version(), "pytest": pytest.__version__, "system": platform.system()}, "source_sha256": hashes, "original_ast_sha256": ast_hashes, "passed_original_test_bodies": passed, "own_controls": controls, "caller_wiring": wiring, "scope": "Original four async test bodies with original fixtures, not native pytest collection/full package imports. Own controls use local asyncio tasks and a state recorder, not Ray ObjectRefs, resource release, dashboard or engine readiness."}, indent=2))


if __name__ == "__main__":
    main()
