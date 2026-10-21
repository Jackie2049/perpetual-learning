#!/usr/bin/env python3
"""Pinned AsyncTokenBucket: 16 original coroutine test bodies and 2 cancellations.

Python >=3.10 and pytest. AST-selects the complete original bucket class and
get_event_loop helper, then loads the complete original test module unchanged.
Directly awaits its methods, preserving decorators/assertions/real timings;
not full pytest, manager integration, Ray, API calls or distributed quota tests.
No fake clock, asyncio substitute, shortened sleeps or source checkout edits.
"""
import argparse
import ast
import asyncio
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
MODULE = "verl.experimental.reward_loop.reward_manager.limited"


def load(root, path, name, selected=None, globals_=None):
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    source = data
    if selected:
        nodes = [node for node in ast.parse(data).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
        assert {node.name for node in nodes} == selected
        source = ast.Module(body=nodes, type_ignores=[])
    module = ModuleType(name)
    module.__dict__.update(globals_ or {})
    sys.modules[name] = module
    exec(compile(source, f"{REV}:{path}", "exec"), module.__dict__)
    return module


async def cancellations(bucket_class):
    for oversize in [False, True]:
        bucket = bucket_class(rate_limit=1.0, max_tokens=1.0)
        if not oversize:
            await bucket.acquire(1.0)
        task = asyncio.create_task(bucket.acquire(5.0 if oversize else 1.0))
        try:
            await asyncio.sleep(0)  # Original acquire runs until its real sleep.
            assert not task.done()
            before = bucket.tokens
            assert (before == -4.0) if oversize else (0.0 <= before < 1.0)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert bucket.tokens == before, "cancellation unexpectedly refunded balance"
        print("cancel", "oversize" if oversize else "standard", "stored balance", before, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    assert "verl" not in sys.modules
    for name in ["verl", "verl.experimental", "verl.experimental.reward_loop",
                 "verl.experimental.reward_loop.reward_manager"]:
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    helper = load(args.source_repo, "verl/utils/ray_utils.py", "_native_loop_helper",
                  {"get_event_loop"}, {"asyncio": asyncio})
    native = load(args.source_repo, "verl/experimental/reward_loop/reward_manager/limited.py", MODULE,
                  {"AsyncTokenBucket"}, {"asyncio": asyncio, "get_event_loop": helper.get_event_loop})
    tests = load(args.source_repo, "tests/experimental/reward_loop/test_async_token_bucket_on_cpu.py",
                 "_original_bucket_tests")
    names = sorted(name for name in vars(tests.TestAsyncTokenBucket) if name.startswith("test_"))
    assert len(names) == 16
    for name in names:
        asyncio.run(getattr(tests.TestAsyncTokenBucket(), name)())
        print("PASS", name, flush=True)
    asyncio.run(cancellations(native.AsyncTokenBucket))
    assert not any(name == "ray" or name.startswith("ray.") for name in sys.modules)
    print("16 original test bodies and 2 cancellation observations passed; Python", sys.version.split()[0])


if __name__ == "__main__":
    main()
