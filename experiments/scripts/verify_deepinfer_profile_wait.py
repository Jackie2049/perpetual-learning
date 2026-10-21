#!/usr/bin/env python3
"""Check the original profiling wait coroutine with native asyncio/Future only.

Python >=3.11. Select unchanged _Reply/_ProfileRequest/_wait AST definitions
from fixed Git objects; postpone annotations, omit unused module imports and
manager initialization. Shortened timeout globals are diagnostic controls, not
production timing. No HTTP, sockets, engine, profiler, torch or PR test execution.
"""
import argparse
import ast
import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
import hashlib
from http import HTTPStatus
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

BASE = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"
HEAD = "8c5f5759b53dd7d003927db57af50563cdaa1d37"


def load(root, revision):
    data = subprocess.check_output(
        ["git", "show", f"{revision}:deepinfer/engine/profile_server.py"], cwd=root,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    tree = ast.parse(data)
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ("_Reply", "_ProfileRequest")]
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "ProfileManager")
    wait = next(node for node in manager.body if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_wait")
    assert len(definitions) == 2
    definitions.append(wait)
    module = ModuleType("_original_profile_wait_" + revision)
    sys.modules[module.__name__] = module
    module.__dict__.update(asyncio=asyncio, Future=Future, dataclass=dataclass, field=field,
                           Any=Any, HTTPStatus=HTTPStatus,
                           _PICKUP_TIMEOUT_S=0.01, _COMPLETION_TIMEOUT_S=0.05)
    annotation = ast.parse("from __future__ import annotations").body[0]
    selected = ast.fix_missing_locations(ast.Module(body=[annotation, *definitions], type_ignores=[]))
    exec(compile(selected, f"{revision}:profile_server.py[selected]", "exec"), module.__dict__)
    print(revision, "profile_server.py sha256=" + hashlib.sha256(data).hexdigest())
    return module, ast.dump(wait, include_attributes=False)


async def check(module):
    request, reply, wait = module._ProfileRequest, module._Reply, module._wait
    completed = request("not-used", 1)
    answer = reply.ok("completed")
    completed.reply.set_result(answer)
    assert await wait(None, completed) is answer

    queued = request("not-used", 1)
    result = await wait(None, queued)
    assert result.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert "did not pick up" in result.body["message"]
    assert queued.reply.cancelled()
    assert not queued.reply.set_running_or_notify_cancel()

    running = request("not-used", 1)
    assert running.reply.set_running_or_notify_cancel()
    timer = asyncio.get_running_loop().call_later(0.025, running.reply.set_result, answer)
    try:
        assert await wait(None, running) is answer
        assert not running.reply.cancelled()
    finally:
        timer.cancel()

    late = request("not-used", 1)
    assert late.reply.set_running_or_notify_cancel()
    result = await wait(None, late)
    assert result.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert "still running" in result.body["message"]
    assert late.reply.running() and not late.reply.cancelled()
    late.reply.set_result(answer)
    assert late.reply.result() is answer  # no exporter is called by _wait
    print("4 original-coroutine controls passed; no shutdown/HTTP/profiler claims")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    base, base_ast = load(args.source_repo, BASE)
    patched, patched_ast = load(args.source_repo, HEAD)
    assert base_ast == patched_ast, "wait logic changed; revisit assumptions"
    print("python", sys.version.split()[0], "base/head _wait AST identical")
    asyncio.run(check(base))
    asyncio.run(check(patched))
    assert "torch" not in sys.modules and "uvicorn" not in sys.modules
    print("NOT RUN: author's 15 tests, manager.close, HTTP/port reuse, captures or engine/GPU")


if __name__ == "__main__":
    main()
