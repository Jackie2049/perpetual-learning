"""Finite CPU control-flow checks for verl #8459; no Ray/model/device imports."""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HEAD = "eafee25507cd54bfe62c5282e02705187d871e6e"
SOURCES = {
    "verl/experimental/teacher_loop/teacher_model.py": "c0da409d626e21247ea5757380956a9da381e9d9d08ca0813bdd639b45bc3f85",
    "verl/utils/ray_utils.py": "1f70a9ac05b4a6f92a3ad2486bee17ac9721a9568e7e72b17e76172a60a7751b",
}


def load(repo):
    trees = []
    for path, digest in SOURCES.items():
        raw = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{HEAD}:{path}"],
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        assert hashlib.sha256(raw).hexdigest() == digest, path
        compile(raw, path, "exec")
        trees.append(ast.parse(raw))
    ns = dict(asyncio=asyncio, concurrent=concurrent, functools=functools, inspect=inspect)
    auto = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == "auto_await")
    nodes = [auto]
    for node in trees[0].body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_all":
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name in ("TeacherModelManager", "MultiTeacherModelManager"):
            names = ({"__init__", "initialize"} if node.name == "TeacherModelManager" else
                     {"_initialize_teacher_model_managers", "_initialize_managers"})
            selected = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
            nodes.append(ast.ClassDef(node.name, [], [], selected, []))
    # Future annotations avoid importing the annotation-only framework types.
    module = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes], [])
    exec(compile(ast.fix_missing_locations(module), "original-methods", "exec"), ns)
    return ns


class Scenario:
    def __init__(self, keys, failure=False):
        self.keys, self.failure = keys, failure
        self.trace, self.started, self.created = [], [], []
        self.all_started, self.alpha_done = asyncio.Event(), asyncio.Event()
        self.release, self.finished = asyncio.Event(), asyncio.Event()
        self.maps = None


def teacher_class(ns):
    class Teacher(ns["TeacherModelManager"]):
        def __init__(self, **kwargs):
            self.scenario = kwargs["teacher_model_config"].scenario
            self.key = kwargs["teacher_model_config"].key
            self.scenario.created.append(self)
            super().__init__(**kwargs)

        async def _initialize_llm_servers(self):
            s = self.scenario
            if s.maps is not None:
                assert all(not value for value in s.maps)
            s.started.append(self.key)
            s.trace.append("start:" + self.key)
            if len(s.started) == len(s.keys):
                s.all_started.set()
            await asyncio.wait_for(s.all_started.wait(), timeout=5)  # hang guard, not a timing assertion
            if s.failure and self.key == "fail":
                s.trace.append("fail:" + self.key)
                raise ValueError("teacher failed")
            if s.failure:
                try:
                    await s.release.wait()
                except asyncio.CancelledError:
                    s.trace.append("cancel:" + self.key)
                    raise
            elif self.key == "zeta":
                await s.alpha_done.wait()
            self.server_addresses, self.server_handles = [self.key], ["handle:" + self.key]
            s.trace.append("ready:" + self.key)
            if self.key == "alpha":
                s.alpha_done.set()

        def _initialize_load_balancer_handle(self):
            assert hasattr(self, "server_addresses")
            if self.scenario.maps is not None:
                assert all(not value for value in self.scenario.maps)
            self.load_balancer_handle = "balance:" + self.key
            self.scenario.trace.append(self.load_balancer_handle)
            self.scenario.finished.set()
    return Teacher


def make(Teacher, scenario, key, defer=True):
    return Teacher(distillation_config=None, teacher_model_config=SimpleNamespace(key=key, scenario=scenario),
                   resource_pool=None, defer_init=defer)


def main(repo):
    ns = load(repo)
    Teacher = teacher_class(ns)
    results = {}
    s = Scenario(["solo"])
    m = make(Teacher, s, "solo", defer=False)
    assert m.load_balancer_handle == "balance:solo"
    assert s.trace == ["start:solo", "ready:solo", "balance:solo"]
    results["default_constructor"] = s.trace

    s = Scenario(["solo"])
    m = make(Teacher, s, "solo")
    coro = m.initialize()
    assert inspect.iscoroutine(coro) and s.trace == [] and not hasattr(m, "server_addresses")
    coro.close()
    results["defer_and_coroutine_creation"] = {"events": s.trace, "addresses_published": False}

    s = Scenario(["zeta", "alpha"])
    outer = SimpleNamespace(
        distillation_config=SimpleNamespace(teacher_models={key: SimpleNamespace(key=key, world_size=1, scenario=s) for key in s.keys}),
        resource_pool=None, teacher_model_managers={}, server_addresses={}, server_handles={}, load_balancer_handle={},
        _initialize_managers=ns["MultiTeacherModelManager"]._initialize_managers,
    )
    s.maps = [outer.teacher_model_managers, outer.server_addresses, outer.server_handles, outer.load_balancer_handle]
    ns.update(TeacherModelManager=Teacher, split_resource_pool=lambda pool, split_size: list(range(len(split_size))))
    ns["MultiTeacherModelManager"]._initialize_teacher_model_managers(outer)
    assert s.trace[:2] == ["start:zeta", "start:alpha"]
    assert s.trace[2:] == ["ready:alpha", "balance:alpha", "ready:zeta", "balance:zeta"]
    assert all(list(value) == ["zeta", "alpha"] for value in s.maps)
    results["concurrent_start_and_publication"] = {"events": s.trace, "keys": list(outer.server_addresses)}

    async def failure_inside_running_loop():
        s = Scenario(["fail", "slow"], failure=True)
        managers = [make(Teacher, s, key) for key in s.keys]
        try:
            await ns["_run_all"]([outer._initialize_managers(managers)])
        except ValueError as error:
            assert str(error) == "teacher failed"
        else:
            raise AssertionError("expected teacher failure")
        assert not s.finished.is_set() and "cancel:slow" not in s.trace
        s.trace.append("error_observed")
        s.release.set()
        await asyncio.wait_for(s.finished.wait(), timeout=5)
        assert s.trace[-2:] == ["ready:slow", "balance:slow"]
        return s.trace
    results["gather_in_running_loop"] = asyncio.run(failure_inside_running_loop())

    s = Scenario(["fail", "slow"], failure=True)
    managers = [make(Teacher, s, key) for key in s.keys]
    try:
        ns["_run_all"]([outer._initialize_managers(managers)])
    except ValueError as error:
        assert str(error) == "teacher failed"
    else:
        raise AssertionError("expected teacher failure")
    assert s.trace == ["start:fail", "start:slow", "fail:fail", "cancel:slow"]
    results["sync_entry_loop_shutdown"] = s.trace
    print(json.dumps({"head": HEAD, "python": sys.version.split()[0], "sources": SOURCES,
                      "scope": "original methods with async teacher/resource substitutes; no Ray, devices or startup timing",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
