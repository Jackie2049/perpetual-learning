"""Thirteen original slot unit-test parameters, no Ray import or cluster.

Original pool classes and two worker initializers are extracted unchanged.
Original fake PGs/mocks stay in force; no actors, resources or devices exist.
"""

import argparse
import ast
import __future__
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

HEAD = "57916cf0af1ffb0a60263077c3df374a2ee5fe47"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
SOURCE = "verl/single_controller/ray/base.py"
PARENT = "verl/single_controller/base/worker_group.py"
TEST = "tests/single_controller/test_resource_pool_slots_on_cpu.py"
POOL_NAMES = {"get_random_string", "RayResourcePool", "SubRayResourcePool",
              "split_resource_pool", "merge_resource_pool", "extract_pg_from_exist"}
METHOD_NAMES = {"_init_with_resource_pool", "_init_with_subresource_pool"}


def execute(nodes, filename, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    sources, hashes = {}, {}
    for path in (SOURCE, PARENT, TEST):
        raw = subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), path, "exec")
        sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    original = ModuleType("isolated_slot_availability")
    sys.modules[original.__name__] = original
    ns = original.__dict__
    platform_seam = SimpleNamespace(supports_colocated_worker_groups=lambda: True,
                                    device_name="cuda", ray_resource_name=lambda: "GPU")

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("unmocked Ray allocation requested")

    ns.update(np=np, logger=logging.getLogger(original.__name__),
              get_platform=lambda: platform_seam, get_device_name=lambda: "cuda",
              placement_group=forbidden_allocation, ray=SimpleNamespace(get=forbidden_allocation),
              sort_placement_group_by_node_ip=lambda pgs: pgs)
    parent = next(n for n in ast.parse(sources[PARENT]).body
                  if isinstance(n, ast.ClassDef) and n.name == "ResourcePool")
    execute([parent], PARENT, ns)
    nodes = [n for n in ast.parse(sources[SOURCE]).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in POOL_NAMES]
    assert {n.name for n in nodes} == POOL_NAMES
    execute(nodes, SOURCE, ns)
    worker = next(n for n in ast.parse(sources[SOURCE]).body
                  if isinstance(n, ast.ClassDef) and n.name == "RayWorkerGroup")
    methods = [n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name in METHOD_NAMES]
    assert {n.name for n in methods} == METHOD_NAMES
    method_ns = dict(ns)
    execute(methods, SOURCE, method_ns)
    ns["RayWorkerGroup"] = SimpleNamespace(**{name: method_ns[name] for name in METHOD_NAMES})

    def isolated_patch(target, *args, **kwargs):
        # Keep original patch behavior, redirect only its module lookup to our AST module.
        prefix = "verl.single_controller.ray.base."
        assert target.startswith(prefix)
        return patch(original.__name__ + "." + target[len(prefix):], *args, **kwargs)

    tests = dict(ns, pytest=pytest, Mock=Mock, SimpleNamespace=SimpleNamespace, patch=isolated_patch)
    functions = [n for n in ast.parse(sources[TEST]).body if isinstance(n, ast.FunctionDef)
                 and n.name != "test_real_cpu_placement_group_rejects_bad_slot_and_launches_on_healthy_slot"]
    for node in functions:
        node.decorator_list = []  # manual original parameter grid, bodies/assertions unchanged
    execute(functions, TEST, tests)
    reports = []
    for node in functions:
        if not node.name.startswith("test_"):
            continue
        parameters = [{}]
        if node.name == "test_out_of_range_slot_is_rejected":
            parameters = [{"index": -1}, {"index": 2}]
        elif node.name == "test_worker_launch_fails_before_master_or_actor_creation":
            parameters = [{"subpool": False}, {"subpool": True}]
        for values in parameters:
            tests[node.name](**values)
            reports.append(dict(test=node.name, parameters=values, outcome="passed"))
    assert len(reports) == 13
    print(json.dumps(dict(head=HEAD, base=BASE, python=platform.python_version(), numpy=np.__version__,
                         pytest=pytest.__version__, source_sha256=hashes, original_test_cases=reports,
                         scope="Complete original ResourcePool, RayResourcePool, SubRayResourcePool classes; "
                         "four original free functions and two full original worker initializer methods. "
                         "Annotations postponed, production bodies unchanged. Eleven original test functions "
                         "plus make_pool extracted, only decorators removed; 13 parameters run manually, not "
                         "native pytest. Original Mock/fake PG fixtures preserved; patch module lookup redirected "
                         "to isolated module. Platform/device/sorter/Ray import seams; cuda/GPU are inert strings, "
                         "no device detection. Three full inputs compile. Real CPU Ray integration case, related "
                         "device-agnostic suite, actual PG/master/actor creation, Ray cluster/scheduler, resource "
                         "release, failure detection/recovery, process/driver restart, GPU/NPU and training not run."),
                     indent=2))


if __name__ == "__main__":
    main()
