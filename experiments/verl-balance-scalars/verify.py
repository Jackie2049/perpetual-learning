"""Run #7281's two original CPU test bodies without importing the trainer stack.

Source functions and tests are loaded unchanged from pinned public Git blobs.
Upstream sources: verl-project/verl, Apache-2.0, ByteDance contributors.
No GPU, trainer execution, timing, synthetic replacement algorithm or downloads.
"""

import __future__
import argparse
import ast
import hashlib
import heapq
import json
import os
from pathlib import Path
import platform
from numbers import Real
import subprocess

import numpy as np
import torch

VERSIONS = {
    "head": "05ef5dcc8d36f3e7d5c652b2d9d14b11ffc5a487",
    "parent": "87c68d2124ead7e1e3611e6ad0a8d0393dfd5274",
    "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
}
MODULE = "verl/utils/seqlen_balancing.py"
TEST_FILE = "tests/utils/test_seqlen_balancing.py"
TESTS = (
    "test_calculate_workload_as_list_uses_host_scalars",
    "test_calculate_workload_as_list_preserves_float32_partitioning",
)
HELPERS = {"calculate_workload", "karmarkar_karp", "get_seqlen_balanced_partitions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(revision, path):
        raw = subprocess.check_output(
            ["git", "show", revision + ":" + path], cwd=args.source_repo,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def definitions(tree, names, namespace):
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        assert all(not node.decorator_list for node in nodes)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    namespaces = {}
    for label, revision in VERSIONS.items():
        names = HELPERS | ({"calculate_workload_as_list"} if label == "head" else set())
        namespace = {"torch": torch, "np": np, "heapq": heapq, "Real": Real}
        definitions(source(revision, MODULE), names, namespace)
        namespaces[label] = namespace

    tests_tree = source(VERSIONS["head"], TEST_FILE)
    head = namespaces["head"]
    definitions(tests_tree, set(TESTS), head)
    passed = []
    for test in TESTS:
        head[test]()
        passed.append(test)

    # Reuse the exact tensor expression from the original float32 regression.
    test = next(node for node in tests_tree.body if getattr(node, "name", None) == TESTS[1])
    first = test.body[0]
    assert isinstance(first, ast.Assign) and first.targets[0].id == "seqlens"
    seqlens = eval(compile(ast.Expression(first.value), "original_regression_input", "eval"), head)
    partitions = {}
    for label, namespace in namespaces.items():
        workloads = namespace["calculate_workload"](seqlens)
        partition = namespace["get_seqlen_balanced_partitions"]
        partitions[label] = {
            str(equal_size): partition(workloads, k_partitions=2, equal_size=equal_size)
            for equal_size in (True, False)
        }
    assert partitions["head"] == partitions["parent"] == partitions["main"]
    workloads = head["calculate_workload"](seqlens)
    as_list = head["calculate_workload_as_list"](seqlens)
    partition = head["get_seqlen_balanced_partitions"]
    candidate = partition(as_list, k_partitions=2, equal_size=True)
    naive = partition(workloads.tolist(), k_partitions=2, equal_size=True)
    assert candidate == partitions["head"]["True"] and naive != candidate

    trainers = (
        "verl/trainer/ppo/ray_trainer.py",
        "verl/trainer/ppo/v1/trainer_base.py",
        "verl/trainer/sft_trainer_ray.py",
    )
    for path in trainers:
        source(VERSIONS["head"], path)
    print(json.dumps({
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "system": platform.system(),
                        "machine": platform.machine(), "device": "cpu"},
        "versions": VERSIONS,
        "passed_original_tests": passed,
        "original_float32_input": seqlens.tolist(),
        "tensor_partitions": partitions,
        "candidate_numpy_float32": candidate,
        "naive_python_float": naive,
        "host_scalar_type": type(as_list[0]).__name__,
        "source_sha256": hashes,
        "scope": "unchanged AST function/test bodies; no trainer/DP/GPU/timing",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
