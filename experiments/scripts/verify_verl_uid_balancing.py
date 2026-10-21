#!/usr/bin/env python3
"""CPU-only UID grouping contract checks on fixed native verl functions.

Python >=3.10 and CPU torch. AST-selects four complete implementation functions
and three complete original test functions, without editing bodies/decorators.
Package shells allow their local imports without initializing verl/Ray/devices.
Calls test functions directly: not full pytest collection or trainer/dispatch.
"""
import argparse
import ast
import hashlib
import heapq
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
FUNCTIONS = {"calculate_workload", "karmarkar_karp", "get_seqlen_balanced_partitions",
             "get_group_balanced_partitions"}
TESTS = {"test_group_balanced_partitions", "test_group_balanced_partitions_single_sample_groups",
         "test_group_balanced_partitions_equal_size"}


def selected(root, path, names, module):
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest())
    tree = ast.parse(data)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{REV}:{path}", "exec"), module.__dict__)


def check_cover(parts, count):
    flat = [i for part in parts for i in part]
    assert sorted(flat) == list(range(count))
    assert all(part == sorted(part) for part in parts)


def split_uids(parts, uids):
    owners = {}
    for rank, part in enumerate(parts):
        for i in part:
            owners.setdefault(uids[i], set()).add(rank)
    return {uid for uid, ranks in owners.items() if len(ranks) > 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    assert "verl" not in sys.modules
    for name in ["verl", "verl.utils"]:
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    native = ModuleType("verl.utils.seqlen_balancing")
    native.__dict__.update(torch=torch, heapq=heapq)
    sys.modules[native.__name__] = native
    selected(args.source_repo, "verl/utils/seqlen_balancing.py", FUNCTIONS, native)
    original = ModuleType("_original_uid_tests")
    selected(args.source_repo, "tests/utils/test_seqlen_balancing.py", TESTS, original)
    for name in sorted(TESTS):
        getattr(original, name)()
    print("3 original test bodies passed (direct calls, not full pytest)")
    group = native.get_group_balanced_partitions
    for groups in [4, 8]:
        for n in [1, 2, 3]:
            for ranks in [1, 2, 4]:
                uids = [uid for uid in range(groups) for _ in range(n)]
                lengths = [(i * 37 + 11) % 1000 + 1 for i in range(len(uids))]
                parts = group(lengths, uids, ranks)
                check_cover(parts, len(uids))
                assert not split_uids(parts, uids)
                assert {len(part) for part in parts} == {len(uids) // ranks}
    print("18 contiguous equal-cardinality grouping controls passed")
    uids = [0, 1, 0, 1]
    parts = group([1] * 4, uids, 2)
    check_cover(parts, 4)
    assert split_uids(parts, uids) == {0, 1}
    print("noncontiguous", parts, "split UIDs", sorted(split_uids(parts, uids)))
    uids = [0, 1, 2] + [3] * 5
    parts = group([1] * 8, uids, 2)
    check_cover(parts, 8)
    assert not split_uids(parts, uids)
    assert sorted(map(len, parts)) == [2, 6]
    flat = [i for part in parts for i in part]
    equal_row_chunks = [flat[:4], flat[4:]]
    assert split_uids(equal_row_chunks, uids) == {3}
    print("unequal group sizes", parts, "equal-row chunk illustration", equal_row_chunks)
    for lengths, uids, ranks in [([1] * 3, [0, 1, 2], 2), ([1] * 2, [0, 0], 2)]:
        try:
            group(lengths, uids, ranks)
        except AssertionError as error:
            print("expected precondition failure", str(error))
        else:
            raise AssertionError("missing grouping precondition")
    assert not any(name == "ray" or name.startswith("ray.") for name in sys.modules)
    print("4 boundary observations passed; no trainer, DataProto, dispatch, distributed or device execution")
    print("Python", sys.version.split()[0], "torch", torch.__version__)


if __name__ == "__main__":
    main()
