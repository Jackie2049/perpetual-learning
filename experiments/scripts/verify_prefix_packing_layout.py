#!/usr/bin/env python3
"""Run a pinned subset of PR #8300's original CPU layout tests without verl imports.

Python >=3.10 and torch with jagged nested tensors. Reads an existing Git object;
does not fetch, change checkout, install packages, or run pytest/conftest.
No model, FlashAttention, engine, distributed batching, or gradient equivalence
test is run. Only original layout helpers and the listed original tests execute.
"""
import argparse
import ast
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType

import torch

REVISION = "d6dfcb284d5acbe8a2ecb860843edee7b9d078be"
HELPER = "verl/workers/utils/prefix_packing.py"
TESTS = "tests/workers/test_prefix_packing_on_cpu.py"


def load(root, path, names, namespace):
    data = subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=root)
    print(path, "sha256=" + hashlib.sha256(data).hexdigest())
    nodes = [n for n in ast.parse(data).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names), "pinned definitions missing"
    for node in nodes:
        # Parameter expansion is explicit below; the function bodies remain original.
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, f"{REVISION}:{path}", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    print("revision", REVISION)
    print("python", sys.version.split()[0], "torch", torch.__version__, "device=cpu")
    module = ModuleType("_original_prefix_layout")
    sys.modules[module.__name__] = module  # dataclass resolves its defining module
    ns = module.__dict__
    ns.update(torch=torch, random=random, dataclass=dataclass, field=field, replace=replace)
    load(args.source_repo, HELPER, [
        "_LayerCounter", "PrefixPackingAttnMeta", "PrefixPackingPlan",
        "find_prefix_groups", "build_prefix_packing_plan", "packed_group_lens", "split_groups_to_budget",
    ], ns)
    load(args.source_repo, TESTS, [
        "_nested", "_empty_nested", "_make_batch", "_positions", "_offsets", "_random_spec",
        "_dense_tree_mask", "_brute_force_mask", "TestFindPrefixGroups", "TestBuildPlan", "TestBudget",
        "test_rolled_labels_on_read_rows", "test_dense_mask_matches_brute_force",
        "test_layer_counter_shared_with_fsdp_kwarg_copy",
    ], ns)
    count = 0
    for name in ("TestFindPrefixGroups", "TestBuildPlan", "TestBudget"):
        obj = ns[name]()
        methods = sorted(n for n in vars(type(obj)) if n.startswith("test_"))
        for method in methods:
            getattr(obj, method)()
        count += len(methods)
        print(name, len(methods), "passed")
    for seed in range(4):
        ns["test_rolled_labels_on_read_rows"](seed)
        count += 1
    for seed in range(8):
        for pad in (0, 3):
            ns["test_dense_mask_matches_brute_force"](seed, pad)
            count += 1
    ns["test_layer_counter_shared_with_fsdp_kwarg_copy"]()
    count += 1
    assert count == 38
    print("38 original CPU cases passed (17 class methods, 4 label, 16 mask, 1 counter)")
    print("NOT RUN: prompt-length/TensorDict, tiny Qwen model/gradients, engine, FA2/3, DP/SP or GPU tests")


if __name__ == "__main__":
    main()
