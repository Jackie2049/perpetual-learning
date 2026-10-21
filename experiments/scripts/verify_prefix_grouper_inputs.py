#!/usr/bin/env python3
"""Check native PrefixGrouper input layouts, not attention, models or speed.

Python >=3.10, CPU torch, and two existing source Git repositories. Loads the
unmodified original package directly from pinned Git blobs (no install/checkout)
and AST-extracts verl's two input-building functions. No CUDA function is called.
The historical verl install command did not pin a library revision: this does not
establish which package was used in any past benchmark.
"""
import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import torch

PG_REV = "1accedc043c9ea79165888f48ab163fb8d6d5bb0"
VERL_REV = "d6bc62964d06588b0728122a1d2414b098165295"


def blob(root, revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))


class GitPackage(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root):
        self.root = root
        self.paths = set(subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", PG_REV, "src/prefix_grouper"], cwd=root
        ).decode().splitlines())
        self.sources = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "prefix_grouper" and not fullname.startswith("prefix_grouper."):
            return None
        base = "src/" + fullname.replace(".", "/")
        for candidate, package in ((base + "/__init__.py", True), (base + ".py", False)):
            if candidate in self.paths:
                self.sources[fullname] = candidate
                return importlib.util.spec_from_loader(fullname, self, is_package=package)
        raise ModuleNotFoundError(fullname)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = self.sources[module.__name__]
        data = blob(self.root, PG_REV, path)
        print(path, "sha256=" + hashlib.sha256(data).hexdigest())
        exec(compile(data, f"{PG_REV}:{path}", "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl_repo", type=Path)
    parser.add_argument("prefix_grouper_repo", type=Path)
    args = parser.parse_args()
    print("verl", VERL_REV, "PrefixGrouper", PG_REV, "device=cpu")
    assert "prefix_grouper" not in sys.modules
    finder = GitPackage(args.prefix_grouper_repo)
    sys.meta_path.insert(0, finder)
    try:
        from prefix_grouper import PrefixGrouper
    finally:
        sys.meta_path.remove(finder)
    path = "verl/trainer/ppo/prefix_grouper_utils.py"
    data = blob(args.verl_repo, VERL_REV, path)
    print(path, "sha256=" + hashlib.sha256(data).hexdigest())
    names = {"build_position_ids_for_prefix_grouper", "build_pg_from_micro_batch"}
    nodes = [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = {"torch": torch, "PrefixGrouper": PrefixGrouper}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<original verl input functions>", "exec"), ns)

    # Each tuple is (uid, unpadded prompt, unpadded response). IDs never use pad=0.
    cases = [
        ([("a", [1, 2, 3], [10, 11]), ("a", [1, 2, 3], [12, 13]),
          ("a", [1, 2, 3], [14, 15])], [[1, 2, 3, 10, 11, 12, 13, 14, 15]], (1, 9), 15),
        ([("a", [1, 2], [10, 11, 12, 13])] * 3 + [("b", [3, 4], [14, 15, 16, 17])],
         [[1, 2] + [10, 11, 12, 13] * 3, [3, 4, 14, 15, 16, 17]], (2, 14), 24),
        ([("a", [1, 2], [10]), ("b", [1, 2], [11]), ("c", [1, 2], [12])],
         [[1, 2, 10], [1, 2, 11], [1, 2, 12]], (3, 3), 9),
        ([("a", [1, 2], [10]), ("b", [3, 4], [11]), ("a", [1, 2], [12])],
         [[1, 2, 10], [3, 4, 11], [1, 2, 12]], (3, 3), 9),
        ([("a", [1, 2, 3], [10]), ("a", [1, 2, 3], [11, 12, 13]),
          ("b", [4, 5], [14, 15])], [[1, 2, 3, 10, 11, 12, 13], [4, 5, 14, 15]], (2, 7), 18),
    ]
    for index, (rows, expected, shape, baseline) in enumerate(cases, 1):
        pmax = max(len(p) for _, p, _ in rows)
        rmax = max(len(r) for _, _, r in rows)
        prompts = torch.tensor([p + [0] * (pmax - len(p)) for _, p, _ in rows])
        responses = torch.tensor([r + [0] * (rmax - len(r)) for _, _, r in rows])
        batch = dict(prompts=prompts, responses=responses, response_mask=responses.ne(0),
                     uid=[uid for uid, _, _ in rows])
        pg, ids, mask, positions, _, _ = ns["build_pg_from_micro_batch"](batch, 0)
        assert tuple(ids.shape) == shape and len(rows) * (pmax + rmax) == baseline
        assert [ids[i][mask[i]].tolist() for i in range(len(expected))] == expected
        assert int(mask.sum()) == sum(map(len, expected))
        for i, group in enumerate(pg.group_info):
            want = list(range(group.prefix_len))
            for length in group.suffix_lens:
                want.extend(range(group.prefix_len, group.prefix_len + length))
            assert positions[i][mask[i]].tolist() == want
        print(f"case {index}: original_padded={baseline} grouped_padded={ids.numel()} valid={int(mask.sum())}")
    print("5 native input-layout cases passed; python", sys.version.split()[0], "torch", torch.__version__)
    print("NOT RUN: attention/model/gradient/FSDP/checkpointing/GPU or performance tests")


if __name__ == "__main__":
    main()
