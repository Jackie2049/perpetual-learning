#!/usr/bin/env python3
"""Pinned native RL collator with one original test and seven CPU controls.

Python >=3.10, NumPy and CPU torch. AST-selects the complete collate_fn and
original test function; no function-body edits or RLHFDataset/tokenizer imports,
Ray/verl initialization, DataProto, StatefulDataLoader or worker processes.
Extra controls use real torch DataLoader, num_workers=0 and pin_memory=False.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.utils.data import DataLoader

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"


def selected(root, path, name, module):
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest())
    nodes = [node for node in ast.parse(data).body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{REV}:{path}", "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    assert "verl" not in sys.modules
    for name in ["verl", "verl.utils", "verl.utils.dataset"]:
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    native = ModuleType("verl.utils.dataset.rl_dataset")
    native.__dict__.update(torch=torch, np=np, defaultdict=defaultdict)
    sys.modules[native.__name__] = native
    selected(args.source_repo, "verl/utils/dataset/rl_dataset.py", "collate_fn", native)
    original = ModuleType("_original_collate_test")
    original.torch = torch
    selected(args.source_repo, "tests/utils/dataset/test_rl_collate_fn_on_cpu.py", "test_rl_collate_fn", original)
    original.test_rl_collate_fn()
    print("1 original test body passed (direct call, not full pytest)")

    def collate(items):
        return next(iter(DataLoader(items, batch_size=len(items), collate_fn=native.collate_fn,
                                    num_workers=0, pin_memory=False, shuffle=False)))

    equal = [[1, 2], [3, 4]]
    result = collate([{"x": item} for item in equal])["x"]
    assert result.shape == (2,) and result[0] is equal[0]
    assert np.array(equal, dtype=object).shape == (2, 2)
    print("equal-length Python lists: native shape", result.shape, "np.array shape", (2, 2))
    variable = [[1], [2, 3]]
    result = collate([{"x": item} for item in variable])["x"]
    assert result.shape == (2,) and result[1] is variable[1]
    result = collate([{"x": value} for value in [None, 7, "text"]])["x"]
    assert result.shape == (3,) and result.dtype == object and result.tolist() == [None, 7, "text"]
    items = [{"ids": torch.tensor([i]), "messages": [{"content": str(i)}]} for i in range(2)]
    result = collate(items)
    assert result["messages"][0] is items[0]["messages"]
    items[0]["messages"][0]["content"] = "changed"
    items[0]["ids"][0] = 99
    assert result["messages"][0][0]["content"] == "changed" and result["ids"][0, 0].item() == 0
    print("same-process mutable objects alias; stacked tensor storage is independent")
    result = collate([{"ids": torch.tensor([0])}, {"ids": torch.tensor([1]), "optional": "second"}])
    assert result["ids"].shape == (2, 1) and result["optional"].tolist() == ["second"]
    print("missing key: tensor batch 2, optional object array length", len(result["optional"]))
    result = collate([{"ids": torch.tensor([0]), "x": torch.tensor([10])},
                      {"ids": torch.tensor([1]), "x": [20]}])
    assert isinstance(result["x"], np.ndarray) and result["x"].shape == (1,) and result["x"][0] == [20]
    print("mixed tensor/non-tensor key: non-tensor dictionary wins; shape", result["x"].shape)
    try:
        collate([{"x": torch.tensor([1])}, {"x": torch.tensor([2, 3])}])
    except RuntimeError as error:
        assert "stack expects each tensor to be equal size" in str(error)
    else:
        raise AssertionError("ragged tensor stack unexpectedly accepted")
    print("7 DataLoader controls passed; no schema validation or full trainer claim")
    print("Python", sys.version.split()[0], "torch", torch.__version__, "NumPy", np.__version__)


if __name__ == "__main__":
    main()
