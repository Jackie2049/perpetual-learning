#!/usr/bin/env python3
"""Pinned masked statistics: 3 original mean cases and 20 CPU/autograd controls.

Python >=3.10, CPU torch and pytest. AST-selects complete original functions and
original parametrized test; bodies, decorators and assertions are unchanged.
No verl initialization, model, estimator, optimizer, distributed or device run.
Pre-sanitized inputs below are diagnostic controls, not an upstream patch.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import pytest
import torch

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"


def selected(root, path, names):
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest())
    nodes = [node for node in ast.parse(data).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ModuleType("_native_masked_stats")
    module.__dict__.update(torch=torch, pytest=pytest)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{REV}:{path}", "exec"), module.__dict__)
    return module


def close(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), rtol=1e-6, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    native = selected(args.source_repo, "verl/utils/torch_functional.py",
                      {"masked_sum", "masked_mean", "masked_var", "masked_whiten"})
    original = selected(args.source_repo, "tests/utils/test_torch_functional.py", {"test_masked_mean"})
    original.masked_mean = native.masked_mean
    marks = original.test_masked_mean.pytestmark
    assert len(marks) == 1 and marks[0].name == "parametrize"
    assert marks[0].args[0] == "value,mask,gt" and len(marks[0].args[1]) == 3
    for values in marks[0].args[1]:
        original.test_masked_mean(*values)
    print("3 original parameter cases passed (direct calls, not full pytest)")
    for dtype in [torch.float32, torch.float64]:
        mask = torch.tensor([1.0, 1.0, 0.0], dtype=dtype)
        for bad in [float("nan"), float("inf")]:
            def leaf():
                return torch.tensor([1.0, 3.0, bad], dtype=dtype, requires_grad=True)

            x = leaf()
            mean = native.masked_mean(x, mask)
            mean.backward()
            close(mean, 2.0)
            close(x.grad, [0.5, 0.5, 0.0])
            x = leaf()
            squared = native.masked_mean(x.square(), mask)
            squared.backward()
            close(squared, 5.0)
            close(x.grad[:2], [1.0, 3.0])
            assert torch.isnan(x.grad[2])
            x = leaf()
            variance = native.masked_var(x, mask)
            variance.backward()
            close(variance, 2.0)
            assert torch.isnan(x.grad).all()
            print(str(dtype), str(bad), "native variance", variance.item(), "gradient", x.grad.tolist())
            x = leaf()
            safe = torch.where(mask.bool(), x, 0.0)
            safe_var = native.masked_var(safe, mask)
            safe_var.backward()
            close(safe_var, 2.0)
            close(x.grad, [-2.0, 2.0, 0.0])
            with torch.no_grad():
                whitened = native.masked_whiten(leaf(), mask)
                assert torch.isfinite(whitened[:2]).all() and not torch.isfinite(whitened[2])
                assert torch.isnan((whitened * mask)[2])
                close(native.masked_mean(whitened, mask), 0.0)
    print("20 CPU controls passed; sanitized input is a diagnostic control, not a patch or training fix")
    print("Python", sys.version.split()[0], "torch", torch.__version__, "pytest", pytest.__version__)


if __name__ == "__main__":
    main()
