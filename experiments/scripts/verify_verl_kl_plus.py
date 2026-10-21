#!/usr/bin/env python3
"""Pinned KL straight-through helpers: 6 original cases, 12+2 CPU controls.

Python >=3.10, CPU torch and pytest. AST-selects two complete original KL
functions and two original tests with decorators/bodies unchanged. Direct calls,
not full pytest. No model, policy loss/aggregation, trainer, DP or device run.
Finite differences intentionally measure displayed forward values, which need
not match straight-through autograd; no unbiased-estimator claim is tested.
"""
import argparse
import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import pytest
import torch

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"


def selected(root, path, names, globals_=None):
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest())
    nodes = [node for node in ast.parse(data).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ModuleType("_native_kl_plus")
    module.__dict__.update(torch=torch, pytest=pytest)
    module.__dict__.update(globals_ or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{REV}:{path}", "exec"), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    native = selected(args.source_repo, "verl/trainer/ppo/core_algos.py", {"kl_penalty", "kl_penalty_forward"})
    tests = selected(args.source_repo, "tests/trainer/ppo/test_core_algos_on_cpu.py",
                     {"test_kl_penalty_straight_through_value_matches_base", "test_kl_penalty_k3_plus_uses_k2_gradient"},
                     dict(kl_penalty=native.kl_penalty))
    marks = tests.test_kl_penalty_straight_through_value_matches_base.pytestmark
    assert len(marks) == 1 and marks[0].name == "parametrize" and len(marks[0].args[1]) == 5
    for case in marks[0].args[1]:
        tests.test_kl_penalty_straight_through_value_matches_base(*case)
    tests.test_kl_penalty_k3_plus_uses_k2_gradient()
    print("6 original cases passed (direct calls, not full pytest)")
    for delta in [-30.0, -0.5, 0.5, 30.0]:
        for mode in ["k1+", "k3+", "abs+"]:
            ref = torch.tensor(-40.0, dtype=torch.float64, requires_grad=True)
            current = torch.tensor(-40.0 + delta, dtype=torch.float64, requires_grad=True)
            value = native.kl_penalty(current, ref, mode)
            base = native.kl_penalty(current, ref, mode[:-1])
            torch.testing.assert_close(value, base)
            grad, ref_grad = torch.autograd.grad(value, (current, ref))
            assert math.isclose(grad.item(), delta) and math.isclose(ref_grad.item(), -delta)
            if mode == "k3+" and abs(delta) == 30:
                base_grad, = torch.autograd.grad(base, current)
                assert value.item() == 10.0 and base_grad.item() == 0.0
            print(mode, "delta", delta, "value", value.item(), "grad", grad.item())
    for delta in [0.5, 30.0]:
        def forward(d):
            return native.kl_penalty(torch.tensor(-40.0 + d, dtype=torch.float64),
                                     torch.tensor(-40.0, dtype=torch.float64), "k3+").item()

        h = 1e-5
        finite_difference = (forward(delta + h) - forward(delta - h)) / (2 * h)
        expected = 1 - math.exp(-delta) if delta == 0.5 else 0.0
        assert math.isclose(finite_difference, expected, rel_tol=1e-6, abs_tol=1e-7)
        current = torch.tensor(-40.0 + delta, dtype=torch.float64, requires_grad=True)
        value = native.kl_penalty(current, torch.tensor(-40.0, dtype=torch.float64), "k3+")
        gradient, = torch.autograd.grad(value, current)
        assert abs(gradient.item() - finite_difference) > 1e-3
        print("k3+", "delta", delta, "finite difference", finite_difference, "autograd", gradient.item())
    print("12 value/gradient controls and 2 finite-difference controls passed")
    print("Python", sys.version.split()[0], "torch", torch.__version__, "pytest", pytest.__version__)


if __name__ == "__main__":
    main()
