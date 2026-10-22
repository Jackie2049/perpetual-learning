#!/usr/bin/env python3
"""Pinned ReMax CPU arithmetic, not the PR's 14 tests or V1 integration.

Python >=3.10 and CPU torch. Selects complete original enum/registry/decorator,
estimator and greedy-parameter helper via AST, deferring annotations only.
No worker, queue, decoding, model, device discovery or dependency substitutes.
"""
import __future__
import argparse
import ast
from enum import Enum
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch

HEAD = "5faa09894b386768976176fecbad9eb72dfda11c"
BASE = "cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3"
CORE = "verl/trainer/ppo/core_algos.py"
AGENT = "verl/trainer/ppo/v1/agent_loop_tq.py"
HASHES = {
    CORE: "369b614f239f237a04a38a40b1877cd6fd14aa7bd5f9ff4fe4d7758b682a6948",
    AGENT: "04f8cc14e9556c48a01d22cb7b0437521b41e15adb9fdbdb1bce8c82950c908d",
    "verl/trainer/ppo/v1/trainer_base.py": "9da613864aea8129ebbfa73fea921091fb77abaa5796541139b69c0ae34ee8cb",
    "verl/trainer/ppo/ray_trainer.py": "95205837037beb4074a8b1cb5c60110bc11cadc46e7c31bf43d70a32d343290b",
    "tests/trainer/ppo/test_remax_v1_on_cpu.py": "4f0b8c9eeeb593bb8151b026f83820d965d25b05f1102fd0a58db34eee44a624",
}


def blob(root, revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))


def selected(data, path, names):
    nodes, found = [], set()
    for node in ast.parse(data).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.target.id if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else None)
        if name in names:
            nodes.append(node)
            found.add(name)
    assert found == names, (found, names)
    module = ModuleType("_native_remax")
    module.__dict__.update(torch=torch, Enum=Enum)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{HEAD}:{path}", "exec",
                 flags=__future__.annotations.compiler_flag), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    root = parser.parse_args().source_repo
    sources = {}
    for path, expected in HASHES.items():
        data = blob(root, HEAD, path)
        digest = hashlib.sha256(data).hexdigest()
        assert digest == expected, (path, digest)
        sources[path] = data
        print(path, "sha256=" + digest)
    assert sources[CORE] == blob(root, BASE, CORE)
    print("Core estimator module unchanged between pinned base and head")
    native = selected(sources[CORE], CORE, {"AdvantageEstimator", "ADV_ESTIMATOR_REGISTRY",
                      "register_adv_est", "get_adv_estimator_fn", "compute_remax_outcome_advantage"})
    estimator = native.get_adv_estimator_fn(native.AdvantageEstimator.REMAX)
    assert estimator is native.get_adv_estimator_fn("remax")
    rewards = torch.tensor([[0., 1., 0., 0.], [0., 0., 4., 0.]], dtype=torch.float64,
                           device="cpu", requires_grad=True)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]], dtype=torch.float64, device="cpu")
    returns_expected = torch.tensor([[1., 1., 0., 0.], [4., 4., 4., 0.]], dtype=torch.float64)
    # Synthetic external greedy reward, not an actual decoded response.
    baseline = torch.tensor([2., 2.], dtype=torch.float64, device="cpu", requires_grad=True)
    advantage, returns = estimator(rewards, baseline, mask)
    torch.testing.assert_close(returns, returns_expected)
    torch.testing.assert_close(advantage, torch.tensor([[-1., -1., 0., 0.], [2., 2., 2., 0.]],
                                                     dtype=torch.float64))
    assert not advantage.requires_grad and not returns.requires_grad
    # Deliberately replace the external baseline with sampled max: different result.
    max_advantage, max_returns = estimator(rewards, torch.tensor([4., 4.], dtype=torch.float64), mask)
    torch.testing.assert_close(max_returns, returns_expected)
    torch.testing.assert_close(max_advantage, torch.tensor([[-3., -3., 0., 0.], [0., 0., 0., 0.]],
                                                         dtype=torch.float64))
    print("Sampled rewards [1,4]; synthetic greedy baseline 2:", advantage[:, 0].tolist())
    print("Counterfactual sampled-max baseline 4:", max_advantage[:, 0].tolist())
    greedy = selected(sources[AGENT], AGENT, {"apply_greedy_sampling_params"})
    params = dict(temperature=1., top_p=.8, top_k=50, max_new_tokens=9)
    greedy.apply_greedy_sampling_params(params)
    assert params == dict(temperature=0, top_p=1., top_k=-1, max_new_tokens=9)
    print("Original greedy helper:", params)
    print("CPU controls passed; no V1 integration, PR-test, decoding or training claim")
    print("Python", sys.version.split()[0], "torch", torch.__version__)


if __name__ == "__main__":
    main()
