#!/usr/bin/env python3
"""Pinned GRPO group behavior: 4 original equivalence cases and 12 controls.

Python >=3.10, CPU torch, NumPy and pytest. AST-selects full original estimators,
enum/registry/decorator and groupwise helpers. Annotation evaluation is deferred;
bodies/decorators remain unchanged. Helpers receive explicit CPU devices, so the
unused device-discovery import is omitted. Not full pytest/registry integration,
policy loss, model training, GPU/DP or a recommendation for rollout.n.
--uniform-pr checks #8349's 16 original parameter cases at fixed base/head,
plus singleton controls; the original historical mode remains the default.
"""
import __future__
import argparse
import ast
from collections import defaultdict
from enum import Enum
import hashlib
from itertools import product
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import numpy as np
import pytest
import torch

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
UNIFORM_BASE = "a4b9354968705a441fb7da7367924b15b902e771"
UNIFORM_HEAD = "0e3c3e3667fc9764c6d0e554c2feb9a904d7519e"


def selected(root, path, names, globals_=None, rev=REV):
    data = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    compile(data, f"{rev}:{path}", "exec")
    print(path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    nodes, found = [], set()
    for node in ast.parse(data).body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.target.id if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else None)
        if name in names:
            nodes.append(node)
            found.add(name)
    assert found == names
    module = ModuleType("_native_grpo_groups")
    module.__dict__.update(torch=torch, np=np, pytest=pytest, random=random, os=os,
                           defaultdict=defaultdict, Enum=Enum)
    module.__dict__.update(globals_ or {})
    code = compile(ast.Module(body=nodes, type_ignores=[]), f"{rev}:{path}", "exec",
                   flags=__future__.annotations.compiler_flag)
    exec(code, module.__dict__)
    return module


def verify_uniform_pr(root):
    test_path = "tests/trainer/ppo/test_core_algos_on_cpu.py"
    for rev in (UNIFORM_BASE, UNIFORM_HEAD):
        print("Revision", rev, flush=True)
        helpers = selected(root, "verl/utils/groupwise.py",
                           {"_resolve_device", "_to_1d_numpy_object_array", "as_torch_index", "group_mean_std"},
                           rev=rev)
        native = selected(root, "verl/trainer/ppo/core_algos.py",
                          {"AdvantageEstimator", "ADV_ESTIMATOR_REGISTRY", "register_adv_est",
                           "compute_grpo_outcome_advantage", "compute_grpo_vectorized_outcome_advantage"},
                          dict(as_torch_index=helpers.as_torch_index, group_mean_std=helpers.group_mean_std), rev=rev)
        functions = {name: getattr(native, name) for name in
                     ("compute_grpo_outcome_advantage", "compute_grpo_vectorized_outcome_advantage")}
        original = selected(root, test_path,
                            {"test_grpo_uniform_reward_group_has_zero_advantage",
                             "test_grpo_uniform_group_next_to_mixed_group",
                             "test_grpo_vectorized_matches_original_for_low_variance_rewards"},
                            functions, rev=UNIFORM_HEAD)
        fn = original.test_grpo_uniform_reward_group_has_zero_advantage
        marks = fn.pytestmark
        assert len(marks) == 2 and all(mark.name == "parametrize" for mark in marks)
        cases = [dict(zip([mark.args[0] for mark in marks], values, strict=True))
                 for values in product(*(mark.args[1] for mark in marks))]
        assert len(cases) == 15
        failed = []
        for case in cases:
            try:
                fn(**case)
            except AssertionError:
                failed.append(case)
        try:
            original.test_grpo_uniform_group_next_to_mixed_group()
        except AssertionError:
            failed.append({"mixed": True})
        assert (not failed) if rev == UNIFORM_HEAD else bool(failed)
        print("Original #8349 cases:", 16 - len(failed), "passed;", len(failed), "failed", failed)
        original.test_grpo_vectorized_matches_original_for_low_variance_rewards()
        for norm in (True, False):
            rewards = torch.tensor([[2.0], [-3.0]])
            mask = torch.ones_like(rewards)
            expected = rewards / (1 + 1e-6) if norm else rewards
            for function in functions.values():
                advantage, returns = function(rewards, mask, np.array(["positive", "negative"], dtype=object),
                                              norm_adv_by_std_in_grpo=norm)
                torch.testing.assert_close(advantage, expected)
                torch.testing.assert_close(returns, expected)
        print("Original low-variance case and singleton +/- controls (both normalization modes) passed")
        for reward, size in ((0.3, 16), (0.7, 64), (0.9, 64)):
            rewards = torch.full((size, 1), reward)
            for name, function in functions.items():
                advantage, _ = function(rewards, torch.ones_like(rewards), np.array(["g"] * size))
                if rev == UNIFORM_HEAD:
                    assert torch.equal(advantage, torch.zeros_like(advantage))
                print("Witness", reward, size, name, advantage[0, 0].item())
    print("Direct original test calls/AST functions with real CPU tensors, not full pytest/training/DP/GPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--uniform-pr", action="store_true")
    args = parser.parse_args()
    if args.uniform_pr:
        verify_uniform_pr(args.source_repo)
        return
    helpers = selected(args.source_repo, "verl/utils/groupwise.py",
                       {"_resolve_device", "_to_1d_numpy_object_array", "as_torch_index", "group_mean_std"})
    native = selected(args.source_repo, "verl/trainer/ppo/core_algos.py",
                      {"AdvantageEstimator", "ADV_ESTIMATOR_REGISTRY", "register_adv_est",
                       "compute_grpo_outcome_advantage", "compute_grpo_vectorized_outcome_advantage"},
                      dict(as_torch_index=helpers.as_torch_index, group_mean_std=helpers.group_mean_std))
    functions = {name: getattr(native, name) for name in
                 ["compute_grpo_outcome_advantage", "compute_grpo_vectorized_outcome_advantage"]}
    original = selected(args.source_repo, "tests/trainer/ppo/test_core_algos_on_cpu.py",
                        {"_make_group_index", "_rand_mask", "test_grpo_and_vectorized_equivalence",
                         "test_grpo_vectorized_matches_original_for_low_variance_rewards"}, functions)
    original.test_grpo_vectorized_matches_original_for_low_variance_rewards()
    marks = original.test_grpo_and_vectorized_equivalence.pytestmark
    assert len(marks) == 1 and marks[0].name == "parametrize" and len(marks[0].args[1]) == 3
    for case in marks[0].args[1]:
        original.test_grpo_and_vectorized_equivalence(*case)
    print("4 original equivalence cases passed (direct calls, not full pytest)", flush=True)
    cases = [("singleton positive", [2.0], ["s"], [2.0], [2.0]),
             ("singleton negative", [-3.0], ["s"], [-3.0], [-3.0]),
             ("singleton zero", [0.0], ["s"], [0.0], [0.0]),
             ("equal pair", [4.0, 4.0], ["t", "t"], [0.0, 0.0], [0.0, 0.0]),
             ("varied pair", [1.0, 3.0], ["g", "g"], [-1.0, 1.0], [-1.0, 1.0]),
             ("mixed groups", [2.0, 1.0, 3.0, 4.0, 4.0], ["s", "g", "g", "t", "t"],
              [2.0, -1.0, 1.0, 0.0, 0.0], [2.0, -1.0, 1.0, 0.0, 0.0])]
    for norm in [True, False]:
        for title, scores, labels, normalized_numerators, unnormalized in cases:
            rewards = torch.zeros(len(scores), 3)
            rewards[:, 1] = torch.tensor(scores)
            mask = torch.tensor([[1.0, 1.0, 0.0]] * len(scores))
            expected = []
            for value, label, unnorm in zip(normalized_numerators, labels, unnormalized, strict=True):
                denom = (1.0 if labels.count(label) == 1 else math.sqrt(2.0)) + 1e-6
                expected.append(value / denom if norm else unnorm)
            expected = torch.tensor(expected).unsqueeze(-1) * mask
            outputs = []
            for fn in functions.values():
                advantage, returns = fn(rewards.clone(), mask, np.array(labels, dtype=object),
                                        epsilon=1e-6, norm_adv_by_std_in_grpo=norm)
                torch.testing.assert_close(advantage, expected, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(returns, expected, rtol=1e-6, atol=1e-6)
                assert torch.isfinite(advantage).all() and not advantage.requires_grad
                outputs.append(advantage)
            torch.testing.assert_close(*outputs)
            print(title, "norm", norm, "first-token advantage", outputs[0][:, 0].tolist())
    print("12 dual-implementation group controls passed; not a training or convergence result")
    print("Python", sys.version.split()[0], "torch", torch.__version__, "NumPy", np.__version__)


if __name__ == "__main__":
    main()
