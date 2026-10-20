#!/usr/bin/env python3
"""Bounded NumPy GSPO/GRPO diagnostic models; no torch or training.

Replaces the invalid old comparison with an unnormalized sequence product.
Paper: https://arxiv.org/html/2507.18071v2 (Eq.7 and Eq.14).
Usage: python3 verify_gspo_detach_gradient.py /path/to/verl-8250
Source shapes/hashes are checked; upstream bodies are NOT executed.
Stop-gradient is represented by fixed base values in finite differences.
"""
import argparse
import ast
import hashlib
from pathlib import Path

import numpy as np


def source_guard(root):
    core = root / "verl/trainer/ppo/core_algos.py"
    helper = root / "verl/trainer/ppo/rollout_corr_helper.py"
    for path in (core, helper):
        print(f"{path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    functions = {n.name: n for n in ast.parse(core.read_text()).body if isinstance(n, ast.FunctionDef)}
    grpo = ast.unparse(functions["compute_grpo_outcome_advantage"])
    gspo = ast.unparse(functions["compute_policy_loss_gspo"])
    assert "torch.std(scores_tensor)" in grpo and "norm_adv_by_std_in_grpo" in grpo
    assert "/ seq_lengths" in gspo and "negative_approx_kl_seq.detach()" in gspo
    assert "log_prob - log_prob.detach()" in gspo and "torch.maximum(pg_losses1, pg_losses2)" in gspo
    assert 'metrics["kl"]' in helper.read_text()


def fd_gradient(fn, point, step=1e-6):
    result = np.zeros_like(point, dtype=float)
    for j in range(point.size):
        delta = np.zeros_like(point, dtype=float)
        delta[j] = step
        result[j] = (fn(point + delta) - fn(point - delta)) / (2 * step)
    return result


def clipped_loss(ratio, advantage):
    return np.maximum(-advantage * ratio, -advantage * np.clip(ratio, 1 - 3e-4, 1 + 4e-4))


def grpo_advantage(rewards):
    # Standard n>1, finite-reward, std-normalized branch only.
    return (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    source_guard(args.source_root)
    rng = np.random.default_rng(8250)
    checks = 0
    # Constant response A, equal valid-token mean, away from clipping/max10
    # kinks. Not arbitrary full-loss equality or a full trainer test.
    for length in (1, 2, 7, 16):
        for ratio in (.5, 1., 2.):
            for advantage in (-1., 0., 1.):
                base = rng.normal(size=length)
                old = base - np.log(ratio)
                frozen_mean = (base - old).mean()
                sequence = lambda x: clipped_loss(np.exp((x - old).mean()), advantage)
                token = lambda x: clipped_loss(np.exp(x - base + frozen_mean), advantage).mean()
                assert np.allclose(sequence(base), token(base))
                assert np.allclose(fd_gradient(sequence, base), fd_gradient(token, base), atol=2e-7)
                checks += 1
    print(f"GSPO / GSPO-token: {checks} restricted value/objective-gradient cases agree.")
    derivatives = {(a, r): fd_gradient(lambda x: clipped_loss(np.exp(x[0]), a), np.array([np.log(r)]))[0]
                   for a, r in ((1., 2.), (-1., 2.), (1., .5), (-1., .5))}
    assert np.allclose(list(derivatives.values()), [0., 2., -.5, 0.], atol=1e-7)
    print("Outside-window d(loss)/d(log ratio):", derivatives)
    # Token-wise changes can cancel in the sequence mean; not a KL bound.
    assert np.isclose(np.exp(np.array([.2, -.2]).mean()), 1.)
    bound = 7 / np.sqrt(8)  # sample std; epsilon only reduces magnitude
    for rewards in rng.normal(size=(1000, 8)):
        assert np.max(np.abs(grpo_advantage(rewards))) <= bound + 1e-12
    extreme = grpo_advantage(np.array([1e8] + [0.] * 7))
    assert np.isclose(extreme[0], bound)
    assert np.all(grpo_advantage(np.ones(8) * .28) == 0)
    total = np.array([0., .1, .2, .3, 0., .1, .2, .3])
    assert np.any(grpo_advantage(total) != 0)  # result=0, but other components vary
    assert derivatives[(1., 2.)] == 0  # A!=0 can still give zero PG derivative
    print(f"Finite normalized GRPO n=8: |A| <= {bound:.9f}; zero result alone does not force A=0.")
    theta = 1.
    probability = 1 / (1 + np.exp(-theta))
    assert theta * probability * (1 - probability) > 0  # toy KL to Bernoulli(.5)
    print("Zero A kills isolated PG, not arbitrary auxiliary objectives.")
    print("PASS analytical guardrails; no torch/autograd, real masks, optimizer, model or causal diagnosis.")


if __name__ == "__main__":
    main()
