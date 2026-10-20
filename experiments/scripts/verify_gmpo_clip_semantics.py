#!/usr/bin/env python3
"""Bounded GMPO full-surrogate NumPy checks; not torch/autograd or training.

Usage: python3 verify_gmpo_clip_semantics.py /path/to/verl-8250
Source hash/AST guards do not execute upstream bodies. Models independently
represent paper v1 Eq.4, Algorithm 1, and the pinned verl arithmetic shape.
Finite differences avoid clip kinks; advantages/masks are fixed inputs.
"""
import argparse
import ast
import hashlib
from pathlib import Path

import numpy as np


def source_guard(root):
    path = root / "verl/trainer/ppo/core_algos.py"
    print("core_algos.py sha256=" + hashlib.sha256(path.read_bytes()).hexdigest())
    fn = next(node for node in ast.parse(path.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == "compute_policy_loss_geo_mean")
    text = ast.unparse(fn)
    for expected in ("torch.sign(advantages)", "-cliprange_low, cliprange_high",
                     "sgn_advantage * negative_approx_kl_clamp", "response_mask_sum + 1e-08",
                     "torch.mean(pg_losses)"):
        assert expected in text, expected
    print("Source arithmetic shape guarded; original function NOT executed.")


def paper_eq4(z, mask, advantage, lower, upper):
    """Direct probability-space signed-ratio formula; nonempty rows only."""
    if advantage == 0:
        return 0.
    sign = np.sign(advantage)
    signed_ratio = np.exp(z) ** sign
    selected = np.minimum(signed_ratio, np.clip(signed_ratio, lower, upper)) ** sign
    return -advantage * np.exp(np.log(selected[mask]).mean())


def paper_algorithm(z, mask, advantage, epsilon):
    sign = np.sign(advantage)
    selected = sign * np.minimum(sign * z, np.clip(sign * z, -epsilon, epsilon))
    return -advantage * np.exp(selected[mask].mean())


def verl_model(z, mask, advantages, low, high, denominator_epsilon=0.):
    """Sequence PG branch only: no rollout IS, auxiliary loss or engine."""
    sign = np.sign(advantages)
    selected = sign * np.minimum(sign * z, sign * np.clip(z, -low, high))
    count = mask.sum(axis=-1) + denominator_epsilon
    ratio = np.exp((selected * mask).sum(axis=-1) / count)
    average_advantage = (advantages * mask).sum(axis=-1) / count
    return -average_advantage * ratio


def ratio_space_ppo_inside_geomean(z, mask, advantage, low, high):
    """Conventional original-ratio PPO min, not generic asymmetric Eq.4."""
    if advantage == 0:
        return 0.
    ratio = np.exp(z)
    selected = np.minimum(advantage * ratio, advantage * np.clip(ratio, low, high))
    return -np.sign(advantage) * np.exp(np.log(np.abs(selected[mask])).mean())


def fd_gradient(fn, point, step=1e-6):
    result = np.zeros_like(point, dtype=float)
    for index in np.ndindex(point.shape):
        delta = np.zeros_like(point, dtype=float)
        delta[index] = step
        result[index] = (fn(point + delta) - fn(point - delta)) / (2 * step)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    source_guard(args.source_root)
    rng = np.random.default_rng(20673)
    checks, max_stabilizer_delta = 0, 0.
    # Symmetric paper settings; partial padding, finite inputs, constant A.
    for width in (1, 2, 7, 16):
        for epsilon in (.0003, .2, .4):
            for advantage in (-2., 0., 1.5):
                z = rng.uniform(-1.2, 1.2, size=width)
                z[np.abs(np.abs(z) - epsilon) < 1e-4] += .01
                mask = rng.random(width) > .35
                mask[0] = True
                advantages = np.full(width, advantage)
                functions = (
                    lambda x: paper_eq4(x, mask, advantage, np.exp(-epsilon), np.exp(epsilon)),
                    lambda x: paper_algorithm(x, mask, advantage, epsilon),
                    lambda x: verl_model(x, mask, advantages, epsilon, epsilon),
                )
                values = [fn(z) for fn in functions]
                gradients = [fd_gradient(fn, z) for fn in functions]
                assert np.allclose(values, values[0], atol=1e-12, rtol=1e-12)
                for gradient in gradients[1:]:
                    assert np.allclose(gradient, gradients[0], atol=2e-8, rtol=2e-8)
                # Independent analytical derivative of the one-sided branch.
                active = z < epsilon if advantage > 0 else z > -epsilon
                expected = values[0] * mask * active / mask.sum()
                assert np.allclose(gradients[0], expected, atol=2e-8, rtol=2e-8)
                actual_shape = verl_model(z, mask, advantages, epsilon, epsilon, 1e-8)
                max_stabilizer_delta = max(max_stabilizer_delta, abs(actual_shape - values[0]))
                checks += 1
    print(f"Symmetric Eq.4 / Algorithm 1 / verl ideal model: {checks} value+gradient cases PASS.")
    print(f"Observed +1e-8 denominator max value delta={max_stabilizer_delta:.12g}; not exact equality.")

    # General original-ratio PPO windows transform exactly, WITH sign/min.
    for advantage in (-2., 1.5):
        z = np.array([-.9, -.3, .05, .8])
        mask = np.array([True, True, False, True])
        a = np.full(z.shape, advantage)
        lhs = lambda x: ratio_space_ppo_inside_geomean(x, mask, advantage, np.exp(-.1), np.exp(.4))
        rhs = lambda x: verl_model(x, mask, a, .1, .4)
        assert np.allclose(lhs(z), rhs(z), atol=1e-12, rtol=1e-12)
        assert np.allclose(fd_gradient(lhs, z), fd_gradient(rhs, z), atol=2e-8, rtol=2e-8)
    # Eq.4 clips r^sign instead: a common asymmetric window needs sign mapping.
    z, mask = np.array([-.3]), np.array([True])
    equation = paper_eq4(z, mask, -1., np.exp(-.1), np.exp(.4))
    implementation = verl_model(z, mask, np.array([-1.]), .1, .4)
    assert np.isclose(equation, np.exp(-.3)) and np.isclose(implementation, np.exp(-.1))
    print(f"Asymmetric signed-window control (not bug proof): Eq.4={equation:.9f}, verl={implementation:.9f}.")

    # Old clamp-only comparison misses one-sided clipping; opposite side live.
    for advantage, token in ((1., -.8), (-1., .8)):
        z = np.array([token])
        full = paper_algorithm(z, mask, advantage, .4)
        clamp_only = -advantage * np.exp(np.clip(z, -.4, .4).mean())
        assert not np.isclose(full, clamp_only)
        assert abs(fd_gradient(lambda x: paper_algorithm(x, mask, advantage, .4), z)[0]) > 0
    print("Clamp-only counterexamples: opposite-direction tails retain nonzero gradient.")

    # Zero A, fixed finite padding, unsupported varying A, and outer mean.
    z = np.array([[.8, -.8], [.1, -.2]])
    mask2 = np.array([[True, True], [True, False]])
    constant = np.array([[-1., -1.], [2., 2.]])
    base = verl_model(z, mask2, constant, .4, .4)
    padded = z.copy()
    padded[1, 1] = 100.
    assert np.allclose(base, verl_model(padded, mask2, constant, .4, .4))
    zero_mask = np.zeros_like(mask2)
    assert np.all(verl_model(z, zero_mask, constant, .4, .4, 1e-8) == 0)
    assert np.isclose(np.r_[base, 0.].mean(), base.mean() * 2 / 3)
    varying = np.array([.2, -.7])
    assert not np.isclose(verl_model(z[0], mask2[0], varying, .4, .4),
                          paper_algorithm(z[0], mask2[0], varying.mean(), .4))
    row_losses = np.array([1., 2., 4., 8.])
    chunks = [row_losses[:1], row_losses[1:]]
    assert not np.isclose(sum(chunk.mean() for chunk in chunks), row_losses.mean())
    assert np.isclose(sum(chunk.sum() / len(row_losses) for chunk in chunks), row_losses.mean())
    print("Mask/constant-A/outer-mean controls PASS; no distributed engine assertion.")
    print("PASS bounded surrogate models; no upstream execution, torch/autograd, GPU or training.")


if __name__ == "__main__":
    main()
