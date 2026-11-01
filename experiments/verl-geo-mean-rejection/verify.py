"""Five own CPU gradient controls for #8421's original geometric-mean loss.

Fixed weights are inputs, not an execution of IcePop's weight producer or the
original regression. No model, trainer, registry, configuration validation or DP.
"""

import argparse
import ast
import __future__
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import torch

BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
HEAD = "aa5c8c037f6e04b7722c08cde2af8d04028a4528"
CORE = "verl/trainer/ppo/core_algos.py"
HELPER = "verl/utils/torch_functional.py"
TEST = "tests/trainer/ppo/test_core_algos_on_cpu.py"
NAME = "compute_policy_loss_geo_mean"


def select(raw, names, ns, path, remove_decorators=False):
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    if remove_decorators:
        for node in nodes:
            node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources, hashes = {}, {}
    for ref, path in ((BASE, CORE), (HEAD, CORE), (HEAD, HELPER), (HEAD, TEST)):
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), f"{ref}:{path}", "exec")
        sources[ref, path] = raw
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
    helpers = dict(torch=torch)
    select(sources[HEAD, HELPER], {"masked_sum", "masked_mean"}, helpers, HELPER)
    functions = {}
    for ref in (BASE, HEAD):
        ns = dict(torch=torch, verl_F=SimpleNamespace(masked_mean=helpers["masked_mean"]),
                  AlgoConfig=type("UnusedAlgoConfigGuard", (), {}))
        select(sources[ref, CORE], {NAME}, ns, CORE, True)
        functions[ref] = ns[NAME]

    def evaluate(ref, weights, delta=None, advantage=None, mask=None):
        old = torch.full_like(weights, -1.0)
        logp = (old + (torch.zeros_like(old) if delta is None else delta)).detach().requires_grad_(True)
        mask = torch.ones_like(old) if mask is None else mask
        advantage = torch.ones_like(old) if advantage is None else advantage
        before = tuple(t.clone() for t in (weights, old, mask, advantage))
        loss, metrics = functions[ref](old, logp, advantage, mask,
                                      config=SimpleNamespace(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2),
                                      rollout_is_weights=weights)
        loss.backward()
        for actual, expected in zip((weights, old, mask, advantage), before, strict=True):
            assert torch.equal(actual, expected)
        assert torch.isfinite(loss) and torch.isfinite(logp.grad).all()
        return float(loss.detach()), logp.grad.detach().clone(), metrics

    controls = []
    length_reports = []
    for length in (16, 64, 256):
        weights = torch.ones(2, length)
        weights[1, 5] = 0
        versions = {}
        for ref in (BASE, HEAD):
            loss, gradient, metrics = evaluate(ref, weights)
            kept_ratio = float(gradient[1, 0] / gradient[0, 0])
            if ref == HEAD:
                assert gradient[1, 5] == 0 and kept_ratio == 1
            else:
                assert gradient[1, 5] < 0
                assert math.isclose(kept_ratio, math.exp(math.log(1e-10) / length), rel_tol=1e-6)
            versions[ref] = dict(loss=loss, rejected_gradient=float(gradient[1, 5]),
                                 kept_gradient=float(gradient[1, 0]), kept_ratio=kept_ratio, metrics=metrics)
        length_reports.append(dict(length=length, versions=versions))
    controls.append(dict(case="one_zero_weight_unit_kept_weights", observations=length_reports))
    weights = torch.ones(2, 16, dtype=torch.float64)
    weights[1] = 0
    # Original denominators include 1e-8 and log(weights+1e-10), so the clean
    # half-batch contribution is near -0.5, not bitwise exactly -0.5 in FP64.
    clean_half_loss = evaluate(HEAD, torch.ones(1, 16, dtype=torch.float64))[0] / 2
    rejected = {}
    for ref in (BASE, HEAD):
        loss, gradient, _ = evaluate(ref, weights)
        if ref == HEAD:
            assert loss == clean_half_loss and torch.count_nonzero(gradient[1]) == 0
        else:
            assert torch.count_nonzero(gradient[1]) == 16 and loss < clean_half_loss, (
                loss, int(torch.count_nonzero(gradient[1])), float(gradient[1].abs().max()))
        rejected[ref] = dict(loss=loss, rejected_gradient_max=float(gradient[1].abs().max()),
                             rejected_gradient_nonzeros=int(torch.count_nonzero(gradient[1])),
                             clean_half_batch_reference=clean_half_loss)
    controls.append(dict(case="fully_rejected_sequence_float64", observations=rejected))
    positive = torch.linspace(.5, 2, 32, dtype=torch.float64).reshape(2, 16)
    delta = torch.linspace(-.3, .3, 32, dtype=torch.float64).reshape(2, 16)
    advantage = torch.tensor([1., -.7], dtype=torch.float64)[:, None].expand_as(positive)
    mask = torch.ones_like(positive)
    mask[1, -3:] = 0
    baseline = evaluate(BASE, positive, delta, advantage, mask)
    candidate = evaluate(HEAD, positive, delta, advantage, mask)
    assert baseline[0] == candidate[0] and torch.equal(baseline[1], candidate[1]) and baseline[2] == candidate[2]
    controls.append(dict(case="all_positive_weights_compatibility", loss=candidate[0],
                         exact_loss_gradient_metrics_equal=True, finite_padding=True,
                         sequence_advantages=[1., -.7], metrics=candidate[2]))
    weights = torch.ones(2, 16, dtype=torch.float64)
    weights[1, 5] = 0
    perturbed = torch.zeros_like(weights)
    perturbed[1, 5] = .3
    clean = evaluate(HEAD, weights)
    changed = evaluate(HEAD, weights, perturbed)
    old_changed = evaluate(BASE, weights, perturbed)
    assert clean[0] == changed[0] and torch.equal(clean[1], changed[1])
    assert clean[2] != changed[2] and changed[2] == old_changed[2]
    controls.append(dict(case="rejected_position_still_in_diagnostics", loss_unchanged=changed[0],
                         gradient_exactly_unchanged=True, clean_metrics=clean[2],
                         perturbed_metrics=changed[2], baseline_candidate_metrics_equal=True))
    weights = torch.full((2, 16), 4., dtype=torch.float64)
    weights[1, 5] = 0
    loss, gradient, _ = evaluate(HEAD, weights)
    kept_ratio = float(gradient[1, 0] / gradient[0, 0])
    assert math.isclose(kept_ratio, 4 ** (-1 / 16), rel_tol=1e-9)
    assert gradient[1, 5] == 0 and kept_ratio < 1
    controls.append(dict(case="nonunit_kept_weights_fixed_denominator", loss=loss,
                         rejected_gradient=0., kept_ratio=kept_ratio,
                         expected_ignoring_log_epsilon=4 ** (-1 / 16),
                         interpretation="No log(1e-10) contribution, but original length denominator remains. "
                         "The kept-gradient ratio need not be one when other IS weights are not one."))
    assert len(controls) == 5
    print(json.dumps(dict(base=BASE, head=HEAD, python=platform.python_version(), torch=torch.__version__,
                         torch_git=torch.version.git_version, source_sha256=hashes, own_controls=controls,
                         scope="Two complete original geo_mean loss bodies, registry decorators removed; "
                         "original masked_sum/mean, real CPU autograd. Annotations postponed, config is an "
                         "attribute-only namespace; AlgoConfig guard type is an import seam, not validation. "
                         "Five own control groups with fixed finite weights/log-probs, not original tests or "
                         "IcePop weight producer. Four full inputs compile. No vanilla loss, actual ActorConfig, "
                         "original regression/pytest, rollout helper/DataProto, trainer, DP/CP, model, GPU/NPU, "
                         "optimizer, training, performance or non-finite-input safety executed."), indent=2))


if __name__ == "__main__":
    main()
