"""Pinned UniRL #557: original correction branch + clip/reducer, CPU controls.

Usage: python -B verify_unirl_rollout_correction.py /path/to/UniRL
Python 3.12+ and torch, immutable local Git blobs; no implicit fetch.
Only original tensor math is executed. Two attribute containers provide cap and
saved behavior/mask tensors; they are NOT TextSegment, GRPO, a stage or trainer.
No prepare_segment/lifecycle, model, backward method, optimizer or distributed test.
"""
import __future__
import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

HEAD = '300170e99209f26908d637971b3146d2ac7f98ff'
BASE = '852c0454a6cb63592b2582358f88623870848df1'
repo = Path(sys.argv[1])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
torch.set_default_device('cpu')
torch.set_num_threads(1)


def source(path):
    data = subprocess.check_output(['git', 'show', f'{HEAD}:{path}'], cwd=repo, env=env)
    print(path, 'sha256=' + hashlib.sha256(data).hexdigest())
    compile(data, path, 'exec')
    return data


ns = {'torch': torch, 'math': math}
exec(compile(source('unirl/types/loss_agg.py'), 'loss_agg.py', 'exec'), ns)
base_tree = ast.parse(source('unirl/algorithms/base.py'))
helpers = [n for n in base_tree.body if isinstance(n, ast.FunctionDef)
           and n.name in {'select_active_tokens', '_grpo_clip_loss', 'aggregate_token_losses'}]
assert len(helpers) == 3
exec(compile(ast.Module(body=helpers, type_ignores=[]), 'original clip/reducer', 'exec',
             flags=__future__.annotations.compiler_flag), ns)
grpo_tree = ast.parse(source('unirl/algorithms/grpo.py'))
grpo = next(n for n in grpo_tree.body if isinstance(n, ast.ClassDef) and n.name == 'GRPO')
method = next(n for n in grpo.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_loss_and_backward')
branch = next(n for n in method.body if isinstance(n, ast.If)
              and any(isinstance(v, ast.Name) and v.id == 'behavior_logp' for v in ast.walk(n)))
branch_code = compile(ast.Module(body=[branch], type_ignores=[]), 'original correction branch', 'exec')
for path in ('unirl/algorithms/base.py', 'unirl/types/loss_agg.py'):
    ids = [subprocess.check_output(['git', 'rev-parse', f'{rev}:{path}'], cwd=repo, env=env).strip()
           for rev in (BASE, HEAD)]
    assert ids[0] == ids[1], path
    print('unchanged base/head', path, ids[0].decode())


def corrected(loss, old, behavior, mask, cap):
    local = dict(ns, self=SimpleNamespace(rollout_is_threshold=cap),
                 segment=SimpleNamespace(rollout_log_probs=behavior, loss_mask=mask),
                 old_logp=old, loss_per_elem=loss, ratio_metrics={})
    exec(branch_code, local)
    return local


lengths = torch.tensor([2, 3])
old = torch.full((5,), -5., dtype=torch.float64)
raw_weights = [.1, 1., 3., 5., .7]
behavior = old - torch.tensor(raw_weights, dtype=torch.float64).log()
ratios = torch.tensor([.5, .9, 1.1, 1.8, 1.], dtype=torch.float64)
advantages = torch.tensor([1., -2., 3., -4., .5], dtype=torch.float64)
comparisons = 0
for cap in (None, .5, 1., 2.):
    for mask in (None, torch.tensor([1., 0., 1., 1., 0.]), torch.zeros(5)):
        for mode in ('token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum-norm'):
            new = (old + ratios.log()).detach().requires_grad_()
            element, _ = ns['_grpo_clip_loss'](new_logp=new, old_logp=old, advantages=advantages,
                                              clip_range=.2, clip_range_high=.4, mask=mask)
            result = corrected(element, old, behavior, mask, cap)
            loss = ns['aggregate_token_losses'](result['loss_per_elem'], lengths=lengths,
                                                loss_mask=mask, loss_agg_mode=mode, horizon=3)
            grad = torch.autograd.grad(loss, new)[0]

            ref_new = new.detach().clone().requires_grad_()
            r = (ref_new - old).exp()
            unbounded = -advantages * r
            bounded = -advantages * r.clamp(.8, 1.4)
            per_token = torch.where(unbounded >= bounded, unbounded, bounded)
            # Independent scalar weights using exactly the production FP32 anchor operands.
            deltas = (old.float() - behavior.float()).tolist()
            weights = [1. if cap is None else math.exp(min(max(-20., min(20., d)), math.log(cap))) for d in deltas]
            weights = torch.tensor(weights, dtype=torch.float32)
            per_token = per_token * weights
            active = torch.ones(5, dtype=torch.float64) if mask is None else mask.double()
            masked = per_token * active
            if mode == 'token-mean':
                reference = masked.sum() / active.sum().clamp_min(1)
            elif mode == 'seq-mean-token-mean':
                reference = (masked[:2].sum() / active[:2].sum().clamp_min(1)
                             + masked[2:].sum() / active[2:].sum().clamp_min(1)) / 2
            else:
                reference = masked.sum() / 2 / 3
            ref_grad = torch.autograd.grad(reference, ref_new)[0]
            torch.testing.assert_close(loss, reference, atol=2e-7, rtol=2e-7)
            torch.testing.assert_close(grad, ref_grad, atol=2e-7, rtol=2e-7)
            if cap is not None:
                assert not result['weights'].requires_grad
                if mask is not None and not mask.any():
                    assert all(v.item() == 0 for v in result['ratio_metrics'].values())
            comparisons += 1
print('PASS:', comparisons, 'CPU value/local-gradient comparisons over 3 reducers, 3 masks and 4 cap settings')

# Isolate the original branch from PPO's own old-anchor dependence.
leaf_loss = torch.ones(2, requires_grad=True)
old_leaf = torch.tensor([-3., -3.], requires_grad=True)
behavior_leaf = torch.tensor([-3., -4.], requires_grad=True)
detached = corrected(leaf_loss, old_leaf, behavior_leaf, None, 2.)
grad_loss, grad_old, grad_behavior = torch.autograd.grad(
    detached['loss_per_elem'].sum(), (leaf_loss, old_leaf, behavior_leaf), allow_unused=True)
torch.testing.assert_close(grad_loss, torch.tensor([1., 2.]))
assert grad_old is None and grad_behavior is None
mean_loss = ns['aggregate_token_losses'](detached['loss_per_elem'], lengths=torch.tensor([2]),
                                         loss_mask=None, loss_agg_mode='token-mean', horizon=1)
assert mean_loss.item() == 1.5  # Original token count, NOT sum(weights).
print('PASS: multiplier detached from both anchor operands; original loss receives weights [1, 2]')

current = torch.tensor([-5.], requires_grad=True)
anchor = torch.tensor([-5.])
behavior_one = anchor - math.log(3)
clipped, _ = ns['_grpo_clip_loss'](new_logp=current, old_logp=anchor, advantages=torch.ones(1),
                                   clip_range=.2, clip_range_high=.4)
separate = corrected(clipped, anchor, behavior_one, None, 2.)['loss_per_elem']
combined, _ = ns['_grpo_clip_loss'](new_logp=current, old_logp=behavior_one, advantages=torch.ones(1),
                                    clip_range=.2, clip_range_high=.4)
torch.testing.assert_close(separate, torch.tensor([-2.]))
torch.testing.assert_close(combined, torch.tensor([-1.4]))
separate_grad = torch.autograd.grad(separate.sum(), current, retain_graph=True)[0]
combined_grad = torch.autograd.grad(combined.sum(), current)[0]
torch.testing.assert_close(separate_grad, torch.tensor([-2.]))
assert combined_grad.item() == 0
print('two-ratio control: clip then correction loss/grad=-2/-2; behavior-anchor clipping=-1.4/0')
for cap, expected in ((2., 1.), (.5, .5)):
    same_policy = corrected(torch.ones(2), torch.zeros(2), torch.zeros(2), None, cap)
    torch.testing.assert_close(same_policy['weights'], torch.full((2,), expected))
print('equal anchors: unit weights at cap=2, weights .5 at cap=.5 (allowed by positive-cap contract)')

for cap, expected, clipped_fraction in ((2., [math.exp(-20), 2.], .5),
                                        (math.exp(30), [math.exp(-20), math.exp(20)], 0.)):
    safety = corrected(torch.ones(2), torch.zeros(2), torch.tensor([100., -100.]), None, cap)
    torch.testing.assert_close(safety['weights'], torch.tensor(expected), atol=0, rtol=1e-6)
    assert safety['ratio_metrics']['rollout_is_clipped_fraction'].item() == clipped_fraction
print('PASS: log safety floor/ceiling and clipped-fraction comparison after safety clamp')

ess = []
for behavior_values in ([0., 0.], [-math.log(2)] * 2, [0., 0., -math.log(2), -math.log(2)]):
    size = len(behavior_values)
    value = corrected(torch.ones(size), torch.zeros(size), torch.tensor(behavior_values), None, 2.)
    ess.append(value['ratio_metrics']['rollout_is_micro_ess_fraction'].item())
assert ess[:2] == [1., 1.] and abs(ess[2] - .9) < 1e-7
print('micro ESS fractions', ess[:2], 'vs concatenated-tensor ESS fraction', ess[2])
print('Python', sys.version.split()[0], 'torch', torch.__version__)
print('No full GRPO/base-vs-head disabled parity, anchor lifecycle, model, optimizer, transport or device claim.')
