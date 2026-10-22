"""Pinned original UniRL reducer: four CPU forward/local-autograd controls.

Usage: python -B verify_unirl_loss_reduction.py /path/to/UniRL
Requires Python >=3.10 and CPU torch. Full original loss-mode module and AST-selected
original aggregate_token_losses only; explicit single-process micro weights.
Not loss_units/resolve_loss_scales, an upstream test, FSDP or training validation.
"""
import __future__
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import torch

repo = Path(sys.argv[1])
head = '9dded4e2fd92732dd9de40252a4ae5b342b164e0'
base = '6fd82cba13f52f394747832344039408699ea378'
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
def source(rev, path):
    data = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=repo, env=env)
    print(rev[:8], path, hashlib.sha256(data).hexdigest())
    return data
namespace = {'torch': torch}
exec(compile(source(head, 'unirl/types/loss_agg.py'), 'loss_agg.py', 'exec'), namespace)
def reduction(data):
    return next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef)
                and n.name == 'aggregate_token_losses')
node = reduction(source(head, 'unirl/algorithms/base.py'))
assert ast.dump(node) == ast.dump(reduction(source(base, 'unirl/algorithms/base.py')))
exec(compile(ast.Module(body=[node], type_ignores=[]), 'original aggregate_token_losses', 'exec',
             flags=__future__.annotations.compiler_flag), namespace)
source(head, 'unirl/train/stack/loss_scale.py')
aggregate = namespace['aggregate_token_losses']
lengths = torch.tensor([2, 3], device='cpu')
for label, mask, totals in [
    ('unequal valid tokens', [1., 0., 1., 1., 1.], [5., 4.]),
    ('fully masked first row', [0., 0., 1., 1., 1.], [6., 3.]),
]:
    mask = torch.tensor(mask, dtype=torch.float64, device='cpu')
    valid = [float(mask[:2].sum()), float(mask[2:].sum())]
    for mode, expected in zip(['token-mean', 'seq-mean-token-mean'], totals):
        values = torch.tensor([2., 999., 4., 6., 8.], dtype=torch.float64, device='cpu', requires_grad=True)
        direct = aggregate(values, lengths=lengths, loss_mask=mask, loss_agg_mode=mode, horizon=1)
        weights = [n / sum(valid) for n in valid] if mode == 'token-mean' else [.5, .5]
        micros = [aggregate(values[s:e], lengths=lengths[i:i+1], loss_mask=mask[s:e],
                            loss_agg_mode=mode, horizon=1) for i, (s, e) in enumerate([(0, 2), (2, 5)])]
        recomposed = sum(w * m for w, m in zip(weights, micros))
        torch.testing.assert_close(direct, torch.tensor(expected, dtype=torch.float64))
        torch.testing.assert_close(recomposed, direct, rtol=1e-12, atol=1e-12)
        grad = torch.autograd.grad(direct, values, retain_graph=True)[0]
        split_grad = torch.autograd.grad(recomposed, values)[0]
        torch.testing.assert_close(split_grad, grad, rtol=1e-12, atol=1e-12)
        assert grad[1] == 0
        print(label, mode, 'valid', valid, 'weights', weights, 'loss', direct.item(), 'grad', grad.tolist())
print('PASS: unchanged original reducer, four CPU forward/local-autograd controls')
print('Weights are explicit single-process arithmetic, not resolve_loss_scales, loss_units, FSDP or training execution')
print('Python', sys.version.split()[0], 'torch', torch.__version__)
