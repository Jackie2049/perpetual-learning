"""UniRL #563 original LM-head chunks and pure metrics: real CPU controls.

Usage: python -B verify_unirl_policy_entropy.py /path/to/UniRL
Requires Python 3.12+, torch, immutable local Git objects; no implicit fetch.
AST-selects original chunk/entropy/metric helpers, executes full ReplayResult and
metrics modules. The LM head is an explicit Linear fixture, NOT a Qwen backbone.
No outer replay (which probes CUDA), checkpoint loop, six algorithm execution,
TrainStack, Ray, distributed reduction, device discovery or hardware validation.
"""
import __future__
import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import torch
from torch import nn

HEAD = '00e5597f6ff6de639d7997a2ea21266d688e9f72'
BASE = '94e8f26b8768a3695b5985c2fa072e07d9dbfa0d'
repo = Path(sys.argv[1])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
torch.set_default_device('cpu')
torch.set_num_threads(1)


def source(path, rev=HEAD):
    data = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=repo, env=env)
    compile(data, path, 'exec')
    print(rev[:8], path, 'sha256=' + hashlib.sha256(data).hexdigest())
    return data


def execute(nodes, namespace, filename):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)


def module(path, name, rev=HEAD):
    result = types.ModuleType(name)
    sys.modules[name] = result
    exec(compile(source(path, rev), path, 'exec'), result.__dict__)
    return result


ReplayResult = module('unirl/models/types/replay_result.py', '_original_entropy_result').ReplayResult
reducer = module('unirl/utils/metrics.py', '_original_entropy_reducer').aggregate_numeric_metrics
old_reducer = module('unirl/utils/metrics.py', '_base_entropy_reducer', BASE).aggregate_numeric_metrics
ar_path = 'unirl/models/qwen3/ar.py'
ar_tree = ast.parse(source(ar_path))
entropy_node = next(n for n in ar_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_categorical_entropy')
forward_node = next(n for n in ar_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_replay_aware_forward')
temperature_node = next(n for n in forward_node.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'T' for t in n.targets))
chunks = [n for n in ast.walk(forward_node) if isinstance(n, ast.FunctionDef)
          and n.name in {'_flat_logp_chunk', '_logp_chunk'}]
assert len(chunks) == 2
ns = {'torch': torch, 'ReplayResult': ReplayResult}
execute([entropy_node], ns, ar_path)
entropy = ns['_categorical_entropy']
metric_tree = ast.parse(source('unirl/algorithms/base.py'))
metric_node = next(n for n in metric_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'policy_entropy_metrics')
execute([metric_node], ns, 'unirl/algorithms/base.py')
metrics = ns['policy_entropy_metrics']

comparisons = 0
for dtype in (torch.float32, torch.bfloat16):
    head = nn.Module()
    head.lm_head = nn.Linear(3, 5, dtype=dtype)
    with torch.no_grad():
        head.lm_head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 13 - .5)
        head.lm_head.bias.copy_(torch.linspace(-.2, .3, 5))
    for temperature in (-2., 0., .5, 1., 2.):
        for chunk_node in chunks:
            shape = (4, 3) if chunk_node.name == '_flat_logp_chunk' else (2, 2, 3)
            local = dict(ns, self=head, temperature=temperature)
            execute([temperature_node, chunk_node], local, ar_path)
            assert local['T'] == (temperature if temperature > 0 else 1.)
            fn = local[chunk_node.name]
            token = torch.tensor([0, 1, 3, 4]).reshape(shape[:-1])
            hidden = (torch.arange(12, dtype=torch.float32).reshape(shape) / 7 - .5).to(dtype).requires_grad_()
            inputs = (hidden, *head.parameters())
            local['return_entropy'] = False
            off = fn(hidden, token)
            off_grad = torch.autograd.grad(off.sum(), inputs)
            local['return_entropy'] = True
            on, value = fn(hidden, token)
            on_grad = torch.autograd.grad(on.sum(), inputs)
            torch.testing.assert_close(off, on, atol=0, rtol=0)
            for a, b in zip(off_grad, on_grad):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert value.dtype == torch.float32 and not value.requires_grad
            logits = head.lm_head(hidden).detach().float().double() / local['T']
            logp = logits.log_softmax(-1)
            reference = -(logp.exp() * logp).sum(-1)
            torch.testing.assert_close(value.double(), reference, atol=3e-7, rtol=3e-7)
            comparisons += 1
print('PASS:', comparisons, 'original CPU LM-head chunk comparisons; exact on/off logp and local gradients')

skew = torch.tensor([[math.log(.9), math.log(.1)]], dtype=torch.float32, requires_grad=True)
skew_entropy = entropy(skew).item()
assert abs(skew_entropy - (-.9 * math.log(.9) - .1 * math.log(.1))) < 1e-7
assert not math.isclose(skew_entropy, -math.log(.9), abs_tol=1e-3)
assert entropy(torch.tensor([[0., -float('inf')]])).item() == 0
print('full categorical entropy', skew_entropy, 'vs chosen-token surprisal', -math.log(.9), '; top-1 entropy=0')


def stats(values, mask=None):
    return metrics(ReplayResult(log_probs=torch.zeros_like(values), entropy=values), mask)


values = torch.tensor([1., 2., 3.], requires_grad=True)
assert stats(values)['policy_entropy'].item() == 2
assert stats(values, torch.tensor([.2, 0., -1.]))['policy_entropy_count'].item() == 2
assert stats(values, torch.tensor([True, False, True]))['policy_entropy_sum'].item() == 4
assert not stats(values)['policy_entropy_sum'].requires_grad
for value, mask in ((torch.empty(0), None),
                    (torch.tensor([float('nan'), float('inf'), 7.]), torch.zeros(3, dtype=torch.bool))):
    assert all(v.item() == 0 for v in stats(value, mask).values())
assert math.isnan(stats(torch.tensor([float('nan')]))['policy_entropy'].item())
for replay, mask, error in (
    (torch.zeros(1), None, TypeError),
    (ReplayResult(torch.zeros(1)), None, TypeError),
    (ReplayResult(torch.zeros(1), entropy=torch.zeros(2)), None, ValueError),
    (ReplayResult(torch.zeros(1), entropy=torch.zeros(1)), torch.ones(2), ValueError),
):
    try:
        metrics(replay, mask)
    except error:
        pass
    else:
        raise AssertionError('Expected original metrics input rejection')
print('PASS: mask boolean conversion, detach, empty/all-masked zeros, active NaN propagation and four input rejections')

a, b = entropy(torch.tensor([[0., 0., -float('inf'), -float('inf')], [0., 0., 0., 0.]]))
micro_a = stats(a.reshape(1))
micro_b = stats(b.repeat(3))
# Real CPU scalar readback; not the TrainStack batched host transfer or collective.
cpu = [{k: v.item() for k, v in m.items()} for m in (micro_a, micro_b)]
weighted = reducer(cpu)
hierarchical = reducer([reducer(cpu[:1]), reducer(cpu[1:])])
old = old_reducer(cpu)
expected = (a.item() + 3 * b.item()) / 4
assert weighted['policy_entropy_count'] == 4 and abs(weighted['policy_entropy'] - expected) < 1e-7
assert weighted == hierarchical
assert abs(old['policy_entropy'] - weighted['policy_entropy']) > .1
print('same synthetic micro dictionaries: base mean', old['policy_entropy'],
      'vs additive token mean', weighted['policy_entropy'], 'count', weighted['policy_entropy_count'])

changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, HEAD], cwd=repo, env=env).decode().splitlines()
paths = [p for p in changed if p.endswith('.py')]
for path in paths:
    data = subprocess.check_output(['git', 'show', f'{HEAD}:{path}'], cwd=repo, env=env)
    compile(data, path, 'exec')
assert len(paths) == 11
print('11 changed Python modules compile only; Python', sys.version.split()[0], 'torch', torch.__version__)
print('No outer replay, backbone, checkpoint loop, full algorithm/TrainStack, TP/SP/DP, Ray or GPU/NPU claim.')
