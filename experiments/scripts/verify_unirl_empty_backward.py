"""UniRL #598 original all-masked branches and independent optimizer CPU controls.

Usage: python -B verify_unirl_empty_backward.py /path/to/UniRL-git
Python 3.12+, torch; fixed local Git objects, no implicit fetch.
AST-selects ONLY the original AlgorithmStepResult and the two all-masked if
branches. Their tensors come from explicit scalar CPU graph fixtures, NOT a
stage/replay, complete algorithm, padding helper, TrainStack or FSDP runtime.
The optimizer controls use ordinary torch optimizers, not UniRL backend code.
"""
import __future__
import ast
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

BASE = '6c828f5ed81649153315d996f0f597e8808ff144'
HEAD = 'fa0c306378a6b4f8cfd6c8abcf8bd342d8ce1f73'
repo = Path(sys.argv[1])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
torch.set_default_device('cpu')
torch.set_num_threads(1)


def source(rev, path):
    raw = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=repo, env=env)
    compile(raw, path, 'exec')
    print(rev[:8], path, hashlib.sha256(raw).hexdigest())
    return ast.parse(raw)


def execute(nodes, ns):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, 'original selected nodes', 'exec',
                 flags=__future__.annotations.compiler_flag), ns)


ns = {'torch': torch, 'dataclass': dataclass, 'Mapping': Mapping, 'Any': Any}
base_tree = source(HEAD, 'unirl/algorithms/base.py')
result_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'AlgorithmStepResult')
execute([result_class], ns)
comparisons = 0
for rev in (BASE, HEAD):
    for name in ('gspo', 'ppo'):
        tree = source(rev, f'unirl/algorithms/{name}.py')
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name.upper())
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'compute_loss_and_backward')
        condition = 'not bool(valid.any())' if name == 'gspo' else 'active_count == 0'
        branch = next(n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == condition)
        wrapper = ast.parse('def selected(new_logp, new_values, valid, active_count): pass').body[0]
        wrapper.body = [branch]
        execute([wrapper], ns)
        for count in (1, 4):
            trunk, policy, value = [torch.nn.Parameter(torch.tensor(2., dtype=torch.float64)) for _ in range(3)]
            x = torch.arange(1, count + 1, dtype=torch.float64)
            new_logp = trunk * policy * x
            new_values = trunk * value * x
            result = ns['selected'](new_logp, new_values, torch.zeros(count, dtype=torch.bool), 0)
            assert result.loss == 0 and result.num_steps_or_tokens == 0 and result.metrics == {}
            assert result.has_backward == (rev == HEAD)
            expected = (trunk, policy, value) if name == 'ppo' else (trunk, policy)
            for parameter in expected:
                if rev == BASE:
                    assert parameter.grad is None
                else:
                    assert parameter.grad is not None and parameter.grad.item() == 0
            if name == 'gspo' or rev == BASE:
                assert value.grad is None
            comparisons += 1
print('PASS:', comparisons, 'original branch/finite CPU graph comparisons; no replay or FSDP')

# Counterfactual graph control: policy-only zero backward leaves an independent critic unused.
trunk, policy, value = [torch.nn.Parameter(torch.tensor(2.)) for _ in range(3)]
policy_output, value_output = trunk * policy, trunk * value
(policy_output * 0).backward()
assert trunk.grad.item() == policy.grad.item() == 0 and value.grad is None
print('PASS: policy-only backward does not allocate critic gradient')


def optimizer_control(kind, warm, zero):
    parameter = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    opt = (torch.optim.AdamW([parameter], lr=.1, weight_decay=.1, foreach=False)
           if kind == 'AdamW' else torch.optim.SGD([parameter], lr=.1, momentum=.9))
    if warm:
        parameter.grad = torch.ones_like(parameter)
        opt.step()
    opt.zero_grad(set_to_none=True)
    before = parameter.item()
    if zero:
        (parameter * 0).backward()
    opt.step()
    state = opt.state.get(parameter, {})
    after = parameter.item()
    if not zero:
        assert after == before
    if kind == 'AdamW':
        expected_steps = int(warm) + int(zero)
        assert int(state.get('step', 0)) == expected_steps
        if not warm and zero:
            assert abs(after - 1.98) < 1e-14  # Pure decoupled decay, no prior moments.
        if warm and zero:
            assert abs(state['exp_avg'].item() - .09) < 1e-14
            assert abs(state['exp_avg_sq'].item() - .000999) < 1e-14
            assert after < before * .99  # Additional residual-momentum update.
    else:
        assert abs(after - (1.81 if zero else 1.9)) < 1e-14
    print(kind, 'warm=' + str(warm), 'zero_grad=' + str(zero),
          'before=' + str(before), 'after=' + str(after),
          ('parameter_step=' + str(int(state.get('step', 0))) if kind == 'AdamW'
           else 'momentum_buffer=' + str(state.get('momentum_buffer'))))


for kind, warm in (('AdamW', False), ('AdamW', True), ('SGD', True)):
    for zero in (False, True):
        optimizer_control(kind, warm, zero)
print('PASS: 6 independent optimizer controls; Python', sys.version.split()[0], 'torch', torch.__version__)
print('No padding/normalization, complete algorithm, accumulation, collectives or device validation.')
