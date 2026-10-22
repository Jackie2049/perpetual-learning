"""Pinned UniRL #593 selection plus real CPU autograd/optimizer controls.

Usage: python -B verify_unirl_block_freeze.py /path/to/UniRL
Python 3.12+ / torch required; immutable local Git blobs, no network fallback.
Runs the original two selection statements on a declared three-linear fixture,
the full original optimizer/config modules, and original checkpoint helper AST.
Not an H3 constructor/model, LoRA, FSDP, replay, resume or hardware test.
"""
import __future__
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types
import warnings

import torch
from torch import nn

HEAD = 'da8d285ac529c2f06819bf3a098e3cc657ae6618'
BASE = '2d9e57fdee65ea69254fdb958d3fca9ce4872661'
repo = Path(sys.argv[1])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
torch.set_default_device('cpu')
torch.set_num_threads(1)


def source(path):
    data = subprocess.check_output(['git', 'show', f'{HEAD}:{path}'], cwd=repo, env=env)
    print(path, 'sha256=' + hashlib.sha256(data).hexdigest())
    compile(data, path, 'exec')
    return data


def original_module(name, path):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source(path), path, 'exec'), module.__dict__)
    return module


bundle_tree = ast.parse(source('unirl/models/minimax_h3/bundle.py'))
selection = next(n for n in ast.walk(bundle_tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Attribute) and n.test.attr == 'full_finetune_blocks')
assert len(selection.body) == 2 and not selection.orelse
config_tree = ast.parse(source('unirl/models/minimax_h3/config.py'))
flag = next(n for n in ast.walk(config_tree) if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name) and n.target.id == 'full_finetune_blocks')
assert isinstance(flag.value, ast.Constant) and flag.value.value is False

base_module = original_module('unirl.train.backend.base', 'unirl/train/backend/base.py')
optim_module = original_module('unirl.train.optim', 'unirl/train/optim.py')
wrap_path = 'unirl/train/backend/fsdp/wrap.py'
wrap_tree = ast.parse(source(wrap_path))
helpers = [n for n in wrap_tree.body if isinstance(n, ast.FunctionDef)
           and n.name in {'_clone_checkpoint_kwarg', '_checkpoint_with_kwarg_snapshots'}]
assert len(helpers) == 2
helper_namespace = {}
exec(compile(ast.Module(body=helpers, type_ignores=[]), wrap_path, 'exec',
             flags=__future__.annotations.compiler_flag), helper_namespace)
checkpoint = helper_namespace['_checkpoint_with_kwarg_snapshots']

# These production files were NOT changed by #593.
for path in ('unirl/train/optim.py', wrap_path,
             'unirl/models/minimax_h3/vendor/transformer_minimax_h3.py'):
    ids = [subprocess.check_output(['git', 'rev-parse', f'{rev}:{path}'], cwd=repo, env=env).strip()
           for rev in (BASE, HEAD)]
    assert ids[0] == ids[1], path
    print('unchanged base/head', path, ids[0].decode())


class ThreeLinear(nn.Module):
    """A mathematical fixture, deliberately not a tiny H3 implementation."""
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.transformer_blocks = nn.ModuleList([nn.Linear(1, 1, bias=False, dtype=torch.float64)])
        self.proj_out = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for layer, value in ((self.proj_in, 3), (self.transformer_blocks[0], 5), (self.proj_out, 7)):
                layer.weight.fill_(value)


def selected(enabled=True):
    model = ThreeLinear().train()
    ids = [id(p) for p in model.parameters()]
    # Only the original conditional/selection; no bundle construction or loader.
    exec(compile(ast.Module(body=[selection], type_ignores=[]), 'original block selection', 'exec'),
         {'config': types.SimpleNamespace(full_finetune_blocks=enabled), 'transformer': model})
    assert ids == [id(p) for p in model.parameters()]
    assert all(m.training for m in model.modules())
    return model


def forward(model, checkpoint_fn=None):
    hidden = model.proj_in(torch.tensor([[2.0]], dtype=torch.float64))
    block = model.transformer_blocks[0]
    hidden = block(hidden) if checkpoint_fn is None else checkpoint_fn(block, hidden)
    return model.proj_out(hidden)


def backward_rejected(value):
    assert not value.requires_grad
    try:
        value.sum().backward()
    except RuntimeError as error:
        assert 'does not require grad' in str(error), str(error)
        return
    raise AssertionError('Expected disconnected autograd graph')


default = selected(False)
assert all(p.requires_grad for p in default.parameters())
model = selected()
assert [name for name, p in model.named_parameters() if p.requires_grad] == ['transformer_blocks.0.weight']
output = forward(model)
reference = forward(default)
torch.testing.assert_close(output, reference, atol=0, rtol=0)
output.sum().backward()
reference.sum().backward()
torch.testing.assert_close(model.transformer_blocks[0].weight.grad, torch.tensor([[42.0]], dtype=torch.float64))
torch.testing.assert_close(model.transformer_blocks[0].weight.grad, default.transformer_blocks[0].weight.grad)
assert model.proj_in.weight.grad is None and model.proj_out.weight.grad is None
print('selection/autograd: output=210; block grad=42; frozen peripheral grads=None; train mode unchanged')

config = base_module.OptimizerConfig(learning_rate=.01, adam_beta1=.9, adam_beta2=.999,
                                   adam_epsilon=1e-8, weight_decay=.1)
optimizer = optim_module.build_optimizer(config, params=model.parameters())
weight = model.transformer_blocks[0].weight
assert [id(p) for group in optimizer.param_groups for p in group['params']] == [id(weight)]
optimizer.step()
expected = 5 * (1 - .01 * .1) - .01 * 42 / (42 + 1e-8)
torch.testing.assert_close(weight, torch.tensor([[expected]], dtype=torch.float64), atol=1e-12, rtol=1e-12)
assert model.proj_in.weight.item() == 3 and model.proj_out.weight.item() == 7
assert list(optimizer.state) == [weight]
print('original optimizer: only block updated to', weight.item(), '; peripherals unchanged')

cut = selected()
hidden = cut.transformer_blocks[0](cut.proj_in(torch.tensor([[2.0]], dtype=torch.float64)))
with torch.no_grad():
    detached_output = cut.proj_out(hidden)
backward_rejected(detached_output)
assert cut.transformer_blocks[0].weight.grad is None
print('negative control: no_grad over frozen output head disconnects the block')

checked = selected()
checked_output = forward(checked, checkpoint)
checked_output.sum().backward()
torch.testing.assert_close(checked_output, reference, rtol=0, atol=0)
torch.testing.assert_close(checked.transformer_blocks[0].weight.grad, default.transformer_blocks[0].weight.grad)
assert checked.proj_in.weight.grad is None and checked.proj_out.weight.grad is None
print('original non-reentrant checkpoint helper: frozen input requires_grad=False; block grad=42')

from torch.utils.checkpoint import checkpoint as torch_checkpoint
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    reentrant_output = forward(selected(), lambda fn, x: torch_checkpoint(fn, x, use_reentrant=True))
assert any('None of the inputs have requires_grad=True' in str(w.message) for w in caught)
backward_rejected(reentrant_output)
print('negative control: reentrant checkpoint with no grad-requiring input disconnects this fixture')
print('PASS: CPU selection, gradient, AdamW and checkpoint controls; Python', sys.version.split()[0], 'torch', torch.__version__)
print('No H3/LoRA/FSDP/TrainStack/model-loader/replay/resume execution, dependency installs or device discovery.')
