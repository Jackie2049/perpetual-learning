"""Run #8026's four original CPU math test bodies and narrow formula controls.

Usage: python -B verify_verl_sampler_logprobs.py /path/to/verl-git
Python3.12+, torch (tested 2.9.1), fixed local Git objects. AST-loads original
sampler/chunk/v2 functions and original test definitions; no native verl import,
vLLM overrides, engine/config tests, GPU discovery, FSDP or model execution.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
args = parser.parse_args()
HEAD = '6a7110b42ae45068b8c7d1c20c56db841c5283af'
SOURCE = 'verl/utils/torch_functional.py'
TEST = 'tests/utils/test_logprobs_from_logits_sampler_on_cpu.py'


def read(path):
    raw = subprocess.check_output(['git', 'show', HEAD + ':' + path], cwd=args.repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
    compile(raw, path, 'exec')
    print(path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    return ast.parse(raw)


ns = dict(torch=torch, F=F, checkpoint=checkpoint)
names = {'logprobs_from_logits_sampler', '_sampler_chunk_logprobs', 'logprobs_from_logits_v2'}
nodes = [n for n in read(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 3
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-functions>', 'exec'), ns)
tests = [n for n in read(TEST).body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
assert len(tests) == 4
exec(compile(ast.Module(body=tests, type_ignores=[]), '<original-test-definitions>', 'exec'), ns)
for test in tests:
    ns[test.name]()
    print('PASS original test body', test.name)

sampler, native = ns['logprobs_from_logits_sampler'], ns['logprobs_from_logits_v2']
torch.manual_seed(123)
logits = torch.randn(50, 4096, dtype=torch.float32) * 4
labels = torch.randint(0, 4096, (50,))
out, old = sampler(logits, labels), native(logits, labels)
assert torch.equal(out, torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1))
different = (out != old).sum().item()
print('FP32 same logits, sampler vs gather-logsumexp:', different, '/50 differ; max abs delta',
      (out - old).abs().max().item())

# Both compatibility flag values leave the input intact and match the same local VJP.
grads = []
for flag in (False, True):
    x = logits[:5, :128].clone().requires_grad_()
    before = x.detach().clone()
    y = torch.arange(5)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append((t.dtype, tuple(t.shape), t.untyped_storage().data_ptr())) or t, lambda t: t
    ):
        actual = sampler(x, y, inplace_backward=flag, chunk_size=2)
    actual.sum().backward()
    ref_x = before.clone().requires_grad_()
    F.log_softmax(ref_x.float(), dim=-1).gather(-1, y[:, None]).sum().backward()
    assert torch.equal(x.detach(), before) and torch.equal(x.grad, ref_x.grad)
    assert all(pointer == x.untyped_storage().data_ptr()
               for dtype, shape, pointer in saved if dtype == torch.float32 and shape and shape[-1] == 128)
    grads.append(x.grad.clone())
assert torch.equal(*grads)
print('PASS both inplace_backward flags/input immutability/exact local VJP; saved FP32 chunks alias inputs, not log-softmax outputs')

# The helper receives already-scaled logits and never applies top-k/top-p renormalization.
x = torch.tensor([[0., 1., 2.]])
y = torch.tensor([2])
full, scaled, truncated = sampler(x, y), sampler(x / .5, y), sampler(x[:, 2:], torch.tensor([0]))
assert full.item() != scaled.item() and full.item() != truncated.item() and truncated.item() == 0
print('PASS full-vocabulary/temperature-input/truncated-support separation:',
      full.item(), scaled.item(), truncated.item())
print('torch', torch.__version__, '; four original test bodies and three supplemental control groups.')
print('Same-logits CPU math only; no native pytest collection, vLLM/kernel override, engine or batch-invariant model proof.')
