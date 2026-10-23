"""Run #8026's four original CPU math test bodies and narrow formula controls.

Usage: python -B verify_verl_sampler_logprobs.py /path/to/verl-git
Add --row-chunk-pr for #8351's CPU-ported test bodies and saved-storage controls.
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
parser.add_argument('--row-chunk-pr', action='store_true')
args = parser.parse_args()
HEAD = '6a7110b42ae45068b8c7d1c20c56db841c5283af'
SOURCE = 'verl/utils/torch_functional.py'
TEST = 'tests/utils/test_logprobs_from_logits_sampler_on_cpu.py'


def read(path, revision=HEAD):
    raw = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=args.repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
    compile(raw, path, 'exec')
    print(revision[:8], path, 'sha256=' + hashlib.sha256(raw).hexdigest())
    return ast.parse(raw)


def row_chunk_pr():
    from collections import Counter
    from types import ModuleType
    import pytest

    base = 'a4b9354968705a441fb7da7367924b15b902e771'
    head = '285e2dd83d131a324b440ed5464c0dc61e3cf0b4'
    modules = []
    trees = []
    for rev in (base, head):
        tree = read(SOURCE, rev)
        trees.append(tree)
        module = ModuleType('original_logprobs_' + rev[:8])
        module.__dict__.update(torch=torch, F=F)
        nodes = [n for n in tree.body if
                 isinstance(n, ast.FunctionDef) and n.name == 'logprobs_from_logits_v2' or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_LOGPROB_CHUNK_BYTES'
                                                   for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-v2>', 'exec'), module.__dict__)
        modules.append(module)
    before, after = [next(n for n in t.body if isinstance(n, ast.FunctionDef)
                         and n.name == 'logprobs_from_logits_v2').body[1] for t in trees]
    assert ast.dump(before.test) == ast.dump(after.test)
    assert [ast.dump(n) for n in before.body] == [ast.dump(n) for n in after.body]
    assert [ast.dump(n) for n in before.orelse] == [ast.dump(n) for n in after.orelse[0].orelse]
    names = {'_per_row_logprobs', 'test_logprobs_from_logits_v2_2d_bitwise_matches_per_row',
             'test_logprobs_from_logits_v2_bf16_2d_does_not_loop_over_tokens'}
    test_nodes = [n for n in read('tests/utils/test_torch_functional.py', head).body
                  if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(test_nodes) == 3
    ns = dict(torch=torch, F=F, pytest=pytest, get_device_name=lambda: 'cpu',
              torch_functional=modules[1], logprobs_from_logits_v2=modules[1].logprobs_from_logits_v2)
    exec(compile(ast.Module(body=test_nodes, type_ignores=[]), '<original-test-bodies>', 'exec'), ns)
    assert torch.get_default_device().type == 'cpu'
    failures = []
    for dtype in (torch.bfloat16, torch.float16):
        for transposed in (False, True):
            for tokens, vocab, budget in ((37, 1000, 1 << 26), (37, 1000, 8192),
                                         (37, 1000, 1), (5, 151936, 1 << 20)):
                with pytest.MonkeyPatch.context() as patch:
                    try:
                        ns['test_logprobs_from_logits_v2_2d_bitwise_matches_per_row'](
                            patch, tokens, vocab, budget, transposed, dtype)
                    except AssertionError:
                        failures.append((str(dtype), transposed, tokens, vocab, budget))
    print('CPU port: 16 original parameter cases; bitwise failures:', failures)
    assert not failures
    for i, module in enumerate(modules):
        ns.update(torch_functional=module, logprobs_from_logits_v2=module.logprobs_from_logits_v2)
        with pytest.MonkeyPatch.context() as patch:
            try:
                ns['test_logprobs_from_logits_v2_bf16_2d_does_not_loop_over_tokens'](patch)
            except AssertionError as error:
                assert i == 0
                trace = error.__traceback__
                while trace.tb_next is not None:
                    trace = trace.tb_next
                expected_line = next(n for n in test_nodes if n.name.endswith('does_not_loop_over_tokens')).body[-1].lineno
                assert trace.tb_lineno == expected_line
                assert trace.tb_frame.f_locals['calls'] == [torch.Size([1000])] * 64
                print('Base original call-count body: expected AssertionError (64 per-token calls).')
            else:
                assert i == 1
                print('Head original call-count body passed.')
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(37, 1000, generator=generator).to(torch.bfloat16)
    labels = torch.arange(37) % 1000
    observations = []
    for index, module in enumerate(modules):
        if index == 1:
            module._LOGPROB_CHUNK_BYTES = 8192
        leaf = x.clone().requires_grad_()
        storages, calls = {}, []
        original = F.log_softmax

        def counted(tensor, *a, **kw):
            calls.append(tuple(tensor.shape))
            return original(tensor, *a, **kw)

        def save(tensor):
            if tensor.ndim and tensor.shape[-1] == 1000 and tensor.dtype == torch.bfloat16:
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(F, 'log_softmax', counted)
            with torch.autograd.graph.saved_tensors_hooks(save, lambda t: t):
                out = module.logprobs_from_logits_v2(leaf, labels)
            seen, stack, counts = set(), [out.grad_fn], Counter()
            while stack:
                node = stack.pop()
                if node is None or node in seen:
                    continue
                seen.add(node)
                counts[type(node).__name__] += 1
                stack.extend(n for n, _ in node.next_functions)
            out.sum().backward()
        observations.append((out.detach(), leaf.grad, len(calls), max(storages.values()), sum(storages.values())))
        print(module.__name__, 'calls', len(calls), 'largest saved vocab storage', max(storages.values()),
              'unique saved vocab bytes', sum(storages.values()), 'backward nodes', dict(counts))
    old, new = observations
    assert torch.equal(old[0].view(torch.int16), new[0].view(torch.int16))
    assert torch.equal(old[1].view(torch.int16), new[1].view(torch.int16))
    assert old[2:] == (37, 2000, 74000) and new[2:] == (10, 8000, 74000)
    with pytest.raises(ValueError, match='labels shape'):
        modules[1].logprobs_from_logits_v2(torch.zeros(2, 3, dtype=torch.bfloat16), torch.zeros(2, 1, dtype=torch.long))
    print('PASS CPU values/gradients and saved-storage accounting; not CUDA peak memory or timing.')
    print('PASS 2-D low-precision shape rejection; unchanged FP32/64 and other-rank branches checked by AST.')
    print('Original test bodies use explicit CPU device binding; no native pytest, GPU or engine execution.')


if args.row_chunk_pr:
    row_chunk_pr()
    raise SystemExit(0)


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
