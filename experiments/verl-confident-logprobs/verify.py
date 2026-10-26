"""Run unchanged PR #8377 tests against isolated original CPU implementations.

Python 3.12, torch and pytest; read-only local Git objects, no model imports.
Usage: python verify.py /absolute/path/to/verl base|head
The original test module is collected in memory; no source files are rewritten.
"""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
import torch.nn.functional as F

BASE = '3c6d9b43b67c74ec343d77874cd8fa6e841523da'
HEAD = 'e9847a3d3db1d0f44ccb919e21677fb02e603464'
checkout, selected = sys.argv[1:]
revision = {'base': BASE, 'head': HEAD}[selected]
torch.set_num_threads(1)


def read(ref, path):
    return subprocess.check_output(['git', '-C', checkout, 'show', f'{ref}:{path}'],
                                   env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})


source = read(revision, 'verl/utils/torch_functional.py')
test_source = read(HEAD, 'tests/utils/test_torch_functional_logprobs_on_cpu.py')
compile(source, revision, 'exec')
compile(test_source, HEAD + ':tests', 'exec')
names = {'gather_from_labels', 'logprobs_from_logits_naive', 'logprobs_from_logits_v2'}
if selected == 'head':
    names.add('_LabelLogprobs')
nodes = [n for n in ast.parse(source).body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
assert {n.name for n in nodes} == names
module = types.ModuleType('verl.utils.torch_functional')
module.__dict__.update(torch=torch, F=F)
exec(compile(ast.Module(body=nodes, type_ignores=[]), revision, 'exec'), module.__dict__)
sys.modules[module.__name__] = module
reports = []


class OriginalTests(pytest.Module):
    def _getobj(self):
        tests = types.ModuleType('upstream_logprob_tests_' + selected)
        tests.__file__ = str(Path(__file__).resolve())
        exec(compile(test_source, HEAD + ':tests', 'exec'), tests.__dict__)
        return tests


class CollectOriginal:
    def pytest_pycollect_makemodule(self, module_path, parent):
        return OriginalTests.from_parent(parent, path=module_path)

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed:
            reports.append({'test': report.nodeid.split('::')[-1], 'outcome': report.outcome})


exit_code = pytest.main([str(Path(__file__).resolve()), '-q', '--tb=no', '--noconftest',
                         '-o', 'addopts=', '-p', 'no:cacheprovider'], plugins=[CollectOriginal()])
counts = dict(Counter(r['outcome'] for r in reports))
assert counts == ({'passed': 8, 'failed': 4} if selected == 'base' else {'passed': 12}), counts
assert int(exit_code) == (1 if selected == 'base' else 0)

controls = []
for name in ('logprobs_from_logits_naive', 'logprobs_from_logits_v2'):
    fn = getattr(module, name)
    # Original issue input, real CPU tensors and autograd, no synthetic backward.
    confident = torch.zeros(1, 32000)
    confident[0, 0] = 30.
    confident.requires_grad_()
    logp = fn(confident, torch.tensor([0]))
    grad, = torch.autograd.grad(logp.sum(), confident)
    confident_result = {'logprob': logp.item(), 'label_grad': grad[0, 0].item(),
                        'other_grad_sum': grad[0, 1:].sum().item()}
    # Existing review's finite low-confidence example, not a new finding.
    low = torch.tensor([[-1e8, 0., 1.]], requires_grad=True)
    low_logp = fn(low, torch.tensor([0]))
    low_grad, = torch.autograd.grad(low_logp.sum(), low)
    reference = torch.tensor([[-1e8, 0., 1.]], dtype=torch.float64, requires_grad=True)
    reference_logp = F.log_softmax(reference, -1)[0, 0]
    reference_grad, = torch.autograd.grad(reference_logp, reference)
    low_result = {'gradient': low_grad.tolist(), 'reference': reference_grad.tolist(),
                  'gradient_sum': low_grad.sum().item(),
                  'max_abs_error': (low_grad.double() - reference_grad).abs().max().item()}
    assert low_result['max_abs_error'] < 1e-6 if selected == 'base' else low_result['max_abs_error'] > 1.
    # One ordinary finite FP64 input tests the separate higher-order contract.
    x = torch.tensor([[.2, -.3, .7]], dtype=torch.float64, requires_grad=True)
    label = torch.tensor([0])
    torch.manual_seed(8377)
    first = torch.autograd.gradcheck(lambda a: fn(a, label), (x,), raise_exception=False)
    second_error = None
    try:
        second = torch.autograd.gradgradcheck(lambda a: fn(a, label), (x,), raise_exception=False)
    except RuntimeError as error:
        assert selected == 'head' and 'modified by an inplace operation' in str(error)
        second = False
        second_error = str(error)
    assert first and second == (selected == 'base')
    controls.append({'function': name, 'issue_repro': confident_result,
                     'existing_review_low_label': low_result,
                     'first_order_gradcheck': first, 'gradgradcheck': second,
                     'gradgradcheck_runtime_error': second_error})

print('RESULT_JSON=' + json.dumps({
    'revision': revision, 'test_revision': HEAD, 'python': sys.version.split()[0],
    'torch': torch.__version__, 'pytest': pytest.__version__, 'threads': torch.get_num_threads(),
    'source_sha256': hashlib.sha256(source).hexdigest(),
    'test_sha256': hashlib.sha256(test_source).hexdigest(),
    'original_tests': reports, 'counts': counts, 'controls': controls,
    'scope': 'original functions/class and unchanged original tests with real CPU autograd; no full verl, flash-attn, NPU, Triton, model, memory or training test'
}, indent=2))
