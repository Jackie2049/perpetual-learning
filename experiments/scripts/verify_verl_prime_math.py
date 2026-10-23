"""Run #8347's 64 original cases and scoped controls through original CPU scorers.

Usage: python -B verify_verl_prime_math.py /path/to/verl --deps /path/to/parser-target
Needs Python3.12+/pytest/sympy1.14/pylatexenc2.10/antlr4-runtime4.9.3/numpy/torch.
Complete original modules load from fixed Git objects; only verl/utils package
initializers are skipped. Uses process-local fork for the native timeout decorator
(Linux-style CPU context), not macOS default spawn. No timeout/parser replacement.
Direct parameter calls, not pytest collection, dataset replay or training proof.
Only hard-coded benign expressions: upstream parsers are not a security sandbox.
"""
import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

BASE = 'd430157d15516d89e0cc853fd10fcd21d9234c44'
HEAD = '7f34e0188697b89ec0def7ce243633ba4076b707'
TARGET = 'bfff4d90b911f412a0856abecffc15eeafb78977'
TEST = 'tests/utils/reward_score/test_prime_math_on_cpu.py'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repo', type=Path)
parser.add_argument('--deps', type=Path, required=True)
parser.add_argument('--rev', choices=(BASE, HEAD))
args = parser.parse_args()
if args.rev is None:
    for rev in (BASE, HEAD):
        subprocess.run([sys.executable, '-B', __file__, str(args.repo), '--deps', str(args.deps), '--rev', rev], check=True)
    sys.exit(0)

sys.dont_write_bytecode = True
sys.path.insert(0, str(args.deps.resolve()))
multiprocessing.set_start_method('fork')


def git(*params):
    return subprocess.check_output(['git', *params], cwd=args.repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))


paths = set(git('ls-tree', '-r', '--name-only', args.rev).decode().splitlines())
for path in ('verl/utils/reward_score/prime_math/__init__.py',
             'verl/utils/reward_score/prime_math/grader.py',
             'verl/utils/reward_score/prime_math/math_normalize.py'):
    assert git('show', BASE + ':' + path) == git('show', TARGET + ':' + path)


class OriginalModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'verl' and not fullname.startswith('verl.'):
            return None
        stem = fullname.replace('.', '/')
        path = stem + '/__init__.py' if stem + '/__init__.py' in paths else stem + '.py'
        if path not in paths:
            return None
        spec = importlib.util.spec_from_loader(fullname, self, is_package=path.endswith('/__init__.py'))
        spec.loader_state = path
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ in ('verl', 'verl.utils'):
            return
        path = module.__spec__.loader_state
        exec(compile(git('show', args.rev + ':' + path), args.rev + ':' + path, 'exec'), module.__dict__)


sys.meta_path.insert(0, OriginalModules())
import pytest
import sympy
import torch
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.prime_math import grade_answer, match_answer
from verl.utils.reward_score.prime_math.grader import math_equal

namespace = {'__name__': 'original_prime_math_tests'}
exec(compile(git('show', HEAD + ':' + TEST), TEST, 'exec'), namespace)
summary = {}
failed = []
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for name, fn in namespace.items():
        if not name.startswith('test_'):
            continue
        cases = next(mark.args[1] for mark in fn.pytestmark if mark.name == 'parametrize')
        passed = 0
        for index, values in enumerate(cases):
            try:
                fn(*values)
            except AssertionError:
                failed.append((name, index))
            else:
                passed += 1
        summary[name] = {'passed': passed, 'failed': len(cases) - passed}
print(args.rev[:8], json.dumps(summary), flush=True)
expected = (64, 0) if args.rev == HEAD else (39, 25)
assert (sum(value['passed'] for value in summary.values()), len(failed)) == expected, failed
print('Captured original scorer diagnostic lines:', len(captured.getvalue().splitlines()))


def score(answer, gold):
    return default_compute_score('numina_cn_k12', r'The answer is $\boxed{' + answer + '}$.', gold)


with contextlib.redirect_stdout(captured):
    controls = {
        'integer early grade vs fallback': [grade_answer('22501', '22500'), math_equal('22501', '22500'), score('22501', '22500')],
        'decimal tolerance retained': score('348.69', '348.68'),
        'equation echo still rejected': score(r'x - \sqrt{3}y + 2 = 0', r'x - \sqrt{3}y + 2 = 0'),
        'equation extraction': match_answer(r'The answer is $\boxed{x - \sqrt{3}y + 2 = 0}$.')[1],
        'short lhs renamed': score('q = 2', 'm = 2'),
        'arrows direct vs full score': [math_equal(r'A \leftarrow B', r'A \rightarrow B'), score(r'A \leftarrow B', r'A \rightarrow B')],
        'scientific notation large integer': score('9.007199254740993e15', '9007199254740992'),
    }
assert controls == {
    'integer early grade vs fallback': [False, False, 0.0] if args.rev == HEAD else [False, True, 1.0],
    'decimal tolerance retained': 1.0,
    'equation echo still rejected': 0.0,
    'equation extraction': '0',
    'short lhs renamed': 1.0,
    'arrows direct vs full score': [False, 1.0],
    'scientific notation large integer': 1.0,
}, controls
print('7 scope controls:', json.dumps(controls), flush=True)
for path in ('verl/utils/reward_score/prime_math/__init__.py',
             'verl/utils/reward_score/prime_math/grader.py',
             'verl/utils/reward_score/prime_math/math_normalize.py'):
    print(path, hashlib.sha256(git('show', args.rev + ':' + path)).hexdigest())
print('Original test sha256', hashlib.sha256(git('show', HEAD + ':' + TEST)).hexdigest())
print('Python', sys.version.split()[0], 'sympy', sympy.__version__, 'torch', torch.__version__,
      'pytest', pytest.__version__, 'multiprocessing', multiprocessing.get_start_method())
assert not torch.distributed.is_initialized() and 'ray' not in sys.modules
print('CPU scorer only; no dataset download, model, training, GPU or security validation.')
