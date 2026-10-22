#!/usr/bin/env python3
"""Run #209 original admission tests in an explicit CPU import harness; arg: git repo."""
import ast
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
repo = Path(sys.argv[1]).resolve()
head = 'bb289c50d1ad9dc61d08efefaa27fc2660d1358a'
base = '9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'
test_path = 'tests/engine/test_golden_token_validation.py'


def blob(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=repo)


if len(sys.argv) == 2:
    for ref, path, expected in (
        (head, 'deepinfer/engine/llm_engine.py', 'c50f82abf9735940442eb7658ab77f7d1897cf84f67ceb7d5928eb8f1e70199e'),
        (base, 'deepinfer/engine/llm_engine.py', 'a636d60ca5711f6b7c21e68370c3bd9c2de6fc3f4439a7429ca4f572f0edeebe'),
        (head, test_path, '8a784c02e341fd8f57193a5459f948b03aa462067285a9586a9f707b81e6a675'),
        (head, 'deepinfer/sampling_params.py', 'f800034efdebe1b21620660a51fa9122d3d7f138d8e60cd738f33d643c3116f8'),
    ):
        actual = hashlib.sha256(blob(ref, path)).hexdigest()
        assert actual == expected, (ref, path, actual)
        print(ref[:8], path, actual, flush=True)
    for path in ('deepinfer/sampling_params.py', 'deepinfer/multimodal_tokens.py', 'deepinfer/engine/engine_type.py'):
        assert blob(base, path) == blob(head, path), path
    with tempfile.TemporaryDirectory(prefix='deepinfer-golden-') as scratch:
        archive = subprocess.check_output(['git', 'archive', head, test_path], cwd=repo)
        subprocess.run(['tar', '-xf', '-', '-C', scratch], input=archive, check=True)
        for ref in (base, head):
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), str(repo), ref, scratch], check=True)
    print('Both versions match expected original-test outcomes; no engine construction, workers or GPU/NPU')
    raise SystemExit(0)

ref, scratch = sys.argv[2:]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import pytest

for name in ('deepinfer', 'deepinfer.engine'):
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    if '.' in name:
        parent, child = name.rsplit('.', 1)
        setattr(sys.modules[parent], child, package)


def load_leaf(name, path):
    module = ModuleType(name)
    module.__file__ = f'{ref}:{path}'
    sys.modules[name] = module
    parent, child = name.rsplit('.', 1)
    setattr(sys.modules[parent], child, module)
    exec(compile(blob(ref, path), module.__file__, 'exec'), module.__dict__)
    return module


params = load_leaf('deepinfer.sampling_params', 'deepinfer/sampling_params.py')
tokens = load_leaf('deepinfer.multimodal_tokens', 'deepinfer/multimodal_tokens.py')
kind = load_leaf('deepinfer.engine.engine_type', 'deepinfer/engine/engine_type.py')
engine = ModuleType('deepinfer.engine.llm_engine')
engine.__file__ = f'{ref}:deepinfer/engine/llm_engine.py'
engine.__dict__.update(np=np, time=time, SamplingParams=params.SamplingParams,
                       ImageTokenOffset=tokens.ImageTokenOffset, EngineType=kind.EngineType,
                       logger=logging.getLogger('admission-control'), Sequence=None,
                       cpu_timer=SimpleNamespace(record=lambda *args, **kwargs: lambda fn: fn))
tree = ast.parse(blob(ref, 'deepinfer/engine/llm_engine.py'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMEngine')
future = ast.parse('from __future__ import annotations').body[0]
exec(compile(ast.Module(body=[future, cls], type_ignores=[]), engine.__file__, 'exec'), engine.__dict__)
sys.modules[engine.__name__] = engine
sys.modules['deepinfer.engine'].llm_engine = engine


class Results:
    def __init__(self):
        self.counts = {'passed': 0, 'failed': 0}
        self.failures = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.counts[report.outcome] += 1
            if report.failed:
                self.failures.append(report.longreprtext)


results = Results()
status = pytest.main(['-q', '--tb=no', '--no-summary', '-p', 'no:cacheprovider',
                      '-o', 'markers=ci: CPU harness\nci_npu: upstream marker only',
                      '--rootdir', scratch, str(Path(scratch) / test_path)], plugins=[results])
expected = {'passed': 30, 'failed': 0} if ref == head else {'passed': 12, 'failed': 18}
assert results.counts == expected and status == (0 if ref == head else 1), (status, results.counts)
assert all('DID NOT RAISE' in failure and 'ValueError' in failure for failure in results.failures)
assert 'deepinfer._C' not in sys.modules and not any(n.startswith('deepinfer.worker') for n in sys.modules)
print(ref[:8], results.counts, 'full original class AST; timer identity stub; original test mocks retained')
print('Python', sys.version.split()[0], 'NumPy', np.__version__, 'pytest', pytest.__version__)
