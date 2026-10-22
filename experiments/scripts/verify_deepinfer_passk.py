#!/usr/bin/env python3
"""Run fixed DeepInfer #207 original pytest and ragged-summary controls; arg: git repo."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
repo = Path(sys.argv[1]).resolve()
head = '6e4c8eeb30f53e87c18569e198a2cf2e6baad5ac'
base = '9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'


def blob(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=repo)


for path, expected in {
    'eval/common/passk.py': 'aee51e7d444205c49cbcef815e62dd918b9d98726a04e1c388c8a2695869c7f7',
    'tests/eval/test_passk.py': 'fdee5510f2f3dd8f6bed3e4c5d8c35f051187773edd65ca85a4ce4030d6b5600',
    'eval/score.py': 'af3755cd5c0a1bb050296b03999773d6aead6360e197234b1bd8438fbcb293a4',
}.items():
    actual = hashlib.sha256(blob(head, path)).hexdigest()
    assert actual == expected, (path, actual)
    if not path.startswith('tests/'):
        assert blob(head, path) == blob(base, path), path
    print(path, actual, flush=True)

with tempfile.TemporaryDirectory(prefix='deepinfer-passk-') as scratch:
    archive = subprocess.check_output([
        'git', 'archive', head, 'eval/common/__init__.py', 'eval/common/passk.py', 'tests/eval/test_passk.py'
    ], cwd=repo)
    subprocess.run(['tar', '-xf', '-', '-C', scratch], input=archive, check=True)
    env = dict(os.environ, PYTHONPATH=scratch, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    subprocess.run([
        sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--rootdir', scratch, 'tests/eval/test_passk.py'
    ], cwd=scratch, env=env, check=True)
    spec = importlib.util.spec_from_file_location('native_passk', Path(scratch) / 'eval/common/passk.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'task_id': task, 'correct': correct} for task, flags in (
        ('full', [True, False, False, False]),
        ('short_correct', [True, False]),
        ('short_wrong', [False]),
    ) for correct in flags]
    summary = module.summarize(rows, n_sample=4, k_list=[1, 2, 3, 4])
    assert summary['num_tasks'] == 3 and summary['num_rollouts'] == 7
    for key, expected in {'score': 2/7, 'Pass@1': 1/4, 'Pass@2': 1/2, 'Pass@3': 7/12, 'Pass@4': 2/3}.items():
        assert abs(summary[key] - expected) < 1e-12, (key, summary[key], expected)
    full_only = module.summarize(rows[:4], n_sample=4, k_list=[4])
    assert full_only['Pass@4'] == 1.0
    declared = module.summarize(rows, n_sample=4, k_list=[4], extra={'expected_tasks': 4})
    assert declared['expected_tasks'] == 4 and declared['num_tasks'] == 3
    assert declared['Pass@4'] == summary['Pass@4']
    empty = module.summarize([], n_sample=4, k_list=[4])
    assert empty['num_tasks'] == 0 and empty['score'] == 0.0 and empty['Pass@4'] == 0.0
    print('ragged summary', summary)
    print('full-only contrast', full_only['Pass@4'], 'declared/observed tasks',
          declared['expected_tasks'], declared['num_tasks'])
    assert not any(n in sys.modules for n in ('torch', 'deepinfer', 'transformers', 'eval.score'))
    print('Three summary contrasts passed; n_sample/extra metadata do not reconstruct missing observations')
    print('Python', sys.version.split()[0], 'no model, judge, score_run, CUDA discovery or GPU/NPU')
