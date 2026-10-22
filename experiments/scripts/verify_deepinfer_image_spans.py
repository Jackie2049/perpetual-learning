#!/usr/bin/env python3
"""Check #208 original MM tests and cached-image endpoints in CPU AST harness; arg: git repo."""
import ast
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType

sys.dont_write_bytecode = True
repo = Path(sys.argv[1]).resolve()
head = '1e262f2b3aa508e0cb348d94662da36907363bda'
base = '9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'
source_path = 'deepinfer/ve/multimodal_processor.py'
test_path = 'tests/engine/test_mm_overwrite_prompt.py'


def blob(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=repo)


if len(sys.argv) == 2:
    for ref, path, expected in (
        (head, source_path, '5a544b357b4761659dcbd5a49c170a2d3ac8f224e5ba34787cc7bed2babf7314'),
        (base, source_path, 'f6b0e03a9138eddead7adfe424d4818a937c5ffbac2bff07d27dc96dd20d75cb'),
        (head, test_path, 'f72dee2beab198bfdb6b7487cea33aebcfffb9143077b546879c9db369498c0a'),
        (head, 'deepinfer/ve/processor/common.py', '2a9de67c82c624ebe8f87247d4d49cd1a86196cc878ea212d023048906c19da1'),
        (head, 'deepinfer/ve/schema.py', '6c2a481bb6ebb51df61968a619835e6a840db90fcf723ed4817f2451c6925485'),
    ):
        actual = hashlib.sha256(blob(ref, path)).hexdigest()
        assert actual == expected, (ref, path, actual)
        print(ref[:8], path, actual, flush=True)
    for path in ('deepinfer/ve/processor/common.py', 'deepinfer/ve/schema.py'):
        assert blob(base, path) == blob(head, path), path
    with tempfile.TemporaryDirectory(prefix='deepinfer-image-spans-') as scratch:
        archive = subprocess.check_output(['git', 'archive', head, test_path], cwd=repo)
        subprocess.run(['tar', '-xf', '-', '-C', scratch], input=archive, check=True)
        for ref in (base, head):
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), str(repo), ref, scratch], check=True)
    raise SystemExit(0)

ref, scratch = sys.argv[2:]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import pytest

for name in ('deepinfer', 'deepinfer.ve', 'deepinfer.ve.processor'):
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    if '.' in name:
        parent, child = name.rsplit('.', 1)
        setattr(sys.modules[parent], child, package)
common = ModuleType('deepinfer.ve.processor.common')
common.__file__ = f'{ref}:deepinfer/ve/processor/common.py'
sys.modules[common.__name__] = common
sys.modules['deepinfer.ve.processor'].common = common
exec(compile(blob(ref, 'deepinfer/ve/processor/common.py'), common.__file__, 'exec'), common.__dict__)
schema = ast.parse(blob(ref, 'deepinfer/ve/schema.py'))
raw_keys = next(n.value for n in schema.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'RAW_IMAGE_KEYS' for t in n.targets))
module = ModuleType('deepinfer.ve.multimodal_processor')
module.__file__ = f'{ref}:{source_path}'
module.__dict__.update(np=np, logger=logging.getLogger('image-spans-control'),
                       RAW_IMAGE_KEYS=ast.literal_eval(raw_keys), ImageInput=common.ImageInput,
                       SegmentInput=common.SegmentInput, TextLengthInput=common.TextLengthInput)
cls = next(n for n in ast.parse(blob(ref, source_path)).body
           if isinstance(n, ast.ClassDef) and n.name == 'MultimodalProcessor')
future = ast.parse('from __future__ import annotations').body[0]
exec(compile(ast.Module(body=[future, cls], type_ignores=[]), module.__file__, 'exec'), module.__dict__)
sys.modules[module.__name__] = module
sys.modules['deepinfer.ve'].multimodal_processor = module


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
status = pytest.main(['-q', '--tb=no', '--no-summary', '-p', 'no:cacheprovider', '-o', 'markers=ci: upstream marker only',
                      '--rootdir', scratch, str(Path(scratch) / test_path)], plugins=[results])
expected = {'passed': 12, 'failed': 0} if ref == head else {'passed': 8, 'failed': 4}
assert results.counts == expected and status == (0 if ref == head else 1), (status, results.counts)
if ref == base:
    assert sum('DID NOT RAISE' in f for f in results.failures) == 3
    assert sum('IndexError: list index out of range' in f for f in results.failures) == 1
spec = importlib.util.spec_from_file_location('native_mm_controls', Path(scratch) / test_path)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
for cached, expected_binaries in ((3, [b'a', b'b']), (4, [None, b'b']), (6, [None, b'b']), (7, [None, None])):
    processor = tests._processor({(10, 20): 3, (30, 40): 2})
    data = {'image_sizes': [(10, 20), (30, 40)], 'image_bytes': [b'a', b'b']}
    processor.overwrite_prompt(np.array([1, tests.IMAGE_TOKEN_ID, 2, tests.IMAGE_TOKEN_ID, 3], dtype=np.int32), data)
    assert data['segment_input'].meta['image_spans'] == [(1, 3), (5, 2)]
    processor._compute_and_strip_cached_images(data, cached)
    binaries = [s.binary for s in data['segment_input'].segments if isinstance(s, common.ImageInput)]
    assert binaries == expected_binaries
    assert data.get('_num_cached_images', 0) == sum(b is None for b in expected_binaries)
    print(ref[:8], 'cache prefix', cached, 'binaries', binaries)
assert 'deepinfer.ve.schema' not in sys.modules and 'deepinfer._C' not in sys.modules
assert not any(n in sys.modules for n in ('transformers', 'deep_ep', 'deep_gemm', 'tilelang'))
print(ref[:8], results.counts, 'plus 4 exact-endpoint controls; original fake image-size estimator retained')
print('Python', sys.version.split()[0], 'NumPy', np.__version__, 'pytest', pytest.__version__)
print('No processor init/workers, CUDA event shim, cache hits, VE encoding or GPU/NPU run')
