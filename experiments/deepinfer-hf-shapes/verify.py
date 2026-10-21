"""Run #153's original CPU safetensors tests from immutable Git objects.

Python >=3.10, torch with float8_e8m0fnu, safetensors, pytest. Full leaf modules
and original tests execute unchanged; package shells skip engine initialization.
No model download, full-model allocation, device runtime initialization or install.
"""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

import pytest
import torch

BASE = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
HEAD = 'c2bbb61623c5120364faa879829feabb6d709434'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('--revision', choices=('base', 'fix'), default='fix')
    args = parser.parse_args()
    revision = BASE if args.revision == 'base' else HEAD

    def read(rev, path):
        data = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=args.repo,
                                       env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})
        print(rev[:8], path, hashlib.sha256(data).hexdigest(), flush=True)
        return data

    def register(name):
        module = ModuleType(name)
        sys.modules[name] = module
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(sys.modules[parent], attr, module)
        return module

    for name in ('deepinfer', 'deepinfer.hugging_face'):
        register(name).__path__ = []
    for path in ('deepinfer/logger.py', 'deepinfer/hugging_face/names.py',
                 'deepinfer/hugging_face/parameters.py', 'deepinfer/hugging_face/weights.py',
                 'deepinfer/hugging_face/loading.py'):
        name = path[:-3].replace('/', '.')
        module = register(name)
        module.__file__ = 'git:' + revision + ':' + path
        module.__package__ = name.rpartition('.')[0]
        exec(compile(read(revision, path), module.__file__, 'exec'), module.__dict__)

    tests = register('_original_hf_shape_tests')
    tests.__file__ = str(Path(__file__).resolve())
    exec(compile(read(HEAD, 'tests/hugging_face/test_checkpoint_shapes.py'),
                 'git:#153:tests/hugging_face/test_checkpoint_shapes.py', 'exec'), tests.__dict__)

    class OriginalModule(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        results = []

        def pytest_configure(self, config):
            for mark in ('ci', 'ci_npu'):
                config.addinivalue_line('markers', mark + ': original upstream regression marker')

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalModule.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                self.results.append((report.nodeid.rsplit('::', 1)[-1], report.outcome, str(report.longrepr)))

    def no_network(event, args):
        if event.startswith('socket.'):
            raise RuntimeError('Network outside the scope of CPU checkpoint tests')

    sys.addaudithook(no_network)
    adapter = Adapter()
    code = pytest.main([str(Path(__file__).resolve()), '--noconftest', '-o', 'addopts=',
                        '-p', 'no:cacheprovider', '-q', '--tb=line'], plugins=[adapter])
    counts = Counter(outcome for _, outcome, _ in adapter.results)
    failures = {name.split('[', 1)[0] for name, outcome, _ in adapter.results if outcome == 'failed'}
    if args.revision == 'base':
        expected = {
            'test_ordinary_fp8_rejects_missing_or_wrong_scale_shape', 'test_ordinary_fp8_rejects_wrong_scale_dtype',
            'test_shared_expert_halves_must_match', 'test_expert_packed_geometry', 'test_engram_geometry',
            'test_engram_local_range_and_alignment_padding', 'test_fp8_shared_halves_and_scales_in_independent_shards',
            'test_every_global_expert_is_validated',
            'test_engram_global_rows_match_vocabulary_metadata', 'test_shared_halves_with_mixed_optional_prefixes_are_validated',
            'test_prefixed_w1_plain_w3_layout', 'test_mixed_prefix_coverage_preserves_gate_up_order_when_loading',
        }
        assert code == 1 and counts == {'passed': 22, 'failed': 33} and failures == expected, (code, counts, failures)
        missed = sum('DID NOT RAISE' in detail for _, outcome, detail in adapter.results if outcome == 'failed')
        assert missed == 32, missed
    else:
        assert code == 0 and counts == {'passed': 55}, (code, counts)
    hf = sys.modules['deepinfer.hugging_face']
    target = 'module.module.1.experts.shared_experts.0.linear1.weight'
    for gate_prefix, up_prefix in ((False, False), (True, True), (True, False), (False, True)):
        w1 = ('model.' if gate_prefix else '') + 'layers.0.ffn.shared_experts.w1.weight'
        w3 = ('model.' if up_prefix else '') + 'layers.0.ffn.shared_experts.w3.weight'
        with TemporaryDirectory(prefix='hf-gate-up-') as directory:
            root = Path(directory)
            tensors = {w1: torch.full((32, 64), 2.0, dtype=torch.bfloat16),
                       w3: torch.full((32, 64), 7.0, dtype=torch.bfloat16)}
            index = tests._checkpoint(root, tensors, split=True)
            coverage = hf.weights.build_coverage(index, gpt_cfg=SimpleNamespace(num_hidden_layers=1))
            model = torch.nn.Module()
            parent = model
            for name in target.split('.')[:-1]:
                child = torch.nn.Module()
                parent.add_module(name, child)
                parent = child
            parent.register_parameter('weight', tests._parameter((64, 64), torch.bfloat16))
            hf.loading.validate_hf_shapes(hf.parameters.model_params(model), coverage, hf_dir=root)
            hf.weights.load_covered_params(model, coverage, hf_dir=root)
            reversed_halves = args.revision == 'base' and gate_prefix and not up_prefix
            first, second = (7.0, 2.0) if reversed_halves else (2.0, 7.0)
            assert torch.all(parent.weight[:32] == first) and torch.all(parent.weight[32:] == second)
            print('native load gate/up prefixes', gate_prefix, up_prefix,
                  'order', coverage.tensors[target], 'loaded halves', first, second)
    assert not torch.cuda.is_initialized()  # Local flag only, not device discovery.
    assert all(name not in sys.modules for name in ('tile_kernels', 'deepinfer._C', 'deep_ep'))
    print(args.revision, dict(counts), 'original CPU tensors/files; no full model, kernels, distributed or device validation')


if __name__ == '__main__':
    main()
