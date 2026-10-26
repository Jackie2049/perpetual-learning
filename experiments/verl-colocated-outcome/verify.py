"""PR #8379 original CPU tests with real containers and original method shells.

Run: python verify.py /absolute/path/to/verl base|head
Requires Python 3.12, torch, TensorDict, NumPy and pytest.
No complete trainer, TransferQueue service, reward inference or GPU is loaded.
Upstream source: verl-project/verl (Apache-2.0, ByteDance contributors).
"""
import __future__
import ast
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tensordict import TensorDict, NonTensorData, NonTensorStack
import tensordict
import torch

BASE = '3c6d9b43b67c74ec343d77874cd8fa6e841523da'
HEAD = '0447a46d59b842da56e38e7d25b0d05e47f6b42e'
checkout, selected = sys.argv[1:]
revision = {'base': BASE, 'head': HEAD}[selected]
torch.set_num_threads(1)
hashes = {}


def source(ref, path):
    raw = subprocess.check_output(['git', '-C', checkout, 'show', f'{ref}:{path}'],
                                  env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})
    hashes[ref + ':' + path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    compile(tree, path, 'exec')
    return tree


def execute(nodes, ns):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, 'original_definitions', 'exec',
                 flags=__future__.annotations.compiler_flag), ns)


def definitions(tree, names, ns, strip_registration=False):
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
    assert {n.name for n in nodes} == set(names)
    if strip_registration:
        for node in nodes:
            node.decorator_list = []
    execute(nodes, ns)


def package(name):
    if name not in sys.modules:
        mod = ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        if '.' in name:
            parent, leaf = name.rsplit('.', 1)
            setattr(package(parent), leaf, mod)
    return sys.modules[name]


ns = package('_colocated_original_' + selected).__dict__
ns.update(torch=torch, np=np, TensorDict=TensorDict, NonTensorData=NonTensorData,
          NonTensorStack=NonTensorStack, dataclass=dataclass, field=field, os=os,
          random=random, Any=Any, Optional=Optional, pytest=pytest,
          MagicMock=MagicMock, patch=patch, SimpleNamespace=SimpleNamespace)
definitions(source(revision, 'verl/protocol.py'),
            {'_DataProtoConfigMeta', 'DataProtoConfig', 'DataProto', 'DataProtoItem'}, ns)
for module_name, path in [('verl.base_config', 'verl/base_config.py'),
                          ('verl.trainer.config', 'verl/trainer/config/algorithm.py')]:
    execute(source(revision, path).body, package(module_name).__dict__)
ns['AlgoConfig'] = sys.modules['verl.trainer.config'].AlgoConfig
definitions(source(revision, 'verl/utils/torch_functional.py'),
            {'masked_sum', 'masked_mean', 'masked_var', 'masked_whiten'}, ns)
ns['verl_F'] = SimpleNamespace(**{k: ns[k] for k in ('masked_sum', 'masked_mean', 'masked_var', 'masked_whiten')})
definitions(source(revision, 'verl/trainer/ppo/core_algos.py'),
            {'compute_gae_advantage_return', 'compute_reinforce_plus_plus_outcome_advantage',
             'compute_remax_outcome_advantage'}, ns, strip_registration=True)


def method_shell(tree, cls_name, names):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    methods = [n for n in cls.body if getattr(n, 'name', None) in names]
    assert {n.name for n in methods} == set(names)
    execute([ast.ClassDef(name=cls_name, bases=[], keywords=[], body=methods,
                          decorator_list=[], type_params=[])], ns)


method_shell(source(revision, 'verl/experimental/reward_loop/reward_manager/base.py'),
             'RewardManagerBase', {'assemble_rm_scores'})
method_shell(source(revision, 'verl/trainer/ppo/v1/trainer_base.py'),
             'PPOTrainer', {'_compute_reward_colocate', '_lengths_to_mask'})
definitions(source(revision, 'verl/utils/tensordict_utils.py'), {'get_tensordict'}, ns)
ns['tu'] = SimpleNamespace(get_tensordict=ns['get_tensordict'])
ns['tq'] = SimpleNamespace(kv_batch_get=MagicMock(), kv_batch_put=MagicMock())
package('verl.trainer.ppo.v1.trainer_base').tq = ns['tq']

assembly = source(HEAD, 'tests/experimental/reward_loop/test_assemble_rm_scores_on_cpu.py')
assembly_names = {n.name for n in assembly.body if isinstance(n, ast.FunctionDef)}
definitions(assembly, assembly_names, ns)
v1_test = source(HEAD, 'tests/trainer/ppo/v1/test_trainer_base_on_cpu.py')
definitions(v1_test, {'_StubTrainer', 'test_colocated_reward_preserves_generation_mask_from_transfer_queue'}, ns)
reports = []


class OriginalTests(pytest.Module):
    def _getobj(self):
        mod = ModuleType('_colocated_selected_tests_' + selected)
        mod.__dict__.update(ns)
        mod.__file__ = str(Path(__file__).resolve())
        return mod


class CollectOriginal:
    def pytest_pycollect_makemodule(self, module_path, parent):
        return OriginalTests.from_parent(parent, path=module_path)

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed:
            reports.append({'test': report.nodeid.split('::')[-1], 'outcome': report.outcome,
                            'failure': str(report.longrepr) if report.failed else None})


exit_code = pytest.main([str(Path(__file__).resolve()), '-q', '--tb=no', '--noconftest',
                        '-o', 'addopts=', '-p', 'no:cacheprovider'], plugins=[CollectOriginal()])
counts = dict(Counter(r['outcome'] for r in reports))
assert counts == ({'failed': 9, 'passed': 4} if selected == 'base' else {'passed': 13}), counts
assert int(exit_code) == (1 if selected == 'base' else 0)
print('RESULT_JSON=' + json.dumps({
    'revision': revision, 'test_revision': HEAD, 'counts': counts,
    'tests': reports, 'source_sha256': hashes,
    'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'tensordict': tensordict.__version__, 'numpy': np.__version__, 'pytest': pytest.__version__},
    'scope': '11 original assembly/estimator items + 2 original V1 items; original method shells, DataProto, typed config, CPU autograd and jagged TensorDict; original TQ/inference mocks, no service/model/full trainer or GPU'
}, indent=2))
