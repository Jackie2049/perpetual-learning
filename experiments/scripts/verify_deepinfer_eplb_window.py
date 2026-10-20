#!/usr/bin/env python3
"""CPU tensor checks for DeepInfer PR #107's expert-count identity contract.

Python 3.12+, torch and pytest required. Reads immutable Git blobs; no checkout
edits, full DeepInfer import, GPU, distributed collectives or weight shuffle.
Runs original PR test function bodies and selected original recording/decoding
logic with real CPU tensors. The process group and placement change are fakes.
Optional --check-step-lifecycle exercises real manager views/step/local stats
below the collective flush threshold; it does not execute count-producing kernels.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

BASE = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
HEAD = '578752f660e68c66412d4d0c08dccfc84c6724dd'
EPLB = 'deepinfer/model_executor/eplb/__init__.py'


def blob(root, revision, path):
    data = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=root)
    print(revision[:7], path, 'sha256=' + hashlib.sha256(data).hexdigest())
    return data.decode()


def load_balancer(source, envs):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExpertLoadBalancer')
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                ('__init__', 'step', '_update_load_balance_stats', '_update_rearrangement',
                 '_local_physical_to_logical_map')]
    # Only isolate the CPU tensor decoding block; no CUDA/shuffle/all_reduce.
    rearrange = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_rearrange_experts')
    start = next(i for i, n in enumerate(rearrange.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id in ('window_size', 'num_physical_experts')
                     for t in ast.walk(n.targets[0])))
    stop = next(i for i, n in enumerate(rearrange.body) if isinstance(n, ast.Expr) and
                isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'dist.all_reduce')
    decode = ast.parse('def decode_window(self):\n    ep_group = self._ep_group\n').body[0]
    decode.body += rearrange.body[start:stop] + [ast.Return(ast.Name('logical_expert_token_counts', ast.Load()))]
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
               ('resolve_num_extra_experts', 'get_initial_global_physical_to_logical_map')]
    isolated = ast.ClassDef(name='ExpertLoadBalancer', bases=[], keywords=[], body=selected + [decode], decorator_list=[])
    module = ast.Module(body=ast.parse('from __future__ import annotations').body + helpers + [isolated], type_ignores=[])
    ns = {'torch': torch, 'envs': envs}
    exec(compile(ast.fix_missing_locations(module), '<original-isolated-eplb>', 'exec'), ns)
    return ns['ExpertLoadBalancer']


def schedule(cls, environ, forced):
    """Controlled legal map replacement, not a real policy-driven weight shuffle."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        for key, value in {
            'DEEPINFER_EXPERT_COUNT_WINDOW_SIZE': '4',
            'DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL': '8',
            'DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE': str(int(forced)),
        }.items():
            monkeypatch.setenv(key, value)
        environ.rearm_all()
        obj = cls(expert_weights=[[]], num_moe_layers=1, num_logical_experts=4,
                  num_physical_experts=8, ep_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
                  num_routed_experts=4, num_extra_experts=4, is_normal_engine=True)
        oracle = torch.zeros((4, 1, 4), dtype=torch.int)
        observations = []
        step_number = 0
        def rearrange():
            actual = obj.decode_window().clone()
            observations.append((step_number, actual, oracle.clone()))
            if len(observations) == 1:
                obj.physical_to_logical_map[0, :4] = torch.tensor([3, 1, 2, 0], dtype=torch.int32)
        obj._rearrange_experts = rearrange
        prompts = [True, False, True, False] if forced else [False] * 10
        for step_number, prompt in enumerate(prompts, 1):
            obj.expert_count[0] = torch.tensor([100, 1, 1, 1], dtype=torch.int)
            if not (prompt and obj._skip_rearrangement_on_prefill):
                index = obj._expert_count_window_index
                oracle[index].zero_().scatter_add_(
                    -1, obj.physical_to_logical_map[:, :4].long(), obj.expert_count)
            obj._update_rearrangement(prompt)
        assert [o[0] for o in observations] == ([2, 4] if forced else [2, 10])
        return observations


def step_lifecycle(cls, source, environ, label):
    tree = ast.parse(source)
    manager_classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in
                       ('ExpertLoadBalancerLayerState', 'ExpertLoadBalancerManager')]
    module = ast.Module(body=ast.parse('from __future__ import annotations').body + manager_classes, type_ignores=[])
    ns = {'torch': torch, 'NamedTuple': NamedTuple}
    exec(compile(module, '<original-eplb-manager>', 'exec'), ns)
    cases = (('prefill_skipped_window', 2, 8, True), ('decode_record', 2, 8, False),
             ('disabled_rearrange', 2, 0, False), ('single_rank', 1, 8, False))
    for name, ep_size, interval, prompt in cases:
        with pytest.MonkeyPatch.context() as monkeypatch:
            for key, value in {
                'DEEPINFER_EXPERT_COUNT_WINDOW_SIZE': '4' if interval else '0',
                'DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL': str(interval),
                'DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE': '0',
                'DEEPINFER_MOE_LB_STAT_INTERVAL': '128',
            }.items(): monkeypatch.setenv(key, value)
            environ.rearm_all()
            obj = cls(expert_weights=[[], []], num_moe_layers=2, num_logical_experts=4,
                      num_physical_experts=8, ep_group=SimpleNamespace(size=lambda: ep_size, rank=lambda: 0),
                      num_routed_experts=4, num_extra_experts=4, is_normal_engine=True)
            manager = ns['ExpertLoadBalancerManager']([(obj, [1, 0])], 2)
            first_layer = manager.layer(0)
            assert first_layer.expert_count.data_ptr() == obj.expert_count[1].data_ptr()
            assert first_layer.logical_count.data_ptr() == obj.logical_count[1].data_ptr()
            obj.expert_count.copy_(torch.arange(1, obj.expert_count.numel() + 1).reshape_as(obj.expert_count))
            first_layer.expert_count[0] += 10
            before = obj.expert_count.clone()
            pointer = obj.expert_count.data_ptr()
            manager.step(prompt)
            assert obj.expert_count.data_ptr() == pointer
            if ep_size == 1:
                assert torch.equal(obj.expert_count, before)
                assert obj._load_balance_stat_index == obj._expert_count_window_index == 0
            else:
                assert obj.expert_count.count_nonzero() == 0 and first_layer.expert_count.count_nonzero() == 0
                assert obj._load_balance_stat_index == 1
                assert torch.equal(obj._local_tokens_per_step[0], before.sum(-1))
                assert obj._expert_count_window_index == (1 if name == 'decode_record' else 0)
                if name == 'decode_record':
                    assert torch.equal(obj.decode_window()[0], before)
            assert manager.take_moe_load_balance_ratio() is None  # No collective flush in these cases.
            print(label, 'original manager/step/local stats PASS:', name)
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--check-step-lifecycle', action='store_true')
    args = parser.parse_args()
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    print('Python/torch CPU:', __import__('sys').version.split()[0], torch.__version__)
    env_ns = {}
    exec(compile(blob(args.source_root, HEAD, 'deepinfer/environ.py'), '<original-environ>', 'exec'), env_ns)
    environ = SimpleNamespace(**env_ns)
    baseline_source, patched_source = blob(args.source_root, BASE, EPLB), blob(args.source_root, HEAD, EPLB)
    baseline = load_balancer(baseline_source, environ.envs)
    patched = load_balancer(patched_source, environ.envs)
    tests = ast.parse(blob(args.source_root, HEAD, 'tests/models/test_eplb_window.py'))
    # Drop only package imports to avoid DeepInfer/platform/CUDA initialization.
    tests.body = [n for n in tests.body if not (isinstance(n, ast.ImportFrom) and n.module.startswith('deepinfer'))]
    ns = {'environ': environ, 'ExpertLoadBalancer': patched}
    exec(compile(tests, '<original-pr107-window-tests>', 'exec'), ns)
    names = [name for name in ns if name.startswith('test_')]
    assert len(names) == 3
    for name in names:
        with pytest.MonkeyPatch.context() as monkeypatch:
            ns[name](monkeypatch)
        print('original PR test body PASS:', name)

    final_observation = None
    for label, cls in (('baseline', baseline), ('pr107', patched)):
        for forced in (False, True):
            observations = schedule(cls, environ, forced)
            matches = [torch.equal(actual, expected) for _, actual, expected in observations]
            assert matches == ([True, False] if label == 'baseline' and forced else [True, True])
            print(label, 'after_prefill=' + str(forced), 'callback_steps=' + str([o[0] for o in observations]),
                  'history_matches_record_time_identity=' + str(matches))
            if label == 'baseline' and forced:
                final_observation = observations[-1]
                print('second trigger stale/current decode:', final_observation[1].squeeze(1).tolist())
                print('second trigger record-time oracle:', final_observation[2].squeeze(1).tolist())

    policy_ns = {}
    exec(compile(blob(args.source_root, BASE, 'deepinfer/model_executor/eplb/policy.py'),
                 '<original-cpu-policy>', 'exec'), policy_ns)
    _, stale, expected = final_observation
    stale_replicas = policy_ns['rebalance_experts_global'](stale, 8, 4)[2]
    expected_replicas = policy_ns['rebalance_experts_global'](expected, 8, 4)[2]
    assert not torch.equal(stale_replicas, expected_replicas)
    print('real CPU policy replica allocation, stale/oracle:', stale_replicas.tolist(), expected_replicas.tolist())

    policy_test_source = blob(args.source_root, BASE, 'tests/models/test_eplb_policy.py')
    policy_tests = ast.parse(policy_test_source)
    policy_tests.body = [n for n in policy_tests.body if not
                        (isinstance(n, ast.ImportFrom) and n.module.startswith('deepinfer'))]
    # Invoke original bodies with the original parameter data, CPU only; do not evaluate
    # the CUDA fixture decorator or import the full package's test conftest.
    for node in policy_tests.body:
        if isinstance(node, ast.FunctionDef): node.decorator_list = []
    exec(compile(policy_tests, '<original-cpu-policy-tests>', 'exec'), policy_ns)
    policy_checks = 0
    for case in policy_ns['CASES']:
        for name in ('test_rebalance_experts_global_structure', 'test_rebalance_experts_global_deterministic'):
            policy_ns[name](*case)
            policy_checks += 1
    for distribution in ('uniform', 'skewed', 'zeros', 'spike'):
        policy_ns['test_rebalance_experts_global_edge_distributions'](distribution)
        policy_checks += 1
    policy_ns['test_rebalance_does_not_modify_history'](torch.device('cpu'))
    policy_checks += 1
    original_test_tree = ast.parse(policy_test_source)
    objective = next(n for n in original_test_tree.body if isinstance(n, ast.FunctionDef) and
                     n.name == 'test_batch_greedy_objective_and_ties')
    cases = ast.literal_eval(objective.decorator_list[0].args[1])
    for history, physical, expected_indices in cases:
        policy_ns['test_batch_greedy_objective_and_ties'](history, physical, expected_indices, torch.device('cpu'))
        policy_checks += 1
    assert policy_checks == 26
    print('PASS: 3 original window test bodies, 4 controlled schedules, 1 CPU policy-input contrast,')
    print('plus 26 original policy CPU cases (decorators removed, parameter data retained).')
    if args.check_step_lifecycle:
        count = step_lifecycle(baseline, baseline_source, environ, 'baseline')
        count += step_lifecycle(patched, patched_source, environ, 'pr107')
        print(f'PASS: {count} additional original manager/view/step CPU controls; no kernels or collective flush.')
    print('No full package, distributed collectives, actual weight shuffle, accelerator or throughput validation.')


if __name__ == '__main__':
    main()
