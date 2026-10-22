#!/usr/bin/env python3
"""CPU tensor checks for DeepInfer PR #107's expert-count identity contract.

Python 3.12+, torch and pytest required. Reads immutable Git blobs; no checkout
edits, full DeepInfer import, GPU, distributed collectives or weight shuffle.
Runs original PR test function bodies and selected original recording/decoding
logic with real CPU tensors. The process group and placement change are fakes.
Optional --check-step-lifecycle exercises real manager views/step/local stats
below the collective flush threshold; it does not execute count-producing kernels.
--tied-allocation-only instead checks #191's original CPU policy test bodies
against base/fix and an independent exact-rational greedy oracle, with no fixtures
evaluated, device discovery, window/manager substitution or weight shuffle.
--peak-gain-only checks #212's original tests with pytest against main/#191/#212.
Only the device fixture's collection is narrowed to CPU before execution; no
availability probe is evaluated. Policy and test assertions remain unchanged.
"""
import argparse
import ast
import hashlib
from itertools import product
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

BASE = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
HEAD = '578752f660e68c66412d4d0c08dccfc84c6724dd'
EPLB = 'deepinfer/model_executor/eplb/__init__.py'
TIED_BASE = '9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'
TIED_HEAD = '090e942c4c4365d9d0e38b41483b92452bd7a10f'
PEAK_HEAD = '3b950ebc0b66fd507ecdff02ef3193136b2fa5b9'


def blob(root, revision, path):
    data = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=root,
                                   env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})
    print(revision[:7], path, 'sha256=' + hashlib.sha256(data).hexdigest())
    return data.decode()


def rational_allocation(history, extras):
    """Brute-force each greedy choice, not a global placement optimum."""
    counts = [1] * len(history[0])
    choices = []
    for _ in range(extras):
        costs = []
        for expert in range(len(counts)):
            cost = sum(max(Fraction(load, counts[j] + int(j == expert))
                           for j, load in enumerate(step)) for step in history)
            total_load = Fraction(sum(step[expert] for step in history), counts[expert])
            costs.append((cost, -total_load, expert))
        selected = min(costs)[2]
        counts[selected] += 1
        choices.append(selected)
    return counts, choices


def tied_allocation(root):
    """Retain upstream functions/assertions/parameter data; omit decorators only."""
    source = blob(root, TIED_HEAD, 'tests/models/test_eplb_policy.py')
    original = ast.parse(source)
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and n.module.startswith('deepinfer'))]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # In particular, do not evaluate CUDA availability.
    device = torch.device('cpu')
    policies = {}
    for label, revision in (('base', TIED_BASE), ('fix', TIED_HEAD)):
        ns = {}
        exec(compile(blob(root, revision, 'deepinfer/model_executor/eplb/policy.py'),
                     '<original-cpu-policy-' + label + '>', 'exec'), ns)
        policies[label] = ns['rebalance_experts_global']
        exec(compile(tree, '<original-policy-test-bodies>', 'exec'), ns)
        calls = []
        for case in ns['CASES']:
            for name in ('test_rebalance_experts_global_structure', 'test_rebalance_experts_global_deterministic'):
                calls.append((name, case))
        distributions = next(n for n in original.body if isinstance(n, ast.FunctionDef) and
                             n.name == 'test_rebalance_experts_global_edge_distributions')
        for distribution in ast.literal_eval(distributions.decorator_list[0].args[1]):
            calls.append((distributions.name, (distribution,)))
        calls.append(('test_rebalance_does_not_modify_history', (device,)))
        tied = next(n for n in original.body if isinstance(n, ast.FunctionDef) and
                    n.name == 'test_rebalance_replicates_tied_hot_experts')
        params = {ast.literal_eval(d.args[0]): eval(compile(ast.Expression(d.args[1]), '<original-test-params>', 'eval'), ns)
                  for d in tied.decorator_list if isinstance(d, ast.Call)}
        for dtype, load in product(params['dtype'], params['load']):
            calls.append((tied.name, (load, dtype, device)))
        for name in ('test_rebalance_tied_hot_experts_with_placement',
                     'test_rebalance_prefers_lower_peak_cost_over_tie_break_load'):
            calls.append((name, (device,)))
        objective = next(n for n in original.body if isinstance(n, ast.FunctionDef) and
                         n.name == 'test_batch_greedy_objective_and_ties')
        for case in ast.literal_eval(objective.decorator_list[0].args[1]):
            calls.append((objective.name, (*case, device)))
        assert len(calls) == 32
        failures = []
        for name, args in calls:
            try:
                ns[name](*args)
            except AssertionError:
                failures.append(name)
        expected = ([tied.name] * 4 + ['test_rebalance_tied_hot_experts_with_placement']) if label == 'base' else []
        assert failures == expected, (label, failures)
        print(label, 'original CPU bodies:', len(calls) - len(failures), 'passed;', len(failures), 'expected failures', failures)
        for physical, local in ((6, 1), (288, 36)):
            history = torch.zeros((1, 1, 4 if physical == 6 else 256), dtype=torch.int32)
            history[..., 1:3] = 5000
            counts = ns['rebalance_experts_global'](history, physical, local)[2][0]
            print(label, 'physical/local', physical, local, 'first four replica counts', counts[:4].tolist())
    checks = 0
    for values in product((0, 5000), repeat=6):
        history = [list(values[:3]), list(values[3:])]
        expected_counts, expected_choices = rational_allocation(history, 6)
        for dtype in (torch.int32, torch.float32):
            physical, _, counts = policies['fix'](torch.tensor(history, dtype=dtype).unsqueeze(1), 9, 1)
            assert counts.tolist() == [expected_counts]
            assert physical[0, 3:].tolist() == expected_choices
            checks += 1
    for load in (5000, 65536):
        scores = torch.tensor([0, load, load, 0], dtype=torch.int32) + torch.arange(4) * 1e-4
        assert scores[1] == scores[2]
        print('FP32 perturbation rounds to tie at', load, ':', scores.tolist())
    print('PASS:', checks, 'CPU dtype/history comparisons to exact greedy oracle; no global-optimality or device claim.')


def peak_gain(root):
    """Original pytest assertions and fixture, restricted to explicit CPU collection."""
    from collections import Counter
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    path = 'tests/models/test_eplb_policy.py'
    source = blob(root, PEAK_HEAD, path)
    tree = ast.parse(source)
    fixture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'policy_device')
    # The original decorator evaluates CUDA availability during module import.
    # Select CPU explicitly instead; preserve the fixture body and every test.
    fixture.decorator_list = ast.parse('@pytest.fixture(params=["cpu"])\ndef selected(): pass').body[0].decorator_list
    ast.fix_missing_locations(tree)
    failures_expected = {
        'main': Counter(test_rebalance_replicates_hot_experts_on_a_peak_plateau=2,
                        test_rebalance_hot_replicas_remain_consistent_after_placement=1,
                        test_rebalance_preserves_small_gains_in_long_histories=2,
                        test_rebalance_replica_choices_match_exact_greedy_costs=1),
        'pr191': Counter(test_rebalance_preserves_small_gains_in_long_histories=2),
        'pr212': Counter(),
    }
    for label, revision in (('main', TIED_BASE), ('pr191', TIED_HEAD), ('pr212', PEAK_HEAD)):
        policy_name = 'deepinfer.model_executor.eplb.policy'
        policy = types.ModuleType(policy_name)
        policy_source = blob(root, revision, 'deepinfer/model_executor/eplb/policy.py')
        exec(compile(policy_source, '<original-policy-' + label + '>', 'exec'), policy.__dict__)
        sys.modules[policy_name] = policy  # Avoid executing the parent package/DeepEP imports.
        tests = types.ModuleType('_original_peak_tests_' + label)
        tests.__file__ = str(Path(__file__).resolve())
        exec(compile(tree, path, 'exec'), tests.__dict__)
        sys.modules[tests.__name__] = tests

        class OriginalTests(pytest.Module):
            def _getobj(self):
                return tests

        class Reports:
            passed = 0
            failed = Counter()
            skipped = 0
            exception_types = Counter()

            def pytest_pycollect_makemodule(self, module_path, parent):
                if module_path == Path(__file__).resolve():
                    return OriginalTests.from_parent(parent, path=module_path)

            def pytest_runtest_logreport(self, report):
                if report.when == 'call':
                    if report.passed:
                        self.passed += 1
                    elif report.failed:
                        self.failed[report.nodeid.rsplit('::', 1)[-1].split('[', 1)[0]] += 1
                if report.skipped:
                    self.skipped += 1

            def pytest_runtest_makereport(self, item, call):
                if call.when == 'call' and call.excinfo is not None:
                    self.exception_types[call.excinfo.typename] += 1

        reports = Reports()
        code = pytest.main(['-q', '--tb=no', '--noconftest', '-p', 'no:cacheprovider',
                            '-o', 'markers=ci: upstream CPU marker\nci_npu: upstream platform marker',
                            str(Path(__file__).resolve())], plugins=[reports])
        expected = failures_expected[label]
        assert reports.failed == expected and reports.skipped == 0, (label, reports.failed, reports.skipped)
        assert reports.exception_types == Counter(AssertionError=sum(expected.values()))
        assert reports.passed + sum(reports.failed.values()) == 32
        assert int(code) == (1 if expected else 0), (label, code)
        print(label, reports.passed, 'passed;', sum(reports.failed.values()),
              'expected assertion failures:', dict(reports.failed), flush=True)
        for dtype in (torch.int32, torch.float32):
            tokens = torch.zeros((1000, 1, 4), dtype=dtype)
            tokens[:-1, :, 1:3] = 32768
            tokens[-1, :, 3] = 1
            print(label, str(dtype), 'long-history replica counts',
                  policy.rebalance_experts_global(tokens, 5, 1)[2].tolist(), flush=True)

    exact = [Fraction(999 * 32768 + 1)] * 3 + [Fraction(999 * 32768) + Fraction(1, 2)]
    rounded = torch.tensor([float(cost) for cost in exact], dtype=torch.float32)
    assert rounded.unique().numel() == 1
    print('exact long-history candidate costs', [str(cost) for cost in exact],
          'FP32 costs', rounded.tolist())
    print('Original #212 oracle covers 729 two-step histories and 6 successive greedy choices each.')
    print('CPU collection only; no CUDA fixture/probe, package root, manager, kernel, shuffle or throughput test.')


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
    parser.add_argument('--tied-allocation-only', action='store_true')
    parser.add_argument('--peak-gain-only', action='store_true')
    args = parser.parse_args()
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    print('Python/torch CPU:', __import__('sys').version.split()[0], torch.__version__)
    if args.peak_gain_only:
        assert not args.check_step_lifecycle and not args.tied_allocation_only, 'Choose one isolated policy check'
        peak_gain(args.source_root)
        return
    if args.tied_allocation_only:
        assert not args.check_step_lifecycle, 'Choose either tied allocation or window lifecycle checks'
        tied_allocation(args.source_root)
        return
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
