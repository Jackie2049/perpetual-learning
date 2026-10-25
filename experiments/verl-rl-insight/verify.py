"""#8315/#8357 original logger tests, isolated original classes; no monitoring service.

Python >=3.10, pytest and packaging. No installs or checkout changes. Original
RLInsightLogger/RolloutTraceConfig classes execute with original decorators;
package shells skip unrelated verl/orjson/Ray imports. Tests retain their own
MagicMock client, fixtures and assertions. This is not native backend integration.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional

from packaging.version import Version
import pytest

VERSIONS = {
    'ancestor': 'cd58f550fb1c4f5a3fe29da38e6f0002ec462fd3',
    'previous': '4d6b3eb8606e24c1d70d648e92a1114129ec6c62',
    'current': '0cea139623cc8913d2241d21223df5682474041c',
    'gauge-base': '3c6d9b43b67c74ec343d77874cd8fa6e841523da',
    'gauge-head': 'cd12877931cdd441adb731978a8b3785858dba27',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    parser.add_argument('--revision', choices=VERSIONS, default='current')
    args = parser.parse_args()
    revision = VERSIONS[args.revision]
    gauge_suite = args.revision.startswith('gauge-')
    hashes = {}

    def read(rev, path):
        data = subprocess.check_output(['git', 'show', f'{rev}:{path}'], cwd=args.repo,
                                       env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})
        print(rev[:8], path, hashlib.sha256(data).hexdigest(), flush=True)
        hashes[rev + ':' + path] = hashlib.sha256(data).hexdigest()
        compile(data, 'git:' + rev + ':' + path, 'exec', dont_inherit=True)
        return data

    for name in ('verl', 'verl.utils'):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    for leaf, class_name in (('tracking', 'RLInsightLogger'), ('rollout_trace', 'RolloutTraceConfig')):
        name = 'verl.utils.' + leaf
        module = ModuleType(name)
        module.__file__ = 'git:' + revision + ':' + leaf
        sys.modules[name] = module
        tree = ast.parse(read(revision, 'verl/utils/' + leaf + '.py'))
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        assert len(nodes) == 1
        module.__dict__.update(os=os, inspect=inspect, wraps=wraps, Version=Version,
                               contextmanager=contextmanager, SimpleNamespace=SimpleNamespace,
                               Any=Any, Callable=Callable, Optional=Optional, logger=logging.getLogger(name))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), module.__file__, 'exec',
                     dont_inherit=True), module.__dict__)

    if not gauge_suite and args.revision != 'ancestor':
        tree = ast.parse(read(revision, 'verl/workers/engine_workers.py'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'update_weights')
        decorator = next(n for n in method.decorator_list if isinstance(n, ast.Call) and
                         ast.unparse(n.func) == 'RLInsightLogger.trace_state_decorator')
        expression = next(k.value for k in decorator.keywords if k.arg == 'state_lane_id')
        lane = eval(compile(ast.Expression(expression), '<original-worker-lane>', 'eval'))(SimpleNamespace(rank=2))
        assert lane == ('rank_2' if args.revision == 'current' else 2)
        print('Original worker lane expression:', repr(lane), type(lane).__name__)

    tests = ModuleType('_original_rl_insight_tests')
    tests.__file__ = str(Path(__file__).resolve())
    sys.modules[tests.__name__] = tests
    test_revision = VERSIONS['gauge-head'] if gauge_suite else VERSIONS['current']
    exec(compile(read(test_revision, 'tests/utils/test_rl_insight_logger_on_cpu.py'),
                 'git:' + test_revision + ':tests/utils/test_rl_insight_logger_on_cpu.py', 'exec'), tests.__dict__)

    class OriginalModule(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        results = []

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalModule.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                self.results.append((report.nodeid.rsplit('::', 1)[-1], report.outcome))

    def no_network(event, args):
        # asyncio creates anonymous local socketpairs for its self-wakeup pipe.
        # Permit only AF_UNIX socket creation, not bind/connect or INET sockets.
        if event == 'socket.__new__' and args[1] == socket.AF_UNIX:
            return
        if event.startswith('socket.'):
            raise RuntimeError('No network permitted in isolated logger test')

    sys.addaudithook(no_network)
    adapter = Adapter()
    code = pytest.main([str(Path(__file__).resolve()), '--noconftest', '-o', 'addopts=',
                        '-p', 'no:cacheprovider', '-q', '--tb=short'], plugins=[adapter])
    counts = Counter(outcome for _, outcome in adapter.results)
    failures = {name for name, outcome in adapter.results if outcome == 'failed'}
    if gauge_suite:
        if args.revision == 'gauge-head':
            assert code == 0 and counts == {'passed': 7}, (code, counts)
        else:
            assert code == 1 and counts == {'passed': 6, 'failed': 1}, (code, counts)
            assert failures == {'test_log_lazy_inits_and_writes_gauge'}, failures
    elif args.revision == 'ancestor':
        expected = {'test_log_uses_metric_log_with_step_when_available',
                    'test_trace_state_decorator_resolves_callable_lane',
                    'test_trace_state_decorator_passes_literal_lane',
                    'test_trace_state_decorator_skips_trace_when_disabled',
                    'test_trace_state_decorator_rejects_sync_method'}
        assert code == 1 and counts == {'passed': 7, 'failed': 5} and failures == expected, (code, counts, failures)
    else:
        assert code == 0 and counts == {'passed': 12}, (code, counts)
    assert all(name not in sys.modules for name in ('torch', 'ray', 'orjson', 'rl_insight'))
    if gauge_suite:
        print('GAUGE_RESULT ' + json.dumps({'revision': revision, 'counts': dict(counts),
              'tests': adapter.results, 'source_sha256': hashes,
              'environment': {'python': platform.python_version(), 'pytest': pytest.__version__,
                              'system': platform.system()},
              'scope': '7 original gauge PR tests and fixtures on isolated original classes; MagicMock client; '
                       'not native verl imports or monitoring service'}, sort_keys=True))
    print(args.revision, dict(counts), 'original fixtures/client mocks; no ingestion, timeline or hardware validation')


if __name__ == '__main__':
    main()
