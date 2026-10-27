"""Fixed rLLM/Slime bookkeeping and cancellation controls, not RL validation."""
import argparse
import ast
import asyncio
import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace as NS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rllm', type=Path, required=True)
parser.add_argument('--slime', type=Path, required=True)
args = parser.parse_args()
R, S = args.rllm.resolve(), args.slime.resolve()
oids = {R: '4e828c81680e5a85b638ceedeaf09d1e90109ca8', S: '540ea919e45b8b929198abce008a103be9cfc779'}
paths = {R: ['rllm/trainer/unified_trainer.py', 'rllm/trainer/sync_coordinator.py',
             'rllm/trainer/buffer.py', 'rllm/trainer/algorithms/config.py',
             'rllm/engine/rollout/rollout_engine.py', 'rllm/trainer/agent_trainer.py'],
         S: ['train.py', 'slime/ray/rollout.py', 'slime/rollout/sglang_rollout.py',
             'slime/data/batch_builder.py', 'slime/utils/arguments.py',
             'slime/rollout/filter_hub/base_types.py', 'slime/data/data_source.py']}
sources = {}
for root, files in paths.items():
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == oids[root]
    for path in files:
        source = (root / path).read_text()
        assert source == subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{path}'], text=True)
        sources[root / path] = source, ast.parse(source)

def module(root, path, name):
    # These two complete modules import only the standard library.
    obj = types.ModuleType(name)
    sys.modules[name] = obj
    exec(compile(sources[root / path][0], str(root / path), 'exec'), obj.__dict__)
    return obj

def body(root, path, name, owner=None, scope=None):
    source, tree = sources[root / path]
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    scope = {} if scope is None else scope
    exec(compile('from __future__ import annotations\n' + textwrap.dedent(ast.get_source_segment(source, node)), str(root / path), 'exec'), scope)
    return scope[name]

coord = module(R, 'rllm/trainer/sync_coordinator.py', '_rllm_c3_coord')
filters = module(S, 'slime/rollout/filter_hub/base_types.py', '_slime_c3_filters')
C, Config = coord.SyncCoordinator, coord.SyncCoordinatorConfig

async def checks():
    # 1. Dispatch quota is a sync-window count, not released by consume.
    c = C(Config(mini_batch_size=2, group_size=3, staleness_threshold=0, trigger_parameter_sync_step=1))
    assert c.config.max_rollout_quota == 2
    c.on_group_dispatched(); c.on_group_dispatched()
    assert not c.has_quota()
    c.on_group_consumed()
    assert c._in_flight == 1 and c._quota_used == 2 and not c.has_quota()
    c.on_group_filtered()
    assert c._in_flight == 0 and c._quota_used == 1 and c.has_quota()

    # 2. A sync carries outstanding groups into the next dispatch window.
    c = C(Config(mini_batch_size=2, group_size=3, staleness_threshold=.5, trigger_parameter_sync_step=2))
    assert c.config.max_rollout_quota == 6
    for _ in range(4): c.on_group_dispatched()
    c.on_group_consumed()
    c.on_training_step_complete()
    assert not c.should_sync()
    c.on_training_step_complete()
    assert c.should_sync()
    c.on_sync_complete()
    assert c.weight_version == 1 and c._quota_used == c._in_flight == 3 and not c.should_sync()

    # 3. Real asyncio generation pause/resume, without any serving backend.
    c.pause_generation()
    waiter = asyncio.create_task(c.wait_for_generation_allowed())
    await asyncio.sleep(0)
    assert not waiter.done()
    c.resume_generation()
    await asyncio.wait_for(waiter, .5)

    # 4. Tracked failure releases waiters and preserves the cause.
    c = C(Config(1, 1, 0, 1))
    failure = ValueError('controlled task failure')
    async def fail(): raise failure
    t = asyncio.create_task(fail())
    c.track_task(t)
    try:
        await asyncio.wait_for(c.wait_for_task_error(), .5)
    except RuntimeError as exc:
        assert exc.__cause__ is failure
    else:
        raise AssertionError('failure was not surfaced')
    await c.wait_for_drain()

    # 5. Tracked cancellation drains without treating cancellation as an error.
    gate = asyncio.Event()
    t = asyncio.create_task(gate.wait())
    c.track_task(t)
    c.cancel_tracked_tasks()
    await asyncio.wait_for(c.wait_for_drain(), .5)
    assert t.cancelled() and not c._task_errors

    # 6. Group version is the minimum observed step version, or zero if absent.
    min_version = body(R, 'rllm/trainer/buffer.py', '_min_weight_version', 'TrajectoryGroupBuffer')
    assert min_version([]) == 0
    episode = NS(trajectories=[NS(steps=[NS(weight_version=None), NS(weight_version=5), NS(weight_version=2)])])
    assert min_version([episode]) == 2

    # 7. Original Slime cancellation helpers with fake state/sample, no HTTP.
    scope = {'asyncio': asyncio, 'Sample': NS(Status=NS(ABORTED='aborted'))}
    run_abort = body(S, 'slime/rollout/sglang_rollout.py', '_run_request_abortable_generate', scope=scope)
    run_server = body(S, 'slime/rollout/sglang_rollout.py', '_run_server_abort_generate')
    state = NS(cancellable_tasks=set(), active_server_generations=0)
    sample = NS(status='pending')
    gate = asyncio.Event()
    t = asyncio.create_task(run_abort(state, sample, gate.wait()))
    await asyncio.sleep(0)
    state.cancellable_tasks.remove(t)  # abort() unregisters its own cancellation
    t.cancel()
    assert await t is sample and sample.status == 'aborted'
    t = asyncio.create_task(run_abort(state, NS(status='pending'), gate.wait()))
    await asyncio.sleep(0)
    t.cancel()  # unrelated cancellation must propagate
    try: await t
    except asyncio.CancelledError: pass
    else: raise AssertionError('unrelated cancellation was swallowed')
    assert not state.cancellable_tasks
    t = asyncio.create_task(run_server(state, gate.wait()))
    await asyncio.sleep(0)
    assert state.active_server_generations == 1
    t.cancel()
    try: await t
    except asyncio.CancelledError: pass
    assert state.active_server_generations == 0

    # 8. Original conditional filter admits a rejected group if explicitly
    # configured to keep it when too few candidates remain.
    for keep in (False, True):
        for insufficient in (False, True):
            for remaining in (2, 3, 4):
                out = filters.DynamicFilterOutput(keep, keep_when_insufficient=insufficient)
                expected = not keep and not (insufficient and remaining <= 3)
                assert filters.should_drop_dynamic_filter_output(out, remaining_batch_size=remaining, target_data_size=3) == expected

asyncio.run(checks())
print(json.dumps({'result': 'pass', 'python': sys.version.split()[0], 'rllm_commit': oids[R],
                  'slime_commit': oids[S], 'source_asts': len(sources), 'control_groups': 8,
                  'limits': 'Real stdlib asyncio coordinator and filter modules; extracted cancellation/version helpers with fake sample/state. No framework import, HTTP, Ray, storage, optimizer, device or performance execution.'}, indent=2))
