#!/usr/bin/env python3
"""Execute pinned original async sync-control methods with local fakes.

Usage: python3 verify_verl_weight_sync_control.py /path/to/verl-8250
Only manager/update adapter control bodies execute. Ray, actor handles, sender,
acknowledgements, cache and weights are fakes; no network, CUDA or real content
verification. Failure propagation is not a native fault reproduction or fix.
"""
import argparse
import ast
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time


class InjectedFailure(RuntimeError):
    pass


def extract(path, class_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'update_weights')
    fn.decorator_list = []
    exec(compile('from __future__ import annotations\n' + ast.unparse(fn), str(path), 'exec'), namespace)
    print(class_name, 'sha256=' + hashlib.sha256(path.read_bytes()).hexdigest())
    return namespace['update_weights']


async def adapter_case(fn, fail=None, has_server=True, version=9, wire_format='named_tensors'):
    trace = []

    def event(name):
        trace.append(name)
        if name == fail:
            raise InjectedFailure(name)

    class Ack:
        def __await__(self):
            async def finish(): event('receive_ack')
            return finish().__await__()

    class Sender:
        def __init__(self, **kw): pass
        async def async_send_weights(self, weights): event('send')

    async def execute(*args, **kw):
        event('execute')
        return Ack() if has_server else None

    async def clear(): event('clear_cache')
    async def tag(step):
        assert step == version
        event('set_version')
    async def delta(*args, **kw): event('delegate_delta')
    adapter = SimpleNamespace(
        config=SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)),
        zmq_handle='fake-no-socket', use_shm=False, _execute_method=execute,
        _has_server=has_server, replica_rank=1, rollout_rank=1, _update_delta_weights=delta,
        server_handle=SimpleNamespace(clear_kv_cache=SimpleNamespace(remote=clear),
                                      set_global_steps=SimpleNamespace(remote=tag)),
    )
    # A separate global namespace per case avoids sender binding across cases.
    fn.__globals__['BucketedWeightSender'] = Sender
    expected = ['delegate_delta'] if wire_format == 'delta_flush' else ['execute', 'send']
    if wire_format == 'named_tensors' and has_server:
        expected += ['receive_ack', 'clear_cache'] + (['set_version'] if version is not None else [])
    try:
        await fn(adapter, iter(()), global_steps=version, wire_format=wire_format)
    except InjectedFailure:
        assert fail is not None
        assert trace == expected[:expected.index(fail) + 1]
    except AssertionError:
        assert wire_format == 'invalid' and trace == []
    else:
        assert fail is None and trace == expected
    return trace


async def manager_case(fn, fail=None, backend='fake-transport'):
    trace = []

    def event(name):
        trace.append(name)
        if fail == name:
            raise InjectedFailure(name)

    class Ref:
        def __init__(self, name, value): self.name, self.value = name, value
        def resolve(self): event(self.name); return self.value

    class Group:
        world_size = 1
        def __init__(self, role): self.role = role
        def update_weights(self, **kw):
            event(self.role + '_submit')
            return [Ref(self.role + '_ack', {'test_metric': 1} if self.role == 'actor' else {})]
        def execute_checkpoint_engine(self, ops):
            assert ops == ['finalize']
            event(self.role + '_finalize_submit')
            return [Ref(self.role + '_finalize_ack', {})]

    def get(refs): return [ref.resolve() for ref in refs]
    def rollout_group(**kw): event('make_rollout_group'); return Group('rollout')
    async def abort(): event('abort')
    async def release(): event('release_cache')
    async def resume_cache(): event('resume_cache')
    async def resume_generation(): event('resume_generation')
    manager = SimpleNamespace(
        backend=backend, actor_wg=Group('actor'), replicas=[SimpleNamespace(workers=['fake-worker'])],
        abort_replicas=abort, release_kv_cache_replicas=release,
        build_process_group=lambda rollout: event('build_group'),
        resume_kv_cache_replicas=resume_cache, resume_generation_replicas=resume_generation,
    )
    fn.__globals__.update(ray=SimpleNamespace(get=get), RayWorkerGroup=rollout_group,
                         RayClassWithInitArgs=lambda **kw: None, _worker_cls=object)
    expected = ['actor_submit', 'actor_ack'] if backend == 'naive' else [
        'abort', 'make_rollout_group', 'release_cache', 'build_group',
        'actor_submit', 'rollout_submit', 'actor_ack', 'rollout_ack',
        'actor_finalize_submit', 'rollout_finalize_submit',
        'actor_finalize_ack', 'rollout_finalize_ack', 'resume_cache', 'resume_generation']
    try:
        result = await fn(manager, global_steps=9)
    except InjectedFailure:
        assert fail is not None and trace == expected[:expected.index(fail) + 1]
    else:
        assert fail is None and trace == expected
        assert result == ({} if backend == 'naive' else {'test_metric': 1})
    return trace


async def check(root):
    adapter_path = root / 'verl/workers/rollout/vllm_rollout/vllm_rollout.py'
    manager_path = root / 'verl/checkpoint_engine/base.py'
    adapter = extract(adapter_path, 'ServerAdapter', {'time': time, 'logger': SimpleNamespace(info=lambda *a: None)})
    manager = extract(manager_path, 'CheckpointEngineManager', {})
    count = 0
    for options in ({}, {'has_server': False}, {'version': None},
                    {'wire_format': 'invalid'}, {'wire_format': 'delta_flush'}):
        await adapter_case(adapter, **options)
        count += 1
    for stage in ('execute', 'send', 'receive_ack', 'clear_cache', 'set_version'):
        trace = await adapter_case(adapter, fail=stage)
        print('adapter failure:', stage, '->', trace)
        count += 1
    await manager_case(manager)
    await manager_case(manager, backend='naive')
    count += 2
    for stage in ('abort', 'release_cache', 'build_group', 'actor_ack', 'rollout_ack',
                  'actor_finalize_ack', 'rollout_finalize_ack', 'resume_cache', 'resume_generation'):
        trace = await manager_case(manager, fail=stage)
        print('manager failure:', stage, '->', trace)
        count += 1
    print(f'PASS {count} control/exception cases; injected errors propagate and subsequent stages do not run.')
    print('No native Ray cancellation/concurrency, transport, device/cache/weights, rollback or recovery proof.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    args = parser.parse_args()
    asyncio.run(check(args.source_root))


if __name__ == '__main__':
    main()
