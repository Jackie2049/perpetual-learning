#!/usr/bin/env python3
"""Old-logprob routing controls and probability/metadata counterexamples.

Usage: python3 verify_verl_logprob_provenance.py /path/to/verl-8250
Executes original base bypass branch and separate-async routing method with
fake queue/actor/base recomputation. CPU slots hold identifiers, not tensors.
NumPy examples are not real model logits, vLLM processing or synchronization.
Optional --check-snapshot-failures adds 21 controller/worker/TaskRunner controls.
Those use scalar snapshots and fake trainer/tracker/queue, not native recovery.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import sys
from textwrap import indent
from types import ModuleType, SimpleNamespace

import numpy as np


def method(path, class_name, name='_compute_old_log_prob'):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    return ast.unparse(fn)


def routing_controls(root):
    paths = {
        'base': root / 'verl/trainer/ppo/v1/trainer_base.py',
        'separate': root / 'verl/trainer/ppo/v1/trainer_separate_async.py',
        'server': root / 'verl/workers/rollout/vllm_rollout/vllm_async_server.py',
        'client': root / 'verl/workers/rollout/llm_server.py',
    }
    for key, path in paths.items():
        print(key, hashlib.sha256(path.read_bytes()).hexdigest())

    class Queue:
        def __init__(self): self.data = {'rollout_log_probs': [-1., -2.]}
        def kv_batch_get(self, **kw): return self.data.copy()
        def kv_batch_put(self, **kw): self.data.update(kw['fields'])

    queue = Queue()
    ns = {'tq': queue}
    exec(compile('from __future__ import annotations\n' + method(paths['base'], 'PPOTrainer'),
                 '<original-base-old-logprob>', 'exec'), ns)
    batch = SimpleNamespace(keys=['sample'], partition_id='fake')
    bypass = SimpleNamespace(config=SimpleNamespace(algorithm={'rollout_correction': {'bypass_mode': True}}))
    assert ns['_compute_old_log_prob'](bypass, batch, {}) is batch
    assert queue.data['old_log_probs'] == [-1., -2.]

    class Actor:
        def __init__(self, current): self.current, self.slots, self.events = current, {0: 10}, []
        def save_model_to_cpu(self, slot):
            self.slots[slot] = self.current
            self.events.append(('save', slot, self.current))
        def restore_model_from_cpu(self, slot):
            self.current = self.slots[slot]
            self.events.append(('restore', slot, self.current))
        def clear_cpu_model(self, slot):
            del self.slots[slot]
            self.events.append(('clear', slot))

    class Base:
        def _compute_old_log_prob(self, batch, metrics):
            self.observed = 'rollout-copy' if self.bypass else self.actor_rollout_wg.current
            return batch

    ns = {'Base': Base}
    source = 'from __future__ import annotations\nclass Probe(Base):\n' + indent(
        method(paths['separate'], 'PPOTrainerSeparateAsync'), '    ')
    exec(compile(source, '<original-separate-routing>', 'exec'), ns)
    for use_bypass in (False, True):
        for trigger in (0, 1, 3):
            probe = ns['Probe']()
            probe.bypass = use_bypass
            probe.config = SimpleNamespace(algorithm={'rollout_correction': {'bypass_mode': use_bypass}})
            probe.local_trigger_step = trigger
            probe.actor_rollout_wg = Actor(10 + trigger)
            assert probe._compute_old_log_prob(batch, {}) is batch
            actor = probe.actor_rollout_wg
            assert actor.current == 10 + trigger
            if use_bypass:
                assert probe.observed == 'rollout-copy' and actor.events == []
            else:
                assert probe.observed == 10
                assert set(actor.slots) == {0}
                expected = [('save', 0, 10)] if trigger == 0 else [
                    ('save', trigger, 10 + trigger), ('restore', 0, 10),
                    ('restore', trigger, 10 + trigger), ('clear', trigger)]
                assert actor.events == expected
    print('7 original-branch/routing success controls PASS; queue, CPU snapshots and recomputation are fakes.')


def log_softmax(logits, temperature=1.):
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled -= scaled.max()
    return scaled - np.log(np.exp(scaled).sum())


def counterexamples():
    logits = np.array([.2, .7, -.4])
    assert not np.allclose(log_softmax(logits, 1.), log_softmax(logits, .7))
    # Selected-token equality does not imply a full distribution match.
    a, b = np.array([.2, .3, .5]), np.array([.2, .5, .3])
    assert np.log(a[0]) == np.log(b[0]) and not np.allclose(a, b)
    # Even all vocabulary probabilities do not identify parameter/logit values.
    assert np.allclose(log_softmax(logits), log_softmax(logits + 10.))
    assert not np.allclose(logits, logits + 10.)
    residuals = np.array([.2, -.2])
    assert residuals.mean() == 0 and np.max(np.abs(residuals)) > 0
    segments = [7, 9]
    assert max(segments) - min(segments) + 1 == 3 and len(set(segments)) == 2
    training_step = 10
    assert (training_step - 1) - max(segments) == 0
    assert (training_step - 1) - min(segments) == 2
    # Same administrative label alone permits different tensor/logit contents.
    records = [(9, log_softmax([0., 1.])), (9, log_softmax([0., 2.]))]
    assert records[0][0] == records[1][0] and not np.allclose(records[0][1], records[1][1])
    # A mixed-version response legitimately differs from all-current recompute.
    old, new = log_softmax([0., 1.]), log_softmax([0., 2.])
    stored = np.array([old[0], new[1]])
    all_current = np.array([new[0], new[1]])
    assert not np.allclose(stored, all_current)
    print('7 probability/metadata counterexamples PASS: processing, selected-token coverage, gauge,')
    print('mean cancellation, interval span, administrative labels and mixed-version recomputation.')


class InjectedSnapshotFailure(RuntimeError):
    pass


def snapshot_failure_controls(root):
    separate = root / 'verl/trainer/ppo/v1/trainer_separate_async.py'
    worker = root / 'verl/experimental/separation/engine_workers.py'
    entry = root / 'verl/trainer/main_ppo.py'
    for path in (separate, worker, entry):
        print(path.name, 'sha256=' + hashlib.sha256(path.read_bytes()).hexdigest())

    class Base:
        def _compute_old_log_prob(self, batch, metrics):
            self.actor_rollout_wg.event('compute')
            return batch

    ns = {'Base': Base}
    exec(compile('from __future__ import annotations\nclass Probe(Base):\n' + indent(
        method(separate, 'PPOTrainerSeparateAsync'), '    '), '<original-snapshot-routing>', 'exec'), ns)
    count = 0
    for bypass in (False, True):
        for trigger in (0, 1, 3):
            failures = ['compute'] if bypass else (['save', 'compute'] if trigger == 0 else
                                                   ['save', 'restore_old', 'compute', 'restore_current', 'clear'])
            for failure in failures:
                events, slots = [], {0: 10}
                actor = SimpleNamespace(current=10 + trigger)
                def event(name):
                    events.append(name)
                    if name == failure:
                        raise InjectedSnapshotFailure(name)
                def save(slot):
                    event('save')
                    slots[slot] = actor.current
                def restore(slot):
                    event('restore_old' if slot == 0 else 'restore_current')
                    actor.current = slots[slot]
                def clear(slot):
                    event('clear')
                    del slots[slot]
                actor.event, actor.save_model_to_cpu = event, save
                actor.restore_model_from_cpu, actor.clear_cpu_model = restore, clear
                probe = ns['Probe']()
                probe.actor_rollout_wg = actor
                probe.config = SimpleNamespace(algorithm={'rollout_correction': {'bypass_mode': bypass}})
                probe.local_trigger_step = trigger
                try:
                    probe._compute_old_log_prob(object(), {})
                except InjectedSnapshotFailure as error:
                    assert str(error) == failure and events[-1] == failure
                else:
                    raise AssertionError('injected failure was swallowed')
                expected = ['compute'] if bypass else (['save', 'compute'] if trigger == 0 else
                                                       ['save', 'restore_old', 'compute', 'restore_current', 'clear'])
                assert events == expected[:expected.index(failure) + 1]
                # Fake operations fail BEFORE their own mutation. Native partial effects unknown.
                old_remains = not bypass and trigger > 0 and failure in ('compute', 'restore_current')
                assert actor.current == (10 if old_remains else 10 + trigger)
                if not bypass and trigger > 0:
                    assert (trigger in slots) == (failure != 'save')
                print(f'snapshot bypass={bypass} trigger={trigger} fail={failure}: events={events}, fake_current={actor.current}')
                count += 1

    ns = {}
    for name in ('save_model_to_cpu', 'restore_model_from_cpu', 'clear_cpu_model'):
        exec(compile('from __future__ import annotations\n' + method(worker, 'DetachActorWorker', name),
                     '<original-detach-worker>', 'exec'), ns)
    for strategy in ('fsdp', 'fsdp2'):
        module = SimpleNamespace(value=11)
        def copy(model): return (model.value, 'fake-spec') if strategy == 'fsdp2' else model.value
        def restore(model, saved, *spec):
            assert spec == (('fake-spec',) if strategy == 'fsdp2' else ())
            model.value = saved
        obj = SimpleNamespace(config=SimpleNamespace(actor=SimpleNamespace(strategy=strategy)),
                              actor=SimpleNamespace(engine=SimpleNamespace(module=module)),
                              copy_handler=copy, restore_handler=restore)
        ns['save_model_to_cpu'](obj, 0)
        module.value = 12
        ns['restore_model_from_cpu'](obj, 0)
        assert module.value == 11
        ns['clear_cpu_model'](obj, 0)
        assert obj.cpu_saved_models == {}
        module.value = 13
        ns['restore_model_from_cpu'](obj, 0)  # Existing dictionary, missing key: no-op.
        assert module.value == 13 and obj.cpu_saved_models == {}
        count += 1
    print(f'{count} snapshot/worker controls PASS; worker handlers and snapshots are scalar fakes.')
    return count


def entry_cleanup_controls(root):
    path = root / 'verl/trainer/main_ppo.py'
    source = method(path, 'TaskRunnerV1', 'run')
    count = 0
    for failure in (None, 'fit', 'init', 'fit+finish'):
        events = []
        tq = ModuleType('transfer_queue')
        tq.init = lambda config: events.append('tq.init')
        tq.close = lambda: events.append('tq.close')
        def finish(exit_code):
            events.append(('finish', exit_code))
            if failure == 'fit+finish':
                raise InjectedSnapshotFailure('finish')
        class Trainer:
            def __init__(self, config): self.logger = SimpleNamespace(finish=finish)
            def init(self):
                events.append('trainer.init')
                if failure == 'init': raise InjectedSnapshotFailure('init')
            def fit(self, manager):
                events.append('fit')
                if failure in ('fit', 'fit+finish'): raise InjectedSnapshotFailure('fit')
        modules = {'transfer_queue': tq}
        for name in ('verl', 'verl.trainer', 'verl.trainer.ppo', 'verl.trainer.ppo.v1'):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        modules['verl.trainer.ppo.v1'].get_trainer_cls = lambda mode: Trainer
        previous = {name: sys.modules.get(name) for name in modules}
        namespace = {'configure_verl_logging': lambda: None, 'pprint': lambda *a: None,
                     'OmegaConf': SimpleNamespace(to_container=lambda *a, **kw: {}, resolve=lambda *a: None)}
        exec(compile('from __future__ import annotations\n' + source, '<original-task-runner>', 'exec'), namespace)
        runner = SimpleNamespace(trainer=None, agent_loop_manager=object(),
                                 init_agent_loop_manager=lambda: events.append('agent.init'))
        config = SimpleNamespace(trainer=SimpleNamespace(v1=SimpleNamespace(trainer_mode='separate_async')),
                                 transfer_queue=SimpleNamespace(enable=False))
        sys.modules.update(modules)
        try:
            try:
                namespace['run'](runner, config)
            except InjectedSnapshotFailure as error:
                assert failure is not None
                assert str(error) == ('finish' if failure == 'fit+finish' else failure)
            else:
                assert failure is None
        finally:
            for name, old in previous.items():
                if old is None: sys.modules.pop(name, None)
                else: sys.modules[name] = old
        expected = ['tq.init', 'trainer.init']
        if failure != 'init': expected += ['agent.init', 'fit']
        expected += [('finish', 0 if failure is None else 1), 'tq.close']
        assert events == expected
        count += 1
    print('4 original TaskRunner success/failure/finally controls PASS; tracker/queue/trainer are fakes.')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--check-snapshot-failures', action='store_true')
    args = parser.parse_args()
    routing_controls(args.source_root)
    counterexamples()
    if args.check_snapshot_failures:
        count = snapshot_failure_controls(args.source_root) + entry_cleanup_controls(args.source_root)
        print(f'PASS {count} additional snapshot/failure-boundary controls; no native continuation/recovery proof.')
    print('No native queue/actor inference, real weights, logits processor, vLLM, GPU or causal diagnosis.')


if __name__ == '__main__':
    main()
