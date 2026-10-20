#!/usr/bin/env python3
"""Old-logprob routing controls and probability/metadata counterexamples.

Usage: python3 verify_verl_logprob_provenance.py /path/to/verl-8250
Executes original base bypass branch and separate-async routing method with
fake queue/actor/base recomputation. CPU slots hold identifiers, not tensors.
NumPy examples are not real model logits, vLLM processing or synchronization.
"""
import argparse
import ast
import hashlib
from pathlib import Path
from textwrap import indent
from types import SimpleNamespace

import numpy as np


def method(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_old_log_prob')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    args = parser.parse_args()
    routing_controls(args.source_root)
    counterexamples()
    print('No native queue/actor inference, real weights, logits processor, vLLM, GPU or causal diagnosis.')


if __name__ == '__main__':
    main()
