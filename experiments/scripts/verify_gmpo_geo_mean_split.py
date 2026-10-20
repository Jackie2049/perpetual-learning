#!/usr/bin/env python3
"""Pinned FSDP host-control probes + independent GMPO accumulation models.

Usage: python3 verify_gmpo_geo_mean_split.py /path/to/verl-7913
Executes only extracted original train_batch/forward_backward_batch methods
with fake data/forward/backward/optimizer/communication. NumPy shared-parameter
finite differences separately model the PG loss. NOT torch/autograd, real DP,
native FSDP, model training, clipping/SGD reproduction, or patch verification.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from verify_gmpo_clip_semantics import fd_gradient, verl_model


def function_at(path, name, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    return fn


def source_checks(root):
    paths = {key: root / path for key, path in {
        'base': 'verl/workers/engine/base.py',
        'fsdp': 'verl/workers/engine/fsdp/transformer_impl.py',
        'loss': 'verl/workers/utils/losses.py',
        'core': 'verl/trainer/ppo/core_algos.py',
        'trainer': 'verl/trainer/ppo/v1/trainer_base.py',
        'worker': 'verl/workers/engine_workers.py',
    }.items()}
    for key, path in paths.items():
        print(key, hashlib.sha256(path.read_bytes()).hexdigest())
    fsdp = function_at(paths['fsdp'], 'forward_step', 'FSDPEngineWithLMHead')
    assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == 'loss_function'
               and ast.dump(n.targets[0]) == ast.dump(ast.parse('loss, metrics = x').body[0].targets[0])
               for n in ast.walk(fsdp))
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
               and [e.id for e in n.value.elts if isinstance(e, ast.Name)] == ['loss', 'output']
               for n in ast.walk(fsdp))
    ppo = ast.unparse(function_at(paths['loss'], 'ppo_loss'))
    assert 'config.global_batch_info[\'global_batch_size\']' in ppo
    core = ast.unparse(function_at(paths['core'], 'compute_policy_loss_geo_mean'))
    assert 'torch.mean(pg_losses)' in core and 'rollout_is_weights + 1e-10' in core
    trainer = paths['trainer'].read_text()
    assert 'ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n' in trainer
    assert '"global_batch_size": ppo_mini_batch_size' in trainer
    worker = paths['worker'].read_text()
    assert 'mini_batch_size // self.engine.get_data_parallel_size()' in worker
    return paths


def extract(fn, namespace):
    fn.decorator_list = []
    exec(compile('from __future__ import annotations\n' + ast.unparse(fn), '<upstream-host-method>', 'exec'), namespace)
    return namespace[fn.name]


def host_probe(paths, sizes):
    trace = []

    class Count:
        def sum(self): return self
        def to(self, device): return self
        def item(self): return 128

    class Loss:
        def __init__(self, tag): self.tag = tag
        def backward(self): trace.append(('backward', self.tag))

    namespace = dict(
        nullcontext=nullcontext,
        get_device_id=lambda: 'fake-cpu',
        maybe_fix_3d_position_ids=lambda data: None,
        tu=SimpleNamespace(assign_non_tensor=lambda data, **kw: data.update(kw),
                           get_non_tensor_data=lambda data, key, default=None: data.get(key, default)),
        torch=SimpleNamespace(distributed=SimpleNamespace(
            all_reduce=lambda *a, **kw: None, ReduceOp=SimpleNamespace(SUM='fake-sum')),
            profiler=SimpleNamespace(record_function=lambda name: nullcontext())),
        prepare_micro_batches=lambda **kw: ([SimpleNamespace(tag=i, size=n) for i, n in enumerate(sizes)], None),
        postprocess_batch_func=lambda **kw: {'metrics': {}, 'outputs': kw['output_lst']},
    )
    loop = extract(function_at(paths['fsdp'], 'forward_backward_batch', 'FSDPEngine'), namespace)
    train = extract(function_at(paths['base'], 'train_batch', 'BaseEngine'), namespace)

    class Engine:
        ulysses_sequence_parallel_size = 1
        scaler = None
        def get_data_parallel_size(self): return 1
        def get_data_parallel_group(self): return None
        def _gradient_sync_context(self, **kw): return nullcontext()
        def forward_step(self, micro_batch, **kw):
            return Loss(micro_batch.tag), {'model_output': object()}
        def forward_backward_batch(self, *args, **kw): return loop(self, *args, **kw)
        def optimizer_zero_grad(self): trace.append(('zero', None))
        def optimizer_step(self): trace.append(('step', None)); return 0.
        def is_mp_src_rank_with_outputs(self): return False

    data = {'loss_mask': Count(), 'global_batch_size': sum(sizes)}
    result = train(Engine(), data, loss_function=None)
    assert trace == [('zero', None)] + [('backward', i) for i in range(len(sizes))] + [('step', None)]
    assert all('model_output' not in item for item in result['outputs'])
    assert data['dp_size'] == 1 and data['global_batch_size'] == sum(sizes)
    return trace


def row_losses(theta, features, initial_z, mask, advantages, weights):
    z = initial_z + np.einsum('blk,k->bl', features, theta)
    losses = verl_model(z, mask, advantages, .4, .4, 1e-8)
    if weights is not None:
        losses *= np.exp((np.log(weights + 1e-10) * mask).sum(-1) / (mask.sum(-1) + 1e-8))
    return losses


def accumulate(losses, rank_sizes, global_size, normalized):
    if normalized and (global_size is None or global_size <= 0):
        raise ValueError('reference contract requires positive global_size')
    dp = len(rank_sizes)
    offset, rank_values = 0, []
    for sizes in rank_sizes:
        value = 0.
        for size in sizes:
            part = losses[offset:offset + size]
            assert len(part) == size
            value += part.sum() * dp / global_size if normalized else part.mean()
            offset += size
        rank_values.append(value)
    assert offset == len(losses)
    return np.mean(rank_values)  # explicit simulated DP gradient/value average


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root', type=Path)
    args = parser.parse_args()
    paths = source_checks(args.source_root)
    for sizes in ([8], [1] * 8, [3, 5]):
        host_probe(paths, sizes)
    print('Original host train/loop: 3 fake-control cases PASS; no arithmetic or communication execution.')

    rng = np.random.default_rng(7913)
    features = rng.normal(size=(8, 5, 4)) * .1
    initial_z = rng.normal(size=(8, 5)) * .2
    mask = np.arange(5)[None, :] < np.array([5, 3, 4, 2, 1, 5, 0, 4])[:, None]
    advantages = np.broadcast_to(np.array([1., -1., .5, -.5, 2., -2., .25, -.25])[:, None], mask.shape)
    theta = np.array([.03, -.02, .01, .04])
    z = initial_z + np.einsum('blk,k->bl', features, theta)
    assert np.min(np.abs(np.abs(z) - .4)) > 1e-4
    plans = [[[8]], [[1] * 8], [[3, 5]], [[2, 2, 4]], [[1, 3], [3, 1]], [[1, 1]] * 4]
    for weighted in (False, True):
        weights = rng.uniform(.8, 1.2, size=mask.shape) if weighted else None
        rows = lambda t: row_losses(t, features, initial_z, mask, advantages, weights)
        full = lambda t: rows(t).sum() / 8
        expected_gradient = fd_gradient(full, theta)
        for plan in plans:
            normalized = lambda t: accumulate(rows(t), plan, 8, True)
            local_mean = lambda t: accumulate(rows(t), plan, 8, False)
            assert np.allclose(normalized(theta), full(theta), atol=1e-12, rtol=1e-12)
            assert np.allclose(fd_gradient(normalized, theta), expected_gradient, atol=2e-9, rtol=2e-8)
            relative = np.linalg.norm(fd_gradient(local_mean, theta) - expected_gradient) / np.linalg.norm(expected_gradient)
            if plan == [[8]]: assert relative < 1e-8
            else: assert relative > .01
            print(f'IS={weighted} rank/micro sizes={plan} local-mean relative gradient delta={relative:.6f}')
        # A fixed G retains fully masked rows in its denominator; excluding one changes objective.
        assert np.isclose(rows(theta)[6], 0.)
        assert not np.isclose(rows(theta).sum() / 7, full(theta))
    for invalid in (None, 0):
        try: accumulate(np.ones(8), [[3, 5]], invalid, True)
        except ValueError: pass
        else: raise AssertionError('invalid reference denominator accepted')
    print('12 shared-parameter value/finite-difference cases PASS, including explicit simulated DP average.')
    print('No native FSDP/DP, autograd, real optimizer, KL/clip update, GPU, or upstream patch verification.')


if __name__ == '__main__':
    main()
