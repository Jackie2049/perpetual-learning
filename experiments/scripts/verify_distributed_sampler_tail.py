#!/usr/bin/env python3
"""Observe DistributedSampler/loader tail policies with real CPU classes.

Python >=3.10 and torch. Enumerates rank indices in one process with explicit
num_replicas/rank; no process group, DDP, workers, device or training test.
Run: python experiments/scripts/verify_distributed_sampler_tail.py
"""
from collections import Counter
from fractions import Fraction
import hashlib
import inspect
import sys

sys.dont_write_bytecode = True
import torch
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler


def load_indices(n, replicas, batch_size, sampler_drop, loader_drop):
    data = list(range(n))
    sampled, batches = [], []
    for rank in range(replicas):
        sampler = DistributedSampler(data, num_replicas=replicas, rank=rank,
                                     shuffle=False, drop_last=sampler_drop)
        sampled.append(list(sampler))
        loader = DataLoader(data, sampler=sampler, batch_size=batch_size,
                            drop_last=loader_drop, num_workers=0, pin_memory=False,
                            generator=torch.Generator(device="cpu").manual_seed(1234))
        rank_batches = [batch.tolist() for batch in loader]
        assert len(loader) == len(rank_batches)
        batches.append(rank_batches)
    emitted = [item for rank_batches in batches for batch in rank_batches for item in batch]
    return sampled, batches, emitted


def main():
    assert not torch.distributed.is_initialized()
    print("Python", sys.version.split()[0], "torch", torch.__version__, "torch git", torch.version.git_version)
    for cls in (DistributedSampler, BatchSampler):
        print(cls.__name__, "class-source-sha256=" + hashlib.sha256(inspect.getsource(cls).encode()).hexdigest())

    padded = [[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 1]]
    truncated = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    for sampler_drop in (False, True):
        for loader_drop in (False, True):
            sampled, batches, emitted = load_indices(10, 3, 2, sampler_drop, loader_drop)
            assert sampled == (truncated if sampler_drop else padded)
            if sampler_drop and loader_drop:
                assert batches == [[[0, 3]], [[1, 4]], [[2, 5]]]
                assert set(emitted) == set(range(6)) and len(emitted) == 6
            elif sampler_drop:
                assert batches == [[[0, 3], [6]], [[1, 4], [7]], [[2, 5], [8]]]
                assert set(emitted) == set(range(9)) and len(emitted) == 9
            else:
                assert len(emitted) == 12 and Counter(emitted) == Counter(list(range(10)) + [0, 1])
            print("N10/R3/B2", "sampler_drop", sampler_drop, "loader_drop", loader_drop,
                  "sampler", sampled, "batches", batches,
                  "count", len(emitted), "unique", len(set(emitted)),
                  "missing", sorted(set(range(10)) - set(emitted)))

    # Synthetic binary correctness: only original item 0 is correct.
    _, _, emitted = load_indices(10, 3, 2, False, False)
    true_mean = Fraction(1, 10)
    sampled_mean = Fraction(emitted.count(0), len(emitted))
    assert sampled_mean == Fraction(1, 6) != true_mean
    print("synthetic evaluation: original mean", true_mean, "padded-observation mean", sampled_mean)

    sampled, batches, emitted = load_indices(2, 4, 1, False, False)
    assert sampled == [[0], [1], [0], [1]] and Counter(emitted) == Counter({0: 2, 1: 2})
    print("N2/R4 padding", sampled)
    sampled, batches, emitted = load_indices(2, 4, 1, True, False)
    assert sampled == [[], [], [], []] and batches == [[], [], [], []] and emitted == []
    print("N2/R4 sampler drop -> all empty")
    _, batches, emitted = load_indices(2, 4, 2, False, True)
    assert batches == [[], [], [], []] and emitted == []
    print("N2/R4/B2 loader drop -> all empty even with padded sampler")

    data = list(range(12))
    samplers = [DistributedSampler(data, num_replicas=3, rank=r, shuffle=True, seed=17) for r in range(3)]
    first = [list(s) for s in samplers]
    assert [list(s) for s in samplers] == first
    for sampler in samplers:
        sampler.set_epoch(1)
    second = [list(s) for s in samplers]
    assert second != first
    for groups in (first, second):
        assert Counter(x for group in groups for x in group) == Counter(data)
    for sampler in samplers:
        sampler.set_epoch(0)
    assert [list(s) for s in samplers] == first
    print("same seed/epoch repeats; epoch1 changes ordering; epoch0 restores; both epochs exactly cover N12")
    assert not torch.distributed.is_initialized()
    print("All tail/coverage/mean/epoch observations passed; real CPU sampler/loader, not original upstream tests")
    print("NOT RUN: multi-process ranks/workers, collective, DDP, model, framework validation or GPU/NPU")


if __name__ == "__main__":
    main()
