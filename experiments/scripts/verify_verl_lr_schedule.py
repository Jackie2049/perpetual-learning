#!/usr/bin/env python3
"""Pinned native LR constructors with real CPU SGD/LambdaLR.

Usage: python verify_verl_lr_schedule.py /path/to/verl
Python >=3.10 and CPU torch. Four traces, two in-memory restore comparisons,
and one explicit LR override observation; not original upstream tests or training.
"""
import __future__
import argparse
import ast
import copy
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"


def load(root):
    path = "verl/utils/torch_functional.py"
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    names = {"get_cosine_schedule_with_warmup", "get_constant_schedule_with_warmup"}
    nodes = [node for node in ast.parse(data).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ModuleType("_original_lr_constructors")
    module.__dict__.update(math=math, LambdaLR=LambdaLR)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{REV}:{path}", "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), module.__dict__)
    return module


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def pair(constructor, *, momentum=0.0, **kwargs):
    parameter = torch.tensor(10.0, dtype=torch.float64, device="cpu", requires_grad=True)
    optimizer = SGD([parameter], lr=1.0, momentum=momentum)
    scheduler = constructor(optimizer=optimizer, **kwargs)
    return parameter, optimizer, scheduler


def advance(state):
    parameter, optimizer, scheduler = state
    used_lr = optimizer.param_groups[0]["lr"]
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    next_lr = optimizer.param_groups[0]["lr"]
    close(next_lr, scheduler.get_last_lr()[0])
    return used_lr, next_lr, parameter.item()


def trace(label, state, expected):
    parameter, optimizer, scheduler = state
    actual = [optimizer.param_groups[0]["lr"]]
    assert scheduler.last_epoch == 0  # constructor has performed the initial scheduling call
    for _ in expected[1:]:
        used, next_lr, _ = advance(state)
        close(used, actual[-1])
        actual.append(next_lr)
    for observed, wanted in zip(actual, expected, strict=True):
        close(observed, wanted)
    close(parameter.item(), 10 - sum(expected[:-1]))
    print(label, "constructor + post-update LRs =", actual)


def restore_comparisons(native):
    constructor = native.get_cosine_schedule_with_warmup
    kwargs = dict(num_warmup_steps=2, num_training_steps=10, min_lr_ratio=0.2)
    reference = pair(constructor, momentum=0.9, **kwargs)
    for _ in range(4):
        advance(reference)
    parameter, optimizer, scheduler = reference
    snapshot = (parameter.detach().clone(), copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()))
    assert snapshot[2]["lr_lambdas"] == [None]
    assert "num_training_steps" not in snapshot[2]
    continuation = [advance(reference) for _ in range(5)]

    for total_steps in (10, 18):
        resumed = pair(constructor, momentum=0.9, **dict(kwargs, num_training_steps=total_steps))
        new_parameter, new_optimizer, new_scheduler = resumed
        with torch.no_grad():
            new_parameter.copy_(snapshot[0])
        # Build scheduler BEFORE restoring optimizer; restore both states, not just last_epoch.
        new_optimizer.load_state_dict(copy.deepcopy(snapshot[1]))
        new_scheduler.load_state_dict(copy.deepcopy(snapshot[2]))
        assert new_scheduler.last_epoch == 4
        close(new_optimizer.param_groups[0]["lr"], snapshot[1]["param_groups"][0]["lr"])
        close(new_scheduler.get_last_lr()[0], snapshot[2]["_last_lr"][0])
        rows = [advance(resumed) for _ in range(5)]
        if total_steps == 10:
            for actual, expected in zip(rows, continuation, strict=True):
                for a, b in zip(actual, expected, strict=True):
                    close(a, b)
            torch.testing.assert_close(new_optimizer.state[new_parameter]["momentum_buffer"],
                                       optimizer.state[parameter]["momentum_buffer"])
            print("same-config restore: all 5 used/next LRs, parameters and final momentum match uninterrupted run")
        else:
            close(rows[0][0], continuation[0][0])
            close(rows[0][2], continuation[0][2])
            assert not math.isclose(rows[0][1], continuation[0][1])
            assert not math.isclose(rows[1][2], continuation[1][2])
            print("changed T=18 restore: state echoes same LR, first update matches, then next LR diverges:",
                  continuation[0][1], "vs", rows[0][1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    native = load(parser.parse_args().source_repo.resolve())
    cosine = native.get_cosine_schedule_with_warmup
    constant = native.get_constant_schedule_with_warmup
    a, b = 0.6 + 0.2 * math.sqrt(2), 0.6 - 0.2 * math.sqrt(2)
    expected = [0, 0.5, 1, a, 0.6, b, 0.2, b, 0.6, a, 1]
    trace("constant W=2", pair(constant, num_warmup_steps=2), [0, 0.5] + [1] * 9)
    trace("cosine W=2 T=6 floor=.2 cycles=.5",
          pair(cosine, num_warmup_steps=2, num_training_steps=6, min_lr_ratio=0.2), expected)
    trace("cosine one-indexed",
          pair(cosine, num_warmup_steps=2, num_training_steps=6, min_lr_ratio=0.2, zero_indexed_step=False),
          expected[1:7])
    trace("cosine cycles=1",
          pair(cosine, num_warmup_steps=2, num_training_steps=6, min_lr_ratio=0.2, num_cycles=1),
          [0, 0.5, 1, 0.6, 0.2, 0.6, 1])
    restore_comparisons(native)
    state = pair(constant, num_warmup_steps=2)
    state[1].param_groups[0]["lr"] = 0.4
    used, next_lr, value = advance(state)
    close(used, 0.4)
    close(next_lr, 0.5)
    close(value, 9.6)
    assert state[2].base_lrs == [1.0]
    print("explicit override: used .4, next scheduler LR .5, base_lrs stays [1.0]")
    assert "verl" not in sys.modules and "ray" not in sys.modules
    print("4 traces + 2 restore comparisons + 1 override passed; Python", sys.version.split()[0], "torch", torch.__version__)
    print("NOT RUN: original upstream tests/configs, package initialization, worker/FSDP, model, disk checkpoint, devices")


if __name__ == "__main__":
    main()
