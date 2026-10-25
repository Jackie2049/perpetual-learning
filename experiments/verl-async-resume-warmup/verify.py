#!/usr/bin/env python3
"""Run the nine pinned original warmup test bodies in an isolated namespace.

Only complete production methods are extracted. Original test assertions,
OmegaConf and MagicMock fixtures are unchanged; only verl imports are replaced.
No trainer initialization, TransferQueue runtime, Ray, model or device discovery.
The load test uses the author's mocked queue/actor loader and a real temp path;
it is NOT a checkpoint round-trip or durable-buffer verification.
"""

import argparse
import ast
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType

import omegaconf
import pytest
import torch

sys.dont_write_bytecode = True
REV = "3c6d9b43b67c74ec343d77874cd8fa6e841523da"
BASE = "verl/trainer/ppo/v1/trainer_base.py"
COLOCATE = "verl/trainer/ppo/v1/trainer_colocate_async.py"
SEPARATE = "verl/trainer/ppo/v1/trainer_separate_async.py"
TEST = "tests/trainer/ppo/v1/test_v1_async_warmup_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl", type=Path, help="read-only Git checkout with pinned object")
    args = parser.parse_args()
    hashes = {}

    def tree(path):
        raw = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=args.verl,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[path] = hashlib.sha256(raw).hexdigest()
        parsed = ast.parse(raw, filename=path)
        compile(parsed, path, "exec")
        return parsed

    logger = logging.getLogger("isolated-warmup-test")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    base = ModuleType("isolated_trainer_base")
    base.__dict__.update(os=os, math=math, torch=torch, logger=logger,
                         tq=ModuleType("queue_not_initialized"))
    source = tree(BASE)
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef)
               and node.name == "PPOTrainer")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_load_checkpoint", "_add_async_warmup_batches")]
    assert len(methods) == 2
    count = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_count_tq_prompt_groups")
    isolated = ast.Module(body=[count, ast.ClassDef(name="PPOTrainer", bases=[], keywords=[],
                                                  body=methods, decorator_list=[])],
                          type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), BASE, "exec"), base.__dict__)

    def entry(path, class_name):
        module = tree(path)
        original = next(node for node in module.body if isinstance(node, ast.ClassDef)
                        and node.name == class_name)
        method = next(node for node in original.body if isinstance(node, ast.FunctionDef)
                      and node.name == "on_train_begin")
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     path, "exec"), ns)
        return type(class_name, (), {"on_train_begin": ns["on_train_begin"]})

    test_ns = {
        "trainer_base": base,
        "PPOTrainer": base.PPOTrainer,
        "PPOTrainerColocateAsync": entry(COLOCATE, "PPOTrainerColocateAsync"),
        "PPOTrainerSeparateAsync": entry(SEPARATE, "PPOTrainerSeparateAsync"),
    }
    tests = tree(TEST)
    tests.body = [node for node in tests.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("verl")
    )]
    exec(compile(tests, TEST, "exec"), test_ns)
    passed = []
    for name, func in test_ns.items():
        if not name.startswith("test_"):
            continue
        with tempfile.TemporaryDirectory(prefix="warmup-original-test-") as temp:
            with pytest.MonkeyPatch.context() as monkeypatch:
                fixtures = {"monkeypatch": monkeypatch, "tmp_path": Path(temp)}
                func(**{key: fixtures[key] for key in inspect.signature(func).parameters})
        passed.append(name)
    assert len(passed) == 9, passed
    print(json.dumps({
        "revision": REV, "python": sys.version.split()[0],
        "torch": torch.__version__, "omegaconf": omegaconf.__version__,
        "pytest": pytest.__version__, "passed": passed, "source_sha256": hashes,
        "scope": "original nine test bodies; extracted methods; mock loaders; no TQ/Ray/model/checkpoint round-trip"
    }, indent=2))


if __name__ == "__main__":
    main()
