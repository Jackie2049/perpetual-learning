"""Verify #8217's original CPU tests with real macOS spawn/ProcessPoolExecutor.

Usage: Python3.12+ verify_verl_prime_pool.py /path/to/verl
Requires psutil7.2.2/pytest8.4.2. No torch/Ray/tokenizer imports or device discovery.
Loads three complete original scoring functions and original get_event_loop via
AST, plus the complete original test module. Package roots/class/manager are not
executed. Direct test calls, not native pytest collection or trainer validation.
"""
import argparse
import ast
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import importlib.abc
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import psutil

BASE = "2035f58dcb06d8cad4caee107b30f4045d724367"
HEAD = "fe9ff514507e6f7a68cf927b2213ad8d8d2501ff"
PRIME = "verl/workers/reward_manager/prime.py"
TEST = "tests/workers/reward_manager/test_prime_on_cpu.py"
NAMES = {"single_compute_score", "parallel_compute_score_async", "run_reward_scoring"}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo", type=Path)
parser.add_argument("--rev", choices=(BASE, HEAD))
args = parser.parse_args()
sys.dont_write_bytecode = True


def source(rev, path):
    return subprocess.check_output(
        ["git", "show", f"{rev}:{path}"], cwd=args.repo,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
    )


class Originals(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"verl", "verl.workers", "verl.workers.reward_manager"}:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        if fullname in {"verl.workers.reward_manager.prime", "original_prime_tests"}:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == "original_prime_tests":
            exec(compile(source(HEAD, TEST), HEAD + ":" + TEST, "exec"), module.__dict__)
        elif module.__name__ == "verl.workers.reward_manager.prime":
            module.__dict__.update(asyncio=asyncio, ProcessPoolExecutor=ProcessPoolExecutor,
                                   partial=partial, psutil=psutil)
            ray = ast.parse(source(args.rev, "verl/utils/ray_utils.py"))
            helper = [node for node in ray.body if isinstance(node, ast.FunctionDef)
                      and node.name == "get_event_loop"]
            tree = ast.parse(source(args.rev, PRIME))
            functions = [node for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name in NAMES]
            assert len(helper) == 1 and {node.name for node in functions} == NAMES
            exec(compile(ast.Module(helper + functions, type_ignores=[]),
                         args.rev + ":" + PRIME, "exec"), module.__dict__)


sys.meta_path.insert(0, Originals())


def mixed_score(task, completion, reference, extra_info):
    """Benign process-picklable controls; not a model/dataset scorer."""
    if completion == "fail":
        raise RuntimeError("expected worker exception")
    return {"float": 0.5, "dict": {"score": 1.0, "acc": True},
            "missing": {"acc": True}, "bad": {"score": "not-a-number"},
            "nan": {"score": float("nan")}}[completion]


def main():
    if args.rev is None:
        for rev in (BASE, HEAD):
            subprocess.run([sys.executable, "-B", __file__, str(args.repo), "--rev", rev], check=True)
        return
    assert multiprocessing.get_start_method() == "spawn", "This probe targets native macOS spawn"
    from verl.workers.reward_manager.prime import run_reward_scoring
    import original_prime_tests as tests

    for path in (PRIME, "verl/utils/ray_utils.py"):
        raw = source(args.rev, path)
        compile(raw, args.rev + ":" + path, "exec")
        print(args.rev[:8], path, "sha256", hashlib.sha256(raw).hexdigest(), flush=True)
    compile(source(HEAD, TEST), HEAD + ":" + TEST, "exec")
    passed, failed = [], []
    for name in sorted(dir(tests)):
        if name.startswith("test_"):
            try:
                getattr(tests, name)()
            except KeyError as exc:
                assert args.rev == BASE and name == "test_dict_result_uses_score_key" and exc.args == (0,)
                failed.append(name)
            else:
                passed.append(name)
    assert (len(passed), len(failed)) == ((3, 1) if args.rev == BASE else (4, 0))

    def score(rows):
        return run_reward_scoring(mixed_score, rows, ["42"] * len(rows),
                                  ["control"] * len(rows), num_processes=1)

    assert score(["float", "fail", "float"]) == [0.5, 0.0, 0.5]
    controls = ["worker failure preserves successful rows"]
    if args.rev == HEAD:
        assert score(["float", "fail", "dict"]) == [0.5, 0.0, 1.0]
        controls.append("mixed dict/float/failure order")
        for kind, error in (("missing", KeyError), ("bad", ValueError)):
            try:
                score(["float", kind])
            except error:
                controls.append(kind + " escapes result normalization")
            else:
                raise AssertionError(kind)
        assert math.isnan(score(["nan"])[0])
        controls.append("nonfinite score not rejected")
    print(json.dumps({"rev": args.rev, "start_method": "spawn", "original_pass": passed,
                      "original_expected_fail": failed, "controls": controls}), flush=True)


if __name__ == "__main__":
    main()
