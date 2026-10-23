#!/usr/bin/env python3
"""Run #8348's 19 original config cases with real OmegaConf, no workers.

Python 3.12+, pytest, OmegaConf. Fixed local Git objects only; optional --deps
points to an existing dependency target. Four complete original helper functions
are AST-executed instead of importing rollout_corr_helper's torch/Ray stack.
The original test module is unchanged; its five tensor/loss cases are deselected.
This is not native verl import, Hydra composition or worker/engine validation.
"""

import argparse
import ast
import hashlib
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Optional

BASE = "a4b9354968705a441fb7da7367924b15b902e771"
HEAD = "4b3b9f84078f83fe5f485e7d8bccf8c1ddbba349"
HELPER = "verl/trainer/ppo/rollout_corr_helper.py"
TEST = "tests/trainer/ppo/test_bypass_mode_config_on_cpu.py"
FUNCTIONS = {
    "_parse_rollout_is_threshold", "_parse_rollout_rs_thresholds",
    "_parsed_setting", "configure_bypass_mode",
}


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ray", "vllm", "sglang", "megatron"}:
            raise ImportError("Out-of-scope framework import: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--deps", type=Path)
    args = parser.parse_args()
    if args.deps:
        sys.path.insert(0, str(args.deps.resolve()))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, BlockFrameworks())
    import omegaconf
    import pytest

    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(args.source_repo.resolve()), *arguments], env=env
        )

    baseline = git("show", BASE + ":" + HELPER)
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "configure_bypass_mode"
                   for n in ast.parse(baseline).body)
    source = git("show", HEAD + ":" + HELPER)
    tree = ast.parse(source, HELPER)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in nodes} == FUNCTIONS
    module = types.ModuleType("verl.trainer.ppo.rollout_corr_helper")
    module.Optional = Optional
    exec(compile(ast.Module(body=nodes, type_ignores=[]), HELPER, "exec"), module.__dict__)
    sys.modules[module.__name__] = module
    original_tests = git("show", HEAD + ":" + TEST)
    for label, data in (("base helper", baseline), ("head helper", source), ("original tests", original_tests)):
        print(label, "SHA256", hashlib.sha256(data).hexdigest(), flush=True)
    print("Python", sys.version.split()[0], "OmegaConf", omegaconf.__version__,
          "pytest", pytest.__version__, flush=True)
    print("Base has no configure_bypass_mode; no baseline test pass/fail claim.", flush=True)

    class Reports:
        passed = 0
        failed = 0
        deselected = 0

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed += 1
            if report.failed:
                self.failed += 1

        def pytest_deselected(self, items):
            self.deselected += len(items)

    report = Reports()
    with tempfile.TemporaryDirectory(prefix="perpetual-bypass-config-") as directory:
        test_path = Path(directory) / Path(TEST).name
        test_path.write_bytes(original_tests)
        code = pytest.main([str(test_path), "-q", "-p", "no:cacheprovider",
                            "-k", "not padding_only"], plugins=[report])
    assert code == 0 and (report.passed, report.failed, report.deselected) == (19, 0, 5)
    print("19 original config cases passed; 5 tensor/loss cases not run. No workers or training.")


if __name__ == "__main__":
    main()
