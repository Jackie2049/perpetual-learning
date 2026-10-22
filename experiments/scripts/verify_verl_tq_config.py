#!/usr/bin/env python3
"""Check #8340's original Hydra tests without importing the trainer/Ray.

Python 3.12+, pytest, hydra-core, OmegaConf. Source and config come from fixed
local Git objects, with no lazy fetch. Only the unchanged default-setting helper
is AST-executed and exposed as main_ppo; the actual module/package is not run.
Configs are composed by real Hydra; no scheduler, actor or storage substitutes.
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

BASE = "dc709afb7a74fd2645640a4e5f2fdf40166b4fbb"
HEAD = "881fb96937f5ddfae5d894bb9a48538d1b049e1e"
MAIN = "verl/trainer/main_ppo.py"
TEST = "tests/trainer/test_tq_storage_units_on_cpu.py"
CONFIG = "verl/trainer/config"
GENERATED = ["_generated_ppo_" + name + "trainer" for name in
             ("", "megatron_", "torchtitan_", "veomni_")]


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ray", "vllm", "sglang", "transfer_queue"}:
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
    import hydra
    import omegaconf
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig
    import pytest

    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(args.source_repo.resolve()), *arguments], env=env)

    source = git("show", HEAD + ":" + MAIN)
    tree = ast.parse(source, MAIN)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "set_default_tq_storage_units")
    module = types.ModuleType("verl.trainer.main_ppo")
    module.DictConfig = DictConfig
    exec(compile(ast.Module(body=[helper], type_ignores=[]), MAIN, "exec"), module.__dict__)
    sys.modules[module.__name__] = module
    print("main_ppo SHA256", hashlib.sha256(source).hexdigest(), flush=True)
    print("original helper AST SHA256", hashlib.sha256(
        ast.dump(helper, include_attributes=False).encode()).hexdigest(), flush=True)

    class Reports:
        passed = 0
        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed += 1

    with tempfile.TemporaryDirectory(prefix="perpetual-tq-config-") as directory:
        root = Path(directory)
        for revision in (BASE, HEAD):
            destination = root / revision
            destination.mkdir()
            paths = [CONFIG] + ([TEST] if revision == HEAD else [])
            subprocess.run(["tar", "-x", "-C", str(destination)],
                           input=git("archive", revision, *paths), check=True)
        test_path = root / HEAD / TEST
        data = test_path.read_bytes()
        compile(data, TEST, "exec")
        print("original tests SHA256", hashlib.sha256(data).hexdigest(), flush=True)
        reports = Reports()
        code = pytest.main(["-q", "--noconftest", "-p", "no:cacheprovider", str(test_path)],
                           plugins=[reports])
        assert code == 0 and reports.passed == 3, (code, reports.passed)
        for name in GENERATED:
            observed = []
            for revision in (BASE, HEAD):
                with initialize_config_dir(config_dir=str(root / revision / CONFIG), version_base=None):
                    config = compose(config_name=name, overrides=["trainer.nnodes=1"])
                storage = config.transfer_queue.backend.SimpleStorage
                before = storage.num_data_storage_units
                if revision == HEAD:
                    module.set_default_tq_storage_units(config)
                observed.append((before, storage.num_data_storage_units))
            assert observed == [(8, 8), (None, 2)], (name, observed)
            print(name, "base/head before-after", observed, flush=True)
    print("Python", sys.version.split()[0], "Hydra", hydra.__version__,
          "OmegaConf", omegaconf.__version__, "pytest", pytest.__version__, flush=True)
    assert not any(name in sys.modules for name in ("torch", "ray", "vllm", "transfer_queue"))
    print("3 original cases + 4 paired generated-config checks passed; no Ray/torch/storage imported")


if __name__ == "__main__":
    main()
