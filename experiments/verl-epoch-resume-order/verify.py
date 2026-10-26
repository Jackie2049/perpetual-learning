"""Original #7974 CPU tests, isolated imports, real torchdata/order/file IO.

head: original 15 SFT + 4 V1 parameter cases; old-v1: original 4 V1 cases.
Not a framework import, trainer construction, model restore or GPU test.
"""

import __future__
import argparse
import ast
import hashlib
import json
import linecache
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from enum import Enum
from types import ModuleType, SimpleNamespace
import uuid

HEAD = "cfad96962d5aa9640cce052774a34b810382b20f"
OLD = "4105a9914658ae2dc411d80cb530681ce759546e"
sys.dont_write_bytecode = True


def module(name):
    obj = ModuleType(name)
    obj.__path__ = []
    sys.modules[name] = obj
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--revision", choices=("head", "old-v1"), default="head")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.deps.resolve()))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import numpy as np
    import pytest
    import torch
    import torchdata
    from tensordict import NonTensorData, NonTensorStack, TensorDict

    torch.set_num_threads(1)
    hashes = {}

    def source(revision, path):
        raw = subprocess.check_output(
            ["git", "show", revision + ":" + path], cwd=args.repo,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
        ast.parse(raw)
        compile(raw, path, "exec")
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        return raw

    def definitions(raw, names, namespace):
        nodes = [n for n in ast.parse(raw).body if getattr(n, "name", None) in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    assert "verl" not in sys.modules
    for name in ("verl", "verl.utils", "verl.utils.checkpoint", "verl.trainer",
                 "verl.trainer.ppo", "verl.trainer.ppo.v1"):
        module(name)
    namespace = {"logging": logging}
    definitions(source(HEAD, "verl/utils/logger/aggregate_logger.py"), {"log_with_rank"}, namespace)
    handler = module("verl.utils.checkpoint.checkpoint_handler")
    handler.__file__ = "isolated_original_checkpoint_handler.py"
    handler.__dict__.update(torch=torch, os=os, json=json, re=re, logging=logging,
                            Enum=Enum, log_with_rank=namespace["log_with_rank"])
    tree = ast.parse(source(HEAD, "verl/utils/checkpoint/checkpoint_handler.py"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "original_checkpoint_handler", "exec",
                 flags=__future__.annotations.compiler_flag), handler.__dict__)

    utils = module("verl.trainer.ppo.utils")
    definitions(source(HEAD, "verl/trainer/ppo/utils.py"), {"create_rl_sampler"}, utils.__dict__)
    td = {"torch": torch, "np": np, "TensorDict": TensorDict,
          "NonTensorData": NonTensorData, "NonTensorStack": NonTensorStack}
    definitions(source(HEAD, "verl/utils/tensordict_utils.py"), {"get_tensordict"}, td)
    v1 = module("verl.trainer.ppo.v1.trainer_base")
    v1.__dict__.update(torch=torch, np=np, uuid=uuid, os=os, logger=logging.getLogger("v1-control"),
                       tu=SimpleNamespace(get_tensordict=td["get_tensordict"]))
    revision = HEAD if args.revision == "head" else OLD
    v1_raw = source(revision, "verl/trainer/ppo/v1/trainer_base.py")
    original = next(n for n in ast.parse(v1_raw).body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer")
    methods = [n for n in original.body if getattr(n, "name", None) in {"_load_checkpoint", "_fetch_one_gen_batch"}]
    assert len(methods) == 2
    # Keep both complete production methods; only their surrounding class is a shell.
    shell = ast.ClassDef(name="PPOTrainer", bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])),
                 "original_v1_methods", "exec", flags=__future__.annotations.compiler_flag), v1.__dict__)

    tests = ModuleType("_original_epoch_tests")
    tests.__file__ = str(Path(__file__).resolve())
    paths = ["tests/trainer/ppo/v1/test_dataloader_resume_on_cpu.py"]
    if args.revision == "head":
        paths.append("tests/utils/ckpt/test_checkpoint_handler_epoch_boundary_on_cpu.py")
    for path in paths:
        # Entire unmodified original test module, including fixtures/decorators.
        raw = source(HEAD, path)
        linecache.cache[path] = (len(raw), None, raw.decode().splitlines(True), path)
        exec(compile(raw, path, "exec"), tests.__dict__)

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.passed, self.failures, self.order_failures = [], [], []

        def pytest_runtest_makereport(self, item, call):
            if call.when == "call" and call.excinfo:
                values = call.excinfo.traceback[-1].frame.f_locals
                if "actual" in values and "expected" in values:
                    actual, expected = values["actual"], values["expected"]
                    first = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
                    self.order_failures.append({
                        "nodeid": item.nodeid, "first_mismatch_batch_index": first,
                        "actual": actual[first], "expected": expected[first],
                        "first_epoch_batch": expected[0],
                    })

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed.append(report.nodeid)
            if report.failed:
                self.failures.append({"nodeid": report.nodeid, "phase": report.when,
                                      "traceback": str(report.longrepr)})

    adapter = Adapter()
    with tempfile.TemporaryDirectory(prefix="verl-7974-control-") as scratch:
        exit_code = int(pytest.main([str(Path(__file__).resolve()), "-q", "--noconftest", "-c", os.devnull,
                                    "--rootdir", str(Path(__file__).resolve().parent),
                                    "--confcutdir", str(Path(__file__).resolve().parent),
                                    "--basetemp", str(Path(scratch).resolve() / "pytest"),
                                    "-p", "no:cacheprovider"], plugins=[adapter]))
    result = {"scope": __doc__, "v1_revision": revision,
              "sft_revision": HEAD if args.revision == "head" else None,
              "original_tests_revision": HEAD, "python": platform.python_version(),
              "torch": torch.__version__, "torchdata": torchdata.__version__,
              "pytest": pytest.__version__, "source_hashes": hashes,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pytest_exit": exit_code, "passed": adapter.passed, "failures": adapter.failures}
    result["order_failures"] = adapter.order_failures
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
    # The old candidate is expected to fail order assertions only at the two boundaries.
    if args.revision == "old-v1":
        assert len(adapter.passed) == 2 and len(adapter.failures) == 2 and exit_code == 1
        assert all(f["phase"] == "call" and "assert actual == expected" in f["traceback"]
                   for f in adapter.failures)
    else:
        assert len(adapter.passed) == 19 and not adapter.failures and exit_code == 0


if __name__ == "__main__":
    main()
