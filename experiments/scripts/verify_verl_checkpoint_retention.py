#!/usr/bin/env python3
"""Run five original verl cleanup tests, not model save/restore or crash recovery.

Usage: python verify_verl_checkpoint_retention.py /path/to/verl
Requires Python 3.10+, CPU torch, numpy and pytest. Reads fixed Git objects without
fetch/checkouts. Deletes only directories under its newly allocated scratch root.
"""

import __future__
import argparse
import ast
import hashlib
import linecache
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pytest
import torch
import torch.distributed

REVISION = "1876b06d0a3e4e71e06230be10af14492ca8a75b"


def load(root, path, name, *, select_class=False):
    data = subprocess.check_output(
        ["git", "show", f"{REVISION}:{path}"], cwd=root,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
    )
    print(path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    filename = str(root / ".git-object-tests" / REVISION / path)
    source = data.decode()
    linecache.cache[filename] = (len(data), None, source.splitlines(True), filename)
    module = ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    flags = 0
    if select_class:
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == "BaseCheckpointManager"]
        assert len(nodes) == 1
        source = ast.Module(body=nodes, type_ignores=[])
        flags = __future__.annotations.compiler_flag
        # Definitions are unchanged; only framework imports/annotation evaluation
        # are bypassed. RNG/device methods are defined but never called.
        module.__dict__.update(os=os, random=random, shutil=shutil, np=np, torch=torch)
    exec(compile(source, filename, "exec", flags=flags, dont_inherit=True), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    root = args.source_repo.resolve()
    assert "verl" not in sys.modules
    for name in ("verl", "verl.utils", "verl.utils.checkpoint"):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)
    implementation = load(root, "verl/utils/checkpoint/checkpoint_manager.py",
                          "verl.utils.checkpoint.checkpoint_manager", select_class=True)
    sys.modules["verl.utils.checkpoint"].checkpoint_manager = implementation
    tests = load(root, "tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py", "_original_cleanup_tests")

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.calls = 0
            self.failures = []
            self.skips = []

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1
            if report.failed:
                self.failures.append((report.nodeid, report.when, str(report.longrepr)))
            if report.skipped:
                self.skips.append(report.nodeid)

    print("revision", REVISION, "python", sys.version.split()[0], "torch", torch.__version__,
          "numpy", np.__version__, "pytest", pytest.__version__, flush=True)
    old_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="verl-checkpoint-retention-") as scratch:
        allowed = Path(scratch).resolve()
        print("scratch", allowed, "(only synthetic checkpoint.txt fixtures)", flush=True)

        def guard(event, arguments):
            if event == "shutil.rmtree":
                target, dir_fd = arguments
                assert dir_fd is None, "relative-fd deletion is outside this experiment"
                resolved = Path(target).resolve()
                assert resolved == allowed or allowed in resolved.parents, resolved

        sys.addaudithook(guard)
        tempfile.tempdir = scratch
        try:
            adapter = Adapter()
            code = pytest.main(
                ["-q", "--noconftest", "-c", os.devnull,
                 "--rootdir", str(Path(__file__).resolve().parent),
                 "--confcutdir", str(Path(__file__).resolve().parent), "-p", "no:cacheprovider",
                 "--basetemp", str(allowed / "pytest"), str(Path(__file__).resolve())],
                plugins=[adapter],
            )
            assert code == 0 and adapter.calls == 5, (code, adapter.calls)
            assert not adapter.failures and not adapter.skips, (adapter.failures, adapter.skips)
        finally:
            tempfile.tempdir = old_tempdir
    assert not allowed.exists(), "scratch cleanup incomplete"
    assert "verl.utils.device" not in sys.modules
    print("5 original tests passed; scratch removed; no user checkpoint touched")
    print("NOT RUN: package/conftest, RNG/device methods, process group, FSDP/Megatron save/restore, crash recovery")


if __name__ == "__main__":
    main()
