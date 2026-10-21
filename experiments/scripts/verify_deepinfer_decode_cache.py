#!/usr/bin/env python3
"""Run original decode-cache PR tests with native CPU/Gloo and real temporary I/O.

Python >=3.12, CPU torch/Gloo, pytest. Original modules import via pinned Git
blobs; only decode_perf_cache.py is overlaid from the selected related PR.
No checkout/install/model/profiling/GPU/remote-server work. The accelerator env
tag only prevents package initialization from probing hardware, not proof of GPU
support. Original tests/conftest.py is loaded explicitly as a plugin.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import linecache
import os
from pathlib import Path
import socket
import subprocess
import sys

from verify_deepinfer_sampling_params import BASE, load
import pytest

REVISIONS = {
    "identity": "4622836fa04acf074845feb27448fabf50a85401",
    "overflow": "4e679e904f5ffee7b410eecb5340f50c0e8a97b6",
}


class GitSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root, revision):
        self.root, self.revision = root, revision
        self.paths = set(subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", BASE, "deepinfer"], cwd=root
        ).decode().splitlines())
        self.sources = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "deepinfer" and not fullname.startswith("deepinfer."):
            return None
        stem = fullname.replace(".", "/")
        for candidate, package in ((stem + "/__init__.py", True), (stem + ".py", False)):
            if candidate in self.paths:
                self.sources[fullname] = candidate
                return importlib.util.spec_from_loader(fullname, self, is_package=package)
        raise ModuleNotFoundError(fullname)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = self.sources[module.__name__]
        revision = self.revision if path == "deepinfer/worker/decode_perf_cache.py" else BASE
        data = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=self.root,
                                       env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        if path == "deepinfer/worker/decode_perf_cache.py":
            print(revision, path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
        module.__file__ = str(self.root / ".git-object-tests" / revision / path)
        linecache.cache[module.__file__] = (len(data), None, data.decode().splitlines(True), module.__file__)
        exec(compile(data, module.__file__, "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--suite", choices=REVISIONS, required=True)
    parser.add_argument("--implementation", choices=("base", "patched"), required=True)
    args = parser.parse_args()
    root = args.source_repo.resolve()
    revision = BASE if args.implementation == "base" else REVISIONS[args.suite]
    assert "deepinfer" not in sys.modules
    os.environ["DEEPINFER_ACCELERATOR"] = "nvidia"  # platform tag only; no local hardware discovery
    interfaces = [name for _, name in socket.if_nameindex() if name in ("lo", "lo0")]
    assert interfaces, "no known loopback interface available for CPU/Gloo"
    os.environ["GLOO_SOCKET_IFNAME"] = interfaces[0]
    finder = GitSources(root, revision)
    sys.meta_path.insert(0, finder)
    tests = load(root, REVISIONS[args.suite], f"tests/worker/test_decode_perf_cache_{args.suite}.py",
                 "_original_decode_cache_tests")
    fixtures = load(root, BASE, "tests/conftest.py", "_original_shared_fixtures")
    import torch
    print("python", sys.version.split()[0], "torch", torch.__version__, "pytest", pytest.__version__,
          "cache implementation", revision, "loopback", interfaces[0], flush=True)

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.calls = 0
            self.failed = {}

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1
                if report.failed:
                    self.failed[report.nodeid.split("::", 1)[1]] = str(report.longrepr)
            elif report.failed:
                raise AssertionError("unexpected setup/teardown failure: " + report.nodeid)

    adapter = Adapter()
    code = pytest.main(["-q", "--tb=short", "--noconftest", "-p", "no:cacheprovider", "-o",
                        "markers=ci: CPU test\nci_npu: shared backend test", str(Path(__file__).resolve())],
                       plugins=[adapter, fixtures])
    total = 9 if args.suite == "identity" else 8
    assert adapter.calls == total, adapter.calls
    expected = 0 if args.implementation == "patched" else (4 if args.suite == "identity" else 8)
    assert len(adapter.failed) == expected, adapter.failed.keys()
    if args.suite == "overflow" and expected:
        assert all("OverflowError" in error for error in adapter.failed.values())
    if args.suite == "identity" and expected:
        assert sum("engram_parallel_size" in name for name in adapter.failed) == 1
        assert sum("engine_type" in name for name in adapter.failed) == 1
        assert sum("test_batch_invariant_mode" in name for name in adapter.failed) == 2
    assert code == (1 if expected else 0), code
    print(args.suite, args.implementation, total - expected, "passed;", expected,
          "expected baseline failures; original pytest exit=", int(code))
    print("NOT RUN: multi-rank Gloo, profile_decode_perf/model, latency/throughput or GPU/NPU tests")


if __name__ == "__main__":
    main()
