#!/usr/bin/env python3
"""Run original #127 or #60 discovery tests without accelerator runtimes.

Python >=3.10 and pytest. Native modules and real temporary files/symlinks;
the original tests supply synthetic _root/_environ, not accelerator hardware.
Two extra lookup controls are distinct from the 12 author tests. No binding,
libnuma/NVML loading, npu-smi/lscpu, torch, runtime or server execution.
Importing ascend computes its original local device-manager existence flag.
Default: #127 tests against unchanged base checkout. --suite vendor: #60's 10
tests against complete original #60 Git modules; original NVIDIA capability
and machine fixtures are test substitutes, not NVML/hardware measurements.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import linecache
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from verify_deepinfer_sampling_params import BASE, load
import pytest

TESTS = "f9dec70d38d81838133e22b3208f43bfc9e40030"
VENDOR = "1f299c270bc21ffafb51d14ac1c160a4a350333c"


class VendorSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root):
        self.root = root
        self.paths = set(subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", VENDOR, "deep_cluster"], cwd=root).decode().splitlines())
        self.sources = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "deep_cluster" and not fullname.startswith("deep_cluster."):
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
        data = subprocess.check_output(["git", "show", f"{VENDOR}:{path}"], cwd=self.root,
                                       env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        module.__file__ = str(self.root / ".git-object-tests" / VENDOR / path)
        linecache.cache[module.__file__] = (len(data), None, data.decode().splitlines(True), module.__file__)
        print(VENDOR, path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
        exec(compile(data, module.__file__, "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--suite", choices=("ascend", "vendor"), default="ascend")
    args = parser.parse_args()
    root = args.source_repo.resolve()
    if args.suite == "vendor":
        assert "deep_cluster" not in sys.modules
        sys.meta_path.insert(0, VendorSources(root))
        tests = load(root, VENDOR, "tests/deep_cluster/test_hardware.py", "_original_vendor_tests")
    else:
        subprocess.run(["git", "diff", "--exit-code", BASE, "--", "deep_cluster"], cwd=root, check=True)
        assert not subprocess.check_output(["git", "status", "--porcelain", "--", "deep_cluster"], cwd=root)
        sys.path.insert(0, str(root))
        for path in ("hardware/ascend.py", "hardware/device.py", "topology/bind.py"):
            data = (root / "deep_cluster" / path).read_bytes()
            print(BASE, path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
        tests = load(root, TESTS, "tests/hardware/test_ascend_discovery.py", "_original_ascend_tests")

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        calls = 0

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1

    adapter = Adapter()
    code = pytest.main(["-q", "--tb=short", "--noconftest", "-p", "no:cacheprovider", "-o",
                        "markers=ci: CPU test\nci_npu: shared backend test", str(Path(__file__).resolve())],
                       plugins=[adapter])
    expected = 10 if args.suite == "vendor" else 12
    assert code == 0 and adapter.calls == expected, (code, adapter.calls)
    print("python", sys.version.split()[0], "pytest", pytest.__version__)
    assert "torch" not in sys.modules and "torch_npu" not in sys.modules
    if args.suite == "vendor":
        print("10 original vendor tests passed; NOT RUN: platform/build/runtime, NVML, binding, kernels or throughput")
        return

    from deep_cluster.topology import which_numa
    device = tests.device_module
    with tempfile.TemporaryDirectory(prefix="ascend-lookup-") as directory, pytest.MonkeyPatch.context() as patch:
        synthetic = Path(directory)
        tests._add_ubcore_device(synthetic, "udma0", numa="3")
        tests._add_ubcore_device(synthetic, "udma1", numa="3")
        patch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,0")
        assert device.physical_device_id(0, _root=synthetic) == "1"
        assert (which_numa(0, _root=synthetic), which_numa(1, _root=synthetic)) == (3, 2)
        assert device.device_numa_node(0, _root=synthetic) == 2  # physical index, no visible remap
        patch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
        assert device.visible_device_count(_root=synthetic) == 0
        assert device.physical_device_count(_root=synthetic) == 2
        with pytest.raises(AssertionError, match="no such GPU"):
            which_numa(0, _root=synthetic)
    print("12 original tests + 2 separate visible-ordinal/empty-mask lookup controls passed")
    assert "torch" not in sys.modules and "torch_npu" not in sys.modules
    print("NOT RUN: real topology/device discovery, NUMA/CPU/memory binding, kernels or throughput")


if __name__ == "__main__":
    main()
