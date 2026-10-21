#!/usr/bin/env python3
"""Run original #127 discovery tests against an unchanged base deep_cluster.

Python >=3.10 and pytest. Native modules and real temporary files/symlinks;
the original tests supply synthetic _root/_environ, not accelerator hardware.
Two extra lookup controls are distinct from the 12 author tests. No binding,
libnuma/NVML loading, npu-smi/lscpu, torch, runtime or server execution.
Importing ascend computes its original local device-manager existence flag.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from verify_deepinfer_sampling_params import BASE, load
import pytest

TESTS = "f9dec70d38d81838133e22b3208f43bfc9e40030"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    root = args.source_repo.resolve()
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
    assert code == 0 and adapter.calls == 12, (code, adapter.calls)

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
    print("python", sys.version.split()[0], "pytest", pytest.__version__)
    assert "torch" not in sys.modules and "torch_npu" not in sys.modules
    print("NOT RUN: real topology/device discovery, NUMA/CPU/memory binding, kernels or throughput")


if __name__ == "__main__":
    main()
