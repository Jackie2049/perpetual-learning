#!/usr/bin/env python3
"""Run #8132's original audit CPU unit tests from fixed Git objects.

Python >=3.10, CPU torch and pytest. Original audit and compressed_tensors
modules run in full; only unused verl/utils/qat package initializers are
namespace shells. Test bodies/fixtures are unchanged and explicitly fake
Megatron/TE/vLLM modules. This is not native engine/quantization/GPU validation.
No installation, source checkout edit, model, distributed RPC or server use.
"""
import argparse
from pathlib import Path
import sys
from types import ModuleType

from verify_deepinfer_sampling_params import load
import pytest

REVISION = "0defd9a231cb876ce56812611b0afb3b2b25831a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    root = args.source_repo.resolve()
    assert "verl" not in sys.modules
    for name in ("verl", "verl.utils", "verl.utils.qat"):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
    for path, name in (
        ("verl/utils/qat/compressed_tensors.py", "verl.utils.qat.compressed_tensors"),
        ("verl/utils/quant_layer_audit.py", "verl.utils.quant_layer_audit"),
    ):
        module = load(root, REVISION, path, name)
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    tests = load(root, REVISION, "tests/utils/test_quant_layer_audit_on_cpu.py", "_original_quant_audit_tests")
    expected = sum(name.startswith("test_") and callable(value) for name, value in vars(tests).items())

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
    code = pytest.main(["-q", "--tb=short", "--noconftest", "-p", "no:cacheprovider",
                        str(Path(__file__).resolve())], plugins=[adapter])
    assert code == 0 and adapter.calls == expected, (code, adapter.calls, expected)
    import torch
    print("python", sys.version.split()[0], "torch", torch.__version__, "pytest", pytest.__version__)
    print(expected, "original CPU unit tests passed; Megatron/TE/vLLM fixtures are substitutes")
    assert all(name not in sys.modules for name in ("compressed_tensors", "transformer_engine", "vllm", "sglang"))
    print("NOT RUN: compressors/FP8 GEMMs, real TE workspaces/offload, engines/RPC/loaders, training or GPU/NPU")


if __name__ == "__main__":
    main()
