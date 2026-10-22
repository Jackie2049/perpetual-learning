#!/usr/bin/env python3
"""Run #8327's original pure-Python sentinel tests from fixed Git objects.

Python >=3.10 and pytest only. Loads complete original module/test unchanged;
preloads its exact qualified module name without importing verl package roots.
A blocking import finder rejects framework/device packages, rather than stubbing
them. No model/quantization/kernel/driver or training test is performed.
Run: python -B experiments/scripts/verify_verl_quant_sentinel.py /path/to/verl-git
"""

import argparse
import importlib.abc
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
sys.dont_write_bytecode = True
from verify_deepinfer_sampling_params import load
import pytest

REV = "a2012b598e2e5eabc9a79847ed0a245dc73bc478"
FORBIDDEN = {"verl", "torch", "ray", "vllm", "sglang", "compressed_tensors", "transformer_engine"}


class BlockRootImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in FORBIDDEN:
            raise ImportError(f"out-of-scope package import blocked: {fullname}")
        return None


def controls(native):
    S = native.QuantizedRolloutSentinel
    clip_key = "response_length/clip_ratio"
    s = S("mxfp8", mode="raise")
    s.check({"rollout_corr/kl": 1.0}, 1)
    with pytest.raises(RuntimeError, match="rollout_corr/kl"):
        s.check({"rollout_corr/kl": 1.0001}, 2)
    print("KL threshold: equal passes, strictly greater raises")

    s = S("mxfp8", mode="raise", clip_steps=2)
    s.check({clip_key: 1.0}, 1)
    s.check({}, 2)
    with pytest.raises(RuntimeError, match="length cap"):
        s.check({clip_key: 1.0}, 3)
    print("missing clip metric preserves the observed streak; not a healthy/reset observation")

    s = S("mxfp8", mode="raise", clip_steps=2)
    s.check({clip_key: 1.0}, 7)
    with pytest.raises(RuntimeError, match="length cap"):
        s.check({clip_key: 1.0}, 7)
    print("step label is not a unique-call key: repeated step=7 counted twice")

    cfg = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout={"quantization": "mxfp8"}))
    trainer = SimpleNamespace(config=cfg)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VERL_QUANT_SENTINEL", "raise")
        mp.setenv("VERL_QUANT_SENTINEL_KL_MAX", "invalid")
        native.check_quantized_rollout_metrics(trainer, {"rollout_corr/kl": 7.2}, 1)
        cached = trainer._quant_sentinel
        assert not cached.enabled
        mp.setenv("VERL_QUANT_SENTINEL_KL_MAX", "1.0")
        native.check_quantized_rollout_metrics(trainer, {"rollout_corr/kl": 7.2}, 2)
        assert trainer._quant_sentinel is cached and not cached.enabled
    print("configuration parse failure caches a disabled sentinel even after env is corrected")
    S("mxfp8", mode="raise").check({}, 99)
    print("empty metric mapping produces no finding, not a health verdict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    for key in ("VERL_QUANT_SENTINEL", "VERL_QUANT_SENTINEL_KL_MAX", "VERL_QUANT_SENTINEL_CLIP_STEPS"):
        os.environ.pop(key, None)
    assert all(name not in sys.modules for name in FORBIDDEN)
    sys.meta_path.insert(0, BlockRootImports())
    root = args.source_repo.resolve()
    native = load(root, REV, "verl/utils/quant_sentinel.py", "verl.utils.quant_sentinel")
    tests = load(root, REV, "tests/utils/test_quant_sentinel_on_cpu.py", "_original_quant_sentinel_tests")

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
    assert code == 0 and adapter.calls == 20, (code, adapter.calls)
    controls(native)
    assert all(name not in sys.modules for name in FORBIDDEN)
    assert all(not name.startswith("torch.") for name in sys.modules)
    print("Python", sys.version.split()[0], "pytest", pytest.__version__)
    print("20 original parameterized cases + 5 additional controls passed; no package shells or engine stubs")
    print("NOT RUN: full pytest/conftest, trainer integration, refit loaders, audit, kernels, model or GPU/NPU")


if __name__ == "__main__":
    main()
