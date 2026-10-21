#!/usr/bin/env python3
"""Pinned V1 metric reduction: 19 original tests and 5 CPU observations.

Usage: python verify_verl_metrics_aggregator.py /path/to/verl
Requires Python 3.10+, CPU torch, numpy, pytest. Reads fixed Git objects without
fetch/checkouts; no framework initialization, trainer, logger or devices run.
"""
import __future__
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import linecache
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pytest
import torch

REV = "1876b06d0a3e4e71e06230be10af14492ca8a75b"


def load(root, path, name, names=None, namespace=None):
    data = subprocess.check_output(["git", "show", f"{REV}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    filename = str(root / ".git-object-tests" / REV / path)
    source = data.decode()
    linecache.cache[filename] = (len(data), None, source.splitlines(True), filename)
    flags = 0
    if names:
        selected = []
        found = set()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                selected.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in names:
                    selected.append(node)
                    found.add(target.id)
        assert found == names, (found, names)
        source = ast.Module(body=selected, type_ignores=[])
        flags = __future__.annotations.compiler_flag
    module = ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    module.__dict__.update(namespace or {})
    exec(compile(source, filename, "exec", flags=flags, dont_inherit=True), module.__dict__)
    return module


def observations(cls, timing):
    # Missing/zero weights are not token counts and do not infer a denominator.
    agg = cls()
    agg.add_step_metrics({"custom/loss/mean": 1.0})
    agg.add_step_metrics({"custom/loss/mean": 3.0})
    assert agg.get_aggregated_metrics()["custom/loss/mean"] == 2.0
    agg = cls()
    agg.add_step_metrics({"custom/loss/mean": 1.0}, sample_count=0)
    agg.add_step_metrics({"custom/loss/mean": 3.0}, sample_count=3)
    assert agg.get_aggregated_metrics()["custom/loss/mean"] == 3.0
    print("zero weights: all zero -> arithmetic mean 2; mixed 0/3 -> 3")

    # Structural batch fixtures only, not real DataProto/TransferQueue batches.
    def batch(lengths):
        response = torch.tensor(lengths, dtype=torch.float64, device="cpu")
        return SimpleNamespace(batch={"prompt_length": torch.zeros_like(response), "response_length": response})

    first = timing.compute_timing_metrics(batch([100]), {"gen": 0.100})
    second = timing.compute_timing_metrics(batch([1, 1, 1]), {"gen": 0.009})
    agg = cls()
    for metrics, count in [(first, 1), (second, 3)]:
        metrics["custom/rate/mean"] = metrics["timing_per_token_ms/gen"]
        agg.add_step_metrics(metrics, sample_count=count)
    out = agg.get_aggregated_metrics()
    pooled = timing.compute_timing_metrics(batch([100, 1, 1, 1]), {"gen": 0.109})
    assert math.isclose(out["timing_s/gen"], 0.109)
    assert math.isclose(out["timing_per_token_ms/gen"], 4.0)
    assert math.isclose(out["custom/rate/mean"], 2.5)
    assert math.isclose(pooled["timing_per_token_ms/gen"], 109 / 103)
    print("synthetic gen rates: local 1,3 ms/token; helper sum=4; sample mean=2.5; pooled=", 109 / 103)

    agg = cls()
    agg.add_step_metrics({"custom/python_bool": True, "custom/tensor_bool": torch.tensor(True, device="cpu")})
    out = agg.get_aggregated_metrics()
    assert "custom/python_bool" not in out and out["custom/tensor_bool"] == 1.0
    print("Python bool ignored; single-element bool tensor becomes scalar 1.0")

    agg = cls()
    agg.add_step_metrics({"custom/max_entropy/mean": 1.0}, sample_count=1)
    agg.add_step_metrics({"custom/max_entropy/mean": 3.0}, sample_count=3)
    assert agg.get_aggregated_metrics()["custom/max_entropy/mean"] == 3.0
    print("synthetic custom/max_entropy/mean uses max before mean, not sample-weighted 2.5")
    print("5 observations passed; synthetic naming/rate cases are NOT trainer defects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    root = parser.parse_args().source_repo.resolve()
    assert "verl" not in sys.modules
    for name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1"):
        package = ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)
    constant = load(root, "verl/trainer/ppo/v1/replay_buffer.py", "verl.trainer.ppo.v1.replay_buffer",
                    {"DAPO_FILTERED_REWARD_COUNTS_KEY"})
    namespace = dict(Counter=Counter, defaultdict=defaultdict, Any=Any, np=np, torch=torch,
                     DAPO_FILTERED_REWARD_COUNTS_KEY=constant.DAPO_FILTERED_REWARD_COUNTS_KEY)
    native = load(root, "verl/trainer/ppo/v1/utils.py", "verl.trainer.ppo.v1.utils",
                  {"MetricsAggregator"}, namespace)
    tests = load(root, "tests/trainer/ppo/v1/test_metrics_aggregator_on_cpu.py", "_original_metric_tests")
    timing = load(root, "verl/trainer/ppo/metric_utils.py", "_native_timing",
                  {"_compute_response_info", "compute_timing_metrics"}, dict(torch=torch))

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.calls, self.failures, self.skips = 0, [], []

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

    adapter = Adapter()
    folder = str(Path(__file__).resolve().parent)
    code = pytest.main(["-q", "--noconftest", "-c", os.devnull, "--rootdir", folder,
                        "--confcutdir", folder, "-p", "no:cacheprovider", str(Path(__file__).resolve())],
                       plugins=[adapter])
    assert code == 0 and adapter.calls == 19, (code, adapter.calls)
    assert not adapter.failures and not adapter.skips, (adapter.failures, adapter.skips)
    observations(native.MetricsAggregator, timing)
    assert "ray" not in sys.modules and "verl.utils.device" not in sys.modules
    print("revision", REV, "Python", sys.version.split()[0], "torch", torch.__version__,
          "numpy", np.__version__, "pytest", pytest.__version__)
    print("NOT RUN: package/conftest, replay buffer, DataProto, trainer/logger, process group, model/devices")


if __name__ == "__main__":
    main()
