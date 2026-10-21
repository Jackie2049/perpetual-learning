#!/usr/bin/env python3
"""Run pinned PR #121 SamplingParams or CacheConfig tests against related fixes.

Python >=3.10 and pytest; sampling mode also uses CPU torch. Reads fixed Git objects, never checks out or
patches upstream. Loads the original standalone sampling module and test module;
only the package root is a namespace shell, avoiding hardware/platform imports.
Uses a pytest collector adapter, no upstream conftest or third-party plugins.
Cache mode AST-selects only original CacheConfig and TestCacheConfig, preserving
the test decorators; unused model/engine imports are omitted, not stubbed.
No Parallel/Scheduler/ModelConfig, bindings, engine, sampler, model or GPU run.
"""
import argparse
import ast
import hashlib
import linecache
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest

BASE = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"
PATCH = "bc668873e5520aa309c709a9b4bfc2935519779a"
CACHE_PATCH = "c78146bfb1cef7ec94326a51b84b1d968aef7d3f"
TESTS = "2afdb3d2d8ed1fb3a83a0169485aa25d281196e8"


def load(root, revision, path, name, select=None, namespace=None):
    data = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(revision, path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    # Virtual diagnostic filename only: no source files are written into the checkout.
    filename = str(root / ".git-object-tests" / revision / path)
    source = data.decode()
    linecache.cache[filename] = (len(data), None, source.splitlines(True), filename)
    module = ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    if namespace:
        module.__dict__.update(namespace)
    if select:
        original = ast.parse(source)
        selected = [n for n in original.body if isinstance(n, ast.ClassDef) and n.name in select]
        assert {n.name for n in selected} == set(select)
        future = [n for n in original.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
        source = ast.Module(body=[*future, *selected], type_ignores=[])
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def boundaries(cls):
    """Observe Python-constructor boundaries; not a downstream validity claim."""
    cases = [
        (dict(temperature=0.0, top_k=0), ValueError),
        (dict(temperature=0.0, top_p=0.0), ValueError),
        (dict(top_p=float("nan")), ValueError),
        (dict(temperature=float("nan")), None),
        (dict(temperature=float("inf")), None),
        (dict(top_k=1.5), None),
        (dict(max_tokens=1.5), None),
        (dict(top_logprobs=0.5), None),
    ]
    for kwargs, error in cases:
        if error:
            with pytest.raises(error):
                cls(seq_seed=7, **kwargs)
        else:
            cls(seq_seed=7, **kwargs)
    print("8 constructor boundary observations passed; acceptance is NOT engine validity", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--suite", choices=("sampling", "cache"), default="sampling")
    parser.add_argument("--implementation", choices=("base", "pr119", "pr118"), required=True)
    parser.add_argument("--runxfail", action="store_true", help="Cache PR #118 only: ignore xfail markers via pytest")
    args = parser.parse_args()
    allowed = ("base", "pr119") if args.suite == "sampling" else ("base", "pr118")
    if args.implementation not in allowed or (args.runxfail and (args.suite, args.implementation) != ("cache", "pr118")):
        parser.error("implementation/suite/runxfail combination not supported")
    root = args.source_repo.resolve()
    assert "deepinfer" not in sys.modules
    package = ModuleType("deepinfer")
    package.__path__ = []
    sys.modules["deepinfer"] = package
    if args.suite == "sampling":
        revision = BASE if args.implementation == "base" else PATCH
        implementation = load(root, revision, "deepinfer/sampling_params.py", "deepinfer.sampling_params")
        package.sampling_params = implementation
        tests = load(root, TESTS, "tests/test_sampling_params.py", "_original_sampling_tests")
        boundaries(implementation.SamplingParams)
    else:
        revision = BASE if args.implementation == "base" else CACHE_PATCH
        implementation = load(root, revision, "deepinfer/config.py", "_original_cache_config", ["CacheConfig"])
        tests = load(root, TESTS, "tests/test_config.py", "_original_cache_tests", ["TestCacheConfig"],
                     dict(pytest=pytest, CacheConfig=implementation.CacheConfig))
        assert "torch" not in sys.modules, "cache isolation unexpectedly imported torch"
    print("python", sys.version.split()[0], "pytest", pytest.__version__, "device=cpu", flush=True)

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.failed = set()
            self.xfailed = set()
            self.xfail_missing_guard = set()
            self.strict_xpass = set()
            self.calls = 0

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1
                if report.failed:
                    self.failed.add(report.nodeid.split("::", 1)[1])
                    if "[XPASS(strict)]" in str(report.longrepr):
                        self.strict_xpass.add(report.nodeid.split("::", 1)[1])
                elif report.skipped and hasattr(report, "wasxfail"):
                    self.xfailed.add(report.nodeid.split("::", 1)[1])
                    if "DID NOT RAISE" in str(report.longrepr):
                        self.xfail_missing_guard.add(report.nodeid.split("::", 1)[1])
            elif report.failed:
                raise AssertionError("unexpected setup/teardown failure: " + report.nodeid)

    adapter = Adapter()
    options = ["-q", "--noconftest", "-p", "no:cacheprovider", str(Path(__file__).resolve())]
    if args.runxfail:
        options.append("--runxfail")
    code = pytest.main(options,
                       plugins=[adapter])
    expected, xfailed, strict_xpass = set(), set(), set()
    if args.suite == "sampling":
        total = 53
        if args.implementation == "base":
            expected = {"TestMaxTokens::test_zero_raises",
                        "TestLogprobsDeprecation::test_logprobs_true_emits_deprecation_warning"}
    else:
        total = 9
        marked = {"TestCacheConfig::test_zero_utilization_raises", "TestCacheConfig::test_negative_utilization_raises"}
        if args.implementation == "base":
            xfailed = marked
        elif not args.runxfail:
            expected = strict_xpass = marked
    assert adapter.calls == total, (adapter.calls, "unexpected test selection")
    assert adapter.failed == expected, (adapter.failed, expected)
    assert adapter.xfailed == xfailed, (adapter.xfailed, xfailed)
    assert adapter.xfail_missing_guard == xfailed, "unexpected reason for a baseline XFAIL"
    assert adapter.strict_xpass == strict_xpass, (adapter.strict_xpass, strict_xpass)
    assert code == (1 if expected else 0), code
    print(args.suite, args.implementation, "pytest exit=", int(code), "checks=", total,
          "passed=", total - len(expected) - len(xfailed), "expected failures=", len(expected - strict_xpass),
          "XFAIL=", len(xfailed), "strict XPASS=", len(strict_xpass))
    print("NOT RUN: upstream conftest/full config module, package initialization, bindings, engine or sampler/GPU")


if __name__ == "__main__":
    main()
