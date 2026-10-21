#!/usr/bin/env python3
"""Run PR #121's original SamplingParams tests against base or PR #119.

Python >=3.10, CPU torch and pytest. Reads fixed Git objects, never checks out or
patches upstream. Loads the original standalone sampling module and test module;
only the package root is a namespace shell, avoiding hardware/platform imports.
Uses a pytest collector adapter, no upstream conftest or third-party plugins.
No config tests, bindings, engine, sampler, model or GPU are run.
"""
import argparse
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
TESTS = "2afdb3d2d8ed1fb3a83a0169485aa25d281196e8"


def load(root, revision, path, name):
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
    parser.add_argument("--implementation", choices=("base", "pr119"), required=True)
    args = parser.parse_args()
    root = args.source_repo.resolve()
    assert "deepinfer" not in sys.modules
    package = ModuleType("deepinfer")
    package.__path__ = []
    sys.modules["deepinfer"] = package
    revision = BASE if args.implementation == "base" else PATCH
    implementation = load(root, revision, "deepinfer/sampling_params.py", "deepinfer.sampling_params")
    package.sampling_params = implementation
    tests = load(root, TESTS, "tests/test_sampling_params.py", "_original_sampling_tests")
    print("python", sys.version.split()[0], "torch", implementation.torch.__version__,
          "pytest", pytest.__version__, "device=cpu", flush=True)
    boundaries(implementation.SamplingParams)

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.failed = set()
            self.calls = 0

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1
                if report.failed:
                    self.failed.add(report.nodeid.split("::", 1)[1])
            elif report.failed:
                raise AssertionError("unexpected setup/teardown failure: " + report.nodeid)

    adapter = Adapter()
    code = pytest.main(["-q", "--noconftest", "-p", "no:cacheprovider", str(Path(__file__).resolve())],
                       plugins=[adapter])
    expected = ({"TestMaxTokens::test_zero_raises",
                 "TestLogprobsDeprecation::test_logprobs_true_emits_deprecation_warning"}
                if args.implementation == "base" else set())
    assert adapter.calls == 53, (adapter.calls, "unexpected test selection")
    assert adapter.failed == expected, (adapter.failed, expected)
    assert code == (1 if expected else 0), code
    print(args.implementation, "53 original tests:", 53 - len(expected), "passed,",
          len(expected), "expected baseline failures")
    print("NOT RUN: upstream conftest/config tests, package initialization, bindings, engine or sampler/GPU")


if __name__ == "__main__":
    main()
