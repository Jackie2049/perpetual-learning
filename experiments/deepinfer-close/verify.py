#!/usr/bin/env python3
"""Pinned #175 control-flow experiment, NOT full engine shutdown validation.

Python >=3.12 and pytest. Read existing Git objects without checkout/lazy fetch.
AST-select original LLMEngine.close/__del__/_run_workers; omit all other class
members and heavyweight imports. Reuse original test bodies, decorators and Mock
fixtures, replacing only their engine-module import with the selected methods.
Exclude the real subprocess cleanup test. Never initialize an engine, resource,
thread, model or device. Baseline warnings are retained, not silenced.
"""
import argparse
import ast
from contextlib import suppress
import hashlib
import linecache
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest

BASE = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
PATCH = "1dc73e9e8bb8d6099a50b163b9fc3a8351c955f2"


def source(repo, revision, path):
    data = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=repo,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    print(revision, path, "sha256=" + hashlib.sha256(data).hexdigest(), flush=True)
    filename = str(repo / ".git-object-close" / revision / path)
    text = data.decode()
    linecache.cache[filename] = (len(data), None, text.splitlines(True), filename)
    return ast.parse(text), filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--implementation", required=True, choices=("base", "pr175"))
    args = parser.parse_args()
    revision = BASE if args.implementation == "base" else PATCH
    tree, filename = source(args.source_repo.resolve(), revision, "deepinfer/engine/llm_engine.py")
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMEngine")
    selected = {"close", "__del__", "_run_workers"}
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in original.body} == selected
    logger = logging.getLogger("isolated.original.close")
    namespace = {"suppress": suppress, "logger": logger}
    future = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    exec(compile(ast.Module(body=[*future, original], type_ignores=[]), filename, "exec"), namespace)

    test_tree, test_filename = source(args.source_repo.resolve(), PATCH, "tests/engine/test_engine_close.py")
    removed = [node for node in test_tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "deepinfer.engine.llm_engine"]
    assert len(removed) == 1
    test_tree.body = [node for node in test_tree.body if node not in removed]
    tests = ModuleType("_original_close_tests")
    tests.__file__ = test_filename
    tests.__dict__.update(LLMEngine=namespace["LLMEngine"], logger=logger)
    sys.modules[tests.__name__] = tests
    exec(compile(test_tree, test_filename, "exec"), tests.__dict__)

    class OriginalTests(pytest.Module):
        def _getobj(self):
            return tests

    class Adapter:
        def __init__(self):
            self.calls = 0
            self.failures = set()

        def pytest_pycollect_makemodule(self, module_path, parent):
            if module_path == Path(__file__).resolve():
                return OriginalTests.from_parent(parent, path=module_path)

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                self.calls += 1
                if report.failed:
                    self.failures.add(report.nodeid.split("::", 1)[1])
            elif report.failed:
                raise AssertionError("unexpected fixture failure: " + report.nodeid)

    adapter = Adapter()
    code = pytest.main(["-q", "--tb=line", "--noconftest", "-p", "no:cacheprovider",
                        "-o", "markers=ci: original core mark\nci_npu: original NPU mark",
                        "-k", "not worker_stop_error_does_not_skip_local_process_cleanup",
                        str(Path(__file__).resolve())], plugins=[adapter])
    expected = set()
    if args.implementation == "base":
        expected = {f"test_close_attempts_other_resources_after_failure[{error}-{stage}]"
                    for error in ("RuntimeError", "KeyboardInterrupt")
                    for stage in ("builder", "workers", "multimodal")}
        expected.update({"test_close_preserves_first_error_when_multiple_resources_fail",
                         "test_close_before_driver_initialization[False]",
                         "test_close_before_driver_initialization[True]",
                         "test_destructor_suppresses_shutdown_error"})
    assert adapter.calls == 13, adapter.calls
    assert adapter.failures == expected, (adapter.failures, expected)
    assert int(code) == (1 if expected else 0), code
    for forbidden in ("torch", "deepinfer", "deep_cluster", "openai", "huggingface_hub"):
        assert forbidden not in sys.modules, forbidden
    print("isolated original methods/tests:", 13 - len(expected), "passed;", len(expected),
          "expected failures; pytest exit", int(code), flush=True)
    print("NOT RUN: full imports/class/engine, subprocess cleanup test, device/collective cleanup", flush=True)


if __name__ == "__main__":
    main()
