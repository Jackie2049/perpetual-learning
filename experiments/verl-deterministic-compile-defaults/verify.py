"""Check #8366's original dictionary helper/tests and the isolated caller gate, without vLLM."""
import __future__
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import pytest


VERSIONS = {"base": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "head": "e1ac734c5db176dfdb1bed74a67306b6ee89da46"}
UTILS = "verl/workers/rollout/vllm_rollout/utils.py"
SERVER = "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
TEST = "tests/workers/rollout/test_vllm_cli_args_on_cpu.py"
FUNCTION = "apply_deterministic_inductor_defaults"
CLASS = "TestApplyDeterministicInductorDefaults"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, trees = {}, {}
    for label, revision in VERSIONS.items():
        for path in (UTILS, SERVER, TEST):
            raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
            tree = ast.parse(raw, filename=path)
            compile(tree, path, "exec")
            trees[label, path] = tree
    assert not any(getattr(node, "name", None) == FUNCTION for node in trees["base", UTILS].body)
    helper = next(node for node in trees["head", UTILS].body if getattr(node, "name", None) == FUNCTION)
    tests = next(node for node in trees["head", TEST].body if getattr(node, "name", None) == CLASS)
    ns = {"pytest": pytest}
    exec(compile(ast.Module(body=[helper, tests], type_ignores=[]), "original_definitions", "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    suite = ns[CLASS]()
    cases = []
    for name in ("test_sets_combo_kernels_without_benchmarking", "test_keeps_other_inductor_options"):
        getattr(suite, name)()
        cases.append(name)
    for value in ({"combo_kernels": False}, {"benchmark_combo_kernel": True}, {"combo_kernels": True}):
        suite.test_user_combo_kernel_settings_win(value)
        cases.append("test_user_combo_kernel_settings_win:" + json.dumps(value, sort_keys=True))
    assert len(cases) == 5
    gates = [node for node in ast.walk(trees["head", SERVER]) if isinstance(node, ast.If)
             and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                     and child.func.id == FUNCTION for child in ast.walk(node))
             and ast.unparse(node.test) == "self.config.full_determinism"]
    assert len(gates) == 1
    gate_code = compile(ast.Module(body=gates, type_ignores=[]), "original_caller_if", "exec")
    gated = {}
    for enabled in (False, True):
        config = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
        exec(gate_code, {FUNCTION: ns[FUNCTION], "compilation_config": config,
                         "self": SimpleNamespace(config=SimpleNamespace(full_determinism=enabled))})
        assert ("inductor_compile_config" in config) is enabled
        gated[str(enabled)] = config
    override = {"inductor_compile_config": {"deterministic": False}}
    ns[FUNCTION](override)
    assert override == {"inductor_compile_config": {"deterministic": False, "combo_kernels": True,
                                                   "benchmark_combo_kernel": False}}
    assert "verl" not in sys.modules and "vllm" not in sys.modules and "torch" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "original_cases_passed": cases,
                      "isolated_gate_results": gated, "per_compile_false_control": override,
                      "environment": {"python": platform.python_version(), "pytest": pytest.__version__,
                                      "system": platform.system()},
                      "scope": "original helper and 5 unchanged CPU test parameter cases; original isolated "
                               "full_determinism if-node on two flag values, not launch_server; "
                               "one per-compile override control; 6 modules syntax checked; "
                               "no full pytest/38-case suite, torch/vLLM import, Inductor compile, engine/GPU/training"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
