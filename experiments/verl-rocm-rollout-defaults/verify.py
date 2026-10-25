"""#8371 original three CPU parameter cases and non-executing shell syntax checks.

Fixed verl Git objects; original test bodies/decorators and selected definitions.
Package/class shells skip platform discovery and optional vLLM imports.
Upstream source: verl-project/verl, Apache-2.0, ByteDance/BAAI contributors.
"""
import __future__
import argparse
import ast
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType

import pytest
import torch

VERSIONS = {"base": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "head": "7b6c38123bdf8ec7c70cadaf2d2796d36de67b8f"}
QUANT = "verl/utils/vllm/vllm_quant_utils.py"
ROCM = "verl/plugin/platform/platform_rocm.py"
CUDA = "verl/plugin/platform/platform_cuda.py"
TEST = "tests/utils/test_vllm_rocm_rollout_defaults_on_cpu.py"
RECIPE = "examples/grpo_trainer/run_deepseek_v4_veomni.sh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, results, syntax = {}, {}, {}

    def read(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        if path.endswith(".sh"):
            subprocess.run(["bash", "-n"], input=raw, check=True, capture_output=True)
            syntax[revision] = "bash -n passed; recipe not executed"
            return None
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def definition(tree, name):
        return next(n for n in tree.body if getattr(n, "name", None) == name)

    def execute(nodes, ns):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), ns)

    for name in ("verl", "verl.utils", "verl.utils.vllm"):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            setattr(sys.modules[parent], leaf, module)
    test_tree = read(VERSIONS["head"], TEST)
    original_tests = [n for n in test_tree.body if isinstance(n, ast.FunctionDef)]
    constants = [n for n in test_tree.body if isinstance(n, ast.Assign)]
    assert len(original_tests) == 2 and len(constants) == 1
    for label, revision in VERSIONS.items():
        module_name = "verl.utils.vllm.vllm_quant_utils"
        module = ModuleType(module_name)
        sys.modules[module_name] = module
        sys.modules["verl.utils.vllm"].vllm_quant_utils = module
        ns = module.__dict__
        ns.update(torch=torch, os=os, Mapping=Mapping, pytest=pytest)
        tree = read(revision, QUANT)
        execute([definition(tree, name) for name in ("_is_nvfp4_quantization", "_uses_fp32_block_scales",
                                                    "_is_hopper", "set_default_vllm_quantization_backends")], ns)
        cuda = definition(read(revision, CUDA), "PlatformCUDA")
        rocm = definition(read(revision, ROCM), "PlatformROCm")
        execute([ast.ClassDef(name="PlatformCUDA", bases=[], keywords=[],
                              body=[definition(cuda, "rollout_env_vars")], decorator_list=[]),
                 ast.ClassDef(name="PlatformROCm", bases=[ast.Name(id="PlatformCUDA", ctx=ast.Load())],
                              keywords=[], body=[definition(rocm, "rollout_env_vars")], decorator_list=[])], ns)
        execute(constants + original_tests, ns)
        cases = {}
        for name, hip in (("hip_none", None), ("hip_7_2", "7.2"), ("environment", None)):
            with pytest.MonkeyPatch.context() as monkeypatch:
                try:
                    if name == "environment":
                        ns["test_rocm_rollout_keeps_deepseek_v4_weights_reloadable"](monkeypatch)
                    else:
                        ns["test_deep_gemm_is_only_the_default_backend_off_rocm"](monkeypatch, hip)
                except (AssertionError, KeyError) as error:
                    expected = AssertionError if name == "hip_7_2" else KeyError
                    assert label == "base" and name != "hip_none" and isinstance(error, expected)
                    cases[name] = "expected_" + type(error).__name__
                else:
                    assert label == "head" or name == "hip_none"
                    cases[name] = "passed"
        results[label] = cases
        read(revision, RECIPE)
    read(VERSIONS["base"], "verl/trainer/main_ppo.py")
    read(VERSIONS["base"], "verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    assert all(name not in sys.modules for name in ("vllm", "ray"))
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "original_cases": results,
                      "shell_syntax": syntax,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "pytest": pytest.__version__, "system": platform.system()},
                      "scope": "original two test bodies / three parameter cases; monkeypatched torch.version.hip "
                               "and _is_hopper; isolated original functions and two environment methods; "
                               "not native pytest/imports, ROCm detection, backend choice, Ray propagation, "
                               "recipe commands, kernel execution, weight reload or training"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
