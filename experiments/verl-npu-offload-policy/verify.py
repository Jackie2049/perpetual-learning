"""Check overlapping #8372/3/4 source and #8374's seven original CPU cases.

Original definitions/tests: verl-project/verl, Apache-2.0, ByteDance contributors.
Tests retain their original mocked transfers; no accelerator is discovered or
used. This does not validate NPU host allocation, synchronization or FSDP runs.
"""
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
import traceback
from types import ModuleType

import pytest
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FSDPModule
from torch.distributed.fsdp._runtime_utils import _lazy_init


VERSIONS = {"main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "base": "e009360c03e708e75d5628c9b7ad584af8a53e4d",
            "8372": "648423c5a5cece57a5412f8c1d96bee7d068d512",
            "8373": "340680d39c84293c2bebdaf292a3941b77958b47",
            "8374": "8b7aa26c055e8f4cbbfc8989e429e4265b5dcb67"}
SOURCE = "verl/utils/fsdp_utils.py"
TEST = "tests/utils/test_fsdp_model_transfer_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, trees = {}, {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    for label, revision in VERSIONS.items():
        trees[label] = source(revision, SOURCE)
    normalized = {label: ast.dump(tree, include_attributes=False) for label, tree in trees.items()}
    assert hashes[VERSIONS["base"] + ":" + SOURCE] == hashes[VERSIONS["main"] + ":" + SOURCE]
    assert normalized["8372"] == normalized["8374"]

    def function_ast(label, name):
        node = next(node for node in trees[label].body if getattr(node, "name", None) == name)
        return ast.dump(node, include_attributes=False)

    assert function_ast("8373", "offload_fsdp_model_to_cpu") == function_ast("8374", "offload_fsdp_model_to_cpu")
    assert function_ast("8373", "offload_fsdp_optimizer") == function_ast("base", "offload_fsdp_optimizer")
    for label in VERSIONS:
        assert function_ast(label, "offload_fsdp2_model_to_cpu") == function_ast("base", "offload_fsdp2_model_to_cpu")
        assert function_ast(label, "load_fsdp_model_to_gpu") == function_ast("base", "load_fsdp_model_to_gpu")
        assert function_ast(label, "load_fsdp_optimizer") == function_ast("base", "load_fsdp_optimizer")

    test_tree = source(VERSIONS["8374"], TEST)
    test_tree.body = [node for node in test_tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "verl.utils")]
    test_code = compile(test_tree, TEST, "exec")
    results = {}

    def unpatched_device_guard():
        raise AssertionError("Original tests must replace device discovery before use")

    for label in ("base", "8374"):
        module = ModuleType("_original_fsdp_policy_" + label)
        ns = module.__dict__
        ns.update(torch=torch, FSDP=FSDP, FSDPModule=FSDPModule, _lazy_init=_lazy_init,
                  get_device_name=unpatched_device_guard, get_torch_device=unpatched_device_guard)
        names = {"fsdp_version", "offload_fsdp_model_to_cpu", "offload_fsdp2_model_to_cpu",
                 "offload_fsdp_optimizer"}
        nodes = [node for node in trees[label].body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), SOURCE, "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        tests = {"fsdp_utils": module}
        exec(test_code, tests)
        cases = []
        for device, non_blocking in (("npu", False), ("cuda", True)):
            for empty_cache in (False, True):
                cases.append(("test_offload_fsdp_model_copy_policy_and_shard_aliases",
                              (device, non_blocking, empty_cache), device))
            cases.append(("test_offload_fsdp_optimizer_copy_policy", (device, non_blocking), device))
        observed = {}
        for name, params, device in cases:
            expected = "expected_assertion_failure" if label == "base" and device == "npu" else "passed"
            try:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    tests[name](monkeypatch, *params)
            except AssertionError as exc:
                assert expected == "expected_assertion_failure"
                if name == "test_offload_fsdp_model_copy_policy_and_shard_aliases":
                    # The upstream call-list assert has no message without pytest
                    # rewriting. Require its exact fixed-source location instead.
                    assert any(frame.filename == TEST and frame.lineno == 65
                               for frame in traceback.extract_tb(exc.__traceback__))
                else:
                    assert "non_blocking" in str(exc)
                status = "expected_assertion_failure"
            else:
                status = "passed"
            assert status == expected, (label, name, params, status)
            observed[name + ":" + repr(params)] = status
        tests["test_offload_fsdp_optimizer_without_state"]()
        observed["test_offload_fsdp_optimizer_without_state"] = "passed"
        results[label] = observed

    assert "verl" not in sys.modules and "torch_npu" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "original_tests": results,
                      "source_checks": {"8372_8374_full_module_ast_equal": True,
                                        "8373_model_ast_equal_8374": True,
                                        "8373_optimizer_unchanged": True,
                                        "fsdp2_and_reload_functions_unchanged": True,
                                        "base_main_full_source_equal": True},
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "pytest": pytest.__version__, "system": platform.system()},
                      "scope": "seven unchanged original #8374 CPU parameter cases on base/head; "
                               "original functions/type dispatch with original mock handles, device modules "
                               "and optimizer tensors; discovery imports replaced by fail-closed guards; "
                               "no full pytest/FSDP construction/NPU or CUDA transfers/allocator metrics"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
