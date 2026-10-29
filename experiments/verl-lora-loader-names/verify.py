"""Run eight original name-normalization test bodies against fixed PR helpers.

Only selected complete AST definitions execute. No verl/vLLM module import,
real worker, weight load, device discovery or training. Original source and
tests: Apache-2.0, Copyright Bytedance Ltd. and/or its affiliates.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

import pytest
import torch


REV = "0ed94ba020747101bb22db6005849c9a6ca06aa6"
PREVIOUS = "2671d862e3fb5e937b0da70da348b86e37df724f"
PRODUCTION = "verl/utils/vllm/utils.py"
TEST_SOURCE = "tests/utils/test_vllm_weight_name_normalization_on_cpu.py"
FUNCTIONS = ["_inner_load_weights_is_strict", "_name_reaches_strict_loader", "resolve_weight_name"]
HELPERS = ["_FakeMapper", "_FakeVllmConfig", "_make_worker", "_resolve", "_FakeStrictInner",
           "_FakeFlatInner", "_OuterModel", "_strict_outer", "_flat_outer"]
TESTS = [
    "test_strict_loader_keeps_base_layer_suffix_on_newer_vllm",
    "test_strict_loader_re_suffixes_plain_name_on_newer_vllm",
    "test_flat_loader_strips_base_layer_suffix_on_newer_vllm",
    "test_strict_loader_vision_merger_still_strips",
    "test_strict_loader_multi_segment_shard_strips_to_unfused_leaf",
    "test_strict_loader_unprefixed_shard_against_model_prefixed_live",
    "test_packed_shard_keeps_suffix_when_fused_target_is_wrapped",
    "test_packed_shard_keeps_suffix_on_released_vllm",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    trees, hashes = {}, {}
    for path in (PRODUCTION, TEST_SOURCE):
        raw = subprocess.check_output(["git", "show", REV + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[path] = hashlib.sha256(raw).hexdigest()
        previous = subprocess.check_output(["git", "show", PREVIOUS + ":" + path], cwd=args.source_repo,
                                           env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        assert previous == raw, f"Source changed since first snapshot: {path}"
        trees[path] = ast.parse(raw, filename=path)
        compile(trees[path], path, "exec")  # Syntax check; no full-module execution.

    def definitions(path, names):
        selected = []
        for name in names:
            node = next(n for n in trees[path].body if getattr(n, "name", None) == name)
            assert not node.decorator_list, f"Unexpected decorator: {name}"
            selected.append(node)
        return selected

    production = types.ModuleType("isolated_original_vllm_utils")
    # Explicit fixtures for import-time feature probes; no vLLM/device import.
    production._HAS_LORA_LOAD_WEIGHTS = False
    production._HAS_LORA_BASE_LAYER_PREFIX = False
    # Keep the original cache and sentinel assignments, not substitutes.
    for name in ("_inner_load_weights_is_strict_cache", "_STRICT_PREFIX_MISS"):
        node = next(n for n in trees[PRODUCTION].body
                    if isinstance(n, (ast.Assign, ast.AnnAssign))
                    and any(isinstance(t, ast.Name) and t.id == name
                            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))
        exec(compile(ast.Module(body=[node], type_ignores=[]), PRODUCTION, "exec"), production.__dict__)
    exec(compile(ast.Module(body=definitions(PRODUCTION, FUNCTIONS), type_ignores=[]),
                 PRODUCTION, "exec"), production.__dict__)

    current_calls = []

    def observed_resolve(model, name, model_weight_names, **kwargs):
        resolved = production.resolve_weight_name(model, name, model_weight_names, **kwargs)
        current_calls.append({"input": name, "output": resolved,
                              "has_lora_load_weights_fixture": production._HAS_LORA_LOAD_WEIGHTS,
                              "has_lora_base_layer_prefix_fixture": production._HAS_LORA_BASE_LAYER_PREFIX,
                              "live_names": sorted(model_weight_names)})
        return resolved

    # The original worker fixture only allocates and assigns Python attributes;
    # _resolve immediately discards it. This empty class is NOT a real worker.
    ns = {"torch": torch, "types": types, "_vllm_utils_real": production,
          "vLLMColocateWorkerExtension": type("EmptyWorkerAllocationFixture", (), {}),
          "resolve_weight_name": observed_resolve}
    exec(compile(ast.Module(body=definitions(TEST_SOURCE, HELPERS + TESTS), type_ignores=[]),
                 TEST_SOURCE, "exec"), ns)
    results = []
    for name in TESTS:
        production._inner_load_weights_is_strict_cache.clear()
        production._HAS_LORA_LOAD_WEIGHTS = False
        production._HAS_LORA_BASE_LAYER_PREFIX = False
        current_calls.clear()
        with pytest.MonkeyPatch.context() as monkeypatch:
            ns[name](monkeypatch)
        results.append({"original_test": name, "passed": True, "calls": list(current_calls)})

    assert len(results) == 8 and sum(len(r["calls"]) for r in results) == 17
    assert "vllm" not in sys.modules and "verl" not in sys.modules
    print(json.dumps({"revision": REV, "previous_snapshot": PREVIOUS,
                      "both_complete_inputs_identical_to_previous": True, "python": platform.python_version(),
                      "torch": torch.__version__, "pytest": pytest.__version__,
                      "source_sha256": hashes, "production_definitions": FUNCTIONS,
                      "original_fixture_definitions": HELPERS,
                      "original_tests_passed": len(results), "resolver_calls": 17,
                      "results": results,
                      "scope": "Eight original test bodies and nine original fake helpers; three original production functions plus cache/sentinel. Direct calls with real pytest.MonkeyPatch, not native pytest collection. CPU empty tensors, fake mapper/loader topology and empty worker class; no real vLLM feature probes, loader execution, refit, quantization, model, GPU/NPU or training."}, indent=2))


if __name__ == "__main__":
    main()
