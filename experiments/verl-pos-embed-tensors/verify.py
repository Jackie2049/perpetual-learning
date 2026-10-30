"""Function-level CPU equivalence controls for verl #8414, not a vision benchmark."""

import ast
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
HEAD = "d8c490b269d8c288b04b140f254592157c4aecd0"
TEST = "tests/models/test_qwen_vl_pos_embed_interpolate_on_cpu.py"
MODULES = ["qwen3_vl", "qwen3_5"]


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main(repo):
    sources = {}

    def read(revision, path):
        value = subprocess.check_output(["git", "-C", repo, "show", f"{revision}:{path}"],
                                        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
        compile(value, path, "exec")
        sources[f"{revision}:{path}"] = hashlib.sha256(value).hexdigest()
        return value

    functions = {}
    for label, revision in [("base", BASE), ("head", HEAD)]:
        functions[label] = {}
        nodes = []
        for module in MODULES:
            path = f"verl/models/transformers/{module}.py"
            source = read(revision, path)
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "fast_pos_embed_interpolate")
            nodes.append(ast.dump(node, include_attributes=False))
            namespace = {"torch": torch}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
            functions[label][module] = namespace[node.name]
        assert nodes[0] == nodes[1]

    test_source = read(HEAD, TEST)
    selected = [n for n in ast.parse(test_source).body if isinstance(n, ast.FunctionDef)
                and n.name in {"_reference_fast_pos_embed_interpolate", "test_fast_pos_embed_interpolate_matches_reference"}]
    assert len(selected) == 2
    namespace = {"torch": torch, "pytest": pytest}
    exec(compile(ast.Module(body=selected, type_ignores=[]), TEST, "exec"), namespace)
    test = namespace["test_fast_pos_embed_interpolate_matches_reference"]
    parameters = {mark.args[0]: mark.args[1] for mark in test.pytestmark if mark.name == "parametrize"}
    torch.set_num_threads(1)

    def make_model(dtype, num_grid_per_side=48, hidden_size=32, spatial_merge_size=2):
        # Same numerical fixture as upstream, but seed only CPU and force CPU allocation.
        torch.random.default_generator.manual_seed(0)
        embedding = torch.nn.Embedding(num_grid_per_side * num_grid_per_side, hidden_size,
                                       device="cpu").to(dtype)
        return SimpleNamespace(pos_embed=embedding, num_grid_per_side=num_grid_per_side,
                               config=SimpleNamespace(spatial_merge_size=spatial_merge_size))

    results = {}
    for label in ("base", "head"):
        records = []

        def load_impl(module):
            implementation = functions[label][module]

            def record(model, grid):
                before_grid = grid.clone()
                before_weight = model.pos_embed.weight.detach().clone()
                output = implementation(model, grid)
                assert torch.equal(before_grid, grid)
                assert torch.equal(before_weight, model.pos_embed.weight)
                assert list(output.shape) == [int(grid.prod(1).sum().item()), 32]
                records.append({"module": module, "dtype": str(output.dtype), "grid_thw": grid.tolist(),
                                "shape": list(output.shape), "device": str(output.device),
                                "output_sha256": tensor_hash(output), "inputs_unchanged": True})
                return output

            return record

        namespace["_load_impl"] = load_impl
        namespace["_make_vision_model"] = make_model
        for module, dtype, grid in itertools.product(parameters["module_name"], parameters["dtype"],
                                                     parameters["grid_thw"]):
            test(module, dtype, grid)
        assert len(records) == 16
        results[label] = records
    assert results["base"] == results["head"]
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "pytest": pytest.__version__, "threads": 1,
                      "base": BASE, "head": HEAD, "source_sha256": sources,
                      "same_function_ast_across_two_models_per_revision": True,
                      "original_test_body_calls": {"base": 16, "head": 16},
                      "base_and_head_records_equal": True, "results": results,
                      "scope": "Original parameter marks, test body and list-based reference; original interpolation functions AST-isolated without framework imports. Two local helper adapters replace importorskip and model factory (CPU-only seed/allocation, same Embedding dimensions/config). Direct calls, not native pytest collection or original Transformers imports. No old-fail/new-pass claim, backward, full vision tower, CUDA/offload, GPU/NPU, timing or training validation."}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
