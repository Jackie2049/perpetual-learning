"""#8442 original CPU converter/tests plus own value and module-MRO controls.

Usage: python experiments/verl-moe-layout-provenance/verify.py /cached/verl-git
No native verl import, FSDP wrapping, NPU patch, vLLM loader or weight IPC.
"""

import __future__
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from types import SimpleNamespace

import torch

HEAD = "3a1ff0432aa3187aabbc700ad5852a658b8656d6"
BASE = "2ab35292039a51175a4b3d3b3b2d76e951a0d3e4"
API_BASE = "a08ea02a784d7d4982b1d81df36d188a476192bb"
UTILS = "verl/workers/engine/fsdp/utils.py"
TESTS = "tests/workers/rollout/test_vllm_moe_param_converter_on_cpu.py"
ENGINE = "verl/workers/engine/fsdp/transformer_impl.py"
PACKED = "model.layers.0.mlp.experts"
GATE_UP = PACKED + ".gate_up_proj"
DOWN = PACKED + ".down_proj"


def execute(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), namespace)


def main(cache):
    hashes = {}
    git_env = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def read(ref, path):
        raw = subprocess.check_output(["git", "-C", cache, "show", f"{ref}:{path}"], env=git_env)
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()
        compile(raw, f"{ref}:{path}", "exec")
        return ast.parse(raw)

    merge_base = subprocess.check_output(
        ["git", "-C", cache, "merge-base", HEAD, BASE], env=git_env
    ).decode().strip()
    assert merge_base == BASE
    parent = subprocess.check_output(
        ["git", "-C", cache, "show", "-s", "--format=%P", HEAD], env=git_env
    ).decode().strip()
    assert parent == BASE
    torch.random.default_generator.manual_seed(0)  # CPU only; no all-device seeding hooks.
    converters, original_tests = {}, {}
    for revision in (BASE, HEAD):
        tree = read(revision, UTILS)
        converter = next(node for node in tree.body if getattr(node, "name", None) == "unfuse_moe_params")
        ns = {"torch": torch}
        execute([converter], ns)
        converters[revision] = ns["unfuse_moe_params"]
        tests = read(revision, TESTS)
        # The sole removed import would eagerly import the full device-aware verl package.
        removed = [node for node in tests.body if isinstance(node, ast.ImportFrom)
                   and node.module == "verl.workers.engine.fsdp.utils"]
        assert len(removed) == 1
        execute([node for node in tests.body if node not in removed], ns)
        results = []
        for node in tests.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                assert not node.args.args and not node.decorator_list
                ns[node.name]()  # Each revision's own unchanged assertions and matching API.
                results.append({"name": node.name, "status": "passed"})
        original_tests[revision] = results
    assert [len(original_tests[ref]) for ref in (BASE, HEAD)] == [2, 5]
    old, new = converters[BASE], converters[HEAD]

    def collect(fn, weights, **kwargs):
        return dict(fn(weights, "qwen3_vl_moe", **kwargs))

    def check_equal(actual, expected):
        assert list(actual) == list(expected)
        for name in expected:
            assert actual[name].device.type == "cpu"
            assert actual[name].is_contiguous()
            assert torch.equal(actual[name], expected[name]), name

    # Own rectangular HF reference; both revisions receive API-compatible calls.
    experts, hidden, intermediate = 2, 8, 3
    gate = torch.arange(experts * 2 * intermediate * hidden, dtype=torch.float32).reshape(
        experts, 2 * intermediate, hidden)
    down = (1000 + torch.arange(experts * hidden * intermediate, dtype=torch.float32)).reshape(
        experts, hidden, intermediate)
    hf_weights = [(GATE_UP, gate), (DOWN, down)]
    expected = {}
    for expert in range(experts):
        expected[f"{PACKED}.{expert}.gate_proj.weight"] = gate[expert, :intermediate].contiguous()
        expected[f"{PACKED}.{expert}.up_proj.weight"] = gate[expert, intermediate:].contiguous()
    for expert in range(experts):
        expected[f"{PACKED}.{expert}.down_proj.weight"] = down[expert].contiguous()
    check_equal(collect(old, hf_weights), expected)
    check_equal(collect(new, hf_weights, hidden_size=hidden), expected)
    npu_weights = [(name, value.transpose(1, 2).contiguous()) for name, value in hf_weights]
    old_npu = collect(old, npu_weights)
    heuristic = collect(new, npu_weights, hidden_size=hidden)
    explicit = collect(new, npu_weights, hidden_size=hidden, npu_gmm_layout=True)
    check_equal(heuristic, expected)
    check_equal(explicit, expected)
    check_equal(collect(new, npu_weights), old_npu)  # No metadata retains old behavior.
    gate_key, down_key = f"{PACKED}.0.gate_proj.weight", f"{PACKED}.0.down_proj.weight"
    assert old_npu[gate_key].shape == (4, 6) != expected[gate_key].shape
    rectangular = {
        "experts": experts, "hidden": hidden, "intermediate": intermediate,
        "expected_gate": list(expected[gate_key].shape), "old_npu_gate": list(old_npu[gate_key].shape),
        "expected_down": list(expected[down_key].shape), "old_npu_down": list(old_npu[down_key].shape),
        "hf_base_head_exact": True, "head_heuristic_exact": True, "head_flag_exact": True,
        "head_no_metadata_matches_old_wrong_axis": True,
    }

    # Separate projection controls: gate square H=2I, down square H=I.
    # These two I values do NOT describe one coherent model configuration.
    square = {}
    for projection, packed_name, size in (("gate_up", GATE_UP, 4), ("down", DOWN, 4)):
        hf = torch.arange(2 * size * size, dtype=torch.float32).reshape(2, size, size)
        npu = hf.transpose(1, 2).contiguous()
        reference = collect(old, [(packed_name, hf)])
        wrong = collect(new, [(packed_name, npu)], hidden_size=size)
        right = collect(new, [(packed_name, npu)], hidden_size=size, npu_gmm_layout=True)
        check_equal(right, reference)
        check_equal(wrong, collect(old, [(packed_name, npu)]))
        assert all(wrong[name].shape == reference[name].shape for name in reference)
        gap = max((wrong[name] - reference[name]).abs().max().item() for name in reference)
        assert gap > 0
        first = next(iter(reference))
        square[projection] = {
            "hidden": size, "intermediate": size // 2 if projection == "gate_up" else size,
            "same_output_shapes_without_flag": True, "max_value_gap_without_flag": gap,
            "flag_matches_hf_values_exactly": True, "first_key": first,
            "hf_expected_first": reference[first].tolist(), "without_flag_first": wrong[first].tolist(),
        }

    # Execute ONLY the original layout-detection RHS, not get_per_tensor_param.
    # Empty own nn.Module fixtures test class-name/MRO logic, not actual FSDP/NPU installation.
    engine = read(HEAD, ENGINE)
    assignments = [node for node in ast.walk(engine) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "npu_gmm_layout"
                           for target in node.targets)]
    assert len(assignments) == 1
    expression = compile(ast.Expression(body=assignments[0].value), "original_mro_rhs", "eval")
    expert_type = type("NPUQwen3VLMoeTextExperts", (torch.nn.Module,), {})
    wrapped_type = type("FSDPNPUQwen3VLMoeTextExperts", (expert_type,), {})
    trees = {"stock": torch.nn.Linear(2, 2), "own_wrapped": wrapped_type(),
             "own_mixed_tree": torch.nn.ModuleList([torch.nn.Linear(2, 2), wrapped_type()])}
    mro_flags = {name: eval(expression, {"self": SimpleNamespace(module=module)})
                 for name, module in trees.items()}
    assert mro_flags == {"stock": False, "own_wrapped": True, "own_mixed_tree": True}
    assert type(trees["own_wrapped"]).__name__ != "NPUQwen3VLMoeTextExperts"

    # GPT-OSS bypass precedes even an explicit flag; unknown keys pass through by identity.
    passed = list(new(npu_weights, "gpt_oss", hidden_size=hidden, npu_gmm_layout=True))
    assert all(name == original_name and value is original_value
               for (name, value), (original_name, original_value) in zip(passed, npu_weights, strict=True))
    extra = torch.ones(2, 2)
    assert list(new([("other.weight", extra)], npu_gmm_layout=True))[0][1] is extra
    print(json.dumps({
        "head": HEAD, "api_target_base_observed": API_BASE, "actual_parent_and_merge_base": merge_base,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_git": torch.version.git_version, "device": "cpu"},
        "source_sha256": hashes, "original_test_bodies_manual": original_tests,
        "rectangular_own_control": rectangular, "independent_square_projection_controls": square,
        "original_mro_expression_on_own_modules": mro_flags,
        "gpt_oss_explicit_flag_preserves_identity": True, "unmatched_key_preserves_identity": True,
        "scope": "Original converter and each revision's original plain test bodies; own value/MRO fixtures. "
                 "No native verl import, full export entry, FSDP/NPU patch, vLLM loader, IPC, model or training.",
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
