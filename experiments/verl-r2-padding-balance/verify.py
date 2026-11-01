"""CPU-only extraction of PR #7524's original helper and one original test.

Run: python verify.py /path/to/read-only/verl-git-cache
No Megatron import, device discovery, packing, dispatcher, or model execution.
"""

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

HEAD = "a79099997aeb95bbdf1771fd7631bd9e67101c4c"
BASE = "3c6d9b43b67c74ec343d77874cd8fa6e841523da"
SOURCE = "verl/utils/megatron/router_replay_utils.py"
TEST = "tests/utils/megatron/test_router_replay_model_walk_on_cpu.py"
HELPER = "_distribute_router_replay_padding"
TEST_NAME = "test_padding_routes_preserve_valid_rows_and_balance_experts"


def main(cache):
    environment = dict(os.environ, GIT_NO_LAZY_FETCH="1")

    def git(*args):
        return subprocess.check_output(["git", "-C", cache, *args], env=environment).decode()

    assert git("merge-base", HEAD, BASE).strip() == BASE
    texts = {path: git("show", f"{HEAD}:{path}") for path in (SOURCE, TEST)}
    trees = {path: ast.parse(text) for path, text in texts.items()}
    helper_node = next(node for node in trees[SOURCE].body if isinstance(node, ast.FunctionDef)
                       and node.name == HELPER)
    test_node = next(node for node in trees[TEST].body if isinstance(node, ast.FunctionDef)
                     and node.name == TEST_NAME)
    namespace = {"torch": torch, "pytest": pytest}
    for path, node in ((SOURCE, helper_node), (TEST, test_node)):
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, f"{HEAD}:{path}", "exec"), namespace)
        if path == SOURCE:
            namespace["router_replay_utils"] = SimpleNamespace(**{HELPER: namespace[HELPER]})
    helper = namespace[HELPER]
    # Preserve the original decorator and obtain its parameters, but invoke the
    # body manually: these are not native pytest collection or module fixtures.
    parameters = ast.literal_eval(test_node.decorator_list[0].args[1])
    original_cases = []
    for values in parameters:
        namespace[TEST_NAME](*values)
        original_cases.append(dict(num_experts=values[0], ep=values[1], padding_rows=values[2], passed=True))

    def accounting(indices, num_experts, ep):
        # Own counting model: duplicate top-k IDs collapse per (row, expert).
        # This is NOT Megatron's dispatch or its runtime storage allocation.
        membership = torch.zeros((indices.shape[0], num_experts), dtype=torch.bool)
        membership.scatter_(1, indices.to(torch.int64), True)
        expert_load = membership.sum(dim=0)
        rank_load = expert_load.reshape(ep, num_experts // ep).sum(dim=1)
        return dict(raw_topk_entries=indices.numel(), unique_row_expert_pairs=int(membership.sum()),
                    max_rank_pairs=int(rank_load.max()), min_rank_pairs=int(rank_load.min()))

    own_counts = []
    for ep, expected_peak in ((32, 60), (8, 240), (4, 480)):
        routes = torch.zeros((1, 240, 1, 8), dtype=torch.int16)
        before = accounting(routes[0, :, 0], 256, ep)
        pointer = routes.data_ptr()
        returned = helper(routes, torch.zeros((1, 240), dtype=torch.bool), 256, ep)
        assert returned is routes and returned.data_ptr() == pointer
        after = accounting(routes[0, :, 0], 256, ep)
        assert before["raw_topk_entries"] == after["raw_topk_entries"] == 1920
        assert before["unique_row_expert_pairs"] == before["max_rank_pairs"] == 240
        assert after["unique_row_expert_pairs"] == 1920
        assert after["max_rank_pairs"] == after["min_rank_pairs"] == expected_peak
        own_counts.append(dict(ep=ep, before=before, after=after))

    no_padding = torch.arange(48, dtype=torch.int16).reshape(1, 3, 2, 8)
    snapshot = no_padding.clone()
    returned = helper(no_padding, torch.ones((1, 3), dtype=torch.bool), 256, 32)
    assert returned is no_padding and torch.equal(no_padding, snapshot)

    entry = next(node for node in trees[SOURCE].body if isinstance(node, ast.FunctionDef)
                 and node.name == "set_router_replay_data")
    guard = next(node.value for node in ast.walk(entry) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "balance_padding"
                         for target in node.targets))
    expression = compile(ast.Expression(guard), f"{HEAD}:{SOURCE}:balance_padding", "eval")
    guard_cases = []
    for has_replay_mask in (False, True):
        for capacity in (None, 0.0, 0.5, 1.0):
            enabled = eval(expression, {"replay_mask": object() if has_replay_mask else None,
                                       "tf_config": SimpleNamespace(moe_expert_capacity_factor=capacity)})
            assert enabled == (not has_replay_mask and capacity is None)
            guard_cases.append(dict(replay_mask_present=has_replay_mask, capacity=capacity, enabled=enabled))

    report = dict(head=HEAD, base=BASE, actual_merge_base=BASE, python=platform.python_version(),
                  torch=torch.__version__, torch_git=torch.version.git_version, pytest=pytest.__version__,
                  source_sha256={path: hashlib.sha256(text.encode()).hexdigest() for path, text in texts.items()},
                  original_parameter_bodies=original_cases, own_pair_accounting=own_counts,
                  own_no_padding_control=True, own_original_guard_expression_controls=guard_cases,
                  scope="Complete original helper and one original parameterized test body via AST; "
                  "three original parameters manually invoked, not native pytest/full module. "
                  "Own boolean row-expert counting, no-padding identity control, and eight original "
                  "guard-expression evaluations only. No production entry, packing, CP/SP, Megatron "
                  "dispatcher, capacity selection, gradients, GPU memory, timing, or training execution.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
