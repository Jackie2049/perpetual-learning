"""Three original unfused CPU test cases plus two own full-mask controls.

No Megatron import, original pytest collection, SP/CP collective or fused kernel.
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
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nested._internal.nested_tensor import NestedTensor

HEAD = "03e1b2951ea7f5aaa49ac4fb8b2c8aa85a4c5eb7"
SOURCE = "verl/models/mcore/response_only_lm_head.py"
TEST = "tests/models/mcore/test_response_only_lm_head_on_cpu.py"
FORWARD = "verl/models/mcore/model_forward.py"
READING = ("verl/models/mcore/model_forward_fused.py", "verl/models/mcore/util.py",
           "verl/workers/engine/megatron/transformer_impl.py", "verl/utils/kernel/kernels.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw_sources, hashes = {}, {}
    for path in (SOURCE, TEST, FORWARD, *READING):
        raw = subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        compile(ast.parse(raw), path, "exec")
        raw_sources[path] = raw
        hashes[path] = hashlib.sha256(raw).hexdigest()
    original = ModuleType("original_response_only_lm_head")
    original.__file__ = f"{HEAD}:{SOURCE}"
    exec(compile(raw_sources[SOURCE], SOURCE, "exec"), original.__dict__)
    ns = {"torch": torch, "pytest": pytest, "SimpleNamespace": SimpleNamespace,
          "NestedTensor": NestedTensor,
          **{key: getattr(original, key) for key in ("response_only_output_projection",
                                                    "restore_response_only_outputs", "select_response_only_inputs")}}
    wanted = {"_OutputLayer", "_Model", "test_unfused_projection_preserves_values_and_gradients",
              "test_unfused_sequence_parallel_rejects_deferred_weight_gradients"}
    nodes = [n for n in ast.parse(raw_sources[TEST]).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # manually run original empty=False/True parameters
    exec(compile(ast.Module(body=nodes, type_ignores=[]), TEST, "exec"), ns)
    seam = ModuleType("verl.utils.megatron_utils")
    seam.unwrap_model = lambda model: model  # plain original CPU fixture, not real Megatron wrappers
    reports = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, seam.__name__, seam)
        for empty in (False, True):
            ns["test_unfused_projection_preserves_values_and_gradients"](empty)
            reports.append({"test": "test_unfused_projection_preserves_values_and_gradients", "empty": empty,
                            "outcome": "passed"})
        ns["test_unfused_sequence_parallel_rejects_deferred_weight_gradients"]()
        reports.append({"test": "test_unfused_sequence_parallel_rejects_deferred_weight_gradients", "outcome": "passed"})

    build_mask = next(n for n in ast.parse(raw_sources[FORWARD]).body
                      if isinstance(n, ast.FunctionDef) and n.name == "_build_full_loss_mask_nested")
    exec(compile(ast.Module(body=[build_mask], type_ignores=[]), FORWARD, "exec"), ns)
    responses = [torch.tensor([1, 1, 0, 1, 0]), torch.tensor([1, 0, 1, 1])]
    expected = [[0, 0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 1]]
    masks = []
    for padded in (False, True):
        response_mask = (torch.nn.utils.rnn.pad_sequence(responses, batch_first=True) if padded else
                         torch.nested.nested_tensor(responses, layout=torch.jagged))
        attention = (torch.nn.utils.rnn.pad_sequence([torch.ones_like(row) for row in responses], batch_first=True)
                     if padded else None)
        actual = ns["_build_full_loss_mask_nested"](response_mask, [8, 8], attention)
        rows = [row.tolist() for row in actual.unbind()]
        assert rows == expected
        masks.append({"padded": padded, "full_loss_mask": rows})
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "pytest": pytest.__version__,
                      "head": HEAD, "source_sha256": hashes, "original_test_cases": reports,
                      "own_full_mask_controls": masks,
                      "scope": "Complete original response-only helper module. Two original test functions and "
                      "two original nn.Module fixture classes extracted via AST; only parametrize decorators "
                      "removed, bodies/assertions unchanged, three cases run manually (not native pytest). "
                      "True CPU linear projection/autograd, mask selection/restoration and empty connected graph; "
                      "original objective is selected logits squared, not policy CE. Identity unwrap import seam "
                      "for plain CPU fixtures. Deferred-SP guard rejects before any gather; no SP executed. "
                      "Original full-mask builder AST plus two own real nested/padded controls; no causal roll or "
                      "CP slicing helper execution. Seven complete inputs compile. Remaining 12 original cases, "
                      "Megatron model/wrapper, true SP/TP/CP/PP, engine, Triton/Liger, GPU/NPU, training, performance "
                      "and allocator peak are not executed."}, indent=2))


if __name__ == "__main__":
    main()
