"""Original batching control flow with explicit recorders, not quantization tests."""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Iterable, Iterator

import torch

HEAD = "b84cd9ac42a50b1018f48a4afa9987728c225b0c"
SOURCE = "verl/utils/qat/compressed_tensors.py"
READING = ("tests/utils/test_batched_expert_export_on_cpu.py",
           "verl/workers/engine/veomni/transformer_impl.py", "verl/workers/engine/veomni/utils.py")


def load(repo, path):
    return subprocess.check_output(["git", "show", f"{HEAD}:{path}"], cwd=repo,
                                   env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    hashes = {}
    for path in (SOURCE, *READING):
        raw = load(args.repo, path)
        compile(ast.parse(raw), path, "exec")
        hashes[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(load(args.repo, SOURCE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CompressedTensorsWeightQuantizer")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    rows = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compress_rows")
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in {"_EXPERT_BATCH_BYTES", "_EXPERT_WEIGHT", "_PACKED_EXPERTS"}
                         for t in n.targets)]
    ns = {"torch": torch, "re": re, "Iterable": Iterable, "Iterator": Iterator, "QuantizationScheme": object}
    exec(compile(ast.Module(body=[*constants, call, rows], type_ignores=[]), SOURCE, "exec"), ns)

    class NoArrivalClone(ast.NodeTransformer):
        replacements = 0

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "tensor" and node.func.attr == "clone":
                self.replacements += 1
                return ast.copy_location(ast.Name(id="tensor", ctx=ast.Load()), node)
            return self.generic_visit(node)

    mutation = NoArrivalClone()
    altered = mutation.visit(copy.deepcopy(call))
    altered.name = "without_arrival_clone"
    assert mutation.replacements == 1
    exec(compile(ast.fix_missing_locations(ast.Module(body=[altered], type_ignores=[])), SOURCE, "exec"), ns)
    ns["_row_local"] = lambda scheme: True  # routing seam only; no CT strategy or compressor
    scheme = object()

    class Recorder:
        def __init__(self):
            self.runs = []

        def _scheme_for(self, name, tensor):
            return scheme

        def _compress_experts(self, run):
            self.runs.append({"bytes": sum(t.nbytes for _, t, _ in run),
                              "values": [t[0, 0].item() for _, t, _ in run]})
            for name, tensor, _ in run:
                yield name, tensor.clone()  # owned output for the recorder, not a quantizer

    def reused_stream():
        buffer = torch.empty(2, 3)
        for expert in range(3):
            buffer.fill_(expert + 1)
            yield f"model.layers.0.mlp.experts.{expert}.gate_proj.weight", buffer

    snapshots = {}
    for key in ("__call__", "without_arrival_clone"):
        recorder = Recorder()
        output = list(ns[key](recorder, reused_stream()))
        snapshots[key] = {"runs": recorder.runs, "values": [tensor[0, 0].item() for _, tensor in output]}
    assert snapshots["__call__"]["values"] == [1, 2, 3]
    assert snapshots["without_arrival_clone"]["values"] == [3, 3, 3]

    ns["_EXPERT_BATCH_BYTES"] = 1
    capped = Recorder()
    assert len(list(ns["__call__"](capped, reused_stream()))) == 3
    assert [run["bytes"] for run in capped.runs] == [24, 24, 24]

    weights = [torch.full((2, 3), value) for value in (1.0, 2.0)]
    calls = []

    def row_recorder(weight, scheme):
        calls.append(list(weight.shape))
        return {"weight": weight}

    ns["_compress_weight"] = row_recorder
    outputs = ns["_compress_rows"](weights, scheme)
    before = [output["weight"].tolist() for output in outputs]
    pointers = [output["weight"].untyped_storage().data_ptr() for output in outputs]
    assert calls == [[4, 3]] and pointers[0] != pointers[1]
    outputs[0]["weight"].fill_(99)
    assert torch.equal(outputs[1]["weight"], weights[1]) and weights[0][0, 0].item() == 1

    def scalar_recorder(weight, scheme):
        calls.append(list(weight.shape))
        return {"global_scale": torch.tensor(1.0)}

    ns["_compress_weight"] = scalar_recorder
    calls.clear()
    fallback = ns["_compress_rows"](weights, scheme)
    assert calls == [[4, 3], [2, 3], [2, 3]] and len(fallback) == 2
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "head": HEAD,
                      "source_sha256": hashes, "arrival_clone_control": snapshots,
                      "one_byte_cap_runs": capped.runs,
                      "row_split_control": {"values_before_mutation": before, "distinct_output_storage": True,
                                            "first_mutation_leaves_second_and_inputs_unchanged": True},
                      "scalar_output_fallback_call_shapes": calls,
                      "scope": "Four full source inputs compile. Exact original __call__ and _compress_rows AST; "
                      "own routing/flush and identity/scalar compressor recorders, no CT import or compression. "
                      "One local in-memory counterfactual removes only arrival clone, never applied upstream. "
                      "Real CPU tensors demonstrate lifetime, ownership, indivisible input cap and fallback call "
                      "order only; no original test bodies, real _compress_experts, packed path, quantization "
                      "bytes, VeOmni/EP/broadcast, model/refit, GPU/NPU, timing or memory-peak benchmark."}, indent=2))


if __name__ == "__main__":
    main()
