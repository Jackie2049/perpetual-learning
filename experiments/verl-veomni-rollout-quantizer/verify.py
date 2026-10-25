"""#8369 original wrapper test, static export edges, and one real CPU byte-view control.

Fixed upstream Git blobs only; no checkout edits or automatic fetch/install.
AST isolation skips optional VeOmni imports, model construction and distributed export.
Upstream source: verl-project/verl, Apache-2.0, ByteDance contributors.
"""
import __future__
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import torch

VERSIONS = {"base": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "head": "ca0aea265f4bc4dab927a106cc7f9d43fc201337"}
ENGINE = "verl/workers/engine/veomni/transformer_impl.py"
TEST = "tests/workers/test_veomni_rollout_quantizer_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, results, engine_classes = {}, {}, {}

    def read(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def definition(tree, name):
        return next(n for n in tree.body if getattr(n, "name", None) == name)

    def execute(nodes, ns):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), ns)

    base = VERSIONS["base"]
    base_engine = definition(read(base, "verl/workers/engine/base.py"), "BaseEngine")
    base_setter = definition(base_engine, "set_rollout_weight_quantizer")
    original_test = definition(read(VERSIONS["head"], TEST),
                               "test_rollout_weight_quantizer_wraps_the_exported_stream")
    for label, revision in VERSIONS.items():
        engine = definition(read(revision, ENGINE), "VeOmniEngine")
        engine_classes[label] = engine
        ns = {}
        execute([ast.ClassDef(name="BaseEngine", bases=[], keywords=[], body=[base_setter],
                              decorator_list=[])], ns)
        methods = [n for n in engine.body if getattr(n, "name", None) in
                   {"set_rollout_weight_quantizer", "_quantize_for_rollout"}]
        execute([ast.ClassDef(name="VeOmniEngine", bases=[ast.Name(id="BaseEngine", ctx=ast.Load())],
                              keywords=[], body=methods or [ast.Pass()], decorator_list=[])], ns)
        ns["transformer_impl"] = SimpleNamespace(VeOmniEngine=ns["VeOmniEngine"])
        execute([original_test], ns)
        try:
            ns[original_test.name]()
        except AttributeError as error:
            assert label == "base" and "_quantize_for_rollout" in str(error)
            results[label] = {"status": "expected_AttributeError", "message": str(error)}
        else:
            assert label == "head"
            results[label] = {"status": "passed"}

    # Static equality after removing exactly the two newly inserted wrappers.
    head_export = copy.deepcopy(definition(engine_classes["head"], "get_per_tensor_param"))

    class Unwrap(ast.NodeTransformer):
        count = 0
        edges = []

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "self._quantize_for_rollout":
                assert len(node.args) == 1 and not node.keywords
                self.count += 1
                self.edges.append(ast.unparse(node.args[0]))
                return node.args[0]
            return self.generic_visit(node)

    unwrapper = Unwrap()
    restored = unwrapper.visit(head_export)
    assert unwrapper.count == 2
    assert ast.dump(restored) == ast.dump(definition(engine_classes["base"], "get_per_tensor_param"))
    head_init = copy.deepcopy(definition(engine_classes["head"], "__init__"))
    assignments = [n for n in head_init.body if isinstance(n, ast.Assign)
                   and any(ast.unparse(t) == "self._rollout_weight_quantizer" for t in n.targets)]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) is None
    head_init.body.remove(assignments[0])
    assert ast.dump(head_init) == ast.dump(definition(engine_classes["base"], "__init__"))
    # Include the fixed initialization/dispatcher source in syntax/hash evidence only.
    read(base, "verl/workers/engine_workers.py")
    quantizer = definition(read(base, "verl/utils/qat/compressed_tensors.py"), "Fp8ConfigWeightQuantizer")
    ns = {"torch": torch}
    execute([quantizer], ns)
    exporter = ns["Fp8ConfigWeightQuantizer"]({"quant_method": "fp8"}, model_types=["deepseek_v4"])
    expert_name = "model.layers.0.mlp.experts.0.weight"
    dense_name = "model.layers.0.self_attn.q_proj.weight"
    packed = torch.tensor([-128, -1, 0, 127], dtype=torch.int8, device="cpu")
    dense = packed.clone()
    exported = dict(exporter(iter([(expert_name, packed), (dense_name, dense)])))
    expert = exported[expert_name]
    assert exporter.passthrough and expert.dtype == torch.uint8
    assert expert.tolist() == [128, 255, 0, 127] and expert.data_ptr() == packed.data_ptr()
    assert exported[dense_name] is dense and dense.dtype == torch.int8
    assert all(name not in sys.modules for name in ("verl", "veomni", "ray"))
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "original_test": results,
                      "static_export_edges": unwrapper.edges,
                      "export_restores_base_when_unwrapped": True,
                      "constructor_only_adds_quantizer_none": True,
                      "cpu_passthrough_control": {"input_dtype": str(packed.dtype),
                          "input_values": packed.tolist(), "expert_output_dtype": str(expert.dtype),
                          "expert_output_values": expert.tolist(), "expert_aliases_input": True,
                          "nonexpert_identity_preserved": True},
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "system": platform.system()},
                      "scope": "one unchanged original test body with engine shell, not original module imports; "
                               "two export edges and constructor static only; original FP8 quantizer class on "
                               "one CPU int8 byte-view input; no VeOmni/converter/DTensor/FP8 kernel, "
                               "distributed export, model init, refit or training"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
