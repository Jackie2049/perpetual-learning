"""Run one original postprocess contract body against pinned base/head AST definitions.

No Megatron import, pytest collection, engine, fused kernel or training execution.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import torch

BASE = "8df5baeed0f5ecdd7f870d83bc3d79194816e2a9"
HEAD = "61064b0fab650a5f228ae73413b7f8885fd9fde1"
PRODUCTION = ["verl/models/mcore/mtp_patch.py", "verl/models/mcore/model_forward_fused.py",
              "verl/workers/engine/megatron/transformer_impl.py"]
TESTS = ["tests/utils/test_megatron_mtp_dcp.py", "tests/models/test_model_forward_fused.py"]
TEST_NAME = "test_postprocess_forwards_output_processor_without_labels"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    hashes, trees = {}, {}
    for ref, path in [(HEAD, p) for p in PRODUCTION + TESTS] + [(BASE, PRODUCTION[0])]:
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=args.repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        tree = ast.parse(raw)
        compile(tree, path, "exec")
        trees[ref, path] = tree
        hashes[f"{ref}:{path}"] = hashlib.sha256(raw).hexdigest()

    original_test = trees[HEAD, TESTS[0]]
    test_nodes = [n for n in original_test.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                  and n.name in ("_OutputLayer", "_model", TEST_NAME)]
    assert len(test_nodes) == 3
    results = {}
    for ref in (BASE, HEAD):
        production = trees[ref, PRODUCTION[0]]
        function = next(n for n in production.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_megatron_gptmodel_postprocess")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)], type_ignores=[])),
                     f"{ref}:original-postprocess", "exec"), namespace)
        test_namespace = {"torch": torch, "SimpleNamespace": SimpleNamespace,
                          "mtp_patch": SimpleNamespace(
                              _megatron_gptmodel_postprocess=namespace[function.name])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(test_nodes), type_ignores=[])),
                     "original-test-nodes", "exec"), test_namespace)
        try:
            test_namespace[TEST_NAME]()
        except AssertionError:
            status = "assertion_failed"
        else:
            status = "passed"
        results[ref] = {"test": TEST_NAME, "status": status}
    assert results[BASE]["status"] == "assertion_failed"
    assert results[HEAD]["status"] == "passed"
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_git": torch.version.git_version, "base": BASE, "head": HEAD,
                      "original_test_body": results, "source_sha256": hashes,
                      "scope": "One unchanged original test body plus original _model/_OutputLayer; "
                      "complete original postprocess function AST at base and head. CPU arange tensors; "
                      "the original test replaces output_layer/output_processor with recording callbacks. "
                      "Six full source inputs compile only. No native pytest/Megatron imports, gate or "
                      "fused-forward execution, labels-context fallback test, TP/SP, fused CE math, "
                      "MTP drafter/loss, model training, GPU/NPU or performance validation."}, indent=2))


if __name__ == "__main__":
    main()
