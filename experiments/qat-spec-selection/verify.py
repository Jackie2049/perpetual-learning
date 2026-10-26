"""Run upstream test bodies with their own fake Bridge fixture, not a native suite.

Usage: python verify.py /path/to/qat-spec-source
Inputs: old.py, new.py, test.py, fetched at the commits recorded below.
No Megatron, Bridge, TE, ModelOpt, model weights, or devices are initialized.
"""
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest

OLD = "cf6eb5181e17b4db67afff110eabc61daf61fb4d"
NEW = "8a697618897d473a43b6d9e83ba6c3341a23067b"
PRODUCTION_PATH = "verl/workers/engine/megatron/transformer_impl.py"
TEST_PATH = "tests/workers/test_megatron_qat_layer_spec_on_cpu.py"
EXPECTED_SHA256 = {
    "old.py": "ae9c0f67e92fd0c31cfcc68572e452047b61256abf4c999768c256d49b97acec",
    "new.py": "1c4a029de4e1248433260b3c84e508d0e9b589a563953750c6d2074d91ec57e3",
    "test.py": "9b2eed71d1f19031e3be6766cc5bbbf935c026285bd46b0c10deb55d17fe1f48",
}


def run(source_dir):
    inputs = {name: (source_dir / name).read_bytes() for name in ("old.py", "new.py", "test.py")}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()} == EXPECTED_SHA256
    test_tree = ast.parse(inputs["test.py"])
    # Omit only the heavy production import; inject the exact extracted helper.
    test_tree.body = [node for node in test_tree.body if not (
        isinstance(node, ast.ImportFrom)
        and node.module == "verl.workers.engine.megatron.transformer_impl"
    )]
    test_names = [node.name for node in test_tree.body
                  if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    result = {"python": platform.python_version(), "pytest": pytest.__version__,
              "scope": "original test bodies and original fake Bridge fixture; AST-extracted original helper; not native pytest or real backend",
              "old_commit": OLD, "new_commit": NEW,
              "production_path": PRODUCTION_PATH, "test_path": TEST_PATH,
              "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
              "runs": {}}
    for version in ("old", "new"):
        tree = ast.parse(inputs[version + ".py"])
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_use_qat_layer_spec")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), version, "exec"), namespace)
        exec(compile(test_tree, TEST_PATH, "exec"), namespace)
        outcomes = []
        for name in test_names:
            with pytest.MonkeyPatch.context() as monkeypatch:
                namespace["_fake_gpt_provider"].__wrapped__(monkeypatch)
                try:
                    namespace[name]()
                except AssertionError:
                    outcomes.append({"test": name, "outcome": "assertion_failed"})
                else:
                    outcomes.append({"test": name, "outcome": "passed"})
        result["runs"][version] = outcomes
    assert [r["outcome"] for r in result["runs"]["old"]] == ["passed", "passed", "assertion_failed"]
    assert all(r["outcome"] == "passed" for r in result["runs"]["new"])
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
