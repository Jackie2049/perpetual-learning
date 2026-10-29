"""Run two original concat test bodies with real CPU TensorDict 0.11.0, not TPU or TensorDict 0.15.1."""

import argparse
import ast
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types

import tensordict
import torch
from packaging.requirements import Requirement
from tensordict.tensorclass import NonTensorData

HEAD = "1ffe98bf14949f0ac1cc000facf056ec0b7808d9"
BASE = "a08ea02a784d7d4982b1d81df36d188a476192bb"
PRODUCTION = "verl/utils/tensordict_utils.py"
TESTS = "tests/test_protocol_v2_on_cpu.py"
NAMES = ["test_concat_tensordict", "test_concat_tensordict_keeps_the_first_metadata"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_repo", type=Path)
parser.add_argument("unimported_tensordict_0151", type=Path)
args = parser.parse_args()
assert tensordict.__version__ == "0.11.0"
hashes = {}


def source(rev, path):
    raw = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=args.source_repo,
                                  env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
    hashes[f"{rev}:{path}"] = hashlib.sha256(raw).hexdigest()
    compile(raw, path, "exec")
    return raw


test_tree = ast.parse(source(HEAD, TESTS))
nodes = [next(n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in NAMES]
assert all(not n.decorator_list for n in nodes)
results = {}
for label, rev in [("base", BASE), ("head", HEAD)]:
    original = types.ModuleType(f"original_tensordict_utils_{label}")
    # The original module imports only numpy/torch/tensordict/packaging/stdlib.
    exec(compile(source(rev, PRODUCTION), PRODUCTION, "exec"), original.__dict__)
    scope = {"tu": original, "torch": torch, "NonTensorData": NonTensorData, "copy": copy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), TESTS, "exec"), scope)
    passed = []
    for name in NAMES:
        scope[name]()
        passed.append(name)
    observations = {}
    for kind in ("regular", "nested"):
        inputs = []
        for peak in (17.9, 18.3):
            value = torch.zeros(2, 4) if kind == "regular" else torch.nested.as_nested_tensor(
                [torch.arange(3), torch.arange(5)], layout=torch.jagged)
            inputs.append(original.get_tensordict({"payload": value}, {"metrics": {"loss": 0.5, "peak": peak}}))
        output = original.concat_tensordict(inputs)
        raw_metadata = output.get("metrics")
        value = original.get(output, "metrics")
        assert isinstance(raw_metadata, NonTensorData) and value == {"loss": 0.5, "peak": 17.9}
        assert output.batch_size[0] == 4
        observations[kind] = {"batch_size": list(output.batch_size), "metadata_type": type(raw_metadata).__name__,
                              "metadata": value}
    results[label] = {"original_test_bodies_passed": passed, "branch_observations": observations}
assert results["base"] == results["head"]

unimported_meta = (args.unimported_tensordict_0151 / "tensordict-0.15.1.dist-info/METADATA").read_bytes()
requirements = [line.removeprefix("Requires-Dist: ") for line in unimported_meta.decode().splitlines()
                if line.startswith("Requires-Dist: torch")]
assert len(requirements) == 1
requirement = Requirement(requirements[0])
assert not requirement.specifier.contains(torch.__version__.split("+")[0])
lib_sources = {}
for label, path in [("0.11.0/tensorclass.py", Path(tensordict.__file__).parent / "tensorclass.py"),
                    ("0.15.1/tensorclass.py", args.unimported_tensordict_0151 / "tensordict/tensorclass.py")]:
    raw = path.read_bytes()
    tree = ast.parse(raw)
    compile(tree, label, "exec")
    lib_sources[label] = {"sha256": hashlib.sha256(raw).hexdigest(),
                          "cat_non_tensor_definitions": sum(isinstance(n, ast.FunctionDef) and n.name == "_cat_non_tensor"
                                                            for n in ast.walk(tree))}
assert "verl" not in sys.modules
print(json.dumps({"head": HEAD, "base": BASE, "python": platform.python_version(), "torch": torch.__version__,
                  "tensordict": tensordict.__version__, "source_sha256": hashes, "results": results,
                  "unexecuted_distribution": {"version": "0.15.1", "metadata_sha256": hashlib.sha256(unimported_meta).hexdigest(),
                                               "torch_requirement": requirements[0], "current_torch_satisfies": False},
                  "library_source_syntax_only": lib_sources,
                  "support_packages": {name: metadata.version(name) for name in
                                       ("numpy", "cloudpickle", "orjson", "packaging", "pyvers", "importlib_metadata", "zipp")},
                  "scope": "Two original test bodies, executed directly against both complete original utils modules and real CPU TensorDict 0.11.0. Both versions pass: no old-fails/new-passes fix evidence here. 0.15.1 metadata and source only; no valid import/test, version-bisect, full native pytest, BatchData/Ray/worker, training or TPU validation."}, indent=2))
