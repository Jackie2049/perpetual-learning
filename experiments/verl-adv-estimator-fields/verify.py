"""Verify #7473's original CPU test bodies without initializing Ray or verl.

Upstream definitions are from verl-project/verl, Apache-2.0, ByteDance
contributors. Full original functions/classes and registry declarations are AST
loaded; test bodies are unchanged. V1 field selection is inspected, not run.
The single local control records forwarding identity, not validity filtering.
"""

import __future__
import argparse
import ast
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Optional

import numpy as np
from tensordict import TensorDict
import tensordict
import torch


VERSIONS = {"base": "b256ebf83b304d83be5c1207fdf6867c04a0d077",
            "head": "c9e1e9150e26d9bc109fe0b142891804885af66e",
            "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da"}
TRAINER = "verl/trainer/ppo/ray_trainer.py"
ALGOS = "verl/trainer/ppo/core_algos.py"
V1 = "verl/trainer/ppo/v1/trainer_base.py"
TEST = "tests/trainer/ppo/test_adv_estimator_kwargs_on_cpu.py"
SIGNATURE = "test_accepts_kwarg_detects_declared_and_catch_all_parameters"
CUSTOM = "test_custom_estimator_receives_non_tensor_batch"
FIXED = "test_estimator_with_a_fixed_signature_is_unaffected"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, results = {}, {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def execute(nodes, namespace):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    def definitions(tree, names, namespace):
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        execute(nodes, namespace)

    test_tree = source(VERSIONS["head"], TEST)
    for label, revision in VERSIONS.items():
        module = ModuleType("_original_adv_fields_" + label)
        sys.modules[module.__name__] = module
        ns = module.__dict__
        ns.update(torch=torch, np=np, TensorDict=TensorDict, dataclass=dataclass, field=field,
                  Any=Any, Optional=Optional, Callable=Callable, random=random, inspect=inspect)
        definitions(source(revision, "verl/protocol.py"), {"DataProto", "DataProtoItem"}, ns)

        core = ModuleType("_original_adv_registry_" + label)
        core.__dict__.update(Enum=Enum, Any=Any)
        algo_tree = source(revision, ALGOS)
        declarations = [node for node in algo_tree.body if isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)
                        and node.target.id == "ADV_ESTIMATOR_REGISTRY"]
        assert len(declarations) == 1
        execute(declarations, core.__dict__)
        definitions(algo_tree, {"AdvantageEstimator", "register_adv_est", "get_adv_estimator_fn"},
                    core.__dict__)
        ns.update(core_algos=core, AdvantageEstimator=core.AdvantageEstimator)
        trainer_tree = source(revision, TRAINER)
        trainer_names = {"compute_advantage"}
        if label == "head":
            trainer_names.add("_accepts_kwarg")
        else:
            assert not any(getattr(node, "name", None) == "_accepts_kwarg" for node in trainer_tree.body)
        definitions(trainer_tree, trainer_names, ns)
        definitions(test_tree, {"_batch", CUSTOM, FIXED} | ({SIGNATURE} if label == "head" else set()), ns)

        tests = {}
        for name in (CUSTOM, FIXED) + ((SIGNATURE,) if label == "head" else ()):
            try:
                ns[name]()
            except AssertionError:
                assert label != "head" and name == CUSTOM
                tests[name] = "expected_assertion_failure"
            else:
                tests[name] = "passed"
            expected = "expected_assertion_failure" if label != "head" and name == CUSTOM else "passed"
            assert tests[name] == expected

        # One local forwarding control, using the upstream two-row input unchanged.
        # The invalid row has zero reward in that fixture; retaining it is NOT a
        # numerical proof about statistical filtering or any policy gradient.
        control = None
        if label == "head":
            seen = {}

            @core.register_adv_est("local_forwarding_identity")
            def recorder(token_level_rewards, response_mask, index=None, config=None, **kwargs):
                seen.update(kwargs)
                values = token_level_rewards * response_mask
                return values, values

            data = ns["_batch"]()
            tensors, non_tensors = data.batch, data.non_tensor_batch
            output = ns["compute_advantage"](data, adv_estimator="local_forwarding_identity")
            assert output is data and seen["batch"] is tensors and seen["non_tensor_batch"] is non_tensors
            assert output.batch["advantages"].shape == (2, 2)
            assert list(seen["non_tensor_batch"]["sample_is_valid"]) == [True, False]
            control = {"same_data_object": True, "same_batch_object": True,
                       "same_non_tensor_object": True, "sample_is_valid": [True, False],
                       "output_rows": 2, "advantages": output.batch["advantages"].tolist()}

        v1_tree = source(revision, V1)
        methods = [node for node in ast.walk(v1_tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "_compute_advantage"]
        assert len(methods) == 1
        read_fields = ast.literal_eval(methods[0].body[1].value)
        assert read_fields == ["uid", "response_mask", "rm_scores", "rollout_log_probs",
                               "old_log_probs", "ref_log_prob", "values"]
        get_calls = [node for node in ast.walk(methods[0]) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "kv_batch_get"]
        assert len(get_calls) == 1
        assert any(kw.arg == "select_fields" and isinstance(kw.value, ast.Name) and kw.value.id == "fields"
                   for kw in get_calls[0].keywords)
        results[label] = {"original_tests": tests, "local_control": control,
                          "v1_source_only_read_fields": read_fields}

    assert "ray" not in sys.modules and "verl" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "tensordict": tensordict.__version__,
                                      "system": platform.system()},
                      "scope": "unchanged original test bodies; real CPU tensors/DataProto/TensorDict; "
                               "original enum/registry/caller; signature case head only; "
                               "one local identity control; V1 whitelist source only; "
                               "no full pytest/estimator filtering/TransferQueue/Ray/rollout/training"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
