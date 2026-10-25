"""Run #8368's two unchanged CPU regression bodies against base and head.

Upstream definitions: verl-project/verl, Apache-2.0, ByteDance contributors.
AST loading bypasses Ray/package initialization and estimator registration only.
One issue-shaped six-token arithmetic control uses original estimators, not a
recipe/tokenizer/model or training run.
"""

import __future__
import argparse
import ast
import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
import pydantic
from tensordict import TensorDict
import tensordict
import torch


VERSIONS = {"base": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
            "head": "9cf35a1d3692e2680c96e5c594e204638fcf08bc"}
AGENT = "verl/experimental/agent_loop/agent_loop.py"
TEST = "tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py"
CASES = {"test_agent_loop_output_as_dict_scores_last_generated_token",
         "test_agent_loop_postprocess_scores_last_generated_token"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, results, algorithm_asts = {}, {}, {}

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

    def definitions(tree, names, namespace, strip_registration=False):
        nodes = [copy.deepcopy(node) for node in tree.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        if strip_registration:
            for node in nodes:
                node.decorator_list = []
        execute(nodes, namespace)

    test_tree = source(VERSIONS["head"], TEST)
    for label, revision in VERSIONS.items():
        module = ModuleType("_original_outcome_" + label)
        sys.modules[module.__name__] = module
        ns = module.__dict__
        ns.update(torch=torch, np=np, TensorDict=TensorDict, BaseModel=BaseModel, ConfigDict=ConfigDict,
                  dataclass=dataclass, field=field, Any=Any, Optional=Optional, random=random)
        definitions(source(revision, "verl/protocol.py"), {"DataProto", "DataProtoItem"}, ns)
        agent_tree = source(revision, AGENT)
        definitions(agent_tree, {"AgentLoopMetrics", "AgentLoopOutput", "_InternalAgentLoopOutput"}, ns)
        worker = next(node for node in agent_tree.body if getattr(node, "name", None) == "AgentLoopWorker")
        method = next(node for node in worker.body if getattr(node, "name", None) == "_postprocess")
        execute([ast.ClassDef(name="AgentLoopWorker", bases=[], keywords=[], body=[method],
                              decorator_list=[], type_params=[])], ns)
        definitions(test_tree, CASES | {"_to_internal", "_pad_1d"}, ns)
        tests = {}
        for name in sorted(CASES):
            try:
                ns[name]()
            except AssertionError:
                assert label == "base"
                tests[name] = "expected_assertion_failure"
            else:
                tests[name] = "passed"
            assert tests[name] == ("passed" if label == "head" else "expected_assertion_failure")

        # Original six-token issue geometry; not a tokenized SWE trajectory.
        output = ns["AgentLoopOutput"](prompt_ids=[101, 102], response_ids=[1, 2, 3, 4, 5, 6],
                                       response_mask=[1, 1, 0, 1, 1, 0], reward_score=1.0,
                                       metrics=ns["AgentLoopMetrics"]())
        fields = output.as_dict()
        rewards = fields["rm_scores"].unsqueeze(0)
        mask = fields["response_mask"].unsqueeze(0).float()
        functional = {"torch": torch}
        definitions(source(revision, "verl/utils/torch_functional.py"),
                    {"masked_sum", "masked_mean", "masked_var", "masked_whiten"}, functional)
        algos = source(revision, "verl/trainer/ppo/core_algos.py")
        names = {"compute_gae_advantage_return", "compute_reinforce_plus_plus_outcome_advantage"}
        algorithm_asts[label] = {node.name: ast.dump(node, include_attributes=False)
                                 for node in algos.body if getattr(node, "name", None) in names}
        original = {"torch": torch, "verl_F": SimpleNamespace(**functional)}
        definitions(algos, names, original, strip_registration=True)
        gae_adv, gae_returns = original["compute_gae_advantage_return"](
            rewards, torch.zeros_like(rewards), mask, gamma=1.0, lam=1.0)
        rf_adv, rf_returns = original["compute_reinforce_plus_plus_outcome_advantage"](
            rewards, mask, config=SimpleNamespace(gamma=1.0))
        expected = 0.0 if label == "base" else 1.0
        assert gae_returns[0, 0].item() == rf_returns[0, 0].item() == expected
        assert rewards.sum().item() == 1.0
        results[label] = {"original_tests": tests,
                          "control": {"rm_scores": rewards.tolist(), "response_mask": mask.tolist(),
                                      "reward_sum": rewards.sum().item(), "gae_returns": gae_returns.tolist(),
                                      "rf_returns": rf_returns.tolist(), "gae_advantages": gae_adv.tolist(),
                                      "rf_advantages": rf_adv.tolist()}}
    assert algorithm_asts["base"] == algorithm_asts["head"]
    assert "ray" not in sys.modules and "transformers" not in sys.modules
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "estimator_ast_unchanged": True,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "pydantic": pydantic.__version__, "numpy": np.__version__,
                                      "tensordict": tensordict.__version__, "system": platform.system()},
                      "scope": "two original test bodies/helper, real CPU tensors/containers/Pydantic; "
                               "original method/class shells; two original estimators with registration bypassed; "
                               "gamma-only local config control; no full pytest/recipe/tokenizer/rollout/trainer"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
