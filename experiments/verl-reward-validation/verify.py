"""Isolate #7217's unchanged regression with real CPU data containers.

Original fake remote handles/tests and selected definitions are loaded from
verl-project/verl Git blobs (Apache-2.0, ByteDance contributors). No Ray,
TransferQueue, model, tokenizer, network, or training stack is initialized.
"""

import __future__
import argparse
import ast
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import ModuleType
from typing import Any, Optional

from codetiming import Timer
import numpy as np
from pydantic import BaseModel
import pytest
from tensordict import TensorDict
import tensordict
import torch

VERSIONS = {
    "head": "e8508932b0a7ae99b97ec8c3512fddbadd35d9d0",
    "ancestor": "2781c1e405fa80871a838bfdf9495b412df19c35",
    "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da",
}
AGENT = "verl/experimental/agent_loop/agent_loop.py"
TQ = "verl/trainer/ppo/v1/agent_loop_tq.py"
TEST = "tests/experimental/agent_loop/test_async_reward_meta_info_on_cpu.py"
CASE = "test_async_reward_data_proto_preserves_validate_meta_info"
METHODS = {"_compute_multi_modal_inputs", "_compute_position_ids",
           "_get_mm_processor_kwargs", "_compute_score"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(revision, path):
        raw = subprocess.check_output(
            ["git", "show", revision + ":" + path], cwd=args.source_repo,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1"),
        )
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
    results = {}
    for label, revision in VERSIONS.items():
        module = ModuleType("_isolated_reward_validation_" + label)
        sys.modules[module.__name__] = module  # dataclass resolves the defining module
        namespace = module.__dict__
        namespace.update(torch=torch, np=np, TensorDict=TensorDict, dataclass=dataclass,
                         field=field, BaseModel=BaseModel, pytest=pytest, Any=Any,
                         Optional=Optional, random=random, contextmanager=contextmanager,
                         Timer=Timer)
        definitions(source(revision, "verl/protocol.py"), {"DataProto", "DataProtoItem"}, namespace)
        definitions(source(revision, "verl/utils/model.py"), {"compute_position_id_with_mask"}, namespace)
        definitions(source(revision, "verl/utils/profiler/performance.py"),
                    {"_timer", "simple_timer"}, namespace)
        agent_tree = source(revision, AGENT)
        definitions(agent_tree, {"AgentLoopMetrics", "AgentLoopOutput"}, namespace)
        worker = next(node for node in agent_tree.body if getattr(node, "name", None) == "AgentLoopWorker")
        methods = [node for node in worker.body if getattr(node, "name", None) in METHODS]
        assert {node.name for node in methods} == METHODS
        assert all(not node.decorator_list for node in methods)
        # Only the class shell is local; all four method bodies remain unchanged.
        execute([ast.ClassDef(name="AgentLoopWorker", bases=[], keywords=[],
                              body=methods, decorator_list=[], type_params=[])], namespace)
        definitions(test_tree, {"_FakeRemoteComputeScore", "_FakeRewardLoopWorkerHandle",
                                "_DummyAgentLoopWorker", CASE}, namespace)
        original_tests = {}
        for validate in (False, True):
            try:
                asyncio.run(namespace[CASE](validate))
                status = "passed"
            except TypeError as error:
                assert label != "head" and "unexpected keyword argument 'validate'" in str(error)
                status = "expected_signature_failure"
            assert status == ("passed" if label == "head" else "expected_signature_failure")
            original_tests[str(validate)] = status

        # A separate boundary control, not another upstream regression test:
        # kwargs survives as per-row fields, not as a general meta_info bag.
        handle = namespace["_FakeRewardLoopWorkerHandle"]()
        dummy = namespace["_DummyAgentLoopWorker"](handle)
        output = namespace["AgentLoopOutput"](
            prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 1],
            metrics=namespace["AgentLoopMetrics"](),
        )
        options = {"validate": True} if label == "head" else {}
        asyncio.run(dummy._compute_score([output], kwargs={"batch_marker": "row-field"}, **options))
        data = handle.compute_score.received_data
        assert data.non_tensor_batch["batch_marker"].tolist() == ["row-field"]
        assert data.meta_info == ({"validate": True} if label == "head" else {})
        assert output.reward_score == 1.0 and output.extra_fields["reward_extra_info"] == {}
        calls = {}
        for path, tree in ((AGENT, agent_tree), (TQ, source(revision, TQ))):
            calls[path] = [
                [keyword.arg for keyword in node.keywords]
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "_compute_score"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
            ]
            expected = [["kwargs", "validate"]] if label == "head" else [["kwargs"]]
            assert calls[path] == expected
        results[label] = {"original_two_parameter_cases": original_tests,
                          "control_meta_info": data.meta_info,
                          "control_non_tensor_marker": data.non_tensor_batch["batch_marker"].tolist(),
                          "control_reward_score": output.reward_score, "caller_keywords": calls}

    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "tensordict": tensordict.__version__,
                                      "system": platform.system(), "machine": platform.machine()},
                      "scope": "original AST definitions and original fake-remote regression; no Ray/TQ/trainer"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
