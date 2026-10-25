"""Run #7608's original producer/consumer regressions with isolated CPU types.

Git sources: verl-project/verl, Apache-2.0, ByteDance contributors.
HTTP/preprocessing/Ray are the original test's deterministic substitutes.
An additional same-output recall control is not a trajectory-recovery test.
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
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock

from codetiming import Timer
import numpy as np
from pydantic import BaseModel
import pytest
from tensordict import TensorDict
import torch

VERSIONS = {"head": "e1e753776781129ae39de77a91fd2ee06ac74508",
            "ancestor": "ea53291385ce764019a2b40733605f21d8317583",
            "main": "3c6d9b43b67c74ec343d77874cd8fa6e841523da"}
AGENT = "verl/experimental/agent_loop/agent_loop.py"
REWARD = "verl/experimental/reward_loop/reward_loop.py"
TEST = "tests/experimental/reward_loop/test_disrm_reward_contract_on_cpu.py"
PRODUCER_TEST = "test_compute_score_disrm_returns_two_field_contract"
CONSUMER_TEST = "test_disrm_result_survives_agent_loop_consumer_indexing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes, consumer_asts = {}, {}

    def source(revision, path):
        raw = subprocess.check_output(["git", "show", revision + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[revision + ":" + path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=path)
        compile(tree, path, "exec")
        return tree

    def module(name):
        value = ModuleType(name)
        value.__path__ = []
        sys.modules[name] = value
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, value)
        return value

    def execute(nodes, namespace):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_definitions", "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

    def definitions(tree, names, namespace):
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        execute(nodes, namespace)

    def method_class(tree, cls_name, method_name, namespace):
        cls = next(node for node in tree.body if getattr(node, "name", None) == cls_name)
        method = next(node for node in cls.body if getattr(node, "name", None) == method_name)
        assert not method.decorator_list
        execute([ast.ClassDef(name=cls_name, bases=[], keywords=[], body=[method],
                             decorator_list=[], type_params=[])], namespace)
        return ast.dump(method, include_attributes=False)

    test_tree = source(VERSIONS["head"], TEST)
    assert "verl" not in sys.modules
    for name in ("verl", "verl.experimental", "verl.experimental.agent_loop", "verl.experimental.reward_loop"):
        module(name)
    results = {}
    for label, revision in VERSIONS.items():
        namespace = module("verl.protocol").__dict__
        namespace.update(torch=torch, np=np, TensorDict=TensorDict, dataclass=dataclass, field=field)
        definitions(source(revision, "verl/protocol.py"), {"DataProto", "DataProtoItem"}, namespace)
        agent = module("verl.experimental.agent_loop.agent_loop")
        agent.__dict__.update(torch=torch, np=np, TensorDict=TensorDict, BaseModel=BaseModel,
                              Any=Any, Optional=Optional, random=random,
                              DataProto=namespace["DataProto"], contextmanager=contextmanager, Timer=Timer)
        definitions(source(revision, "verl/utils/profiler/performance.py"), {"_timer", "simple_timer"},
                    agent.__dict__)
        tree = source(revision, AGENT)
        definitions(tree, {"AgentLoopMetrics", "AgentLoopOutput"}, agent.__dict__)
        consumer_asts[label] = method_class(tree, "AgentLoopWorker", "_compute_score", agent.__dict__)
        reward = module("verl.experimental.reward_loop.reward_loop")
        method_class(source(revision, REWARD), "RewardLoopWorker", "compute_score_disrm", reward.__dict__)
        tests = module("_original_disrm_" + label)
        execute(test_tree.body, tests.__dict__)
        parameter_mark = next(mark for mark in getattr(tests, PRODUCER_TEST).pytestmark
                              if mark.name == "parametrize")
        original_cases = {}
        for params in parameter_mark.args[1]:
            try:
                getattr(tests, PRODUCER_TEST)(*params)
                status = "passed"
            except AssertionError:
                assert label != "head"
                status = "expected_contract_assertion_failure"
            assert status == ("passed" if label == "head" else "expected_contract_assertion_failure")
            original_cases[params[0]] = status
        try:
            getattr(tests, CONSUMER_TEST)()
            status = "passed"
        except KeyError as error:
            assert label != "head" and error.args == ("reward_extra_info",)
            status = "expected_KeyError_reward_extra_info"
        assert status == ("passed" if label == "head" else "expected_KeyError_reward_extra_info")
        original_cases["consumer"] = status

        # Same fixed input as the original consumer test, with output retained
        # for inspection and a second call. No workflow/automatic retry added.
        producer = tests._make_worker("vllm")
        producer._post_request = AsyncMock(return_value={"data": [{"probs": [0.125, 0.875]}]})
        remote = SimpleNamespace(remote=producer.compute_score_disrm)
        consumer = object.__new__(agent.AgentLoopWorker)
        consumer.reward_loop_worker_handles = [SimpleNamespace(compute_score=remote)]
        consumer._compute_multi_modal_inputs = lambda output, input_ids: {}
        consumer._compute_position_ids = (
            lambda input_ids, attention_mask, mm, mm_processor_kwargs=None:
            torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
        )
        consumer._get_mm_processor_kwargs = lambda audio_data=None: {}
        output = agent.AgentLoopOutput(prompt_ids=[1, 2], response_ids=[3], response_mask=[1],
                                       metrics=agent.AgentLoopMetrics(), extra_fields={})
        first_error = None
        try:
            asyncio.run(consumer._compute_score([output], kwargs={}))
        except KeyError as error:
            assert error.args == ("reward_extra_info",) and label != "head"
            first_error = "reward_extra_info"
        after_first = {"reward_score": output.reward_score, "extra_fields": dict(output.extra_fields)}
        assert output.reward_score == 0.875 and producer._post_request.await_count == 1
        asyncio.run(consumer._compute_score([output], kwargs={}))
        after_second = {"reward_score": output.reward_score, "extra_fields": dict(output.extra_fields)}
        assert producer._post_request.await_count == 1 and after_second == after_first
        assert ("reward_extra_info" in output.extra_fields) == (label == "head")
        results[label] = {"original_cases": original_cases, "first_control_error": first_error,
                          "after_first": after_first, "after_second": after_second,
                          "post_request_await_count": producer._post_request.await_count}
    print(json.dumps({"versions": VERSIONS, "source_sha256": hashes, "results": results,
                      "consumer_method_ast_equal": len(set(consumer_asts.values())) == 1,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                      "numpy": np.__version__, "pytest": pytest.__version__,
                                      "system": platform.system(), "machine": platform.machine()},
                      "scope": "original AST methods/types/test bodies; original transport/preprocess fakes; no services"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
