"""PR #8479: original async state-machine/CT code, attribute-only output seams.

python verify.py /path/to/pre-cached/verl-git
No Pydantic validation, tokenizer, engine, Ray, tool execution or device work.
"""

import __future__
import ast
import asyncio
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import io
import json
import logging
import os
import platform
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Literal
import unittest
from uuid import uuid4

HEAD = "1fcab32aecf3a7dc9cc4198fc31504df2dd11f0f"
BASE = "8df5baeed0f5ecdd7f870d83bc3d79194816e2a9"
LOOP = "verl/experimental/agent_loop/tool_agent_loop.py"
AGENT = "verl/experimental/agent_loop/agent_loop.py"
CT = "verl/utils/tokenizer/continuous_token.py"
TEST = "tests/experimental/agent_loop/test_tool_agent_loop_empty_response_on_cpu.py"


def token_output(**kwargs):
    # Explicit attribute carrier, not original TokenOutput/Pydantic validation.
    return SimpleNamespace(**dict(dict(log_probs=None, routed_experts=None, stop_reason=None,
                                       num_preempted=None, extra_fields={}), **kwargs))


def main(cache):
    hashes = {}
    sources = {}

    def read(ref, path):
        key = f"{ref}:{path}"
        if key not in sources:
            raw = subprocess.check_output(["git", "-C", cache, "show", key],
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
            sources[key] = raw.decode()
            hashes[key] = hashlib.sha256(raw).hexdigest()
        return ast.parse(sources[key], filename=key)

    def execute(nodes, namespace):
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, "original_ast", "exec", flags=__future__.annotations.compiler_flag), namespace)

    def find(tree, name):
        return next(node for node in tree.body if getattr(node, "name", None) == name)

    merge_base = subprocess.check_output(["git", "-C", cache, "merge-base", HEAD, BASE],
                                        env=dict(os.environ, GIT_NO_LAZY_FETCH="1")).decode().strip()
    assert merge_base == BASE
    reports = {}
    namespaces = {}
    base_methods = {"_assert_mm_supported", "_get_mm_processor_kwargs", "process_multi_modal_info",
                    "build_sglang_video_payload", "ct_merge_assistant_token"}
    for ref in (BASE, HEAD):
        mod = ModuleType(f"_empty_response_{ref}")
        sys.modules[mod.__name__] = mod  # dataclass requires registered module
        ns = mod.__dict__
        ns.update(dataclass=dataclass, field=field, Enum=Enum, uuid4=uuid4, Literal=Literal, logging=logging,
                  logger=logging.getLogger("isolated_empty_response"),
                  TokenOutput=token_output, AgentLoopOutput=SimpleNamespace,
                  simple_timer=lambda *args, **kwargs: nullcontext())
        ct = read(ref, CT)
        constants = [node for node in ct.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
        execute([*constants, find(ct, "MergeResult"), find(ct, "ContinuousTokenBuilder")], ns)
        base = find(read(ref, AGENT), "AgentLoopBase")
        methods = [node for node in base.body if getattr(node, "name", None) in base_methods]
        assert len(methods) == len(base_methods)
        execute([ast.ClassDef(name="AgentLoopBase", bases=[], keywords=[], body=methods,
                              decorator_list=[])], ns)
        loop = read(ref, LOOP)
        cls = find(loop, "ToolAgentLoop")
        cls.decorator_list = []  # registration is not exercised
        run = find(cls, "run")
        run.decorator_list = []  # tracing wrapper is not exercised
        constants = [node for node in loop.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "SPEC_DECODE_EXTRA_KEYS"
                             for target in node.targets)]
        execute([*constants, find(loop, "AgentState"), find(loop, "AgentData"), cls], ns)
        test = read(HEAD, TEST)  # same new tests applied to both versions
        nodes = [node for node in test.body if not (
            isinstance(node, ast.ImportFrom) and node.module.startswith("verl."))]
        execute(nodes, ns)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ns["TestToolAgentLoopEmptyResponseOnCpu"])
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
        failures = [{"test": case.id().split(".")[-1], "traceback": trace} for case, trace in result.failures]
        assert not result.errors, stream.getvalue()
        assert result.testsRun == 2
        assert len(failures) == (1 if ref == BASE else 0), stream.getvalue()
        if ref == BASE:
            assert failures[0]["test"] == "test_empty_first_turn_keeps_prompt_out_of_response"
            assert "AssertionError" in failures[0]["traceback"]
        reports[ref] = dict(tests_run=2, passed=2-len(failures), failures=failures)
        namespaces[ref] = ns

    async def own_runs(ns):
        observed = []
        for ids, logp, reason, cap in (([], None, None, 2),
                                       ([21, 22, 23], [-.1, -.2, -.3], "aborted", 2),
                                       ([21, 22, 23], None, None, 8)):
            loop = ns["_make_loop"](token_output(token_ids=ids, log_probs=logp, stop_reason=reason),
                                    response_length=cap)
            result = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hi"}])
            expected = ids[:cap] if ids or ns is namespaces[HEAD] else ns["PROMPT_IDS"][:cap]
            assert result.prompt_ids == ns["PROMPT_IDS"] and result.response_ids == expected
            assert result.response_mask == [1] * min(len(ids), cap)
            assert result.response_logprobs == (logp[:cap] if logp else None)
            observed.append(dict(token_ids=ids, stop_reason=reason, cap=cap,
                                 response_ids=result.response_ids, mask=result.response_mask,
                                 logprobs=result.response_logprobs, num_turns=result.num_turns))
        return observed

    own = {ref: asyncio.run(own_runs(ns)) for ref, ns in namespaces.items()}
    # Own finalizer-only control: list length, not any(mask), defines response span.
    finalizers = {}
    for ref in (BASE, HEAD):
        run = find(find(read(ref, LOOP), "ToolAgentLoop"), "run")
        assignments = [node for node in run.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in {"response_ids", "prompt_ids"}
                               for target in node.targets)]
        assert len(assignments) == 2
        ns = {"agent_data": SimpleNamespace(prompt_ids=[11, 12, 91, 92], response_mask=[0, 0])}
        execute(assignments, ns)
        assert ns["prompt_ids"] == [11, 12] and ns["response_ids"] == [91, 92]
        finalizers[ref] = dict(prompt_ids=ns["prompt_ids"], response_ids=ns["response_ids"], mask=[0, 0])

    print(json.dumps(dict(base=BASE, head=HEAD, actual_merge_base=merge_base,
                         python=platform.python_version(), source_sha256=hashes,
                         original_unittest_bodies=reports, own_async_controls=own,
                         own_all_zero_mask_finalizer=finalizers,
                         scope="Original AgentData/AgentState/ToolAgentLoop class (registration and run tracing "
                         "decorators removed), five original base methods, full original CT base builder and "
                         "MergeResult; original two unittest bodies collected by unittest with original fakes. "
                         "TokenOutput and AgentLoopOutput are attribute-only seams, no Pydantic validation; "
                         "timer is nullcontext, base initialization bypassed by original test factory. "
                         "No real tokenizer/templates, postprocess, reward, Ray, engine, cancellation or training."), indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
