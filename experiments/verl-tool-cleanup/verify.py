"""Run original _call_tool CPU tests plus one isolated dual-failure control.

Pinned sources: verl-project/verl (Apache-2.0, ByteDance contributors).
All tools are original test fakes or a local recorder; no sandbox/network/code
execution, rollout, retry implementation or GPU work is performed.
"""

import __future__
import argparse
import ast
import asyncio
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable
import unittest

import pydantic

REVISION = "3c6d9b43b67c74ec343d77874cd8fa6e841523da"
TOOL_LOOP = "verl/experimental/agent_loop/tool_agent_loop.py"
TEST = "tests/experimental/agent_loop/test_call_tool_on_cpu.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    args = parser.parse_args()
    hashes = {}

    def source(path):
        raw = subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=args.source_repo,
                                      env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        hashes[path] = hashlib.sha256(raw).hexdigest()
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

    assert "verl" not in sys.modules
    for name in ("verl", "verl.tools", "verl.experimental", "verl.experimental.agent_loop"):
        module(name)
    schema = module("verl.tools.schemas")
    execute(source("verl/tools/schemas.py").body, schema.__dict__)
    functions = module("verl.tools.function_tool")
    functions.__dict__.update(dataclass=dataclass, Any=Any, Callable=Callable,
                              asyncio=asyncio, json=json, ToolResponse=schema.ToolResponse,
                              OpenAIFunctionToolSchema=schema.OpenAIFunctionToolSchema)
    wanted = {"FunctionTool", "normalize_function_tool_return", "_coerce_response"}
    nodes = [node for node in source("verl/tools/function_tool.py").body
             if getattr(node, "name", None) in wanted]
    assert len(nodes) == len(wanted)
    execute(nodes, functions.__dict__)
    tool_module = module("verl.experimental.agent_loop.tool_agent_loop")
    logger = logging.getLogger("isolated_call_tool")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    tool_module.__dict__.update(json=json, logger=logger, ToolResponse=schema.ToolResponse,
                               FunctionTool=functions.FunctionTool,
                               normalize_function_tool_return=functions.normalize_function_tool_return)
    loop_tree = source(TOOL_LOOP)
    cls = next(node for node in loop_tree.body if getattr(node, "name", None) == "ToolAgentLoop")
    method = next(node for node in cls.body if getattr(node, "name", None) == "_call_tool")
    execute([ast.ClassDef(name="ToolAgentLoop", bases=[], keywords=[], body=[method],
                         decorator_list=[], type_params=[])], tool_module.__dict__)
    tests = module("_original_call_tool_tests")
    execute(source(TEST).body, tests.__dict__)  # __name__ is not __main__
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.TestCallToolErrorHandling)
    case_names = [case.id().split(".")[-1] for case in suite]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    assert result.testsRun == 8 and result.wasSuccessful(), stream.getvalue()

    events = []

    class DualFailureTool(tests.FakeTool):
        async def create(self, create_kwargs=None):
            events.append("create")
            return await super().create(create_kwargs)

        async def execute(self, instance_id, parameters, **kwargs):
            events.append("execute")
            raise RuntimeError("execute-primary")

        async def release(self, instance_id):
            events.append("release")
            raise OSError("release-secondary")

    loop = tests._make_tool_agent_loop({"dual": DualFailureTool("dual")})
    try:
        asyncio.run(loop._call_tool(tests.FakeFunctionCall("dual", "{}"), {}, tests.FakeAgentData()))
        raise AssertionError("cleanup error unexpectedly became a tool response")
    except OSError as error:
        assert str(error) == "release-secondary"
        control = {"events": events, "raised": type(error).__name__, "message": str(error)}
    assert events == ["create", "execute", "release"]
    print(json.dumps({"revision": REVISION, "source_sha256": hashes,
                      "original_tests_passed": case_names, "dual_failure_control": control,
                      "environment": {"python": platform.python_version(), "pydantic": pydantic.__version__,
                                      "system": platform.system(), "machine": platform.machine()},
                      "scope": "original method/schema/test fakes; no FunctionTool branch, sandbox or agent runtime"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
