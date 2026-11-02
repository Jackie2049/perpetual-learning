"""Minimal reproduction: rLLM Tool.use_async auto-detect never engages.

Against rllm-org/rllm main @ 4e828c8 (2026-10-08).

The docstring of Tool.__call__ and docs/api/tools.mdx both state:
    "If use_async is None (default), uses async_forward if implemented, otherwise forward"

Two claims are checked here:
  1) The signature default is False, not None -> the documented auto-detect
     branch is unreachable by default.
  2) Even when None is passed explicitly, has_async is computed in a way that
     is always True for every subclass, so a sync-only tool returns an
     un-awaited coroutine instead of its result.

Run with a Python >= 3.10 interpreter that can import rllm.tools.tool_base.
"""

import asyncio
import inspect
import sys

from rllm.tools.tool_base import Tool


class SyncOnlyTool(Tool):
    """A tool that only implements forward(), i.e. synchronous only."""

    def __init__(self):
        super().__init__(name="sync_only", description="synchronous only")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {"name": "sync_only", "description": "synchronous only", "parameters": {}},
        }

    def forward(self, x):
        return x * 2


class AsyncOnlyTool(Tool):
    """A tool that only implements async_forward(), i.e. asynchronous only."""

    def __init__(self):
        super().__init__(name="async_only", description="asynchronous only")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {"name": "async_only", "description": "asynchronous only", "parameters": {}},
        }

    async def async_forward(self, x):
        return x * 2


def main() -> int:
    failures = []

    sig = inspect.signature(Tool.__call__)
    default = sig.parameters["use_async"].default
    print(f"[probe] Tool.__call__ signature default: use_async={default!r}")
    if default is not False:
        failures.append("signature default is expected to be False in this snapshot")
    print("  -> documented 'default is None' does NOT match the signature default False")

    has_async_sync_only = (
        inspect.isfunction(SyncOnlyTool.async_forward) or SyncOnlyTool.async_forward is Tool.async_forward
    )
    print(f"[probe] has_async computed for a sync-only subclass: {has_async_sync_only}")
    if not has_async_sync_only:
        failures.append("has_async expression is not constantly True; premise changed upstream")
    print("  -> the expression is True for every subclass, so it cannot discriminate")

    tool = SyncOnlyTool()
    print(f"[probe] sync-only tool, default call t(3) -> {tool(3)!r}")
    assert tool(3) == 6, "baseline default path must still work (use_async=False)"

    result = tool(3, use_async=None)
    print(f"[probe] sync-only tool, documented auto-detect t(3, use_async=None) -> {type(result).__name__}")
    if not asyncio.iscoroutine(result):
        failures.append("expected an un-awaited coroutine; premise changed upstream")
    else:
        result.close()

    guarded = tool(3, use_async=True)
    print(f"[probe] sync-only tool, t(3, use_async=True) -> {type(guarded).__name__} (guard expected to raise)")
    if not asyncio.iscoroutine(guarded):
        failures.append("expected a coroutine instead of NotImplementedError; premise changed upstream")
    else:
        guarded.close()

    async_tool = AsyncOnlyTool()
    async_result = async_tool(3, use_async=None)
    awaited = asyncio.run(async_result) if asyncio.iscoroutine(async_result) else async_result
    print(f"[probe] async-only tool, auto-detect awaited -> {awaited!r}")

    if failures:
        print("\n[FAIL]")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[PASS] defect reproduced: auto-detect is unreachable and has_async is constant True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
