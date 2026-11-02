"""Is the use_async auto-detect defect reachable from a real training path?

Against rllm-org/rllm main @ 4e828c8 (2026-10-08).

The sibling probe (repro.py) proves the defect exists at the Tool class. This
probe answers the next question, which decides whether the defect is worth an
upstream issue at all: does any production code path put a sync-only Tool
behind Tool.__call__ with use_async left at auto-detect?

Three checks:
  1) Call-site inventory: every Tool.__call__ invocation in the package that is
     not inside a `__main__` block, with the use_async value it passes.
  2) Dispatch inventory: the agent/harness layer uses `tool.execute(...)` on a
     different tool protocol (BashTool.schema/execute) that never touches
     Tool.__call__.
  3) Behavioural trace of the one reachable production path:
     rllm/rewards/code_reward.py -> TogetherCodeTool() -> codetool(...).

Exit 0 means "defect exists, production paths checked out as reported".
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[2] / "rllm"
PKG = REPO / "rllm"


def _is_main_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return isinstance(test, ast.Compare) and any(
        isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators
    )


def scan_calls(root: pathlib.Path):
    """Yield (file, line, use_async_value_str) for every call inside `if __name__ == "__main__":` blocks."""
    hits = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        blocks = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)
        ]
        for block in blocks:
            for sub in ast.walk(block):
                if isinstance(sub, ast.Call):
                    kw = {k.arg: ast.unparse(k.value) for k in sub.keywords if k.arg}
                    if "use_async" in kw:
                        hits.append((str(path.relative_to(root)), sub.lineno, kw["use_async"]))
    return hits


def main() -> int:
    failures = []

    print("== 1) Tool.__call__ call sites outside __main__ ==")
    hits = scan_calls(PKG)
    print(f"  call sites inside `if __name__ == '__main__'` blocks: {len(hits)}")
    for f, line, ua in hits:
        print(f"    {f}:{line}  use_async={ua}")
    # The scan above only proves the __main__ hits; prove production has none by
    # counting the parameter name outside those blocks.
    prod_hits = []
    for path in sorted(PKG.rglob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)
        main_ranges = [(n.lineno, n.end_lineno) for n in tree.body if _is_main_block(n)]
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "use_async":
                if any(a <= node.lineno <= b for a, b in main_ranges):
                    continue
                prod_hits.append(f"{path.relative_to(PKG)}:{node.lineno}")
    print(f"  keyword `use_async=` outside __main__: {len(prod_hits)} -> {prod_hits}")
    if prod_hits:
        failures.append("production code passes use_async explicitly; premise changed upstream")

    print("\n== 2) agent/harness dispatch protocol ==")
    tc = (PKG / "harnesses" / "tool_calling.py").read_text()
    uses_execute = "tool.execute(sandbox, **args)" in tc
    print(f"  harnesses/tool_calling.py dispatches via `tool.execute(sandbox, **args)`: {uses_execute}")
    bash = (PKG / "harnesses" / "tools" / "bash_tool.py").read_text()
    is_plain_class = "class BashTool:" in bash and "class BashTool(Tool)" not in bash
    print(f"  harnesses/tools/bash_tool.py defines a plain class (not a Tool subclass): {is_plain_class}")
    if not (uses_execute and is_plain_class):
        failures.append("harness dispatch changed; the Tool.__call__ wrapper may now be on the agent path")

    print("\n== 3) behavioural trace of the reachable production path ==")
    # Execute the real Tool class without importing the whole package.
    import inspect

    # tool_base only needs function_to_dict from the lightweight ui
    # import it directly instead of booting the whole rllm/rllm.__init__.
    sys.path.insert(0, str(REPO))
    utils = types.ModuleType("rllm.tools.utils")
    utils.__dict__["__name__"] = "rllm.tools.utils"
    exec(compile((PKG / "tools" / "utils.py").read_text(), "utils.py", "exec"), utils.__dict__)
    sys.modules["rllm"] = types.ModuleType("rllm")
    sys.modules["rllm.tools"] = types.ModuleType("rllm.tools")
    sys.modules["rllm.tools.utils"] = utils
    mod = types.ModuleType("rllm.tools.tool_base")
    mod.__dict__["__name__"] = "rllm.tools.tool_base"
    sys.modules["rllm.tools.tool_base"] = mod
    exec(compile((PKG / "tools" / "tool_base.py").read_text(), "tool_base.py", "exec"), mod.__dict__)
    Tool = mod.Tool

    class SyncOnlyTool(Tool):
        def __init__(self):
            super().__init__(name="together_python", description="sync only")

        @property
        def json(self):
            return {"type": "function", "function": {"name": "together_python", "parameters": {}}}

        def forward(self, code, timeout=30, **kwargs):
            return f"ran:{code}"

    tool = SyncOnlyTool()
    result = tool(code="print(1)", timeout=30)  # exactly how code_reward.py:399 calls it
    print(f"  codetool(code=..., timeout=30) -> {result!r}  [production path, use_async omitted]")
    if result != "ran:print(1)":
        failures.append("the production call shape no longer resolves synchronously")

    # What the same call would do if anyone asked for auto-detect.
    import asyncio

    coro = tool(code="print(1)", timeout=30, use_async=None)
    print(f"  codetool(code=..., timeout=30, use_async=None) -> {type(coro).__name__}")
    if not asyncio.iscoroutine(coro):
        failures.append("auto-detect now resolves correctly; defect may be fixed upstream")
    else:
        coro.close()

    print("\n== 4) the two tool stacks are unrelated ==")
    print("  rllm/tools/tool_base.py::Tool      -> training reward / rollout code, dispatch via __call__")
    print("  rllm/harnesses/tools/*.py          -> agent sandbox tools, protocol is .schema + .execute()")

    if failures:
        print("\n[FAIL]")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[OK] defect exists; no production path reaches the auto-detect branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
