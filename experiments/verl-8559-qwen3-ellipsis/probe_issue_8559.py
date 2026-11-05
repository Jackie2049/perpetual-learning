#!/usr/bin/env python3
"""Reproduce verl#8559 and measure how general the defect is.

The Qwen3 XML tool parser converts raw parameter text with ``ast.literal_eval``
and then serialises the whole parameter dict with ``json.dumps``.  ``literal_eval``
accepts Python literals that JSON has no representation for (``...``, tuples, sets,
bytes, complex).  Those reach ``json.dumps`` unchanged and raise ``TypeError``,
which propagates out of ``extract_tool_calls`` and drops the tool call.

Exit 0 = the defect is present exactly as described (at least one literal type
still crashes).  Exit 2 = every literal type round-trips (bug fixed upstream).
"""

import argparse
import json
import sys
import warnings

# Empty text repr for each Python literal ``ast.literal_eval`` accepts but JSON rejects.
NON_JSON_LITERALS = {
    "ellipsis": "...",
    "tuple": "(1, 2)",
    "set": "{1, 2}",
    "bytes": "b'abc'",
    "complex": "1+2j",
    "frozenset": "frozenset([1])",
}

# and a couple that JSON can represent, as positive controls
JSON_LITERALS = {
    "list": "[1, 2]",
    "dict": "{'a': 1}",
    "int": "7",
    "bool": "True",
    "none": "None",
    "string": "hello",
}


def make_tool(param_type: str):
    from verl.tools.schemas import (
        OpenAIFunctionParametersSchema,
        OpenAIFunctionPropertySchema,
        OpenAIFunctionSchema,
        OpenAIFunctionToolSchema,
    )

    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="list_tool",
            description="a tool used for testing",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={"items": OpenAIFunctionPropertySchema(type=param_type)},
                required=[],
            ),
        ),
    )


def run_case(raw_value: str, param_type: str = "array") -> dict:
    from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser

    parser = Qwen3XMLToolParser(tokenizer=None)
    tool = make_tool(param_type)
    call = f"list_tool><parameter=items>{raw_value}</parameter>"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = parser._parse_xml_function_call(call, [tool])
    except Exception as exc:  # noqa: BLE001 - the point is to classify any escape
        return {"raw": raw_value, "status": "raise", "error": f"{type(exc).__name__}: {exc}"}
    try:
        parsed = json.loads(res.arguments)
    except Exception as exc:  # noqa: BLE001
        return {"raw": raw_value, "status": "bad-json", "error": f"{type(exc).__name__}: {exc}"}
    return {"raw": raw_value, "status": "ok", "arguments": res.arguments, "value": parsed.get("items")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="inject a crash into the control cases")
    args = ap.parse_args()

    report = {"non_json": {}, "json_controls": {}}
    for name, raw in NON_JSON_LITERALS.items():
        report["non_json"][name] = run_case(raw)
    for name, raw in JSON_LITERALS.items():
        report["json_controls"][name] = run_case(raw)

    crashing = [k for k, v in report["non_json"].items() if v["status"] != "ok"]
    good_controls = [k for k, v in report["json_controls"].items() if v["status"] == "ok"]

    if args.selftest:
        # A control case that cannot crash would make "controls pass" vacuous.
        must_pass = {"list", "dict", "int", "bool", "none", "string"}
        assert set(good_controls) == must_pass, f"control cases failed: {set(must_pass) - set(good_controls)}"

    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"non-json literal types crashing: {crashing}")
    print(f"json control cases ok: {len(good_controls)}/{len(JSON_LITERALS)}")
    if crashing:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
