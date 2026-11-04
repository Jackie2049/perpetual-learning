"""Check whether verl's `tiled_mlp` option is reachable on current main.

The option was wired up by PR #4649 (merged 2026-01-04) in `verl/workers/fsdp_workers.py`.
That file was deleted by the engine refactor (#6067, 2026-04-20), and this script
checks whether the wiring came along into the new engine.

Everything here is static: AST + text, no verl import, no GPU. Snapshots used:

    transformer_impl_main_74543265.py     verl/workers/engine/fsdp/transformer_impl.py at main 74543265
    fsdp_workers_pre_refactor_63e8a0d1.py verl/workers/fsdp_workers.py at 63e8a0d1 (just before deletion)

Usage:
    python3 check_wiring.py            # exit 0 if every stated fact holds
    python3 check_wiring.py --json
"""

import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolved against this file so the script also runs when invoked by path.
CURRENT_ENGINE = os.path.join(HERE, "transformer_impl_main_74543265.py")
PRE_REFACTOR_WORKER = os.path.join(HERE, "fsdp_workers_pre_refactor_63e8a0d1.py")
CALL = "apply_monkey_patch"

# Facts this script claims. Each is a constant expectation, so a change upstream
# shows up as a failure instead of a silently different reading.
EXPECT_CURRENT_CALLS = 1        # exactly one production call site in the FSDP engine
EXPECT_CURRENT_FORWARDS = 0     # and it forwards no tiled_mlp value
EXPECT_PRE_FORWARDS = True      # the pre-refactor worker did forward it


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def call_sites(source, callee):
    """Every call whose callee name is `callee`, with its keyword argument names."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == callee:
            found.append(
                {
                    "line": node.lineno,
                    "keywords": [kw.arg for kw in node.keywords],
                }
            )
    return found


def mentions(source, needle):
    return [i + 1 for i, line in enumerate(source.splitlines()) if needle in line]


def main():
    current = read(CURRENT_ENGINE)
    pre = read(PRE_REFACTOR_WORKER)

    current_calls = call_sites(current, CALL)
    forwards_tiled = [
        call for call in current_calls if any(k and k.startswith("use_tiled_mlp") for k in call["keywords"])
    ]
    pre_calls = call_sites(pre, CALL)
    pre_forwards_tiled = [
        call for call in pre_calls if any(k and k.startswith("use_tiled_mlp") for k in call["keywords"])
    ]

    engine_mentions = mentions(current, "tiled_mlp")
    pre_mentions = mentions(pre, "tiled_mlp")
    # The pre-refactor file also carried the FSDP1 refusal; keep it visible because
    # the new engine builds the module before it branches on strategy == "fsdp"/"fsdp2".
    pre_has_fsdp1_guard = bool(re.search(r"if use_tiled_mlp and .*strategy == \"fsdp\"", pre))

    report = {
        "current_engine": {
            "file": CURRENT_ENGINE,
            "call_sites_of_apply_monkey_patch": current_calls,
            "lines_mentioning_tiled_mlp": engine_mentions,
            "forwards_use_tiled_mlp": len(forwards_tiled) > 0,
        },
        "pre_refactor_worker": {
            "file": PRE_REFACTOR_WORKER,
            "call_sites_of_apply_monkey_patch": pre_calls,
            "lines_mentioning_tiled_mlp": pre_mentions,
            "forwards_use_tiled_mlp": len(pre_forwards_tiled) > 0,
            "has_fsdp1_refusal": pre_has_fsdp1_guard,
        },
    }

    ok = len(current_calls) == EXPECT_CURRENT_CALLS
    ok = ok and (len(forwards_tiled) > 0) == bool(EXPECT_CURRENT_FORWARDS)
    ok = ok and (len(pre_forwards_tiled) > 0) == EXPECT_PRE_FORWARDS
    report["expected"] = {
        "current_call_sites": EXPECT_CURRENT_CALLS,
        "current_forwards_use_tiled_mlp": EXPECT_CURRENT_FORWARDS,
        "pre_refactor_forwards_use_tiled_mlp": EXPECT_PRE_FORWARDS,
    }
    report["wiring_is_lost_on_main"] = ok

    if "--json" in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"current engine {CURRENT_ENGINE}")
        print(f"  calls of {CALL}: {len(current_calls)} at lines {[c['line'] for c in current_calls]}")
        print(f"  keywords passed: {current_calls[0]['keywords'] if current_calls else None}")
        print(f"  lines mentioning tiled_mlp: {engine_mentions or 'none'}")
        print(f"  forwards use_tiled_mlp: {len(forwards_tiled) > 0}")
        print()
        print(f"pre-refactor worker {PRE_REFACTOR_WORKER}")
        print(f"  calls of {CALL}: {len(pre_calls)} at lines {[c['line'] for c in pre_calls]}")
        print(f"  lines mentioning tiled_mlp: {pre_mentions}")
        print(f"  forwards use_tiled_mlp: {len(pre_forwards_tiled) > 0}")
        print(f"  had FSDP1 refusal: {pre_has_fsdp1_guard}")
        print()
        print(f"wiring lost on main (as claimed): {ok}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
