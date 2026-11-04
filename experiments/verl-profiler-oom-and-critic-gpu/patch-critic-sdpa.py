#!/usr/bin/env python3
"""Re-apply the local, window-only tree adaptation that `run-critic-pdbg.sh` depends on.

Why this exists
---------------
`load_valuehead_model` in `verl/utils/model.py` hardcodes
``attn_implementation="flash_attention_2"`` at both of its `from_pretrained` calls (two
occurrences at main `3fdf2b54`), and the only call site (the FSDP engine,
`transformer_impl.py:304`) does not pass that parameter. 4090-1 has no flash-attn installed,
so a run with a critic dies while building the value-head model. The Hydra override
`+actor_rollout_ref.model.override_config.attn_implementation=sdpa` covers actor/rollout
only and cannot reach the critic.

The previous round made this adaptation by hand in the server tree under `/tmp`. `/tmp` is
not durable and the tree is rebuilt from a fresh clone whenever it disappears, so the
adaptation is a script now: the critic window must not be lost to a hand edit that no
longer exists.

What it does
------------
Inside the `load_valuehead_model` block only, rewrite
    attn_implementation="flash_attention_2"
to
    attn_implementation="<impl>"          (default: sdpa)

Nothing outside that function is touched. Idempotent; `--check` reports state without
writing; `--revert` restores the literal.

Fail-loud, on purpose
---------------------
The anchor count inside the function must equal `--expect` (2 at `3fdf2b54`). If upstream
adds an `attn_implementation` parameter, drops one of the two calls, or renames the literal,
this script exits 2 instead of reporting success. A tree that claims to be patched but is
not would burn the whole GPU window, and a silently skipped patch is exactly the failure
mode this repository keeps finding in other people's code.

Boundary
--------
This verifies a *text* transformation only. Whether `sdpa` is a numerically acceptable
stand-in for the critic has NOT been measured -- that is one of the things the critic window
is for. Do not quote this script as runtime evidence.

Exit codes
----------
0  ok (apply/revert done, or already in the requested state); `--check`: already patched
1  `--check`: not patched yet
2  anchor shape changed (function missing, anchor count != --expect) -- inspect by hand
3  tree / file not found
4  `--selftest` failure
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

FUNC = "def load_valuehead_model("
FA2 = 'attn_implementation="flash_attention_2"'


def sha256_file(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def block_span(lines):
    """Return (start, end) line indices of the `load_valuehead_model` block."""
    starts = [i for i, ln in enumerate(lines) if ln.startswith(FUNC)]
    if not starts:
        return None
    start = starts[0]
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln[0].isspace():
            return start, j
    return start, len(lines)


def analyze(text, impl):
    """Classify the file: patched / unpatched / shape-changed. Pure function."""
    lines = text.splitlines(keepends=True)
    span = block_span(lines)
    if span is None:
        return {"state": "shape-changed", "reason": "def load_valuehead_model( not found",
                "func_count": 0, "anchors_in_block": 0, "impl_in_block": 0}
    start, end = span
    func_count = sum(1 for ln in lines if ln.startswith(FUNC))
    block = "".join(lines[start:end])
    target = f'attn_implementation="{impl}"'
    body = {
        "state": None,
        "func_count": func_count,
        "block": [start + 1, end],          # 1-based, inclusive
        "anchors_in_block": block.count(FA2),
        "impl_in_block": block.count(target),
        "anchors_outside_block": "".join(lines[:start] + lines[end:]).count(FA2),
        "anchor_lines": [i + 1 for i in range(start, end) if FA2 in lines[i]],
        "impl_lines": [i + 1 for i in range(start, end) if target in lines[i]],
        "impl": impl,
    }
    if func_count != 1:
        body["state"] = "shape-changed"
        body["reason"] = f"{func_count} definitions of load_valuehead_model"
    elif body["anchors_in_block"] == 0 and body["impl_in_block"] > 0:
        body["state"] = "patched"
    elif body["anchors_in_block"] > 0 and body["impl_in_block"] == 0:
        body["state"] = "unpatched"
    else:
        body["state"] = "shape-changed"
        body["reason"] = (f"mixed anchors: fa2={body['anchors_in_block']} "
                          f"{impl}={body['impl_in_block']}")
    return body


def rewrite(text, impl, direction):
    """direction: 'apply' (fa2 -> impl) or 'revert' (impl -> fa2). Returns (text, n)."""
    lines = text.splitlines(keepends=True)
    span = block_span(lines)
    if span is None:
        return text, 0
    start, end = span
    src, dst = ((FA2, f'attn_implementation="{impl}"') if direction == "apply"
                else (f'attn_implementation="{impl}"', FA2))
    n = 0
    for i in range(start, end):
        if src in lines[i]:
            lines[i] = lines[i].replace(src, dst)
            n += 1
    return "".join(lines), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", required=False,
                    help="verl tree root (the dir that contains verl/utils/model.py)")
    ap.add_argument("--impl", default="sdpa", help="replacement attn_implementation (default sdpa)")
    ap.add_argument("--expect", type=int, default=2,
                    help="expected anchor count inside the function at the pinned commit")
    ap.add_argument("--check", action="store_true", help="report only, never write")
    ap.add_argument("--revert", action="store_true", help="restore the flash_attention_2 literal")
    ap.add_argument("--json", action="store_true", help="machine-readable result on stdout")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if not args.tree:
        ap.error("--tree is required unless --selftest")
    model_py = Path(args.tree) / "verl" / "utils" / "model.py"
    if not model_py.is_file():
        out = {"state": "missing", "file": str(model_py)}
        print(json.dumps(out) if args.json else f"[critic-sdpa] missing: {model_py}")
        return 3

    text = model_py.read_text()
    before_sha = sha256_file(model_py)
    state = analyze(text, args.impl)
    state["file"] = str(model_py)
    state["sha256_before"] = before_sha

    def emit(rc, action):
        state["action"] = action
        state["sha256_after"] = sha256_file(model_py)
        if args.json:
            print(json.dumps(state, sort_keys=True))
        else:
            pre = state.get("pre") or {}
            print(f"[critic-sdpa] {action}: state={state['state']} "
                  f"anchors_before={pre.get('anchors_in_block', state.get('anchors_in_block'))} "
                  f"lines_before={pre.get('anchor_lines', state.get('anchor_lines'))} "
                  f"sha={state['sha256_after'][:12]}")
            if state.get("reason"):
                print(f"[critic-sdpa] reason: {state['reason']}")
        return rc

    if state["state"] == "shape-changed":
        return emit(2, "abort")

    if state["state"] == "unpatched" and state["anchors_in_block"] != args.expect:
        state["state"] = "shape-changed"
        state["reason"] = (f"anchor count {state['anchors_in_block']} != expected {args.expect} "
                           f"at pinned commit -- upstream shape changed, inspect by hand")
        return emit(2, "abort")

    if args.check:
        return emit(0 if state["state"] == "patched" else 1, "check")

    if args.revert:
        if state["state"] == "unpatched":
            return emit(0, "revert-noop")
        state["pre"] = {k: state[k] for k in ("anchors_in_block", "anchor_lines", "impl_in_block")}
        new_text, n = rewrite(text, args.impl, "revert")
        if n != state["impl_in_block"] or n == 0:
            state["state"] = "shape-changed"
            state["reason"] = f"revert touched {n} lines, expected {state['impl_in_block']}"
            return emit(2, "abort")
        model_py.write_text(new_text)
        state.update(analyze(new_text, args.impl))
        state["lines_rewritten"] = n
        return emit(0 if state["state"] == "unpatched" else 2, "revert")

    if state["state"] == "patched":
        return emit(0, "apply-noop")
    state["pre"] = {k: state[k] for k in ("anchors_in_block", "anchor_lines", "impl_in_block")}
    new_text, n = rewrite(text, args.impl, "apply")
    if n != args.expect:
        state["state"] = "shape-changed"
        state["reason"] = f"apply touched {n} lines, expected {args.expect}"
        return emit(2, "abort")
    model_py.write_text(new_text)
    state.update(analyze(new_text, args.impl))
    state["lines_rewritten"] = n
    return emit(0 if state["state"] == "patched" else 2, "apply")


# --------------------------------------------------------------------------- selftest

FIXTURE = '''\
def load_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code):
    from transformers import AutoModelForCausalLM, AutoModelForTokenClassification

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=local_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
        return model
    except BaseException:
        pass

    ori_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local_path,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )
    return ori_model


def other_function():
    # an identical-looking literal OUTSIDE the block must never be touched
    return dict(attn_implementation="flash_attention_2")
'''

FIXTURE_ONE_ANCHOR = FIXTURE.replace('''        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )
    return ori_model''', '''        trust_remote_code=trust_remote_code,
    )
    return ori_model''')

FIXTURE_NO_FUNC = "# nothing here\nattn_implementation=\"flash_attention_2\"\n"


def _run(args):
    return main(args)


def selftest():
    """Every injected fault must be caught; the clean fixture must round-trip byte-exact."""
    results = []

    def ck(name, got, want):
        results.append((name, got == want, got, want))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "verl" / "utils").mkdir(parents=True)
        model_py = root / "verl" / "utils" / "model.py"

        # 1. check on an unpatched clean tree -> 1, no write
        model_py.write_text(FIXTURE)
        orig_sha = sha256_file(model_py)
        ck("check_unpatched_rc", _run(["--tree", td, "--check"]), 1)
        ck("check_no_write", sha256_file(model_py) == orig_sha, True)

        # 2. apply -> 0, exactly two anchors rewritten, outside literal untouched
        ck("apply_rc", _run(["--tree", td]), 0)
        text = model_py.read_text()
        ck("apply_count", text.count('attn_implementation="sdpa"'), 2)
        ck("apply_outside_untouched", text.count(FA2), 1)   # the other_function one
        ck("apply_func_still_there", text.count(FUNC), 1)
        sha_after_apply = sha256_file(model_py)

        # 3. idempotent second apply -> 0, no further writes
        ck("apply_twice_rc", _run(["--tree", td]), 0)
        ck("apply_twice_idempotent", sha256_file(model_py) == sha_after_apply, True)

        # 4. check on a patched tree -> 0
        ck("check_patched_rc", _run(["--tree", td, "--check"]), 0)

        # 5. revert -> 0 and byte-identical to the fixture
        ck("revert_rc", _run(["--tree", td, "--revert"]), 0)
        ck("revert_byte_exact", model_py.read_text() == FIXTURE, True)
        ck("revert_sha_exact", sha256_file(model_py) == orig_sha, True)

        # 6. revert on a clean tree -> 0 (no-op)
        ck("revert_noop_rc", _run(["--tree", td, "--revert"]), 0)

        # 7. one anchor missing inside the block -> 2, and NOT written
        model_py.write_text(FIXTURE_ONE_ANCHOR)
        sha_one = sha256_file(model_py)
        ck("one_anchor_rc", _run(["--tree", td]), 2)
        ck("one_anchor_no_write", sha256_file(model_py) == sha_one, True)

        # 8. function missing -> 2
        model_py.write_text(FIXTURE_NO_FUNC)
        ck("no_func_rc", _run(["--tree", td]), 2)

        # 9. missing tree -> 3
        ck("missing_tree_rc", _run(["--tree", str(root / "nope")]), 3)

        # 10. three anchors -> 2 (the --expect guard, not just count==expect on the happy path)
        three = FIXTURE.replace('''    return ori_model''', '''    _ = dict(attn_implementation="flash_attention_2")
    return ori_model''')
        model_py.write_text(three)
        ck("three_anchor_rc", _run(["--tree", td]), 2)

    failed = [r for r in results if not r[1]]
    for name, ok, got, want in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    print(f"[selftest] {len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 4


if __name__ == "__main__":
    sys.exit(main())
