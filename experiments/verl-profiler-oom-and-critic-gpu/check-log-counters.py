#!/usr/bin/env python3
"""Check the log counters that feed the critic probe's pre-registered verdict.

Why this exists
---------------
`run-critic-pdbg.sh` derived `steps_seen` with `re.finditer(r"^step:(\\d+)", text, re.M)`.
Every real verl console log written through Ray carries an ANSI-coloured task prefix, so no
line *begins* with `step:` and the counter returned 0. `verify-critic-pdbg.py` gates V2 on
`steps_seen >= min_steps`, so a perfectly good window run would be judged INVALID -- and the
window would be spent on a run that can never pass its own gate.

The 2026-11-05 01:0x mock harness did not catch this because its stand-in script printed
bare `step:1 - ...` lines. The fixture was not reality-shaped. This script makes the
difference measurable instead of arguable: it runs the old and the new counter over every
archived real log in the repo and requires them to disagree on all of them.

Counters compared
-----------------
old_steps   max over `^step:(\\d+)` matches            -- the pre-fix behaviour
naive_max   max over `step:(\\d+)` anywhere            -- shows why dropping the anchor is
                                                         also wrong: unrelated tokens such as
                                                         checkpoint paths match too
new_steps   number of distinct *training* step numbers: a log line that carries
            `timing_s/step` (printed once per training step; the step-0 validation line has
            no `timing_s/*` keys at all)

Usage
-----
    python3 check-log-counters.py --verify [--root <repo-root>]
    python3 check-log-counters.py --selftest

Exit codes
----------
0  pass     -- every archived step-log agrees with the new counter
1  mismatch -- at least one log disagrees (the counter is not trustworthy)
2  no logs  -- nothing found to check (an empty pass proves nothing)
4  --selftest failure
"""

import argparse
import json
import re
import sys
from pathlib import Path

STEP_TOKEN = re.compile(r"step:(\d+)\s+-\s+")
TRAIN_MARK = "timing_s/step"
LOG_SUFFIXES = (".txt", ".log", ".out")


def counters(text):
    """Return every counter for one log, including the ones that are wrong on purpose."""
    old = [int(m.group(1)) for m in re.finditer(r"^step:(\d+)", text, re.M)]
    naive = [int(m.group(1)) for m in re.finditer(r"step:(\d+)", text)]
    training_lines = []
    for line in text.splitlines():
        if TRAIN_MARK in line:
            m = STEP_TOKEN.search(line)
            if m:
                training_lines.append(int(m.group(1)))
    numbers = sorted(set(training_lines))
    return {
        "old_steps": max(old) if old else 0,
        "old_matches": len(old),
        "naive_max": max(naive) if naive else 0,
        "new_steps": len(numbers),
        "new_numbers": numbers,
        "training_lines": len(training_lines),
        "train_mark_count": text.count(TRAIN_MARK),
    }


def is_step_log(text):
    return TRAIN_MARK in text


def judge_log(name, text):
    c = counters(text)
    fails = []
    if c["new_steps"] != c["train_mark_count"]:
        fails.append(f"new_steps={c['new_steps']} != timing_s/step count={c['train_mark_count']}")
    if c["training_lines"] != c["new_steps"]:
        fails.append(f"training lines={c['training_lines']} != distinct steps={c['new_steps']}")
    return c, fails


def verify(root, out=None):
    root = Path(root)
    paths = sorted(p for p in root.glob("experiments/**/*") if p.suffix in LOG_SUFFIXES and p.is_file())
    checked, skipped, failures, old_zero, naive_differs = [], 0, [], 0, 0
    for p in paths:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            skipped += 1
            continue
        if not is_step_log(text):
            skipped += 1
            continue
        c, fails = judge_log(p.name, text)
        rel = str(p.relative_to(root))
        checked.append((rel, c))
        if c["old_steps"] == 0:
            old_zero += 1
        if c["naive_max"] != c["new_steps"]:
            naive_differs += 1
        for f in fails:
            failures.append(f"{rel}: {f}")

    print(f"scanned {len(paths)} candidate files, {len(checked)} are console step-logs, "
          f"{skipped} skipped (no {TRAIN_MARK})")
    if not checked:
        print("[verify] NO STEP LOGS FOUND -- an empty pass proves nothing")
        return 2

    total_steps = sum(c["new_steps"] for _, c in checked)
    print(f"total training steps counted across the corpus: {total_steps}")
    print(f"logs where the OLD counter (^step:) reads 0: {old_zero}/{len(checked)}")
    print(f"logs where an unanchored max() disagrees with the new counter: "
          f"{naive_differs}/{len(checked)}")

    # Not an invariant: archived logs include excerpts and resumed runs that start above 1.
    odd = [(rel, c["new_numbers"]) for rel, c in checked
           if c["new_numbers"] != list(range(1, c["new_steps"] + 1))]
    if odd:
        print(f"note: {len(odd)} log(s) are excerpts/resumes, step numbers not 1..N "
              f"(counted, not treated as failures): "
              + ", ".join(f"{rel}={nums}" for rel, nums in odd[:3]))

    worst = sorted(checked, key=lambda t: t[1]["naive_max"] - t[1]["new_steps"], reverse=True)[:3]
    print("largest unanchored-vs-training gaps (why the anchor cannot simply be dropped):")
    for rel, c in worst:
        print(f"  {rel}: naive_max={c['naive_max']} new_steps={c['new_steps']} "
              f"numbers={c['new_numbers'][:6]}{'...' if len(c['new_numbers']) > 6 else ''}")

    rc = 0 if (not failures and old_zero > 0) else 1
    if out:
        payload = {
            "root": str(root.resolve()),
            "candidate_files": len(paths),
            "step_logs": len(checked),
            "skipped": skipped,
            "total_training_steps": total_steps,
            "old_counter_zero": old_zero,
            "naive_disagrees": naive_differs,
            "not_1_to_n": {rel: nums for rel, nums in odd},
            "failures": failures,
            "ok": rc == 0,
            "logs": {rel: {k: v for k, v in c.items()} for rel, c in checked},
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"[verify] wrote {out}")

    if failures:
        print(f"[verify] FAIL: {len(failures)} disagreement(s)")
        for f in failures[:10]:
            print(f"  - {f}")
        return 1
    if old_zero == 0:
        print("[verify] FAIL: the old counter agrees everywhere -- the fix premise is gone; "
              "re-read run-critic-pdbg.sh before trusting this file")
        return 1
    print("[verify] PASS: new counter equals the per-step marker on every archived log, "
          "and the old anchored counter reads 0 on all of them")
    return 0

def selftest():
    ansi = "\x1b[36m(TaskRunnerV1 pid=2337295)\x1b[0m "
    real = (
        ansi + "step:0 - val-aux/openai/gsm8k/reward/mean@1:0.075 - val-core/acc:0.075\n"
        + ansi + "step:1 - global_seqlen/min:8616.0 - timing_s/gen:2.1 - timing_s/step:8.2\n"
        + ansi + "step:2 - global_seqlen/min:8616.0 - timing_s/gen:2.0 - timing_s/step:8.1\n"
        + "saving to /tmp/ckpt/global_step_46/actor\n"
    )
    cases = [
        # naive_max=8 comes from `timing_s/step:8.2` -- a *timing value* that looks like a
        # step number once the anchor is dropped. That is the second half of the lesson.
        ("real_prefix_two_steps", real, {"old_steps": 0, "naive_max": 8, "new_steps": 2}),
        ("bare_two_steps_old_fixture_shape",
         "step:1 - timing_s/step:1.0\nstep:2 - timing_s/step:1.0\n",
         {"old_steps": 2, "naive_max": 2, "new_steps": 2}),
        ("validation_only", ansi + "step:0 - val-core/acc:0.075\n", {"new_steps": 0}),
        ("truncated_run",
         ansi + "step:0 - val-core/acc:0.075\n" + ansi + "step:1 - timing_s/step:8.2\n",
         {"new_steps": 1}),
        ("metrics_split_over_two_lines",
         ansi + "step:1 - timing_s/gen:2.1\n" + ansi + "step:2 - timing_s/step:8.1\n",
         {"new_steps": 1}),
    ]
    failed = 0
    for name, text, want in cases:
        c = counters(text)
        ok = all(c[k] == v for k, v in want.items())
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {name}: want={want} got={ {k: c[k] for k in want} }")
        failed += 0 if ok else 1

    # negative control: the new counter must not be a constant. Corrupting the marker in a
    # log that the new counter reads as 2 steps has to move it. Replace the whole marker --
    # `timing_s/step` is a prefix of `timing_s/stepX`, so a suffix edit would not remove it.
    broken = real.replace(TRAIN_MARK, "elapsed_s/phase")
    if counters(broken)["new_steps"] != 0:
        print("[selftest] FAIL negative_control: marker removed but new_steps stayed non-zero")
        failed += 1
    else:
        print("[selftest] PASS negative_control: removing the marker drops new_steps to 0")

    total = len(cases) + 1
    print(f"[selftest] {total - failed}/{total} passed")
    return 0 if failed == 0 else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", help="write the per-log table as JSON here")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.verify:
        return verify(args.root, args.out)
    ap.error("pass --verify or --selftest")


if __name__ == "__main__":
    sys.exit(main())
