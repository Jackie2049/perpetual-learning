#!/usr/bin/env python3
"""TM-008 async branch: independent re-derivation of the headline numbers.

Reads the same console logs as make-summary-async.py but recomputes the
bimodality counts, per-run means and the pooled delta from scratch, so a bug in
the summary parser cannot silently pass. Exits non-zero on any mismatch with
summary.json.
"""
import json
import pathlib
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z0-9_/\.\+\-]+):(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))")
RUNS = ["a200", "a010", "a200b", "a010b"]
WINDOW = range(3, 15)
EXPECTED_POLL = {"a200": 2.0, "a010": 0.1, "a200b": 2.0, "a010b": 0.1}


def rows(path):
    out, resolved = {}, None
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if resolved is None and "'poll_interval':" in line:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                resolved = float(m.group(1))
        if "timing_s/step" not in line or "step:" not in line:
            continue
        rec = {k: float(v) for k, v in PAIR.findall(line)}
        if "step" in rec:
            out[int(rec["step"])] = rec
    return resolved, out


def main():
    raw = pathlib.Path("raw")
    fails = []
    per_run = {}
    for mode in RUNS:
        resolved, steps = rows(raw / f"rollout-poll-async-{mode}.txt")
        if resolved != EXPECTED_POLL[mode]:
            fails.append(f"{mode}: resolved poll {resolved} != {EXPECTED_POLL[mode]}")
        if sorted(steps) != list(range(1, 16)):
            fails.append(f"{mode}: steps seen {sorted(steps)} != 1..15")
        win = [steps[s] for s in WINDOW]
        gens = [r["timing_s/gen"] for r in win]
        step = [r["timing_s/step"] for r in win]
        per_run[mode] = {
            "poll": resolved,
            "n": len(win),
            "gen_mean": statistics.fmean(gens),
            "gen_min": min(gens),
            "gen_max": max(gens),
            "gen_hits_gt2": sum(1 for g in gens if g > 2.0),
            "step_mean": statistics.fmean(step),
            "resp_len": statistics.fmean(r["response_length/mean"] for r in win if "response_length/mean" in r),
            "aborted": statistics.fmean(r.get("response/aborted_ratio", 0.0) for r in win),
            "staleness": statistics.fmean(
                r.get("training/off_policy/trajectory_staleness/mean", -1) for r in win
            ),
            "num_turns": statistics.fmean(r.get("training/num_turns/mean", -1) for r in win),
            "gen_values": [round(g, 3) for g in gens],
        }

    summary = json.loads((pathlib.Path("summary.json")).read_text())
    for mode in RUNS:
        s = summary[mode]
        if abs(s["timing"]["gen"]["mean"] - per_run[mode]["gen_mean"]) > 1e-6:
            fails.append(f"{mode}: gen mean mismatch vs summary.json")
        if abs(s["timing"]["step"]["mean"] - per_run[mode]["step_mean"]) > 1e-6:
            fails.append(f"{mode}: step mean mismatch vs summary.json")

    a200 = statistics.fmean([per_run["a200"]["step_mean"], per_run["a200b"]["step_mean"]])
    a010 = statistics.fmean([per_run["a010"]["step_mean"], per_run["a010b"]["step_mean"]])
    g200 = statistics.fmean([per_run["a200"]["gen_mean"], per_run["a200b"]["gen_mean"]])
    g010 = statistics.fmean([per_run["a010"]["gen_mean"], per_run["a010b"]["gen_mean"]])
    raw_means_200 = [per_run["a200"]["step_mean"], per_run["a200b"]["step_mean"]]
    raw_means_010 = [per_run["a010"]["step_mean"], per_run["a010b"]["step_mean"]]
    sd200 = statistics.stdev(raw_means_200)
    sd010 = statistics.stdev(raw_means_010)
    se = (sd200**2 / 2 + sd010**2 / 2) ** 0.5

    print(f"{'run':7s} {'poll':>4s} {'gen mean':>9s} {'hits>2s':>7s} {'gen range':>16s} {'step mean':>9s} {'resp len':>8s} {'abort':>6s} {'stale':>5s}")
    for mode in RUNS:
        r = per_run[mode]
        print(
            f"{mode:7s} {r['poll']:>4} {r['gen_mean']:9.3f} {r['gen_hits_gt2']:>3d}/12 "
            f"{r['gen_min']:7.3f}~{r['gen_max']:.3f} {r['step_mean']:9.3f} "
            f"{r['resp_len']:8.1f} {r['aborted']:6.2f} {r['staleness']:5.2f}"
        )
    print()
    print(f"pooled poll=2.0: gen {g200:.3f}s step {a200:.3f}s")
    print(f"pooled poll=0.1: gen {g010:.3f}s step {a010:.3f}s")
    print(f"delta: gen {g010 - g200:+.3f}s ({(g010 / g200 - 1) * 100:+.1f}%), "
          f"step {a010 - a200:+.3f}s ({(a010 / a200 - 1) * 100:+.1f}%), "
          f"run-level SE {se:.3f}s ({(a010 - a200) / se:+.1f} SE)")
    print(f"per-run step deltas: {raw_means_010[0] - raw_means_200[0]:+.3f}s, {raw_means_010[1] - raw_means_200[1]:+.3f}s")

    if fails:
        print("\nFAIL:")
        for f in fails:
            print(" -", f)
        return 1
    print("\nOK: resolved poll per run, 15 steps each, summary.json means match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
