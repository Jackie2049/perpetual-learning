#!/usr/bin/env python3
"""TM-008 async branch: parse colocate_async poll-interval A/B logs into summary.json.

Usage: python3 make-summary-async.py [raw_dir]

Same log format as the sync experiment: per-step metrics on one console line as
``key:value - key:value ...``. Steady-state window is steps 3-14 (step 1 carries
warmup; the last step carries the final checkpoint).
"""
import json
import pathlib
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z0-9_/\.\+\-]+):(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))")
PHASES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]
NUM_KEYS = [
    "critic/rewards/mean",
    "response_length/mean",
    "response/aborted_ratio",
    "response_length/clip_ratio",
    "training/off_policy/trajectory_spans/mean",
    "training/off_policy/trajectory_staleness/mean",
    "perf/throughput",
    "training/num_turns/mean",
]
RUNS = [
    ("a200", "2 GPU (4,5), patched tree, poll_interval=2.0, run 1"),
    ("a010", "2 GPU (4,5), patched tree, poll_interval=0.1, run 1"),
    ("a200b", "2 GPU (4,5), patched tree, poll_interval=2.0, run 2"),
    ("a010b", "2 GPU (4,5), patched tree, poll_interval=0.1, run 2"),
]
WINDOW = range(3, 15)


def parse(path: pathlib.Path):
    steps = {}
    resolved = None
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if resolved is None and "'poll_interval':" in line:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                resolved = float(m.group(1))
        if "step:" not in line or "timing_s/step" not in line:
            continue
        rec = {}
        for key, value in PAIR.findall(line):
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        if "timing_s/step" in rec and "step" in rec:
            steps[int(rec["step"])] = rec
    return resolved, steps


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def ratio(a, b, rel_se=None):
    """Ratio of two means (floats); pass rel_se when a list-based SE is available."""
    return {"a": a, "b": b, "delta": a - b, "ratio": a / b if b else None, "rel_se": rel_se}


def ratio_lists(a, b):
    """Ratio of two samples' means with a Welch-style SE on the ratio."""
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = statistics.stdev(a) if len(a) > 1 else 0.0
    sb = statistics.stdev(b) if len(b) > 1 else 0.0
    se = ((sa**2 / len(a)) + (sb**2 / len(b))) ** 0.5
    return ratio(ma, mb, se / mb if mb else None)


def main():
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    out = {}
    for mode, desc in RUNS:
        path = raw / f"rollout-poll-async-{mode}.txt"
        if not path.exists():
            print(f"missing {path}")
            continue
        resolved, steps = parse(path)
        window = [steps[s] for s in WINDOW if s in steps]
        timing = {key: stats([r[f"timing_s/{key}"] for r in window if f"timing_s/{key}" in r]) for key in PHASES + ["step"]}
        out[mode] = {
            "description": desc,
            "resolved_poll_interval": resolved,
            "steps_seen": sorted(steps),
            "window": [min(WINDOW), max(WINDOW)],
            "timing": timing,
            "gen_values": [r["timing_s/gen"] for r in window],
            "step_values": [r["timing_s/step"] for r in window],
            "numeric": {key: [r.get(key) for s, r in sorted(steps.items())] for key in NUM_KEYS},
        }

    # paired same-window comparison: a010 vs a200 (run 1), a010b vs a200b (run 2)
    for arm, ctrl in (("a010", "a200"), ("a010b", "a200b")):
        if arm in out and ctrl in out:
            out[arm]["vs_" + ctrl] = {
                key: ratio(out[arm]["timing"][key]["mean"], out[ctrl]["timing"][key]["mean"])
                if out[arm]["timing"][key] and out[ctrl]["timing"][key]
                else None
                for key in PHASES + ["step"]
            }
            out[arm]["vs_" + ctrl]["step_values"] = ratio_lists(out[arm]["step_values"], out[ctrl]["step_values"])

    # pooled arm comparison across the two repeats
    pooled = {}
    for arm in ("a200", "a010"):
        runs = [m for m in (arm, arm + "b") if m in out]
        if not runs:
            continue
        # run-level means (one value per run) keep repeats independent
        pooled[arm] = {
            "runs": runs,
            "run_means": {key: [out[m]["timing"][key]["mean"] for m in runs] for key in PHASES + ["step"]},
            "step_run_means": [out[m]["timing"]["step"]["mean"] for m in runs],
        }
    if "a200" in pooled and "a010" in pooled:
        cmp = {}
        for key in PHASES + ["step"]:
            cmp[key] = ratio_lists(pooled["a010"]["run_means"][key], pooled["a200"]["run_means"][key])
        cmp["step_stepwise"] = ratio_lists(
            [v for m in ("a010", "a010b") if m in out for v in out[m]["step_values"]],
            [v for m in ("a200", "a200b") if m in out for v in out[m]["step_values"]],
        )
        pooled["a010_vs_a200"] = cmp
    out["pooled"] = pooled

    text = json.dumps(out, indent=2, ensure_ascii=False)
    target = raw.parent / "summary.json"
    target.write_text(text + "\n")

    print("mode      poll  gen(s)            step(s)           n")
    for mode, desc in RUNS:
        if mode not in out:
            continue
        r = out[mode]
        print(
            f"{mode:8s} {r['resolved_poll_interval']!s:5s} "
            f"{r['timing']['gen']['mean']:.3f}±{r['timing']['gen']['sd']:.3f}  "
            f"{r['timing']['step']['mean']:.3f}±{r['timing']['step']['sd']:.3f}  {r['timing']['step']['n']}"
        )
    if "a010_vs_a200" in pooled:
        c = pooled["a010_vs_a200"]
        print(
            f"\npooled a010 vs a200: step {c['step']['delta']:+.3f}s ({(c['step']['ratio'] - 1) * 100:+.1f}%), "
            f"gen {c['gen']['delta']:+.3f}s ({(c['gen']['ratio'] - 1) * 100:+.1f}%)"
        )
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
