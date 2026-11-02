#!/usr/bin/env python3
"""TM-008 separate_async branch: parse the 2.0 / 0.1 poll A/B logs into summary.json.

Usage: python3 make-summary-separate.py [raw_dir]

Log format is the same console line per step: ``key:value - key:value ...``.
separate_async differs from colocate_async in one respect that matters for the
poll question: generation runs in a standalone rollout pool, so ``timing_s/gen``
is the trainer's wait for a trainable batch, not generation time on shared GPUs.
"""
import json
import pathlib
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# separate_async prints some counters as np.float64(1.0) / np.int64(2) reprs.
PAIR = re.compile(
    r"([A-Za-z0-9_/\.\+\-]+):(?:np\.float64\(|np\.int64\()?"
    r"(-?(?:\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?|nan|inf))"
)
PHASES = ["gen", "old_log_prob", "adv", "update_actor", "update_weights"]
NUM_KEYS = [
    "critic/rewards/mean",
    "response_length/mean",
    "response/aborted_ratio",
    "training/off_policy/trajectory_spans/mean",
    "training/off_policy/trajectory_staleness/mean",
    "perf/throughput",
]
RUNS = [
    ("sa20a", "2 train (4,5) + 1 rollout (1), patched main tree, poll_interval=2.0, run 1"),
    ("sa01a", "2 train (4,5) + 1 rollout (1), patched main tree, poll_interval=0.1, run 1"),
    ("sa20b", "2 train (4,5) + 1 rollout (1), patched main tree, poll_interval=2.0, run 2"),
    ("sa01b", "2 train (4,5) + 1 rollout (1), patched main tree, poll_interval=0.1, run 2"),
]
WINDOW = range(3, 9)  # 8 steps; step 1 carries warmup, step 2 the first sync boundary
STEP_KEYS = ["timing_s/" + p for p in PHASES] + ["timing_s/step"] + NUM_KEYS


def parse(path: pathlib.Path):
    steps = {}
    resolved = None
    logged = 0
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if resolved is None and "'poll_interval':" in line:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                resolved = float(m.group(1))
        if "step:" not in line or "timing_s/step" not in line:
            continue
        logged += 1
        rec = {}
        for key, value in PAIR.findall(line):
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        if "timing_s/step" in rec and "step" in rec:
            steps[int(rec["step"])] = rec
    return resolved, logged, steps


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


def compare(a, b):
    """mean(a) - mean(b) with a Welch-style SE from the per-step samples."""
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = statistics.variance(a) if len(a) > 1 else 0.0
    vb = statistics.variance(b) if len(b) > 1 else 0.0
    se = (va / len(a) + vb / len(b)) ** 0.5
    return {
        "a": ma,
        "b": mb,
        "delta": ma - mb,
        "ratio": ma / mb if mb else None,
        "se": se,
        "t": (ma - mb) / se if se else None,
    }


def pick(steps, keys, window):
    values = {k: [] for k in keys}
    for step in sorted(steps):
        if step not in window:
            continue
        for key in keys:
            if key in steps[step]:
                values[key].append(steps[step][key])
    return values


def main():
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    out = {"engine": "verl 0.10.0.dev (main 3fdf2b54 + TM-008 patch) + vLLM 0.24.0",
           "host": "4090-1, train GPUs 4,5 (same socket) + rollout GPU 1",
           "model": "Qwen2.5-0.5B-Instruct, gsm8k train_sub480/test_sub120",
           "window_steps": [WINDOW.start, WINDOW.stop - 1], "runs": {}}
    per_run = {}
    for name, desc in RUNS:
        path = raw / f"{name}.txt"
        if not path.exists():
            out["runs"][name] = {"missing": str(path)}
            continue
        resolved, logged, steps = parse(path)
        vals = pick(steps, STEP_KEYS, WINDOW)
        per_run[name] = vals
        out["runs"][name] = {
            "desc": desc,
            "resolved_poll_interval": resolved,
            "logged_step_lines": logged,
            "steps_parsed": sorted(steps),
            "stats": {k: stats(v) for k, v in vals.items()},
        }
    # pooled comparison per poll value
    pooled = {}
    for poll in ("2.0", "0.1"):
        names = [n for n, d in RUNS if (poll in d.split("poll_interval=")[1][:3])]
        merged = {k: [] for k in STEP_KEYS}
        for n in names:
            for k, v in per_run.get(n, {}).items():
                merged[k].extend(v)
        pooled[poll] = merged
        out.setdefault("pooled", {})[poll] = {k: stats(v) for k, v in merged.items() if v}
    if all(pooled.get(p) for p in ("2.0", "0.1")):
        out["delta"] = {k: compare(pooled["2.0"][k], pooled["0.1"][k]) for k in STEP_KEYS
                        if pooled["2.0"].get(k) and pooled["0.1"].get(k)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
