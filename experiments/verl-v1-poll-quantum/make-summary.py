#!/usr/bin/env python3
"""TM-008: parse the poll-interval A/B training logs into summary.json.

Usage: python3 make-summary.py [raw_dir]

Each log is a verl console log; per-step metrics appear on one line as
``key:value - key:value ...``. The steady-state window is steps 3-14 (step 1
carries warmup, the last step carries the final checkpoint).
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
    "actor/entropy",
    "actor/pg_loss",
    "actor/loss",
    "global_seqlen/mean",
    "response_length/mean",
    "training/rollout_probs_diff_mean",
]
RUNS = [
    ("pbase", "2 GPU, pristine tree, default 2.0s"),
    ("p200", "2 GPU, patched tree, poll_interval=2.0"),
    ("p050", "2 GPU, patched tree, poll_interval=0.5"),
    ("p010", "2 GPU, patched tree, poll_interval=0.1"),
    ("p4base", "4 GPU (0,1,4,5), pristine tree, default 2.0s"),
    ("p4p010", "4 GPU (0,1,4,5), patched tree, poll_interval=0.1"),
]
CONTROLS = {"p200": "pbase", "p050": "p200", "p010": "p200", "p4p010": "p4base"}
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


def main():
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    out = {}
    for mode, desc in RUNS:
        path = raw / f"rollout-poll-ab-{mode}.txt"
        if not path.exists():
            print(f"missing {path}")
            continue
        resolved, steps = parse(path)
        window = [steps[s] for s in WINDOW if s in steps]
        timing = {key: stats([r[f"timing_s/{key}"] for r in window]) for key in PHASES + ["step"]}
        phase_sum = statistics.fmean(sum(r[f"timing_s/{k}"] for k in PHASES) for r in window)
        out[mode] = {
            "description": desc,
            "resolved_poll_interval": resolved,
            "steps_seen": sorted(steps),
            "window": [min(WINDOW), max(WINDOW)],
            "timing": timing,
            "phase_sum_mean": phase_sum,
            "residual_mean": timing["step"]["mean"] - phase_sum,
            "gen_gt_3s": sum(1 for r in window if r["timing_s/gen"] > 3.0),
            "gen_values": [r["timing_s/gen"] for r in window],
            "step_values": [r["timing_s/step"] for r in window],
            "numeric": {key: [r.get(key) for s, r in sorted(steps.items())] for key in NUM_KEYS},
        }

    for mode, control in CONTROLS.items():
        if mode not in out or control not in out:
            continue
        a, b = out[control], out[mode]
        row = {}
        for key in PHASES + ["step"]:
            ca, cb = a["timing"][key]["mean"], b["timing"][key]["mean"]
            row[key] = {"control": ca, "mode": cb, "delta": cb - ca, "ratio": cb / ca}
        out[mode]["vs_control"] = {"control": control, "timing": row}

    text = json.dumps(out, indent=2, ensure_ascii=False)
    target = raw.parent / "summary.json"
    target.write_text(text + "\n")

    print("mode      gen(s)        slow  step(s)       vs control")
    for mode, desc in RUNS:
        if mode not in out:
            continue
        r = out[mode]
        vc = r.get("vs_control", {}).get("timing", {}).get("step")
        vg = r.get("vs_control", {}).get("timing", {}).get("gen")
        extra = ""
        if vc:
            extra = f" step {vc['delta']:+.3f}s ({(vc['ratio'] - 1) * 100:+.1f}%), gen {vg['delta']:+.3f}s"
        print(
            f"{mode:8s} {r['timing']['gen']['mean']:.3f}±{r['timing']['gen']['sd']:.3f} "
            f"{r['gen_gt_3s']:>3d}/{r['timing']['gen']['n']:<2d} "
            f"{r['timing']['step']['mean']:.3f}±{r['timing']['step']['sd']:.3f}{extra}"
        )
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
