#!/usr/bin/env python3
"""Parse candidate-G placement A/B/A/B repeat logs (2026-11-02).

Modes: pA* = GPUs 0,1,3,4 (NUMA0 x3 + NUMA1 x1, two PCIe roots)
       pB* = GPUs 4,5,6,7 (all NUMA1, one root)
Suffix digit = repeat index inside the same idle window.
p*3 modes are the taskset CPU/memory-locality controls (all CPUs on one node).

Usage: python3 parse-candidate-g-ab.py /tmp/rollout-g8-candidate-g-p{A,B}*.log --json-out ab-summary.json
"""
import argparse
import json
import re
import statistics

STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]


def load(path):
    rows = []
    for line in open(path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            d["gs"] = int(re.search(r"training/global_step:(\d+)", line).group(1))
            m = re.search(r"perf/throughput:([0-9.]+)", line)
            if m:
                d["throughput"] = float(m.group(1))
            rows.append(d)
    return rows


def steady(rows, lo=3):
    hi = max(r["gs"] for r in rows) - 1
    rs = [r for r in rows if lo <= r["gs"] <= hi]
    out = {}
    for k in STAGES + ["step", "throughput"]:
        vals = [r[k] for r in rs if k in r]
        if vals:
            out[k] = statistics.mean(vals)
            out[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    out["n"] = len(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    modes = {}
    for path in args.logs:
        label = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        m = re.search(r"-(p[A-Za-z0-9]+)$", label)
        name = m.group(1) if m else label
        modes[name] = steady(load(path))

    names = sorted(modes)
    print(f"{'mode':6s} {'step':>8s} {'sd':>7s} {'n':>3s} " +
          " ".join(f"{k:>11s}" for k in STAGES))
    for name in names:
        m = modes[name]
        print(f"{name:6s} {m['step']:8.3f} {m['step_sd']:7.3f} {m['n']:3d} " +
              " ".join(f"{m[k]:11.3f}" for k in STAGES))

    pairs = {}
    def ratio(a, b):
        if a not in modes or b not in modes:
            return None
        ma, mb = modes[a], modes[b]
        return {
            "a": a, "b": b,
            "step_a_s": round(ma["step"], 4), "step_b_s": round(mb["step"], 4),
            "step_ratio_b_over_a": round(mb["step"] / ma["step"], 4),
            "stage_ratio_b_over_a": {k: round(mb[k] / ma[k], 4) for k in STAGES},
        }
    for a, b in [("pA1", "pB1"), ("pA2", "pB2"), ("pA1", "pA2"), ("pB1", "pB2"),
                 ("pA1", "pA3"), ("pB1", "pB3")]:
        r = ratio(a, b)
        if r:
            pairs[f"{a}_vs_{b}"] = r
    # mean of the two A runs vs mean of the two B runs, when all four exist
    if all(k in modes for k in ("pA1", "pA2", "pB1", "pB2")):
        a_step = (modes["pA1"]["step"] + modes["pA2"]["step"]) / 2
        b_step = (modes["pB1"]["step"] + modes["pB2"]["step"]) / 2
        pairs["meanA_vs_meanB"] = {
            "step_a_s": round(a_step, 4), "step_b_s": round(b_step, 4),
            "step_ratio_b_over_a": round(b_step / a_step, 4),
            "stage_ratio_b_over_a": {
                k: round((modes["pB1"][k] + modes["pB2"][k]) /
                         (modes["pA1"][k] + modes["pA2"][k]), 4)
                for k in STAGES},
        }

    for key, v in pairs.items():
        print(f"{key}: step {v['step_a_s']}s -> {v['step_b_s']}s "
              f"ratio={v['step_ratio_b_over_a']:.3f} "
              f"update_actor={v['stage_ratio_b_over_a']['update_actor']:.3f}")

    summary = {"meta": {
        "executed_utc": "2026-11-02",
        "placement_a": "GPUs 0,1,3,4 (NUMA0 x3 + NUMA1 x1; PCIe roots 4a/4a/4a/c9)",
        "placement_b": "GPUs 4,5,6,7 (all NUMA1; single PCIe root c9)",
        "window_rule": "steps 3 .. max_step-1",
        "config": "same as candidate G strong scaling: batch 32 x n=4, 15 iters, seed 42",
    }, "modes": {n: {k: round(v, 4) for k, v in modes[n].items() if k != "n"}
                 | {"n": modes[n]["n"]} for n in names}, "pairs": pairs}
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
