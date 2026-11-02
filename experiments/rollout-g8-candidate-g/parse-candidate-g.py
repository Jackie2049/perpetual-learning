#!/usr/bin/env python3
"""Parse candidate-G (same-stack multi-GPU scaling) verl logs.

Usage:
  python3 parse-candidate-g.py g1.log g2.log g4.log

All logs are 15-step runs (TOTAL_EPOCHS=1) of the identical workload/config; only
NGPUS_PER_NODE differs (1/2/4). Steady-state window is steps 3-14: step 1-2 contain
vLLM/FSDP warmup, step 15 includes final testing (+ possible final checkpoint save).

Outputs per-mode stage means, whole-step speedup vs g1, stage scaling factor G,
parallel efficiency, and the unaccounted residual (step - sum(stages)).
"""
import json
import re
import statistics
import sys

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
            m = re.search(r"perf/total_num_tokens:(\d+)", line)
            if m:
                d["tokens"] = int(m.group(1))
            rows.append(d)
    return rows


def steady(rows, lo=3, hi=None):
    # The final step of every run includes validation (and possibly a final save),
    # so it is excluded from the steady window. Steps 1-2 are vLLM/FSDP warmup.
    if hi is None:
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
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    modes = {}
    for path in sys.argv[1:]:
        label = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        m = re.search(r"-g(\d+w?)$", label)
        modes[m.group(1) if m else label] = steady(load(path))
    def sort_key(k):
        m = re.match(r"g?(\d+)", k)
        return (int(m.group(1)) if m else 0, k.endswith("w"))

    names = sorted(modes, key=sort_key)
    base = modes[names[0]]
    print(f"BASE {names[0]} n={base['n']} step={base['step']:.3f}s "
          f"(sd {base['step_sd']:.3f}) throughput={base['throughput']:.1f} tok/s")
    print(f"{'stage':15s} {'base_s':>8s}", end="")
    for name in names[1:]:
        print(f" {name+'_s':>9s} {'G':>7s}", end="")
    print()
    for k in STAGES:
        print(f"{k:15s} {base[k]:8.3f}", end="")
        for name in names[1:]:
            v = modes[name][k]
            print(f" {v:9.3f} {base[k]/v:7.3f}", end="")
        print()

    summary = {"modes": {}, "scaling": {}}
    for name in names:
        m = modes[name]
        summary["modes"][name] = {
            **{k: round(m[k], 4) for k in STAGES + ["step", "throughput"]},
            "step_sd": round(m["step_sd"], 4),
            "n": m["n"],
            "stage_sum": round(sum(m[k] for k in STAGES), 4),
            "residual": round(m["step"] - sum(m[k] for k in STAGES), 4),
        }
        if name != names[0]:
            s = base["step"] / m["step"]
            n = int(re.match(r"g?(\d+)", name).group(1))
            entry = {
                "speedup": round(s, 4),
                "parallel_efficiency": round(s / n, 4),
                "throughput_ratio": round(m["throughput"] / base["throughput"], 4),
                "aggregate_throughput_ratio": round(m["throughput"] * n / base["throughput"], 4),
                "stage_scaling": {k: round(base[k] / m[k], 4) for k in STAGES},
                "residual": round(m["step"] - sum(m[k] for k in STAGES), 4),
            }
            if name.endswith("w"):
                entry["scale_kind"] = "weak (per-GPU batch fixed; global batch scaled)"
            else:
                entry["scale_kind"] = "strong (global batch fixed)"
            summary["scaling"][name] = entry
    print()
    for name in names:
        m = summary["modes"][name]
        print(f"{name}: step={m['step']:.3f}s sd={m['step_sd']:.3f} "
              f"tok/s={m['throughput']:.1f} stage_sum={m['stage_sum']:.3f} "
              f"residual={m['residual']:+.3f}s")
    for name, v in summary["scaling"].items():
        print(f"{name} vs {names[0]}: speedup={v['speedup']:.3f}x "
              f"efficiency={v['parallel_efficiency']:.3f} "
              f"throughput={v['throughput_ratio']:.3f}x")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
