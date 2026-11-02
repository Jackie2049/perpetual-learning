#!/usr/bin/env python3
"""Parse candidate-F (update-side PPO mini-batch structure) verl logs.

Usage:
  python3 parse-candidate-f.py BASE.log MODE1.log MODE2.log ...

Each mode log must be a candidate-F run (15 steps, TOTAL_EPOCHS=1, f1/f16/f2).
BASE is the candidate-A 90-step log (same workload/config, no lever). Steady-state window is
steps 3-14 for both: the first two steps contain vLLM/FSDP warmup, and step 15
of a 15-step run includes final testing + checkpoint save (e2 step 15 was 37.9s
vs 28.7s steady), which would contaminate the whole-step comparison.

Outputs per-mode stage means, stage-local G, identity prediction and the measured
whole-step ratio. The prediction uses the single-stage form of the multi-stage
identity (research doc §13.27):
  step_x_pred = 1 / (1 - s_k_base * (1 - 1/G_k)),  s_k_base = stage_k_base / step_base
"""
import re
import sys

STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]


def load(path):
    rows = []
    for line in open(path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            d["gs"] = int(re.search(r"training/global_step:(\d+)", line).group(1))
            rows.append(d)
    return rows


def steady(rows, lo=3, hi=14):
    rs = [r for r in rows if lo <= r["gs"] <= hi]
    return {k: sum(r[k] for r in rs) / len(rs) for k in STAGES + ["step"]}, len(rs)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    base, base_n = steady(load(sys.argv[1]))
    print(f"BASE  n={base_n}  step={base['step']:.3f}")
    print(f"{'stage':14s} {'base':>8s} {'mode':>8s} {'delta%':>8s} {'G':>7s} {'identity_step_x':>17s} {'actual_step_x':>14s}")
    for path in sys.argv[2:]:
        mode, n = steady(load(path))
        print(f"--- {path}  n={n}  step={mode['step']:.3f}")
        for k in STAGES:
            g = base[k] / mode[k]
            s = base[k] / base["step"]
            pred = 1 / (1 - s * (1 - 1 / g))
            print(f"{k:14s} {base[k]:8.3f} {mode[k]:8.3f} {(mode[k]/base[k]-1)*100:+8.2f} "
                  f"{g:7.3f} {pred:17.4f} {base['step']/mode['step']:14.4f}")
        # whole-step reconstruction from all measured stage deltas
        rebuilt = base["step"] + sum(mode[k] - base[k] for k in STAGES)
        print(f"   all-stage rebuild step={rebuilt:.3f} vs measured={mode['step']:.3f} "
              f"residual={mode['step']-rebuilt:+.3f}s (base unaccounted={base['step']-sum(base[k] for k in STAGES):+.3f}s)")


if __name__ == "__main__":
    main()
