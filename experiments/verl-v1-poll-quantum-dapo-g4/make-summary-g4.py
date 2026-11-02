#!/usr/bin/env python3
"""4 卡（2+2）DAPO 组过滤下的 poll_interval A/B：解析两份训练日志成 summary.json。

Usage: python3 make-summary-g4.py [raw_dir]

口径与 `experiments/verl-v1-poll-quantum-dapo/make-summary-dapo.py` 相同
（逐 step 一行 `key:value - ...`，稳态窗口 steps 3..14），便于与 2 卡组直接比较。
"""
import json
import pathlib
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PHASES = ["gen", "old_log_prob", "ref", "update_actor", "update_weights", "adv"]
RUNS = [
    ("g4base", "4 GPU (0,1,4,5, 2+2), pristine tree, default 2.0s, filter on, inflight=2"),
    ("g4p010", "4 GPU (0,1,4,5, 2+2), patched tree, poll_interval=0.1, filter on, inflight=2"),
]
WINDOW = range(3, 15)
TRAIN_BATCH = 32


def parse(path: pathlib.Path):
    steps = {}
    resolved = None
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if resolved is None and "'poll_interval':" in line:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                resolved = float(m.group(1))
        if "timing_s/step" not in line:
            continue
        rec = {}
        for field in line.split(" - "):
            if ":" not in field:
                continue
            key, _, value = field.rpartition(":")
            key = key.strip().split()[-1] if key.strip() else ""
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        if "timing_s/step" in rec and "step" in rec:
            steps[int(rec["step"])] = rec
    return resolved, steps


def stats(values):
    values = [v for v in values if v is not None]
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
        path = raw / f"rollout-poll-dapo-{mode}.txt"
        if not path.exists():
            print(f"missing {path}")
            continue
        resolved, steps = parse(path)
        window = [steps[s] for s in WINDOW if s in steps]
        timing = {key: stats([r.get(f"timing_s/{key}") for r in window]) for key in PHASES + ["step"]}
        phase_sum = statistics.fmean(sum(r.get(f"timing_s/{k}", 0.0) for k in PHASES) for r in window)
        evicted = [r.get("training/filter_groups/evicted_samples") for r in window]
        surplus = [r.get("training/filter_groups/discarded_surplus_samples") for r in window]
        generated = [TRAIN_BATCH + (e or 0.0) + (s or 0.0) for e, s in zip(evicted, surplus, strict=False)]
        gen_values = [r["timing_s/gen"] for r in window]
        out[mode] = {
            "description": desc,
            "resolved_poll_interval": resolved,
            "steps_seen": sorted(steps),
            "window": [min(WINDOW), max(WINDOW)],
            "timing": timing,
            "phase_sum_mean": phase_sum,
            "residual_mean": timing["step"]["mean"] - phase_sum,
            "gen_values": gen_values,
            "step_values": [r["timing_s/step"] for r in window],
            "filter": {
                "evicted_groups": stats(evicted),
                "discarded_surplus_groups": stats(surplus),
                "generated_groups": stats(generated),
                "selected_groups": TRAIN_BATCH,
                "useful_fraction": TRAIN_BATCH / statistics.fmean(generated) if generated else None,
                "gen_per_group": statistics.fmean(gen_values) / statistics.fmean(generated) if generated else None,
                "filter_rate": statistics.fmean(evicted) / statistics.fmean(generated) if generated else None,
                "evicted_values": evicted,
                "surplus_values": surplus,
                "generated_values": generated,
            },
            "numeric": {
                key: [r.get(key) for s, r in sorted(steps.items())]
                for key in ["critic/rewards/mean", "actor/entropy", "actor/pg_loss", "response_length/mean"]
            },
        }

    if len(out) == 2:
        a, b = out["g4base"], out["g4p010"]
        row = {}
        for key in PHASES + ["step"]:
            ca = a["timing"][key]["mean"]
            cb = b["timing"][key]["mean"]
            row[key] = {"control": ca, "mode": cb, "delta": cb - ca, "ratio": cb / ca if ca else None}
        out["g4p010"]["vs_control"] = {
            "control": "g4base",
            "timing": row,
            "generated_groups": {
                "control": a["filter"]["generated_groups"]["mean"],
                "mode": b["filter"]["generated_groups"]["mean"],
            },
        }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    (raw.parent / "summary.json").write_text(text + "\n")
    print("mode   poll  gen(s)      step(s)     gen'd  useful  filter%  residual")
    for mode, _desc in RUNS:
        if mode not in out:
            continue
        r = out[mode]
        f = r["filter"]
        print(
            f"{mode:7s} {str(r['resolved_poll_interval']):5s} "
            f"{r['timing']['gen']['mean']:7.3f}±{r['timing']['gen']['sd']:.2f} "
            f"{r['timing']['step']['mean']:7.3f}±{r['timing']['step']['sd']:.2f} "
            f"{f['generated_groups']['mean']:6.1f} {f['useful_fraction']*100:5.1f}% "
            f"{f['filter_rate']*100:6.1f}% {r['residual_mean']:.3f}"
        )
    if "g4p010" in out and "vs_control" in out["g4p010"]:
        v = out["g4p010"]["vs_control"]["timing"]
        print(
            f"\ng4p010 vs g4base: step {v['step']['delta']:+.3f}s ({100*(v['step']['ratio']-1):+.1f}%), "
            f"gen {v['gen']['delta']:+.3f}s ({100*(v['gen']['ratio']-1):+.1f}%)"
        )


if __name__ == "__main__":
    main()
