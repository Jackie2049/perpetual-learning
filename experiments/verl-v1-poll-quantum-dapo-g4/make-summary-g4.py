#!/usr/bin/env python3
"""4 卡（2+2）DAPO 组过滤下的 poll_interval A/B：解析两份训练日志成 summary.json。

Usage: python3 make-summary-g4.py [raw_dir] [--group=main|rerun]

``main`` 是 2026-11-02 15:03 的首轮（只有 g4base 成功，写 summary.json）；
``rerun`` 是 16:0x 独占窗口重跑的两臂（写 summary-rerun.json）。

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
    ("g4base", "g4base", "4 GPU (0,1,4,5, 2+2), pristine tree, default 2.0s, filter on, inflight=2"),
    ("g4p010", "g4p010", "4 GPU (0,1,4,5, 2+2), patched tree, poll_interval=0.1, filter on, inflight=2"),
]
# 独占窗口重跑：每个臂两跑（*R2 为重复），臂级统计在 out["arms"] 里按臂合并
RUNS_RERUN = [
    ("g4baseR", "g4base", "4 GPU (0,1,4,5, 2+2), pristine tree, default 2.0s, filter on, inflight=2, exclusive window"),
    ("g4baseR2", "g4base", "same arm, repeat #2"),
    ("g4p050R", "g4p050", "4 GPU (0,1,4,5, 2+2), patched tree, poll_interval=0.5, filter on, inflight=2, exclusive window"),
    ("g4p050R2", "g4p050", "same arm, repeat #2"),
    ("g4p010R", "g4p010", "4 GPU (0,1,4,5, 2+2), patched tree, poll_interval=0.1, filter on, inflight=2, exclusive window"),
    ("g4p010R2", "g4p010", "same arm, repeat #2"),
]
GROUPS = {
    "main": (RUNS, "summary.json"),
    "rerun": (RUNS_RERUN, "summary-rerun.json"),
}
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
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    group = "main"
    for a in sys.argv[1:]:
        if a.startswith("--group="):
            group = a.split("=", 1)[1]
    runs, filename = GROUPS[group]
    raw = pathlib.Path(argv[0] if argv else "raw")
    out = {}
    for mode, arm, desc in runs:
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
            "phase_values": {key: [r.get(f"timing_s/{key}") for r in window] for key in PHASES + ["step"]},
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

    # 臂级合并：同臂的重复跑把窗口内所有 step 合成一个样本（n = 12 × 重复数）
    arm_order = []
    for mode, arm, _desc in runs:
        if mode in out and arm not in arm_order:
            arm_order.append(arm)
    arms = {}
    for arm in arm_order:
        modes = [m for m, a, _ in runs if a == arm and m in out]
        pooled = {key: sum((out[m]["phase_values"][key] for m in modes), []) for key in PHASES + ["step"]}
        pooled_gen = sum((out[m]["filter"]["generated_values"] for m in modes), [])
        pooled_evict = sum((out[m]["filter"]["evicted_values"] for m in modes), [])
        arms[arm] = {
            "runs": modes,
            "resolved_poll_interval": out[modes[0]]["resolved_poll_interval"],
            "timing": {key: stats(pooled[key]) for key in PHASES + ["step"]},
            "generated_groups": stats(pooled_gen),
            "evicted_groups": stats(pooled_evict),
            "useful_fraction": TRAIN_BATCH / statistics.fmean(pooled_gen),
            "gen_per_group": statistics.fmean(pooled["gen"]) / statistics.fmean(pooled_gen),
        }
        arms[arm]["phase_sum_mean"] = statistics.fmean(
            sum(v[i] for k, v in pooled.items() if k != "step") for i in range(len(pooled["step"]))
        )
        arms[arm]["residual_mean"] = arms[arm]["timing"]["step"]["mean"] - arms[arm]["phase_sum_mean"]
    control_arm = arm_order[0] if arm_order else None
    for arm in arm_order[1:]:
        a, b = arms[control_arm], arms[arm]
        arms[arm]["vs_control"] = {
            "control": control_arm,
            "timing": {
                key: {
                    "control": a["timing"][key]["mean"],
                    "mode": b["timing"][key]["mean"],
                    "delta": b["timing"][key]["mean"] - a["timing"][key]["mean"],
                    "ratio": b["timing"][key]["mean"] / a["timing"][key]["mean"] if a["timing"][key]["mean"] else None,
                }
                for key in PHASES + ["step"]
            },
            "generated_groups": {
                "control": a["generated_groups"]["mean"],
                "mode": b["generated_groups"]["mean"],
            },
        }
    if len(arm_order) > 1:
        out["arms"] = arms

    text = json.dumps(out, indent=2, ensure_ascii=False)
    (raw.parent / filename).write_text(text + "\n")
    print("mode     poll  gen(s)      step(s)     gen'd  useful  filter%  residual")
    for mode, _arm, _desc in runs:
        if mode not in out:
            continue
        r = out[mode]
        f = r["filter"]
        print(
            f"{mode:8s} {str(r['resolved_poll_interval']):5s} "
            f"{r['timing']['gen']['mean']:7.3f}±{r['timing']['gen']['sd']:.2f} "
            f"{r['timing']['step']['mean']:7.3f}±{r['timing']['step']['sd']:.2f} "
            f"{f['generated_groups']['mean']:6.1f} {f['useful_fraction']*100:5.1f}% "
            f"{f['filter_rate']*100:6.1f}% {r['residual_mean']:.3f}"
        )
    if len(arm_order) > 1:
        print("\n臂级（重复合并）")
        print("arm      runs  gen(s)      step(s)     gen'd  useful  residual")
        for arm in arm_order:
            r = arms[arm]
            print(
                f"{arm:8s} {len(r['runs']):2d}    {r['timing']['gen']['mean']:7.3f}±{r['timing']['gen']['sd']:.2f} "
                f"{r['timing']['step']['mean']:7.3f}±{r['timing']['step']['sd']:.2f} "
                f"{r['generated_groups']['mean']:6.1f} {r['useful_fraction']*100:5.1f}% {r['residual_mean']:+.3f}"
            )
        for arm in arm_order[1:]:
            v = arms[arm]["vs_control"]["timing"]
            g = arms[arm]["vs_control"]["generated_groups"]
            print(
                f"\n{arm} vs {control_arm}: step {v['step']['delta']:+.3f}s ({100*(v['step']['ratio']-1):+.1f}%), "
                f"gen {v['gen']['delta']:+.3f}s ({100*(v['gen']['ratio']-1):+.1f}%), "
                f"gen'd {g['mode']-g['control']:+.1f} groups"
            )


if __name__ == "__main__":
    main()
