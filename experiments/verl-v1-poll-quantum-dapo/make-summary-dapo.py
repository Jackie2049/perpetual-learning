#!/usr/bin/env python3
"""TM-008 边界实验：把 DAPO 过滤 + 多 in-flight 的 poll A/B 训练日志解析成 summary.json。

Usage: python3 make-summary-dapo.py [raw_dir] [--group=main|inflight1|inflight3]

``main`` 是六跑的 inflight=2 组（写 summary.json），``inflight1`` 是 DAPO 默认
``max_inflight_gen_batches=1`` 的两跑（写 summary-inflight1.json），``inflight3``
是 2026-11-03 补测的 ``max_inflight_gen_batches=3`` 四跑（写 summary-inflight3.json）。

与 `experiments/verl-v1-poll-quantum/make-summary.py` 同口径（逐 step 一行 `key:value - ...`，
稳态窗口 steps 3..14），另外加入本实验特有的三个量：

- ``training/filter_groups/evicted_samples``：本步被 DAPO 过滤掉的组数；
- ``training/filter_groups/discarded_surplus_samples``：选批时多出来的有效组（被丢弃）；
- ``generated_groups`` = 32（选中的 batch）+ 上面两项 ⇒ 本步实际生成组数。

``gen_per_group`` 用 generated_groups 归一，用来区分"等更久"与"生成更多"。
"""
import json
import pathlib
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z0-9_/\.\+\-]+):(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))")
PHASES = ["gen", "old_log_prob", "ref", "update_actor", "update_weights"]
# `adv` 在 V1 里恒为毫秒级，仍保留以便逐相位相加时对账
ALL_PHASES = PHASES + ["adv"]
NUM_KEYS = [
    "critic/rewards/mean",
    "actor/entropy",
    "actor/pg_loss",
    "actor/loss",
    "globalseqlen/mean",
    "global_seqlen/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "training/rollout_probs_diff_mean",
]
RUNS = [
    ("dbase", "2 GPU (0,1), pristine tree, default 2.0s, filter on, inflight=2"),
    ("dp200", "2 GPU (0,1), patched tree, poll_interval=2.0, filter on, inflight=2"),
    ("dp050", "2 GPU (0,1), patched tree, poll_interval=0.5, filter on, inflight=2"),
    ("dp010", "2 GPU (0,1), patched tree, poll_interval=0.1, filter on, inflight=2"),
    ("dp200b", "repeat of dp200"),
    ("dp010b", "repeat of dp010"),
]
CONTROLS = {
    "dp200": "dbase",
    "dp050": "dp200",
    "dp010": "dp200",
    "dp200b": "dp200",
    "dp010b": "dp010",
}
# 第二组：DAPO 默认的 max_inflight_gen_batches=1（投机空间更小），只测 2.0/0.1 两点
RUNS_I1 = [
    ("di1base", "2 GPU (0,1), patched tree, poll_interval=2.0, filter on, inflight=1"),
    ("di1p010", "2 GPU (0,1), patched tree, poll_interval=0.1, filter on, inflight=1"),
    ("di1basec", "repeat of di1base"),
    ("di1p010c", "repeat of di1p010"),
]
CONTROLS_I1 = {"di1p010": "di1base", "di1basec": "di1base", "di1p010c": "di1p010"}
# 第三组：max_inflight_gen_batches=3（2026-11-03 补测），测 2.0/0.1 两点
RUNS_I3 = [
    ("i3p200", "2 GPU (4,5), patched tree, poll_interval=2.0, filter on, inflight=3"),
    ("i3p010", "2 GPU (4,5), patched tree, poll_interval=0.1, filter on, inflight=3"),
    ("i3p200b", "repeat of i3p200"),
    ("i3p010b", "repeat of i3p010"),
]
CONTROLS_I3 = {"i3p010": "i3p200", "i3p200b": "i3p200", "i3p010b": "i3p010"}
GROUPS = {
    "main": (RUNS, CONTROLS, "summary.json"),
    "inflight1": (RUNS_I1, CONTROLS_I1, "summary-inflight1.json"),
    "inflight3": (RUNS_I3, CONTROLS_I3, "summary-inflight3.json"),
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
        for key, value in PAIR.findall(line):
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
    runs, controls, filename = GROUPS[group]
    raw = pathlib.Path(argv[0] if argv else "raw")
    out = {}
    for mode, desc in runs:
        path = raw / f"rollout-poll-dapo-{mode}.txt"
        if not path.exists():
            print(f"missing {path}")
            continue
        resolved, steps = parse(path)
        window = [steps[s] for s in WINDOW if s in steps]
        timing = {key: stats([r.get(f"timing_s/{key}") for r in window]) for key in ALL_PHASES + ["step"]}
        phase_sum = statistics.fmean(sum(r.get(f"timing_s/{k}", 0.0) for k in ALL_PHASES) for r in window)
        evicted = [r.get("training/filter_groups/evicted_samples") for r in window]
        surplus = [r.get("training/filter_groups/discarded_surplus_samples") for r in window]
        generated = [
            TRAIN_BATCH + (e or 0.0) + (s or 0.0) for e, s in zip(evicted, surplus, strict=False)
        ]
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
                # 观测过滤率：被过滤组 / 全部生成组。用于核对跨运行的准确率漂移（p 越低过滤越多）
                "filter_rate": statistics.fmean(evicted) / statistics.fmean(generated) if generated else None,
                "evicted_values": evicted,
                "surplus_values": surplus,
                "generated_values": generated,
            },
            "numeric": {key: [r.get(key) for s, r in sorted(steps.items())] for key in NUM_KEYS},
        }

    for mode, control in controls.items():
        if mode not in out or control not in out:
            continue
        a, b = out[control], out[mode]
        row = {}
        for key in ALL_PHASES + ["step"]:
            ca = a["timing"][key]["mean"] if a["timing"][key] else None
            cb = b["timing"][key]["mean"] if b["timing"][key] else None
            row[key] = {
                "control": ca,
                "mode": cb,
                "delta": (cb - ca) if (ca is not None and cb is not None) else None,
                "ratio": (cb / ca) if (ca and cb) else None,
            }
        fa, fb = a["filter"], b["filter"]
        out[mode]["vs_control"] = {
            "control": control,
            "timing": row,
            "generated_groups": {
                "control": fa["generated_groups"]["mean"],
                "mode": fb["generated_groups"]["mean"],
                "delta": fb["generated_groups"]["mean"] - fa["generated_groups"]["mean"],
            },
            "evicted_groups": {
                "control": fa["evicted_groups"]["mean"],
                "mode": fb["evicted_groups"]["mean"],
            },
        }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    target = raw.parent / filename
    target.write_text(text + "\n")

    print("mode    gen(s)      step(s)     evicted  surplus  gen'd  useful  vs control(step)")
    for mode, _desc in runs:
        if mode not in out:
            continue
        r = out[mode]
        f = r["filter"]
        vc = r.get("vs_control", {}).get("timing", {}).get("step")
        extra = f" {vc['delta']:+.3f}s ({(vc['ratio'] - 1) * 100:+.1f}%)" if vc and vc["delta"] is not None else ""
        print(
            f"{mode:7s} {r['timing']['gen']['mean']:8.3f}±{r['timing']['gen']['sd']:.2f} "
            f"{r['timing']['step']['mean']:8.3f}±{r['timing']['step']['sd']:.2f} "
            f"{f['evicted_groups']['mean']:8.1f} {f['discarded_surplus_groups']['mean']:8.1f} "
            f"{f['generated_groups']['mean']:6.1f} {f['useful_fraction'] * 100:5.1f}%{extra}"
        )
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
