#!/usr/bin/env python3
"""TM-008 stale 逐出补测：把 colocate_async + threshold=1 的四份训练日志解析成 summary.json。

与 `../../experiments/verl-v1-poll-quantum-dapo/make-summary-dapo.py` 同口径（逐 step
一行、稳态窗口 steps 3..14），但本配置的逐出通道是 **off-policy**（`training/off_policy/
evicted_samples`，每步 32 组），`filter_groups` 关闭，所以：

- ``generated_groups = 32（选中） + off_policy evicted``；
- 另记 `trajectory_spans`／`trajectory_staleness`（日志里是 `np.float64(x)` 形式）。

Usage: python3 make-summary-stale.py [raw_dir]
"""
import json
import pathlib
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z0-9_/\.\+\-]+):(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))")
NP_PAIR = re.compile(r"([A-Za-z0-9_/\.\+\-]+):np\.(?:float64|int64)\(([-\d.e+]+)\)")
PHASES = ["gen", "old_log_prob", "ref", "update_actor", "update_weights", "adv"]
WINDOW = range(3, 15)
TRAIN_BATCH = 32
RUNS = [
    ("st1p200", "2 GPU (4,5), patched tree, poll_interval=2.0, colocate_async, threshold=1"),
    ("st1p010", "2 GPU (4,5), patched tree, poll_interval=0.1, colocate_async, threshold=1"),
    ("st1p200b", "repeat of st1p200"),
    ("st1p010b", "repeat of st1p010"),
]
ARMS = {"2.0": ["st1p200", "st1p200b"], "0.1": ["st1p010", "st1p010b"]}


def parse(path):
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
        for key, value in NP_PAIR.findall(line):
            rec[key] = float(value)
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
    out = {"window": [min(WINDOW), max(WINDOW)], "runs": {}, "arms": {}}
    for mode, desc in RUNS:
        path = raw / f"rollout-poll-stale-{mode}.txt"
        if not path.exists():
            print(f"missing {path}")
            continue
        resolved, steps = parse(path)
        window = [steps[s] for s in WINDOW if s in steps]
        evicted = [r.get("training/off_policy/evicted_samples") for r in window]
        generated = [TRAIN_BATCH + (e or 0.0) for e in evicted]
        out["runs"][mode] = {
            "description": desc,
            "resolved_poll_interval": resolved,
            "steps_seen": sorted(steps),
            "timing": {key: stats([r.get(f"timing_s/{key}") for r in window]) for key in PHASES + ["step"]},
            "off_policy_evicted": stats(evicted),
            "generated_groups": stats(generated),
            "gen_per_group_ms": 1000.0
            * statistics.fmean([r["timing_s/gen"] for r in window])
            / statistics.fmean(generated),
            "spans": stats([r.get("training/off_policy/trajectory_spans/mean") for r in window]),
            "staleness": stats([r.get("training/off_policy/trajectory_staleness/mean") for r in window]),
            "aborted_ratio": stats([r.get("response/aborted_ratio") for r in window]),
        }

    for arm, runs in ARMS.items():
        available = [m for m in runs if m in out["runs"]]
        if not available:
            continue
        pooled = {}
        for key in ("gen", "step", "old_log_prob", "ref", "update_actor", "update_weights"):
            pooled[key] = statistics.fmean(out["runs"][m]["timing"][key]["mean"] for m in available)
        pooled["evicted_mean"] = statistics.fmean(out["runs"][m]["off_policy_evicted"]["mean"] for m in available)
        pooled["generated_mean"] = statistics.fmean(out["runs"][m]["generated_groups"]["mean"] for m in available)
        pooled["runs"] = available
        out["arms"][arm] = pooled

    if "2.0" in out["arms"] and "0.1" in out["arms"]:
        a, b = out["arms"]["2.0"], out["arms"]["0.1"]
        out["pooled_delta"] = {
            key: {"control": a[key], "mode": b[key], "delta": b[key] - a[key], "ratio": b[key] / a[key]}
            for key in ("gen", "step", "evicted_mean", "generated_mean")
        }

    target = raw.parent / "summary.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print("run       poll  gen(s)     step(s)    evicted  gen'd  span  staleness")
    for mode, _desc in RUNS:
        if mode not in out["runs"]:
            continue
        r = out["runs"][mode]
        spans = r["spans"]["mean"] if r["spans"] else float("nan")
        stale = r["staleness"]["mean"] if r["staleness"] else float("nan")
        print(
            f"{mode:9s} {r['resolved_poll_interval']:4.1f} {r['timing']['gen']['mean']:8.3f}±{r['timing']['gen']['sd']:.2f}"
            f" {r['timing']['step']['mean']:8.3f}±{r['timing']['step']['sd']:.2f}"
            f" {r['off_policy_evicted']['mean']:7.1f} {r['generated_groups']['mean']:6.1f} {spans:5.2f} {stale:8.2f}"
        )
    if "pooled_delta" in out:
        print("\npooled 2.0 vs 0.1:")
        for key in ("gen", "step", "evicted_mean", "generated_mean"):
            d = out["pooled_delta"][key]
            print(f"  {key:14s} {d['control']:8.3f} -> {d['mode']:8.3f}  {d['delta']:+.3f} ({d['ratio'] - 1:+.1%})")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
