#!/usr/bin/env python3
"""Independent re-parse of batch-56 (inflight=3) raw logs.

Deliberately does not import make-summary-dapo.py or analyze_inflight3.py:
token-by-token splitting instead of one global regex, and a different
bootstrap implementation (percentile over stratified run resamples).
"""
import json, math, pathlib, random, re, statistics

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = re.compile(r"^-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?$")
RUNS = ["i3p200", "i3p200b", "i3p010", "i3p010b"]
ARM = {"i3p200": "2.0", "i3p200b": "2.0", "i3p010": "0.1", "i3p010b": "0.1"}
WINDOW = range(3, 15)


def parse(path):
    steps = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if "timing_s/step" not in line:
            continue
        m_step = re.search(r"\bstep:(\d+)\b", line)
        if not m_step:
            continue
        fields = {"step": float(m_step.group(1))}
        for token in line.split(" - ")[1:]:
            k, sep, v = token.partition(":")
            if not sep:
                continue
            v = v.strip()
            if NUM.match(v):
                try:
                    fields[k.strip()] = float(v)
                except ValueError:
                    pass
        if "timing_s/step" in fields and "step" in fields:
            steps[int(fields["step"])] = fields
    return steps


def mean_sd(vals):
    return (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)


per_run = {}
for run in RUNS:
    steps = parse(RAW / f"rollout-poll-dapo-{run}.txt")
    win = [steps[s] for s in WINDOW if s in steps]
    assert len(win) == 12, (run, len(win))
    rec = {}
    for key in ["gen", "step", "old_log_prob", "ref", "update_actor", "update_weights", "adv"]:
        vals = [r[f"timing_s/{key}"] for r in win]
        m, sd = mean_sd(vals)
        rec[key] = {"mean": m, "sd": sd}
    ev = [r["training/filter_groups/evicted_samples"] for r in win]
    su = [r["training/filter_groups/discarded_surplus_samples"] for r in win]
    gen_groups = [32 + e + s for e, s in zip(ev, su)]
    m_ev, sd_ev = mean_sd(ev)
    m_su, sd_su = mean_sd(su)
    m_gg, sd_gg = mean_sd(gen_groups)
    rec["evicted"] = {"mean": m_ev, "sd": sd_ev}
    rec["surplus"] = {"mean": m_su, "sd": sd_su}
    rec["generated"] = {"mean": m_gg, "sd": sd_gg}
    rec["useful_fraction"] = 32.0 * len(win) / sum(gen_groups)
    per_run[run] = rec

pooled = {}
for arm in ["2.0", "0.1"]:
    runs = [r for r in RUNS if ARM[r] == arm]
    allsteps = []
    for r in runs:
        steps = parse(RAW / f"rollout-poll-dapo-{r}.txt")
        allsteps += [steps[s] for s in WINDOW]
    gen = statistics.fmean(s["timing_s/gen"] for s in allsteps)
    stp = statistics.fmean(s["timing_s/step"] for s in allsteps)
    gg = statistics.fmean(32 + s["training/filter_groups/evicted_samples"] + s["training/filter_groups/discarded_surplus_samples"] for s in allsteps)
    pooled[arm] = {"gen": gen, "step": stp, "generated": gg, "n_steps": len(allsteps)}

# within-arm fixed-effect slope: deviations from each run's own mean
sxy_gen = sxx = sxy_step = 0.0
for run in RUNS:
    steps = parse(RAW / f"rollout-poll-dapo-{run}.txt")
    win = [steps[s] for s in WINDOW]
    xs = [32 + r["training/filter_groups/evicted_samples"] + r["training/filter_groups/discarded_surplus_samples"] for r in win]
    ys_gen = [r["timing_s/gen"] for r in win]
    ys_step = [r["timing_s/step"] for r in win]
    mx, mg, ms = statistics.fmean(xs), statistics.fmean(ys_gen), statistics.fmean(ys_step)
    for x, g, s in zip(xs, ys_gen, ys_step):
        dx = x - mx
        sxx += dx * dx
        sxy_gen += dx * (g - mg)
        sxy_step += dx * (s - ms)
slope_gen = sxy_gen / sxx
slope_step = sxy_step / sxx

dgen = pooled["0.1"]["gen"] - pooled["2.0"]["gen"]
dstep = pooled["0.1"]["step"] - pooled["2.0"]["step"]
dgg = pooled["0.1"]["generated"] - pooled["2.0"]["generated"]
fixed_gen = dgen - slope_gen * dgg
fixed_step = dstep - slope_step * dgg

# percentile bootstrap: resample runs within arm, then steps within run ("stratified")
random.seed(20261103)
boot = {"dgen": [], "dstep": [], "dgg": []}
data = {}
for run in RUNS:
    steps = parse(RAW / f"rollout-poll-dapo-{run}.txt")
    win = [steps[s] for s in WINDOW]
    data[run] = [(r["timing_s/gen"], r["timing_s/step"], 32 + r["training/filter_groups/evicted_samples"] + r["training/filter_groups/discarded_surplus_samples"]) for r in win]
for _ in range(4000):
    acc = {}
    for arm, runs in [("2.0", ["i3p200", "i3p200b"]), ("0.1", ["i3p010", "i3p010b"])]:
        vals = []
        for run in runs:
            picks = [random.choice(data[run]) for _ in range(12)]
            vals += picks
        acc[arm] = (
            statistics.fmean(v[0] for v in vals),
            statistics.fmean(v[1] for v in vals),
            statistics.fmean(v[2] for v in vals),
        )
    boot["dgen"].append(acc["0.1"][0] - acc["2.0"][0])
    boot["dstep"].append(acc["0.1"][1] - acc["2.0"][1])
    boot["dgg"].append(acc["0.1"][2] - acc["2.0"][2])


def ci(vals):
    vals = sorted(vals)
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


out = {
    "per_run": per_run,
    "pooled": pooled,
    "deltas": {
        "gen_s": dgen, "gen_pct": 100 * dgen / pooled["2.0"]["gen"],
        "step_s": dstep, "step_pct": 100 * dstep / pooled["2.0"]["step"],
        "generated_groups": dgg, "generated_pct": 100 * dgg / pooled["2.0"]["generated"],
        "gen_ci": ci(boot["dgen"]), "step_ci": ci(boot["dstep"]), "gen_groups_ci": ci(boot["dgg"]),
    },
    "fixed_effect": {
        "slope_gen_ms_per_group": 1000 * slope_gen,
        "slope_step_ms_per_group": 1000 * slope_step,
        "fixed_wait_gen_s": fixed_gen,
        "fixed_wait_step_s": fixed_step,
        "residual_gen_s": dgen - (slope_gen * dgg + fixed_gen),
        "residual_step_s": dstep - (slope_step * dgg + fixed_step),
    },
}
print(json.dumps(out, indent=2))

ref_summary = json.load(open(HERE / "summary-inflight3.json"))
ref_analysis = json.load(open(HERE / "inflight3-analysis.json"))
mismatch = 0
print("\n=== compare ===")
for run in RUNS:
    a = per_run[run]
    b = ref_summary[run]
    pairs = [
        ("gen", a["gen"]["mean"], b["timing"]["gen"]["mean"]),
        ("gen_sd", a["gen"]["sd"], b["timing"]["gen"]["sd"]),
        ("step", a["step"]["mean"], b["timing"]["step"]["mean"]),
        ("generated", a["generated"]["mean"], b["filter"]["generated_groups"]["mean"]),
        ("evicted", a["evicted"]["mean"], b["filter"]["evicted_groups"]["mean"]),
        ("surplus", a["surplus"]["mean"], b["filter"]["discarded_surplus_groups"]["mean"]),
        ("useful_fraction", a["useful_fraction"], b["filter"]["useful_fraction"]),
    ]
    bad = [(k, x, y) for k, x, y in pairs if abs(x - y) > 1e-9]
    print(f"{run}: " + ("OK" if not bad else f"MISMATCH {bad}"))
    mismatch += len(bad)

print("\n=== pooled vs analysis ===")
for arm in ["2.0", "0.1"]:
    for key, ref_key in [("gen", "gen_mean"), ("step", "step_mean"), ("generated", "generated_mean")]:
        mine, ref = pooled[arm][key], ref_analysis["pooled"][arm][ref_key]
        if abs(mine - ref) > 1e-9:
            print(f"pooled {arm} {key}: MISMATCH {mine} vs {ref}")
            mismatch += 1
print("pooled: OK" if not mismatch else f"{mismatch} mismatches")

print("\n=== deltas (independent) ===")
print(json.dumps(out["deltas"], indent=1))
print("slope gen/step ms per group: %.2f / %.2f" % (
    1000 * slope_gen, 1000 * slope_step))
print("fixed wait gen/step s: %.3f / %.3f" % (fixed_gen, fixed_step))
raise SystemExit(1 if mismatch else 0)
