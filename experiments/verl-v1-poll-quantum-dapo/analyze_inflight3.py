#!/usr/bin/env python3
"""inflight=3 补测（2026-11-03）的池化统计与"省等待／多生成"分解。

数据是四份训练日志（i3p200／i3p010 各两跑，稳态 steps 3–14），解析口径与
`make-summary-dapo.py` 相同；本脚本额外做两件事：

1. 两臂池化（每跑等权）——因为单跑会偏离臂均值约 ±0.4s（§13.48 的重复性结论）；
2. 与 `verification.md` 第 3 节同形的共斜率回归：每步 gen／step 对本步生成组数回归，
   斜率各臂共用、截距按 poll 平移（最小二乘 + 按跑分层的 bootstrap），把
   "固定等待收益"与"多生成代价"分开。

Usage: python3 analyze_inflight3.py [raw_dir]
"""
import json
import pathlib
import random
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z0-9_/\.\+\-]+):(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))")
PHASES = ["gen", "old_log_prob", "ref", "update_actor", "update_weights", "adv"]
WINDOW = range(3, 15)
TRAIN_BATCH = 32
ARMS = {"2.0": ["i3p200", "i3p200b"], "0.1": ["i3p010", "i3p010b"]}


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
        if "timing_s/step" in rec and "step" in rec:
            steps[int(rec["step"])] = rec
    return resolved, steps


def window_rows(steps):
    rows = []
    for s in WINDOW:
        rec = steps.get(s)
        if rec is None:
            continue
        evicted = rec.get("training/filter_groups/evicted_samples") or 0.0
        surplus = rec.get("training/filter_groups/discarded_surplus_samples") or 0.0
        rows.append(
            {
                "step": s,
                "gen": rec["timing_s/gen"],
                "step_s": rec["timing_s/step"],
                "generated": TRAIN_BATCH + evicted + surplus,
                "evicted": evicted,
                "surplus": surplus,
                **{f"phase_{k}": rec.get(f"timing_s/{k}") for k in PHASES if k not in ("gen",)},
            }
        )
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def sd(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ols_shared_slope(points):
    """points: [(arm, x, y)] -> (intercepts {arm: a}, slope b)."""
    arms = sorted({a for a, _x, _y in points})
    n = len(points)
    if len(arms) != 2:
        raise SystemExit(f"expected 2 arms, got {arms}")
    xm = sum(x for _a, x, _y in points) / n
    ym = sum(y for _a, _x, y in points) / n
    num = sum((x - xm) * (y - ym) for _a, x, y in points)
    den = sum((x - xm) ** 2 for _a, x, _y in points)
    b = num / den
    intercepts = {}
    for arm in arms:
        ys = [y for a, _x, y in points if a == arm]
        xs = [x for a, x, _y in points if a == arm]
        intercepts[arm] = sum(ys) / len(ys) - b * (sum(xs) / len(xs))
    return intercepts, b


def fe_fit(points):
    """固定效应拟合：臂内中心化后求共同斜率，截距按臂分离。

    与 verification.md 的「共斜率＋臂截距」同模型，但斜率从**臂内**离差估计。跨臂 OLS
    会混入处理效应（poll 同时降低时间、抬高生成量），把斜率往 0 压，只作诊断量保留。
    """
    arms = sorted({a for a, _x, _y in points})
    if len(arms) != 2:
        raise SystemExit(f"expected 2 arms, got {arms}")
    centered = []
    means = {}
    for arm in arms:
        arm_points = [(x, y) for a, x, y in points if a == arm]
        xm = sum(x for x, _y in arm_points) / len(arm_points)
        ym = sum(y for _x, y in arm_points) / len(arm_points)
        means[arm] = (xm, ym)
        centered.extend((x - xm, y - ym) for x, y in arm_points)
    num = sum(x * y for x, y in centered)
    den = sum(x * x for x, _y in centered)
    slope = num / den
    intercepts = {arm: means[arm][1] - slope * means[arm][0] for arm in arms}
    return intercepts, slope


def bootstrap_fe_delta(rows_by_run, metric, n_boot=4000, seed=0):
    rng = random.Random(seed)
    runs_a = [r for r in rows_by_run if r.startswith("i3p200")]
    runs_b = [r for r in rows_by_run if r.startswith("i3p010")]
    deltas, slopes = [], []
    for _ in range(n_boot):
        points = []
        for arm, group in (("2.0", runs_a), ("0.1", runs_b)):
            for _i in range(len(group)):
                run = rng.choice(group)
                for row in rows_by_run[run]:
                    points.append((arm, row["generated"], row[metric]))
        intercepts, slope = fe_fit(points)
        deltas.append(intercepts["0.1"] - intercepts["2.0"])
        slopes.append(slope)
    deltas.sort()
    slopes.sort()
    lo, hi = 0.025 * n_boot, 0.975 * n_boot
    return {
        "delta": statistics.fmean(deltas),
        "ci": [deltas[int(lo)], deltas[int(hi)]],
        "slope": statistics.fmean(slopes),
        "slope_ci": [slopes[int(lo)], slopes[int(hi)]],
    }


def main():
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    rows_by_run, resolved = {}, {}
    for runs in ARMS.values():
        for mode in runs:
            path = raw / f"rollout-poll-dapo-{mode}.txt"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            resolved[mode], steps = parse(path)
            rows_by_run[mode] = window_rows(steps)

    out = {"resolved_poll_interval": resolved, "arms": {}, "pooled": {}, "regression": {}}
    for arm, runs in ARMS.items():
        out["arms"][arm] = {}
        for mode in runs:
            rows = rows_by_run[mode]
            out["arms"][arm][mode] = {
                "gen_mean": mean([r["gen"] for r in rows]),
                "step_mean": mean([r["step_s"] for r in rows]),
                "generated_mean": mean([r["generated"] for r in rows]),
                "evicted_mean": mean([r["evicted"] for r in rows]),
                "surplus_mean": mean([r["surplus"] for r in rows]),
                "useful_fraction": TRAIN_BATCH / mean([r["generated"] for r in rows]),
                "gen_per_group_ms": 1000.0
                * mean([r["gen"] for r in rows])
                / mean([r["generated"] for r in rows]),
                "phases": {k: mean([r.get(f"phase_{k}") for r in rows]) for k in PHASES if k != "gen"},
                "n_steps": len(rows),
            }

    for arm, runs in ARMS.items():
        pooled = {}
        for key in ("gen_mean", "step_mean", "generated_mean", "evicted_mean", "surplus_mean"):
            pooled[key] = statistics.fmean(out["arms"][arm][m][key] for m in runs)
        pooled["useful_fraction"] = TRAIN_BATCH / pooled["generated_mean"]
        pooled["gen_per_group_ms"] = 1000.0 * pooled["gen_mean"] / pooled["generated_mean"]
        # 单跑极差：量子化臂的跨跑稳定性
        pooled["gen_run_range"] = max(out["arms"][arm][m]["gen_mean"] for m in runs) - min(
            out["arms"][arm][m]["gen_mean"] for m in runs
        )
        pooled["step_run_range"] = max(out["arms"][arm][m]["step_mean"] for m in runs) - min(
            out["arms"][arm][m]["step_mean"] for m in runs
        )
        out["pooled"][arm] = pooled

    a, b = out["pooled"]["2.0"], out["pooled"]["0.1"]
    out["pooled_delta"] = {
        "gen": {"abs": b["gen_mean"] - a["gen_mean"], "ratio": b["gen_mean"] / a["gen_mean"]},
        "step": {"abs": b["step_mean"] - a["step_mean"], "ratio": b["step_mean"] / a["step_mean"]},
        "generated": {
            "abs": b["generated_mean"] - a["generated_mean"],
            "ratio": b["generated_mean"] / a["generated_mean"],
        },
        "surplus": {"abs": b["surplus_mean"] - a["surplus_mean"], "ratio": b["surplus_mean"] / a["surplus_mean"]},
    }

    for metric, key in (("gen", "gen"), ("step_s", "step")):
        points = [
            ("2.0" if run.startswith("i3p200") else "0.1", row["generated"], row[metric])
            for run, rows in rows_by_run.items()
            for row in rows
        ]
        intercepts, slope = fe_fit(points)
        shared_intercepts, shared_slope = ols_shared_slope(points)
        boot = bootstrap_fe_delta(rows_by_run, metric)
        out["regression"][key] = {
            "fe_slope_ms_per_group": 1000.0 * slope,
            "fe_intercepts": intercepts,
            "shared_slope_ms_per_group": 1000.0 * shared_slope,
            "shared_intercept_delta": shared_intercepts["0.1"] - shared_intercepts["2.0"],
            "fixed_offset_0.1_vs_2.0": {
                "delta": boot["delta"],
                "ci95": boot["ci"],
                "slope_ms": 1000.0 * boot["slope"],
                "slope_ci95_ms": [1000.0 * boot["slope_ci"][0], 1000.0 * boot["slope_ci"][1]],
            },
        }
        # 两笔账对账：原始差 ≈ 固定等待 + 斜率 × 生成量差（用固定效应斜率）
        obs = out["pooled_delta"]["gen" if key == "gen" else "step"]["abs"]
        gen_delta = out["pooled_delta"]["generated"]["abs"]
        decomposed = boot["delta"] + boot["slope"] * gen_delta
        out["regression"][key]["decomposition"] = {
            "observed_delta": obs,
            "fixed_wait": boot["delta"],
            "generation_cost": boot["slope"] * gen_delta,
            "reconstructed": decomposed,
            "residual": obs - decomposed,
        }

    print("run      poll  gen(s)     step(s)    gen'd  surplus  useful")
    for arm, runs in ARMS.items():
        for mode in runs:
            r = out["arms"][arm][mode]
            print(
                f"{mode:9s}{arm:5s}{r['gen_mean']:8.3f}  {r['step_mean']:9.3f}"
                f"  {r['generated_mean']:6.1f}  {r['surplus_mean']:6.1f}  {r['useful_fraction'] * 100:5.1f}%"
            )
    print("\npooled   2.0 vs 0.1")
    for key in ("gen", "step"):
        d = out["pooled_delta"][key]
        print(f"  {key:5s} {a[key + '_mean'] if key == 'gen' else a['step_mean']:.3f} -> "
              f"{b['gen_mean'] if key == 'gen' else b['step_mean']:.3f}  "
              f"{d['abs']:+.3f}s ({d['ratio'] - 1:+.1%})")
    g = out["pooled_delta"]["generated"]
    print(f"  generated {a['generated_mean']:.1f} -> {b['generated_mean']:.1f}  "
          f"{g['abs']:+.1f} ({g['ratio'] - 1:+.1%})")
    print(f"  run ranges: 2.0 gen {a['gen_run_range']:.3f}s step {a['step_run_range']:.3f}s | "
          f"0.1 gen {b['gen_run_range']:.3f}s step {b['step_run_range']:.3f}s")
    for key in ("gen", "step"):
        reg = out["regression"][key]
        dep = reg["decomposition"]
        print(f"  regression[{key}]: FE slope {reg['fe_slope_ms_per_group']:.1f} ms/group "
              f"(shared-slope {reg['shared_slope_ms_per_group']:.1f} is cross-arm confounded); "
              f"fixed offset {reg['fixed_offset_0.1_vs_2.0']['delta']:+.3f}s "
              f"CI [{reg['fixed_offset_0.1_vs_2.0']['ci95'][0]:+.3f},{reg['fixed_offset_0.1_vs_2.0']['ci95'][1]:+.3f}]")
        print(f"    decomposition: observed {dep['observed_delta']:+.3f} = "
              f"fixed {dep['fixed_wait']:+.3f} + gen {dep['generation_cost']:+.3f} "
              f"(residual {dep['residual']:+.3f})")

    target = raw.parent / "inflight3-analysis.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
