#!/usr/bin/env python3
"""Independent re-parse of the threshold=2 dropless `wait` A/B (swt2p*).

Separate implementation from verify-wait.py on purpose: this one scans the whole
line with one metric regex instead of splitting on " - ", and it adds the check
that makes the t2 arm informative at all — the window must actually contain
steps whose selected trajectories span 2 model versions. If every span is 1 the
run degenerated to the threshold=1 barrier and says nothing new.

Structural expectation, read off replay_buffer.py rather than the data:
  ``_has_enough_samples`` blocks only while an in-flight key has
  ``global_steps - prompt_global_steps + 1 >= threshold``; with threshold=2 the
  batch dispatched in the *current* step (span 1) does not block, the previous
  step's batch (span 2) does. One step of overlap is therefore legal, spans may
  reach 2, never 3, and strategy="wait" still evicts nothing.

Usage: python3 verify-wait-t2.py [--raw DIR] [--summary FILE] [--write-summary]
Exit 0 only when every structural check passes.
"""
import argparse
import json
import pathlib
import random
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(
    r"([A-Za-z][A-Za-z0-9_/]*):\s*(np\.\w+\((?:[^()]|\([^()]*\))*\)|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
NP = re.compile(r"^np\.\w+\((.*)\)$")

RUNS = ["swt2p200", "swt2p200b", "swt2p010", "swt2p010b"]
ARM = {"swt2p200": "2.0", "swt2p200b": "2.0", "swt2p010": "0.1", "swt2p010b": "0.1"}
EXPECT_POLL = {"2.0": 2.0, "0.1": 0.1}
WINDOW = range(3, 15)
TRAIN_BATCH = 32
MAX_SPAN = 2
PHASES = ["gen", "old_log_prob", "ref", "update_actor", "update_weights", "adv"]
TOL = 1e-9


def parse(path):
    """Return (config, {step: {metric: float}}) using a whole-line metric scan."""
    cfg = {}
    steps = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        for key in ("max_off_policy_threshold", "max_off_policy_strategy", "poll_interval"):
            if key in cfg:
                continue
            m = re.search(rf"'{key}':\s*([0-9.]+|'[a-z]+')", line)
            if m:
                cfg[key] = m.group(1).strip("'")
        if "timing_s/step" not in line:
            continue
        rec = {}
        for key, value in METRIC.findall(line):
            raw = value
            wrapper = NP.match(raw)
            if wrapper:
                raw = wrapper.group(1)
            try:
                rec[key] = float(raw)
            except ValueError:
                pass
        if "timing_s/step" not in rec or "step" not in rec:
            continue
        steps[int(rec["step"])] = rec
    return cfg, steps


def bootstrap(strat, iters=5000, seed=11):
    rng = random.Random(seed)
    out = []
    for _ in range(iters):
        means = {}
        for arm, runs in strat.items():
            pool = []
            for values in runs:
                pool += rng.choices(values, k=len(values))
            means[arm] = statistics.fmean(pool)
        out.append(means["0.1"] - means["2.0"])
    out.sort()
    return out[int(0.025 * iters)], out[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--summary", default="summary-wait-t2.json")
    ap.add_argument("--write-summary", action="store_true")
    args = ap.parse_args()

    raw = pathlib.Path(args.raw)
    summary_path = pathlib.Path(args.summary)
    failures = []
    notes = []
    per_run = {}
    strat_step = {"2.0": [], "0.1": []}
    strat_gen = {"2.0": [], "0.1": []}

    for run in RUNS:
        path = raw / f"rollout-poll-stale-{run}.txt"
        if not path.exists():
            failures.append(f"missing log {path}")
            continue
        cfg, steps = parse(path)
        arm = ARM[run]
        win = [steps[s] for s in WINDOW if s in steps]
        if len(steps) != 15:
            failures.append(f"{run}: saw {len(steps)} steps, expected 15")
        if len(win) != 12:
            failures.append(f"{run}: window has {len(win)}/12 steps (seen {sorted(steps)})")
            continue

        if cfg.get("max_off_policy_threshold") != "2":
            failures.append(f"{run}: threshold {cfg.get('max_off_policy_threshold')} != 2")
        if cfg.get("max_off_policy_strategy") != "wait":
            failures.append(f"{run}: strategy {cfg.get('max_off_policy_strategy')} != wait")
        poll = cfg.get("poll_interval")
        if poll is None or abs(float(poll) - EXPECT_POLL[arm]) > TOL:
            failures.append(f"{run}: resolved poll_interval {poll} != {EXPECT_POLL[arm]}")

        spans_mean = [r["training/off_policy/trajectory_spans/mean"] for r in win]
        spans_max = [r["training/off_policy/trajectory_spans/max"] for r in win]
        stale_max = [r["training/off_policy/trajectory_staleness_worst/max"] for r in win]
        has_evict = any("training/off_policy/evicted_samples" in r for r in win)

        if any(v > MAX_SPAN + TOL for v in spans_mean) or any(v > MAX_SPAN for v in spans_max):
            failures.append(f"{run}: span above {MAX_SPAN} (mean={sorted(set(spans_mean))}, max={sorted(set(spans_max))})")
        if any(v > MAX_SPAN for v in stale_max):
            failures.append(f"{run}: staleness_worst/max above {MAX_SPAN} ({sorted(set(stale_max))})")
        if has_evict:
            failures.append(f"{run}: evicted_samples present — wait must never evict")

        # The point of the t2 arm: overlapping batches must actually happen.
        span2_steps = sum(1 for v in spans_max if v == 2)
        if span2_steps == 0:
            failures.append(f"{run}: span never reaches 2 — arm degenerated into the t1 barrier")

        rec = {"arm": arm, "poll": float(poll), "steps": sorted(steps)}
        for key in PHASES + ["step"]:
            values = [r[f"timing_s/{key}"] for r in win]
            rec[key] = statistics.fmean(values)
            rec[key + "_sd"] = statistics.stdev(values)
        rec.update(
            span_mean=statistics.fmean(spans_mean),
            span2_steps=span2_steps,
            span_max_max=max(spans_max),
            staleness_worst_max=max(stale_max),
            evicted_metric_present=has_evict,
            gen_per_group_ms=1000.0 * rec["gen"] / TRAIN_BATCH,
            gen_slow_steps=sum(1 for r in win if r["timing_s/gen"] > 1.0),
        )
        per_run[run] = rec
        strat_step[arm].append([r["timing_s/step"] for r in win])
        strat_gen[arm].append([r["timing_s/gen"] for r in win])

    arms = {}
    for arm in ("2.0", "0.1"):
        runs = [r for r in RUNS if ARM[r] == arm and r in per_run]
        if not runs:
            continue
        arms[arm] = {"runs": runs}
        for key in PHASES + ["step"]:
            arms[arm][key] = statistics.fmean(per_run[r][key] for r in runs)
        arms[arm]["span_mean"] = statistics.fmean(per_run[r]["span_mean"] for r in runs)
        arms[arm]["span2_steps"] = sum(per_run[r]["span2_steps"] for r in runs)
        arms[arm]["gen_slow_steps"] = sum(per_run[r]["gen_slow_steps"] for r in runs)
        arms[arm]["gen_per_group_ms"] = statistics.fmean(per_run[r]["gen_per_group_ms"] for r in runs)

    print(f"{'run':10s} {'poll':>4s} {'gen':>8s} {'step':>8s} {'span':>5s} {'span2':>5s} {'upd_act':>8s} {'upd_w':>7s}")
    for run in RUNS:
        if run not in per_run:
            continue
        r = per_run[run]
        print(
            f"{run:10s} {r['poll']:4.1f} {r['gen']:8.3f} {r['step']:8.3f}"
            f" {r['span_mean']:5.2f} {r['span2_steps']:5d} {r['update_actor']:8.3f} {r['update_weights']:7.3f}"
        )

    if "2.0" in arms and "0.1" in arms:
        print("\npooled (2 runs per arm, steps 3..14):")
        print(f"{'phase':16s} {'2.0':>9s} {'0.1':>9s} {'delta':>9s} {'ratio':>8s}")
        for key in PHASES + ["step"]:
            a, b = arms["2.0"][key], arms["0.1"][key]
            print(f"{key:16s} {a:9.3f} {b:9.3f} {b - a:+9.3f} {b / a:8.3f}")
        print(f"{'span_mean':16s} {arms['2.0']['span_mean']:9.2f} {arms['0.1']['span_mean']:9.2f}")
        print(f"{'gen_slow_steps':16s} {arms['2.0']['gen_slow_steps']:9d} {arms['0.1']['gen_slow_steps']:9d}"
              f"   (>1.0 s per window, out of 24)")
        lo_s, hi_s = bootstrap(strat_step)
        lo_g, hi_g = bootstrap(strat_gen)
        print(f"bootstrap Δstep 95% CI: [{lo_s:+.3f}, {hi_s:+.3f}] s (5000 iters, seed 11)")
        print(f"bootstrap Δgen  95% CI: [{lo_g:+.3f}, {hi_g:+.3f}] s")

        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            matched = 0
            for arm in ("2.0", "0.1"):
                for key in ("gen", "step"):
                    theirs = summary.get("arms", {}).get(arm, {}).get(key)
                    if theirs is None:
                        failures.append(f"summary missing arms.{arm}.{key}")
                        continue
                    if abs(arms[arm][key] - theirs) > 1e-9:
                        failures.append(f"summary arms.{arm}.{key}={theirs} != mine {arms[arm][key]}")
                    matched += 1
            for run in RUNS:
                if run not in per_run:
                    continue
                node = summary.get("runs", {}).get(run)
                if not node:
                    failures.append(f"summary missing run {run}")
                    continue
                for key in ("gen", "step"):
                    theirs = node["timing"][key]["mean"]
                    if abs(per_run[run][key] - theirs) > 1e-9:
                        failures.append(f"summary runs.{run}.{key}={theirs} != mine {per_run[run][key]}")
                    matched += 1
            print(f"summary cross-check: {matched} values within 1e-9")
        else:
            notes.append(f"summary {summary_path} absent — arm numbers only")

    # Directional cross-window comparison against the threshold=1 wait arm.
    t1_path = raw / "summary-wait.json"
    if t1_path.exists() and "2.0" in arms and "0.1" in arms:
        t1 = json.loads(t1_path.read_text())["arms"]
        print("\nvs threshold=1 wait (adjacent window, directional only):")
        for key in ("gen", "step"):
            print(f"{key:16s} t1 {t1['2.0'][key]:7.3f}/{t1['0.1'][key]:7.3f}"
                  f"  t2 {arms['2.0'][key]:7.3f}/{arms['0.1'][key]:7.3f}"
                  f"  t2-t1 {arms['2.0'][key]-t1['2.0'][key]:+7.3f}/{arms['0.1'][key]-t1['0.1'][key]:+7.3f}")

    if args.write_summary:
        summary = {
            "group": "t2",
            "window": [min(WINDOW), max(WINDOW)],
            "description": "2 GPU (4,5), patched tree, colocate_async, threshold=2, strategy=wait",
            "arms": {},
            "runs": {},
        }
        for arm, a in arms.items():
            summary["arms"][arm] = {k: v for k, v in a.items() if k != "runs"}
        for run, r in per_run.items():
            summary["runs"][run] = {k: v for k, v in r.items() if k != "steps"} | {"steps_seen": r["steps"]}
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {summary_path}")

    for note in notes:
        print("note:", note)
    if failures:
        print("\nFAIL:")
        for item in failures:
            print("  -", item)
        return 1
    print("\nPASS: 4/4 runs, 15 steps, window 3..14, threshold=2 wait, spans reach 2 and stay <=2, zero evictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
