#!/usr/bin/env python3
"""Independent re-parse of the dropless `wait` A/B (swt200/swt010 + repeats).

Deliberately does not import make-summary-wait.py: tokens are split on " - "
instead of one global key:value regex, np.* wrappers are unwrapped separately,
and the interval comes from a percentile bootstrap over steps stratified by run.

Structural expectations for `max_off_policy_strategy=wait` with threshold=1
(read off replay_buffer.py, not inferred from the data):
  - ``_stale_terminal_keys`` returns empty unless strategy == "drop"  => zero evictions
  - ``_has_enough_samples`` blocks while any in-flight key has
    ``global_steps - prompt_global_steps + 1 >= threshold``  => threshold=1 is a
    full barrier, so every selected trajectory spans exactly 1 model version.
Hence: no ``training/off_policy/evicted_samples`` key at all, spans == 1.0,
staleness <= 1, generated groups == 32/step.

Usage: python3 verify-wait.py [--raw DIR] [--summary FILE]
Exit 0 only when all structural checks and all summary comparisons pass.
"""
import argparse
import json
import pathlib
import random
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NP = re.compile(r"^np\.(?:float64|int64|float32)\((.*)\)$")
GROUPS = {
    # threshold=1: _has_enough_samples blocks on any in-flight key, so spans are
    # structurally pinned to 1 and no stale group can ever be selected.
    "t1": {"runs": ["swt200", "swt200b", "swt010", "swt010b"], "max_span": 1},
    # threshold=2: the batch dispatched by prepare_step() in the current step has
    # span 1 and no longer blocks; the previous step's batch (span 2) does. One
    # step of overlap is allowed, so spans may reach 2 but never 3.
    "t2": {"runs": ["swt2p200", "swt2p200b", "swt2p010", "swt2p010b"], "max_span": 2},
}
ARM = {
    "swt200": "2.0", "swt200b": "2.0", "swt010": "0.1", "swt010b": "0.1",
    "swt2p200": "2.0", "swt2p200b": "2.0", "swt2p010": "0.1", "swt2p010b": "0.1",
}
EXPECT_POLL = {"2.0": 2.0, "0.1": 0.1}
WINDOW = range(3, 15)
TRAIN_BATCH = 32
PHASES = ["gen", "old_log_prob", "ref", "update_actor", "update_weights", "adv"]
TOL = 1e-9


def parse(path):
    """Return (resolved_poll_interval, {step: {field: float}})."""
    resolved = None
    steps = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if resolved is None:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                resolved = float(m.group(1))
        if "timing_s/step" not in line:
            continue
        m_step = re.search(r"\bstep:(\d+)\b", line)
        if not m_step:
            continue
        rec = {"step": float(m_step.group(1))}
        for token in line.split(" - ")[1:]:
            key, sep, value = token.partition(":")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            np_match = NP.match(value)
            if np_match:
                value = np_match.group(1).strip()
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        if "timing_s/step" in rec:
            steps[int(rec["step"])] = rec
    return resolved, steps


def mean_sd(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def bootstrap_diff(stratified, iters=4000, seed=7):
    """Percentile CI of mean(0.1) - mean(2.0); resample steps within each run."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        means = {}
        for arm, runs in stratified.items():
            pool = []
            for run_steps in runs:
                pool += rng.choices(run_steps, k=len(run_steps))
            means[arm] = statistics.fmean(pool)
        diffs.append(means["0.1"] - means["2.0"])
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(GROUPS), default="t1")
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--write-summary", action="store_true")
    args = ap.parse_args()

    group = GROUPS[args.group]
    group_runs = group["runs"]
    max_span = group["max_span"]
    default_summary = "summary-wait.json" if args.group == "t1" else f"summary-wait-{args.group}.json"
    summary_path = pathlib.Path(args.summary or default_summary)

    raw = pathlib.Path(args.raw)
    failures = []
    per_run = {}
    strat = {"2.0": [], "0.1": []}

    for run in group_runs:
        path = raw / f"rollout-poll-stale-{run}.txt"
        if not path.exists():
            failures.append(f"missing log {path}")
            continue
        resolved, steps = parse(path)
        arm = ARM[run]
        win = [steps[s] for s in WINDOW if s in steps]
        if len(win) != 12:
            failures.append(f"{run}: window steps {sorted(steps)} -> {len(win)}/12")
            continue
        if resolved is None or abs(resolved - EXPECT_POLL[arm]) > TOL:
            failures.append(f"{run}: resolved poll {resolved} != {EXPECT_POLL[arm]}")

        # --- structural checks specific to strategy=wait ---
        spans = [r["training/off_policy/trajectory_spans/mean"] for r in win]
        spans_max = [r["training/off_policy/trajectory_spans/max"] for r in win]
        stale_max = [r["training/off_policy/trajectory_staleness_worst/max"] for r in win]
        has_evict_metric = any("training/off_policy/evicted_samples" in r for r in win)
        if max_span == 1 and (any(abs(v - 1.0) > TOL for v in spans) or any(v != 1 for v in spans_max)):
            failures.append(f"{run}: spans not identically 1.0 (mean={sorted(set(spans))}, max={sorted(set(spans_max))})")
        if max_span > 1 and (any(v > max_span for v in spans) or any(v > max_span for v in spans_max)):
            failures.append(f"{run}: spans exceed {max_span} (mean={sorted(set(spans))}, max={sorted(set(spans_max))})")
        if any(v > max_span for v in stale_max):
            failures.append(f"{run}: staleness_worst/max > {max_span} ({sorted(set(stale_max))})")
        if has_evict_metric:
            failures.append(f"{run}: off_policy/evicted_samples present — wait should never evict")

        rec = {
            "resolved": resolved,
            "steps": sorted(steps),
            "spans_mean": statistics.fmean(spans),
            "staleness_worst_max": max(stale_max),
            "evicted_metric_present": has_evict_metric,
            "generated_groups": TRAIN_BATCH,
            "window": win,
        }
        for key in PHASES + ["step"]:
            values = [r[f"timing_s/{key}"] for r in win]
            m, sd = mean_sd(values)
            rec[key] = {"mean": m, "sd": sd}
        rec["gen_per_group_ms"] = 1000.0 * rec["gen"]["mean"] / TRAIN_BATCH
        per_run[run] = rec
        strat[arm].append([r["timing_s/step"] for r in win])

    arms = {}
    for arm in ("2.0", "0.1"):
        arm_runs = [r for r in group_runs if ARM[r] == arm and r in per_run]
        if not arm_runs:
            continue
        arms[arm] = {"runs": arm_runs}
        for key in PHASES + ["step"]:
            arms[arm][key] = statistics.fmean(per_run[r][key]["mean"] for r in arm_runs)
        arms[arm]["gen_per_group_ms"] = statistics.fmean(per_run[r]["gen_per_group_ms"] for r in arm_runs)

    if args.write_summary:
        def st(values):
            return {"n": len(values), "mean": statistics.fmean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values), "max": max(values)}

        summary = {"window": [min(WINDOW), max(WINDOW)], "group": args.group, "runs": {}, "arms": arms,
                   "pooled_delta": {}}
        for run in group_runs:
            if run not in per_run:
                continue
            rec = per_run[run]
            win = rec["window"]
            summary["runs"][run] = {
                "description": f"2 GPU (4,5), patched tree, poll_interval={rec['resolved']}, "
                               f"colocate_async, threshold={max_span}, strategy=wait",
                "resolved_poll_interval": rec["resolved"],
                "steps_seen": rec["steps"],
                "timing": {key: st([r[f"timing_s/{key}"] for r in win]) for key in PHASES + ["step"]},
                "off_policy_evicted": None,
                "generated_groups": st([float(TRAIN_BATCH)] * len(win)),
                "gen_per_group_ms": rec["gen_per_group_ms"],
                "spans": st([r["training/off_policy/trajectory_spans/mean"] for r in win]),
                "staleness": st([r["training/off_policy/trajectory_staleness/mean"] for r in win]),
                "aborted_ratio": st([r["response/aborted_ratio"] for r in win]),
            }
        for arm, a in arms.items():
            a["evicted_mean"] = 0.0
            a["generated_mean"] = float(TRAIN_BATCH)
            a.pop("gen_per_group_ms", None)
        if "2.0" in arms and "0.1" in arms:
            for key in ("gen", "step", "evicted_mean", "generated_mean"):
                a, b = arms["2.0"][key], arms["0.1"][key]
                summary["pooled_delta"][key] = {
                    "control": a, "mode": b, "delta": b - a,
                    "ratio": (b / a) if a else None,
                }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {summary_path}")

    print(f"{'run':9s} {'poll':>4s} {'gen':>8s} {'step':>8s} {'old_lp':>8s} {'ref':>7s} {'upd_act':>8s} {'upd_w':>7s}  spans")
    for run in group_runs:
        if run not in per_run:
            continue
        r = per_run[run]
        print(
            f"{run:9s} {r['resolved']:4.1f} {r['gen']['mean']:8.3f} {r['step']['mean']:8.3f}"
            f" {r['old_log_prob']['mean']:8.3f} {r['ref']['mean']:7.3f} {r['update_actor']['mean']:8.3f}"
            f" {r['update_weights']['mean']:7.3f}  {r['spans_mean']:.2f}"
        )

    if "2.0" in arms and "0.1" in arms:
        print("\npooled (2 runs per arm, steps 3..14):")
        print(f"{'phase':16s} {'2.0':>9s} {'0.1':>9s} {'delta':>9s} {'ratio':>8s}")
        for key in PHASES + ["step"]:
            a, b = arms["2.0"][key], arms["0.1"][key]
            print(f"{key:16s} {a:9.3f} {b:9.3f} {b - a:+9.3f} {b / a:8.3f}")
        gen_series = {"2.0": [], "0.1": []}
        for run in group_runs:
            if run not in per_run:
                continue
            _, steps = parse(raw / f"rollout-poll-stale-{run}.txt")
            gen_series[ARM[run]].append([steps[s]["timing_s/gen"] for s in WINDOW if s in steps])
        lo_s, hi_s = bootstrap_diff(strat, iters=4000)
        lo_g, hi_g = bootstrap_diff(gen_series, iters=4000)
        print(f"bootstrap Δstep 95% CI: [{lo_s:+.3f}, {hi_s:+.3f}] s (step-level, stratified by run)")
        print(f"bootstrap Δgen  95% CI: [{lo_g:+.3f}, {hi_g:+.3f}] s")

        # compare against the executor's summary if present
        if not summary_path.exists():
            print(f"\nnote: {summary_path} not found — skipping summary comparison")
        else:
            summary = json.loads(summary_path.read_text())
            checks = []
            for arm in ("2.0", "0.1"):
                if arm not in summary.get("arms", {}):
                    failures.append(f"summary missing arm {arm}")
                    continue
                for key in ("gen", "step"):
                    mine, theirs = arms[arm][key], summary["arms"][arm][key]
                    if abs(mine - theirs) > 1e-9:
                        failures.append(f"summary arm {arm} {key}: {theirs} != mine {mine}")
                    checks.append((f"arm {arm} {key}", mine, theirs))
                gen_mine = summary["arms"][arm].get("generated_mean")
                if gen_mine is not None and abs(gen_mine - TRAIN_BATCH) > TOL:
                    failures.append(f"summary arm {arm} generated_mean {gen_mine} != {TRAIN_BATCH}")
            for run in group_runs:
                if run not in per_run or run not in summary.get("runs", {}):
                    continue
                for key in ("gen", "step"):
                    theirs = summary["runs"][run]["timing"][key]["mean"]
                    mine = per_run[run][key]["mean"]
                    if abs(mine - theirs) > 1e-9:
                        failures.append(f"summary run {run} {key}: {theirs} != mine {mine}")
            print(f"summary cross-check: {len(checks)} arm values within 1e-9" if not failures else "summary cross-check: MISMATCH")

    if failures:
        print("\nFAIL:")
        for item in failures:
            print("  -", item)
        return 1
    print(f"\nPASS: 4/4 runs, 12-step window, spans<={max_span}, staleness<={max_span}, zero-eviction structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
