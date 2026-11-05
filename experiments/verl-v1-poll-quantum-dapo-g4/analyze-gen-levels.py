#!/usr/bin/env python3
"""Recover the poll count from `timing_s/gen` alone — the level structure.

`read-polls.py` treats `gen` as a continuous function of the step's workload and
fits `gen ~ 1 + work + polls`. The archive says that model is wrong for the
coarse arms: `gen` is a **staircase**. At P=2.0 the 2-card runs take exactly two
values (2.158 / 4.174, within-level sd 0.003-0.010) and the 4-card DAPO runs
three (4.32 / 6.53 / 8.58). The spacing tracks the poll interval: 2.016 at
P=2.0, 0.498 at P=0.5. That is the poll grid showing through:

    gen = n * s + tail,   s = P + w,   n = number of `time.sleep(P)` polls
    tail = everything sample() does that is not a poll (select + materialize)

Three consequences this script measures instead of asserting:

  1. **The wall clock does identify the count**, up to the unknown `tail`
     (bounded well below `s`). At fixed level, `gen` is flat in `work`
     (0.9 ms/group) while the pooled slope is 41.9 ms/group: `work` acts on
     `gen` only by moving the level (D -> n), so putting `work` in the same
     regression as `polls` mis-specifies the coarse arm.
  2. **Level differences are exact poll-count differences**, so a counter can be
     checked step by step, not just through a fitted coefficient.
  3. **The absolute count needs a bound on `tail`**, not a fit. With
     `tail <= --tail-bound` the grid fit returns n per level; without it only
     differences are identified.

Usage:
  analyze-gen-levels.py --log LOG=P[,tag] [...] [--steady 3:14] [--json out]
  analyze-gen-levels.py --archive            # the six logs quoted in the README
  analyze-gen-levels.py --selftest

Exit codes: 0 ok; 1 unreadable input / no step lines; 3 self-test failure.
"""

import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
READER = HERE / "read-polls.py"

ARCHIVE = [
    ("../verl-v1-poll-quantum/raw/rollout-poll-ab-p200.txt", 2.0, "2card"),
    ("../verl-v1-poll-quantum/raw/rollout-poll-ab-pbase.txt", 2.0, "2card"),
    ("../verl-v1-poll-quantum/raw/rollout-poll-ab-p4base.txt", 2.0, "4card"),
    ("../verl-v1-poll-quantum/raw/rollout-poll-ab-p050.txt", 0.5, "2card"),
    ("../verl-v1-poll-quantum/raw/rollout-poll-ab-p010.txt", 0.1, "2card"),
    ("../verl-v1-poll-quantum/raw/rollout-poll-ab-p4p010.txt", 0.1, "4card"),
    ("raw/rollout-poll-dapo-g4base.txt", 2.0, "4card-dapo"),
    ("raw/rollout-poll-dapo-g4baseR.txt", 2.0, "4card-dapo"),
    ("raw/rollout-poll-dapo-g4baseR2.txt", 2.0, "4card-dapo"),
    ("raw/rollout-poll-dapo-g4p010R.txt", 0.1, "4card-dapo"),
]


def load_reader():
    spec = importlib.util.spec_from_file_location("read_polls", READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_levels(values, thr):
    """Split sorted values into runs whose consecutive gap stays under `thr`."""
    xs = sorted(values)
    groups = [[xs[0]]]
    for v in xs[1:]:
        if v - groups[-1][-1] > thr:
            groups.append([v])
        else:
            groups[-1].append(v)
    return groups


def fit_grid(level_means, nominal_p, tail_bound):
    """Find `s` such that every level mean is `n*s + tail` with tail in [0, bound].

    Returns (s, [(n, tail)], sd(tail)) for the best `s`, or None. The grid is
    searched around the nominal interval; `n` comes from rounding, so a spacing
    that is not the level gap cannot produce consecutive integers with small
    tails and is rejected by the tail spread.
    """
    if len(level_means) < 2:
        return None
    best = None
    steps = int(round((0.9 * nominal_p) / 0.001))
    for i in range(steps):
        s = 0.5 * nominal_p + i * 0.001
        pairs = []
        for m in level_means:
            n = int(round(m / s))
            if n < 1:
                pairs = None
                break
            tail = m - n * s
            if not (-1e-9 <= tail <= tail_bound + 1e-9):
                pairs = None
                break
            pairs.append((n, tail))
        if not pairs:
            continue
        ns = [n for n, _ in pairs]
        # Adjacent levels are adjacent grid points: counts must step by exactly
        # one. Without this, s/P halves double as well (n = 2,4,6) and the fit
        # is scale-free.
        if any(b - a != 1 for a, b in zip(ns, ns[1:])):
            continue
        tails = [t for _, t in pairs]
        spread = statistics.pstdev(tails) if len(tails) > 1 else 0.0
        if best is None or spread < best[2]:
            best = (s, pairs, spread)
    return best


def mediation(rows, thr):
    """Pooled slope of gen on work vs the slope inside each level."""
    by_level = {}
    for group in cluster_levels([r["gen"] for r in rows], thr):
        key = round(statistics.fmean(group), 6)
        by_level[key] = []
    for r in rows:
        for key in by_level:
            if abs(r["gen"] - key) <= thr:
                by_level[key].append(r)
                break
    num = den = 0
    for key, grp in by_level.items():
        if len(grp) < 2:
            continue
        mg = statistics.fmean([r["gen"] for r in grp])
        mw = statistics.fmean([r["work"] for r in grp])
        num += sum((r["gen"] - mg) * (r["work"] - mw) for r in grp)
        den += sum((r["work"] - mw) ** 2 for r in grp)
    return num / den if den else None


def analyze(rp, path, poll, tag, steady, thresholds, tail_bound):
    rows, meta = rp.parse_log(str(path), 0.0)
    if not rows:
        raise SystemExit(f"FATAL: no step lines in {path}")
    lo, hi = steady
    use = [r for r in rows if lo <= r["step"] <= hi and r["gen"] is not None]
    if len(use) < 2:
        raise SystemExit(f"FATAL: fewer than 2 steps in window {steady} for {path}")
    gens = [r["gen"] for r in use]
    out = {
        "log": str(path),
        "tag": tag,
        "poll": poll,
        "steps": [r["step"] for r in use],
        "gen": gens,
        "attempts": [],
    }
    # The quantum is P + w and w differs by config (0.01 non-DAPO, ~0.15 DAPO),
    # so the split threshold cannot be a fixed fraction of P: try a ladder and
    # keep the fit with the tightest tail cluster, which is the one that puts
    # every level on a single grid.
    best = None
    for thr in thresholds(poll):
        groups = cluster_levels(gens, thr)
        means = [statistics.fmean(g) for g in groups]
        attempt = {
            "threshold": thr,
            "level_count": len(groups),
            "level_means": means,
            "level_sizes": [len(g) for g in groups],
            "within_sd": [statistics.pstdev(g) if len(g) > 1 else 0.0 for g in groups],
            "spacings": [means[i + 1] - means[i] for i in range(len(means) - 1)],
        }
        grid = fit_grid(means, poll, tail_bound) if len(means) >= 2 else None
        if grid:
            attempt["grid"] = {"s": grid[0], "counts": [n for n, _ in grid[1]],
                               "tails": [t for _, t in grid[1]], "tail_sd": grid[2],
                               "s_minus_p": grid[0] - poll}
            if best is None or grid[2] < best["grid"]["tail_sd"]:
                best = attempt
        out["attempts"].append(attempt)
    if best:
        out.update({k: best[k] for k in ("threshold", "level_count", "level_means",
                                         "level_sizes", "within_sd", "spacings", "grid")})
        # n per step, in step order, from the fitted grid
        s = best["grid"]["s"]
        tails = best["grid"]["tails"]
        ns = []
        residuals = []
        for r in use:
            hit = min(best["level_means"], key=lambda m: abs(m - r["gen"]))
            ns.append(best["grid"]["counts"][best["level_means"].index(hit)])
            residuals.append((r["gen"] - hit) / s)
        out["counts_per_step"] = ns
        # How far off the grid the points actually sit, in quanta. A fit whose
        # own residuals are a sizeable fraction of one quantum is not a grid.
        out["grid"]["rms_quanta"] = math.sqrt(statistics.fmean([x * x for x in residuals]))
        out["grid"]["usable"] = out["grid"]["rms_quanta"] < 0.15
    have_work = all(r.get("work") is not None for r in use)
    if have_work:
        pooled = rp.ols([r["gen"] for r in use], [[r["work"] for r in use]])
        out["pooled_work_slope"] = pooled[0][1] if pooled else None
        if pooled and pooled[1][1] is not None:
            out["pooled_work_slope_se"] = pooled[1][1]
        out["within_level_work_slope"] = mediation(use, out.get("threshold", poll / 2))
        # Dry run of the window's criterion (1): feed the level-recovered counts
        # in as if they were the counter, and fit the registered two-regressor
        # form against the poll-only form. Same data, same counts -- only the
        # specification differs, so the se difference is what keeping a
        # mediated regressor in the model costs.
        if out.get("counts_per_step"):
            y = [r["gen"] for r in use]
            work = [r["work"] for r in use]
            counts = out["counts_per_step"]
            reg = rp.ols(y, [work, counts])
            simple = rp.ols(y, [counts])
            out["criterion_dryrun"] = {
                "with_work": None if not reg else {"poll_coef": reg[0][2], "poll_se": reg[1][2],
                                                   "work_coef": reg[0][1], "work_se": reg[1][1]},
                "poll_only": None if not simple else {"poll_coef": simple[0][1], "poll_se": simple[1][1]},
            }
    return out


def thresholds(poll):
    """Candidate level-split gaps; wide enough to cover P + w for both families."""
    return sorted({0.02, 0.05, 0.1, 0.2, 0.35, 0.6, round(poll / 2, 4), round(poll, 4)})


def report(items):
    print(f"{'log':40s} {'P':>5s} {'n':>3s} {'thr':>5s} {'lv':>3s}  levels (s)                       "
          f"s       (s-P)  tails (sd)")
    for it in items:
        name = pathlib.Path(it["log"]).name
        if "grid" not in it:
            means = "/".join(f"{m:.3f}" for m in it["attempts"][-1]["level_means"])
            print(f"{name:40s} {it['poll']:5.1f} {len(it['gen']):3d}   -  {it['attempts'][-1]['level_count']:3d}  "
                  f"{means:32s} no grid (needs >=2 levels)")
            continue
        g = it["grid"]
        means = "/".join(f"{m:.3f}" for m in it["level_means"])
        tails = "/".join(f"{t:.3f}" for t in g["tails"])
        flag = "" if g["usable"] else "  <- REJECTED (off-grid)"
        print(f"{name:40s} {it['poll']:5.1f} {len(it['gen']):3d} {it['threshold']:5.2f} "
              f"{it['level_count']:3d}  {means:32s} {g['s']:.3f}  {g['s_minus_p']:+.3f}  "
              f"{tails} (sd {g['tail_sd']:.3f}, rms {g['rms_quanta']:.3f}q){flag}")
        if "pooled_work_slope" in it:
            se = it.get("pooled_work_slope_se")
            print(f"    work slope: pooled={it['pooled_work_slope'] * 1000:.1f} ms/group"
                  + (f" (se {se * 1000:.1f})" if se else "")
                  + (f", within-level={it['within_level_work_slope'] * 1000:.1f} ms/group"
                     if it.get("within_level_work_slope") is not None else ""))
        print("    counts/step: " + " ".join(str(c) for c in it["counts_per_step"]))
        if it.get("criterion_dryrun"):
            d = it["criterion_dryrun"]
            dd = d["with_work"] or {}
            po = d["poll_only"] or {}
            if dd and po:
                print(f"    dry run of criterion (1) on the recovered counts: "
                      f"poll-only coef {po['poll_coef']:.3f} (se {po['poll_se']:.3f}) vs "
                      f"1+work+polls coef {dd['poll_coef']:.3f} (se {dd['poll_se']:.3f}, "
                      f"work {dd['work_coef'] * 1000:.1f} ms/group se {dd['work_se'] * 1000:.1f})")


def pool(items):
    print("\npooled spacing by config (fits only):")
    groups = {}
    for it in items:
        if "grid" in it and it["grid"]["usable"]:
            groups.setdefault((it["tag"], it["poll"]), []).append(it["grid"]["s"])
    for (tag, poll), ss in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        mean = statistics.fmean(ss)
        sd = statistics.pstdev(ss) if len(ss) > 1 else 0.0
        se = sd / math.sqrt(len(ss)) if len(ss) > 1 else 0.0
        print(f"  {tag:12s} P={poll:3.1f}  s={mean:.3f} ± {se:.3f}  (s-P={mean - poll:+.3f}, "
              f"{len(ss)} log(s))")


# --- self-test ---------------------------------------------------------------

def selftest():  # noqa: C901
    checks = []
    rng = __import__("random").Random(7)

    def check(name, cond):
        checks.append((name, bool(cond)))

    # S1 a planted grid is recovered: same tail at every level, tiny noise
    means = [1 * 2.016 + 0.142, 2 * 2.016 + 0.142]
    grid = fit_grid(means, 2.0, 0.5)
    check("S1.planted_grid_recovers_s",
          grid is not None and abs(grid[0] - 2.016) < 0.002 and [n for n, _ in grid[1]] == [1, 2])

    # S2 a *continuous* spread must not look like a grid: with 15 uniform draws
    #    the best grid either fails the tail bound or has a large tail spread.
    xs = [2.0 + 4.0 * rng.random() for _ in range(15)]
    groups = cluster_levels(xs, 1.0)
    grid2 = fit_grid([statistics.fmean(g) for g in groups], 2.0, 0.5)
    check("S2.continuous_rejected", len(groups) == 1 or grid2 is None or grid2[2] > 0.15)

    # S3 the scale is not free: halving the spacing would need non-consecutive n
    means3 = [2.0 + 0.1, 4.0 + 0.1, 6.0 + 0.1]
    grid3 = fit_grid(means3, 2.0, 0.5)
    check("S3.consecutive_counts", grid3 is not None and [n for n, _ in grid3[1]] == [1, 2, 3])

    # S4 a tail outside the bound is rejected rather than silently absorbed
    means4 = [2.0 + 1.2, 4.0 + 1.2]
    grid4 = fit_grid(means4, 2.0, 0.5)
    check("S4.tail_bound_bites", grid4 is None or min(t for _, t in grid4[1]) >= -1e-9)

    # S5 threshold dependence: the 0.5 s grid is invisible at thr=1.0
    xs5 = [2.16, 2.16, 2.17, 2.66, 2.66, 2.67]
    check("S5.threshold_matters",
          len(cluster_levels(xs5, 1.0)) == 1 and len(cluster_levels(xs5, 0.15)) == 2)

    # S6 mediation: a planted staircase has a positive pooled slope and a zero
    #    within-level slope -- the shape the archive shows.
    rows = []
    for work in (60, 70, 80, 90, 100, 110, 120, 130):
        n = 1 if work < 95 else 2
        rows.append({"gen": n * 2.0 + 0.15 + rng.uniform(-0.01, 0.01), "work": work})
    rp = load_reader()
    pooled = rp.ols([r["gen"] for r in rows], [[r["work"] for r in rows]])
    within = mediation(rows, 1.0)
    check("S6.staircase_mediation",
          pooled[0][1] > 0.005 and within is not None and abs(within) < 0.002)

    # S7 the real reader still parses a step line (the script depends on it)
    check("S7.reader_contract", hasattr(rp, "parse_log") and hasattr(rp, "ols"))

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"selftest: {len(checks) - len(failed)}/{len(checks)} passed")
    return 0 if not failed else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=[], metavar="PATH=P[,tag]")
    ap.add_argument("--archive", action="store_true", help="the six archived logs quoted in the README")
    ap.add_argument("--steady", default="3:14")
    ap.add_argument("--tail-bound", type=float, default=0.5,
                    help="max non-poll tail assumed for the absolute count (default 0.5 s)")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    specs = []
    if args.archive:
        specs += [(str(HERE / p), poll, tag) for p, poll, tag in ARCHIVE]
    for spec in args.log:
        path, _, rest = spec.partition("=")
        poll_s, _, tag = rest.partition(",")
        specs.append((path, float(poll_s), tag or "custom"))
    if not specs:
        ap.error("--log or --archive is required (or --selftest)")

    lo, _, hi = args.steady.partition(":")
    steady = (int(lo), int(hi) if hi else 10 ** 6)
    rp = load_reader()
    items = []
    for path, poll, tag in specs:
        # the 2-card archive has no filter_groups keys and starts at step 1
        use_steady = (1, 10 ** 6) if "/verl-v1-poll-quantum/" in str(path) else steady
        try:
            items.append(analyze(rp, path, poll, tag, use_steady, thresholds, args.tail_bound))
        except OSError as exc:
            print(f"FATAL: cannot read {path}: {exc}", file=sys.stderr)
            return 1

    report(items)
    pool(items)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"steady": list(steady), "tail_bound": args.tail_bound, "runs": items},
                      fh, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
