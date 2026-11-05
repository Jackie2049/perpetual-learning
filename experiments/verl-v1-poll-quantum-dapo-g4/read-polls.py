#!/usr/bin/env python3
"""Read `training/sampler/blocking_polls` out of a verl V1 trainer console log.

The 4-card `inflight=1` arm exists to answer one question: how many full
`poll_interval` sleeps does a training step actually take? `poll-count.patch`
adds the counter, but until now nothing read it — the window's output would be
eyeballed out of a 1200-line log with `(TaskRunnerV1 pid=...)` prefixes and
ANSI escapes, and "no counter in this line" would be indistinguishable from
"counter is zero".

This script is the read side of that measurement, and it also pre-registers how
the number is interpreted:

  * a step line's `timing_s/gen` includes the sampler wait, so
        gen_i = a + b * work_i + P * polls_i
    with `b` the shared per-group generation cost (~31.8 ms/group, measured in
    this directory's six-run window) and `P` the poll interval. The fitted
    coefficient on `polls` must come out at ~`P`; if it does, the counter counts
    what the patch says it counts.
  * across two arms that differ only in `poll_interval`, the quantized-release
    model gives an exact per-step identity:
        wait(P) = ceil(D/P)*P - D          (D = time until the batch is ready)
        gen(P_1) - gen(P_2) = P_1*n_1 - P_2*n_2 + b*(work_1 - work_2)
    Two independent paths (wall clock vs counter) have to agree, so the residual
    per poll must be ~0. What the wall clock alone can NOT identify is `n`: the
    wait only exposes `D mod P`, so any `D` sharing the same residue gives the
    same saving. That is why the counter exists at all.

Usage:
  read-polls.py --log raw/rollout-poll-dapo-g4baseR.txt=2.0 \
                --log raw/rollout-poll-dapo-g4p010R.txt=0.1
  read-polls.py --log /tmp/rollout-poll-dapo-d2.log=2.0 --expect-counter --json out.json
  read-polls.py --selftest

Exit codes: 0 ok; 1 unreadable input / no step lines; 2 counter expected but
absent (a window spent on an un-instrumented tree); 3 self-test failure.
"""

import argparse
import json
import math
import re
import sys

STEP_RE = re.compile(r"^step:(\d+) - ")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX_RE = re.compile(r"^(?:\([^()]*\)\s*)")
NP_WRAP_RE = re.compile(r"^np\.(?:float|int|uint)\d*\((.+)\)$")

GEN_KEY = "timing_s/gen"
STEP_KEY = "timing_s/step"
POLL_KEY = "training/sampler/blocking_polls"
EVICTED_KEY = "training/filter_groups/evicted_samples"
SURPLUS_KEY = "training/filter_groups/discarded_surplus_samples"
SELECTED_KEY = "training/filter_groups/selected_groups"  # absent in logs: selected == train batch size


def strip_line(line: str) -> str:
    line = ANSI_RE.sub("", line).rstrip("\n")
    return RAY_PREFIX_RE.sub("", line)


def parse_number(text: str):
    text = text.strip()
    m = NP_WRAP_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_log(path: str, selected_groups: float = 0.0):
    """Return (rows, meta) for one console log.

    rows: list of dicts, one per `step:N - ...` line, in first-seen order; a
    repeated step number keeps the last occurrence (Ray can replay a line).
    """
    with open(path, "r", errors="replace") as fh:
        raw = fh.read()

    rows = {}
    order = []
    duplicates = 0
    counter_lines = 0
    for line in raw.splitlines():
        s = strip_line(line)
        m = STEP_RE.match(s)
        if not m:
            continue
        # A genuine metric line has many ` - key:value` segments; this also keeps
        # `... timing_s/step:8.2 ...` from being read as `step:8` (it is not at
        # line start), which is the failure mode the 02:0x round hit.
        if s.count(" - ") < 4:
            continue
        step = int(m.group(1))
        fields = {}
        for seg in s.split(" - ")[1:]:
            if ":" not in seg:
                continue
            key, _, val = seg.partition(":")
            num = parse_number(val)
            if num is not None:
                fields[key.strip()] = num
        if POLL_KEY in fields:
            counter_lines += 1
        row = {"step": step, "gen": fields.get(GEN_KEY), "step_s": fields.get(STEP_KEY)}
        if EVICTED_KEY in fields or SURPLUS_KEY in fields:
            ev = fields.get(EVICTED_KEY, 0.0)
            su = fields.get(SURPLUS_KEY, 0.0)
            sel = fields.get(SELECTED_KEY)
            row["evicted"] = ev
            row["surplus"] = su
            row["selected"] = sel
            # `selected_groups` is not printed by the trainer (it is the constant
            # train batch size), so it comes in as a CLI constant.
            row["work"] = ev + su + (sel if sel is not None else 0.0) + selected_groups
        row["polls"] = fields.get(POLL_KEY)
        if step in rows:
            duplicates += 1
            order.remove(step)
        rows[step] = row
        order.append(step)

    meta = {
        "log": path,
        "step_lines": len(order),
        "duplicate_steps": duplicates,
        "counter_lines": counter_lines,
        "steps": sorted(rows),
    }
    return [rows[s] for s in sorted(rows)], meta


def _solve(a, b):
    """Gaussian elimination for a small dense system; None if singular."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(y, xs):
    """y ~ 1 + xs[0] + xs[1] + ...; returns (coefs, ses, r2) or None."""
    n = len(y)
    k = len(xs) + 1
    if n <= k:
        return None
    design = [[1.0] + [x[i] for x in xs] for i in range(n)]
    xtx = [[sum(design[r][i] * design[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    xty = [sum(design[r][i] * y[r] for r in range(n)) for i in range(k)]
    coefs = _solve(xtx, xty)
    if coefs is None:
        return None
    resid = [y[r] - sum(coefs[i] * design[r][i] for i in range(k)) for r in range(n)]
    dof = n - k
    if dof <= 0:
        return coefs, [None] * k, None
    sigma2 = sum(e * e for e in resid) / dof
    inv_diag = []
    for i in range(k):
        e = [0.0] * k
        e[i] = 1.0
        sol = _solve([row[:] for row in xtx], e)
        inv_diag.append(sol[i] if sol else None)
    ses = [math.sqrt(sigma2 * d) if d is not None and d >= 0 else None for d in inv_diag]
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    r2 = 1.0 - sum(e * e for e in resid) / sst if sst > 0 else None
    return coefs, ses, r2


def summarize(rows, poll, steady):
    lo, hi = steady
    use = [r for r in rows if lo <= r["step"] <= hi and r["gen"] is not None]
    if not use:
        return None
    out = {
        "n": len(use),
        "window": [use[0]["step"], use[-1]["step"]],
        "gen_mean": sum(r["gen"] for r in use) / len(use),
        "step_mean": sum(r["step_s"] for r in use if r["step_s"] is not None) / max(
            1, sum(1 for r in use if r["step_s"] is not None)
        ),
    }
    have_work = all(r.get("work") is not None for r in use)
    if have_work:
        out["work_mean"] = sum(r["work"] for r in use) / len(use)
    counts = [r["polls"] for r in use]
    if all(c is not None for c in counts):
        out["polls_per_step"] = sum(counts) / len(counts)
        out["polls_series"] = counts
        out["polls_total"] = sum(counts)
        out["steps_with_zero_polls"] = sum(1 for c in counts if c == 0)
        if poll is not None:
            out["sleep_s_per_step"] = poll * out["polls_per_step"]
    return out


def regression(rows, steady):
    lo, hi = steady
    use = [r for r in rows if lo <= r["step"] <= hi and r["gen"] is not None and r.get("work") is not None]
    if not use or any(r["polls"] is None for r in use):
        return None
    y = [r["gen"] for r in use]
    fit = ols(y, [[r["work"] for r in use], [r["polls"] for r in use]])
    if fit is None:
        return None
    coefs, ses, r2 = fit
    return {
        "n": len(use),
        "intercept": coefs[0],
        "b_per_group": coefs[1],
        "b_per_group_se": ses[1],
        "poll_coef": coefs[2],
        "poll_coef_se": ses[2],
        "r2": r2,
    }


def two_path(left, right, b_per_group):
    """left/right: dicts with summary+poll. Check gen difference against counts."""
    if left.get("polls_per_step") is None or right.get("polls_per_step") is None:
        return None
    if left.get("poll") is None or right.get("poll") is None:
        return None
    pred = left["poll"] * left["polls_per_step"] - right["poll"] * right["polls_per_step"]
    work_delta = 0.0
    if left.get("work_mean") is not None and right.get("work_mean") is not None:
        work_delta = b_per_group * (left["work_mean"] - right["work_mean"])
    pred += work_delta
    measured = left["gen_mean"] - right["gen_mean"]
    resid = measured - pred
    n_left = left["polls_per_step"] * left["n"]
    return {
        "predicted_delta_gen": pred,
        "measured_delta_gen": measured,
        "work_adjustment": work_delta,
        "residual": resid,
        "residual_per_poll": resid / n_left if n_left else None,
        "left_poll": left["poll"],
        "right_poll": right["poll"],
        # Wall clock spent inside sampler sleeps in the coarse arm. This is the
        # measured quantity; "how much of it was waste" is not identifiable here.
        "sleep_s_per_step": left["poll"] * left["polls_per_step"],
    }


def run(args):
    runs = []
    for spec in args.log:
        path, _, poll_s = spec.partition("=")
        try:
            rows, meta = parse_log(path, args.selected_groups)
        except OSError as exc:
            print(f"FATAL: cannot read {path}: {exc}", file=sys.stderr)
            return 1
        poll = float(poll_s) if poll_s else None
        meta["poll"] = poll
        if not rows:
            print(f"FATAL: no `step:N - ...` lines in {path}", file=sys.stderr)
            return 1
        summary = summarize(rows, poll, args.steady)
        if summary is None:
            print(f"FATAL: no steps in steady window {args.steady} for {path}", file=sys.stderr)
            return 1
        counter = "present" if meta["counter_lines"] else "absent"
        runs.append({"log": path, "meta": meta, "rows": rows, "summary": summary, "counter": counter,
                     "poll": poll})

    out = {"steady": list(args.steady), "runs": []}
    for r in runs:
        item = {"log": r["log"], "counter": r["counter"], "poll": r["poll"], "meta": r["meta"],
                "summary": r["summary"]}
        item["regression"] = regression(r["rows"], args.steady)
        out["runs"].append(item)

    b_per_group = None
    for item in out["runs"]:
        reg = item.get("regression")
        if reg and reg.get("b_per_group") is not None and b_per_group is None:
            b_per_group = reg["b_per_group"]
    if b_per_group is None and args.b_per_group is not None:
        b_per_group = args.b_per_group
    out["b_per_group_used"] = b_per_group

    if len(out["runs"]) >= 2 and b_per_group is not None:
        left = out["runs"][0]["summary"] | {"poll": out["runs"][0]["poll"]}
        right = out["runs"][1]["summary"] | {"poll": out["runs"][1]["poll"]}
        out["two_path"] = two_path(left, right, b_per_group)

    # --- report -----------------------------------------------------------
    print(f"steady window: steps {args.steady[0]}..{args.steady[1]}")
    for item in out["runs"]:
        s, meta = item["summary"], item["meta"]
        print(f"\n{item['log']}")
        print(f"  step lines {meta['step_lines']} (dups {meta['duplicate_steps']}), "
              f"counter {item['counter']}, poll {item['poll']}")
        line = f"  n={s['n']} steps {s['window'][0]}..{s['window'][1]}  gen={s['gen_mean']:.3f}s"
        if s.get("step_mean") is not None:
            line += f"  step={s['step_mean']:.3f}s"
        if s.get("work_mean") is not None:
            line += f"  work={s['work_mean']:.1f} groups"
        print(line)
        if s.get("polls_per_step") is not None:
            print(f"  polls/step={s['polls_per_step']:.3f} (total {s['polls_total']}, "
                  f"zero-poll steps {s['steps_with_zero_polls']}/{s['n']})"
                  + (f"  sleep={s['sleep_s_per_step']:.3f}s/step" if s.get("sleep_s_per_step") is not None else ""))
        reg = item.get("regression")
        if reg:
            print(f"  regression gen ~ 1 + work + polls: b={reg['b_per_group'] * 1000:.1f} ms/group, "
                  f"poll_coef={reg['poll_coef']:.3f}s (se {reg['poll_coef_se']:.3f})"
                  if reg['poll_coef_se'] is not None else
                  f"  regression gen ~ 1 + work + polls: poll_coef={reg['poll_coef']:.3f}s")
    if out.get("two_path"):
        tp = out["two_path"]
        print("\ntwo-path check (wall clock vs counter):")
        print(f"  predicted Δgen {tp['predicted_delta_gen']:+.3f}s "
              f"(work adj {tp['work_adjustment']:+.3f}s)  measured {tp['measured_delta_gen']:+.3f}s "
              f"  residual {tp['residual']:+.3f}s")
        if tp.get("sleep_s_per_step") is not None:
            print(f"  sleep in sampler polls: {tp['sleep_s_per_step']:.3f}s/step "
                  f"({tp['left_poll']}s x {tp['left_poll'] and (tp['sleep_s_per_step'] / tp['left_poll']):.2f} polls)")
    if args.json:
        payload = json.loads(json.dumps(out, default=str))
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")

    if args.expect_counter and any(item["counter"] == "absent" for item in out["runs"]):
        print("FATAL: --expect-counter but a log has no blocking_polls: "
              "the run did not execute on an instrumented tree.", file=sys.stderr)
        return 2
    return 0


# --- self-test -------------------------------------------------------------

BARE = (
    "step:3 - training/filter_groups/evicted_samples:91.0 - "
    "training/filter_groups/discarded_surplus_samples:13.0 - timing_s/gen:6.517 - "
    "timing_s/step:18.392 - training/sampler/blocking_polls:2.0"
)
RAY = "\x1b[36m(TaskRunnerV1 pid=3749792)\x1b[0m " + BARE
NO_COUNTER = (
    "step:4 - training/filter_groups/evicted_samples:80.0 - "
    "training/filter_groups/discarded_surplus_samples:12.0 - timing_s/gen:7.100 - "
    "timing_s/step:19.000"
)


def selftest():  # noqa: C901 - flat list of named checks reads better here
    import contextlib
    import io
    import tempfile
    import os

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    def quiet(fn, ns):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return fn(ns)

    d = tempfile.mkdtemp(prefix="read-polls-st")
    logs = {}

    def add(name, text):
        path = os.path.join(d, name)
        with open(path, "w") as fh:
            fh.write(text)
        logs[name] = path
        return path

    # S1 bare line
    rows, meta = parse_log(add("bare.log", BARE + "\n"))
    check("S1.bare_parses", len(rows) == 1 and rows[0]["polls"] == 2.0 and abs(rows[0]["gen"] - 6.517) < 1e-9)

    # S2 ANSI + Ray prefix gives the same row
    rows2, meta2 = parse_log(add("ray.log", RAY + "\n"))
    check("S2.ray_prefix_identical", rows2 == rows and meta2["counter_lines"] == 1)

    # S3 `timing_s/step:8.2` must not be read as step 8
    tricky = ("step:1 - timing_s/gen:5.0 - timing_s/step:8.2 - "
              "training/filter_groups/evicted_samples:1.0 - training/filter_groups/discarded_surplus_samples:1.0\n")
    rows3, _ = parse_log(add("tricky.log", tricky))
    check("S3.step_key_not_confused", [r["step"] for r in rows3] == [1] and rows3[0]["polls"] is None)

    # S4 counter absent is reported as absent, and --expect-counter exits 2
    path_nc = add("nc.log", NO_COUNTER + "\n")
    rows4, meta4 = parse_log(path_nc)
    check("S4a.absent_detected", meta4["counter_lines"] == 0 and rows4[0]["polls"] is None)
    ns = argparse.Namespace(log=[path_nc], steady=(1, 99), json=None, expect_counter=True,
                            b_per_group=None, selected_groups=0.0)
    check("S4b.expect_counter_exit2", quiet(run, ns) == 2)

    # S5 two-path identity: 2.0 arm has 1 poll/step, 0.1 arm 20 polls/step, Δgen = 2.0*1 - 0.1*20 = 0
    left = {"n": 5, "gen_mean": 6.0, "work_mean": 100.0, "polls_per_step": 1.0, "poll": 2.0}
    right = {"n": 5, "gen_mean": 6.0, "work_mean": 100.0, "polls_per_step": 20.0, "poll": 0.1}
    tp = two_path(left, right, 0.0318)
    check("S5a.identity_zero", abs(tp["residual"]) < 1e-9)
    left_bad = dict(left, gen_mean=7.0)
    tp_bad = two_path(left_bad, right, 0.0318)
    check("S5b.identity_breaks", abs(tp_bad["residual"] - 1.0) < 1e-9)
    check("S5c.residual_per_poll", abs(tp_bad["residual_per_poll"] - 0.2) < 1e-9)

    # S6 regression recovers a planted poll coefficient and per-group slope
    #    gen = 2.5 + 0.0318*work + 2.0*polls, work varies, polls 0..2
    lines = []
    plan = [(120, 1), (100, 2), (140, 0), (90, 1), (130, 2), (110, 0), (125, 1)]
    for i, (work, polls) in enumerate(plan, start=1):
        gen = 2.5 + 0.0318 * work + 2.0 * polls
        lines.append(
            f"step:{i} - training/filter_groups/evicted_samples:{work}.0 - "
            f"training/filter_groups/discarded_surplus_samples:0.0 - timing_s/gen:{gen:.6f} - "
            f"timing_s/step:{gen + 12.0:.6f} - training/sampler/blocking_polls:{float(polls)}"
        )
    rows6, _ = parse_log(add("reg.log", "\n".join(lines) + "\n"))
    reg = regression(rows6, (1, 99))
    check("S6.regression_recovers", reg is not None and abs(reg["poll_coef"] - 2.0) < 1e-6
          and abs(reg["b_per_group"] - 0.0318) < 1e-6 and abs(reg["intercept"] - 2.5) < 1e-6)

    # S7 np.float64 wrappers
    wrapped = ("step:2 - training/filter_groups/evicted_samples:10.0 - "
               "training/filter_groups/discarded_surplus_samples:1.0 - timing_s/gen:6.0 - "
               "timing_s/step:18.0 - training/sampler/blocking_polls:np.float64(1.0)")
    rows7, _ = parse_log(add("wrap.log", wrapped + "\n"))
    check("S7.np_wrapper", rows7[0]["polls"] == 1.0)

    # S8 duplicate step keeps the last occurrence
    dup = BARE + "\n" + BARE.replace("blocking_polls:2.0", "blocking_polls:3.0") + "\n"
    rows8, meta8 = parse_log(add("dup.log", dup))
    check("S8.duplicate_last_wins", len(rows8) == 1 and rows8[0]["polls"] == 3.0
          and meta8["duplicate_steps"] == 1)

    # S9 no step lines at all -> exit 1
    path_empty = add("empty.log", "nothing here\n")
    ns2 = argparse.Namespace(log=[path_empty], steady=(1, 99), json=None, expect_counter=False,
                             b_per_group=None, selected_groups=0.0)
    check("S9.no_steps_exit1", quiet(run, ns2) == 1)

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"selftest: {len(checks) - len(failed)}/{len(checks)} passed")
    return 0 if not failed else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=[], metavar="PATH[=POLL]",
                    help="console log; optional =POLL sets the poll_interval for that arm")
    ap.add_argument("--steady", default="3:",
                    help="steady-state step window, e.g. 3: (default) or 3:14")
    ap.add_argument("--selected-groups", type=float, default=32.0,
                    help="selected groups per step (constant train batch size; the "
                         "trainer does not print it, default 32 for this recipe)")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--expect-counter", action="store_true",
                    help="exit 2 if a log lacks training/sampler/blocking_polls")
    ap.add_argument("--b-per-group", type=float, default=None,
                    help="per-group generation cost for the two-path check when no "
                         "regression can be fitted (default 0.0318 from this directory's window)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.log:
        ap.error("--log is required (or --selftest)")
    lo, _, hi = args.steady.partition(":")
    args.steady = (int(lo), int(hi) if hi else 10 ** 6)
    if args.b_per_group is None:
        args.b_per_group = 0.0318
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
