#!/usr/bin/env python3
"""Is the poll wait a grid quantization? Test it on the archived A/B logs.

Model under test
----------------
A training step dispatches its generation batch, then blocks in
`ReplayBuffer.sample()`, which re-reads the queue every `poll_interval` P and
only returns once the batch is terminal.  If the batch's own generation takes
D seconds, the release lands on the first grid point after D:

    gen_i(P) = ceil(D_i / P) * P + c

with `c` the fixed work inside the timed block after the last sleep (queue
re-read, selection, materialisation).  The model says nothing about GPU speed:
it only claims *when* the trainer is released, so at P=2.0 a step must take one
of the values k*2.0 + c, never anything in between.

Three readings
--------------
1. Quantization signature (no pairing, no fitted slope): in the 2-GPU
   fixed-batch arm the steady-state `timing_s/gen` must fall into clusters whose
   gap is 2.00 s and whose internal spread is far below 2.0 s.
2. Paired prediction: the same config at P=0.1 gives D_i per step; feeding those
   into the P=2.0 grid is a per-step prediction, valid only while the two runs
   see the same workload in the same step order (fixed-batch arms).
3. DAPO arms run several dispatch rounds per step, so a single rounding cannot
   describe their wait -- that is the expected negative result, and it is why
   `poll-count.patch` exists.

Reads only archived raw logs; no GPU, no training.
"""

import argparse
import json
import math
import random
import re
import statistics
import sys

SYNC = "experiments/verl-v1-poll-quantum/raw"
DAPO = "experiments/verl-v1-poll-quantum-dapo/raw"
G4 = "experiments/verl-v1-poll-quantum-dapo-g4/raw"

STEP_RE = re.compile(r"step:(\d+) ")
GEN_RE = re.compile(r"timing_s/gen:(?:np\.float64\()?([\d.]+)")
WINDOW = range(3, 15)


def per_step_gen(path):
    out = {}
    for line in open(path, errors="replace"):
        if "timing_s/gen" not in line:
            continue
        m, g = STEP_RE.search(line), GEN_RE.search(line)
        if m and g:
            out[int(m.group(1))] = float(g.group(1))
    return out


def predict(dur, poll):
    return math.ceil(dur / poll) * poll


def paired_table(dur_by_step, meas_by_step, poll, window=WINDOW):
    rows = []
    for step in sorted(set(dur_by_step) & set(meas_by_step)):
        if step not in window:
            continue
        d = dur_by_step[step]
        rows.append(
            {
                "step": step,
                "D_from_P010": round(d, 4),
                "pred": round(predict(d, poll), 4),
                "measured": round(meas_by_step[step], 4),
                "resid": round(meas_by_step[step] - predict(d, poll), 4),
            }
        )
    return rows


def resid_stats(rows):
    if not rows:
        return {"n": 0}
    resid = [r["resid"] for r in rows]
    c = statistics.median(resid)
    dev = [abs(r - c) for r in resid]
    return {
        "n": len(rows),
        "median_offset_c": round(c, 4),
        "max_dev_from_c": round(max(dev), 4),
        "sd": round(statistics.pstdev(resid), 4),
        "within_0.05": sum(1 for d in dev if d <= 0.05),
        "within_0.25": sum(1 for d in dev if d <= 0.25),
    }


def clusters(values, tol=0.1):
    """Group steady-state gen values that sit on the same grid point."""
    out = []
    for v in sorted(values):
        if out and v - out[-1]["values"][-1] <= tol:
            out[-1]["values"].append(v)
        else:
            out.append({"values": [v]})
    for c in out:
        c["n"] = len(c["values"])
        c["mean"] = round(statistics.mean(c["values"]), 4)
        c["spread"] = round(max(c["values"]) - min(c["values"]), 4)
        c.pop("values")
    gaps = [round(b["mean"] - a["mean"], 4) for a, b in zip(out, out[1:])]
    return {"clusters": out, "gaps": gaps}


def run():
    p2 = {k: v for k, v in per_step_gen(f"{SYNC}/rollout-poll-ab-pbase.txt").items() if k in WINDOW}
    p05 = {k: v for k, v in per_step_gen(f"{SYNC}/rollout-poll-ab-p050.txt").items() if k in WINDOW}
    p01 = {k: v for k, v in per_step_gen(f"{SYNC}/rollout-poll-ab-p010.txt").items() if k in WINDOW}
    p4 = {k: v for k, v in per_step_gen(f"{SYNC}/rollout-poll-ab-p4base.txt").items() if k in WINDOW}
    p4_01 = {k: v for k, v in per_step_gen(f"{SYNC}/rollout-poll-ab-p4p010.txt").items() if k in WINDOW}

    out = {
        "model": "gen_i(P) = ceil(D_i/P)*P + c, D_i taken from the P=0.1 leg",
        "quantization_signature": {
            "cluster_tolerance_s": 0.1,
            "2gpu_fixed_batch_P2.0": clusters(p2.values()),
            "2gpu_fixed_batch_P0.5": clusters(p05.values()),
            "4gpu_fixed_batch_P2.0": clusters(p4.values()),
        },
        "paired": {},
        "negative_controls": {},
        "dapo_multi_round": {},
    }

    # 1/2. paired prediction on the two fixed-batch arms
    out["paired"]["2gpu_fixed_batch_P2.0"] = paired_table(p01, p2, 2.0)
    out["paired"]["4gpu_fixed_batch_P2.0"] = paired_table(p4_01, p4, 2.0)
    for name, rows in out["paired"].items():
        out.setdefault("paired_stats", {})[name] = resid_stats(rows)

    # 3. controls, run on the arm that actually has variance (2 GPU)
    rows2 = out["paired"]["2gpu_fixed_batch_P2.0"]
    real = resid_stats(rows2)
    variants = {}
    for wrong_poll in (1.0, 0.5, 3.0):
        alt = [{"resid": r["measured"] - predict(r["D_from_P010"], wrong_poll)} for r in rows2]
        variants[f"grid_{wrong_poll}"] = resid_stats(alt)
    rng = random.Random(0)
    ds = [r["D_from_P010"] for r in rows2]
    worst, hits = [], []
    for _ in range(500):
        rng.shuffle(ds)
        alt = [{"resid": r["measured"] - predict(d, 2.0)} for r, d in zip(rows2, ds)]
        st = resid_stats(alt)
        worst.append(st["max_dev_from_c"])
        hits.append(st["within_0.25"])
    variants["shuffled_pairing"] = {
        "median_max_dev": round(statistics.median(worst), 4),
        "min_max_dev": round(min(worst), 4),
        "median_within_0.25": round(statistics.median(hits), 4),
    }
    out["negative_controls"] = {"real": real, "variants": variants}

    # 4. DAPO arms: several dispatch rounds per step
    for name, (m_path, d_path) in {
        "2gpu_dapo_inflight1": (f"{DAPO}/rollout-poll-dapo-di1basec.txt", f"{DAPO}/rollout-poll-dapo-di1p010c.txt"),
        "2gpu_dapo_inflight2": (f"{DAPO}/rollout-poll-dapo-dp200.txt", f"{DAPO}/rollout-poll-dapo-dp010.txt"),
        "4gpu_dapo_inflight2": (f"{G4}/rollout-poll-dapo-g4baseR.txt", f"{G4}/rollout-poll-dapo-g4p010R.txt"),
    }.items():
        rows = paired_table(per_step_gen(d_path), per_step_gen(m_path), 2.0)
        out["dapo_multi_round"][name] = {"rows": rows, "stats": resid_stats(rows)}
    return out


def verify(out):
    problems = []
    sig = out["quantization_signature"]["2gpu_fixed_batch_P2.0"]
    if len(sig["clusters"]) != 2:
        problems.append(f"expected two clusters at P=2.0, got {sig['clusters']}")
    else:
        if abs(sig["gaps"][0] - 2.0) > 0.03:
            problems.append(f"cluster gap {sig['gaps'][0]} != 2.0")
        if max(c["spread"] for c in sig["clusters"]) > 0.05:
            problems.append(f"within-cluster spread {sig['clusters']} too wide for a grid")

    stats = out["paired_stats"]["2gpu_fixed_batch_P2.0"]
    if stats["within_0.25"] < 9:
        problems.append(f"paired prediction degraded: {stats}")
    clean = out["paired_stats"]["4gpu_fixed_batch_P2.0"]
    if clean["max_dev_from_c"] > 0.05:
        problems.append(f"4-GPU arm no longer sits on one grid point: {clean}")

    real = out["negative_controls"]["real"]["within_0.25"]
    for key, val in out["negative_controls"]["variants"].items():
        got = val["median_within_0.25"] if key == "shuffled_pairing" else val["within_0.25"]
        if got >= real:
            problems.append(f"{key} did not degrade (within_0.25={got} vs real {real})")

    for name, arm in out["dapo_multi_round"].items():
        if arm["stats"]["within_0.25"] > 6:
            problems.append(f"{name} unexpectedly fits the single-round model: {arm['stats']}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    out = run()
    print(json.dumps({k: v for k, v in out.items() if k not in ("paired", "dapo_multi_round")},
                     ensure_ascii=False, indent=2))
    for name, rows in out["paired"].items():
        print(f"\n{name}: {out['paired_stats'][name]}")
        for r in rows:
            print(f"  step {r['step']:2d} D={r['D_from_P010']:6.2f} pred={r['pred']:6.2f} "
                  f"meas={r['measured']:6.2f} resid={r['resid']:+.2f}")
    for name, arm in out["dapo_multi_round"].items():
        print(f"\n{name}: {arm['stats']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
    if args.verify:
        problems = verify(out)
        print("\nverify:", "OK" if not problems else problems)
        return 0 if not problems else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
