#!/usr/bin/env python3
"""Parse per-request / per-step instrumentation from the gen-bimodality run.

Usage:
  python3 parse-gen-timing.py /tmp/m5-gen-timing-g2t /tmp/rollout-g8-gen-bimodality-g2t.log
  ... --json-out summary.json

Records (JSONL, per pid):
  m5req  t0,ta,t1 (monotonic), sid (load-balancer server id = replica), rid, np, nout
  m5step phase in {gen_begin, gen_generate_end, gen_end}, gs (global step)

Requests are assigned to a step by their completion time falling inside the step's
[gen_begin, gen_end] window; windows are separated by several seconds of training.
"""

import argparse
import json
import os
import re
import statistics
import sys


def read_jsonl(d):
    recs = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(d, name)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recs.append(json.loads(line))
    return recs


def read_train_log(path):
    """timing_s/gen per global step from the console log."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            gs = int(re.search(r"training/global_step:(\d+)", line).group(1))
            out[gs] = d
    return out


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(round((p / 100.0) * (len(vals) - 1)))))
    return vals[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timing_dir")
    ap.add_argument("train_log", nargs="?")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    recs = read_jsonl(args.timing_dir)
    steps = [r for r in recs if r.get("kind") == "m5step"]
    reqs = [r for r in recs if r.get("kind") == "m5req"]
    if not steps:
        sys.exit("no m5step markers found -- instrumentation inactive?")

    by_gs = {}
    for r in steps:
        by_gs.setdefault(r["gs"], {})[r["phase"]] = r["t"]

    windows = []
    for gs, ph in sorted(by_gs.items()):
        if not {"gen_begin", "gen_generate_end", "gen_end"} <= set(ph):
            continue
        windows.append((gs, ph["gen_begin"], ph["gen_generate_end"], ph["gen_end"]))

    log = read_train_log(args.train_log)
    sids = sorted({r["sid"] for r in reqs})
    rows = []
    for gs, b, g_end, e in windows:
        got = [r for r in reqs if b - 0.5 <= r["t1"] <= e + 0.5]
        if not got:
            continue
        per = {}
        for r in got:
            per.setdefault(r["sid"], []).append(r)
        row = {
            "gs": gs,
            "gen_log_s": log.get(gs, {}).get("gen"),
            "generate_s": g_end - b,
            "sleep_s": e - g_end,
            "n_req": len(got),
        }
        for sid in sids:
            rs = per.get(sid, [])
            short = sid.split(":")[-1] if ":" in sid else sid
            row[f"n_{short}"] = len(rs)
            if rs:
                row[f"span_{short}"] = max(x["t1"] for x in rs) - min(x["t0"] for x in rs)
                row[f"end_{short}"] = max(x["t1"] for x in rs) - b
                row[f"first_{short}"] = min(x["t0"] for x in rs) - b
                row[f"p50_{short}"] = statistics.median(x["t1"] - x["t0"] for x in rs)
        durs = [x["t1"] - x["t0"] for x in got]
        nouts = [x["nout"] for x in got if x.get("nout", -1) >= 0]
        row.update(
            {
                "req_p50": statistics.median(durs),
                "req_max": max(durs),
                "nout_mean": statistics.mean(nouts) if nouts else None,
                "nout_max": max(nouts) if nouts else None,
                "preempt": sum((x.get("pre") or 0) for x in got),
            }
        )
        rows.append(row)

    hdr = ["gs", "gen_log_s", "generate_s", "sleep_s", "n_req"]
    for sid in sids:
        short = sid.split(":")[-1] if ":" in sid else sid
        hdr += [f"n_{short}", f"span_{short}", f"p50_{short}"]
    hdr += ["req_p50", "req_max", "nout_mean", "preempt"]
    print("\t".join(hdr))
    for r in rows:
        vals = []
        for h in hdr:
            v = r.get(h)
            vals.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print("\t".join(vals))

    gens = [r["generate_s"] for r in rows]
    if len(gens) >= 4:
        lo, hi = min(gens), max(gens)
        mid = (lo + hi) / 2
        slow = [r for r in rows if r["generate_s"] >= mid]
        fast = [r for r in rows if r["generate_s"] < mid]
        print(f"\n# generate_sequences split: {len(slow)} slow (>= {mid:.2f}s), {len(fast)} fast")
        for label, grp in (("slow", slow), ("fast", fast)):
            if not grp:
                continue
            print(
                f"# {label}: generate={statistics.mean(r['generate_s'] for r in grp):.3f}s "
                f"sleep={statistics.mean(r['sleep_s'] for r in grp):.3f}s "
                f"gen_log={statistics.mean([r['gen_log_s'] for r in grp if r['gen_log_s']]):.3f}s"
                if any(r["gen_log_s"] for r in grp)
                else f"# {label}: generate={statistics.mean(r['generate_s'] for r in grp):.3f}s"
            )
            for sid in sids:
                short = sid.split(":")[-1] if ":" in sid else sid
                ns = [r.get(f"n_{short}", 0) for r in grp]
                sp = [r.get(f"span_{short}", 0) for r in grp]
                print(
                    f"#   {short}: requests/step={statistics.mean(ns):.1f} "
                    f"span={statistics.mean(sp):.3f}s"
                )

    if args.json_out:
        payload = {
            "timing_dir": args.timing_dir,
            "train_log": args.train_log,
            "replicas": sids,
            "steps": rows,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
