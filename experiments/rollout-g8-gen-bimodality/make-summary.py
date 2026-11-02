#!/usr/bin/env python3
"""Build summary.json for the gen-bimodality experiment from raw/ artifacts.

Usage:
  python3 make-summary.py            # reads raw/*.txt and raw/timing-*/*.jsonl.gz

The raw JSONL are archived gzipped; this script decompresses in memory.
"""

import glob
import gzip
import importlib.util
import json
import os
import re
import statistics
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load_timeline_module():
    path = os.path.join(HERE, "parse-gen-timeline.py")
    spec = importlib.util.spec_from_file_location("pgt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unpack(timing_subdir):
    tmp = tempfile.mkdtemp(prefix="m5timing-")
    for gz in glob.glob(os.path.join(HERE, "raw", timing_subdir, "*.jsonl.gz")):
        out = os.path.join(tmp, os.path.basename(gz)[:-3])
        with gzip.open(gz, "rb") as f, open(out, "wb") as g:
            g.write(f.read())
    return tmp


def stage_table(log_path, lo=2, hi=14):
    rows = []
    for line in open(log_path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            d["gs"] = int(re.search(r"training/global_step:(\d+)", line).group(1))
            rows.append(d)
    rs = [r for r in rows if lo <= r["gs"] <= hi]
    out = {}
    for k in ("step", "gen", "old_log_prob", "ref", "update_actor", "update_weights"):
        vals = [r[k] for r in rs if k in r]
        if vals:
            out[k] = {
                "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    out["gen_gt_3s"] = sum(1 for r in rs if r.get("gen", 0) > 3.0)
    return out


def run1_rounds(recs):
    recs = sorted(recs, key=lambda r: r["t0"])
    rounds, cur = [], [recs[0]]
    for r in recs[1:]:
        if r["wall"] - cur[-1]["wall"] > 3:
            rounds.append(cur)
            cur = [r]
        else:
            cur.append(r)
    rounds.append(cur)
    out = []
    for rd in rounds:
        t0 = min(r["t0"] for r in rd)
        per = {}
        for r in rd:
            short = r["sid"].split(":")[-1]
            per.setdefault(short, []).append(r)
        out.append(
            {
                "n": len(rd),
                "span_s": max(r["t1"] for r in rd) - t0,
                "dispatch_spread_s": max(r["t0"] for r in rd) - t0,
                "dur_p50_s": statistics.median(r["t1"] - r["t0"] for r in rd),
                "dur_max_s": max(r["t1"] - r["t0"] for r in rd),
                "nout_mean": statistics.mean(r["nout"] for r in rd),
                "replicas": {
                    k: {"n": len(v), "span_s": max(x["t1"] for x in v) - min(x["t0"] for x in v)}
                    for k, v in per.items()
                },
            }
        )
    return out


def main():
    pgt = load_timeline_module()

    runs = {}
    for name, sub, log in (
        ("g2t", "timing-g2t", "raw/rollout-g8-gen-bimodality-g2t.txt"),
        ("g2t2", "timing-g2t2", "raw/rollout-g8-gen-bimodality-g2t2.txt"),
        ("g2t3", "timing-g2t3", "raw/rollout-g8-gen-bimodality-g2t3.txt"),
    ):
        runs[name] = {"log": log, "stages_steps_2_14": stage_table(os.path.join(HERE, log))}

    # run 1: per-request rounds across 30 steps (no V1 markers in this run)
    d1 = unpack("timing-g2t")
    recs1 = pgt.read_jsonl(d1)
    runs["g2t"]["rounds"] = run1_rounds(recs1)

    # run 2 / 3: full per-step timeline
    for name, sub in (("g2t2", "timing-g2t2"), ("g2t3", "timing-g2t3")):
        d = unpack(sub)
        rows, sids, off = pgt.build_rows(d, os.path.join(HERE, runs[name]["log"]))
        runs[name]["replicas"] = sids
        runs[name]["clock_offset"] = off
        runs[name]["timeline"] = rows

    a, b = runs["g2t2"]["stages_steps_2_14"], runs["g2t3"]["stages_steps_2_14"]
    delta = {k: {"mean_delta": b[k]["mean"] - a[k]["mean"]} for k in a if isinstance(a[k], dict) and k in b}

    def split(rows):
        gens = [r["gen_window_s"] for r in rows]
        mid = (min(gens) + max(gens)) / 2
        out = {}
        for label, pred in (("slow", lambda r: r["gen_window_s"] >= mid), ("fast", lambda r: r["gen_window_s"] < mid)):
            grp = [r for r in rows if pred(r)]
            out[label] = {
                "n": len(grp),
                "wait_s": statistics.mean(r["wait_s"] for r in grp),
                "sleep_s": statistics.mean(r["sleep_s"] for r in grp),
                "round_span_s": statistics.mean(r["round_span_s"] for r in grp),
                "per_replica": {
                    s: {
                        "n_per_step": statistics.mean(r.get("n_" + s, 0) for r in grp),
                        "span_s": statistics.mean(r.get("span_" + s, 0) for r in grp),
                    }
                    for s in runs["g2t2"]["replicas"]
                },
            }
        return out

    summary = {
        "meta": {
            "executed": "2026-11-02",
            "hardware": "4090-1, 2x RTX 4090 (GPU 0,1 both NUMA0)",
            "stack": "verl 0.10.0.dev (24f25b03 tarball) + vLLM 0.24.0, torch 2.11.0+cu129",
            "model": "Qwen2.5-1.5B-Instruct",
            "data": "gsm8k train_sub480 / test_sub120",
            "config": "candidate-G g2: batch 32 prompts x n=4 = 128 samples/step, 2 rollout replicas",
            "purpose": "isolate the gen two-state (4.17s vs 2.15s) by per-replica request timing",
            "instrumentation": "local additive patch (instrument-verl-patch.py), reverted after the runs",
        },
        "runs": {
            "g2t": {
                "mode": "per-replica request timing, 30 steps, replay poll 2.0s",
                "window": "10:04:57-10:18:35 UTC",
                "exit": 0,
            },
            "g2t2": {
                "mode": "V1 phase markers, 15 steps, replay poll 2.0s (default)",
                "window": "10:19:28-10:27:53 UTC",
                "exit": 0,
                "split": split(runs["g2t2"]["timeline"]),
            },
            "g2t3": {
                "mode": "causal intervention, 15 steps, replay poll 0.1s",
                "window": "10:29:44-10:37:50 UTC",
                "exit": 0,
            },
        },
        "intervention_poll_2s_vs_0p1s_steps_2_14": {"poll2.0": a, "poll0.1": b, "delta": delta},
        "detail": runs,
    }
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote summary.json")
    print(json.dumps({k: {kk: round(vv["mean"], 3) for kk, vv in v.items() if isinstance(vv, dict)} for k, v in
                      (("poll2.0", a), ("poll0.1", b))}, indent=2))


if __name__ == "__main__":
    main()
