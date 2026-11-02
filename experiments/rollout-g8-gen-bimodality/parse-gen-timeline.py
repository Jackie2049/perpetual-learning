#!/usr/bin/env python3
"""Join V1-trainer phase markers with per-request records into a per-step timeline.

Usage:
  python3 parse-gen-timeline.py /tmp/m5-gen-timing-g2t2 /tmp/rollout-g8-gen-bimodality-g2t2.log
  ... --json-out timeline.json

Markers (m5v1, driver process):
  prepare_begin/prepare_end   -- trainer submits the batch to the rollout manager
  gen_begin                   -- enters marked_timer("gen") (= replay-buffer wait)
  sample_end                  -- replay_buffer.sample() returned
  sleep_begin/sleep_end       -- on_sample_end() -> checkpoint_manager.sleep_replicas()
  gen_end                     -- leaves marked_timer("gen")
  update_weights_begin/end    -- on_step_end() -> wake_up + weight sync

Requests (m5req, agent-loop workers): t0 (dispatch), t1 (completion), sid (replica).

The decisive split for the gen two-state question:
  wait  = sample_end - gen_begin     (is the batch late?)
  sleep = gen_end - sample_end       (is the post-sample sleep the extra time?)
  generation itself = span of request t0..t1 inside the step
"""

import argparse
import json
import os
import re
import statistics


def read_jsonl(d):
    recs = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".jsonl"):
            recs += [json.loads(line) for line in open(os.path.join(d, name)) if line.strip()]
    return recs


def read_train_log(path):
    out = {}
    for line in open(path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            gs = int(re.search(r"training/global_step:(\d+)", line).group(1))
            out[gs] = d
    return out


def build_rows(timing_dir, train_log):
    """Return (per-step rows, replica ids, clock offset) for one instrumented run."""
    recs = read_jsonl(timing_dir)
    marks = [r for r in recs if r.get("kind") == "m5v1"]
    reqs = [r for r in recs if r.get("kind") == "m5req"]
    log = read_train_log(train_log)

    # Markers carry only wall-clock; requests carry monotonic t0/t1 plus the wall
    # captured right after t1. Estimate the constant offset and put both in the
    # requests' monotonic domain (CLOCK_MONOTONIC is system-wide on Linux).
    offsets = sorted(r["wall"] - r["t1"] for r in reqs)
    off = offsets[len(offsets) // 2]

    steps = {}
    for r in marks:
        steps.setdefault(r["gs"], {})[r["phase"]] = r["wall"] - off

    rows = []
    for gs, ph in sorted(steps.items()):
        need = {"gen_begin", "sample_end", "sleep_begin", "sleep_end", "gen_end"}
        if not need <= set(ph):
            continue
        gb = ph["gen_begin"]
        got = [r for r in reqs if gb - 0.5 <= r["t1"] <= ph["gen_end"] + 0.5]
        row = {
            "gs": gs,
            "gen_log": log.get(gs, {}).get("gen"),
            "update_weights_log": log.get(gs, {}).get("update_weights"),
            "wait_s": ph["sample_end"] - ph["gen_begin"],
            "sleep_s": ph["sleep_end"] - ph["sleep_begin"],
            "gen_window_s": ph["gen_end"] - ph["gen_begin"],
            "prepare_s": (ph["prepare_end"] - ph["prepare_begin"]) if "prepare_end" in ph else None,
            "uw_s": (ph["update_weights_end"] - ph["update_weights_begin"])
            if "update_weights_end" in ph
            else None,
            "n_req": len(got),
        }
        if got:
            row["round_span_s"] = max(r["t1"] for r in got) - min(r["t0"] for r in got)
            row["dispatch_offset_s"] = min(r["t0"] for r in got) - gb
            row["gen_end_offset_s"] = max(r["t1"] for r in got) - gb
            for sid in sorted({r["sid"] for r in got}):
                rs = [r for r in got if r["sid"] == sid]
                short = sid.split(":")[-1]
                row[f"n_{short}"] = len(rs)
                row[f"span_{short}"] = max(r["t1"] for r in rs) - min(r["t0"] for r in rs)
        rows.append(row)

    sids = sorted({k[2:] for r in rows for k in r if k.startswith("n_") and k != "n_req"})
    return rows, sids, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timing_dir")
    ap.add_argument("train_log")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    rows, sids, off = build_rows(args.timing_dir, args.train_log)
    print(f"# monotonic<-wall offset (median of wall-t1): {off:.6f}")

    hdr = ["gs", "gen_log", "wait_s", "sleep_s", "gen_window_s", "dispatch_offset_s", "round_span_s", "uw_s", "n_req"]
    hdr += [f"{p}_{s}" for s in sids for p in ("n_", "span_")]
    print("\t".join(hdr))
    for r in rows:
        out = []
        for h in hdr:
            v = r.get(h)
            out.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print("\t".join(out))

    def group(pred):
        return [r for r in rows if pred(r)]

    if len(rows) >= 4:
        gens = [r["gen_window_s"] for r in rows]
        mid = (min(gens) + max(gens)) / 2
        slow, fast = group(lambda r: r["gen_window_s"] >= mid), group(lambda r: r["gen_window_s"] < mid)
        print(f"\n# split: {len(slow)} slow (>= {mid:.2f}s), {len(fast)} fast")
        for lbl, grp in (("slow", slow), ("fast", fast)):
            if not grp:
                continue
            print(
                f"# {lbl}: n={len(grp)} "
                f"wait={statistics.mean(r['wait_s'] for r in grp):.3f} "
                f"sleep={statistics.mean(r['sleep_s'] for r in grp):.3f} "
                f"dispatch_offset={statistics.mean(r['dispatch_offset_s'] for r in grp):.3f} "
                f"round_span={statistics.mean(r['round_span_s'] for r in grp):.3f} "
                f"uw={statistics.mean([r['uw_s'] for r in grp if r['uw_s']] or [0]):.3f}"
            )
            for s in sids:
                print(
                    f"#   {s}: n/step={statistics.mean(r.get('n_' + s, 0) for r in grp):.1f} "
                    f"span={statistics.mean(r.get('span_' + s, 0) for r in grp):.3f}"
                )

    if args.json_out:
        json.dump({"rows": rows, "replicas": sids}, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
