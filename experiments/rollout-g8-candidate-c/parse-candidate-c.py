#!/usr/bin/env python
"""Aggregate candidate C per-combo outputs into summary.json.

Reads OUTDIR/lat_<mode>_in<il>_b<batch>_{stdout,stderr}.txt:
  - stdout: the measurement JSON row (wall_s, decode_tok_per_s, ...)
  - stderr: vLLM engine log; for spec mode, extract the LAST
    "SpecDecoding metrics:" line (per-position acceptance rate curve A(p),
    mean acceptance length, accepted/drafted throughput).
Writes summary.json (string keys only) + anchor.json, prints human summary.
"""
import json
import os
import re
import sys

SPEC_RE = re.compile(
    r"SpecDecoding metrics: "
    r"Mean acceptance length: ([0-9.]+), "
    r"Accepted throughput: ([0-9.]+) tokens/s, "
    r"Drafted throughput: ([0-9.]+) tokens/s, "
    r"Accepted: (\d+) tokens, "
    r"Drafted: (\d+) tokens, "
    r"Per-position acceptance rate: ([0-9., ]+), "
    r"Avg Draft acceptance rate: ([0-9.]+)%"
)
POS_RE = re.compile(r"([0-9.]+)")


def main():
    outdir = sys.argv[1]
    rows = {}
    combos = []
    for mode in ("ar", "spec"):
        for il in (512, 2048):
            for b in (8, 16):
                combos.append((mode, il, b))
    for mode, il, b in combos:
        tag = "lat_%s_in%d_b%d" % (mode, il, b)
        key = "%d_%d_%s" % (il, b, mode)
        stdout_f = os.path.join(outdir, tag + "_stdout.txt")
        stderr_f = os.path.join(outdir, tag + "_stderr.txt")
        row = {}
        if os.path.exists(stdout_f):
            try:
                # Locate the measurement JSON line by prefix (engine
                # shutdown INFO logs may trail the row in stdout).
                for line in reversed(open(stdout_f).read().splitlines()):
                    if line.lstrip().startswith("{"):
                        row = json.loads(line)
                        break
            except Exception as exc:
                row = {"parse_error": str(exc)}
        else:
            row = {"missing": True}
        if mode == "spec" and os.path.exists(stderr_f):
            # vLLM logs can land on either stream depending on environment;
            # search stdout first (observed), fall back to stderr.
            text = ""
            stdout_f2 = os.path.join(outdir, tag + "_stdout.txt")
            if os.path.exists(stdout_f2):
                text = open(stdout_f2).read()
            if "SpecDecoding metrics" not in text:
                text += "\n" + open(stderr_f).read()
            matches = SPEC_RE.findall(text)
            if matches:
                m = matches[-1]  # last interval = final flushed aggregate
                acc_len, acc_tp, draft_tp, acc_tok, draft_tok, pos_str, avg = m
                pos = [float(p) for p in POS_RE.findall(pos_str)]
                row["spec_metrics"] = {
                    "mean_accept_len": float(acc_len),
                    "accepted_tok_per_s": float(acc_tp),
                    "drafted_tok_per_s": float(draft_tp),
                    "accepted_tokens": int(acc_tok),
                    "drafted_tokens": int(draft_tok),
                    "acceptance_rate_pos": pos,  # A(p), p=1..k
                    "avg_draft_accept_rate_pct": float(avg),
                }
            else:
                row["spec_metrics"] = {"missing": "no SpecDecoding metrics line"}
        rows[key] = row
    summary = {
        "meta": {
            "experiment": "rollout-g8-candidate-c",
            "model": "Qwen2.5-1.5B-Instruct",
            "hardware": "4090-1 single RTX 4090 (GPU 5)",
            "engine": "vllm 0.24.0+cu129 (anchor-drift 0.24.0, see anchor.json)",
            "spec_method": "ngram proposer k=5, prompt_lookup_max=5/min=1/ratio=0.8",
            "sampling": "temp=0.7 top_p=0.9",
            "matrix": "input_len {512,2048} x batch {8,16} x mode {ar,spec}, max_tokens=128",
            "caliber": "engine-level; A(p) = engine-internal per-position acceptance",
        },
        "rows": rows,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    anchor = {
        "label": "anchor-drift 0.24.0",
        "registered_anchor": "vllm eaf1627c (2026-10-27 main, §8.9)",
        "used_wheel": "0.24.0+cu129 (tag commit ee0da84a 2026-06-28)",
        "drift": "6033 commits ahead + 18 behind; same V1 engine generation; decode throughput NOT cross-version extrapolatable (batch 21 quantification)",
        "reason": "candidate B precedent (batch 24 launch decision); engine-level baseline, version label suffices",
    }
    with open(os.path.join(outdir, "anchor.json"), "w") as f:
        json.dump(anchor, f, indent=2, sort_keys=True)
    # Human summary
    for mode, il, b in combos:
        r = rows["%d_%d_%s" % (il, b, mode)]
        sm = r.get("spec_metrics", {})
        line = "in=%d b=%d %s wall=%.3fs decode=%.1f tok/s" % (
            il, b, mode, r.get("wall_s", -1), r.get("decode_tok_per_s", -1))
        if sm and "acceptance_rate_pos" in sm:
            pos = ", ".join("%.3f" % p for p in sm["acceptance_rate_pos"])
            line += " | acc_len=%.2f acc_tp=%.1f A(p)=[%s]" % (
                sm["mean_accept_len"], sm["accepted_tok_per_s"], pos)
        print(line, flush=True)
    print("wrote summary.json + anchor.json", flush=True)


if __name__ == "__main__":
    main()
