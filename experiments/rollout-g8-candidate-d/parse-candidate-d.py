#!/usr/bin/env python
"""Aggregate candidate D per-combo outputs into summary.json.

Reads OUTDIR/lat_<mode>_<comp>_i<iter>.{stdout,stderr}.txt:
  - stdout: measurement JSON rows (wall_s, decode_tok_per_s, e2e_s, ...)
  - stderr: vLLM engine log; for spec mode, extract the LAST
    "SpecDecoding metrics:" line (A(p) curve, mean accept len).
Writes summary.json (string keys only) + anchor.json and prints the
axis-3 findings: makespan vs batch-internal length distribution
(MIX vs H2048 with identical mean/total tokens), bubble factor
wall/mean(T_i), spec/AR, and A(p) per composition.
"""
import json
import os
import re
import statistics
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
COMPS = ("H512", "H2048", "MIX", "RX8")
MODES = ("ar", "spec")
ITERS = (1, 2)


def _mean(xs):
    return round(statistics.mean(xs), 4) if xs else None


def _spec_metrics(tag, outdir):
    text = ""
    for fname in (tag + ".stdout.txt", tag + ".stderr.txt"):
        f = os.path.join(outdir, fname)
        if os.path.exists(f):
            text += "\n" + open(f).read()
    matches = SPEC_RE.findall(text)
    if not matches:
        return {"missing": "no SpecDecoding metrics line"}
    m = matches[-1]  # last interval = final flushed aggregate
    acc_len, acc_tp, draft_tp, acc_tok, draft_tok, pos_str, avg = m
    return {
        "mean_accept_len": float(acc_len),
        "accepted_tok_per_s": float(acc_tp),
        "drafted_tok_per_s": float(draft_tp),
        "accepted_tokens": int(acc_tok),
        "drafted_tokens": int(draft_tok),
        "acceptance_rate_pos": [float(p) for p in POS_RE.findall(pos_str)],
        "avg_draft_accept_rate_pct": float(avg),
    }


def main():
    outdir = sys.argv[1]
    rows = {}
    for mode in MODES:
        for comp in COMPS:
            runs = []
            for it in ITERS:
                tag = "lat_d_%s_%s_i%d" % (mode, comp, it)
                stdout_f = os.path.join(outdir, tag + ".stdout.txt")
                row = {}
                if os.path.exists(stdout_f):
                    try:
                        for line in reversed(open(stdout_f).read().splitlines()):
                            if line.lstrip().startswith("{"):
                                row = json.loads(line)
                                break
                    except Exception as exc:
                        row = {"parse_error": str(exc)}
                else:
                    row = {"missing": True}
                runs.append(row)
            agg = {
                "runs": runs,
                "wall_s": _mean([r.get("wall_s") for r in runs
                                 if r.get("wall_s") is not None]),
                "decode_tok_per_s": _mean([
                    r.get("decode_tok_per_s") for r in runs
                    if r.get("decode_tok_per_s") is not None]),
                "finished_full": [r.get("finished_full") for r in runs],
                "e2e_all": sorted([
                    x for r in runs for x in (r.get("e2e_s") or [])
                    if isinstance(x, (int, float))]),
            }
            if mode == "spec":
                agg["spec_metrics"] = _spec_metrics("lat_d_spec_%s_i1" % comp,
                                                    outdir)
            rows["%s_%s" % (mode, comp)] = agg
    # Axis-3 findings (batch makespan vs internal length distribution)
    findings = {}
    for mode in MODES:
        w_h512 = rows["%s_H512" % mode]["wall_s"]
        w_h2048 = rows["%s_H2048" % mode]["wall_s"]
        w_mix = rows["%s_MIX" % mode]["wall_s"]
        findings[mode] = {
            "makespan_s": {
                "H512": w_h512, "H2048": w_h2048, "MIX": w_mix,
            },
            "MIX_over_H2048": round(w_mix / w_h2048, 3) if w_h2048 else None,
            "MIX_over_H512": round(w_mix / w_h512, 3) if w_h512 else None,
            "RX8_over_H2048": round(rows["%s_RX8" % mode]["wall_s"] / w_h2048, 3) if w_h2048 else None,
            "spec_over_ar": {
                "H512": round(rows["spec_H512"]["wall_s"] / w_h512, 3) if w_h512 else None,
                "H2048": round(rows["spec_H2048"]["wall_s"] / w_h2048, 3) if w_h2048 else None,
                "MIX": round(rows["spec_MIX"]["wall_s"] / w_mix, 3) if w_mix else None,
                "RX8": round(rows["spec_RX8"]["wall_s"] / rows["ar_RX8"]["wall_s"], 3) if rows["ar_RX8"]["wall_s"] else None,
            },
        }
        for comp in COMPS:
            e2e = rows["%s_%s" % (mode, comp)]["e2e_all"]
            wall = rows["%s_%s" % (mode, comp)]["wall_s"]
            if e2e and wall:
                findings[mode]["bubble_factor_%s" % comp] = round(
                    wall / statistics.mean(e2e), 3)
                findings[mode]["e2e_max_%s" % comp] = round(max(e2e), 4)
                findings[mode]["e2e_median_%s" % comp] = round(
                    statistics.median(e2e), 4)
    summary = {
        "meta": {
            "experiment": "rollout-g8-candidate-d",
            "model": "Qwen2.5-1.5B-Instruct",
            "hardware": "4090-1 single RTX 4090 (GPU 5)",
            "engine": "vllm 0.24.0+cu129 (anchor-drift 0.24.0, see anchor.json)",
            "spec_method": "ngram proposer k=5, prompt_lookup_max=5/min=1",
            "sampling": "temp=0.7 top_p=0.9",
            "matrix": "mode {ar,spec} x comp {H512,H2048,MIX,RX8} x iter "
                      "{1,2}, batch=8, max_tokens=128 (RX8: per-request "
                      "64..256 mean 128), max_model_len=4608",
            "caliber": "engine-level; axis-3 batch-internal length "
                       "distribution -> sync makespan (G3 memo); distinct "
                       "natural-text prompts per request (gsm8k windows)",
        },
        "rows": rows,
        "findings": findings,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    anchor = {
        "label": "anchor-drift 0.24.0",
        "registered_anchor": "vllm eaf1627c (2026-10-27 main, §8.9)",
        "used_wheel": "0.24.0+cu129 (tag commit ee0da84a 2026-06-28)",
        "drift": "6033 commits ahead + 18 behind; same V1 engine generation; "
                 "decode throughput NOT cross-version extrapolatable (batch 21)",
        "reason": "candidate B/C precedent; engine-level baseline, version "
                  "label suffices; max_model_len=4608 (vs 4096 in C/B) for "
                  "MIX in=4096+128 -- within-candidate-D consistent",
        "preheat": "spec H512 untimed run before matrix (Triton JIT, batch 31)",
    }
    with open(os.path.join(outdir, "anchor.json"), "w") as f:
        json.dump(anchor, f, indent=2, sort_keys=True)
    # Human summary
    print("== candidate D axis-3 findings ==", flush=True)
    for mode in MODES:
        fd = findings[mode]
        print("%s makespan H512/H2048/MIX = %s/%s/%s s, "
              "MIX/H2048 = %s, spec/AR %s"
              % (mode, fd["makespan_s"]["H512"], fd["makespan_s"]["H2048"],
                 fd["makespan_s"]["MIX"], fd["MIX_over_H2048"],
                 fd["spec_over_ar"]), flush=True)
        for comp in COMPS:
            r = rows["%s_%s" % (mode, comp)]
            bf = fd.get("bubble_factor_%s" % comp)
            print("  %-5s wall=%s tok/s=%s e2e_med=%s e2e_max=%s bubble=%s"
                  % (comp, r["wall_s"], r["decode_tok_per_s"],
                     fd.get("e2e_median_%s" % comp),
                     fd.get("e2e_max_%s" % comp), bf), flush=True)
            if mode == "spec":
                sm = r.get("spec_metrics", {})
                if "acceptance_rate_pos" in sm:
                    pos = ", ".join("%.3f" % p
                                    for p in sm["acceptance_rate_pos"])
                    print("    A(p)=[%s] acc_len=%s" % (
                        pos, sm.get("mean_accept_len")), flush=True)
    print("wrote summary.json + anchor.json", flush=True)


if __name__ == "__main__":
    main()
