#!/usr/bin/env python
"""Aggregate candidate C2 16-combo outputs into summary.json (string keys).

Combos: {ar,spec} x in{512,2048} x b{32,64} (axis-3 batch scaling) +
        {ar,draft} x in{512,2048} x b{8,16} (strong proposer draft_model,
        same-session AR). Reuses candidate-C parse regex for the engine
        "SpecDecoding metrics:" line (per-position acceptance A(p)).
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

MODES = ["ar", "spec", "draft"]
LENS = [512, 2048]
BATCHES = [8, 16, 32, 64]


def last_json_line(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def main():
    outdir = sys.argv[1]
    rows = {}
    for mode in MODES:
        for il in LENS:
            for b in BATCHES:
                if mode == "ar" and b in (8, 16) and il == 512:
                    pass
                # all 16 combos exist; no skip needed (ar b8/b16 rerun for
                # same-session draft pairing; ar b32/b64 for batch scaling)
                tag = f"lat_{mode}_in{il}_b{b}"
                sp = os.path.join(outdir, tag + "_stdout.txt")
                if not os.path.exists(sp):
                    rows[f"{mode}_{il}_{b}"] = {"missing": True}
                    continue
                text = open(sp, encoding="utf-8", errors="replace").read()
                row = last_json_line(text)
                if row is None:
                    rows[f"{mode}_{il}_{b}"] = {"parse_fail": True}
                    continue
                if mode != "ar":
                    m = None
                    for line in reversed(text.splitlines()):
                        if "SpecDecoding metrics" in line:
                            m = SPEC_RE.search(line)
                            if m:
                                break
                    if m:
                        row["mean_accept_len"] = float(m.group(1))
                        row["accepted_tok_per_s"] = float(m.group(2))
                        row["drafted_tok_per_s"] = float(m.group(3))
                        row["accepted_tokens"] = int(m.group(4))
                        row["drafted_tokens"] = int(m.group(5))
                        row["A_p"] = [float(x) for x in POS_RE.findall(m.group(6))]
                        row["avg_draft_accept_pct"] = float(m.group(7))
                        row["E_L"] = round(1.0 + sum(row["A_p"]), 4)
                rows[f"{mode}_{il}_{b}"] = row
    summary = {"combos": rows, "n": len(rows), "outdir": outdir}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # human table
    print(f"{'combo':<18}{'wall_s':>8}{'dec_tok/s':>10}{'E[L]':>7}  A(p)")
    for mode in MODES:
        for il in LENS:
            for b in BATCHES:
                r = rows.get(f"{mode}_{il}_{b}")
                if not r or "wall_s" not in r:
                    continue
                ap = " ".join(f"{x:.3f}" for x in r.get("A_p", []))
                print(f"{mode}_{il}_{b:<14}{r['wall_s']:>8.2f}{r['decode_tok_per_s']:>10.1f}{r.get('E_L', float('nan')):>7.2f}  {ap}")


if __name__ == "__main__":
    main()
