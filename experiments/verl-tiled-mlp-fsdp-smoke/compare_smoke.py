#!/usr/bin/env python3
"""Compare TiledMLP smoke runs against their same-shape baselines.

Run on 4090-1 from /tmp/m5-tiled-mlp after run_smoke.sh.
Writes out/summary.json and prints a table.
"""

import glob
import json
import os

PAIRS = [
    # (label, candidate, baseline) — same seq_len/batch, baseline has tiled_mlp disabled
    ("fsdp2 16tok s4 vs baseline", "wired_fsdp2_e1_s4", "base_fsdp2_e0"),
    ("fsdp2 16tok s8 vs baseline", "wired_fsdp2_e1_s8", "base_fsdp2_e0"),
    ("fsdp2 16tok s16 vs baseline", "wired_fsdp2_s16", "base_fsdp2_e0"),
    ("fsdp2 18tok s4 vs baseline", "wired_fsdp2_uneven", "base_fsdp2_e0_n18"),
    ("fsdp2 3tok s4 vs baseline", "wired_fsdp2_e1_n3", "base_fsdp2_e0_n3"),
    ("fsdp2 5tok s4 vs baseline", "wired_fsdp2_e1_n5", "base_fsdp2_e0_n5"),
    ("UNPATCHED 16tok s4 vs baseline", "unwired_fsdp2_e1", "base_fsdp2_e0"),
]


def load(name):
    path = os.path.join("out", f"{name}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rel(a, b):
    if a is None or b is None:
        return None
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom


def main():
    rows = []
    for label, cand, base in PAIRS:
        c, b = load(cand), load(base)
        gc, gb = c.get("grad_norms") or {}, b.get("grad_norms") or {}
        row = {
            "label": label,
            "candidate": cand,
            "baseline": base,
            "same_init": abs(c.get("param_fingerprint", 0) - b.get("param_fingerprint", 0)) < 1e-9,
            "both_ok": bool(c.get("ok")) and bool(b.get("ok")),
            "mlp_grad_rel": rel(gc.get("mlp"), gb.get("mlp")),
            "other_grad_rel": rel(gc.get("other"), gb.get("other")),
            "mlp_grad": gc.get("mlp"),
            "other_grad": gb.get("other"),
        }
        rows.append(row)

    print(f"{'case':34s} {'ok':5s} {'sameInit':8s} {'mlpΔrel':>10s} {'otherΔrel':>10s}")
    for r in rows:
        mlp = "-" if r["mlp_grad_rel"] is None else f"{r['mlp_grad_rel']:.3e}"
        oth = "-" if r["other_grad_rel"] is None else f"{r['other_grad_rel']:.3e}"
        print(f"{r['label']:34s} {str(r['both_ok']):5s} {str(r['same_init']):8s} {mlp:>10s} {oth:>10s}")

    failures = {}
    for name in ("wired_fsdp1_e1_s4", "wired_fsdp1_e1_orig", "wired_fsdp1_e1_noresh"):
        d = load(name)
        failures[name] = {
            "ok": d.get("ok"),
            "patched": d.get("patched"),
            "error_type": d.get("error_type"),
            "error": (d.get("error") or "")[:200],
        }
    print("\nFSDP1 (strategy=fsdp) with tiled_mlp enabled:")
    for k, v in failures.items():
        print(f"  {k}: ok={v['ok']} patched={v['patched']} {v['error_type']}: {v['error']}")

    with open(os.path.join("out", "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"pairs": rows, "fsdp1": failures}, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
