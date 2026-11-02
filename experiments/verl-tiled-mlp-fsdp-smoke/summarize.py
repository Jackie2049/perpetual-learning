#!/usr/bin/env python3
"""Consolidate every TiledMLP smoke run into results.json.

Run on 4090-1 from /tmp/m5-tiled-mlp after run_smoke.sh and the extra
0.5B / precision runs.
"""

import json
import os
import platform

EQUIVALENCE = [
    ("tiny 16tok s4", "wired_fsdp2_e1_s4", "base_fsdp2_e0"),
    ("tiny 16tok s8", "wired_fsdp2_e1_s8", "base_fsdp2_e0"),
    ("tiny 16tok s16", "wired_fsdp2_s16", "base_fsdp2_e0"),
    ("tiny 18tok s4", "wired_fsdp2_uneven", "base_fsdp2_e0_n18"),
    ("tiny 3tok s4", "wired_fsdp2_e1_n3", "base_fsdp2_e0_n3"),
    ("tiny 5tok s4", "wired_fsdp2_e1_n5", "base_fsdp2_e0_n5"),
    ("tiny 18tok bf16", "tiny_bf16_e1", "tiny_bf16_e0"),
    ("tiny 18tok fp32", "tiny_fp32_e1", "tiny_fp32_e0"),
    ("CONTROL unpatched 16tok", "unwired_fsdp2_e1", "base_fsdp2_e0"),
    ("0.5B 8192tok bf16 s8", "wired_05b_e1s8", "base_05b_e0"),
]

MEMORY = [
    ("0.5B 8192tok ckpt-off s4", "wired_05b_e1s4", "base_05b_e0"),
    ("0.5B 8192tok ckpt-off s8", "wired_05b_e1s8", "base_05b_e0"),
    ("0.5B 8192tok ckpt-off s16", "wired_05b_e1s16", "base_05b_e0"),
    ("0.5B 8192tok ckpt-ON s8", "wired_05b_e1s8_ckpt", "base_05b_e0_ckpt"),
]

FSDP1 = ["wired_fsdp1_e1_s4", "wired_fsdp1_e1_orig", "wired_fsdp1_e1_noresh"]


def load(name):
    with open(os.path.join("out", f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    eq = []
    for label, cand, base in EQUIVALENCE:
        c, b = load(cand), load(base)
        gc, gb = c.get("grad_norms") or {}, b.get("grad_norms") or {}
        lp_c, lp_b = (c.get("logprobs") or {}).get("sum"), (b.get("logprobs") or {}).get("sum")
        eq.append(
            {
                "label": label,
                "candidate": cand,
                "baseline": base,
                "patched": c.get("patched"),
                "both_ok": bool(c.get("ok")) and bool(b.get("ok")),
                "same_init": abs(c.get("param_fingerprint", 0) - b.get("param_fingerprint", 0)) < 1e-9,
                "mlp_grad_rel": rel(gc.get("mlp"), gb.get("mlp")),
                "other_grad_rel": rel(gc.get("other"), gb.get("other")),
                "logprob_rel": rel(lp_c, lp_b),
            }
        )

    mem = []
    for label, cand, base in MEMORY:
        c, b = load(cand), load(base)
        mem.append(
            {
                "label": label,
                "baseline_gb": round(b["peak_mem_bytes"] / 1e9, 3),
                "tiled_gb": round(c["peak_mem_bytes"] / 1e9, 3),
                "delta_pct": round(100 * (c["peak_mem_bytes"] / b["peak_mem_bytes"] - 1), 1),
                "step_s_base": b.get("step_s"),
                "step_s_tiled": c.get("step_s"),
                "ok": bool(c.get("ok")) and bool(b.get("ok")),
            }
        )

    fsdp1 = {}
    for name in FSDP1:
        d = load(name)
        fsdp1[name] = {
            "ok": d.get("ok"),
            "patched": d.get("patched"),
            "error_type": d.get("error_type"),
            "error": (d.get("error") or "")[:200],
        }

    cpu = json.load(open("out/cpu_equiv.json", encoding="utf-8"))

    result = {
        "env": {
            "python": platform.python_version(),
            "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | head -1").read().strip(),
            "verl_commit": "6879e1ab07",
            "patch_sha256": os.popen("sha256sum tiled-mlp-wiring.patch").read().split()[0],
        },
        "equivalence": eq,
        "memory": mem,
        "precision_note": "fp64 CPU shows exactness; GPU drift is bf16 GEMM-shape rounding",
        "fsdp1": fsdp1,
        "cpu_fp64": cpu["rows"],
    }
    with open("out/results.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"{'case':28s} {'ok':5s} {'init':5s} {'mlpΔ':>9s} {'otherΔ':>9s} {'lpΔ':>9s}")
    for r in eq:
        f = lambda x: "-" if x is None else f"{x:.2e}"
        print(
            f"{r['label']:28s} {str(r['both_ok']):5s} {str(r['same_init']):5s} "
            f"{f(r['mlp_grad_rel']):>9s} {f(r['other_grad_rel']):>9s} {f(r['logprob_rel']):>9s}"
        )
    print("\nmemory (peak allocated, 0.5B / 8192 tok):")
    for m in mem:
        print(f"  {m['label']:26s} {m['baseline_gb']:6.2f} -> {m['tiled_gb']:6.2f} GB ({m['delta_pct']:+.1f}%)")
    print("\nfsdp1:", json.dumps(fsdp1, ensure_ascii=False))


if __name__ == "__main__":
    main()
