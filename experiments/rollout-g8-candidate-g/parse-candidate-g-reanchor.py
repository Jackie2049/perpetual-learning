#!/usr/bin/env python3
"""Parse the candidate-G 2+2 placement re-anchor runs (2026-11-02, M5 compute side).

Family: `g1p` = 1 GPU (0), `g2p` = 2 GPUs 1+1 (0,4), `g4p` = 4 GPUs 2+2 (0,1,4,5).
Same workload/config as the candidate-G strong-scaling family (batch 32 prompts x
n=4, 15 iterations, seed 42), so the numbers are directly comparable with g1/g2/g4
and with the placement-probe runs pC1 (2+2, 4 cards) and pD1 (1+1, 2 cards).

The legacy `parse-candidate-g.py` intentionally only recognises
`-g<N>[w][n]` labels; this script keeps the placement family separate so the
archived `summary.json` semantics do not change.

Usage:
  python3 parse-candidate-g-reanchor.py raw/rollout-g8-candidate-g-g1p.txt \
      raw/rollout-g8-candidate-g-g2p.txt raw/rollout-g8-candidate-g-g4p.txt \
      --reference raw/rollout-g8-candidate-g-pD1.txt \
      --reference raw/rollout-g8-candidate-g-pC1.txt \
      --json-out reanchor-summary.json
"""
import argparse
import json
import re
import statistics

STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]
PLACEMENT = {"g1p": "1 卡（GPU 0）", "g2p": "1+1（GPU 0,4）", "g4p": "2+2（GPU 0,1,4,5）",
             "g2wp": "1+1（GPU 0,4）", "g4wp": "2+2（GPU 0,1,4,5）",
             "g2w": "2+0（GPU 0,1）", "g4w": "3+1（GPU 0,1,3,4）",
             "pD1": "1+1（GPU 0,4）", "pC1": "2+2（GPU 0,1,4,5）"}


def load(path):
    rows = []
    for line in open(path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            d["gs"] = int(re.search(r"training/global_step:(\d+)", line).group(1))
            m = re.search(r"perf/throughput:([0-9.]+)", line)
            if m:
                d["throughput"] = float(m.group(1))
            rows.append(d)
    return rows


def steady(rows, lo=3, hi=None):
    if hi is None:
        hi = max(r["gs"] for r in rows) - 1
    rs = [r for r in rows if lo <= r["gs"] <= hi]
    out = {}
    for k in STAGES + ["step", "throughput"]:
        vals = [r[k] for r in rs if k in r]
        if vals:
            out[k] = statistics.mean(vals)
            out[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    out["n"] = len(rs)
    out["last_step"] = max(r["gs"] for r in rows)
    out["step_median"] = statistics.median([r["step"] for r in rs])
    # §13.44: V1's default 2.0 s ReplayBuffer poll quantum shows up as a two-mode
    # `gen` (slow >= 3 s, fast <= 3 s). Count the modes so the re-anchor can be
    # compared with the earlier anchors under the same measurement regime.
    slow = [r["step"] for r in rs if r.get("gen", 0) >= 3.0]
    fast = [r["step"] for r in rs if r.get("gen", 0) < 3.0]
    out["gen_modes"] = {
        "slow_steps": len(slow),
        "fast_steps": len(fast),
        "slow_step_mean": round(statistics.mean(slow), 4) if slow else None,
        "fast_step_mean": round(statistics.mean(fast), 4) if fast else None,
    }
    return out


def extract(path):
    txt = open(path, errors="ignore").read()

    def last_int(pat):
        vals = re.findall(pat, txt)
        return int(vals[-1]) if vals else None

    cfg = {
        "train_batch_size": last_int(r"data\.train_batch_size=(\d+)"),
        "ppo_mini_batch_size": last_int(r"actor\.ppo_mini_batch_size=(\d+)"),
        "rollout_n": last_int(r"actor_rollout_ref\.rollout\.n=(\d+)"),
        "n_gpus_per_node": last_int(r"trainer\.n_gpus_per_node=(\d+)"),
    }
    vals = re.findall(r"acc/mean@1.{0,160}?np\.float64\(([0-9.]+)\)", txt, re.S)
    cfg["val_acc_final"] = float(vals[-1]) if vals else None
    mem = [float(x) for x in re.findall(r"actor/perf/max_memory_reserved_gb:([0-9.]+)", txt)]
    cfg["max_reserved_gb"] = max(mem) if mem else None
    return cfg


def label_of(path):
    base = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.search(r"-(g\d+w?p|g\d+w|p[CD]\d)$", base)
    return m.group(1) if m else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--reference", action="append", default=[],
                    help="earlier placement-probe logs used only for cross-run checks")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    modes, meta = {}, {}
    for path in args.logs + args.reference:
        name = label_of(path)
        modes[name] = steady(load(path))
        meta[name] = extract(path)

    base = modes["g1p"]
    primaries = [n for n in ("g1p", "g2p", "g4p") if n in modes]
    weak = [n for n in ("g2wp", "g4wp") if n in modes]
    summary = {
        "meta": {
            "batch": "candidate-g placement re-anchor (2+2)",
            "executed_utc": "2026-11-02",
            "stack": "verl 24f25b03 + vLLM 0.24.0+cu129, Qwen2.5-1.5B-Instruct, "
                     "gsm8k train_sub480/test_sub120, 4090-1",
            "window_rule": "steps 3 .. max_step-1 (drop 2 warmup steps and the final "
                           "validation/save step)",
            "placement": {k: PLACEMENT.get(k, "unknown") for k in modes},
            "note": "Same workload as the candidate-G strong-scaling runs; "
                    "default ReplayBuffer poll interval kept so the anchors stay "
                    "comparable with §13.36/§13.42.",
        },
        "modes": {}, "scaling": {}, "weak_scaling": {}, "cross_run": {},
    }

    for name, m in modes.items():
        cfg = meta[name]
        summary["modes"][name] = {
            **{k: round(m[k], 4) for k in STAGES + ["step", "throughput"]},
            "step_sd": round(m["step_sd"], 4),
            "step_median": round(m["step_median"], 4),
            "gen_modes": m["gen_modes"],
            "n": m["n"],
            "window": [3, m["last_step"] - 1],
            "stage_sum": round(sum(m[k] for k in STAGES), 4),
            "residual": round(m["step"] - sum(m[k] for k in STAGES), 4),
            "config": cfg,
            "placement": PLACEMENT.get(name, "unknown"),
        }

    for name in primaries[1:]:
        m = modes[name]
        n = meta[name]["n_gpus_per_node"]
        summary["scaling"][name] = {
            "n_gpus": n,
            "speedup": round(base["step"] / m["step"], 4),
            "parallel_efficiency": round(base["step"] / m["step"] / n, 4),
            "aggregate_throughput_ratio": round(m["throughput"] * n / base["throughput"], 4),
            "stage_scaling": {k: round(base[k] / m[k], 4) for k in STAGES},
            "stage_time_ratio": {k: round(m[k] / base[k], 4) for k in STAGES},
        }

    for name in weak:
        m = modes[name]
        n = meta[name]["n_gpus_per_node"]
        # Weak scaling: per-GPU batch fixed, global batch = n x base batch.
        summary["weak_scaling"][name] = {
            "n_gpus": n,
            "global_batch": meta[name]["train_batch_size"],
            "speedup_work_normalized": round(n * base["step"] / m["step"], 4),
            "parallel_efficiency": round(base["step"] / m["step"], 4),
            "aggregate_throughput_ratio": round(m["throughput"] * n / base["throughput"], 4),
            "per_gpu_throughput_ratio": round(m["throughput"] / base["throughput"], 4),
            "stage_time_ratio": {k: round(m[k] / base[k], 4) for k in STAGES},
            "stage_scaling": {k: round(base[k] / m[k], 4) for k in STAGES},
        }

    pairs = [("g4p", "pC1"), ("g2p", "pD1"), ("g2wp", "g2w"), ("g4wp", "g4w")]
    for new, old in pairs:
        if new in modes and old in modes:
            a, b = modes[new], modes[old]
            summary["cross_run"][new] = {
                "reference": old,
                "step_s": round(a["step"], 4),
                "reference_step_s": round(b["step"], 4),
                "ratio": round(a["step"] / b["step"], 4),
                "stage_ratio": {k: round(a[k] / b[k], 4) for k in STAGES},
            }

    print(f"{'mode':5s} {'GPU':>4s} {'step_s':>8s} {'sd':>6s} {'speedup':>8s} "
          f"{'eff':>6s} " + " ".join(f"{k:>13s}" for k in STAGES) + "  placement")
    for name in primaries:
        m = modes[name]
        n = meta[name]["n_gpus_per_node"] or 1
        print(f"{name:5s} {n:4d} {m['step']:8.3f} {m['step_sd']:6.3f} "
              f"{base['step']/m['step']:8.3f} {base['step']/m['step']/n:6.3f} "
              + " ".join(f"{m[k]:13.3f}" for k in STAGES)
              + f"  {PLACEMENT.get(name, '?')}")
    for new, ref in summary["cross_run"].items():
        print(f"cross-run {new} vs {ref['reference']}: step {ref['step_s']:.3f}s -> "
              f"{ref['reference_step_s']:.3f}s = {ref['ratio']:.3f}x "
              f"(stage ratios {ref['stage_ratio']})")
    for name in weak:
        m, w = modes[name], summary["weak_scaling"][name]
        n = w["n_gpus"]
        print(f"weak {name} ({n} GPU, batch {w['global_batch']}, "
              f"{PLACEMENT.get(name, '?')}): step={m['step']:.3f}s "
              f"aggregate_tp={w['aggregate_throughput_ratio']:.3f}x "
              f"eff={w['parallel_efficiency']:.3f}")
    for name in modes:
        cfg = meta[name]
        print(f"cfg {name}: batch={cfg['train_batch_size']} mini={cfg['ppo_mini_batch_size']} "
              f"n={cfg['rollout_n']} gpus={cfg['n_gpus_per_node']} "
              f"acc={cfg['val_acc_final']} peak_reserved={cfg['max_reserved_gb']}GB")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
