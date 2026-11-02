#!/usr/bin/env python3
"""Parse candidate-H (update-weights bucket sweep) verl logs.

Usage:
  python3 parse-candidate-h.py raw/rollout-g8-candidate-h-h256.txt \
      raw/rollout-g8-candidate-h-h1024.txt raw/rollout-g8-candidate-h-h4096.txt

All modes are the g4 configuration (4 GPUs, global batch 32 prompts x n=4,
15 iterations, candidate A stack) and differ only in
rollout.checkpoint_engine.update_weights_bucket_megabytes (256/1024/4096).

Steady window = steps 3 .. max_step-1 (drop 2 warmup steps and the final
validation/save step), same rule as candidate G.

h256 is the same-session baseline (replicate of candidate G's g4, which used
bucket=256). For the single-stage identity we use only the update_weights
channel:
  step_x_pred = 1 / (1 - s_w * (1 - 1/G_w)),   s_w = update_weights_h256/step_h256
The full-stage rebuild is printed as a check.
"""
import json
import re
import statistics
import sys

STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]


def load(path):
    rows = []
    for line in open(path, errors="ignore"):
        if "timing_s/step:" in line and "training/global_step:" in line:
            d = {k: float(v) for k, v in re.findall(r"timing_s/([a-z_]+):([0-9.]+)", line)}
            d["gs"] = int(re.search(r"training/global_step:(\d+)", line).group(1))
            m = re.search(r"perf/throughput:([0-9.]+)", line)
            if m:
                d["throughput"] = float(m.group(1))
            m = re.search(r"perf/total_num_tokens:(\d+)", line)
            if m:
                d["tokens"] = int(m.group(1))
            rows.append(d)
    return rows


def steady(rows, lo=3, hi=None):
    if hi is None:
        hi = max(r["gs"] for r in rows) - 1
    rs = [r for r in rows if lo <= r["gs"] <= hi]
    out = {}
    for k in STAGES + ["step", "throughput", "tokens"]:
        vals = [r[k] for r in rs if k in r]
        if vals:
            out[k] = statistics.mean(vals)
            out[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    out["n"] = len(rs)
    out["last_step"] = max(r["gs"] for r in rows)
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
        "total_epochs": last_int(r"trainer\.total_epochs=(\d+)"),
        "n_gpus_per_node": last_int(r"trainer\.n_gpus_per_node=(\d+)"),
        "bucket_mb": last_int(r"update_weights_bucket_megabytes=(\d+)"),
    }
    vals = re.findall(r"acc/mean@1.{0,160}?np\.float64\(([0-9.]+)\)", txt, re.S)
    cfg["val_acc_final"] = float(vals[-1]) if vals else None
    mem = [float(x) for x in re.findall(r"actor/perf/max_memory_reserved_gb:([0-9.]+)", txt)]
    cfg["max_reserved_gb"] = max(mem) if mem else None
    return cfg


def mode_name(path):
    label = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.search(r"-(n?h?\d+)$", label)
    return m.group(1) if m else label


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    json_out = None
    for a in sys.argv[1:]:
        if a.startswith("--json-out="):
            json_out = a.split("=", 1)[1]
    if not args:
        print(__doc__)
        sys.exit(2)
    modes, meta = {}, {}
    for path in args:
        name = mode_name(path)
        modes[name] = steady(load(path))
        meta[name] = extract(path)

    names = sorted(modes, key=lambda k: (k.startswith("n"), int(re.search(r"(\d+)", k).group(1))))
    base_name = "h256" if "h256" in modes else ("n256" if "n256" in modes else names[0])
    base = modes[base_name]
    s_w = base["update_weights"] / base["step"]
    print(f"BASE {base_name} n={base['n']} step={base['step']:.3f}s")
    print(f"{'stage':15s} {'base_s':>8s} {'mode_s':>8s} {'G':>7s} {'identity_step_x':>16s} {'actual_step_x':>14s}")
    results = {}
    for name in names:
        if name == base_name:
            continue
        m = modes[name]
        print(f"--- {name} n={m['n']} step={m['step']:.3f}s bucket={meta[name]['bucket_mb']}MB")
        g_w = base["update_weights"] / m["update_weights"]
        pred = 1.0 / (1.0 - s_w * (1.0 - 1.0 / g_w))
        stage_sum = sum(m[k] for k in STAGES)
        results[name] = {
            "step_s": round(m["step"], 4),
            "step_x": round(m["step"] / base["step"], 4),
            "stages": {k: round(m[k], 4) for k in STAGES},
            "G": {k: round(base[k] / m[k], 4) for k in STAGES},
            "update_weights_identity": round(pred, 4),
            "stage_sum": round(stage_sum, 4),
            "residual_s": round(m["step"] - stage_sum, 4),
            "config": meta[name],
        }
        for k in STAGES:
            print(f"{k:15s} {base[k]:8.3f} {m[k]:8.3f} {base[k]/m[k]:7.3f} "
                  f"{pred if k == 'update_weights' else float('nan'):16.4f} {m['step']/base['step']:14.4f}")
        print(f"  all-stage rebuild step={stage_sum:.3f} vs measured={m['step']:.3f} "
              f"residual={m['step']-stage_sum:+.3f}s")
        print(f"  update_weights {base['update_weights']:.3f}->{m['update_weights']:.3f}s "
              f"G_w={g_w:.3f}; single-channel prediction={pred:.4f} vs actual={m['step']/base['step']:.4f}")
    out = {
        "batch": "candidate-h",
        "stack": "verl 24f25b03 + vLLM 0.24.0+cu129, Qwen2.5-1.5B-Instruct, "
                 "gsm8k train_sub480/test_sub120, 4090-1 2 GPUs (0,1); "
                 "group A (h*) = socket transport (NCCL_SHM_DISABLE=1), "
                 "group B (n*) = SHM transport (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1)",
        "lever": "rollout.checkpoint_engine.update_weights_bucket_megabytes (256/1024/4096)",
        "base": {
            "n": base["n"],
            "step_s": round(base["step"], 4),
            "stages": {k: round(base[k], 4) for k in STAGES},
            "bucket_mb": meta[base_name]["bucket_mb"],
            "val_acc_final": meta[base_name]["val_acc_final"],
            "max_reserved_gb": meta[base_name]["max_reserved_gb"],
        },
        "modes": results,
    }
    print(json.dumps(out, indent=1))
    if json_out:
        with open(json_out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
