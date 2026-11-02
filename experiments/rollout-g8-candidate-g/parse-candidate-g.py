#!/usr/bin/env python3
"""Parse candidate-G (same-stack data-parallel scaling) verl logs.

Usage:
  python3 parse-candidate-g.py g1.log g2.log g4.log g2w.log g4w.log
  python3 parse-candidate-g.py raw/*.txt --json-out summary.json

Two families are expected (mode is taken from the trailing `-g<N>[w]` of the filename):

  strong (no `w`): global batch fixed at 32 prompts x n=4 = 128 samples/iter,
                   only NGPUS_PER_NODE varies (1/2/4). `speedup = T1/Tn`,
                   `parallel_efficiency = speedup/n`.
  weak (`w` suffix): per-GPU batch fixed at 128 samples/GPU/iter, so mode g<n>w
                   processes n x 128 samples/iter. `speedup = n*T1/Tn`,
                   `parallel_efficiency = T1/Tn` (time ratio), and the honest
                   throughput view is `aggregate_throughput_ratio`.

Steady-state window is steps 3 .. max_step-1 for every log: steps 1-2 contain
vLLM/FSDP warmup and the final step includes final validation (+ checkpoint save).
That yields 3-14 for the 15-step strong runs, 3-13 for the 14-step g2w and
3-11 for the 12-step g4w.
"""
import argparse
import json
import re
import statistics

STAGES = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights"]

META = {
    "batch": "candidate-g",
    "executed_utc": "2026-11-02",
    "stack": "verl 24f25b03 + vLLM 0.24.0+cu129, Qwen2.5-1.5B-Instruct, "
             "gsm8k train_sub480/test_sub120, single node 4090-1",
    "window_rule": "steps 3 .. max_step-1 (drop 2 warmup steps and the final "
                   "validation/save step)",
    "semantics": {
        "strong": "global batch fixed; speedup = T1/Tn; efficiency = speedup/n",
        "weak": "per-GPU batch fixed (n x global batch); speedup = n*T1/Tn; "
                "parallel_efficiency = T1/Tn (time-based); throughput_efficiency "
                "= aggregate_throughput_ratio/n (work-normalized; README quotes this)",
    },
    "note": "NCCL_P2P_DISABLE=1 on consumer 4090: communication stages are "
            "worst-case. Multi-GPU runs used the non-contiguous mask 0,1,3,4 "
            "(GPU 2 occupied) with a local worker.py physical->ordinal patch.",
}


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
    # The final step of every run includes validation (and possibly a final save),
    # so it is excluded from the steady window. Steps 1-2 are vLLM/FSDP warmup.
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
    """Config echo, final validation acc and peak reserved memory from the log text."""
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
    }
    # The final "Final validation metrics" dict is printed across wrapped lines and
    # each log line carries an ANSI/TaskRunner prefix, so allow a short gap and the
    # last numeric match is the final validation acc.
    vals = re.findall(r"acc/mean@1.{0,160}?np\.float64\(([0-9.]+)\)", txt, re.S)
    cfg["val_acc_final"] = float(vals[-1]) if vals else None
    mem = [float(x) for x in re.findall(r"actor/perf/max_memory_reserved_gb:([0-9.]+)", txt)]
    cfg["max_reserved_gb"] = max(mem) if mem else None
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    modes, meta = {}, {}
    for path in args.logs:
        label = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        m = re.search(r"-(g\d+w?)$", label)
        name = m.group(1) if m else label
        modes[name] = steady(load(path))
        meta[name] = extract(path)

    def sort_key(k):
        m = re.match(r"g(\d+)", k)
        return (int(m.group(1)) if m else 0, k.endswith("w"))

    names = sorted(modes, key=sort_key)
    base_name = next((n for n in names if not n.endswith("w")), names[0])
    base = modes[base_name]
    print(f"BASE {base_name} n={base['n']} step={base['step']:.3f}s "
          f"(sd {base['step_sd']:.3f}) throughput={base['throughput']:.1f} tok/s")
    print(f"{'stage':15s} {'base_s':>8s}", end="")
    for name in names:
        if name == base_name:
            continue
        print(f" {name+'_s':>9s} {'G':>7s}", end="")
    print()
    for k in STAGES:
        print(f"{k:15s} {base[k]:8.3f}", end="")
        for name in names:
            if name == base_name:
                continue
            v = modes[name][k]
            print(f" {v:9.3f} {base[k]/v:7.3f}", end="")
        print()

    summary = {"meta": META, "modes": {}, "scaling": {}}
    for name in names:
        m = modes[name]
        cfg = meta[name]
        summary["modes"][name] = {
            **{k: round(m[k], 4) for k in STAGES + ["step", "throughput", "tokens"]},
            "step_sd": round(m["step_sd"], 4),
            "n": m["n"],
            "window": [3, m["last_step"] - 1],
            "stage_sum": round(sum(m[k] for k in STAGES), 4),
            "residual": round(m["step"] - sum(m[k] for k in STAGES), 4),
            "stage_share_pct": {
                **{k: round(m[k] / m["step"] * 100, 1) for k in STAGES},
                "residual": round((m["step"] - sum(m[k] for k in STAGES)) / m["step"] * 100, 1),
            },
            "config": {k: cfg[k] for k in
                       ["train_batch_size", "ppo_mini_batch_size", "rollout_n",
                        "total_epochs", "n_gpus_per_node"]},
            "global_samples_per_iter": (
                cfg["train_batch_size"] * cfg["rollout_n"]
                if cfg["train_batch_size"] and cfg["rollout_n"] else None),
            "val_acc_final": cfg["val_acc_final"],
            "max_reserved_gb": cfg["max_reserved_gb"],
        }
        if name == base_name:
            continue
        n = cfg["n_gpus_per_node"] or int(re.match(r"g(\d+)", name).group(1))
        t_ratio = m["step"] / base["step"]
        weak = name.endswith("w")
        speedup = (n * base["step"] / m["step"]) if weak else (base["step"] / m["step"])
        eff = (base["step"] / m["step"]) if weak else (base["step"] / m["step"] / n)
        entry = {
            "n_gpus": n,
            "scale_kind": "weak (per-GPU batch fixed; global batch scaled)" if weak
                          else "strong (global batch fixed)",
            "time_ratio_vs_base": round(t_ratio, 4),
            "speedup": round(speedup, 4),
            "parallel_efficiency": round(eff, 4),
            "throughput_efficiency": round(
                (m["throughput"] * n / base["throughput"]) / n, 4),
            "per_gpu_throughput_ratio": round(m["throughput"] / base["throughput"], 4),
            "aggregate_throughput_ratio": round(m["throughput"] * n / base["throughput"], 4),
            "stage_scaling": {k: round(base[k] / m[k], 4) for k in STAGES},
            "stage_time_ratio": {k: round(m[k] / base[k], 4) for k in STAGES},
            "residual": round(m["step"] - sum(m[k] for k in STAGES), 4),
        }
        summary["scaling"][name] = entry
    print()
    for name in names:
        m = summary["modes"][name]
        print(f"{name}: step={m['step']:.3f}s sd={m['step_sd']:.3f} "
              f"tok/s={m['throughput']:.1f} tokens={m['tokens']:.0f} "
              f"stage_sum={m['stage_sum']:.3f} residual={m['residual']:+.3f}s "
              f"acc={m['val_acc_final']}")
    for name, v in summary["scaling"].items():
        print(f"{name}: {v['scale_kind']} speedup={v['speedup']:.3f}x "
              f"efficiency={v['parallel_efficiency']:.3f} "
              f"time_ratio={v['time_ratio_vs_base']:.3f} "
              f"aggregate_tp={v['aggregate_throughput_ratio']:.3f}x")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
