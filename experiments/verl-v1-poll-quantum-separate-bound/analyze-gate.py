#!/usr/bin/env python3
"""GATE + cost-ratio analysis for the separate_async rollout-bound probe.

Reads only archived artefacts (`raw/`): three real console logs from the smoke
runs and the two engine-level generation-cost probe dumps.

Outputs, per load point:
  * steady-state `timing_s/gen` (the trainer's wait) and the pre-registered GATE;
  * the trainer's per-token and fixed costs, back-solved from the same step;
  * the generation budget implied by the step and how far the measured
    generation throughput sits from it (slack factor).

Exit codes mirror verify-bound.py: 0 = GATE passed (A/B may run), 2 = the
interval did not flip (A/B must not run), 1 = the artefact is mechanically
inconsistent.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW = Path(__file__).parent / "raw"

# (label, log file, probe dump matched to the same batch shape)
ARMS = [
    ("n16-r1024", "gate-n16-r1024-poll20.txt", "gen-cost-n16.json", "prompts=4 n=16 max_tokens=1024"),
    ("n16-r1536", "gate-n16-r1536-poll20.txt", "gen-cost-n16.json", "prompts=4 n=16 max_tokens=1024 (cap not reached)"),
    ("n32-r1536", "gate-n32-r1536-poll20.txt", "gen-cost-n32.json", "prompts=4 n=32 max_tokens=1536"),
]

FIELDS = [
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "perf/total_num_tokens",
    "response_length/mean",
    "prompt_length/mean",
]
FLOAT_FIELDS = set(FIELDS) - {"perf/total_num_tokens"}

GATE_MIN_WAIT = 1.9  # seconds; see README "预登记判定规则"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_steps(text: str):
    """Return {step: {field: value}} from `step:N - k:v - k:v ...` console lines."""
    steps = {}
    for line in ANSI.sub("", text).splitlines():
        m = re.search(r"\bstep:(\d+) - ", line)
        if not m:
            continue
        n = int(m.group(1))
        row = {}
        for key in FIELDS:
            km = re.search(re.escape(key) + r":(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)(?!\d)", line)
            if not km:
                continue
            val = float(km.group(1))
            row[key] = int(val) if key == "perf/total_num_tokens" else val
        if row:
            steps[n] = row
    return steps


def parse_config(text: str):
    """Read the dumped config values that were only verifiable in a real window."""
    flat = ANSI.sub("", text)
    out = {}
    for key, pat in [
        ("poll_interval", r"'poll_interval':\s*([0-9.]+)"),
        ("max_response_length", r"'max_response_length':\s*([0-9]+)"),
        ("parameter_sync_step", r"'parameter_sync_step':\s*([0-9]+)"),
        ("rollout_n", r"'n':\s*([0-9]+),"),
    ]:
        m = re.search(pat, flat)
        if m:
            out[key] = float(m.group(1))
    return out


def probe_throughput(path: Path):
    d = json.loads(path.read_text())
    rows = d["rows"][1:] or d["rows"]  # drop the first repeat: JIT/compile warmup
    tok = sorted(r["gen_tok_per_s"] for r in rows)
    mid = tok[len(tok) // 2]
    return mid, [r["wall_s"] for r in rows], d


def analyse():
    problems = []
    results = []
    for label, logname, probename, probe_desc in ARMS:
        log = RAW / logname
        if not log.exists():
            problems.append(f"{label}: missing {log}")
            continue
        text = log.read_text(errors="replace")
        steps = parse_steps(text)
        cfg = parse_config(text)
        if len(steps) < 3:
            problems.append(f"{label}: only {len(steps)} step lines parsed")
            continue

        steady = [steps[k] for k in sorted(steps)[1:]]  # drop step 1 (warmup)
        waits = [s["timing_s/gen"] for s in steady]
        frac_waiting = sum(w >= GATE_MIN_WAIT for w in waits) / len(waits)

        # accounting gap on steady steps (validates the parser against the log)
        gaps = []
        for s in steady:
            parts = [s[k] for k in ("timing_s/gen", "timing_s/old_log_prob", "timing_s/adv",
                                    "timing_s/update_actor", "timing_s/update_weights") if k in s]
            if len(parts) == 5:
                gaps.append(s["timing_s/step"] - sum(parts))

        last = steady[-1]
        tokens = last["perf/total_num_tokens"]
        actor = last["timing_s/old_log_prob"] + last["timing_s/update_actor"]
        fixed = last["timing_s/update_weights"] + last["timing_s/adv"]
        per_tok_ms = 1000.0 * actor / tokens
        step_s = last["timing_s/step"]
        budget_tok_s = tokens / step_s  # generation rate at which the trainer would wait

        q, walls, probe = probe_throughput(RAW / probename)
        slack = q / budget_tok_s

        results.append(dict(
            label=label, cfg=cfg, steps=len(steps), steady=len(steady),
            waits=waits, frac_waiting=frac_waiting, gaps=gaps,
            tokens=tokens, actor_s=actor, fixed_s=fixed,
            per_tok_ms=per_tok_ms, step_s=step_s, budget_tok_s=budget_tok_s,
            q_tok_s=q, probe_walls=walls, slack=slack, probe_desc=probe_desc,
            resp_mean=last.get("response_length/mean"), prompt_mean=last.get("prompt_length/mean"),
        ))
    return results, problems


def main():
    results, problems = analyse()
    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1

    print("=== GATE (pre-registered: >half of steady steps must have timing_s/gen >= 1.9 s) ===")
    for r in results:
        waits = ", ".join(f"{w:.4f}" for w in r["waits"])
        print(f"{r['label']:10s} steady gen=[{waits}] -> waiting {r['frac_waiting']*100:.0f}% "
              f"=> GATE {'PASS' if r['frac_waiting'] > 0.5 else 'FAIL'}")
        print(f"{'':10s} config: poll={r['cfg'].get('poll_interval')} "
              f"max_response={r['cfg'].get('max_response_length')} "
              f"sync_step={r['cfg'].get('parameter_sync_step')} n={r['cfg'].get('rollout_n')} "
              f"| resp_mean={r['resp_mean']:.1f} prompt_mean={r['prompt_mean']:.1f}")

    print("\n=== accounting gap on steady steps (step - sum(phases), s) ===")
    for r in results:
        thr = 0.02 * r["step_s"]
        print(f"{r['label']:10s} gaps={['%.4f' % g for g in r['gaps']]} "
              f"threshold={max(0.1, thr):.3f} -> {'OK' if max(r['gaps']) <= max(0.1, thr) else 'OVER'}")

    print("\n=== why the trainer never waits: per-token cost, same step ===")
    print(f"{'arm':10s} {'tokens':>7s} {'actor_s':>8s} {'fixed_s':>8s} {'ms/token':>9s} "
          f"{'step_s':>7s} {'budget_tok/s':>12s} {'probe tok/s':>11s} {'slack':>6s}")
    for r in results:
        print(f"{r['label']:10s} {r['tokens']:7d} {r['actor_s']:8.3f} {r['fixed_s']:8.3f} "
              f"{r['per_tok_ms']:9.3f} {r['step_s']:7.3f} {r['budget_tok_s']:12.0f} "
              f"{r['q_tok_s']:11.0f} {r['slack']:6.2f}x")

    print("\nprobe scope: engine level only (vLLM offline API, same GPU/model/version), "
          "not the verl stack; the first repeat is dropped as compile warmup.")
    for r in results:
        print(f"  {r['label']:10s} {r['probe_desc']}; repeats wall={['%.2f' % w for w in r['probe_walls']]} s")

    passed = any(r["frac_waiting"] > 0.5 for r in results)
    print("\n=== verdict ===")
    if passed:
        print("GATE passed on at least one load point -> A/B may run.")
        return 0
    print("GATE failed on every load point: the trainer never blocks in sample(), so the "
          "rollout-bound interval was not reached on this stack. Do NOT run the poll A/B "
          "(the pre-registered rule forbids reading a null A/B here).")
    print(f"Slack grows with load ({', '.join('%s: %.1fx' % (r['label'], r['slack']) for r in results)}): "
          "training cost per token exceeds generation cost per token, so adding tokens widens the gap "
          "instead of closing it.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
