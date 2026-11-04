#!/usr/bin/env python3
"""Stale-wait arm: did the poll quantum enter the critical path?

Pre-registered before the runs (see README "stale-wait 臂"): with
`max_off_policy_threshold=1` + `strategy=drop` the trainer must block in
`ReplayBufferAsync.sample()` every step, so `timing_s/gen` is a real wait.

  1. GATE'  : poll-2.0 arm, >half of steady steps have timing_s/gen >= 1.9 s
  2. Delta gen in [0, ~2 s]  (the quantum rounds up, it is not additive)
  3. Delta step ~= Delta gen, other phases |Delta| <= 0.2 s
  4. the two poll-2.0 runs agree in sign

Reads only `raw-stale/`.  Exit 1 if the artefact is mechanically inconsistent;
otherwise prints PASS/FAIL per prediction and the phase decomposition.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

RAW = Path(__file__).parent / "raw-stale"
RUNS = [("sb20a", 2.0), ("sb01a", 0.1), ("sb20b", 2.0), ("sb01b", 0.1)]
WINDOW = range(3, 9)  # steady steps, same window as verify-bound.py
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PHASES = ["timing_s/old_log_prob", "timing_s/adv", "timing_s/update_actor", "timing_s/update_weights"]
NUM = r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"


def parse(path: Path):
    steps, cfg, evicted = {}, {}, []
    for line in ANSI.sub("", path.read_text(errors="ignore")).splitlines():
        for key, pat in (("poll_interval", r"'poll_interval':\s*([0-9.]+)"),
                         ("max_off_policy_threshold", r"'max_off_policy_threshold':\s*([0-9]+)")):
            m = re.search(pat, line)
            if m:
                cfg[key] = float(m.group(1))
        m = re.search(r"\bstep:(\d+) - ", line)
        if not m:
            continue
        row = {}
        for key in ["timing_s/gen", "timing_s/step", "perf/total_num_tokens"] + PHASES:
            km = re.search(re.escape(key) + ":" + NUM, line)
            if km:
                row[key] = float(km.group(1))
        if "timing_s/step" in row:
            steps[int(m.group(1))] = row
            em = re.search(r"training/off_policy/evicted_samples:" + NUM, line)
            evicted.append(float(em.group(1)) if em else 0.0)
    return cfg, steps, evicted


def main():
    data, problems = {}, []
    for name, poll in RUNS:
        path = RAW / f"{name}.txt"
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        cfg, steps, evicted = parse(path)
        if len(steps) < 8:
            problems.append(f"{name}: only {len(steps)} step lines")
            continue
        if cfg.get("poll_interval") != poll:
            problems.append(f"{name}: poll_interval={cfg.get('poll_interval')} expected {poll}")
        if cfg.get("max_off_policy_threshold") != 1.0:
            problems.append(f"{name}: threshold={cfg.get('max_off_policy_threshold')} expected 1")
        data[name] = (cfg, steps, evicted)
    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1

    def steady(name, key):
        steps = data[name][1]
        return [steps[s][key] for s in sorted(steps) if s in WINDOW]

    print("=== per-run steady window (steps 3-8) ===")
    for name, poll in RUNS:
        gen = steady(name, "timing_s/gen")
        extra = [g for g in gen if g >= 3.5]
        print(f"{name} (poll={poll}) gen={['%.2f' % g for g in gen]} mean={st.mean(gen):.3f}s "
              f"floor={min(gen):.2f}s two-quantum-steps={len(extra)}/{len(gen)} "
              f"evicted/step={data[name][2][1:]}")
        print(f"{'':13s} step={st.mean(steady(name, 'timing_s/step')):.3f}s "
              f"tokens={st.mean(steady(name, 'perf/total_num_tokens')):.0f}")

    arms20 = [n for n, p in RUNS if p == 2.0]
    arms01 = [n for n, p in RUNS if p == 0.1]
    pool = lambda arms, key: st.mean([st.mean(steady(n, key)) for n in arms])
    d_step = pool(arms20, "timing_s/step") - pool(arms01, "timing_s/step")
    d_gen = pool(arms20, "timing_s/gen") - pool(arms01, "timing_s/gen")

    print("\n=== phase decomposition (pooled steady means, 2.0 - 0.1) ===")
    for key in ["timing_s/gen"] + PHASES + ["timing_s/step"]:
        a, b = pool(arms20, key), pool(arms01, key)
        print(f"{key:26s} 2.0={a:7.3f}s 0.1={b:7.3f}s delta={a - b:+7.3f}s")
    tok_a, tok_b = pool(arms20, "perf/total_num_tokens"), pool(arms01, "perf/total_num_tokens")
    per_tok_a = pool(arms20, "timing_s/update_actor") / tok_a * 1e3
    per_tok_b = pool(arms01, "timing_s/update_actor") / tok_b * 1e3
    print(f"{'tokens/step':26s} 2.0={tok_a:7.0f}  0.1={tok_b:7.0f}  delta={tok_a - tok_b:+7.0f} "
          f"({(tok_a / tok_b - 1) * 100:+.1f}%)")
    print(f"{'update_actor ms/token':26s} 2.0={per_tok_a:7.3f}  0.1={per_tok_b:7.3f} "
          f"delta={per_tok_a - per_tok_b:+7.3f}")

    print("\n=== pre-registered predictions ===")
    verdicts = {}

    gate = {n: sum(g >= 1.9 for g in steady(n, "timing_s/gen")) * 2 >= len(WINDOW) for n in arms20}
    verdicts["1 GATE': both 2.0 arms wait on >half steady steps"] = all(gate.values())
    print(f"{'PASS' if all(gate.values()) else 'FAIL'} 1. GATE' per arm: {gate}")

    verdicts["2 Delta gen in [0, 2]s"] = 0.0 <= d_gen <= 2.0
    print(f"{'PASS' if verdicts['2 Delta gen in [0, 2]s'] else 'FAIL'} 2. Delta gen={d_gen:+.3f}s")

    others = {k: pool(arms20, k) - pool(arms01, k) for k in PHASES}
    worst = max(others, key=lambda k: abs(others[k]))
    # update_actor is token-proportional; normalise before judging "other phases".
    norm_gap = (pool(arms20, "timing_s/update_actor") * (tok_b / tok_a)) - pool(arms01, "timing_s/update_actor")
    ok3 = all(abs(v) <= 0.2 for k, v in others.items() if k != "timing_s/update_actor") and abs(norm_gap) <= 0.2
    verdicts["3 Delta step ~= Delta gen (other phases |d|<=0.2s)"] = ok3
    print(f"{'PASS' if ok3 else 'FAIL'} 3. other phases: "
          + ", ".join(f"{k.split('/')[-1]}={v:+.3f}" for k, v in others.items())
          + f"; update_actor token-normalised={norm_gap:+.3f}s (raw {others['timing_s/update_actor']:+.3f}s)")
    print(f"     Delta step={d_step:+.3f}s vs Delta gen={d_gen:+.3f}s -> residual {d_step - d_gen:+.3f}s")

    sign = [st.mean(steady(n, "timing_s/gen")) for n in arms20]
    verdicts["4 the two 2.0 runs agree in direction"] = len(sign) == 2 and all(
        a > b for a in sign for b in [st.mean(steady(n, "timing_s/gen")) for n in arms01])
    print(f"{'PASS' if verdicts['4 the two 2.0 runs agree in direction'] else 'FAIL'} 4. "
          f"2.0 arms {['%.3f' % x for x in sign]} vs 0.1 arms "
          f"{['%.3f' % st.mean(steady(n, 'timing_s/gen')) for n in arms01]}")

    print("\n=== quantisation signature (per-step, all 8 steps) ===")
    for name, poll in RUNS:
        gen = [data[name][1][s]["timing_s/gen"] for s in sorted(data[name][1])]
        below = sum(g < 1.9 for g in gen)
        two = sum(g >= 3.5 for g in gen)
        print(f"{name} (poll={poll}) steps<1.9s={below}/8 two-quantum={two}/8 "
              f"min={min(gen):.2f}s max={max(gen):.2f}s")
    print("A 2.0 arm that never drops below one quantum while the 0.1 arm does is the "
          "quantisation floor; steps landing just above 2.0 s pay a second full quantum (>=3.5 s).")

    print("\n" + ("all pre-registered predictions hold" if all(verdicts.values())
                  else "NOT all predictions hold: " + ", ".join(k for k, v in verdicts.items() if not v)))
    return 0 if all(verdicts.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
