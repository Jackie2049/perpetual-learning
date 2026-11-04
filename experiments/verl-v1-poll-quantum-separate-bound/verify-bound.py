#!/usr/bin/env python3
"""Independent re-parse of the separate_async *rollout-bound* poll A/B.

Usage: python3 verify-bound.py [raw_dir]

Exit codes are deliberately split into three kinds of verdict, because the
rollout-bound regime is a *claim about the load*, not a guarantee:

  0  measurement valid AND the regime flipped (poll-2.0 arm shows the quantum)
  1  measurement invalid (missing steps, config drift, step != sum(phases)+gap)
  2  measurement valid but the regime did NOT flip (both arms ~0.03 s of gen)

A 2 means "do not read this A/B as rollout-bound; raise the load and re-smoke",
never "the data are broken".  Effect sizes are printed as INFO only: they are
the pre-registered hypothesis, not a validity condition.

Deliberately different from make-summary-bound.py: it splits a step line on
" - ", keeps only keys of interest, and recomputes means with a plain mean.
"""
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUNS = {"sb20a": 2.0, "sb01a": 0.1, "sb20b": 2.0, "sb01b": 0.1}
WINDOW = set(range(3, 9))
WANT = ["timing_s/gen", "timing_s/step", "timing_s/old_log_prob",
        "timing_s/update_actor", "timing_s/update_weights", "response_length/mean"]
fails, gate_fails = [], []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def gate(name, cond, detail=""):
    print(f"{'GATE-PASS' if cond else 'GATE-FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        gate_fails.append(name)


def info(name, value):
    print(f"INFO {name}: {value}")


def num(token):
    token = token.strip()
    m = re.fullmatch(r"np\.(?:float64|int64)\((-?[0-9.]+)\)", token)
    if m:
        return float(m.group(1))
    try:
        return float(token)
    except ValueError:
        return None


def parse(path):
    steps, cfg = {}, {}
    for line in ANSI.sub("", path.read_text(errors="ignore")).splitlines():
        if "'trainer_mode':" in line:
            m = re.search(r"'trainer_mode':\s*'([a-z_]+)'", line)
            if m:
                cfg["trainer_mode"] = m.group(1)
        if "'poll_interval':" in line:
            m = re.search(r"'poll_interval':\s*([0-9.]+)", line)
            if m:
                cfg["poll_interval"] = float(m.group(1))
        if "'parameter_sync_step':" in line:
            m = re.search(r"'parameter_sync_step':\s*([0-9]+)", line)
            if m:
                cfg["parameter_sync_step"] = int(m.group(1))
        if "'max_response_length':" in line:
            m = re.search(r"'max_response_length':\s*([0-9]+)", line)
            if m:
                cfg["max_response_length"] = int(m.group(1))
        if re.search(r"'backend':\s*'nccl'", line):
            cfg["ckpt_backend"] = "nccl"
        if " step:" not in line or "timing_s/step" not in line:
            continue
        m = re.search(r"step:(\d+)\s", line)
        rec = {"step": float(m.group(1))} if m else {}
        for part in line.split(" - ")[1:]:
            if ":" not in part:
                continue
            key, _, value = part.partition(":")
            value = num(value)
            if value is not None:
                rec[key.strip()] = value
        if "timing_s/step" in rec and "step" in rec:
            steps[int(rec["step"])] = rec
    return cfg, steps


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    steady = {}
    for name, want_poll in RUNS.items():
        cfg, steps = parse(raw / f"{name}.txt")
        check(f"{name}: 8 step lines", len(steps) == 8, f"got {len(steps)}")
        check(f"{name}: trainer_mode=separate_async", cfg.get("trainer_mode") == "separate_async",
              str(cfg.get("trainer_mode")))
        check(f"{name}: resolved poll_interval={want_poll}", cfg.get("poll_interval") == want_poll,
              str(cfg.get("poll_interval")))
        check(f"{name}: checkpoint backend=nccl", cfg.get("ckpt_backend") == "nccl")
        check(f"{name}: parameter_sync_step=1", cfg.get("parameter_sync_step") == 1,
              str(cfg.get("parameter_sync_step")))
        check(f"{name}: max_response_length=1024", cfg.get("max_response_length") == 1024,
              str(cfg.get("max_response_length")))
        window = [s for s in sorted(steps) if s in WINDOW]
        check(f"{name}: steady window steps 3-8 present", window == sorted(WINDOW), str(window))
        steady[name] = {k: mean([steps[s][k] for s in window if k in steps[s]]) for k in WANT}
        gaps = [steps[s]["timing_s/step"] - sum(
            steps[s][k] for k in ("timing_s/gen", "timing_s/old_log_prob",
                                  "timing_s/adv", "timing_s/update_actor",
                                  "timing_s/update_weights")) for s in window]
        check(f"{name}: step = sum(phases) + gap, gap<0.1s", max(gaps) < 0.1,
              f"max gap={max(gaps):.3f}")

    # Regime gate: with poll=2.0 the *only* way `timing_s/gen` can reach 1.9 s in
    # separate_async is that the trainer slept in `_wait_for_next_poll`, so a
    # majority of >=1.9 s steps is exactly the statement "the regime flipped".
    # The 0.1 arm is the control and gets no threshold of its own -- in this
    # regime its true wait can legitimately exceed 2 s, so "gen < 1.9 s" there
    # would reject a perfectly good run.
    for name, want_poll in RUNS.items():
        _, steps = parse(raw / f"{name}.txt")
        gen = [steps[s]["timing_s/gen"] for s in sorted(steps) if s in WINDOW]
        pinned = sum(1 for g in gen if g >= 1.9)
        if want_poll == 2.0:
            gate(f"{name}: steady gen reaches the 2.0s quantum",
                 pinned * 2 >= len(gen), f"{pinned}/{len(gen)} steps >=1.9s")
        else:
            info(f"{name} control arm gen", f"{pinned}/{len(gen)} steps >=1.9s, "
                                            f"mean={mean(gen):.3f}s")

    poll20 = [steady[n] for n in ("sb20a", "sb20b")]
    poll01 = [steady[n] for n in ("sb01a", "sb01b")]
    d_step = mean([r["timing_s/step"] for r in poll20]) - mean([r["timing_s/step"] for r in poll01])
    d_gen = mean([r["timing_s/gen"] for r in poll20]) - mean([r["timing_s/gen"] for r in poll01])
    info("pooled delta step (2.0-0.1)", f"{d_step:+.3f}s")
    info("pooled delta gen (2.0-0.1)", f"{d_gen:+.3f}s")
    info("pre-registered expectation", "d_gen>0.2s and d_gen > d_step-0.2s")

    print(f"\nmechanical: {len(fails)} failure(s); regime gate: {len(gate_fails)} failure(s)")
    if fails:
        return 1
    if gate_fails:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
